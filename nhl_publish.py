"""Tonight's NHL board, published as the page's own data file.

    docs/data/nhl_dk_classic_<draft group>.json
    docs/data/manifest.json

The payload is the same shape the baseball build emits, deliberately: the page
is already sport-agnostic - it reads the roster, the players and the loadings
out of the slate rather than knowing anything about a sport - so one page
renders both and there is one place to fix a rendering bug.

The join is by NAME, and that is the weak point
-----------------------------------------------
MoneyPuck keys on the league's player id (8478402); DraftKings keys on its own
(878032). There is no crosswalk between them, so the only thing the two sides
share is a name. That is the least reliable join in this project, so it is
measured rather than assumed: the coverage is logged, the misses are listed
with their salaries, and a board that joins badly is refused rather than
published half-empty. Three players missing from an NFL showdown board were
invisible to every measurement until a finished contest was joined against it,
and that is exactly the failure this reporting exists to prevent.

What is NOT in this model yet, stated plainly
---------------------------------------------
**Confirmed lines and power-play units.** `nhl_sport` is built to use them and
they are the largest single driver of a skater's night - the hockey equivalent
of a batting order. MoneyPuck does not publish tonight's deployment, only what
happened in past games, so the columns arrive as the player's own recent
average rather than tonight's announced unit. The model degrades to one that
cannot see a promotion, which is where every model without a lines feed
already is, and it says so on the page.

**The confirmed starting goalie.** A backup who does not dress scores zero in
a slot that has no alternative. DraftKings' own board is the only signal
available here, and it is weak. Until a goalie feed is added, the goalie slot
is the least trustworthy thing on the board and the page says that too.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import nhl_data as ND
import nhl_fetch as NF
import nhl_projections as NP
import nhl_sport as NS
from engine import model as EM
from engine import optimise as O
from engine import ownership as OWN
from engine import simulate as S

log = logging.getLogger("nhl_publish")

SPORT = "nhl"
EASTERN = ZoneInfo("America/New_York")
DOCS = Path("docs")
DATA = DOCS / "data"
ARCHIVE_DAYS = 120

ROSTER = NS.DK_CLASSIC

# The board and the history are mapped by the SAME function, in nhl_sport.
# Two copies of this drifted apart once already in this project - the two
# ownership modules - and a position map that disagrees between the frame the
# model was fitted on and the board it is applied to would be invisible.
roster_position = NS.roster_position


# ------------------------------------------------------------- which slate
def next_slate_day(now: pd.Timestamp | None = None) -> str:
    """The day the next UNLOCKED slate belongs to."""
    listed = ND.slates()
    now = now or pd.Timestamp.now(tz="UTC")
    live = listed[listed["starts"].notna() & (listed["starts"] > now)]
    if live.empty:
        day = now.tz_convert(EASTERN).strftime("%Y-%m-%d")
        log.warning("no slate on sale has a future lock time - falling back "
                    "to the Eastern date, %s", day)
        return day
    soonest = live["starts"].min()
    day = soonest.tz_convert(EASTERN).strftime("%Y-%m-%d")
    log.info("next slate locks %s Eastern - building for %s",
             soonest.tz_convert(EASTERN).strftime("%H:%M"), day)
    return day


def pinned_slate_day(draft_group: int | None) -> str | None:
    """The day a NAMED draft group belongs to.

    Naming a draft group has to pin the date as well as the board. Leaving the
    date to "the next slate that has not locked" is a split brain: once
    tonight's main slate locks that answer becomes TOMORROW, and the build
    would then pair tonight's board with tomorrow's context. The baseball
    build shipped exactly that bug.
    """
    if not draft_group:
        return None
    try:
        listed = ND.slates()
        row = listed[listed["draft_group"] == int(draft_group)]
        if not len(row) or pd.isna(row["starts"].iloc[0]):
            return None
        return row["starts"].iloc[0].tz_convert(EASTERN).strftime("%Y-%m-%d")
    except Exception as exc:                                   # noqa: BLE001
        log.warning("no lock time for draft group %s (%s)", draft_group,
                    type(exc).__name__)
        return None


def candidate_slates(draft_group: int | None, look: int) -> list[tuple]:
    """Which boards to publish, biggest first.

    Ranked by how many TEAMS a board covers rather than by contest count.
    Cheap single-game contests are numerous, so counting them picks a showdown
    over the main slate - which is the mistake the baseball build made and
    fixed.
    """
    listed = ND.slates()
    if draft_group:
        row = listed[listed["draft_group"] == int(draft_group)]
        when = row["starts"].iloc[0] if len(row) else pd.NaT
        label = str(row["example"].iloc[0]) if len(row) else "(given)"
        return [(int(draft_group), label, ND.board(int(draft_group)), when)]

    out = []
    for r in listed.head(look).to_dict("records"):
        try:
            b = ND.board(int(r["draft_group"]))
        except ND.DataUnavailable as exc:
            log.warning("draft group %s unreadable: %s", r["draft_group"],
                        str(exc)[:80])
            continue
        if len(b) < 20:
            continue

        # Not every NHL "draft group" is a salary-cap slate. The lobby also
        # lists pick'em and tiers contests, and their rows carry the same
        # field names with completely different meanings - tonight four of
        # nine came back reading "salaries $0-$0" and "$1-$272", which is a
        # rank or a tier index, not money.
        #
        # A board like that builds a lineup that cannot be entered anywhere.
        # The cheapest real NHL skater is about $2,000, so anything whose
        # dearest player costs less than a thousand is not a cap game.
        top = pd.to_numeric(b["salary"], errors="coerce").max()
        if not (top and top >= 1000):
            log.info("draft group %s is not a salary-cap slate (dearest "
                     "player $%s) - skipped", r["draft_group"],
                     f"{0 if pd.isna(top) else int(top):,}")
            continue

        out.append((int(r["draft_group"]), str(r["example"]), b, r["starts"]))
    out.sort(key=lambda t: t[2]["team"].nunique(), reverse=True)
    return out


# --------------------------------------------------------------- projecting
def project_side(hist: pd.DataFrame, which: str) -> pd.DataFrame:
    """Fit one side of the sport and project every player's next game.

    `latest_rows` is the last feature row per player, which describes games
    already played. That is the point: the upcoming game has not happened, so
    a projection is made from what the player has done, never from a row about
    tonight.
    """
    spec = NS.SPECS[which]
    built = NS.build(hist, which)
    fitted = EM.Projections(spec).fit(built)
    latest = EM.latest_rows(built)
    proj = fitted.predict(latest)
    for c in ("player_id", "name", "team", "position"):
        if c in latest.columns and c not in proj.columns:
            proj[c] = latest[c].to_numpy()
    log.info("%s: fitted on %d rows, projected %d players",
             which, getattr(fitted, "trained_rows", len(built)), len(proj))
    return proj


def with_names(proj: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """Make sure every projected player carries a name.

    The join to DraftKings is BY NAME - MoneyPuck keys on the league's id and
    DraftKings on its own, with no crosswalk - so a projection without a name
    cannot reach a board at all.

    Older caches were written before the game logs carried one. Rather than
    force a twelve-minute re-fetch of every career to recover a column the
    season summary already holds, the names are looked up from the summary,
    which is one CSV per side. A cache written from here on carries them and
    this becomes a no-op.
    """
    if "name" in proj.columns and proj["name"].notna().any():
        return proj

    log.info("the cached history carries no player names - filling them from "
             "the season summary rather than re-fetching every career")
    table = {}
    for season in seasons:
        for side in ("skaters", "goalies"):
            try:
                who = ND.season_summary(season, side)
            except ND.DataUnavailable as exc:
                log.warning("no %s directory for %d (%s)", side, season,
                            str(exc)[:70])
                continue
            table.update(dict(zip(who["player_id"].astype(str), who["name"])))

    if not table:
        raise RuntimeError(
            "no player directory could be built, so the projections have no "
            "names and cannot be joined to a DraftKings board.")

    out = proj.copy()
    out["name"] = out["player_id"].astype(str).map(table)
    hit = int(out["name"].notna().sum())
    log.info("named %d of %d projected players (%.0f%%)",
             hit, len(out), 100 * hit / max(len(out), 1))
    return out[out["name"].notna()].copy()


def join_board(board: pd.DataFrame, proj: pd.DataFrame) -> pd.DataFrame:
    """The slate's players with their projections, joined by name.

    Measured, not assumed. The only key the two sources share is a normalised
    name, so this reports what matched, what did not, and how much of the
    slate's SALARY the misses represent - because ten missing minimum-priced
    fourth-liners is a different problem from two missing first-line wingers.
    """
    left = board.copy()
    right = proj.copy()
    if "norm" not in right.columns:
        right["norm"] = right["name"].map(ND.normalise)
    right = right.drop_duplicates("norm", keep="first")

    # THE BOARD WINS EVERY COLUMN IT ALREADY HAS.
    #
    # Dropped by overlap rather than by a hand-written list of three names,
    # because the hand-written list is a bug waiting for its column. When the
    # projection side gained an `opponent` column, pandas silently produced
    # `opponent_x` and `opponent_y`, the plain `opponent` ceased to exist, and
    # every game on the page was labelled "? v TOR" - a whole slate that
    # looked like a data outage and was in fact a name collision.
    #
    # The board is the right winner in every case: it is tonight's actual
    # draft group, so its team, position, salary and opponent describe the
    # game being played, while the projection's describe the game it was
    # built from.
    before = len(left)
    overlap = [c for c in right.columns if c in left.columns and c != "norm"]
    if overlap:
        log.info("the projection carries %d column(s) the board already has, "
                 "and the board's are kept: %s", len(overlap),
                 ", ".join(sorted(overlap)))
    out = left.merge(right.drop(columns=overlap), on="norm", how="left")
    if len(out) != before:
        raise RuntimeError(f"the projection join fanned out {before} rows "
                           f"to {len(out)}")

    have = out["median"].notna() if "median" in out.columns else pd.Series(
        False, index=out.index)
    salary = pd.to_numeric(out["salary"], errors="coerce").fillna(0)
    cover = float(have.mean())
    by_salary = float(salary[have].sum() / max(salary.sum(), 1))
    log.info("joined %d of %d players (%.0f%%), %.0f%% of slate salary",
             int(have.sum()), len(out), 100 * cover, 100 * by_salary)

    missed = out[~have].sort_values("salary", ascending=False)
    if len(missed):
        log.warning("%d players carry no projection. The dearest are: %s",
                    len(missed),
                    ", ".join(f"{r['name']} (${int(r['salary']):,})"
                              for _, r in missed.head(8).iterrows()))
    return out


# --------------------------------------------------------------- the lineups
def server_lineups(pool: pd.DataFrame, draws: np.ndarray,
                   own: pd.Series, field_size: int) -> dict:
    """The exact integer program's answer, checked against the site's rules.

    `check_entry` is not decoration. The optimiser understands the cap, the
    slots and a maximum per team; it does NOT understand that DraftKings
    requires skaters from three different teams, because that is a minimum and
    a minimum cannot be written as a maximum. A lineup that breaks it looks
    perfectly legal right up to the moment the entry is refused.
    """
    out = {}
    for objective in ("cash", "gpp"):
        try:
            built = O.build(pool, ROSTER, draws, objective=objective,
                            entries=1, own=own, field_size=field_size)
        except Exception as exc:                               # noqa: BLE001
            log.error("the %s integer program did not solve (%s: %s)",
                      objective, type(exc).__name__, str(exc)[:120])
            continue
        if not len(built):
            continue

        problems = NS.check_entry(built, ROSTER)
        if problems:
            log.error("the %s lineup CANNOT BE ENTERED: %s", objective,
                      "; ".join(problems))
        out[objective] = {
            "players": [str(n) for n in built["name"]],
            "salary": int(pd.to_numeric(
                built.get("charged", built["salary"])).sum()),
            "ceiling": round(float(pd.to_numeric(
                built["ceiling"], errors="coerce").sum()), 1),
            "illegal": problems,
        }
    return out


def num(v, places: int = 2, default=None):
    """A JSON-safe number, or `default`.

    NaN IS NOT VALID JSON. Python's json.dumps writes a bare `NaN` quite
    happily; JavaScript's JSON.parse throws a SyntaxError on it. The fetch
    resolves, the parse dies, nothing catches it, and the page sits on
    "Loading data/..." forever with no error anywhere on screen.

    A goalie has no `ewm_time_on_ice` and a skater has no save history, so on
    a board holding both this is guaranteed, not hypothetical.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f or f in (float("inf"), float("-inf")):   # NaN or infinite
        return default
    return round(f, places)


def game_label(team, opponent) -> str:
    """One label per game, whichever side asks for it."""
    a, b = str(team or "?"), str(opponent or "?")
    return " v ".join(sorted([a, b]))


def slate_games(board: pd.DataFrame, pool: pd.DataFrame) -> list[dict]:
    """Every game on the draft group, and whether anyone in it is projected.

    Derived from the BOARD, not the pool. Built from the pool, a game with no
    projected players simply vanishes - which is how a sixteen-team slate came
    out looking like a five-game one, with nothing on screen to say that ten
    clubs had been dropped on the way through.
    """
    have = {}
    if len(pool):
        have = pool.groupby("team").size().to_dict()
    seen, out = set(), []
    for _, r in board.iterrows():
        t, o = str(r["team"]), str(r.get("opponent") or "?")
        key = game_label(t, o)
        if key in seen:
            continue
        seen.add(key)
        out.append({"label": key, "teams": sorted([t, o]),
                    "players": int(have.get(t, 0)) + int(have.get(o, 0))})
    return sorted(out, key=lambda g: g["label"])


def slate_players(pool: pd.DataFrame, quantiles: list[float],
                  own: pd.Series, lev: pd.Series) -> list[dict]:
    qcols = [f"q{int(q * 100)}" for q in quantiles]
    rows = []
    for i, (_, r) in enumerate(pool.iterrows()):
        rows.append({
            "name": str(r["name"]),
            "pos": str(r["position"]),
            "team": str(r["team"]),
            "opp": str(r.get("opponent") or ""),
            # CANONICAL, so one game produces ONE label. Built as
            # "{opponent} v {team}" this was directional: a Vancouver player
            # gave "EDM v VAN" and an Edmonton player "VAN v EDM", so the
            # page listed one game twice and counted it twice. Sorting the
            # pair makes the label a property of the GAME rather than of
            # whichever side you happened to be looking from.
            "game": game_label(r["team"], r.get("opponent")),
            "salary": int(r["salary"]),
            # The quantile curve must never carry a hole - the simulator reads
            # it directly - so a missing one falls back to the median rather
            # than to null.
            "q": [num(r[c], 3, num(r["median"], 3, 0.0)) for c in qcols],
            "med": num(r["median"], 2, 0.0),
            "ceil": num(r["ceiling"], 2, 0.0),
            "own": num(own.iloc[i], 5, 0.0),
            "lev": num(lev.iloc[i], 3, 0.0),
            "pp": num(r.get("p_play", 1.0), 4, 1.0),
            # Ice time is hockey's opportunity number, the way a batting slot
            # is baseball's. Shipped so the board can be sorted by it. A
            # goalie has none, and null is what the page can render.
            "toi": num(r.get("ewm_time_on_ice"), 2),
            "pptoi": num(r.get("ewm_pp_time_on_ice"), 2),
            # Games in the last thirty days, and whether that is few enough
            # that he should not be in a lineup at all. DraftKings drops the
            # players it knows are out; this is the rest - the healthy
            # scratch, the long-term injury, the man sent down - who otherwise
            # show up as cheap value because their salary fell and their
            # projection, built from October, did not.
            "gp30": int(r["gp_30d"]) if pd.notna(r.get("gp_30d")) else None,
            "last": (str(r.get("last_played"))[:10]
                     if r.get("last_played") not in (None, "", float("nan"))
                     else None),
            # Last night, kept apart from the projection on purpose. `lsh` is
            # his slice of his team's shots in his most recent game and `ush`
            # the slice he usually takes, built from the games BEFORE it;
            # `mate` is the biggest night anyone else on his team had measured
            # against that man's own norm. Null where the published file does
            # not carry them, which the page renders and a zero cannot.
            "lsog": num(r.get("last_sog"), 0),
            "lsh": num(r.get("last_share"), 4),
            "ush": num(r.get("share_norm"), 4),
            "mate": num(r.get("mate_lift"), 2),
            "doubtful": bool(r.get("doubtful", False)),
            "status": "clear",
        })
    return rows


def caveats_for(pmeta: dict | None) -> list[str]:
    """What the page should say about itself, given where the numbers came
    from. The two sources have genuinely different weaknesses, and printing
    the wrong set is worse than printing none."""
    goalie = ("The starting goalie is not confirmed here. A backup who does "
              "not dress scores zero in a slot with no alternative.")
    own = ("Ownership is modelled and has never been graded against a real "
           "hockey field.")
    if pmeta is None:
        return [
            "Lines and power-play units are the player's own recent average, "
            "not tonight's announced deployment. A promotion to PP1 is "
            "invisible to this board.",
            goalie, own,
        ]
    out = [
        "Every rate is split by game state - power play, penalty kill and "
        "everything else - and scored by simulating the actual stat line, so "
        "DraftKings' threshold bonuses (five shots, three blocks, three "
        "points) are priced as the steps they are rather than smeared.",
        "Power-play MINUTES are projected from recent deployment. Tonight's "
        "announced units are still not known to this board - a promotion "
        "decided at this morning's skate is not in it.",
        goalie, own,
    ]
    if pmeta.get("days_stale"):
        out.insert(0, (
            f"THESE PROJECTIONS ARE {pmeta['days_stale']} DAY(S) OLD "
            f"({pmeta['date']}). Ice time is still roughly right; the "
            f"opponent adjustments in them are for the wrong games."))
    return out


def loadings_for_page() -> dict:
    out = {}
    for kind in ("game", "team", "compete"):
        table = {}
        for spec in (NS.SKATERS, NS.GOALIES):
            table.update(spec.loadings.get(kind, {}))
        out[kind] = table
    return out


def board_spec():
    """A spec whose loadings are keyed the way the BOARD is keyed.

    The simulator looks a player's loadings up by his `position`, and by the
    time the pool reaches it that column holds DraftKings roster slots - C, W,
    D, G - while the sport's own spec also lists LW and RW. Handing it the raw
    spec drops every winger onto a default loading, which would mean the
    server's two lineups were solved against a correlation structure the
    page's browser search does not share. The two are meant to be comparable;
    that is the entire point of shipping both.
    """
    import dataclasses
    return dataclasses.replace(
        NS.SKATERS,
        positions=["C", "W", "D", "G"],
        loadings=loadings_for_page())


def slate_label(pool: pd.DataFrame, starts, raw: str) -> str:
    teams = pool["team"].nunique()
    when = ""
    if starts is not None and pd.notna(starts):
        when = pd.Timestamp(starts).tz_convert(EASTERN).strftime("%-I:%M%p")
    bits = [f"{teams} teams"]
    if when:
        bits.append(when.lower())
    if raw:
        bits.append(raw.strip())
    return " · ".join(bits)


# ------------------------------------------------------------------- output
def write(payloads: list[dict]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    archive = DATA / "archive"

    keep = set()
    slates = []
    for p in payloads:
        name = f"{SPORT}_dk_{p['kind']}_{p['draft_group']}.json"
        # allow_nan=False, and that is the whole point of writing it out.
        #
        # By default Python emits a bare `NaN`, which is not valid JSON and
        # which JavaScript's JSON.parse refuses. The file writes, the commit
        # succeeds, the run goes green, and the page sits on "Loading ..."
        # forever with nothing on screen to say why. Refusing here turns a
        # silent dead page into a loud build failure.
        try:
            body = json.dumps(p, indent=1, allow_nan=False)
        except ValueError as exc:
            log.error("draft group %s produced a payload that is not valid "
                      "JSON (%s). It is NOT being written - a file the page "
                      "cannot parse is worse than no file, because the page "
                      "gives no sign of why it is stuck.",
                      p["draft_group"], exc)
            continue
        (DATA / name).write_text(body)
        keep.add(name)
        slates.append({"sport": SPORT, "site": "dk", "kind": p["kind"],
                       "label": p["label"], "file": f"data/{name}",
                       "locks": p["locks"]})
        log.info("wrote %s (%d players)", name, len(p["players"]))

    # Archived, never deleted. A board that is gone cannot be graded against a
    # finished contest, and grading published boards against real results is
    # the only feedback loop this project has.
    for old in DATA.glob(f"{SPORT}_dk_*.json"):
        if old.name in keep:
            continue
        try:
            when = json.loads(old.read_text()).get("generated_at", "")[:10]
        except Exception:                                      # noqa: BLE001
            when = ""
        when = when or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        dest = archive / when
        dest.mkdir(parents=True, exist_ok=True)
        old.replace(dest / old.name)
        log.info("archived %s -> archive/%s/", old.name, when)

    cutoff = datetime.now(timezone.utc) - timedelta(days=ARCHIVE_DAYS)
    for day in sorted(archive.glob("[0-9]" * 4 + "-*")):
        try:
            on = datetime.strptime(day.name, "%Y-%m-%d").replace(
                tzinfo=timezone.utc)
        except ValueError:
            continue
        if on < cutoff:
            for f in day.glob("*.json"):
                f.unlink()
            day.rmdir()

    # The manifest is REBUILT from every slate file present, not just the ones
    # this run produced - otherwise publishing hockey would hide baseball.
    listed = []
    for f in sorted(DATA.glob("*_dk_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:                                      # noqa: BLE001
            continue
        listed.append({"sport": d.get("sport", "?"), "site": d.get("site", "dk"),
                       "kind": d.get("kind", "classic"),
                       "label": d.get("label", ""), "file": f"data/{f.name}",
                       "locks": d.get("locks")})
    (DATA / "manifest.json").write_text(json.dumps({
        "updated_at": payloads[0]["generated_at"],
        "slates": listed,
    }, indent=1))
    log.info("manifest lists %d slate(s) across %d sport(s)", len(listed),
             len({s["sport"] for s in listed}))


# --------------------------------------------------------------------- main
def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draft-group", type=int, default=None)
    p.add_argument("--first-season", type=int, default=2025)
    p.add_argument("--field", type=int, default=100_000)
    p.add_argument("--sims", type=int, default=20_000)
    p.add_argument("--slates", type=int, default=6)
    p.add_argument("--look", type=int, default=12)
    p.add_argument("--date", default=None)
    p.add_argument("--min-salary-pct", type=float, default=99.0,
                   help="the least a LINEUP may spend, %% of the $50k cap. "
                        "Unspent salary is points declined; 0 turns it off.")
    p.add_argument("--refresh-season", type=int, default=None,
                   help="re-fetch this season's history even if cached")
    p.add_argument("--stat-projections", default="on", choices=("on", "off"),
                   help="read the published per-stat projections and convert "
                        "them to DraftKings points, instead of fitting this "
                        "repo's own quantile model. ON by default: the stat "
                        "model splits every rate by game state, which this "
                        "one does not, and simulating counts is the only way "
                        "to price DraftKings' threshold bonuses at all.")
    p.add_argument("--projections-url", default=NP.BASE,
                   help="where the published stat projections live")
    args = p.parse_args(argv)

    pct = max(0.0, min(100.0, float(args.min_salary_pct))) / 100.0
    ROSTER["min_salary_pct"] = pct
    log.info("lineup salary floor $%s of $%s",
             f"{int(pct * ROSTER['salary_cap']):,}",
             f"{ROSTER['salary_cap']:,}")

    # A hockey season is named for the year it STARTS. On 29 September 2026
    # the 2026-27 season has zero games played, so the model necessarily fits
    # on 2025-26 until a few nights are in the books. Said out loud because a
    # model quietly trained on last season's linemates is worth knowing about.
    now = datetime.now(timezone.utc)
    latest = now.year if now.month >= 8 else now.year - 1
    seasons = list(range(args.first_season, latest + 1))

    day_hint = args.date or pinned_slate_day(args.draft_group) or next_slate_day()

    # NOTHING is refreshed unless asked for, and that is deliberate.
    #
    # This defaulted to refreshing the current season on every publish, which
    # is wrong twice over. On an opening night it demands a season MoneyPuck
    # has no data for and the build dies. Once that season does have data, it
    # re-downloads a thousand careers on every run - a twelve-minute job
    # wearing a two-minute job's name, which is exactly the failure the
    # workflow's own cache guard warns about.
    #
    # Keeping the current season fresh is the FETCH workflow's job, on its own
    # schedule. Publishing reads what is there.
    refresh = [args.refresh_season] if args.refresh_season else []
    day = day_hint
    log.info("building for %s", day)

    # The published stat model first.
    #
    # It is a better projection than this repo can make, for two reasons that
    # are structural rather than a matter of tuning. It splits every rate by
    # GAME STATE - power play, penalty kill, everything else - so a promotion
    # to the top unit is visible to it and invisible here; that is the largest
    # single move a hockey projection makes and this file's own caveats admit
    # it cannot see it. And converting stats to points by SIMULATING COUNTS
    # prices DraftKings' threshold bonuses exactly, where a quantile fitted on
    # a points total cannot represent a step function at five shots at all.
    #
    # It is also a smaller dependency: reading one dated CSV replaces loading
    # a thousand cached careers, so the whole history fetch below only happens
    # when this path fails.
    proj, pmeta, hist = None, None, None
    if args.stat_projections == "on":
        try:
            proj, pmeta = NP.build(day, sims=args.sims,
                                   base=args.projections_url)
            log.info("using published stat projections from %s (%d players, "
                     "%d day(s) stale)", pmeta["date"], pmeta["players"],
                     pmeta["days_stale"])
        except Exception as exc:                               # noqa: BLE001
            log.error("the published stat projections could not be used (%s: "
                      "%s). Falling back to this repo's own model, which does "
                      "NOT split by game state.", type(exc).__name__,
                      str(exc)[:160])

    if proj is None:
        log.info("history: seasons %s%s", seasons,
                 f" (refreshing {refresh})" if refresh else " (cache as-is)")
        hist = NF.load(seasons, refresh=refresh)
        log.info("%d player-games of history", len(hist))

        this_season = hist[hist["season"] == latest] if "season" in hist else hist
        if len(this_season) < 200:
            log.warning("only %d player-games from the %d-%d season so far, so "
                        "this board is essentially last season's model. Early-"
                        "season lines and roles are not in it yet.",
                        len(this_season), latest, latest + 1)

        skaters = project_side(hist, "skaters")
        goalies = project_side(hist, "goalies")
        proj = with_names(pd.concat([skaters, goalies], ignore_index=True),
                          seasons)

    boards = candidate_slates(args.draft_group, args.look)
    if not boards:
        sys.exit("no readable NHL draft group is on sale right now, so "
                 "nothing is published and the page keeps what it had.")

    payloads = []
    for dg, label, board, starts in boards[:args.slates]:
        try:
            board = board.copy()
            board["position"] = board["position"].map(roster_position)
            unknown = board["position"].isna().sum()
            if unknown:
                log.info("dropped %d players whose position is not a slot on "
                         "this roster", int(unknown))
            board = board[board["position"].notna()]

            merged = join_board(board, proj)
            pool = merged[merged["median"].notna()].copy()
            pool = pool[pd.to_numeric(pool["salary"],
                                      errors="coerce").notna()]
            if len(pool) < 40:
                log.error("draft group %s: only %d players survived the join, "
                          "which is too few to build a legal lineup from. "
                          "Skipped.", dg, len(pool))
                continue

            # A COUNT IS NOT COVERAGE. 109 players out of 413 cleared the
            # forty-player bar comfortably and was still a broken slate: it
            # was six clubs out of sixteen, because the projections file was
            # for a different night and only the overlapping teams joined.
            # The page then showed a sixteen-team slate as five games with no
            # hint that ten clubs had been dropped.
            board_teams = set(board["team"].astype(str))
            per = pool.groupby("team").size()
            full = {t for t in board_teams if int(per.get(t, 0)) >= 8}
            share = len(full) / max(len(board_teams), 1)
            if share < 0.75:
                empty = sorted(board_teams - full)
                log.error("draft group %s: only %d of %d clubs have a usable "
                          "set of projections (%.0f%%). Missing or nearly "
                          "empty: %s. This is what a slate built from the "
                          "WRONG NIGHT's projections looks like, so it is "
                          "skipped rather than published half full.",
                          dg, len(full), len(board_teams), 100 * share,
                          ", ".join(empty))
                continue
            if share < 1.0:
                log.warning("draft group %s: %s have few or no projections and "
                            "will be thin on the board",
                            dg, ", ".join(sorted(board_teams - full)))

            missing = [s for s in set(ROSTER["slots"]) if s != "UTIL"
                       and not (pool["position"] == s).any()]
            if missing:
                log.error("draft group %s has nobody at %s, so no legal "
                          "lineup exists. Skipped.", dg, missing)
                continue

            quantiles = list(NS.SKATERS.quantiles)
            own = OWN.project(pool, ROSTER)
            lev = OWN.leverage(pool, own)
            draws = S.simulate(pool, quantiles, args.sims, spec=board_spec())

            payloads.append({
                "generated_at": now.isoformat(timespec="seconds"),
                "sport": SPORT,
                "site": "dk",
                "kind": "classic",
                "draft_group": dg,
                "label": slate_label(pool, starts, label),
                "locks": (pd.Timestamp(starts).isoformat()
                          if starts is not None and pd.notna(starts) else None),
                "field_size": args.field,
                "quantiles": quantiles,
                "loadings": loadings_for_page(),
                "roster": ROSTER,
                "players": slate_players(pool, quantiles, own, lev),
                "games": slate_games(board, pool),
                "server_lineups": server_lineups(pool, draws, own, args.field),
                # The page prints these verbatim. A model that cannot see
                # tonight's lines should say so where somebody will read it,
                # not in a log nobody opens.
                "caveats": caveats_for(pmeta),
            })
        except Exception as exc:                               # noqa: BLE001
            # One bad board must not take the others down. The baseball build
            # once discarded fourteen good slates because a single dead one
            # raised past the handler.
            log.error("draft group %s failed (%s: %s) - the other boards are "
                      "unaffected", dg, type(exc).__name__, str(exc)[:160])
            continue

    if not payloads:
        sys.exit("no board could be built, so nothing is published and the "
                 "page keeps what it had.")
    write(payloads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
