"""One live baseball slate, turned into the file the page reads.

    python mlb_publish.py                    # the busiest live classic slate
    python mlb_publish.py --draft-group 1234
    python mlb_publish.py --field 50000

The page is the same page the football build uses, and it does the heavy work
in the browser: it rebuilds the factor model from the loadings, simulates the
slate, and solves for lineups under whatever locks and fades you set. So this
file's whole job is to emit a JSON document of exactly the shape that page
expects, plus a manifest listing the slates available.

The contract is read off the page's own JavaScript rather than guessed:

    docs/data/manifest.json
        {updated_at, slates: [{sport, site, kind, label, file}]}

    docs/data/<file>.json
        {generated_at, field_size, quantiles, loadings, roster,
         players: [{name, pos, team, game, salary, q, med, ceil, own, lev,
                    status}],
         server_lineups: {cash: {...}, gpp: {...}}}

Two positions vocabularies, kept apart on purpose
-------------------------------------------------
The history speaks box-score positions - LF, CF, RF, DH - and DraftKings
speaks roster positions, where all three outfielders are OF. Translating one
into the other before the fit would corrupt the model's own position dummies;
translating after it would leave the roster rules unenforceable.

So they never meet. The projection is made on HISTORY rows carrying history
positions, and only then joined onto the board, which keeps its own. The board
position is what reaches the page, because that is what the roster rules are
written in.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import numpy as np
import pandas as pd

import mlb_data as MD
import mlb_sport as MS
import mlb_ownership_file as OWF
import mlb_statcast as MSC
from engine import cache as C
from engine import model as M
from engine import optimise as O
from engine import ownership as OWN
from engine import simulate as S

log = logging.getLogger("mlb_publish")

SPORT = "mlb"
DOCS = Path("docs")
DATA = DOCS / "data"

# How long a published board is kept after its slate is over. Results
# arrive days later - a contest export downloaded on Friday grades a slate
# played on Tuesday - so the board has to outlive the day it was for.
ARCHIVE_DAYS = 120

# DraftKings MLB Classic. Ten slots, $50,000, and no more than five hitters
# from any one team.
#
# Two things are deliberately STRICTER here than DraftKings actually requires,
# because a lineup that is too constrained can still be entered and one that is
# not constrained enough cannot:
#
#   * `max_per_team` is applied to every player rather than to hitters only,
#     so a lineup can never exceed the real limit;
#   * a player eligible at several positions is pinned to the first one
#     DraftKings lists, so the optimiser never uses eligibility it might have
#     read wrong.
#
# Both cost a little optimality and neither can produce a rejected entry.
ROSTER = {
    "slots": ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"],
    "flex_positions": [],
    "salary_cap": 50_000,
    "max_per_team": 5,
}

# Everyone who pitches fills a P slot, whatever DraftKings calls him.
PITCHER_CODES = {"P", "SP", "RP"}
HITTER_SLOTS = {"C", "1B", "2B", "3B", "SS", "OF"}


def roster_position(raw: str) -> str | None:
    """A DraftKings position string, reduced to the slot it can fill.

    DraftKings writes multi-eligibility with a slash - "1B/OF" - and the first
    listed position is the one used. A player whose position matches no slot
    at all is dropped rather than guessed at.
    """
    first = str(raw or "").split("/")[0].strip().upper()
    if first in PITCHER_CODES:
        return "P"
    return first if first in HITTER_SLOTS else None


def _id_text(s: pd.Series) -> pd.Series:
    """A player id as text, via a number, and never via `astype(str)`.

    A column of integers that contains a single missing value becomes float64
    in pandas, and `str(658796.0)` is "658796.0" - which does not equal
    "658796" and never will. That is the whole explanation for a join that
    matched 0 of 278 rows on an id both sides genuinely carry.

    It failed silently because the name fallback picked up 255 of them, so the
    only visible symptom was a log line nobody had to act on. This is the
    "a 96% join looks fine and is not" failure, in its exact original form.

    Missing stays MISSING - never the empty string. An empty string is a
    value, and a merge happily matches it against every other empty string on
    the far side, so one board player without an id fans out into a row for
    every projection without one. Null never matches null, which is exactly
    the behaviour wanted here.
    """
    num = pd.to_numeric(s, errors="coerce")
    return num.astype("Int64").astype("string")


# Plate appearances by batting-order slot, per nine-inning game.
#
# A team takes about 38 plate appearances. They are handed out in order, so
# each slot down the card loses roughly an eighth of a turn: leading off is
# about 4.7 and batting ninth about 3.9. That is a ~20% difference in how many
# chances a man gets, which is the single largest thing separating one hitter's
# day from another's - larger than almost any difference in ability between two
# players on the same board.
#
# These are league averages, not a fitted effect. The right version measures
# the slot's effect from history, and cannot be built until the history carries
# a lineup slot, which it does not yet. So this is arithmetic applied on top of
# a projection rather than something the model learned - stated plainly here
# and on the page, because the difference matters.
PA_BY_SLOT = {1: 4.72, 2: 4.61, 3: 4.50, 4: 4.39, 5: 4.28,
              6: 4.17, 7: 4.06, 8: 3.95, 9: 3.84}

# A projection is never moved more than this. The multiplier divides by the
# player's own recent plate appearances, and a small or noisy denominator can
# otherwise produce a wild number from a arithmetic mistake rather than from
# information.
ORDER_CLIP = (0.75, 1.30)

# Columns that scale with opportunity. p_play does not - where a man bats has
# nothing to do with whether he is in the lineup, and that question has already
# been answered by the lineup card itself.
SCALED = ["q10", "q25", "q50", "q75", "q90", "q97",
          "median", "cond_mean", "mean", "ceiling"]


def tonight_opposing_starter(hist: pd.DataFrame, probables: dict
                             ) -> pd.DataFrame:
    """For each team playing tonight, the form of the starter they FACE.

    Returns team -> (opp_sp_k_rate, opp_sp_baserunners).

    Without this the model is handed a real coefficient and the wrong
    pitcher. `latest_rows` gives each hitter the row from his LAST game, and
    the opposing-starter columns on that row describe the pitcher he faced
    then - which is exactly the pitcher he is not facing tonight. Fitting the
    feature and failing to refresh it at the board would be worse than not
    having it: the model would confidently adjust every hitter for the wrong
    man.
    """
    ids = probables.get("ids") or {}                 # pitcher id -> his team
    opponent_of = probables.get("opponent_of") or {}  # team -> team tonight
    if not ids or not opponent_of:
        log.warning("no probable pitchers or no schedule for tonight, so the "
                    "opposing-starter feature cannot be refreshed and is "
                    "being cleared rather than left stale")
        return pd.DataFrame(columns=["team", "opp_sp_k_rate",
                                     "opp_sp_baserunners"])

    form = MS.current_starter_form(hist)
    form["opp_sp_id"] = form["opp_sp_id"].astype(str)
    by_pitcher = form.set_index("opp_sp_id")

    rows = []
    for pid, pitcher_team in ids.items():
        facing = opponent_of.get(str(pitcher_team))
        if not facing or str(pid) not in by_pitcher.index:
            continue
        r = by_pitcher.loc[str(pid)]
        rows.append({"team": str(facing),
                     "opp_sp_k_rate": float(r["opp_sp_k_rate"]),
                     "opp_sp_baserunners": float(r["opp_sp_baserunners"])})
    out = pd.DataFrame(rows).drop_duplicates("team")
    log.info("tonight's opposing starter known for %d of %d teams on the "
             "schedule", len(out), len(opponent_of))
    return out


def project_half(hist: pd.DataFrame, which: str,
                 opp_tonight: pd.DataFrame | None = None,
                 statcast=None) -> pd.DataFrame:
    """Fit one half of the sport and project every player's next outing."""
    spec = MS.SPECS[which]
    built = MS.build(hist, which, statcast)
    proj = M.Projections(spec).fit(built)
    latest = M.latest_rows(built)

    if which == "hitters" and "opp_sp_k_rate" in latest.columns:
        # Whatever happens, the stale value must not survive - it describes
        # the wrong pitcher. Cleared first, then refilled where tonight's
        # starter is known; a hitter whose opponent has not named a starter
        # is left missing, which the model handles, rather than carrying a
        # number that is confidently about somebody else.
        latest = latest.copy()
        latest["opp_sp_k_rate"] = np.nan
        latest["opp_sp_baserunners"] = np.nan
        if opp_tonight is not None and len(opp_tonight):
            m = opp_tonight.set_index("team")
            t = latest["team"].astype(str)
            for c in ("opp_sp_k_rate", "opp_sp_baserunners"):
                latest[c] = t.map(m[c]).astype(float)
            known = float(latest["opp_sp_k_rate"].notna().mean())
            log.info("hitters: tonight's opposing starter attached to %.0f%% "
                     "of them", 100 * known)
        else:
            log.warning("hitters: NO opposing starter for tonight - every "
                        "hitter is projected as if the pitcher were unknown")

    q = proj.predict(latest)
    keep = ["player_id", "name", "team", "position"]
    # Carried so the batting order can be applied RELATIVE to what this player
    # has actually been getting, rather than to a league average. A man who
    # already leads off should not be paid twice for leading off tonight.
    if "ewm_plate_appearances" in latest.columns:
        keep.append("ewm_plate_appearances")
    out = latest[keep].copy()
    for c in q.columns:
        out[c] = q[c].to_numpy()
    out["half"] = which
    log.info("%s: fitted on %d rows, projected %d players",
             spec.name, proj.trained_rows, len(out))
    return out


def join_board(board: pd.DataFrame, proj: pd.DataFrame) -> pd.DataFrame:
    """Board joined to projections on the LEAGUE's player id.

    This is an integer comparison, not a name match, and that is the whole
    reason the box-score fetch bothered to carry `mlb_id`. Names are kept only
    as a fallback and reported separately, because a name join that quietly
    works at 80% is how the players who changed teams disappear.
    """
    left = board.copy()
    left["mlb_id"] = _id_text(left["mlb_id"])
    right = proj.copy()
    right["player_id"] = _id_text(right["player_id"])

    n_in = len(left)
    merged = left.merge(right.drop(columns=["team", "name"]),
                        left_on="mlb_id", right_on="player_id", how="left")

    # A left join must not change the row count. If it does, both sides shared
    # a key that repeats - and the board would carry a player several times,
    # each copy with a different projection, which is how an optimiser ends up
    # fielding the same man twice or filling a slate with rows nobody put on
    # it. The college build lost 700 rows to 28,350 this way.
    if len(merged) != n_in:
        raise SystemExit(
            f"the projection join fanned {n_in} board rows into "
            f"{len(merged)}. Duplicate ids on one side; the board is not "
            f"trustworthy and nothing has been published.")
    by_id = int(merged["player_id"].notna().sum())

    miss = merged["player_id"].isna()
    if miss.any():
        keys = {MD.normalise(n): p for n, p in
                zip(proj["name"], proj["player_id"])}
        found = merged.loc[miss, "name"].map(
            lambda n: keys.get(MD.normalise(n)))
        take = found.notna()
        if take.any():
            fill = proj.set_index(proj["player_id"].astype(str))
            for idx, pid in found[take].items():
                row = fill.loc[str(pid)]
                for c in row.index:
                    if c in merged.columns and c not in ("team", "name"):
                        merged.at[idx, c] = row[c]
        log.info("join: %d of %d on MLB id, %d more on name",
                 by_id, len(merged), int(take.sum()))
    else:
        log.info("join: %d of %d on MLB id", by_id, len(merged))
    return merged


def slate_players(merged: pd.DataFrame, spec_quantiles: list[float],
                  own: pd.Series, lev: pd.Series) -> list[dict]:
    qcols = [f"q{int(round(q * 100)):02d}" for q in spec_quantiles]
    rows = []
    for i, r in merged.reset_index(drop=True).iterrows():
        rows.append({
            "name": str(r["name"]),
            # `position` by this point, not `slot` - the engine's own column
            # name, because the pool has already been renamed for it.
            "pos": str(r["position"]),
            "team": str(r["team"]),
            "game": str(r.get("game") or ""),
            # Who he is facing. Without this the solver cannot know that a
            # hitter and the pitcher he is batting against are the same bet
            # twice, in opposite directions.
            "opp": str(r.get("opponent") or ""),
            "salary": int(r["salary"]),
            "q": [round(float(r[c]), 3) for c in qcols],
            "med": round(float(r["median"]), 2),
            "ceil": round(float(r["ceiling"]), 2),
            "own": round(float(own.iloc[i]), 5),
            "lev": round(float(lev.iloc[i]), 3),
            # Batting order where the card is posted, and an honest blank
            # where it is not. The page shows the difference rather than
            # letting a rested hitter look identical to a confirmed leadoff.
            "bat": (int(r["bat"]) if pd.notna(r.get("bat")) else None),
            # The probability he takes part at all. The browser gates on this
            # exactly as the server's simulator does; without it an unconfirmed
            # player simulates as though he is certain to play.
            "pp": round(float(r.get("p_play", 1.0) or 1.0), 4),
            "of": round(float(r.get("order_factor", 1.0) or 1.0), 3),
            "status": ("clear" if str(r["position"]) == "P"
                       or pd.notna(r.get("bat")) else "unconfirmed"),
        })
    return rows


def loadings_for_page() -> dict:
    """One loadings table keyed by the positions the PAGE will see.

    The specs are keyed by box-score positions; the board is keyed by roster
    positions. The page looks these up by whatever `pos` each player carries in
    the JSON, so they are rewritten here into that vocabulary rather than left
    to fall through to a default that would silently flatten every correlation
    in the slate.
    """
    out: dict[str, dict[str, float]] = {}
    for kind in ("game", "team", "compete"):
        table = {}
        hit = MS.HITTERS.loadings.get(kind, {})
        # Hitter loadings do not vary by position - a catcher and a centre
        # fielder sit in the same batting order - so any one of them stands
        # for the lot.
        if hit:
            value = float(next(iter(hit.values())))
            for slot in sorted(HITTER_SLOTS):
                table[slot] = value
        pit = MS.PITCHERS.loadings.get(kind, {})
        if pit:
            table["P"] = float(pit.get("P", next(iter(pit.values()))))
        out[kind] = table
    return out


def apply_batting_order(pool: pd.DataFrame) -> pd.DataFrame:
    """Scale every hitter's distribution by the slot he is actually batting in.

    The projection is built from a player's own recent games, which already
    reflect wherever he has been batting. So the adjustment is a RATIO - the
    plate appearances tonight's slot is worth, over the plate appearances he
    has lately been getting - and not a raw slot factor. A regular leadoff man
    confirmed to lead off comes out at about 1.0, which is right; paying him a
    leadoff bonus on top of a projection already built from leadoff games would
    count the same thing twice.

    A hitter promoted from eighth to first moves about 4.72/3.95 = 1.19.
    A hitter dropped the other way moves about 0.84. Those are real and they
    are roughly the size of the gap between a good hitter and an average one.

    Pitchers and unconfirmed hitters are untouched: no slot, no adjustment.
    """
    out = pool.copy()
    out["order_factor"] = 1.0
    if "bat" not in out.columns:
        return out

    has = out["bat"].notna() & (out["slot"] != "P")
    if not has.any():
        log.info("no confirmed batting orders - projections unadjusted")
        return out

    want = out.loc[has, "bat"].map(lambda s: PA_BY_SLOT.get(int(s)))
    base = pd.to_numeric(out.loc[has].get("ewm_plate_appearances"),
                         errors="coerce")
    # A man with no plate-appearance history gets the middle of the card as his
    # baseline, which makes the adjustment his slot against an average one.
    base = base.fillna(float(np.mean(list(PA_BY_SLOT.values()))))
    base = base.clip(lower=2.5)

    factor = (want / base).clip(*ORDER_CLIP)
    out.loc[has, "order_factor"] = factor.to_numpy()

    for c in SCALED:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") * out["order_factor"]

    moved = out.loc[has].assign(f=factor.to_numpy())
    moved = moved.reindex(moved["f"].sub(1).abs().sort_values(
        ascending=False).index)
    log.info("batting order applied to %d hitters (mean factor %.3f)",
             int(has.sum()), float(factor.mean()))
    for r in moved.head(8).itertuples(index=False):
        log.info("    %-22s bats %s  x%.2f", str(r.name)[:22],
                 int(r.bat), float(r.f))
    return out


def slate_label(pool: pd.DataFrame, starts, raw: str) -> str:
    """A name that says which board this is.

    Games and lock time, because those are what distinguish an early slate
    from the main one; DraftKings' contest name is kept on the end because it
    is the string you will see in their lobby.
    """
    games = int(pool["game"].nunique()) if "game" in pool.columns else 0
    when = ""
    if starts is not None and pd.notna(starts):
        t = starts.tz_convert(EASTERN)
        when = " · locks " + t.strftime("%-I:%M%p ET").lower().replace(
            "am", "am").replace("pm", "pm")
    tail = str(raw or "").strip()
    for cut in (" [", " ["):
        if cut in tail:
            tail = tail.split(cut)[0]
    return f"{games} games{when} · {tail[:38]}"


def board_spec():
    """A spec whose loadings are keyed the way the BOARD is keyed.

    The simulator looks a player's loadings up by his `position`, and by the
    time the pool reaches it that column holds DraftKings roster positions -
    OF and P - while the sport's own spec is keyed by box-score positions -
    LF, CF, RF, DH, SP, RP. Handing it the raw spec silently dropped every
    outfielder and every pitcher onto a default loading, which is to say the
    server's own two lineups were solved against a correlation structure the
    page's browser search does not share. The two are supposed to be
    comparable; that is the entire point of shipping both.
    """
    import dataclasses
    return dataclasses.replace(
        MS.HITTERS,
        positions=sorted(HITTER_SLOTS | {"P"}),
        loadings=loadings_for_page())


def server_lineups(pool: pd.DataFrame, draws: np.ndarray,
                   own: pd.Series, field_size: int) -> dict:
    """The exact integer program's answer, shipped alongside the browser's.

    The page says this is from the last scheduled run and solved exactly. It
    is the only thing on the page the browser's own near-optimal search can be
    checked against, so a failure here is reported and left empty rather than
    filled with the browser's kind of answer wearing the server's label.
    """
    out = {}
    for objective in ("cash", "gpp"):
        try:
            built = O.build(pool, ROSTER, draws, objective=objective,
                            entries=1, own=own, field_size=field_size)
        except Exception as exc:                           # noqa: BLE001
            log.error("the %s integer program did not solve (%s: %s)",
                      objective, type(exc).__name__, str(exc)[:120])
            continue
        if not len(built):
            continue
        out[objective] = {
            "players": [str(n) for n in built["name"]],
            "salary": int(pd.to_numeric(built.get("charged",
                                                  built["salary"])).sum()),
            "ceiling": round(float(pd.to_numeric(
                built["ceiling"], errors="coerce").sum()), 1),
        }
    return out


EASTERN = ZoneInfo("America/New_York")


def next_slate_day(now: pd.Timestamp | None = None) -> str:
    """The date of the next DraftKings baseball slate that has not locked.

    Baseball's day does not end at midnight and a lobby does not care what
    day it is. What matters is which slate a person can still enter, and after
    the last first pitch of an evening that is tomorrow's. Reading it off the
    lock times means the answer is right at four in the afternoon and right
    again at eleven at night, without either case being special-cased.
    """
    listed = MD.slates()
    now = now or pd.Timestamp.now(tz="UTC")
    live = listed[listed["starts"].notna() & (listed["starts"] > now)]
    if live.empty:
        day = now.tz_convert(EASTERN).strftime("%Y-%m-%d")
        log.warning("no slate on sale has a future lock time - falling back "
                    "to the Eastern date, %s", day)
        return day
    soonest = live["starts"].min()
    day = soonest.tz_convert(EASTERN).strftime("%Y-%m-%d")
    log.info("next slate locks %s (%s Eastern) - building for %s",
             soonest.strftime("%Y-%m-%d %H:%M UTC"),
             soonest.tz_convert(EASTERN).strftime("%H:%M"), day)
    return day


def candidate_slates(draft_group: int | None, look: int,
                     probables: dict) -> list[tuple]:
    """Which boards to publish, biggest first.

    Ranking by contest COUNT - which is what the football build does - picked
    a three-game early slate over the main evening board, because cheap early
    contests are numerous. Baseball's useful slate is the one with the most
    games in it, and the only way to know how many games a draft group covers
    is to fetch it, so the top few by contest count are fetched and then
    re-sorted by how many teams they actually contain.

    Every one that survives is published. The page already has a slate picker;
    filling it is more useful than guessing which single board you wanted.
    """
    listed = MD.slates()
    if listed.empty:
        sys.exit("DraftKings is listing no baseball slates right now")

    if draft_group:
        row = listed[listed["draft_group"] == draft_group]
        label = str(row["example"].iloc[0]) if len(row) else "(given)"
        when = row["starts"].iloc[0] if len(row) else pd.NaT
        return [(int(draft_group), label, MD.board(int(draft_group)), when)]

    classic = listed[~listed["game_type"].astype(str)
                     .str.contains("showdown", case=False, na=False)]

    # A slate that has already locked cannot be entered, so it is not a
    # candidate no matter how big it is.
    now = pd.Timestamp.now(tz="UTC")
    before = len(classic)
    open_now = classic[classic["starts"].isna() | (classic["starts"] > now)]
    if len(open_now) < before:
        log.info("%d classic slate(s) have already locked and are ignored",
                 before - len(open_now))
    classic = open_now

    # Which day a board is for, decided by whether ITS players are the ones
    # the league has named for TODAY.
    #
    # DraftKings sells tomorrow's slates today, and tomorrow's main slate has
    # more games in it than whatever is left of this afternoon. Ranking by
    # size therefore picks tomorrow, confidently, every evening - which is how
    # a board came back showing BOS @ TB on a night the Athletics were at
    # Tampa Bay.
    #
    # No date is parsed and no timestamp is trusted. Today's probable starters
    # and today's posted lineups are already in hand, so the test is simply
    # how many of a board's players are in them. Tomorrow's board scores
    # nearly zero against today's card, which is exactly the signal wanted.
    today_keys = (set(probables["ids"]) | set(probables["order_id"]))
    today_names = (set(probables["names"]) | set(probables["order_name"]))

    out = []
    for r in classic.sort_values(
            ["starts", "contests"], ascending=[True, False]
    ).head(look).itertuples(index=False):
        try:
            board = MD.board(int(r.draft_group))
        except Exception as exc:                               # noqa: BLE001
            log.info("draft group %s did not load (%s)", r.draft_group,
                     str(exc)[:70])
            continue
        teams = int(board["team"].nunique())
        if teams < 2:
            continue
        ids = _id_text(board["mlb_id"]).fillna("")
        norm = board["name"].map(MD.normalise)
        overlap = int((ids.isin(today_keys) | norm.isin(today_names)).sum())
        out.append((int(r.draft_group), str(r.example), board, teams,
                    overlap, r.starts))

    if not out:
        sys.exit("no baseball board could be loaded")

    log.info("boards considered:")
    for dg, label, _, teams, overlap, starts in sorted(out, key=lambda t: -t[4]):
        when = (starts.tz_convert(EASTERN).strftime("%H:%M ET")
                if pd.notna(starts) else "  ?  ")
        log.info("    %-8s %2d teams  %4d named  locks %s  %s",
                 dg, teams, overlap, when, label[:40])

    # At least half a board's teams must have someone on today's card. A board
    # for another day comes nowhere near that.
    today = [t for t in out if t[4] >= max(2, t[3] // 2)]
    if not today:
        sys.exit("no board on sale has more than a handful of players the "
                 "league has named for today - every one of them appears to "
                 "be for a different day. Nothing published.")
    if len(today) < len(out):
        log.info("dropped %d board(s) that are not for today",
                 len(out) - len(today))

    # Earliest lock first, so the dropdown reads Early, Afternoon, Main,
    # Night - the order a person actually thinks in. Size is only a tiebreak.
    today.sort(key=lambda t: (t[5] if pd.notna(t[5]) else pd.Timestamp.max
                              .tz_localize("UTC"), -t[3]))
    return [(dg, label, board, starts)
            for dg, label, board, _, _, starts in today]


def build_slate(proj: pd.DataFrame, dg: int, label: str,
                board: pd.DataFrame, probables: dict, field_size: int,
                sims: int, confirmed_only: bool = False,
                starts=None, status: dict | None = None,
                min_salary: int = 0,
                ownership_file: dict | None = None) -> dict | None:
    log.info("draft group %s: %s", dg, label)

    board["slot"] = board["position"].map(roster_position)
    unknown = board[board["slot"].isna()]
    if len(unknown):
        log.info("%d priced players fill no slot (%s) - dropped",
                 len(unknown),
                 ", ".join(sorted(set(unknown["position"].astype(str)))[:6]))
    board = board[board["slot"].notna()].copy()

    gone = board["disabled"].fillna(False).astype(bool)
    if gone.any():
        log.info("%d players are flagged unavailable and are dropped",
                 int(gone.sum()))
        board = board[~gone].copy()

    # The injured list, hours before any lineup card exists.
    #
    # A man on the sixty-day list is priced at the minimum, carries a full
    # projection built from the games he played before he got hurt, and is
    # therefore the best points-per-dollar on the board. Byron Buxton, hip
    # impingement, in every lineup it produced.
    #
    # Only a player the league has ON a roster and NOT listed active is
    # dropped. Someone on no roster we read is unknown, not out, and is left
    # alone - because deleting a man from a slate on the strength of not
    # having heard of him is how a filter empties a board.
    if status and status.get("teams_read"):
        ids0 = _id_text(board["mlb_id"]).fillna("")
        norm0 = board["name"].map(MD.normalise)
        seen = ids0.isin(status["seen_ids"]) | norm0.isin(status["seen_names"])
        active = (ids0.isin(status["active_ids"])
                  | norm0.isin(status["active_names"]))
        hurt = seen & ~active
        if hurt.any():
            log.info("dropped %d players the league does not list as active "
                     "(injured list, optioned, suspended): %s",
                     int(hurt.sum()), ", ".join(sorted(board.loc[hurt,
                                                                "name"])[:15]))
            board = board[~hurt].copy()
        log.info("roster check: %d of %d priced players found on a 40-man, "
                 "%d of those active", int(seen.sum()), len(seen),
                 int(active.sum()))

    # A salary floor, off by default and available when you want it.
    #
    # The instinct is right - almost every unavailable player sits at the
    # minimum - but the arrow points the other way. Being hurt makes a man
    # cheap; being cheap does not make him hurt. September call-ups and
    # rookies are priced at the minimum too, they start, and they are some of
    # the best value on a board. Deleting the price band would throw those
    # away to catch something the roster check above catches by its cause.
    if min_salary:
        cheap = pd.to_numeric(board["salary"], errors="coerce") < min_salary
        if cheap.any():
            log.info("--min-salary %d: dropping %d players priced below it",
                     min_salary, int(cheap.sum()))
            board = board[~cheap].copy()

    # Only today's announced starters may fill a P slot.
    #
    # Without this the board prices every pitcher on every 26-man roster, the
    # ones not starting are cheap, and points per dollar - the statistic a
    # pitcher who throws no innings maximises - puts one of them in every
    # single lineup. That is not a subtle mis-ranking; it is the optimiser
    # working perfectly on a board that lied to it.
    # Matched on the league id AND on the name, because neither key survives on
    # its own. DraftKings' copy of the league id came back empty for all 278
    # rows of a live board, and a filter keyed only on that would not have
    # dropped the pitchers who are not starting - it would have dropped every
    # pitcher on the slate, skipped the board, published nothing, and left the
    # page showing yesterday's lineups. Which is indistinguishable, from the
    # outside, from the filter not working at all.
    ids = _id_text(board["mlb_id"]).fillna("")
    norm = board["name"].map(MD.normalise)
    is_p = board["slot"] == "P"
    starting = ids.isin(probables["ids"]) | norm.isin(probables["names"])

    kept = board[is_p & starting]
    drop = is_p & ~starting
    log.info("pitchers priced %d, announced starters among them %d",
             int(is_p.sum()), len(kept))
    if len(kept):
        log.info("  STARTING: %s", ", ".join(sorted(kept["name"])))
    if drop.any():
        names = sorted(board.loc[drop, "name"])
        log.info("  dropped %d not starting: %s%s", len(names),
                 ", ".join(names[:12]), " ..." if len(names) > 12 else "")
    if len(kept) < 2:
        log.error("draft group %s prices %d pitchers and only %d match the "
                  "league's probables, so a legal lineup cannot be filled. "
                  "Both the id and the name failed to match. NOT publishing - "
                  "the alternative is a page that starts a man who is not "
                  "playing.", dg, int(is_p.sum()), len(kept))
        return None

    # The pitchers who are not starting leave now, before anything below reads
    # the board again. Dropping rows and then reusing an index built from the
    # old ones is its own class of bug.
    board = board[~drop].copy()
    ids, norm = ids[board.index], norm[board.index]
    is_p = board["slot"] == "P"

    # Hitters: the posted batting order, where it exists.
    #
    # A hitter who is rested scores zero, and DraftKings prices him anyway, so
    # the same arithmetic that put a non-starting pitcher in every lineup puts
    # a benched hitter there too. The difference is that a lineup card only
    # goes up about two hours before first pitch, so an afternoon run has to
    # cope with not knowing yet - and "I do not know" and "he is out" are
    # different answers. Only teams whose card IS posted can have anyone
    # dropped; everyone else is carried and marked unconfirmed.
    board["bat"] = pd.NA
    slot_by_id = probables["order_id"]
    slot_by_name = probables["order_name"]
    board.loc[~is_p, "bat"] = [
        slot_by_id.get(i) or slot_by_name.get(n)
        for i, n in zip(ids[~is_p], norm[~is_p])]

    # Whether a team's card is posted is decided BY THE PLAYERS, not by
    # matching team codes.
    #
    # The first version compared DraftKings' team string to the league's
    # abbreviation. When those disagree - and they do; the Athletics alone
    # have been OAK, ATH and SAC inside two years - no team looks posted, no
    # hitter is ever dropped, and Nick Kurtz stays on the board at the minimum
    # salary on a night Tommy White is playing first base. A filter that
    # depends on two organisations spelling a team the same way is not a
    # filter.
    #
    # So: count how many of each team's priced hitters turned up in a posted
    # lineup. Five or more and that card is clearly up, and anyone from that
    # team who is NOT in it is not playing. No team code is ever compared.
    hit = (~is_p)
    matched_by_team = (board[hit & board["bat"].notna()]["team"]
                       .astype(str).value_counts())
    CARD_IS_UP = 5
    up = set(matched_by_team[matched_by_team >= CARD_IS_UP].index)

    if up:
        log.info("  lineup cards up for %d teams: %s", len(up),
                 ", ".join(f"{t}({matched_by_team[t]})" for t in sorted(up)))
    thin = matched_by_team[(matched_by_team > 0)
                           & (matched_by_team < CARD_IS_UP)]
    if len(thin):
        log.warning("  %d teams matched only 1-%d hitters (%s) - too few to "
                    "call the card posted, so nobody from them is dropped",
                    len(thin), CARD_IS_UP - 1,
                    ", ".join(f"{t}({n})" for t, n in thin.items()))

    benched = hit & board["team"].astype(str).isin(up) & board["bat"].isna()
    if benched.any():
        names = sorted(board.loc[benched, "name"])
        log.info("  dropped %d hitters NOT in their team's posted lineup: %s",
                 len(names), ", ".join(names))
        board = board[~benched].copy()
        ids, norm = ids[board.index], norm[board.index]
        is_p = board["slot"] == "P"

    # Once most of the slate's cards are up, an unconfirmed hitter is not an
    # unknown - he is a man his manager has left out, and the board is simply
    # slower than the manager. So this stops being opt-in and becomes the
    # default, because the failure it prevents is the expensive one and the
    # failure it causes is a slightly smaller pool.
    #
    # Waiting for a flag to be ticked is not a safety mechanism. It is a way
    # of being wrong on the nights somebody forgets.
    teams_total = int(board["team"].nunique())
    broadly_posted = len(up) >= max(1, teams_total // 2)
    loose = (~is_p) & board["bat"].isna()
    if loose.any() and (confirmed_only or broadly_posted):
        why = ("--confirmed-only" if confirmed_only
               else f"{len(up)} of {teams_total} teams have posted")
        log.info("  dropping %d hitters with no confirmed lineup slot (%s): "
                 "%s", int(loose.sum()), why,
                 ", ".join(sorted(board.loc[loose, "name"])[:15]))
        board = board[~loose].copy()
        ids, norm = ids[board.index], norm[board.index]
        is_p = board["slot"] == "P"
    elif loose.any():
        log.warning("  %d hitters have no confirmed slot and are being kept - "
                    "only %d of %d cards are up. They can be rostered, and a "
                    "rested hitter scores zero.",
                    int(loose.sum()), len(up), teams_total)

    in_order = int(board["bat"].notna().sum())
    log.info("hitters: %d on the board, %d confirmed in a batting order",
             int((~is_p).sum()), in_order)

    merged = join_board(board, proj)
    pool = merged[merged["player_id"].notna()].copy()
    pool = pool[pd.to_numeric(pool["salary"], errors="coerce").notna()]
    if pool.empty:
        log.error("nothing on draft group %s joined to the history", dg)
        return None

    have = float(pd.to_numeric(pool["salary"]).sum()
                 / pd.to_numeric(merged["salary"], errors="coerce").sum())
    log.info("%d of %d priced players projected, %.0f%% of slate salary",
             len(pool), len(merged), 100 * have)

    # A real ten-game classic board carries hundreds of players. Sixty is not
    # a slate - it is the wreckage of a filter that matched almost nothing, and
    # publishing it produces lineups drawn from whoever happened to survive.
    per_slot = pool["slot"].value_counts()
    thin = {s: int(per_slot.get(s, 0)) for s in set(ROSTER["slots"])
            if int(per_slot.get(s, 0)) < 2}
    games_here = int(pool["game"].nunique())
    if len(pool) < 12 * games_here:
        log.warning("draft group %s has only %d projectable players across %d "
                    "games. A classic board of that size normally carries "
                    "several hundred - treat everything below with suspicion.",
                    dg, len(pool), games_here)

    missing = [s for s in set(ROSTER["slots"])
               if not (pool["slot"] == s).any()]
    if missing:
        log.error("draft group %s has no projected player for %s - a legal "
                  "lineup does not exist, so it is skipped rather than "
                  "published as a board that cannot be built from", dg,
                  missing)
        return None

    pool = apply_batting_order(pool)

    # The engine wants its own column names.
    pool = pool.rename(columns={"slot": "position"})
    pool["position"] = pool["position"].astype(str)
    pool = pool.reset_index(drop=True)

    own = OWN.project(pool, ROSTER)
    # A vendor's projected ownership, if one has been dropped in the repo.
    # Replaces the modelled number where it has an answer and leaves it
    # where it does not. Ownership is half the edge formula and the half
    # that has never been graded against a baseball field, so a source with
    # a real feedback loop beats ours by default.
    own = OWF.apply(pool, own, ownership_file, len(ROSTER["slots"]))
    lev = OWN.leverage(pool, own)
    pool["ownership"] = own
    pool["leverage"] = lev

    quantiles = list(MS.HITTERS.quantiles)
    draws = S.simulate(pool, quantiles, sims, spec=board_spec())

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "sport": SPORT,
        "site": "dk",
        "kind": "classic",
        "draft_group": dg,
        # What the dropdown says. DraftKings' own contest name is a marketing
        # string - "MLB $3.9K Perfect Game [$2K to 1st] (Early)" - that barely
        # identifies the board. The two facts which actually tell one slate
        # from another are how many games it covers and when it locks, so
        # those lead and the marketing trails.
        "label": slate_label(pool, starts, label),
        "locks": (starts.isoformat()
                  if starts is not None and pd.notna(starts) else None),
        "field_size": field_size,
        "quantiles": quantiles,
        "loadings": loadings_for_page(),
        "roster": ROSTER,
        "players": slate_players(pool, quantiles, own, lev),
        "server_lineups": server_lineups(pool, draws, own, field_size),
    }
    return payload


def write(payloads: list[dict]) -> None:
    """Every slate this run produced, plus a manifest listing exactly those.

    The manifest is rebuilt rather than appended to, and yesterday's files are
    moved into an archive. A stale slate left in the DROPDOWN is worse than a
    missing one: it loads, it looks current, and every salary in it is a day
    old. But the dropdown is the manifest, not the directory - so keeping the
    file costs nothing, and deleting it costs the only record of what we
    published.

    They were deleted, until a contest export arrived on 2026-09-18 for a
    slate played on the 16th and there was nothing left to grade it against.
    score_slates.py matched it to that morning's board instead - 318 players
    against the contest's 98 - and produced a page of numbers comparing one
    slate's projections to another slate's results. Every one of them was
    wrong and not one of them looked wrong.

    A board is the only record of what the model believed on a given day.
    Results arrive days later, so it has to outlive the slate.
    """
    DATA.mkdir(parents=True, exist_ok=True)
    fresh = set()
    slates = []
    for payload in payloads:
        name = (f"{payload['sport']}_{payload['site']}_{payload['kind']}_"
                f"{payload['draft_group']}.json")
        path = DATA / name
        path.write_text(json.dumps(payload, separators=(",", ":")))
        fresh.add(name)
        log.info("wrote %s (%.0f KB, %d players)", path,
                 path.stat().st_size / 1024, len(payload["players"]))
        slates.append({"sport": payload["sport"], "site": payload["site"],
                       "kind": payload["kind"], "label": payload["label"],
                       "file": f"data/{name}"})

    # Archived under the date the board was FOR, not the date it is being
    # moved, so a contest played on the 16th is looked up by the 16th.
    archive = DATA / "archive"
    for old in DATA.glob(f"{SPORT}_*.json"):
        if old.name in fresh:
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

    # Kept for a season, not forever. A board is about 15 KB and a day holds
    # eight of them, so a year is under a megabyte - but an unbounded
    # directory in a repo that publishes a page is a slow leak, and the
    # contests worth grading are the recent ones.
    keep_after = (datetime.now(timezone.utc) - timedelta(days=ARCHIVE_DAYS))
    for day in sorted(archive.glob("[0-9]" * 4 + "-*")):
        try:
            on = datetime.strptime(day.name, "%Y-%m-%d").replace(
                tzinfo=timezone.utc)
        except ValueError:
            continue
        if on < keep_after:
            for f in day.glob("*.json"):
                f.unlink()
            day.rmdir()
            log.info("archive: dropped %s (older than %d days)",
                     day.name, ARCHIVE_DAYS)

    (DATA / "manifest.json").write_text(json.dumps({
        "updated_at": payloads[0]["generated_at"],
        "slates": slates,
    }, indent=1))
    log.info("manifest lists %d slate(s)", len(slates))


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draft-group", type=int, default=None)
    p.add_argument("--first-season", type=int, default=2025)
    p.add_argument("--field", type=int, default=100_000)
    p.add_argument("--sims", type=int, default=20_000)
    p.add_argument("--slates", type=int, default=8,
                   help="how many boards to publish, biggest first")
    p.add_argument("--date", default=None,
                   help="slate date YYYY-MM-DD (default: today, US Eastern)")
    p.add_argument("--confirmed-only", action="store_true",
                   help="drop hitters whose lineup card is not up yet")
    p.add_argument("--min-salary", type=int, default=0,
                   help="drop PLAYERS priced below this (0 = off). This is a "
                        "board filter and has nothing to do with "
                        "--min-salary-pct, which is a LINEUP floor.")
    p.add_argument("--min-salary-pct", type=float, default=98.0,
                   help="the least a LINEUP may spend, as a percent of the "
                        "$50,000 cap (0 = off). 98 leaves at most $1,000. "
                        "Unspent salary is points declined - about 1.16 "
                        "projected points per $1,000 in MLB - and in a small "
                        "player pool the cheap plays you free up salary for "
                        "are usually MORE owned, not less. If no legal lineup "
                        "can reach the floor it is dropped with a loud log "
                        "line and the board still publishes.")
    p.add_argument("--look", type=int, default=25,
                   help="how many draft groups to fetch before ranking them")
    p.add_argument("--ownership", default=None,
                   help="a CSV of projected ownership to use instead of the "
                        "modelled numbers (default: the first one found in "
                        "the repo)")
    p.add_argument("--no-ownership-file", action="store_true",
                   help="ignore any ownership CSV and use the model")
    args = p.parse_args(argv)

    # The lineup salary floor lives on the roster spec, because that is what
    # every layer of the optimiser already carries. Stored as a FRACTION so
    # the same setting means the same thing under any cap.
    pct = max(0.0, min(100.0, float(args.min_salary_pct))) / 100.0
    ROSTER["min_salary_pct"] = pct
    if pct > 0:
        floor = int(pct * ROSTER["salary_cap"])
        log.info("lineup salary floor $%s (%.1f%% of $%s) - at most $%s may "
                 "be left unspent",
                 f"{floor:,}", 100 * pct, f"{ROSTER['salary_cap']:,}",
                 f"{ROSTER['salary_cap'] - floor:,}")
    else:
        log.info("no lineup salary floor; the optimiser may leave any amount "
                 "of the cap unspent")

    seasons = list(range(args.first_season,
                         datetime.now(timezone.utc).year + 1))
    hist = C.load(SPORT, seasons)
    log.info("%d player-games of history", len(hist))

    # The schedule is read BEFORE anything is projected, and the order is
    # load-bearing rather than tidy. A hitter's projection now depends on the
    # pitcher he is about to face, so the matchups have to be in hand before
    # the model is asked for a number. Projecting first and adjusting after
    # would mean adjusting a quantile that was fitted against a different
    # opponent, which is not the same thing at all.

    # WHICH DAY, decided by the next slate that has not locked - not by the
    # clock, and not by "today".
    #
    # "Today" is the wrong question and asking it is what produced a board
    # full of tomorrow's games. At four in the afternoon the next slate is
    # tonight's; at eleven at night every one of today's games has finished
    # and the next slate on sale is tomorrow's, which is the one worth
    # publishing. Pinning the target to the calendar meant that after the last
    # first pitch the league had no games left to name, every board on sale
    # was correctly judged "not today", and NOTHING was published - so the
    # page kept showing the stale file it already had.
    #
    # So the slate is chosen first, from its lock time, and the lineup card is
    # then fetched for whatever day that slate belongs to.
    day = args.date or next_slate_day()
    probables = MD.probable_pitchers(day)
    today = day
    if not probables["announced"] and not probables["posted"]:
        sys.exit(f"the league has announced no probable pitchers for {today}. "
                 f"Publishing without them puts a pitcher who is not playing "
                 f"into every lineup, so nothing is published and the page "
                 f"keeps what it had.")

    # Now the schedule is known, so the hitters can be projected against the
    # pitchers they are actually facing.
    opp_tonight = tonight_opposing_starter(hist, probables)
    # Optional by design. No cache means the Statcast columns are absent and
    # the model is exactly what it was, rather than a board that fails to go
    # out because a supplementary feed was down.
    statcast = MSC.load(seasons)
    proj = pd.concat([project_half(hist, "hitters", opp_tonight, statcast),
                      project_half(hist, "pitchers", statcast=statcast)],
                     ignore_index=True)

    # A vendor's ownership projection, if one has been dropped in the repo.
    # Read once and reused for every slate, and entirely optional: no file
    # means the modelled ownership, exactly as before.
    ownership_file: dict = {}
    if not args.no_ownership_file:
        path = OWF.find_file(args.ownership)
        if path is None:
            log.info("no ownership CSV found - using the modelled ownership")
        else:
            ownership_file = OWF.read(path)

    # Roster status for every team playing, read once and reused per slate.
    try:
        status = MD.roster_status(probables.get("team_ids") or [])
    except Exception as exc:                                   # noqa: BLE001
        log.error("could not read the rosters (%s: %s) - injured players "
                  "cannot be filtered out before lineups post",
                  type(exc).__name__, str(exc)[:90])
        status = None

    payloads = []
    for dg, label, board, starts in candidate_slates(
            args.draft_group, args.look, probables):
        if len(payloads) >= args.slates:
            break
        got = build_slate(proj, dg, label, board, probables,
                          args.field, args.sims, args.confirmed_only,
                          starts, status, args.min_salary, ownership_file)
        if got:
            payloads.append(got)

    payloads.sort(key=lambda p: (p.get("locks") or "9999"))
    if not payloads:
        sys.exit("no slate could be built - nothing was published, so the "
                 "page keeps whatever it had")
    write(payloads)

    print()
    print("=" * 70)
    for payload in payloads:
        print(f"{payload['label']}  -  {len(payload['players'])} players "
              f"(draft group {payload['draft_group']})")
        for objective, lu in payload["server_lineups"].items():
            print(f"  {objective:<5} ${lu['salary']:,}  "
                  f"ceiling {lu['ceiling']}")
            print(f"        {', '.join(lu['players'])}")
        if not payload["server_lineups"]:
            print("  NO server lineup solved - the page will show only the "
                  "browser's own search")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
