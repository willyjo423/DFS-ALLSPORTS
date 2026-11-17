"""College football data: the board, ten seasons of history, and the join.

This is the CFB counterpart to the NFL project's data.py, and it is a rewrite
rather than a port because every input differs. What carries over is the set
of mistakes already paid for, each of which is prevented here by construction:

* **The board comes from the lobby, not the API.** api.draftkings.com answers
  403 to GitHub runners. www.draftkings.com/lineup/getavailableplayers answers
  200 with the whole slate, in an abbreviated dialect - fn/ln, s, pn.

* **Position is `pn`, never `pp`.** `pp` is an integer that is zero for every
  player on the board. Mapping position to it once produced 850 players
  sharing a single position called `0`, which makes every positional
  constraint vacuous and a lineup of eight quarterbacks perfectly legal.

* **Team codes do not join; fixtures do.** DraftKings writes BAMA, TA&M, RU;
  CFBD writes Alabama, Texas A&M, Rutgers. Two of twenty-four matched by
  string. `UL` is Louisiana to DraftKings and reads equally well as
  Louisville, and nothing about the string settles it - but a fixture must
  match exactly one CFBD fixture on BOTH sides, and that does settle it.

* **Names need two spellings, not one.** DraftKings writes "AJ Swann", CFBD
  writes "A.J. Swann". Period-to-space gives "aj swann" and "a j swann",
  which do not match; period-to-nothing breaks "St.Brown" instead. Both rules
  are right and they contradict, so both keys are carried and either may hit.

* **The live week is fetched, not calculated.** Probing showed the season's
  completed-game count equal to weeks 1 and 2 exactly while week 3 sat
  scheduled with zero played. A clock-based guess would have asked for a week
  that does not exist yet and quietly returned nothing.

* **Captain salary is charged, not assumed.** On Showdown the captain costs
  1.5x salary as well as scoring 1.5x points. Reading only the points
  multiplier builds lineups that cannot be entered.
"""

from __future__ import annotations

import logging
import re
import time

import numpy as np
import pandas as pd
import requests

from availability import dk_status  # noqa: E402

log = logging.getLogger(__name__)

UA = {"User-Agent": "Mozilla/5.0 (dfs research)"}
TIMEOUT = 45
RETRIES = 3

CFBD = "https://api.collegefootballdata.com"
DK_CONTESTS = "https://www.draftkings.com/lobby/getcontests?sport=CFB"
DK_PLAYERS = ("https://www.draftkings.com/lineup/getavailableplayers"
              "?draftGroupId={dg}")

# DraftKings college football. Same shape as the NFL game with one difference
# that matters for pricing: college is full PPR, so slot receivers on
# pass-happy teams are worth more than their yardage suggests.
SCORING = {
    "pass_yards": 0.04, "pass_td": 4.0, "interception": -1.0,
    "rush_yards": 0.1, "rush_td": 6.0,
    "rec": 1.0, "rec_yards": 0.1, "rec_td": 6.0,
    "return_td": 6.0, "fumble_lost": -1.0,
}
# The bonuses are why a projection has to be a distribution and not a mean. A
# 95-yard rusher and a 105-yard rusher are one yard of talent apart and three
# points apart, and only a simulation prices that step.
BONUS = {"pass_yards": (300, 3.0), "rush_yards": (100, 3.0),
         "rec_yards": (100, 3.0)}

STAT_MAP = {
    ("passing", "YDS"): "pass_yards", ("passing", "TD"): "pass_td",
    ("passing", "INT"): "interception", ("passing", "C/ATT"): "completions",
    ("rushing", "YDS"): "rush_yards", ("rushing", "TD"): "rush_td",
    ("rushing", "CAR"): "carries",
    ("receiving", "YDS"): "rec_yards", ("receiving", "TD"): "rec_td",
    ("receiving", "REC"): "rec",
    ("fumbles", "LOST"): "fumble_lost",
    ("kickReturns", "TD"): "kick_return_td",
    ("puntReturns", "TD"): "punt_return_td",
}

STAT_FIELDS = sorted(set(STAT_MAP.values()))

# Seasons the depth probe cleared. 2014-2015 lack fumbles/LOST entirely, and
# scoring them would credit every fumble as zero rather than -1 - wrong in a
# way that looks fine. 2020 is present but is a different sport: 20 games in
# week 3 against a normal 65, opt-outs, and a schedule rewritten weekly.
USABLE_FIRST = 2016
COVID_SEASON = 2020

SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]
_SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


class Unavailable(RuntimeError):
    """A source this build cannot proceed without did not answer."""


# --------------------------------------------------------------------- names

def name_keys(name) -> set[str]:
    """Every spelling two sources might agree on, not just one.

    Deliberately not fuzzy. Every key is an exact reduction, because an
    edit-distance match would pair two different players with similar names,
    and in college football - where rosters run to 120 and brothers are
    common - that is a far worse failure than a miss.
    """
    raw = str(name or "").strip()
    if "," in raw:
        last, _, first = raw.partition(",")
        raw = f"{first.strip()} {last.strip()}"
    keys = set()
    for dot in (" ", ""):
        s = raw.lower().replace("-", " ").replace(".", dot)
        s = s.replace("'", "").replace("`", "")
        s = _SUFFIXES.sub(" ", s)
        s = " ".join(s.split())
        if s:
            keys.add(s)
    return keys


def _split_name(key: str) -> tuple[str, list[str]]:
    """A normalised key as (first name, surname tokens)."""
    parts = key.split()
    if not parts:
        return "", []
    return parts[0], parts[1:] or [parts[0]]


def first_names_compatible(a: str, b: str) -> bool:
    """Is one of these the short form of the other?

    Measured, not assumed. Across the misses this rule was built from, every
    correct pair shared a prefix of at least three characters - matt/matthew
    and benji/benjamin and zacharyus/zach share four, maxence/max shares
    three - and every wrong pair shared at most one: mason/stanley zero,
    quintrevion/tre zero, terrence/t.j. one.

    Edit-distance ratio does NOT separate those two groups; prefix length
    does, cleanly, with nothing in between. So the rule is prefix length.
    It costs the nickname cases that share no prefix, which is the right
    trade: Tre Wisner goes unmatched rather than matching by guesswork.
    """
    if a == b:
        return True
    if not a or not b:
        return False
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n >= 3


def surnames_compatible(a: list[str], b: list[str]) -> bool:
    """Same family name, allowing for compound forms and typos.

    Three real shapes: DraftKings carrying a name CFBD does not
    (Coleman-Williams vs Williams), CFBD carrying one DraftKings does not
    (Riley vs Riley-Ducker), and an outright transposition in CFBD
    (Remenowksy for Remenowsky, which scores 0.90).
    """
    if set(a) & set(b):
        return True
    if not a or not b:
        return False
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a[-1], b[-1]).ratio() >= 0.85


def primary_key(name) -> str:
    """One key, for grouping. The period-to-space spelling."""
    keys = name_keys(name)
    return sorted(keys)[0] if keys else ""


# ----------------------------------------------------------------- transport

def _get_json(url: str, headers=None, params=None):
    last = None
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, headers=headers or UA, params=params,
                             timeout=TIMEOUT)
        except Exception as exc:                   # noqa: BLE001
            last = f"{type(exc).__name__}: {str(exc)[:100]}"
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code == 200:
            try:
                return r.json()
            except Exception as exc:               # noqa: BLE001
                raise Unavailable(
                    f"{url[:70]} returned 200 but not JSON "
                    f"({type(exc).__name__}): {r.content[:100]!r}") from exc
        # 429 and 5xx are worth another try; 4xx is not.
        last = f"HTTP {r.status_code}: {r.text[:100]}"
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2.0 * (attempt + 1))
            continue
        break
    raise Unavailable(f"{url[:70]} -> {last}")


def cfbd(path: str, key: str, **params):
    hdr = dict(UA)
    hdr["Authorization"] = f"Bearer {key}"
    return _get_json(f"{CFBD}/{path}", headers=hdr, params=params)


# ---------------------------------------------------------------- draftkings

def slates() -> pd.DataFrame:
    """Every college football draft group DraftKings is currently selling."""
    payload = _get_json(DK_CONTESTS)
    contests = payload.get("Contests") or []
    if not contests:
        raise Unavailable("the college football lobby listed no contests")
    rows = {}
    for c in contests:
        dg = c.get("dg")
        if not dg:
            continue
        r = rows.setdefault(dg, {
            "draft_group": dg, "contests": 0,
            "game_type": c.get("gameType"),
            "starts_text": c.get("sdstring"),
            "biggest_prize": 0, "biggest_field": 0,
            "example": c.get("n"),
        })
        r["contests"] += 1
        r["biggest_prize"] = max(r["biggest_prize"], c.get("po") or 0)
        r["biggest_field"] = max(r["biggest_field"], c.get("m") or 0)
    out = pd.DataFrame(rows.values())
    log.info("lobby: %d draft groups, %d contests",
             len(out), int(out["contests"].sum()))
    return out.sort_values("contests", ascending=False).reset_index(drop=True)


def board(draft_group: int, captain_multiplier: float | None = None
          ) -> pd.DataFrame:
    """Who is priced on a slate.

    `charged_salary` is what the cap actually sees. On Showdown the captain
    costs 1.5x as well as scoring 1.5x, and a solver that charges face value
    builds lineups DraftKings will not accept.
    """
    payload = _get_json(DK_PLAYERS.format(dg=draft_group))
    raw = (payload.get("playerList") or payload.get("draftables")
           or payload.get("players") or [])
    if not raw:
        raise Unavailable(f"draft group {draft_group} returned no players "
                          f"(top-level keys: {sorted(payload)[:10]})")

    rows = []
    for p in raw:
        home, away = p.get("htabbr"), p.get("atabbr")
        tid, htid, atid = p.get("tid"), p.get("htid"), p.get("atid")
        team = home if tid == htid else away if tid == atid else None
        opponent = away if tid == htid else home if tid == atid else None
        rows.append({
            "dk_player_id": p.get("pid"),
            "name": " ".join(x for x in (p.get("fn"), p.get("ln")) if x),
            "position": p.get("pn"),
            "team": team,
            "opponent": opponent,
            "is_home": 1 if tid == htid else 0 if tid == atid else np.nan,
            "salary": pd.to_numeric(p.get("s"), errors="coerce"),
            "dk_points_per_game": pd.to_numeric(p.get("ppg"), errors="coerce"),
            "disabled": bool(p.get("IsDisabledFromDrafting")),
            "dk_status": dk_status(p),
            "roster_slot": p.get("rosposid"),
            "game": f"{away} @ {home}" if home and away else None,
        })
    df = pd.DataFrame(rows)

    # A board where every player shares one position, or where every salary is
    # NaN, is a mapping failure that looks exactly like a successful parse.
    if df["position"].nunique() <= 1:
        raise Unavailable(
            f"every player on draft group {draft_group} has position "
            f"{df['position'].iloc[0]!r}. The position field moved; look at a "
            f"raw row before trusting anything downstream.")
    if df["salary"].notna().sum() == 0:
        raise Unavailable(f"draft group {draft_group}: no player has a "
                          f"salary. The salary field moved.")

    if df["team"].isna().any():
        n = int(df["team"].isna().sum())
        log.warning("%d of %d players could not be assigned to a side of "
                    "their fixture - dropped rather than guessed", n, len(df))
        df = df.dropna(subset=["team"])

    # Showdown: DraftKings lists the captain as a separate, pre-multiplied
    # row. Detect it from the prices themselves rather than from the slate's
    # name, because the name is a label and the prices are the contract.
    df["is_captain"] = 0
    if captain_multiplier:
        df = _mark_captains(df, captain_multiplier)
    df["charged_salary"] = df["salary"] * np.where(
        df["is_captain"] == 1, captain_multiplier or 1.0, 1.0)

    df["keys"] = df["name"].map(name_keys)
    df["key"] = df["name"].map(primary_key)

    log.info("draft group %s: %d players, %d teams, %d games, $%s-$%s",
             draft_group, len(df), df["team"].nunique(), df["game"].nunique(),
             int(df["salary"].min()), int(df["salary"].max()))
    return df.reset_index(drop=True)


def _mark_captains(df: pd.DataFrame, mult: float) -> pd.DataFrame:
    """Flag rows DraftKings has already multiplied.

    On a Showdown board the same player appears twice - once at face value and
    once at roughly `mult` times it. Pairing by name and comparing the two
    prices identifies the captain row without trusting a slot id whose meaning
    changes between sports.
    """
    out = df.copy()
    by_key = out.groupby(out["name"].map(primary_key))["salary"]
    lo = by_key.transform("min")
    ratio = out["salary"] / lo.replace(0, np.nan)
    out["is_captain"] = ((ratio > (mult + 1.0) / 2.0) & (ratio.notna())
                         ).astype(int)
    n = int(out["is_captain"].sum())
    if n:
        log.info("marked %d captain rows (salary ~%.2fx the base row)",
                 n, mult)
    return out


# --------------------------------------------------------------------- cfbd

def live_week(key: str, season: int) -> int:
    """The newest week that has actually been played, fetched not guessed.

    Asking the schedule which games are COMPLETE is the only reliable answer.
    A clock-based estimate gets this wrong in both directions - byes, Tuesday
    MACtion, and a season whose week 1 does not sit where the calendar says.
    """
    played = 0
    for week in range(1, 20):
        try:
            games = cfbd("games", key, year=season, week=week,
                         seasonType="regular")
        except Unavailable as exc:
            log.warning("week %d schedule unavailable: %s", week, exc)
            break
        if not games:
            break
        done = sum(1 for g in games
                   if g.get("completed") is True
                   or (g.get("home_points") is not None
                       and g.get("away_points") is not None))
        if done == 0:
            break
        played = week
    if played == 0:
        raise Unavailable(f"no completed {season} games in any week")
    log.info("%d: newest completed week is %d", season, played)
    return played


def upcoming_games(key: str, season: int, after_week: int,
                   span: int = 2) -> list[dict]:
    """The schedule a slate might draw from, not one week of it.

    DraftKings builds a Saturday slate from kickoffs, not from CFBD's week
    numbering, and the two do not always agree - a Friday or a late-window
    game can sit in the next week's bucket. Fetching one week left SMU @
    Louisiana unresolvable even though both codes proposed exactly the right
    school: the fixture simply was not in the list being searched.
    """
    out, seen = [], set()
    for week in range(after_week + 1, after_week + 1 + span):
        try:
            games = cfbd("games", key, year=season, week=week,
                         seasonType="regular")
        except Unavailable as exc:
            log.warning("week %d schedule unavailable: %s", week, exc)
            continue
        for g in games or []:
            gid = g.get("id")
            if gid not in seen:
                seen.add(gid)
                out.append(g)
    log.info("schedule window: weeks %d-%d, %d games",
             after_week + 1, after_week + span, len(out))
    return out


def roster_positions(key: str, season: int) -> pd.DataFrame:
    """athlete_id -> position for a season.

    games/players carries no position and the roster endpoint is the only
    source of one. Blank positions are dropped rather than kept as empty
    strings: a row that matched an entry with no position is not a match, and
    counting it as one overstates coverage by about six points.
    """
    payload = cfbd("roster", key, year=season)
    if not payload:
        raise Unavailable(f"{season} roster is empty")
    sample = payload[0]
    id_field = next((f for f in ("id", "athlete_id", "athleteId", "player_id")
                     if f in sample), None)
    if id_field is None:
        raise Unavailable(f"{season} roster has no id field; "
                          f"keys are {sorted(sample)[:12]}")
    rows = [{"athlete_id": str(p[id_field]),
             "position": str(p.get("position") or "").upper().strip(),
             "roster_name": " ".join(
                 x for x in (p.get("first_name") or p.get("firstName"),
                             p.get("last_name") or p.get("lastName")) if x)}
            for p in payload if p.get(id_field) is not None]
    df = pd.DataFrame(rows)
    blank = int((df["position"] == "").sum())
    df = df[df["position"] != ""].drop_duplicates("athlete_id")
    log.info("%d roster: %d positioned athletes (%d blank, dropped)",
             season, len(df), blank)
    return df.reset_index(drop=True)


def _number(value):
    """A stat as a number. CFBD sends every value as a string.

    `C/ATT` arrives as "18/25", which float() rejects; the part before the
    slash is completions. A bare float() would turn every quarterback's
    completion count into NaN, and NaN is not zero - it propagates into the
    feature and deletes the row.
    """
    if value is None:
        return np.nan
    s = str(value).strip().replace(",", "")
    if "/" in s:
        s = s.split("/")[0]
    try:
        return float(s)
    except ValueError:
        return np.nan


def flatten_player_games(payload: list[dict]) -> pd.DataFrame:
    """One row per player per game, columns named for the scoring rule."""
    rows = []
    for game in payload or []:
        for team in game.get("teams") or []:
            school = team.get("team") or team.get("school")
            opponent = None
            for other in game.get("teams") or []:
                if (other.get("team") or other.get("school")) != school:
                    opponent = other.get("team") or other.get("school")
            homeaway = team.get("homeAway")
            for cat in team.get("categories") or []:
                cname = cat.get("name")
                for typ in cat.get("types") or []:
                    field = STAT_MAP.get((cname, typ.get("name")))
                    if field is None:
                        continue
                    for ath in typ.get("athletes") or []:
                        rows.append({
                            "game_id": game.get("id"), "school": school,
                            "opponent": opponent,
                            "is_home": 1 if homeaway == "home" else 0,
                            "athlete_id": str(ath.get("id")),
                            "name": ath.get("name"),
                            "field": field,
                            "value": _number(ath.get("stat")),
                        })
    if not rows:
        return pd.DataFrame()
    long = pd.DataFrame(rows)
    wide = (long.pivot_table(
        index=["game_id", "school", "opponent", "is_home",
               "athlete_id", "name"],
        columns="field", values="value", aggfunc="sum").reset_index())
    wide.columns.name = None
    for field in STAT_FIELDS:
        if field not in wide:
            wide[field] = 0.0
    return wide.fillna({f: 0.0 for f in STAT_FIELDS})


def fantasy_points(df: pd.DataFrame) -> pd.Series:
    """DraftKings college football points for each row."""
    pts = pd.Series(0.0, index=df.index)
    for field, weight in SCORING.items():
        if field == "return_td":
            pts = pts + weight * (df.get("kick_return_td", 0)
                                  + df.get("punt_return_td", 0))
        elif field in df:
            pts = pts + weight * df[field]
    for field, (threshold, bonus) in BONUS.items():
        if field in df:
            pts = pts + bonus * (df[field] >= threshold)
    return pts.round(2)


def season_history(key: str, season: int, through_week: int,
                   positions: pd.DataFrame | None = None,
                   require_position: bool = False) -> pd.DataFrame:
    """Every settled game in a season, scored, with positions attached.

    `through_week` is inclusive of the last PLAYED week. Callers predicting
    week W must pass W-1; including W puts a player's own result into the
    features used to predict it, which is the leak that makes a model look
    brilliant in backtest and lose money on Saturday.
    """
    if positions is None:
        positions = roster_positions(key, season)
    frames = []
    for week in range(1, through_week + 1):
        try:
            payload = cfbd("games/players", key, year=season, week=week,
                           seasonType="regular")
        except Unavailable as exc:
            log.warning("%d week %d unavailable: %s", season, week, exc)
            continue
        wide = flatten_player_games(payload)
        if wide.empty:
            log.warning("%d week %d returned no player stats", season, week)
            continue
        wide["season"], wide["week"] = season, week
        frames.append(wide)
        time.sleep(0.35)
    if not frames:
        raise Unavailable(f"no {season} player stats through week "
                          f"{through_week}")
    out = pd.concat(frames, ignore_index=True)
    out["points"] = fantasy_points(out)

    before = len(out)
    out = out.merge(positions[["athlete_id", "position"]],
                    on="athlete_id", how="left")
    unplaced = int(out["position"].isna().sum())

    # Dropping unpositioned rows here was a real mistake, and an expensive
    # one: it deleted 9% of 2026 and then the join reported those players as
    # missing history. They had history. This function threw it away.
    #
    # A position is needed to FIT the model - position features, position
    # baselines, position-restricted grading. It is needed for none of the
    # other things history is used for, and DraftKings supplies a position
    # for every player it prices, so the live board never needs CFBD's.
    #
    # So the filter moved to the one caller that requires it.
    if require_position:
        out = out.dropna(subset=["position"])
    log.info("%d: %d player-games, %d athletes, %d without a position "
             "(%.1f%%, %s), points %.1f to %.1f",
             season, len(out), out["athlete_id"].nunique(), unplaced,
             100 * unplaced / max(1, before),
             "dropped" if require_position else "kept",
             out["points"].min(), out["points"].max())

    out["key"] = out["name"].map(primary_key)
    out["keys"] = out["name"].map(name_keys)
    return out


def training_history(key: str, first: int, last: int,
                     through_week: dict[int, int] | None = None,
                     skip_covid: bool = True,
                     require_position: bool = False) -> pd.DataFrame:
    """Several seasons, concatenated, for fitting.

    2020 is excluded by default. It is not a thin season - it is a different
    sport, played by teams missing opt-outs against a schedule rewritten every
    week. Including it adds rows and subtracts signal. The flag exists so the
    walk-forward can measure that claim rather than leaving it an assertion.
    """
    through_week = through_week or {}
    frames = []
    for season in range(max(first, USABLE_FIRST), last + 1):
        if skip_covid and season == COVID_SEASON:
            log.info("skipping %d (covid season)", COVID_SEASON)
            continue
        try:
            positions = roster_positions(key, season)
            weeks = through_week.get(season)
            if weeks is None:
                weeks = live_week(key, season) if season == last else 15
            frames.append(season_history(key, season, weeks, positions,
                                         require_position=require_position))
        except Unavailable as exc:
            log.warning("season %d skipped: %s", season, exc)
    if not frames:
        raise Unavailable(f"no usable seasons between {first} and {last}")
    out = pd.concat(frames, ignore_index=True)
    log.info("training history: %d player-games across %d seasons "
             "(%s), %d athletes",
             len(out), out["season"].nunique(),
             ", ".join(str(s) for s in sorted(out["season"].unique())),
             out["athlete_id"].nunique())
    return out


# ----------------------------------------------------------------- the join

def attach_history(board_df: pd.DataFrame, hist: pd.DataFrame,
                   team_map: dict | None = None) -> pd.DataFrame:
    """Give every priced player his own scored games, or nothing.

    Matching is on EITHER spelling. The index is built from the history side's
    key set so a DraftKings spelling that differs only in periods still lands.
    """
    index: dict[str, str] = {}
    for athlete, keys in zip(hist["athlete_id"], hist["keys"]):
        for k in keys:
            index.setdefault(k, athlete)

    # Matches are accumulated in a plain Python list and written to the frame
    # exactly once, at the end.
    #
    # This is not stylistic. Assigning a mixed list of strings and Nones to a
    # column makes pandas 3 infer its string dtype, which silently converts
    # every None to NaN - and `nan is not None` is True, so a guard of
    # "already matched, skip" then skips every UNMATCHED row instead.
    #
    # All three fallback passes below were dead on live data for that reason,
    # while passing their tests, because a test board where every player is
    # unmatched stays object dtype and keeps real Nones. The fixtures differed
    # from production in exactly the dimension being tested.
    ids: list = [next((index[k] for k in keys if k in index), None)
                 for keys in board_df["keys"]]
    out = board_df.copy()
    strict = sum(1 for a in ids if a is not None)

    # Second pass for the one difference the exact keys cannot absorb: a
    # middle name or initial present on one side only. "Tyler J. Williams" on
    # the board is "Tyler Williams" in CFBD, and no exact reduction bridges
    # that, because deleting a middle token is not a spelling difference.
    #
    # Dropping middles is therefore allowed ONLY when it is unambiguous. The
    # reduced key must identify exactly one athlete in the whole history; a
    # college roster has enough Williamses that a reduced key matching two
    # people is a coin flip, and a coin flip attached to a real salary is
    # worse than a miss. Ambiguous ones stay unmatched, deliberately.
    # The reduction is applied to BOTH sides. A first version reduced only
    # history keys of three or more parts, which is exactly the side that
    # usually has two - "Tyler Williams" in CFBD, "Tyler J. Williams" on the
    # board - so the index it built never contained the name being looked up.
    # It also made the ambiguous case pass by accident, because only one of
    # the two colliding athletes was ever indexed.
    def _short(k: str) -> str | None:
        parts = k.split()
        return f"{parts[0]} {parts[-1]}" if len(parts) >= 2 else None

    reduced: dict[str, set[str]] = {}
    for athlete, keys in zip(hist["athlete_id"], hist["keys"]):
        for k in keys:
            s = _short(k)
            if s:
                reduced.setdefault(s, set()).add(athlete)
    unique = {k: next(iter(v)) for k, v in reduced.items() if len(v) == 1}

    # The same reduction, restricted to one school. "Ryan Williams" is
    # ambiguous across four seasons of college football and completely
    # unambiguous at Alabama, and the team map - now that it resolves all
    # twenty-four codes - tells us which school every priced player is on.
    # This is the payoff for solving the team map properly.
    by_school: dict[tuple, set[str]] = {}
    if team_map is not None and "school" in hist:
        for athlete, school, keys in zip(hist["athlete_id"], hist["school"],
                                         hist["keys"]):
            for k in keys:
                s = _short(k)
                if s:
                    by_school.setdefault((school, s), set()).add(athlete)
    school_unique = {k: next(iter(v)) for k, v in by_school.items()
                     if len(v) == 1}

    # (school, athlete) -> that athlete's key spellings, for the structured
    # name-form pass below.
    roster_keys: dict[tuple, set[str]] = {}
    if team_map is not None and "school" in hist:
        for athlete, school_, keys in zip(hist["athlete_id"], hist["school"],
                                          hist["keys"]):
            roster_keys.setdefault((school_, athlete), set()).update(keys)

    ambiguous = 0
    by_team = 0
    fuzzy = 0
    teams_col = (list(out["team"]) if "team" in out
                 else [None] * len(out))
    names_col = list(out["name"])
    for i, keys in enumerate(out["keys"]):
        if ids[i] is not None:
            continue
        school = (team_map or {}).get(teams_col[i])
        hit = None
        for k in keys:
            short = _short(k)
            if not short:
                continue
            if short in unique:
                hit = unique[short]
                break
            if school is not None and (school, short) in school_unique:
                hit = school_unique[(school, short)]
                by_team += 1
                break
            if short in reduced:
                ambiguous += 1
        # Last resort, and only within one school: a structured name-form
        # match. Not blind edit distance - the first name must be a short
        # form of the other and the surname must actually correspond - and it
        # must identify exactly one athlete on that roster of ~80.
        if hit is None and school is not None and by_school:
            want_first, want_last = _split_name(primary_key(names_col[i]))
            found = {a for (sch, _), ids in by_school.items() if sch == school
                     for a in ids}
            hits = set()
            for athlete in found:
                for k in roster_keys.get((school, athlete), ()):  # noqa
                    f, l = _split_name(k)
                    if (first_names_compatible(want_first, f)
                            and surnames_compatible(want_last, l)):
                        hits.add(athlete)
                        break
            if len(hits) == 1:
                hit = next(iter(hits))
                fuzzy += 1
            elif len(hits) > 1:
                ambiguous += 1
        if hit is not None:
            ids[i] = hit

    # Object dtype, explicitly, so a caller checking `is None` or `isna` both
    # behave and no later assignment re-triggers the coercion above.
    out["athlete_id"] = pd.Series(ids, index=out.index, dtype=object)
    n = sum(1 for a in ids if a is not None)
    if n - strict - by_team - fuzzy > 0:
        log.info("join: %d extra matched by dropping a middle name, each "
                 "unambiguous in the full history",
                 n - strict - by_team - fuzzy)
    if fuzzy:
        log.info("join: %d matched on name FORM within their own school "
                 "(short form of a first name, compound or mistyped "
                 "surname), each identifying exactly one man on that roster",
                 fuzzy)
    if by_team:
        log.info("join: %d matched by name within their own school, where a "
                 "name ambiguous across all of college football is unique",
                 by_team)
    if ambiguous:
        log.info("join: %d left unmatched because dropping the middle name "
                 "matched more than one athlete - a guess there is worse "
                 "than a miss", ambiguous)
    log.info("join: %d of %d priced players matched to history (%.1f%%)",
             n, len(out), 100 * n / max(1, len(out)))
    return out


def join_quality(joined: pd.DataFrame) -> str:
    """What the misses cost, in salary terms rather than in count.

    A join is allowed to fail at the bottom of the price range, where a
    missing projection costs nothing because nobody was rostering them. It is
    not allowed to fail on an $8,500 quarterback. Counting misses cannot tell
    those apart; pricing them can.
    """
    miss = joined[joined["athlete_id"].isna()]
    hit = joined[joined["athlete_id"].notna()]
    if miss.empty:
        return "join: every priced player matched"
    lines = [f"join: {len(miss)} of {len(joined)} unmatched "
             f"({100*len(miss)/len(joined):.1f}%)",
             f"  unmatched salary: ${miss['salary'].min():,.0f}-"
             f"${miss['salary'].max():,.0f}, median "
             f"${miss['salary'].median():,.0f}",
             f"  matched   salary: ${hit['salary'].min():,.0f}-"
             f"${hit['salary'].max():,.0f}, median "
             f"${hit['salary'].median():,.0f}"]
    # Price is a weak proxy. DraftKings prices a true freshman third-string
    # quarterback at $4,500 precisely BECAUSE it has no data on him, so a
    # miss above the median price is not evidence of anything by itself.
    #
    # The published points per game is the sharp test, because it is
    # DraftKings telling us whether the player has played. A miss with 0.0
    # ppg is a player with no snaps and nothing to match - correct and
    # harmless. A miss with a POSITIVE ppg is a real failure: DraftKings
    # found his production and we did not.
    if "dk_points_per_game" in miss:
        played = miss[pd.to_numeric(miss["dk_points_per_game"],
                                    errors="coerce").fillna(0) > 0]
        lines.append("")
        lines.append(f"  of the {len(miss)} misses, {len(miss) - len(played)} "
                     f"have never scored a point for DraftKings either")
        if played.empty:
            lines.append("  EVERY unmatched player has 0.0 published points "
                         "per game - nothing was lost")
        else:
            lines.append(f"  {len(played)} unmatched players HAVE published "
                         f"production - these are real misses:")
            for r in played.nlargest(12, "dk_points_per_game").itertuples(
                    index=False):
                lines.append(f"    ${int(r.salary):>6,}  {r.position:<4} "
                             f"{str(r.name)[:26]:<27} {r.team:<6} "
                             f"{float(r.dk_points_per_game):>6.1f} ppg")
    return "\n".join(lines)


def diagnose_misses(joined: pd.DataFrame, hist: pd.DataFrame,
                    team_map: dict | None = None, limit: int = 12) -> str:
    """For every real miss, what the history DOES contain for his school.

    Three rounds of this were spent guessing at causes - the position filter,
    then name ambiguity - and fixing the guess. Each guess was plausible and
    the first was even a genuine bug, but neither moved this number.

    A miss has a small number of possible explanations and they are trivially
    distinguishable by looking: a near-identical name is a spelling variant,
    a same-school roster with nobody similar means he is absent from CFBD
    entirely, and an empty school means the team map or the school label is
    wrong. So print the evidence instead of theorising about it.
    """
    from difflib import SequenceMatcher

    bad = real_misses(joined)
    if bad.empty:
        return "no real misses to diagnose"
    lines = [f"Diagnosing {len(bad)} real misses - what the history holds "
             f"for each man's school:"]
    for r in bad.nlargest(limit, "dk_points_per_game").itertuples(index=False):
        school = (team_map or {}).get(getattr(r, "team", None))
        lines.append("")
        lines.append(f"  {str(r.name)[:30]:<31}{r.team} "
                     f"({school or 'SCHOOL UNMAPPED'})  "
                     f"{float(r.dk_points_per_game):.1f} ppg")
        if school is None or "school" not in hist:
            lines.append("    cannot search - no school for this player")
            continue
        pool = hist[hist["school"] == school]
        if pool.empty:
            lines.append(f"    NO history rows at all for {school} - the "
                         f"school label or the team map is wrong")
            continue
        target = primary_key(r.name)
        names = pool.drop_duplicates("athlete_id")[["name", "athlete_id"]]
        scored = sorted(
            ((SequenceMatcher(None, target, primary_key(n)).ratio(), n)
             for n in names["name"]), reverse=True)[:3]
        lines.append(f"    {len(names)} athletes on record for {school}; "
                     f"closest names:")
        for ratio, n in scored:
            lines.append(f"      {ratio:.2f}  {n}")
    return "\n".join(lines)


def real_misses(joined: pd.DataFrame) -> pd.DataFrame:
    """Unmatched players DraftKings says have actually produced.

    This is the number to gate on. The raw match rate counts third-string
    quarterbacks who have never taken a snap, and on a college board that is
    most of the roster - 850 priced players across 24 teams is about 35 a
    side, where perhaps 18 are fantasy-relevant. Failing a build because 40%
    of a board is unrosterable backups measures the wrong thing.
    """
    if "dk_points_per_game" not in joined:
        return joined.iloc[0:0]
    ppg = pd.to_numeric(joined["dk_points_per_game"],
                        errors="coerce").fillna(0)
    return joined[joined["athlete_id"].isna() & (ppg > 0)]


# ------------------------------------------------------------------ fixtures

_STOP = {"of", "the", "at", "and"}


def _tokens(school: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9&]+", str(school).lower()) if t]


def _clean_code(code) -> str:
    return re.sub(r"[^A-Z0-9&]", "", str(code).upper())


def team_aliases(teams: list[dict]) -> tuple[dict, dict]:
    """(official, constructed) code sets per school.

    CFBD's teams endpoint publishes an abbreviation and up to three alternate
    names, and those carry the codes no rule can derive: Georgia's
    abbreviation is literally UGA, which is not a prefix, a substring or an
    initialism of "Georgia". An earlier version worked from school names alone
    and mapped 6 of 24 codes; the difference is entirely this endpoint.

    The two tiers exist because guessing pollutes. Constructing "initials + U"
    gives Kentucky -> KU, and constructing "U + first letter" gives Kansas ->
    UK, so each code proposes both schools and the two fixtures deadlock, each
    waiting on the other to resolve first. CFBD's official abbreviations say
    Kentucky is UK and Kansas is KU with no ambiguity at all.

    So a code that any school claims OFFICIALLY is matched only against
    official sets. Constructed forms - CLEM, STAN, MSST - are consulted only
    for codes no school officially claims, where a guess is all there is.
    """
    official: dict[str, set[str]] = {}
    constructed: dict[str, set[str]] = {}
    for t in teams or []:
        school = t.get("school")
        if not school:
            continue
        toks = _tokens(school)
        flat = "".join(toks)

        strong = set()
        for field in ("abbreviation", "alt_name1", "alt_name2", "alt_name3",
                      "altName1", "altName2", "altName3"):
            v = t.get(field)
            if v:
                strong.add(_clean_code(v))
        strong.add(_clean_code(flat))          # the school's own name

        weak = set()
        for n in (2, 3, 4, 5, 6):
            if len(flat) >= n:
                weak.add(_clean_code(flat[:n]))
        meaningful = [x for x in toks if x not in _STOP]
        initials = "".join(x[0] for x in meaningful)
        if initials:
            weak.update({_clean_code(initials), _clean_code(initials + "u"),
                         _clean_code(initials + "st")})
        if meaningful:
            first = meaningful[0]
            weak.update({_clean_code(first[:4] + "st"),
                         _clean_code(first[:3] + "st"),
                         _clean_code("u" + first[0]),
                         _clean_code(first)})
        # Two-token schools get truncated per token: Utah State -> UTST.
        # "UT Martin" reaches UTST through first[:4] + "st" above, so this
        # code is claimed by several schools and none of them officially -
        # which is exactly the situation the zero-candidate relaxation in
        # fixture_team_map exists to recover from.
        if len(meaningful) >= 2:
            a, b = meaningful[0], meaningful[1]
            for n in (1, 2, 3, 4):
                weak.add(_clean_code(a[:n] + b[:2]))
                weak.add(_clean_code(a[:n] + b[0]))

        official[school] = {c for c in strong if c}
        constructed[school] = {c for c in weak if c} - official[school]
    return official, constructed


def fixture_team_map(board_df: pd.DataFrame, games: list[dict],
                     teams: list[dict] | None = None) -> dict:
    """DraftKings team code -> CFBD school, solved from the schedule.

    A hand-written map is twenty-four guesses that look like knowledge, and at
    least one is wrong in a way nobody can see.

    `UL` is the case that shaped this function. It is Louisiana to
    DraftKings, it reads equally well as Louisville, and - the part that
    matters - NO string rule can propose it, because the letters stand for
    "University of Louisiana" and neither appears in the school name. An
    earlier version required the string to nominate candidates and therefore
    nominated none, leaving UL unmapped: not wrong, but not solved either.

    So the string does not have to propose anything. A code that matches no
    school at all is treated as UNCONSTRAINED rather than as hopeless, and the
    fixture resolves it from the other side: TXST is plainly Texas State,
    exactly one CFBD game has Texas State at home, and whoever is visiting is
    UL. Solving propagates - each code resolved tightens every fixture it
    appears in - so this runs to a fixpoint rather than in one pass.

    Anything still ambiguous at the end is reported, never guessed. A wrong
    team code does not drop a player; it hands him the wrong opponent, the
    wrong implied total and the wrong correlation group, and produces a full
    board of confident numbers that are wrong in every row.
    """
    cfbd_fixtures = []
    for g in games:
        home = g.get("home_team") or g.get("homeTeam")
        away = g.get("away_team") or g.get("awayTeam")
        if home and away:
            cfbd_fixtures.append((str(away), str(home)))
    schools = {s for fx in cfbd_fixtures for s in fx}

    # Aliases from CFBD when available, school names alone otherwise. The
    # fallback works - it solved 6 of 24 - which is exactly why it is not
    # silent about being the fallback.
    official, constructed = team_aliases(teams) if teams else ({}, {})
    official = {s: a for s, a in official.items() if s in schools}
    constructed = {s: a for s, a in constructed.items() if s in schools}
    claimed = {c for codes in official.values() for c in codes}
    if teams:
        log.info("aliases for %d of %d scheduled schools, %d official codes",
                 len(official), len(schools), len(claimed))
    else:
        log.warning("no teams payload - falling back to name matching alone, "
                    "which maps roughly a quarter of DraftKings' codes")

    # How much a proposal is worth, for breaking ties between relaxations.
    # A code identical to the school's own name is near-certain. An
    # abbreviation is a convention, and conventions are exactly what the two
    # sources disagree about - UL is Louisiana to CFBD and Louisville to
    # DraftKings. A constructed guess is worth least of all.
    name_code = {s: _clean_code("".join(_tokens(s))) for s in schools}

    def strength(code: str) -> int:
        c = _clean_code(code)
        if any(c == v for v in name_code.values()):
            return 3                       # the school's own name
        if c in claimed:
            return 2                       # an official abbreviation
        if any(c in a for a in constructed.values()):
            return 1                       # something we made up
        return 0

    def proposes(code: str, school: str) -> bool:
        c = _clean_code(code)
        if not official:
            return _overlap(code, school)
        # A code some school claims officially is matched ONLY officially.
        if c in claimed:
            return c in official.get(school, set())
        return c in constructed.get(school, set())

    dk_fixtures = []
    for game in (board_df.dropna(subset=["game"])["game"].drop_duplicates()):
        away_code, _, home_code = str(game).partition(" @ ")
        if away_code.strip() and home_code.strip():
            dk_fixtures.append((away_code.strip(), home_code.strip()))

    # A code is "free" when nothing in this week's schedule proposes it. UL is
    # free - University of Louisiana shares no letters with "Louisiana" - and
    # the fixture resolves it from the other side.
    free = {code for fx in dk_fixtures for code in fx
            if not any(proposes(code, s) for s in schools)}
    if free:
        log.info("codes no string rule can propose, left to the fixtures: %s",
                 ", ".join(sorted(free)))

    mapping: dict[str, str] = {}

    def compatible(code: str, school: str) -> bool:
        if code in mapping:
            return mapping[code] == school
        if school in mapping.values():
            return False                  # another code already claimed it
        return code in free or proposes(code, school)

    swap = {"on": False}

    def candidates(away_code: str, home_code: str) -> list:
        """Schools for (away_code, home_code), in that order.

        The swap phase exists because "home" is not always a fact. At a
        neutral site it is a bookkeeping choice, and DraftKings and CFBD are
        free to make it differently - which is how SMU and Louisiana both
        appeared in the fetched schedule, each proposing exactly one correct
        school, with no fixture containing them in the orientation DraftKings
        printed. Trying the reverse is not a loosening: both sides must still
        match, and the pair must still be unique.
        """
        out = [(a, h) for a, h in cfbd_fixtures
               if compatible(away_code, a) and compatible(home_code, h)]
        if not out and swap["on"]:
            out = [(h, a) for a, h in cfbd_fixtures
                   if compatible(away_code, h) and compatible(home_code, a)]
        return out

    def run(pending: list) -> list:
        for _ in range(len(pending) + 1):
            progressed = False
            still = []
            for away_code, home_code in pending:
                cands = candidates(away_code, home_code)
                if len(cands) == 1:
                    a, h = cands[0]
                    if mapping.get(away_code, a) == a and \
                       mapping.get(home_code, h) == h:
                        mapping[away_code], mapping[home_code] = a, h
                        progressed = True
                        continue
                still.append((away_code, home_code))
            pending = still
            if not progressed:
                break
        return pending

    unsolved = run(list(dk_fixtures))

    # Only once the strict orientation has extracted everything it can. Doing
    # this earlier would let a reversed match win where a correct forward one
    # existed, and every code it maps is one the strict pass could not.
    if unsolved:
        swap["on"] = True
        before = len(mapping)
        unsolved = run(unsolved)
        if len(mapping) > before:
            log.info("%d code(s) matched only with home and away reversed - "
                     "neutral-site games, where the two sources disagree "
                     "about which side is nominally home",
                     len(mapping) - before)

    # A fixture with ZERO candidates is not ambiguous - it is misinformed.
    # Every school its codes propose is wrong, which is strictly worse than
    # proposing nothing, because proposing nothing earns the free treatment
    # that solved BAMA and this does not.
    #
    # UTST is the case. "UT Martin" builds UTST from its first token and Utah
    # Tech from its initials, while Utah State builds USST and UTAHST and
    # never UTST. So UTST confidently proposed four schools, none of them the
    # right one, and no fixture could contain any of them opposite Utah.
    #
    # Rather than chase every shortening DraftKings might invent, discard the
    # proposals that produced nothing and let the fixture decide, which is
    # what it is for.
    for _ in range(3):
        stuck = [fx for fx in unsolved if not candidates(*fx)]
        if not stuck:
            break
        # Relax ONE code at a time and keep the relaxation that yields a
        # unique answer. Two earlier rules both failed here:
        #
        #   Relaxing BOTH codes threw away the good constraint along with the
        #   bad one and turned zero candidates into several.
        #
        #   Relaxing only codes that proposed several schools assumed a code
        #   proposing exactly one was right. UL is the counterexample and it
        #   is not rare: CFBD's official abbreviation UL belongs to
        #   Louisiana, DraftKings used UL for Louisville on this board, and
        #   the fixture list settles it - SMU plays Louisville. A confident
        #   single proposal can simply be wrong, and an official alias is
        #   still only a proposal.
        #
        # Dropping one code's proposals at a time asks the narrowest question
        # that could help: is this fixture determined by the OTHER side
        # alone? If exactly one of the two relaxations gives a unique game,
        # that is the answer. If both do, they disagree, and a disagreement
        # is ambiguity - refuse it.
        progressed = False
        for fx in list(stuck):
            away_code, home_code = fx
            solutions = []
            for code in (away_code, home_code):
                if code in mapping or code in free:
                    continue
                free.add(code)
                found = candidates(away_code, home_code)
                free.discard(code)
                if len(found) == 1:
                    solutions.append((code, found[0]))
            if not solutions:
                continue
            if len(solutions) > 1:
                # Both sides determine a game, and they disagree. Give up the
                # weaker proposal: relaxing the code whose evidence is
                # flimsiest is the smallest concession that resolves it.
                # SMU is a school's actual name; UL is an abbreviation. Only
                # if the evidence is equally strong on both sides is this
                # genuine ambiguity, and then it is refused.
                solutions.sort(key=lambda s: strength(s[0]))
                if strength(solutions[0][0]) == strength(solutions[1][0]):
                    log.warning("  %s @ %s: both sides determine a different "
                                "game and the evidence is equally strong - "
                                "refusing to guess", away_code, home_code)
                    continue
                solutions = solutions[:1]
            code, (a, h) = solutions[0]
            if a in mapping.values() or h in mapping.values():
                continue
            mapping[away_code], mapping[home_code] = a, h
            unsolved.remove(fx)
            progressed = True
            log.info("%s @ %s had no candidate; ignoring %s's proposals "
                     "resolves it uniquely to %s @ %s",
                     away_code, home_code, code, a, h)
        if not progressed:
            break

    if unsolved:
        log.warning("%d fixtures did not resolve to exactly one CFBD game: %s",
                    len(unsolved),
                    ", ".join(f"{a} @ {h}" for a, h in unsolved[:6]))
        # Say WHY, per code, rather than leaving the next run to guess. A code
        # proposing zero schools is a missing alias; one proposing several is
        # a fixture the schedule could not narrow.
        for away_code, home_code in unsolved[:8]:
            for code in (away_code, home_code):
                if code in mapping:
                    continue
                hits = sorted(s for s in schools if proposes(code, s))
                if not hits:
                    log.warning("  %s proposes nothing (treated as free)",
                                code)
                else:
                    log.warning("  %s proposes %d: %s", code, len(hits),
                                ", ".join(hits[:5]))
            # Distinguish "the proposals are wrong" from "the game is not in
            # the fetched weeks at all". Both look like an unsolved fixture
            # and they need completely different fixes: the first is a naming
            # rule, the second is a wider week window.
            # Print what each proposed school is ACTUALLY doing that week.
            # Absence of a pairing has several causes that look identical -
            # wrong week, wrong code, neutral site - and listing the real
            # fixtures distinguishes them in one line instead of one run.
            any_seen = False
            for code in (away_code, home_code):
                for s in sorted(x for x in schools if proposes(code, x))[:2]:
                    real = [f"{a} @ {h}" for a, h in cfbd_fixtures
                            if a == s or h == s]
                    if real:
                        any_seen = True
                        log.warning("    %s (%s) actually plays: %s",
                                    code, s, "; ".join(real[:3]))
                    else:
                        log.warning("    %s (%s) plays in NO fetched fixture",
                                    code, s)
            if not any_seen:
                log.warning("  no proposed school appears in ANY fetched "
                            "fixture - widen the week window")
    log.info("team map: %d codes solved from %d fixtures",
             len(mapping), len(dk_fixtures))
    return mapping


def _overlap(code: str, school: str) -> bool:
    """Loose enough to propose, never loose enough to decide alone.

    This only nominates candidates; the both-sides fixture constraint is what
    actually decides. A permissive test here is safe and a strict one is not,
    because a strict test rejects BAMA/Alabama and leaves the fixture unsolved.
    """
    c = re.sub(r"[^a-z0-9]", "", str(code).lower())
    toks = _tokens(school)
    flat = "".join(toks)
    if not c or not flat:
        return False
    if c in flat or flat.startswith(c):
        return True
    initials = "".join(t[0] for t in toks)
    if c == initials:
        return True
    return any(t.startswith(c) or c.startswith(t[:4]) for t in toks if t)
