"""Hockey data: MoneyPuck for history, DraftKings for the board.

Every column name in this file is a FACT, read from a probe run against the
live sources on 2026-09-29, not a guess. Where a name is pinned rather than
searched for, the comment says what the wrong match would have been - because
in five separate places the obvious search finds a real column that means
something else entirely, and each of those is a model that trains cleanly and
is wrong.

Re-run `python nhl_data.py --probe` whenever a source might have moved. It
prints every column, dtype and sample value and builds nothing.

The five traps, all confirmed from the real files
-------------------------------------------------
**`shotsBlockedByPlayer` is not `I_F_blockedShotAttempts`.** Both exist. The
first is blocks HE made, which DraftKings pays 1.3 for. The second is HIS OWN
shots that got blocked, which pays nothing and is close to a negative. A
search for "blocked" finds the wrong one on a coin flip.

**In the game logs the team column is `playerTeam`, and there is no `team`.**
A contains-match on "team" finds `playerTeam` AND `opposingTeam`, and with
longest-first tie-breaking `opposingTeam` wins - silently assigning every
player to the club he was playing against.

**`gameDate` is an INTEGER like 20191008.** Handing that to `pd.to_datetime`
without a format reads it as nanoseconds since 1970 and dates every game to
January 1970. Parsed through a string with an explicit format instead.

**Goalies have no saves column.** They have `ongoal` (shots faced) and
`goals` (goals allowed), and saves is the difference. A search for "saves"
finds nothing at all, which is the good outcome; a search for "goals" finds
`goals` and quietly reads goals-allowed as goals-scored.

**Power-play ice time is not a column, it is a ROW.** MoneyPuck publishes one
row per player per game state, so `5on4` icetime IS power-play ice time. The
file has no `ppIcetime` at all.

Season aggregates leak, and are not used for features
-----------------------------------------------------
`season_summary` is a whole-season average, so it already contains the outcome
of any game you would use it to predict. It is loaded for the player directory
- who exists, what they are called, where they play - and nothing else. The
model is fitted on the game-by-game files, which carry dates.
"""
from __future__ import annotations

import argparse
import io
import logging
import re
import sys
import time
import unicodedata

import pandas as pd
import requests

log = logging.getLogger("nhl_data")

# DraftKings' lobby reports lock times in Eastern. See `_lock_time`.
EASTERN = "America/New_York"

TIMEOUT = 30
RETRIES = 3

MP = "https://moneypuck.com/moneypuck"
SEASON_SKATERS = [f"{MP}/playerData/seasonSummary/{{season}}/regular/skaters.csv"]
SEASON_GOALIES = [f"{MP}/playerData/seasonSummary/{{season}}/regular/goalies.csv"]
GAME_SKATERS = [f"{MP}/playerData/careers/gameByGame/regular/skaters/{{pid}}.csv"]
GAME_GOALIES = [f"{MP}/playerData/careers/gameByGame/regular/goalies/{{pid}}.csv"]

# The probe confirmed api.draftkings.com is still refused and the lobby answers,
# so the lobby's abbreviated dialect is what this parses. The API is still tried
# first so the code reverts by itself if the block ever lifts.
DK_CONTESTS = "https://www.draftkings.com/lobby/getcontests?sport=NHL"
DK_DRAFTABLES = ("https://api.draftkings.com/draftgroups/v1/draftgroups/"
                 "{dg}/draftables?format=json")
DK_PLAYERS = ("https://www.draftkings.com/lineup/getavailableplayers"
              "?draftGroupId={dg}")

# MoneyPuck's per-game-state rows. "all" is the everything-state; "5on4" is the
# power play, which is the only way to get power-play ice time out of this feed.
SITUATION_ALL = "all"
SITUATION_PP = "5on4"


class DataUnavailable(RuntimeError):
    """A source did not answer. The caller decides whether that is fatal."""


def _get(url: str, as_json: bool = False):
    """One fetch, retried only where retrying can possibly help.

    A 403 IS NOT RETRIED, and that is the fix for a bug I wrote. A refusal is
    a decision the far end has already made; asking twice more, a second and a
    half apart, is three offences instead of one against a host that has just
    said no. Six publishes in forty minutes, each making about fourteen calls,
    each call retried three times on refusal, is how a working integration
    turns into a blocked one.

    5xx and timeouts ARE retried: those are the far end having a bad moment,
    which is exactly what a retry is for. 429 gets a long wait, because it is
    the one refusal that explicitly means "later".
    """
    last = None
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, timeout=TIMEOUT, headers={
                "User-Agent": "Mozilla/5.0 (compatible; dfs-model/1.0)"})
            if r.status_code == 200:
                return r.json() if as_json else r.content
            last = f"HTTP {r.status_code}"

            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After") or 20)
                log.warning("%s asked us to slow down (429); waiting %.0fs",
                            url.split("/")[2], wait)
                time.sleep(min(wait, 60))
                continue
            if 400 <= r.status_code < 500:
                raise DataUnavailable(
                    f"{url} -> HTTP {r.status_code}. This is a refusal, not a "
                    f"hiccup, so it is not being retried. DraftKings rate-"
                    f"limits by IP and a burst of publishes will earn a 403 "
                    f"for a while; wait and run again rather than retrying "
                    f"immediately, which only extends it.")
        except DataUnavailable:
            raise
        except Exception as exc:                               # noqa: BLE001
            last = f"{type(exc).__name__}: {str(exc)[:80]}"
        if attempt < RETRIES - 1:
            time.sleep(1.5 * (attempt + 1))
    raise DataUnavailable(f"{url} -> {last}")


def _read_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_get(url)), low_memory=False)


def _first_that_answers(patterns, **fmt):
    tried = []
    for p in patterns:
        url = p.format(**fmt)
        try:
            df = _read_csv(url)
        except Exception as exc:                               # noqa: BLE001
            tried.append(f"{url} -> {str(exc)[:70]}")
            continue
        if len(df):
            return df, url
        tried.append(f"{url} -> empty")
    raise DataUnavailable("no source answered:\n  " + "\n  ".join(tried))


def need(df: pd.DataFrame, col: str, what: str) -> pd.Series:
    """A column this file knows the exact name of, or a complete explanation.

    Deliberately NOT a fuzzy search. Every name here was read off the real
    file, and a fuzzy fallback is what puts `opposingTeam` in the team column.
    If a name has moved, the right response is a probe run and a one-line
    edit, not a guess that might land on a column meaning the opposite.
    """
    if col in df.columns:
        return df[col]
    raise DataUnavailable(
        f"the {what} column '{col}' is not in this file - MoneyPuck has "
        f"renamed it.\nEVERY COLUMN PRESENT:\n  "
        + "\n  ".join(map(str, df.columns))
        + "\n\nRun `python nhl_data.py --probe` and update the constant.")


def normalise(name) -> str:
    """The project's shared normalisation. A period becomes a SPACE."""
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace(".", " ").replace("-", " ").replace("'", "")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s).strip()


def _situation(df: pd.DataFrame, keep: str, where: str) -> pd.DataFrame:
    """One row per player-game, at one game state.

    Without this every player appears five times - all, 5on5, 5on4, 4on5,
    other - and the history silently quintuples. It trains cleanly and it is
    wrong, which is the expensive kind of wrong.
    """
    col = "situation"
    if col not in df.columns:
        log.info("%s: no situation column; rows are already one per game",
                 where)
        return df
    values = df[col].astype(str).str.strip()
    out = df[values == keep].copy()
    if not len(out):
        log.error("%s: no rows at situation '%s'. Present: %s", where, keep,
                  ", ".join(sorted(set(values))[:8]))
    else:
        log.info("%s: kept %d of %d rows at situation '%s'",
                 where, len(out), len(df), keep)
    return out


def _minutes(s: pd.Series, where: str = "") -> pd.Series:
    """Ice time in minutes. MoneyPuck publishes SECONDS.

    Decided by the data rather than asserted, because ice time is the most
    important column in this model and a factor-of-sixty error would not look
    like an error - it would look like every skater being unremarkable.
    """
    num = pd.to_numeric(s, errors="coerce")
    med = float(num.dropna().median()) if num.notna().any() else 0.0
    if med > 200:
        log.info("%s: ice time is SECONDS (median %.0f), divided by 60",
                 where, med)
        return num / 60.0
    return num


def _dates(s: pd.Series) -> pd.Series:
    """`gameDate` is an integer like 20191008, not an epoch.

    `pd.to_datetime(20191008)` reads nanoseconds and returns 1 January 1970,
    which sorts every game into one day and destroys every rolling feature
    without raising anything.
    """
    text = pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


# ------------------------------------------------------------- the sources
def season_summary(season: int, side: str = "skaters") -> pd.DataFrame:
    """The player directory. NOT features - every column here is a leak."""
    pats = SEASON_SKATERS if side == "skaters" else SEASON_GOALIES
    df, url = _first_that_answers(pats, season=season)
    log.info("%s %s: %d rows from %s", season, side, len(df),
             url.rsplit("/", 1)[-1])
    df = _situation(df, SITUATION_ALL, f"{season} {side}")

    out = pd.DataFrame({
        "player_id": need(df, "playerId", "player id").astype(str),
        "name": need(df, "name", "player name").astype(str),
        "team": need(df, "team", "team").astype(str).str.upper(),
        "position": need(df, "position", "position").astype(str).str.upper(),
        "games_played": pd.to_numeric(df.get("games_played"), errors="coerce"),
        "season": season,
    })
    out["norm"] = out["name"].map(normalise)
    return out


def game_by_game(player_ids, side: str = "skaters",
                 pause: float = 0.05) -> pd.DataFrame:
    """One row per player per game - the only thing a model may be fitted on.

    Fetched per player, because MoneyPuck publishes careers rather than
    slates. One player failing is logged and skipped: losing a skater's
    history costs a little accuracy, losing the fetch costs the night.
    """
    pats = GAME_SKATERS if side == "skaters" else GAME_GOALIES
    frames, failed = [], []
    ids = [str(p) for p in player_ids]
    for n, pid in enumerate(ids, start=1):
        try:
            df, _ = _first_that_answers(pats, pid=pid)
        except DataUnavailable as exc:
            failed.append(pid)
            if len(failed) <= 5:
                log.warning("no game log for %s (%s)", pid, str(exc)[:70])
            continue
        frames.append(df)
        if n % 100 == 0:
            log.info("  %d of %d %s fetched", n, len(ids), side)
        time.sleep(pause)

    if not frames:
        raise DataUnavailable(
            f"not one {side} game log could be fetched. Run --probe.")
    if failed:
        log.warning("%d of %d %s had no game log and were skipped",
                    len(failed), len(ids), side)

    raw = pd.concat(frames, ignore_index=True)
    return (_tidy_skaters(raw) if side == "skaters" else _tidy_goalies(raw))


def _identity(df: pd.DataFrame) -> pd.DataFrame:
    """The columns every game log shares, with the team trap handled.

    `playerTeam`, PINNED. There is no bare `team` column in these files, and a
    contains-match on "team" finds `playerTeam` and `opposingTeam` both - then
    prefers `opposingTeam` for being longer, which assigns every player to the
    club he was playing against. Everything downstream - stacks, correlation,
    the three-team rule - would be built on the wrong side of every game.
    """
    return pd.DataFrame({
        "player_id": need(df, "playerId", "player id").astype(str),
        # The NAME, which is the only key this player has in common with
        # DraftKings. MoneyPuck uses the league's id (8478402) and DraftKings
        # its own (878032), with no crosswalk between them, so a history
        # without names cannot be joined to a board at all.
        "name": need(df, "name", "player name").astype(str),
        "game_id": need(df, "gameId", "game id").astype(str),
        "date": _dates(need(df, "gameDate", "game date")),
        "team": need(df, "playerTeam", "the player's OWN team")
                .astype(str).str.upper(),
        "opponent": need(df, "opposingTeam", "opponent").astype(str).str.upper(),
        "is_home": (need(df, "home_or_away", "home or away")
                    .astype(str).str.upper().eq("HOME").astype(float)),
        "position": need(df, "position", "position").astype(str).str.upper(),
    })


def _tidy_skaters(raw: pd.DataFrame) -> pd.DataFrame:
    all_rows = _situation(raw, SITUATION_ALL, "skater game logs")
    out = _identity(all_rows)

    out["time_on_ice"] = _minutes(need(all_rows, "icetime", "ice time"),
                                  "skater ice time")
    out["shifts"] = pd.to_numeric(all_rows.get("shifts"), errors="coerce")
    out["shots_on_goal"] = pd.to_numeric(
        need(all_rows, "I_F_shotsOnGoal", "shots on goal"), errors="coerce")
    out["goals"] = pd.to_numeric(
        need(all_rows, "I_F_goals", "goals"), errors="coerce")

    # DraftKings pays 5 for a primary assist and 5 for a secondary one, so they
    # are SUMMED. Taking whichever column matched first would drop roughly a
    # third of every playmaker's scoring and look entirely plausible.
    a1 = pd.to_numeric(need(all_rows, "I_F_primaryAssists", "primary assists"),
                       errors="coerce").fillna(0)
    a2 = pd.to_numeric(need(all_rows, "I_F_secondaryAssists",
                            "secondary assists"), errors="coerce").fillna(0)
    out["assists"] = a1 + a2

    # BLOCKS HE MADE. `I_F_blockedShotAttempts` is the other thing entirely -
    # his own shots that got blocked - and it also matches a search for
    # "blocked". DraftKings pays 1.3 for this column and nothing for that one.
    out["blocked_shots"] = pd.to_numeric(
        need(all_rows, "shotsBlockedByPlayer", "blocks the player MADE"),
        errors="coerce")

    out["faceoffs_taken"] = (
        pd.to_numeric(all_rows.get("faceoffsWon"), errors="coerce").fillna(0)
        + pd.to_numeric(all_rows.get("faceoffsLost"), errors="coerce").fillna(0))

    # Expected goals, which predict future scoring far better than goals do -
    # the whole reason this model reads MoneyPuck rather than a box score.
    out["xgoals"] = pd.to_numeric(all_rows.get("I_F_xGoals"), errors="coerce")
    out["onice_xgoals"] = pd.to_numeric(all_rows.get("OnIce_F_xGoals"),
                                        errors="coerce")

    # POWER-PLAY ICE TIME IS A ROW, NOT A COLUMN. There is no ppIcetime in
    # this feed; the 5on4 rows' icetime is it. Joined back on player+game.
    pp = _situation(raw, SITUATION_PP, "power-play rows")
    if len(pp):
        pptoi = pd.DataFrame({
            "player_id": need(pp, "playerId", "player id").astype(str),
            "game_id": need(pp, "gameId", "game id").astype(str),
            "pp_time_on_ice": _minutes(need(pp, "icetime", "pp ice time"),
                                       "pp ice time"),
        }).drop_duplicates(["player_id", "game_id"])
        before = len(out)
        out = out.merge(pptoi, on=["player_id", "game_id"], how="left")
        if len(out) != before:
            raise DataUnavailable(
                f"the power-play join fanned out {before} rows to {len(out)}")
        hit = int(out["pp_time_on_ice"].notna().sum())
        log.info("power-play ice time on %d of %d skater-games (%.0f%%)",
                 hit, len(out), 100 * hit / max(len(out), 1))
    else:
        out["pp_time_on_ice"] = float("nan")
        log.warning("no 5on4 rows found - power-play ice time is empty, and "
                    "it is the largest single difference between two "
                    "otherwise identical forwards")

    out["played"] = (out["time_on_ice"].fillna(0) > 0).astype(int)
    return _finish(out, "skaters")


def _tidy_goalies(raw: pd.DataFrame) -> pd.DataFrame:
    all_rows = _situation(raw, SITUATION_ALL, "goalie game logs")
    out = _identity(all_rows)
    out["time_on_ice"] = _minutes(need(all_rows, "icetime", "ice time"),
                                  "goalie ice time")

    # THERE IS NO SAVES COLUMN. `ongoal` is shots on goal faced and `goals` is
    # goals allowed, so saves is the difference. Note especially that `goals`
    # here means goals AGAINST - a model that read it as goals scored would
    # reward a goalie for being beaten.
    shots = pd.to_numeric(need(all_rows, "ongoal", "shots on goal faced"),
                          errors="coerce")
    against = pd.to_numeric(need(all_rows, "goals", "goals ALLOWED"),
                            errors="coerce")
    out["shots_against"] = shots
    out["goals_against"] = against
    out["saves"] = (shots - against).clip(lower=0)
    out["xgoals_against"] = pd.to_numeric(all_rows.get("xGoals"),
                                          errors="coerce")
    out["position"] = "G"
    out["played"] = (out["time_on_ice"].fillna(0) > 0).astype(int)
    return _finish(out, "goalies")


def _finish(out: pd.DataFrame, side: str) -> pd.DataFrame:
    out = out[out["date"].notna()].sort_values(["player_id", "date"])
    dup = out.duplicated(["player_id", "game_id"]).sum()
    if dup:
        log.error("%d duplicate player-game rows survived the situation "
                  "filter - every rolling feature is now double-counting",
                  int(dup))
    log.info("%s: %d player-games, %s to %s", side, len(out),
             out["date"].min().date() if len(out) else "-",
             out["date"].max().date() if len(out) else "-")
    return out.reset_index(drop=True)


# --------------------------------------------------------------- the board
def _lock_time(value):
    """DraftKings' `StartDateEst`, as a real instant.

    THE FIELD NAME IS NOT DECORATION, AND READING IT AS UTC COSTS FOUR HOURS.
    It arrives naive - "2026-10-05T19:00:00" for a seven o'clock Eastern first
    puck - and `pd.to_datetime(..., utc=True)` stamps UTC onto it. Seven in the
    evening Eastern becomes seven in the evening UTC, which is three in the
    afternoon Eastern: every slate appears to lock four hours before it does.

    What that does downstream is worse than being four hours out. `slates()`
    feeds `next_slate_day`, which keeps only slates whose lock is still in the
    future - so from about three in the afternoon Eastern onwards, every one of
    tonight's boards looks ALREADY LOCKED. The publisher concludes the next
    enterable slate is tomorrow's, asks the stat model for tomorrow's file,
    does not find one, and falls back to the weaker in-repo model. The run goes
    green, boards are published, and the only visible trace is a log line
    saying a slate locks at 15:00 Eastern - which is not a time NHL hockey
    starts, and is exactly 19:00 with the offset eaten.

    A value that already carries an offset is trusted as-is, so this keeps
    working if DraftKings ever starts sending one.
    """
    t = pd.to_datetime(value, errors="coerce")
    if t is pd.NaT or pd.isna(t):
        return pd.NaT
    if t.tzinfo is not None:
        return t.tz_convert("UTC")
    try:
        # `ambiguous` and `nonexistent` matter twice a year: the hour that
        # happens twice in November and the one that does not exist in March
        # would otherwise raise and take the whole lobby read down.
        return t.tz_localize(EASTERN, ambiguous=True,
                             nonexistent="shift_forward").tz_convert("UTC")
    except Exception:                                          # noqa: BLE001
        log.warning("could not place %r on a clock; that slate carries no "
                    "lock time rather than a wrong one", value)
        return pd.NaT


_SLATES: pd.DataFrame | None = None


def slates(refresh: bool = False) -> pd.DataFrame:
    """Tonight's DraftKings NHL draft groups, busiest first. Fetched ONCE.

    The publisher asks three times in a single run - to pick the day, to pin a
    named draft group, and to list the boards - and the answer cannot change
    between them. Three identical calls is three times the rate-limit budget
    spent on one fact, which is a third of the way to the 403 that stopped a
    build tonight.
    """
    global _SLATES
    if _SLATES is not None and not refresh:
        return _SLATES

    payload = _get(DK_CONTESTS, as_json=True)
    rows = [{
        "draft_group": g.get("DraftGroupId"),
        "game_type": g.get("GameTypeId"),
        "starts": _lock_time(g.get("StartDateEst")),
        "example": g.get("ContestStartTimeSuffix") or "",
    } for g in payload.get("DraftGroups") or []]
    df = pd.DataFrame(rows).dropna(subset=["draft_group"])
    if df.empty:
        raise DataUnavailable("DraftKings is listing no NHL slates right now")
    counts = (pd.Series([c.get("dg") for c in payload.get("Contests") or []])
              .value_counts())
    df["contests"] = df["draft_group"].map(counts).fillna(0).astype(int)
    # SAID OUT LOUD, in the timezone a person thinks in, because the failure
    # this guards against is silent. An NHL board locks at first puck - seven
    # o'clock Eastern, give or take. A soonest lock in the early afternoon
    # means the offset has been eaten again.
    future = df[df["starts"].notna()]
    if len(future):
        soon = future["starts"].min()
        local = soon.tz_convert(EASTERN)
        log.info("%d NHL draft groups listed; soonest lock %s Eastern (%s UTC)",
                 len(df), local.strftime("%Y-%m-%d %H:%M"),
                 soon.strftime("%H:%M"))
        if 4 <= local.hour < 16:
            log.warning("that lock time is before four in the afternoon "
                        "Eastern, which is not when NHL hockey starts. If "
                        "every board looks locked from mid-afternoon, the "
                        "StartDateEst offset is being read wrongly again - "
                        "see nhl_data._lock_time.")
    else:
        log.info("%d NHL draft groups listed, none with a lock time", len(df))
    _SLATES = df.sort_values("contests", ascending=False).reset_index(drop=True)
    return _SLATES


def _draft_rows(draft_group: int):
    try:
        payload = _get(DK_DRAFTABLES.format(dg=draft_group), as_json=True)
        rows = payload.get("draftables") or []
        if rows:
            log.info("draft group %s: %d rows from the draftgroups API",
                     draft_group, len(rows))
            return rows, "api"
    except (DataUnavailable, ValueError) as exc:
        log.info("draftgroups API unavailable (%s); using the lobby endpoint",
                 str(exc)[:70])
    payload = _get(DK_PLAYERS.format(dg=draft_group), as_json=True)
    rows = (payload.get("playerList") or payload.get("draftables")
            or payload.get("players") or [])
    if not rows:
        raise DataUnavailable(
            f"draft group {draft_group} returned no players "
            f"(top-level keys: {sorted(payload)[:10]})")
    log.info("draft group %s: %d rows from the lobby endpoint",
             draft_group, len(rows))
    return rows, "lobby"


# The lobby's abbreviated dialect, read off the real payload. None of these
# are guessable and several are actively misleading: `s` is salary, `pn` is
# position, `or` is not a rank you want, and the name arrives in two halves.
LOBBY = {"id": "pid", "first": "fn", "last": "ln", "salary": "s",
         "position": "pn", "team_id": "tid", "away_id": "atid",
         "away": "atabbr", "home_id": "htid", "home": "htabbr",
         "disabled": "IsDisabledFromDrafting", "dk_ppg": "ppg"}
API = {"id": "playerId", "name": "displayName", "salary": "salary",
       "position": "position", "team": "teamAbbreviation",
       "disabled": "isDisabled"}


def board(draft_group: int) -> pd.DataFrame:
    """Who is on the slate and what they cost, one row per player."""
    rows, dialect = _draft_rows(draft_group)
    df = pd.json_normalize(rows)

    if dialect == "lobby":
        k = LOBBY
        name = (need(df, k["first"], "first name").astype(str).str.strip()
                + " "
                + need(df, k["last"], "last name").astype(str).str.strip())
        # The team arrives as an ID that must be matched against the game's
        # two sides. There is no team abbreviation on the player row itself.
        tid = pd.to_numeric(need(df, k["team_id"], "team id"), errors="coerce")
        atid = pd.to_numeric(df.get(k["away_id"]), errors="coerce")
        team = need(df, k["home"], "home team").astype(str).str.upper()
        away = need(df, k["away"], "away team").astype(str).str.upper()
        team = team.where(tid != atid, away)
        opp = away.where(tid != atid, need(df, k["home"], "home team")
                         .astype(str).str.upper())
        out = pd.DataFrame({
            "dk_player_id": need(df, k["id"], "player id").astype(str),
            "name": name,
            "salary": pd.to_numeric(need(df, k["salary"], "salary"),
                                    errors="coerce"),
            "position": need(df, k["position"], "position")
                        .astype(str).str.upper().str.strip(),
            "team": team, "opponent": opp,
            "dk_ppg": pd.to_numeric(df.get(k["dk_ppg"]), errors="coerce"),
            "disabled": df.get(k["disabled"], False).astype(bool),
        })
    else:
        k = API
        out = pd.DataFrame({
            "dk_player_id": need(df, k["id"], "player id").astype(str),
            "name": need(df, k["name"], "name").astype(str),
            "salary": pd.to_numeric(need(df, k["salary"], "salary"),
                                    errors="coerce"),
            "position": need(df, k["position"], "position")
                        .astype(str).str.upper().str.strip(),
            "team": need(df, k["team"], "team").astype(str).str.upper(),
            "opponent": "",
            "dk_ppg": float("nan"),
            "disabled": df.get(k["disabled"], False).astype(bool),
        })

    # DraftKings repeats a player once per roster slot he is eligible for. Two
    # rows means his ownership is counted twice and a lineup could roster him
    # twice, which is not a legal entry.
    #
    # Deduped by ID and NEVER by name. There are two Sebastian Ahos in this
    # league, a forward and a defenceman, and a name-based dedup deletes one
    # of them - leaving no row to be wrong about, so nothing downstream can
    # detect him. Exactly the two-Max-Muncys problem.
    ids = pd.to_numeric(out["dk_player_id"], errors="coerce")
    if ids.notna().any():
        before = len(out)
        out = out[ids.notna()].drop_duplicates("dk_player_id", keep="first")
        if len(out) != before:
            log.info("collapsed %d multi-eligibility rows", before - len(out))
        repeats = out["name"].value_counts()
        for nm, n in repeats[repeats > 1].items():
            log.info("'%s' appears %d times under different ids - different "
                     "players, both kept", nm, int(n))
    else:
        log.error("no usable DraftKings ids; duplicates cannot be collapsed")

    if out["disabled"].any():
        gone = sorted(out.loc[out["disabled"], "name"])
        log.info("dropping %d players DraftKings has disabled: %s",
                 len(gone), ", ".join(gone[:10]))
        out = out[~out["disabled"]]

    out["norm"] = out["name"].map(normalise)
    log.info("draft group %s (%s): %d players, %d teams, salaries $%s-$%s",
             draft_group, dialect, len(out), out["team"].nunique(),
             f"{int(out['salary'].min()):,}", f"{int(out['salary'].max()):,}")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- the probe
def probe(season: int = 2025, draft_group: int | None = None) -> int:
    """Print what every source contains and change nothing."""
    def show(title, get):
        print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)
        try:
            df = get()
        except Exception as exc:                               # noqa: BLE001
            print(f"  FAILED  {type(exc).__name__}: {exc}")
            return
        if isinstance(df, list):
            print(f"  {len(df)} rows; keys of the first:")
            for kk, v in sorted((df[0] or {}).items())[:60]:
                print(f"    {kk:<32} {str(v)[:44]}")
            return
        print(f"  {len(df):,} rows x {len(df.columns)} columns\n  COLUMNS:")
        for c in df.columns:
            ex = str(df[c].dropna().iloc[0])[:30] if df[c].notna().any() else "-"
            print(f"    {str(c):<38} {str(df[c].dtype):<10} e.g. {ex}")
        print("\n  FIRST 3 ROWS:")
        print(df.head(3).to_string(max_colwidth=18))

    show(f"MONEYPUCK season summary, skaters, {season}",
         lambda: _first_that_answers(SEASON_SKATERS, season=season)[0])
    show(f"MONEYPUCK season summary, goalies, {season}",
         lambda: _first_that_answers(SEASON_GOALIES, season=season)[0])

    def one_log():
        s, _ = _first_that_answers(SEASON_SKATERS, season=season)
        pid = str(s["playerId"].dropna().iloc[0])
        print(f"  (using player id {pid})")
        return _first_that_answers(GAME_SKATERS, pid=pid)[0]
    show("MONEYPUCK game-by-game, one skater", one_log)

    show("DRAFTKINGS NHL slates", slates)

    def dk_rows():
        dg = draft_group or int(slates()["draft_group"].iloc[0])
        print(f"  (using draft group {dg})")
        return _draft_rows(int(dg))[0]
    show("DRAFTKINGS draftables, raw rows", dk_rows)
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--draft-group", type=int, default=None)
    args = p.parse_args(argv)
    if args.probe:
        return probe(args.season, args.draft_group)
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
