"""Hockey data: MoneyPuck for history, DraftKings for the board.

Why MoneyPuck
-------------
It publishes the two things this model needs and the league's own API makes
awkward: per-game player rows going back years, and expected-goals columns
that are far better predictors of future scoring than goals are. It is free,
it is a plain CSV, and it does not rate-limit a nightly job.

THIS FILE WAS WRITTEN WITHOUT SEEING THE DATA
---------------------------------------------
MoneyPuck is not reachable from the machine this was written on, so every
column name below is a CANDIDATE rather than a fact. That is a real risk and
this project has paid for it before: seven guesses at a DraftKings player id
matched zero of 652 rows before somebody printed the payload.

So nothing here guesses twice. Every column this module needs is resolved by
`pick()`, which tries a list of likely names and, when none of them match,
PRINTS EVERY COLUMN IN THE FILE and stops. One run then answers the question
completely instead of one field per round trip.

Run `python nhl_data.py --probe` to dump every source's real shape - columns,
dtypes, row counts, three sample rows - without building anything. On a
GitHub runner, where the network works, that is the fastest way to turn this
file from careful guesses into fact.

Three traps this handles by construction
----------------------------------------
**The `situation` column.** MoneyPuck publishes one row per player PER GAME
STATE - all, 5on5, 5on4, 4on5, other. Reading the file without filtering
multiplies every player by five, and the resulting history looks plausible,
trains cleanly, and is wrong. Filtered explicitly, and the filter reports how
many rows it removed so a silent change is visible.

**Ice time units.** MoneyPuck publishes `icetime` in SECONDS; the league
publishes "18:42"; a human writes 18.7. All three are decided by looking at
the data rather than by assumption, because ice time is the single most
important column here and a factor-of-sixty error would not look like one.

**Season-aggregate leakage.** The season summary is a whole-season average,
so a feature built from it knows the outcome of the game it predicts. It is
loaded for reference and for the player directory ONLY, and the model is fit
on the game-by-game files, which carry dates. Stated here because using the
summary would be easier, faster, and completely invalid.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
import time
import unicodedata

import pandas as pd
import requests

log = logging.getLogger("nhl_data")

TIMEOUT = 30
RETRIES = 3

# --------------------------------------------------------------- MoneyPuck
MP = "https://moneypuck.com/moneypuck"

# Several shapes per source, because a public file that moves is a file that
# silently stops updating. The first that answers wins and which one answered
# is logged, so a move shows up as a line in the log rather than as a model
# that quietly trains on nothing.
SEASON_SKATERS = [
    f"{MP}/playerData/seasonSummary/{{season}}/regular/skaters.csv",
    f"{MP}/playerData/seasonSummary/{{season}}/regular/skaters.csv.gz",
]
SEASON_GOALIES = [
    f"{MP}/playerData/seasonSummary/{{season}}/regular/goalies.csv",
    f"{MP}/playerData/seasonSummary/{{season}}/regular/goalies.csv.gz",
]
GAME_SKATERS = [
    f"{MP}/playerData/careers/gameByGame/regular/skaters/{{pid}}.csv",
]
GAME_GOALIES = [
    f"{MP}/playerData/careers/gameByGame/regular/goalies/{{pid}}.csv",
]

# ------------------------------------------------------------- DraftKings
# The same two endpoints the baseball build settled on, and for the same
# reason: api.draftkings.com returns 403 to GitHub's runners - an IP block,
# not a wrong path - while the lobby's own host keeps answering. The API is
# tried first so this reverts by itself if the block ever lifts.
DK_CONTESTS = "https://www.draftkings.com/lobby/getcontests?sport=NHL"
DK_DRAFTABLES = ("https://api.draftkings.com/draftgroups/v1/draftgroups/"
                 "{dg}/draftables?format=json")
DK_PLAYERS = ("https://www.draftkings.com/lineup/getavailableplayers"
              "?draftGroupId={dg}")


class DataUnavailable(RuntimeError):
    """A source did not answer. The caller decides whether that is fatal."""


def _get(url: str, as_json: bool = False):
    last = None
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, timeout=TIMEOUT, headers={
                "User-Agent": "Mozilla/5.0 (compatible; dfs-model/1.0)"})
            if r.status_code == 200:
                return r.json() if as_json else r.content
            last = f"HTTP {r.status_code}"
        except Exception as exc:                               # noqa: BLE001
            last = f"{type(exc).__name__}: {str(exc)[:80]}"
        if attempt < RETRIES - 1:
            time.sleep(1.5 * (attempt + 1))
    raise DataUnavailable(f"{url} -> {last}")


def _read_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_get(url)), low_memory=False)


def _first_that_answers(patterns: list[str], **fmt) -> tuple[pd.DataFrame, str]:
    tried = []
    for p in patterns:
        url = p.format(**fmt)
        try:
            df = _read_csv(url)
        except DataUnavailable as exc:
            tried.append(str(exc)[:90])
            continue
        except Exception as exc:                               # noqa: BLE001
            tried.append(f"{url} did not parse: {str(exc)[:60]}")
            continue
        if len(df):
            return df, url
        tried.append(f"{url} -> empty")
    raise DataUnavailable("no source answered:\n  " + "\n  ".join(tried))


# --------------------------------------------------- resolving column names
def _key(h) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(h or "").strip().lower())


def pick(df: pd.DataFrame, wanted: list[str], what: str,
         required: bool = True) -> str | None:
    """The column that holds `what`, or a complete explanation and a stop.

    Exact match first, then a contains-match with the longest header winning,
    so "iceTimeSeconds" beats "iceTime" when a file carries both and the
    longer name is the more specific one.

    When nothing matches and the column is required this raises with EVERY
    header in the file attached. That is the whole point: guessing field names
    one at a time is what turns a build into an afternoon, and a single run
    that prints the truth ends it.
    """
    keys = {_key(c): c for c in df.columns}
    for w in wanted:
        if _key(w) in keys:
            return keys[_key(w)]
    for w in wanted:
        hits = sorted((c for k, c in keys.items() if _key(w) in k),
                      key=len, reverse=True)
        if hits:
            return hits[0]
    if not required:
        return None
    raise DataUnavailable(
        f"could not find the {what} column. Tried: {wanted}\n"
        f"EVERY COLUMN IN THIS FILE:\n  " + "\n  ".join(map(str, df.columns))
        + f"\n\nAdd the right name to the candidate list for {what}.")


# Candidates, in the order they are preferred. MoneyPuck's own names first,
# then the league's, then the obvious human spellings.
C_PLAYER_ID = ["playerId", "player_id", "id", "nhl_id"]
C_NAME = ["name", "playerName", "player", "fullName", "skaterFullName"]
C_TEAM = ["team", "teamAbbrev", "playerTeam", "triCode", "teamCode"]
C_POSITION = ["position", "pos", "positionCode", "primaryPosition"]
C_SEASON = ["season", "seasonId", "year"]
C_SITUATION = ["situation", "strength", "state", "gameState"]
C_GAME_ID = ["gameId", "game_id", "gamePk"]
C_DATE = ["gameDate", "date", "game_date"]
C_OPPONENT = ["opposingTeam", "opponent", "opponentTeam", "opp"]
C_HOME = ["home_or_away", "homeOrAway", "isHome", "home"]
C_ICETIME = ["icetime", "iceTime", "timeOnIce", "toi", "ice_time"]
C_PP_ICETIME = ["ppIcetime", "powerPlayTimeOnIce", "ppToi", "pp_icetime"]
C_SHOTS = ["I_F_shotsOnGoal", "shotsOnGoal", "shots", "sog"]
C_GOALS = ["I_F_goals", "goals", "g"]
C_ASSISTS = ["I_F_primaryAssists", "assists", "a"]
C_SECOND_ASSISTS = ["I_F_secondaryAssists", "secondaryAssists"]
C_BLOCKS = ["shotsBlockedByPlayer", "blockedShotAttempts", "blockedShots",
            "blocks"]
C_SHIFTS = ["shifts", "numShifts"]
C_GAMES = ["games_played", "gamesPlayed", "games", "gp"]
C_SAVES = ["saves", "savesFor"]
C_SHOTS_AGAINST = ["shotsAgainst", "ongoal_against", "shots_against"]
C_GOALS_AGAINST = ["goalsAgainst", "goals_against", "ga"]

# The situation value that means "everything". Detected rather than assumed,
# because "all" is the likely spelling and not a certainty.
SITUATION_ALL = ["all", "All", "ALL", "5on5,4on5,5on4,other"]


def _one_situation(df: pd.DataFrame, where: str) -> pd.DataFrame:
    """Collapse MoneyPuck's per-game-state rows to one row per player-game.

    Without this every player appears about five times - once for all, once
    for 5on5, once for each special-teams state - and the history silently
    quintuples. It trains cleanly and it is wrong, which is the expensive
    kind of wrong.
    """
    col = pick(df, C_SITUATION, "game situation", required=False)
    if col is None:
        log.info("%s: no situation column, so every row is already one "
                 "player-game", where)
        return df

    values = df[col].astype(str).str.strip()
    counts = values.value_counts()
    log.info("%s: situation column '%s' holds %s", where, col,
             ", ".join(f"{v}({n})" for v, n in counts.head(8).items()))

    keep = None
    for cand in SITUATION_ALL:
        if cand in set(values):
            keep = cand
            break
    if keep is None:
        keep = counts.index[0]
        log.warning("%s: no situation value named 'all'; using the most "
                    "common one, '%s'. If that is not the everything-state, "
                    "the history is a SUBSET and every rate will be wrong.",
                    where, keep)

    out = df[values == keep].copy()
    log.info("%s: kept %d of %d rows at situation '%s'",
             where, len(out), len(df), keep)
    return out


def to_minutes(s: pd.Series, where: str = "") -> pd.Series:
    """Ice time in minutes, whatever unit it arrived in.

    MoneyPuck publishes seconds, the league publishes "18:42", a spreadsheet
    holds 18.7. Decided by the DATA - a median near a thousand is seconds, a
    median near twenty is minutes - because ice time is the most important
    column in this model and a factor-of-sixty error would not look like one.
    It would look like every skater being unremarkable.
    """
    text = s.astype(str).str.strip()
    num = pd.to_numeric(s, errors="coerce")

    mmss = text.str.match(r"^\d{1,3}:\d{2}$", na=False)
    if mmss.any():
        parts = text[mmss].str.split(":", expand=True).astype(float)
        num.loc[mmss] = parts[0] + parts[1] / 60.0
        log.info("%s: %d ice-time values parsed from mm:ss", where,
                 int(mmss.sum()))

    med = float(num.dropna().median()) if num.notna().any() else 0.0
    if med > 200:
        log.info("%s: ice time looks like SECONDS (median %.0f), divided by 60",
                 where, med)
        return num / 60.0
    if med > 0:
        log.info("%s: ice time looks like MINUTES (median %.1f)", where, med)
    return num


def normalise(name) -> str:
    """The project's shared name normalisation. A period becomes a SPACE."""
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace(".", " ").replace("-", " ").replace("'", "")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s).strip()


# ------------------------------------------------------------- the sources
def season_summary(season: int, side: str = "skaters") -> pd.DataFrame:
    """The whole-season table. FOR THE PLAYER DIRECTORY, NOT FOR FEATURES.

    Every column here is a season average, which means it already contains
    the outcome of any game you might use it to predict. Fitting on it would
    produce a model that grades beautifully and cannot forecast. It is loaded
    to learn who exists, what they are called, what they play and where - and
    that is all.
    """
    pats = SEASON_SKATERS if side == "skaters" else SEASON_GOALIES
    df, url = _first_that_answers(pats, season=season)
    log.info("%s %s: %d rows from %s", season, side, len(df),
             url.rsplit("/", 1)[-1])
    df = _one_situation(df, f"{season} {side}")

    out = pd.DataFrame({
        "player_id": df[pick(df, C_PLAYER_ID, "player id")].astype(str),
        "name": df[pick(df, C_NAME, "player name")].astype(str),
        "team": df[pick(df, C_TEAM, "team")].astype(str),
        "season": season,
    })
    pos = pick(df, C_POSITION, "position", required=False)
    out["position"] = (df[pos].astype(str).str.upper().str.strip()
                       if pos else ("G" if side == "goalies" else ""))
    games = pick(df, C_GAMES, "games played", required=False)
    if games:
        out["games_played"] = pd.to_numeric(df[games], errors="coerce")
    out["norm"] = out["name"].map(normalise)
    return out


def game_by_game(player_ids, side: str = "skaters",
                 pause: float = 0.05) -> pd.DataFrame:
    """One row per player per game, which is what a model may be fitted on.

    Fetched per player, which is slow and unavoidable - MoneyPuck publishes
    careers, not slates. A failure on one player is logged and skipped rather
    than raised: losing one skater's history costs a little accuracy, and
    losing the whole fetch costs the night.
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
        df["__pid"] = pid
        frames.append(df)
        if n % 100 == 0:
            log.info("  %d of %d %s fetched", n, len(ids), side)
        time.sleep(pause)

    if not frames:
        raise DataUnavailable(
            f"not one {side} game log could be fetched. Either the URL shape "
            f"has moved or the player ids are wrong - run --probe to see.")
    if failed:
        log.warning("%d of %d %s had no game log and were skipped",
                    len(failed), len(ids), side)

    raw = pd.concat(frames, ignore_index=True)
    raw = _one_situation(raw, f"{side} game logs")
    return _tidy_games(raw, side)


def _tidy_games(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Per-game rows in the engine's vocabulary, or a loud explanation."""
    pid = pick(df, C_PLAYER_ID, "player id", required=False)
    out = pd.DataFrame({
        "player_id": (df[pid] if pid else df["__pid"]).astype(str),
        "game_id": df[pick(df, C_GAME_ID, "game id")].astype(str),
        "date": pd.to_datetime(df[pick(df, C_DATE, "game date")],
                               errors="coerce"),
    })
    for target, cands, need in (
            ("team", C_TEAM, True),
            ("opponent", C_OPPONENT, False),
            ("position", C_POSITION, False)):
        col = pick(df, cands, target, required=need)
        if col is not None:
            out[target] = df[col].astype(str).str.upper().str.strip()

    home = pick(df, C_HOME, "home or away", required=False)
    if home is not None:
        raw = df[home].astype(str).str.upper().str.strip()
        out["is_home"] = raw.isin(["H", "HOME", "1", "TRUE"]).astype(float)

    out["time_on_ice"] = to_minutes(df[pick(df, C_ICETIME, "ice time")],
                                    f"{side} ice time")
    pp = pick(df, C_PP_ICETIME, "power-play ice time", required=False)
    out["pp_time_on_ice"] = (to_minutes(df[pp], "pp ice time") if pp
                             else float("nan"))

    if side == "skaters":
        num = {
            "shots_on_goal": C_SHOTS, "goals": C_GOALS,
            "blocked_shots": C_BLOCKS, "shifts": C_SHIFTS,
        }
        for target, cands in num.items():
            col = pick(df, cands, target, required=(target != "shifts"))
            out[target] = (pd.to_numeric(df[col], errors="coerce") if col
                           else float("nan"))
        # Assists are published as primary and secondary. DraftKings pays the
        # same 5 points for both, so they are summed - and summed explicitly
        # rather than by taking whichever column matched first, which would
        # quietly drop a third of every playmaker's scoring.
        a1 = pick(df, C_ASSISTS, "primary assists", required=False)
        a2 = pick(df, C_SECOND_ASSISTS, "secondary assists", required=False)
        s1 = pd.to_numeric(df[a1], errors="coerce") if a1 else 0
        s2 = pd.to_numeric(df[a2], errors="coerce") if a2 else 0
        out["assists"] = pd.Series(s1).fillna(0) + pd.Series(s2).fillna(0)
        if a2 is None:
            log.warning("no secondary-assist column found, so assists may be "
                        "PRIMARY ONLY - every playmaker will be understated")
    else:
        for target, cands in (("saves", C_SAVES),
                              ("shots_against", C_SHOTS_AGAINST),
                              ("goals_against", C_GOALS_AGAINST)):
            col = pick(df, cands, target, required=False)
            out[target] = (pd.to_numeric(df[col], errors="coerce") if col
                           else float("nan"))
        out["position"] = "G"

    out["played"] = (out["time_on_ice"].fillna(0) > 0).astype(int)
    out = out[out["date"].notna()].sort_values(["player_id", "date"])
    log.info("%s: %d player-games, %s to %s", side, len(out),
             out["date"].min().date() if len(out) else "-",
             out["date"].max().date() if len(out) else "-")
    return out.reset_index(drop=True)


# --------------------------------------------------------------- the board
def slates() -> pd.DataFrame:
    """Tonight's DraftKings NHL draft groups, biggest first."""
    payload = _get(DK_CONTESTS, as_json=True)
    rows = []
    for g in payload.get("DraftGroups") or []:
        rows.append({
            "draft_group": g.get("DraftGroupId"),
            "game_type": g.get("GameTypeId"),
            "starts": pd.to_datetime(g.get("StartDateEst"), errors="coerce",
                                     utc=True),
            "example": g.get("ContestStartTimeSuffix") or "",
        })
    df = pd.DataFrame(rows).dropna(subset=["draft_group"])
    if df.empty:
        raise DataUnavailable("DraftKings is listing no NHL slates right now")
    counts = (pd.Series([c.get("dg") for c in payload.get("Contests") or []])
              .value_counts())
    df["contests"] = df["draft_group"].map(counts).fillna(0).astype(int)
    log.info("%d NHL draft groups listed", len(df))
    return df.sort_values("contests", ascending=False)


def _draft_rows(draft_group: int):
    """The board, from whichever DraftKings endpoint still answers."""
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
            f"draft group {draft_group} returned no players from either "
            f"endpoint (lobby top-level keys: {sorted(payload)[:10]})")
    log.info("draft group %s: %d rows from the lobby endpoint",
             draft_group, len(rows))
    return rows, "lobby"


def board(draft_group: int) -> pd.DataFrame:
    """Who is on the slate and what they cost, one row per player."""
    rows, dialect = _draft_rows(draft_group)
    df = pd.json_normalize(rows)

    out = pd.DataFrame({
        "dk_player_id": df[pick(df, ["playerId", "pid", "playerID"],
                                "DraftKings player id")].astype(str),
        "name": df[pick(df, ["displayName", "fullName", "name", "firstName"],
                        "player name")].astype(str),
        "salary": pd.to_numeric(df[pick(df, ["salary", "Salary"], "salary")],
                                errors="coerce"),
    })
    for target, cands in (("position", ["position", "rosterSlotName", "pos"]),
                          ("team", ["teamAbbreviation", "team", "teamAbbrev"])):
        col = pick(df, cands, target, required=False)
        out[target] = (df[col].astype(str).str.upper().str.strip() if col
                       else "")
    status = pick(df, ["status", "rosterStatus", "playerStatus"],
                  "status", required=False)
    out["dk_status"] = df[status].astype(str) if status else ""

    # DraftKings repeats a player once per roster slot he is eligible for, so
    # a centre who can fill UTIL arrives twice. Twice on the board means his
    # ownership is counted twice and a lineup could roster him twice, which is
    # not a legal entry.
    #
    # Deduped by ID and never by name. There are two Sebastian Ahos in this
    # league - a forward and a defenceman - and a name-based dedup deletes one
    # of them. A missing player leaves no row to be wrong about, so nothing
    # downstream can detect him.
    ids = pd.to_numeric(out["dk_player_id"], errors="coerce")
    if ids.notna().any():
        before = len(out)
        out = out[ids.notna()].drop_duplicates("dk_player_id", keep="first")
        if len(out) != before:
            log.info("collapsed %d multi-eligibility rows to one per player",
                     before - len(out))
    else:
        log.error("NOT ONE row carries a usable DraftKings player id, so "
                  "duplicate eligibilities cannot be collapsed. The board is "
                  "being used as-is and may contain a player twice.")

    out["norm"] = out["name"].map(normalise)
    log.info("draft group %s (%s dialect): %d players", draft_group, dialect,
             len(out))
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- the probe
def probe(season: int = 2025, draft_group: int | None = None) -> int:
    """Print what every source actually contains, and change nothing.

    This exists so the candidate lists above can stop being candidates. Run it
    once on a GitHub runner, paste the output back, and every guess in this
    file becomes a fact in one round trip instead of five.
    """
    def show(title, get):
        print("\n" + "=" * 72)
        print(title)
        print("=" * 72)
        try:
            df = get()
        except Exception as exc:                               # noqa: BLE001
            print(f"  FAILED  {type(exc).__name__}: {exc}")
            return
        if isinstance(df, list):
            print(f"  {len(df)} rows; keys of the first:")
            for k, v in sorted((df[0] or {}).items())[:60]:
                print(f"    {k:<32} {str(v)[:44]}")
            return
        print(f"  {len(df):,} rows x {len(df.columns)} columns")
        print("  COLUMNS:")
        for c in df.columns:
            print(f"    {str(c):<34} {str(df[c].dtype):<10} "
                  f"e.g. {str(df[c].dropna().iloc[0])[:30] if df[c].notna().any() else '-'}")
        print("\n  FIRST 3 ROWS:")
        print(df.head(3).to_string(max_colwidth=18))

    show(f"MONEYPUCK season summary, skaters, {season}",
         lambda: _first_that_answers(SEASON_SKATERS, season=season)[0])
    show(f"MONEYPUCK season summary, goalies, {season}",
         lambda: _first_that_answers(SEASON_GOALIES, season=season)[0])

    # One real player's game log. The id is taken from the summary rather than
    # typed in, so this cannot fail for the boring reason.
    def one_log():
        s, _ = _first_that_answers(SEASON_SKATERS, season=season)
        col = pick(s, C_PLAYER_ID, "player id")
        pid = str(s[col].dropna().iloc[0])
        print(f"  (using player id {pid})")
        return _first_that_answers(GAME_SKATERS, pid=pid)[0]
    show("MONEYPUCK game-by-game, one skater", one_log)

    show("DRAFTKINGS NHL slates", lambda: slates())

    def dk_rows():
        dg = draft_group
        if dg is None:
            dg = int(slates()["draft_group"].iloc[0])
            print(f"  (using draft group {dg})")
        return _draft_rows(int(dg))[0]
    show("DRAFTKINGS draftables, raw rows", dk_rows)

    print("\n" + "=" * 72)
    print("Paste this whole output back and the candidate lists become facts.")
    print("=" * 72)
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", action="store_true",
                   help="print every source's real shape and exit")
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--draft-group", type=int, default=None)
    args = p.parse_args(argv)
    if args.probe:
        return probe(args.season, args.draft_group)
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
