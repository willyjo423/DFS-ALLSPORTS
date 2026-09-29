"""Baseball: the board, the box scores, and the join.

Why this should be far quicker than college football was
--------------------------------------------------------
The CFB build lost most of its time to identity: DraftKings' team codes did
not match CFBD's schools, names disagreed in four separate ways, 622 schools
were in scope, and the API had a monthly quota that ran out mid-build.

None of that applies here. The MLB StatsAPI is free, needs no key, has no
quota, covers thirty fixed teams, and publishes a STABLE NUMERIC PLAYER ID on
every box score. DraftKings publishes the same player's id in its own feed.
So the join is on an integer, not on a name, and the entire Matt/Matthew
Fuller problem simply does not arise.

What is genuinely harder than football
--------------------------------------
**Two sports in one.** Hitters and pitchers score under different rules, fill
different roster slots, and need different features. They get two specs and
two fitted models, not one model with a position flag - a strikeout means
opposite things to the two of them.

**Innings pitched are not decimal.** "6.2" means six and two-thirds, not six
point two. Reading it as a float understates every start by about a third of
an inning and silently mis-scores every pitcher on the board.

**Lineups land two hours before first pitch.** Batting order is the single
biggest driver of a hitter's day and is not knowable earlier. That is a live
problem for projections, not a historical one, and it is why the board is
fetched close to lock rather than the night before.
"""

from __future__ import annotations

import logging
import re
import time

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

UA = {"User-Agent": "Mozilla/5.0 (dfs research)"}
TIMEOUT = 45
RETRIES = 3

STATS = "https://statsapi.mlb.com/api/v1"
DK_CONTESTS = "https://www.draftkings.com/lobby/getcontests?sport=MLB"
DK_PLAYERS = ("https://www.draftkings.com/lineup/getavailableplayers"
              "?draftGroupId={dg}")

# DraftKings MLB Classic. Two scoring systems, because they are two games.
#
# These values are asserted, not derived, which makes them the most likely
# thing in this file to be wrong - so the verifier recomputes DraftKings'
# own published points per game from these rules and compares. That check
# caught nothing in college football because the rules were right; it is
# here precisely so that being wrong is loud rather than silent.
HITTER_SCORING = {
    "single": 3.0, "double": 5.0, "triple": 8.0, "home_run": 10.0,
    "rbi": 2.0, "run": 2.0, "walk": 2.0, "hbp": 2.0, "stolen_base": 5.0,
}
PITCHER_SCORING = {
    "innings": 2.25, "strikeout": 2.0, "win": 4.0,
    "earned_run": -2.0, "hit_allowed": -0.6, "walk_allowed": -0.6,
    "hbp_allowed": -0.6, "complete_game": 2.5, "shutout": 2.5,
    "no_hitter": 5.0,
}

HITTER_POSITIONS = ["C", "1B", "2B", "3B", "SS", "OF"]
PITCHER_POSITIONS = ["SP", "RP", "P"]


class Unavailable(RuntimeError):
    """A source this build cannot proceed without did not answer."""


# ----------------------------------------------------------------- transport

def _get(url: str, params=None, headers=None):
    last = None
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, params=params, headers=headers or UA,
                             timeout=TIMEOUT)
        except Exception as exc:                   # noqa: BLE001
            last = f"{type(exc).__name__}: {str(exc)[:110]}"
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code == 200:
            try:
                return r.json()
            except Exception as exc:               # noqa: BLE001
                raise Unavailable(
                    f"{url[:70]} returned 200 but not JSON "
                    f"({type(exc).__name__}): {r.content[:100]!r}") from exc
        last = f"HTTP {r.status_code}: {r.text[:110]}"
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2.0 * (attempt + 1))
            continue
        break
    raise Unavailable(f"{url[:70]} -> {last}")


# --------------------------------------------------------------- statsapi

def innings(value) -> float:
    """Innings pitched, which are thirds and not decimals.

    "6.2" is six and TWO THIRDS. Read as a float it is 6.2, which understates
    the start by about a sixth of an inning and, at 2.25 points an inning,
    mis-scores every pitcher on the board in the same direction. The error is
    small enough per start to look like rounding and large enough across a
    slate to change which pitcher the optimiser picks.
    """
    if value is None:
        return 0.0
    s = str(value).strip()
    if not s:
        return 0.0
    if "." not in s:
        try:
            return float(s)
        except ValueError:
            return 0.0
    whole, _, frac = s.partition(".")
    try:
        w = float(whole or 0)
    except ValueError:
        return 0.0
    thirds = {"0": 0.0, "1": 1.0 / 3.0, "2": 2.0 / 3.0}
    return w + thirds.get(frac[:1], 0.0)


# Which games count. The league's own codes: R is the regular season, then
# F wild card, D division series, L championship series, W the World Series.
#
# This was "R" alone, and on the last Tuesday of September that stops being a
# harmless simplification. The 2026 regular season ended Sunday 27 September
# and the wild card round began Tuesday the 29th, so a schedule call for that
# Tuesday returned ZERO games - no probable pitchers, no lineup cards - and
# the publisher correctly refused to build a board out of nothing. The page
# simply stopped updating, on the first day of the only baseball anybody is
# still playing.
#
# Spring training (S), exhibitions (E) and the All-Star game (A) stay out.
# Nobody drafts them and their rosters would poison a history fit.
GAME_TYPES = "R,F,D,L,W"


def schedule(season: int, start: str | None = None, end: str | None = None
             ) -> pd.DataFrame:
    """Every countable game, with its id and whether it finished."""
    params = {"sportId": 1, "season": season, "gameType": GAME_TYPES}
    if start and end:
        params.update({"startDate": start, "endDate": end})
    payload = _get(f"{STATS}/schedule", params=params)
    rows = []
    for day in payload.get("dates") or []:
        for g in day.get("games") or []:
            state = (g.get("status") or {}).get("codedGameState")
            teams = g.get("teams") or {}
            rows.append({
                "game_pk": g.get("gamePk"),
                "date": day.get("date"),
                "final": state == "F",
                "home_id": ((teams.get("home") or {}).get("team") or {}
                            ).get("id"),
                "away_id": ((teams.get("away") or {}).get("team") or {}
                            ).get("id"),
                "home": ((teams.get("home") or {}).get("team") or {}
                         ).get("name"),
                "away": ((teams.get("away") or {}).get("team") or {}
                         ).get("name"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise Unavailable(f"{season}: the schedule returned no games")
    log.info("%d: %d regular-season games, %d final",
             season, len(df), int(df["final"].sum()))
    return df


def probable_pitchers(date: str) -> dict[str, str]:
    """Today's announced starters: player id -> the game he starts in.

    The single most expensive thing a baseball board can get wrong. Every
    pitcher on a 26-man roster is priced; two of them start. The one who is
    not starting is cheap, and points per dollar is precisely the statistic a
    zero-inning pitcher maximises - so every objective picks him, every time.
    A star appearing in all ten lineups on a day he is not pitching is not a
    strange result; it is the only result a board without this can give.

    The league announces probables days ahead and publishes them free on the
    same endpoint the box scores come from. There is no excuse for a page that
    does not read them.

    An empty result means the league lists none, which the caller must treat
    as "do not publish" rather than as "nobody is starting today".
    """
    payload = _get(f"{STATS}/schedule",
                   params={"sportId": 1, "date": date,
                           "gameType": GAME_TYPES,
                           "hydrate": "lineups,probablePitcher"})
    ids, names = {}, {}
    opponent_of: dict[str, str] = {}
    order_id, order_name, posted_teams = {}, {}, set()
    team_ids: set[int] = set()
    games = sides = posted = 0

    for day in payload.get("dates") or []:
        for g in day.get("games") or []:
            games += 1
            teams = g.get("teams") or {}
            lineups = g.get("lineups") or {}

            # Who plays whom tonight. Needed because a hitter's own history
            # knows the last pitcher he faced and nothing about the next one,
            # and "the next one" is the largest thing about his evening that
            # his own record cannot tell you.
            sides_named = {}
            for s_ in ("home", "away"):
                i_ = (teams.get(s_) or {}).get("team") or {}
                sides_named[s_] = (i_.get("abbreviation") or i_.get("name"))
            if sides_named.get("home") and sides_named.get("away"):
                opponent_of[str(sides_named["home"])] = str(sides_named["away"])
                opponent_of[str(sides_named["away"])] = str(sides_named["home"])

            for side in ("home", "away"):
                info = (teams.get(side) or {}).get("team") or {}
                team = info.get("abbreviation") or info.get("name") or "?"
                if info.get("id") is not None:
                    team_ids.add(int(info["id"]))

                p = (teams.get(side) or {}).get("probablePitcher") or {}
                pid, full = p.get("id"), p.get("fullName")
                if pid is not None or full:
                    sides += 1
                    if pid is not None:
                        ids[str(pid)] = team
                    if full:
                        names[normalise(full)] = team

                # The batting order, once it is posted - which is about two
                # hours before first pitch. Nine names, IN ORDER, and the
                # order is the point: leading off is roughly one extra plate
                # appearance a game over batting eighth, which is the single
                # largest thing separating one hitter's day from another's.
                nine = lineups.get(f"{side}Players") or []
                if nine:
                    posted += 1
                    posted_teams.add(str(team))
                    for slot, pl in enumerate(nine, start=1):
                        if pl.get("id") is not None:
                            order_id[str(pl["id"])] = slot
                        if pl.get("fullName"):
                            order_name[normalise(pl["fullName"])] = slot

    # Both keys are kept for both things, because neither survives alone. The
    # league's id is correct, and DraftKings' copy of it came back empty for a
    # whole 278-row board - the run where a filter keyed only on the id would
    # have dropped every pitcher on the slate rather than only the ones
    # sitting.
    log.info("%s: %d games, %d of %d starters announced, %d of %d lineups "
             "posted (%d batters in order)", date, games, sides, games * 2,
             posted, games * 2, len(order_id))
    if games and posted < games * 2:
        log.warning("%d of %d lineups are not posted yet - those teams' "
                    "hitters cannot be confirmed, and a hitter who is rested "
                    "scores zero. Lineups go up about two hours before first "
                    "pitch.", games * 2 - posted, games * 2)
    return {"ids": ids, "names": names, "opponent_of": opponent_of,
            "order_id": order_id, "order_name": order_name,
            "posted_teams": posted_teams, "team_ids": sorted(team_ids),
            "games": games, "announced": sides, "posted": posted}


# A player is available only if the league says "Active". Everything else -
# the injured lists, reassignment to the minors, suspension, the restricted
# list - means he cannot appear tonight whatever DraftKings charges for him.
#
# Listing the ACTIVE code rather than the unavailable ones is deliberate. New
# status codes appear every year, and a blocklist silently treats an unknown
# one as fine; an allowlist treats it as unavailable, which is the safe
# direction when the cost of being wrong is a zero in a ten-man lineup.
ACTIVE_STATUS = {"A"}


def roster_status(team_ids: list[int]) -> dict:
    """Who is actually available, per the league's own roster.

    The lineup card answers this too, but only about two hours before first
    pitch. Before that a man on the sixty-day injured list is simply a cheap
    player with a full projection, and points per dollar picks him every time.
    Byron Buxton, out with a hip impingement, priced at the minimum, in every
    lineup the board produced.

    Returns both what is ACTIVE and everyone SEEN, because the difference
    between them is what makes this safe: a board player found on a roster and
    not active is out, and a board player found on no roster at all is unknown
    and is left alone. Not knowing and knowing he is out are different
    answers, and only one of them justifies deleting a man from the slate.
    """
    active_ids, active_names = set(), set()
    seen_ids, seen_names = set(), set()
    out_rows, failed = [], []

    for tid in team_ids:
        try:
            payload = _get(f"{STATS}/teams/{tid}/roster",
                           params={"rosterType": "40Man"})
        except Unavailable as exc:
            failed.append(tid)
            log.warning("roster for team %s unavailable (%s)", tid,
                        str(exc)[:70])
            continue
        for row in payload.get("roster") or []:
            person = row.get("person") or {}
            pid, full = person.get("id"), person.get("fullName")
            code = ((row.get("status") or {}).get("code") or "").strip()
            if pid is None and not full:
                continue
            key_id = str(pid) if pid is not None else None
            key_nm = normalise(full) if full else None
            if key_id:
                seen_ids.add(key_id)
            if key_nm:
                seen_names.add(key_nm)
            if code in ACTIVE_STATUS:
                if key_id:
                    active_ids.add(key_id)
                if key_nm:
                    active_names.add(key_nm)
            else:
                out_rows.append((full, code,
                                 (row.get("status") or {}).get("description")))

    log.info("rosters: %d teams read, %d active players, %d listed "
             "unavailable", len(team_ids) - len(failed), len(active_ids),
             len(out_rows))
    if failed:
        log.warning("%d team roster(s) did not load - their players cannot be "
                    "checked and are left alone", len(failed))
    return {"active_ids": active_ids, "active_names": active_names,
            "seen_ids": seen_ids, "seen_names": seen_names,
            "out": out_rows, "teams_read": len(team_ids) - len(failed)}


def boxscore(game_pk: int) -> dict:
    return _get(f"{STATS}/game/{game_pk}/boxscore")


def describe_payload(box: dict) -> str:
    """What the box score actually contains, for the first run to print.

    Written because this file's scoring rules are asserted from memory and
    the field names are not. A run that prints the real keys turns "the
    numbers look a bit off" into "hitByPitch is called hitByPitches here".
    """
    lines = ["box score shape:"]
    teams = box.get("teams") or {}
    lines.append(f"  top-level keys : {sorted(box)[:10]}")
    lines.append(f"  team keys      : {sorted(teams)}")
    for side in ("away", "home"):
        players = (teams.get(side) or {}).get("players") or {}
        if not players:
            continue
        lines.append(f"  {side}: {len(players)} players")
        for _, p in list(players.items())[:2]:
            st = p.get("stats") or {}
            bat = st.get("batting") or {}
            pit = st.get("pitching") or {}
            lines.append(f"    {(p.get('person') or {}).get('fullName')} "
                         f"({(p.get('position') or {}).get('abbreviation')})")
            if bat:
                lines.append(f"      batting keys : {sorted(bat)}")
            if pit:
                lines.append(f"      pitching keys: {sorted(pit)}")
        break
    return "\n".join(lines)


def _decisions(box: dict) -> set:
    """Player ids credited with a win, if the payload carries a decisions
    block at all.

    The boxscore endpoint does NOT: its top-level keys are copyright, info,
    officials, pitchingNotes, teams and topPerformers. The first live run
    found zero wins across 120 games, which is four points missing from
    every winning pitcher.

    The win is in fact sitting in each pitcher's own stats block as `wins`,
    which is where it is read from now. This stays as a fallback for the
    live-feed payload shape, which does carry decisions.
    """
    out = set()
    w = (box.get("decisions") or {}).get("winner") or {}
    if w.get("id") is not None:
        out.add(str(w["id"]))
    return out


def player_games(box: dict, game: dict) -> list[dict]:
    """One row per player who appeared, hitting and pitching side by side."""
    rows = []
    teams = box.get("teams") or {}
    winners = _decisions(box)
    for side in ("away", "home"):
        blob = teams.get(side) or {}
        team = ((blob.get("team") or {}).get("name")
                or game.get(side))
        opp = game.get("home" if side == "away" else "away")
        for _, p in (blob.get("players") or {}).items():
            person = p.get("person") or {}
            pid = person.get("id")
            if pid is None:
                continue
            st = p.get("stats") or {}
            bat, pit = st.get("batting") or {}, st.get("pitching") or {}
            if not bat and not pit:
                continue
            pos = (p.get("position") or {}).get("abbreviation")
            # Where he batted, which the box score has carried all along.
            #
            # "100" is leading off as a starter; "201" is the first
            # substitute in the two hole. The hundreds digit is the slot and
            # the last two digits are how deep into the substitutions he is,
            # so a pinch hitter is distinguishable from the man he replaced -
            # and only starters should teach the model what a slot is worth.
            #
            # This is the feature the projections have been missing. Without
            # it the model cannot tell a leadoff hitter from a nine-hole
            # hitter except through the plate appearances that result, which
            # is the effect rather than the cause and arrives a game late.
            raw_order = str(p.get("battingOrder") or "").strip()
            slot = started = None
            if raw_order.isdigit():
                slot = int(raw_order) // 100
                started = 1.0 if int(raw_order) % 100 == 0 else 0.0
                if not 1 <= slot <= 9:
                    slot = None
            hits = _n(bat.get("hits"))
            doubles = _n(bat.get("doubles"))
            triples = _n(bat.get("triples"))
            hr = _n(bat.get("homeRuns"))
            rows.append({
                "player_id": str(pid),
                "name": person.get("fullName"),
                "team": team, "opponent": opp,
                "is_home": 1 if side == "home" else 0,
                "position": pos,
                "game_pk": game.get("game_pk"),
                "date": game.get("date"),
                "bat_slot": slot,
                "bat_started": started,
                # hitting
                "single": max(0.0, hits - doubles - triples - hr),
                "double": doubles, "triple": triples, "home_run": hr,
                "rbi": _n(bat.get("rbi")), "run": _n(bat.get("runs")),
                "walk": _n(bat.get("baseOnBalls")),
                "hbp": _n(bat.get("hitByPitch")),
                "stolen_base": _n(bat.get("stolenBases")),
                "at_bats": _n(bat.get("atBats")),
                "plate_appearances": _n(bat.get("plateAppearances")),
                "strikeouts_batting": _n(bat.get("strikeOuts")),
                # pitching
                "innings": innings(pit.get("inningsPitched")),
                "strikeout": _n(pit.get("strikeOuts")),
                "earned_run": _n(pit.get("earnedRuns")),
                "hit_allowed": _n(pit.get("hits")),
                "walk_allowed": _n(pit.get("baseOnBalls")),
                # `hitBatsmen` is the canonical field for batters this
                # pitcher hit. `hitByPitch` also appears in the pitching
                # block and is not reliably the same thing, so it is only a
                # fallback.
                "hbp_allowed": _n(pit.get("hitBatsmen",
                                          pit.get("hitByPitch"))),
                "complete_game": _n(pit.get("completeGames")),
                "shutout": _n(pit.get("shutouts")),
                "batters_faced": _n(pit.get("battersFaced")),
                # From the pitcher's own line first. The boxscore endpoint
                # has no decisions block, and trusting one that is not there
                # cost every winning pitcher four points on the first run.
                "win": (1.0 if _n(pit.get("wins")) > 0
                        or str(pid) in winners else 0.0),
                "pitched": 1.0 if pit else 0.0,
            })
    return rows


def _n(v) -> float:
    if v is None:
        return 0.0
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------- scoring

def hitter_points(df: pd.DataFrame) -> pd.Series:
    pts = pd.Series(0.0, index=df.index)
    for field, weight in HITTER_SCORING.items():
        if field in df:
            pts = pts + weight * pd.to_numeric(df[field],
                                               errors="coerce").fillna(0)
    return pts.round(2)


def pitcher_points(df: pd.DataFrame) -> pd.Series:
    """Pitcher score, including the two bonuses that are conditional.

    A complete-game shutout pays the complete game AND the shutout, and a
    no-hitter pays on top of both. They are stacked rather than exclusive,
    which is easy to get backwards and worth stating.
    """
    pts = pd.Series(0.0, index=df.index)
    for field, weight in PITCHER_SCORING.items():
        if field in ("complete_game", "shutout", "no_hitter"):
            continue
        if field in df:
            pts = pts + weight * pd.to_numeric(df[field],
                                               errors="coerce").fillna(0)
    cg = pd.to_numeric(df.get("complete_game", 0), errors="coerce").fillna(0)
    so = pd.to_numeric(df.get("shutout", 0), errors="coerce").fillna(0)
    hits = pd.to_numeric(df.get("hit_allowed", 0), errors="coerce").fillna(0)
    pts = pts + PITCHER_SCORING["complete_game"] * (cg > 0)
    pts = pts + PITCHER_SCORING["shutout"] * (so > 0)
    pts = pts + PITCHER_SCORING["no_hitter"] * ((cg > 0) & (hits == 0))
    return pts.round(2)


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Add `points`, using whichever rule set the row belongs to."""
    out = df.copy()
    is_pitcher = pd.to_numeric(out.get("pitched", 0),
                               errors="coerce").fillna(0) > 0
    out["is_pitcher"] = is_pitcher.astype(int)
    out["points"] = np.where(is_pitcher, pitcher_points(out),
                             hitter_points(out))
    return out


# ------------------------------------------------------------- draftkings

_DOTNET_DATE = re.compile(r"/Date\((-?\d+)(?:[+-]\d{4})?\)/")


def start_time(raw) -> pd.Timestamp:
    """When a contest locks, in whichever shape DraftKings sends it.

    The lobby sends a .NET date - "/Date(1757894400000)/" - which
    `pd.to_datetime(..., unit="ms")` cannot read. With errors="coerce" every
    start time silently becomes NaT, and anything that filters on "has this
    started" then matches nothing at all. That exact parse cost the football
    build a whole capability once.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return pd.NaT
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return pd.to_datetime(int(raw), unit="ms", utc=True, errors="coerce")
    s = str(raw).strip()
    m = _DOTNET_DATE.search(s)
    if m:
        return pd.to_datetime(int(m.group(1)), unit="ms", utc=True,
                              errors="coerce")
    if s.lstrip("-").isdigit():
        return pd.to_datetime(int(s), unit="ms", utc=True, errors="coerce")
    return pd.to_datetime(s, utc=True, errors="coerce")


def slates() -> pd.DataFrame:
    """Every baseball draft group DraftKings is currently selling.

    Carries WHEN each one locks, which is the column that decides which slate
    is the next one to play rather than merely the biggest one on sale.
    """
    payload = _get(DK_CONTESTS)
    contests = payload.get("Contests") or []
    if not contests:
        raise Unavailable("the baseball lobby listed no contests")
    rows = {}
    for c in contests:
        dg = c.get("dg")
        if not dg:
            continue
        r = rows.setdefault(dg, {"draft_group": dg, "contests": 0,
                                 "game_type": c.get("gameType"),
                                 "starts_text": c.get("sdstring"),
                                 "starts": pd.NaT,
                                 "biggest_prize": 0, "biggest_field": 0,
                                 "example": c.get("n")})
        r["contests"] += 1
        r["biggest_prize"] = max(r["biggest_prize"], c.get("po") or 0)
        r["biggest_field"] = max(r["biggest_field"], c.get("m") or 0)
        t = start_time(c.get("sd"))
        if pd.notna(t) and (pd.isna(r["starts"]) or t < r["starts"]):
            r["starts"] = t
    out = pd.DataFrame(rows.values())
    got = int(out["starts"].notna().sum())
    log.info("lobby: %d draft groups, %d contests, %d with a start time",
             len(out), int(out["contests"].sum()), got)
    if len(out) and got == 0:
        log.error("NO draft group has a readable start time - sample %r. "
                  "Anything choosing the next slate to play will be choosing "
                  "blind.", (contests[0] or {}).get("sd"))
    return out.sort_values("contests", ascending=False).reset_index(drop=True)


def board(draft_group: int) -> pd.DataFrame:
    """Who is priced on a slate.

    The same lobby endpoint the football build uses, because api.draftkings
    .com answers 403 to GitHub's runners. Position is `pn`; `pp` is an
    integer that is zero for everybody, and mapping position to it once
    produced a board where every player shared one position and every
    positional constraint was vacuous.
    """
    payload = _get(DK_PLAYERS.format(dg=draft_group))
    raw = (payload.get("playerList") or payload.get("draftables")
           or payload.get("players") or [])
    if not raw:
        raise Unavailable(f"draft group {draft_group} returned no players "
                          f"(top-level keys: {sorted(payload)[:10]})")
    rows = []
    for p in raw:
        home, away = p.get("htabbr"), p.get("atabbr")
        tid, htid, atid = p.get("tid"), p.get("htid"), p.get("atid")
        rows.append({
            "dk_player_id": p.get("pid"),
            # DraftKings carries the MLB id, which is the whole point: the
            # join is an integer comparison, not a name match.
            "mlb_id": _first_id(p),
            "name": " ".join(x for x in (p.get("fn"), p.get("ln")) if x),
            "position": p.get("pn"),
            "team": home if tid == htid else away if tid == atid else None,
            "opponent": away if tid == htid else home if tid == atid else None,
            "is_home": 1 if tid == htid else 0 if tid == atid else np.nan,
            "salary": pd.to_numeric(p.get("s"), errors="coerce"),
            "dk_points_per_game": pd.to_numeric(p.get("ppg"),
                                                errors="coerce"),
            "disabled": bool(p.get("IsDisabledFromDrafting")),
            "game": f"{away} @ {home}" if home and away else None,
        })
    df = pd.DataFrame(rows)
    if df["position"].nunique() <= 1:
        raise Unavailable(
            f"every player on draft group {draft_group} has position "
            f"{df['position'].iloc[0]!r}. The position field moved.")
    if df["salary"].notna().sum() == 0:
        raise Unavailable(f"draft group {draft_group}: no player has a "
                          f"salary. The salary field moved.")
    # One row per PLAYER, where "same player" means the same DraftKings id -
    # never the same name.
    #
    # The lobby lists a man once per draftable entry, so somebody eligible at
    # two positions arrives twice and reaches the page twice: his ownership is
    # counted twice and a lineup can roster him twice. Max Muncy appeared
    # twice on the 2026-09-18 board.
    #
    # And Max Muncy is exactly why this collapses on the ID. There are two of
    # them in this league, on different teams, and a name-based dedup would
    # have silently deleted a real player from the board - a worse bug than
    # the one being fixed, and an invisible one, because a missing player
    # leaves no row to be wrong about. The same reasoning killed a name-based
    # join earlier in this project.
    #
    # So: collapse identical ids, keep distinct ids, and SAY which case each
    # repeated name was. The log then answers the question rather than
    # leaving it to be guessed at next time.
    before = len(df)
    ids = pd.to_numeric(df["dk_player_id"], errors="coerce")
    dupe_names = df["name"][df["name"].duplicated(keep=False)].unique()
    for who in dupe_names:
        rows = df[df["name"] == who]
        n_ids = rows["dk_player_id"].nunique(dropna=False)
        if n_ids > 1:
            log.info("%s is %d DIFFERENT players (%s) - both kept",
                     who, n_ids,
                     ", ".join(f"{t}/{p}" for t, p in
                               zip(rows["team"], rows["position"])))
        else:
            log.info("%s is listed %d times under one id (%s) - collapsed to "
                     "one row", who, len(rows),
                     "/".join(str(x) for x in rows["position"]))

    if ids.notna().any():
        df = df[ids.notna()].drop_duplicates("dk_player_id", keep="first")
        # A row with no id at all cannot be deduped and cannot be joined
        # either, so it is dropped rather than kept as an unmatched mystery.
        lost = before - len(df)
        if lost:
            log.info("collapsed %d duplicate or id-less rows (%d -> %d)",
                     lost, before, len(df))
    else:
        log.error("no player on draft group %s carries a DraftKings id, so "
                  "duplicates cannot be collapsed. A multi-position player "
                  "will appear twice on the page.", draft_group)

    log.info("draft group %s: %d players, %d teams, $%s-$%s",
             draft_group, len(df), df["team"].nunique(),
             int(df["salary"].min()), int(df["salary"].max()))
    return df.reset_index(drop=True)


# `pdkid` first, because the field dump showed it holding a real MLB id -
# 658796 for Jacob Misiorowski, where `pid` is DraftKings' own 1217479 and
# `tsid` is a third party's. Seven guessed names missed it because none of
# them guessed that a field called "player DK id" would carry the LEAGUE's
# id. Printing the payload found in one line what guessing had not.
#
# Order matters: the first field holding anything wins, so the one known to
# be right leads and the rest stay only as fallbacks.
_ID_FIELDS = ("pdkid", "mlbId", "MlbId", "mlbid", "sportsRadarId",
              "playerId", "externalId", "srid")


def board_row_keys(draft_group: int) -> str:
    """Every key DraftKings actually sends for a player, printed.

    Seven guessed field names found a league id on zero of 652 rows. Rather
    than guess an eighth, this prints what is really there - the same move
    that turned the box-score question from three rounds into one line.
    """
    payload = _get(DK_PLAYERS.format(dg=draft_group))
    raw = (payload.get("playerList") or payload.get("draftables")
           or payload.get("players") or [])
    if not raw:
        return "no players on that draft group"
    lines = [f"DraftKings sends {len(raw[0])} fields per player:"]
    for k, v in sorted(raw[0].items()):
        lines.append(f"    {k:<22} {str(v)[:44]}")
    return "\n".join(lines)


def _first_id(p: dict):
    """DraftKings' copy of the league's own player id, wherever it lives.

    Field naming in this feed has moved before, so several are tried and the
    verifier reports how many rows found one. If none does, the join falls
    back to names and the build says so loudly rather than quietly getting
    worse.
    """
    for f in _ID_FIELDS:
        v = p.get(f)
        if v not in (None, "", 0):
            return str(v)
    return None


def normalise(name) -> str:
    s = str(name or "").lower().strip()
    if "," in s:
        last, _, first = s.partition(",")
        s = f"{first.strip()} {last.strip()}"
    s = s.replace("-", " ").replace(".", " ").replace("'", "")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    return " ".join(s.split())


def attach_history(board_df: pd.DataFrame, hist: pd.DataFrame
                   ) -> pd.DataFrame:
    """Join the board to history. By id where possible, by name otherwise."""
    out = board_df.copy()
    ids = set(hist["player_id"].astype(str))
    matched, how = [], []
    by_name = {}
    for pid, nm in zip(hist["player_id"].astype(str), hist["name"]):
        by_name.setdefault(normalise(nm), pid)
    for mlb_id, nm in zip(out["mlb_id"], out["name"]):
        if mlb_id is not None and str(mlb_id) in ids:
            matched.append(str(mlb_id))
            how.append("id")
            continue
        hit = by_name.get(normalise(nm))
        matched.append(hit)
        how.append("name" if hit else "none")
    out["player_id"] = pd.Series(matched, index=out.index, dtype=object)
    out["matched_by"] = how
    n = sum(1 for m in matched if m is not None)
    log.info("join: %d of %d matched (%d by id, %d by name)",
             n, len(out), how.count("id"), how.count("name"))
    return out
