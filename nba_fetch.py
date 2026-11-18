"""NBA game history from ESPN's public box scores, one row per player-game.

Why ESPN: it is free, keyless, reachable from GitHub Actions (stats.nba.com
routinely refuses cloud machines), and every box score carries what the
projection needs and nothing else does in one place:

* minutes, the full counting line, and whether he started;
* who did NOT play and why ("COACH'S DECISION", "INJURY/ILLNESS") - the
  raw material for "who was missing" in every past game, which is how the
  usage-redistribution effect is measured rather than assumed;
* the closing DraftKings spread and total ("pickcenter"), so the Vegas and
  blowout effects are fitted on real lines.

Incremental: games already in the cache are never fetched again.

    python nba_fetch.py --since 2025-10-21          # backfill
    python nba_fetch.py                             # yesterday and before
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import time

import pandas as pd
import requests

log = logging.getLogger("nba_fetch")

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "nba_games.csv.gz")
UA = {"User-Agent": "Mozilla/5.0 (compatible; dfs-model/1.0)"}
PAUSE = 0.25
SEASON_START = "2025-10-21"

# ESPN's abbreviations, put on DraftKings' vocabulary.
# ESPN and DraftKings each have their own spellings; both are put on this one.
TEAM = {"GS": "GSW", "NO": "NOP", "NOR": "NOP", "NY": "NYK", "SA": "SAS",
        "UTAH": "UTA", "WSH": "WAS", "PHO": "PHX", "BRK": "BKN", "BK": "BKN",
        "CHO": "CHA", "CHH": "CHA"}


def canon(t: str) -> str:
    t = str(t or "").upper()
    return TEAM.get(t, t)


def _get(url, **params):
    last = None
    for wait in (0, 2, 6, 15):
        if wait:
            time.sleep(wait)
        try:
            r = requests.get(url, params=params, headers=UA, timeout=30)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
            if r.status_code in (400, 404):
                break
        except requests.RequestException as exc:
            last = str(exc)[:80]
    raise RuntimeError(f"{url} {params}: {last}")


def games_on(day: dt.date) -> list[dict]:
    sb = _get(SCOREBOARD, dates=day.strftime("%Y%m%d"))
    out = []
    for ev in sb.get("events") or []:
        st = ((ev.get("status") or {}).get("type") or {})
        season_type = (ev.get("season") or {}).get("type")
        if st.get("completed") and season_type in (2, 3, 5):   # reg, post, play-in
            out.append({"id": str(ev["id"]), "date": day.isoformat()})
    return out


def _split(s):
    try:
        a, b = str(s).split("-")
        return int(a), int(b)
    except (ValueError, AttributeError):
        return 0, 0


def _num(s):
    try:
        return float(str(s).replace("+", ""))
    except (TypeError, ValueError):
        return 0.0


def parse(summary: dict, game_id: str, date: str) -> list[dict]:
    teams = {}
    for c in ((summary.get("header") or {}).get("competitions") or [{}])[0].get(
            "competitors") or []:
        teams[str(c.get("id"))] = {
            "team": canon((c.get("team") or {}).get("abbreviation")),
            "home": c.get("homeAway") == "home",
            "score": _num(c.get("score"))}
    # Closing line: spread is quoted for the home side.
    spread_home = total = None
    for pc in summary.get("pickcenter") or []:
        if pc.get("overUnder") is not None and spread_home is None:
            total = _num(pc.get("overUnder"))
            sp = pc.get("spread")
            fav_home = ((pc.get("homeTeamOdds") or {}).get("favorite"))
            if sp is not None:
                sp = _num(sp)
                # ESPN reports the favourite's number; make it the HOME line.
                spread_home = -abs(sp) if fav_home else abs(sp)
    rows = []
    for side in (summary.get("boxscore") or {}).get("players") or []:
        tid = str((side.get("team") or {}).get("id"))
        me = teams.get(tid, {})
        opp = next((v for k, v in teams.items() if k != tid), {})
        if not me:
            continue
        spread_me = (spread_home if me["home"] else
                     (-spread_home if spread_home is not None else None))
        for block in side.get("statistics") or []:
            keys = block.get("names") or block.get("labels") or []
            for a in block.get("athletes") or []:
                ath = a.get("athlete") or {}
                st = dict(zip(keys, a.get("stats") or []))
                fgm, fga = _split(st.get("FG"))
                tpm, tpa = _split(st.get("3PT"))
                ftm, fta = _split(st.get("FT"))
                played = bool(a.get("stats")) and not a.get("didNotPlay")
                rows.append({
                    "game_id": game_id, "date": date,
                    "team": me["team"], "opp": opp.get("team"),
                    "home": int(me["home"]),
                    "team_pts": me["score"], "opp_pts": opp.get("score"),
                    "spread": spread_me, "total": total,
                    "player_id": str(ath.get("id")),
                    "name": ath.get("displayName"),
                    "pos": ((ath.get("position") or {}).get("abbreviation")),
                    "starter": int(bool(a.get("starter"))),
                    "played": int(played),
                    "reason": a.get("reason") or "",
                    "min": _num(st.get("MIN")) if played else 0.0,
                    "pts": _num(st.get("PTS")), "fgm": fgm, "fga": fga,
                    "tpm": tpm, "tpa": tpa, "ftm": ftm, "fta": fta,
                    "reb": _num(st.get("REB")), "oreb": _num(st.get("OREB")),
                    "dreb": _num(st.get("DREB")), "ast": _num(st.get("AST")),
                    "tov": _num(st.get("TO")), "stl": _num(st.get("STL")),
                    "blk": _num(st.get("BLK")), "pf": _num(st.get("PF")),
                })
    return rows


def dk_points(df: pd.DataFrame) -> pd.Series:
    """DraftKings NBA classic scoring, bonuses included."""
    base = (df["pts"] + 0.5 * df["tpm"] + 1.25 * df["reb"] + 1.5 * df["ast"]
            + 2 * df["stl"] + 2 * df["blk"] - 0.5 * df["tov"])
    tens = sum((df[c] >= 10).astype(int) for c in ("pts", "reb", "ast", "stl", "blk"))
    return base + 1.5 * (tens >= 2) + 3.0 * (tens >= 3)


def load() -> pd.DataFrame:
    if not os.path.exists(CACHE):
        return pd.DataFrame()
    df = pd.read_csv(CACHE, dtype={"game_id": str, "player_id": str})
    return df


def update(since: str | None = None, until: dt.date | None = None,
           limit: int | None = None) -> pd.DataFrame:
    old = load()
    have = set(old["game_id"]) if len(old) else set()
    start = dt.date.fromisoformat(since or (
        old["date"].max() if len(old) else SEASON_START)[:10])
    end = until or (dt.date.today() - dt.timedelta(days=1))
    rows, n = [], 0
    day = start
    while day <= end:
        try:
            ids = games_on(day)
        except RuntimeError as exc:
            log.warning("%s: %s", day, exc)
            ids = []
        for g in ids:
            if g["id"] in have:
                continue
            try:
                rows += parse(_get(SUMMARY, event=g["id"]), g["id"], g["date"])
                have.add(g["id"])
                n += 1
            except Exception as exc:                           # noqa: BLE001
                log.warning("game %s: %s", g["id"], str(exc)[:80])
            time.sleep(PAUSE)
            if limit and n >= limit:
                break
        if limit and n >= limit:
            break
        day += dt.timedelta(days=1)
        time.sleep(PAUSE)
    new = pd.DataFrame(rows)
    out = pd.concat([old, new], ignore_index=True) if len(old) else new
    if len(out):
        out = out.drop_duplicates(["game_id", "player_id"], keep="last")
        out = out.sort_values(["date", "game_id", "team"]).reset_index(drop=True)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        out.to_csv(CACHE, index=False, compression="gzip")
    log.info("nba history: %d new games, %d rows total, %s to %s", n, len(out),
             out["date"].min() if len(out) else "-",
             out["date"].max() if len(out) else "-")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--since")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    update(a.since, limit=a.limit)
