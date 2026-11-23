"""Late-news watcher for the slate clock.

Most NBA slates are decided by a scratch announced 30-60 minutes before lock,
and the fixed publish schedule can miss it by half an hour. The clock calls
`changes()` every few minutes through the evening. It compares the current
ESPN NBA injury statuses, the NBA's posted starting lineups, DailyFaceoff
starting goalies and DailyFaceoff line combinations with the last look and returns which sports have news worth a rebuild. The clock then
dispatches a lock-lane run (`soon=90`), which rebuilds only the slates locking
in the next 90 minutes and does nothing if none are.

Only standard library + requests, so the clock job needs no other installs.
"""
from __future__ import annotations

import json
import re

import requests

UA = {"User-Agent": "Mozilla/5.0 (compatible; dfs-model/1.0)"}
ESPN_INJ = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
DFO_GOALIES = "https://www.dailyfaceoff.com/starting-goalies"
DFO = "https://www.dailyfaceoff.com"
NBA_LINEUPS = "https://stats.nba.com/js/data/leaders/00_daily_lineups_{day}.json"

# Statuses that change what a player is worth tonight. A note that changes
# while the status stays "Questionable" is not news.
NBA_LEVELS = {"out": 3, "doubtful": 2, "questionable": 1, "day-to-day": 1}


def nba_snapshot() -> dict:
    r = requests.get(ESPN_INJ, headers=UA, timeout=20)
    r.raise_for_status()
    out = {}
    for t in r.json().get("injuries") or []:
        for i in t.get("injuries") or []:
            nm = (i.get("athlete") or {}).get("displayName") or ""
            st = str(i.get("status") or "").lower()
            if nm:
                out[nm] = NBA_LEVELS.get(st, 0)
    return out


def nhl_snapshot() -> dict:
    r = requests.get(DFO_GOALIES, headers={**UA, "Accept": "text/html"}, timeout=20)
    r.raise_for_status()
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, re.S)
    if not m:
        return {}
    data = json.loads(m.group(1))["props"]["pageProps"].get("data") or []
    out = {}
    for g in data:
        for side in ("home", "away"):
            team = g.get(f"{side}TeamAbbreviation")
            if team:
                out[team] = (g.get(f"{side}GoalieName") or "",
                             g.get(f"{side}NewsStrengthName") or "Unconfirmed")
    return out


def _today_et() -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")


def nba_lineups_snapshot() -> dict:
    """{team: (confirmed?, starters, inactives)} from the NBA's own daily
    lineups file. A team flipping to Confirmed, a different starting five,
    or a new inactive is news: it moves minutes for the whole rotation."""
    r = requests.get(NBA_LINEUPS.format(day=_today_et()), timeout=20,
                     headers={**UA, "Referer": "https://www.nba.com/",
                              "Accept": "application/json"})
    r.raise_for_status()
    out = {}
    for g in r.json().get("games") or []:
        for side in ("homeTeam", "awayTeam"):
            t = g.get(side) or {}
            team = t.get("teamAbbreviation")
            pls = t.get("players") or []
            if not team or not any(p.get("lineupStatus") for p in pls):
                continue
            conf = any(p.get("lineupStatus") == "Confirmed" for p in pls)
            five = tuple(sorted(p.get("playerName") or "" for p in pls
                                if p.get("lineupStatus") and p.get("position")))
            inact = tuple(sorted(p.get("playerName") or "" for p in pls
                                 if str(p.get("rosterStatus") or "") == "Inactive"))
            out[team] = (conf, five, inact)
    return out


def diff_nba_lineups(old: dict, new: dict) -> list[str]:
    moved = []
    for team in sorted(new):
        if team not in old:
            moved.append(f"{team} lineup posted")
            continue
        (c0, f0, i0), (c1, f1, i1) = old[team], new[team]
        if c1 and not c0:
            moved.append(f"{team} lineup confirmed")
        if f0 != f1:
            moved.append(f"{team} starters " + ", ".join(sorted(set(f1) - set(f0))) + " in")
        if set(i1) - set(i0):
            moved.append(f"{team} inactive " + ", ".join(sorted(set(i1) - set(i0))))
    return moved


def _next_data(url: str) -> dict:
    r = requests.get(url, headers={**UA, "Accept": "text/html"}, timeout=20)
    r.raise_for_status()
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, re.S)
    return json.loads(m.group(1))["props"]["pageProps"] if m else {}


def nhl_lines_snapshot() -> dict:
    """{team: (top-9 forwards by line, PP1, out tags)} from Daily Faceoff's
    line-combinations pages for teams playing today. A new top-six, a PP1
    change or a new scratch is news."""
    slugs = {}
    for g in _next_data(DFO_GOALIES).get("data") or []:
        for side in ("home", "away"):
            if g.get(f"{side}TeamAbbreviation") and g.get(f"{side}TeamSlug"):
                slugs[g[f"{side}TeamAbbreviation"]] = g[f"{side}TeamSlug"]
    out = {}
    for team, slug in slugs.items():
        try:
            c = _next_data(f"{DFO}/teams/{slug}/line-combinations").get("combinations") or {}
        except Exception:                                   # noqa: BLE001
            continue
        lines, pp1, inj = [], [], []
        for p in c.get("players") or []:
            grp = str(p.get("groupIdentifier") or "").lower()
            cat = str(p.get("categoryIdentifier") or "").lower()
            nm = p.get("name") or ""
            if cat in ("ev", "") and grp in ("f1", "f2", "f3"):
                lines.append(f"{grp}:{nm}")
            elif cat in ("pp", "") and grp == "pp1":
                pp1.append(nm)
            if str(p.get("injuryStatus") or "").lower() in ("out", "ir", "ltir", "dtd"):
                inj.append(nm)
        if lines:
            out[team] = (tuple(sorted(lines)), tuple(sorted(pp1)), tuple(sorted(inj)))
    return out


def diff_nhl_lines(old: dict, new: dict) -> list[str]:
    moved = []
    for team in sorted(set(old) & set(new)):
        (l0, p0, i0), (l1, p1, i1) = old[team], new[team]
        if l0 != l1:
            moved.append(f"{team} lines changed")
        if p0 != p1:
            moved.append(f"{team} PP1 changed")
        if set(i1) - set(i0):
            moved.append(f"{team} out " + ", ".join(sorted(set(i1) - set(i0))))
    return moved


def diff_nba(old: dict, new: dict) -> list[str]:
    """Players whose status moved. Someone appearing on the report as out
    counts; someone dropping off it (cleared to play) counts too."""
    moved = []
    for nm in set(old) | set(new):
        a, b = old.get(nm, 0), new.get(nm, 0)
        if a != b and max(a, b) >= 2:       # out/doubtful in or out
            moved.append(f"{nm} {a}->{b}")
    return sorted(moved)


def diff_nhl(old: dict, new: dict) -> list[str]:
    moved = []
    for team in set(old) | set(new):
        if team in old and team in new and old[team] != new[team]:
            moved.append(f"{team} {old[team][0]} {old[team][1]} -> "
                         f"{new[team][0]} {new[team][1]}")
    return sorted(moved)


class Watcher:
    # (sport, snapshot function name, diff function name, every Nth look)
    # Line pages are one request per team, so they are read every other look.
    SOURCES = (("nba", "nba_snapshot", "diff_nba", 1),
               ("nba", "nba_lineups_snapshot", "diff_nba_lineups", 1),
               ("nhl", "nhl_snapshot", "diff_nhl", 1),
               ("nhl", "nhl_lines_snapshot", "diff_nhl_lines", 2))

    def __init__(self):
        self.last = {}
        self.looks = 0

    def changes(self) -> dict:
        """{sport: [what changed]} since the previous call. The first look
        at each source only records a baseline."""
        found = {}
        self.looks += 1
        for sport, snap, diff, every in self.SOURCES:
            if (self.looks - 1) % every:
                continue
            try:
                cur = globals()[snap]()
            except Exception as exc:                       # noqa: BLE001
                print(f"news watch: {snap} unavailable ({str(exc)[:80]})", flush=True)
                continue
            if self.last.get(snap) is not None:
                moved = globals()[diff](self.last[snap], cur)
                if moved:
                    found.setdefault(sport, []).extend(moved)
            self.last[snap] = cur
        return found
