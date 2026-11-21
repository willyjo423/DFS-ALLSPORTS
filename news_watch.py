"""Late-news watcher for the slate clock.

Most NBA slates are decided by a scratch announced 30-60 minutes before lock,
and the fixed publish schedule can miss it by half an hour. The clock calls
`changes()` every few minutes through the evening. It compares the current
ESPN NBA injury statuses and DailyFaceoff starting-goalie statuses with the
last look and returns which sports have news worth a rebuild. The clock then
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
    def __init__(self):
        self.last = {"nba": None, "nhl": None}

    def changes(self) -> dict:
        """{sport: [what changed]} since the previous call. The first call
        only records a baseline."""
        found = {}
        for sport, snap, diff in (("nba", nba_snapshot, diff_nba),
                                  ("nhl", nhl_snapshot, diff_nhl)):
            try:
                cur = snap()
            except Exception as exc:                       # noqa: BLE001
                print(f"news watch: {sport} unavailable ({str(exc)[:80]})", flush=True)
                continue
            if self.last[sport] is not None:
                moved = diff(self.last[sport], cur)
                if moved:
                    found[sport] = moved
            self.last[sport] = cur
        return found
