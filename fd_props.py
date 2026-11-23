"""FanDuel sportsbook player props (public endpoints, no login) for the NHL
and NBA boards, in the same shapes the Bovada readers produce so the two
books merge and everything downstream is unchanged.

NHL  -> {"Name|": {stat: [(k, P(X >= k)), ...]}}  (nhl_props ladders)
        goals (anytime scorer, 2+), points / assists / shots / blocks / saves
        milestones and shots over/unders.
NBA  -> {"Name|": {stat: median}}                  (nba_props medians)
        points, rebounds, assists, threes, steals, blocks: main over/under,
        alternate "Over x.5" ladder and "To score 25+" milestones, fitted to
        a median with nba_props.ladder_median.
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
import urllib.parse
import urllib.request

log = logging.getLogger("fd_props")
AK = "FhMFpcPWXMeyZxOx"
BASE = "https://sbapi.nj.sportsbook.fanduel.com/api"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0 Safari/537.36")
ONE_SIDED_POW = 1.12
TABS = {"nhl": ("", "popular", "goals", "points-assists", "shots", "goalies"),
        "nba": ("popular", "player-points", "player-rebounds", "player-assists",
                "player-threes", "player-defense", "player-combos")}
SKIP = ("PERIOD", "QUARTER", "HALF", "OVERTIME", "FIRST_", "1ST_", "2ND_", "3RD_",
        "GAME_SPECIALS", "TEAM_", "EACH_PERIOD", "COMBINE", "MOST_", "DOUBLE", "TRIPLE")


def _get(path: str, **params):
    params["_ak"] = AK
    url = f"{BASE}/{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def _prob(odds) -> float | None:
    try:
        v = float(odds)
    except (TypeError, ValueError):
        return None
    return 100 / (v + 100) if v > 0 else -v / (-v + 100)


def _odds(r):
    return ((r.get("winRunnerOdds") or {}).get("americanDisplayOdds") or {}).get("americanOdds")


def games(page: str) -> list[dict]:
    js = _get("content-managed-page", page="CUSTOM", customPageId=page)
    return [e for e in ((js.get("attachments") or {}).get("events") or {}).values()
            if " @ " in str(e.get("name")) and "(" not in str(e.get("name"))]


def markets_for(sport: str) -> list[dict]:
    """All player markets, one dict per game (marketId -> market)."""
    out = []
    for e in games(sport):
        mk: dict = {}
        for tab in TABS[sport]:
            try:
                ev = _get("event-page", eventId=e["eventId"], **({"tab": tab} if tab else {}))
                mk.update((ev.get("attachments") or {}).get("markets") or {})
            except Exception as exc:                           # noqa: BLE001
                log.debug("fanduel %s %s: %s", e.get("name"), tab, str(exc)[:60])
            time.sleep(0.25)
        out.append(mk)
    return out


# --------------------------------------------------------------------- NHL
NHL_STAT = (("SHOTS_ON_GOAL", "SOG"), ("BLOCKED_SHOTS", "BLK"), ("BLOCKS", "BLK"),
            ("ASSISTS", "AST"), ("POINTS", "PTS"), ("SAVES", "SV"), ("GOALS", "G"))


def _nhl_stat(text: str):
    t = text.upper().replace(" ", "_")
    return next((s for k, s in NHL_STAT if k in t), None)


def nhl_parse(markets: dict) -> dict:
    out: dict = {}

    def add(name, stat, k, p):
        if name and p is not None and 0.005 < p < 0.995:
            out.setdefault(f"{name.strip()}|", {}).setdefault(stat, []).append((int(k), p))

    for m in markets.values():
        mt = str(m.get("marketType") or "")
        if any(s in mt for s in SKIP):
            continue
        runners = m.get("runners") or []
        if mt == "ANY_TIME_GOAL_SCORER":
            for r in runners:
                p = _prob(_odds(r))
                add(r.get("runnerName"), "G", 1, p ** ONE_SIDED_POW if p else None)
            continue
        mm = re.match(r"PLAYER_TO_(?:RECORD|SCORE)_(\d+)\+_(.+)$", mt)
        if mm:                                   # milestone list: runners are players
            stat = _nhl_stat(mm.group(2))
            if stat:
                for r in runners:
                    p = _prob(_odds(r))
                    add(r.get("runnerName"), stat, int(mm.group(1)), p ** ONE_SIDED_POW if p else None)
            continue
        stat = _nhl_stat(mt) or _nhl_stat(str(m.get("marketName") or ""))
        if not stat or "PLAYER" not in mt:
            continue
        sides: dict = {}
        for r in runners:
            rn = str(r.get("runnerName") or "")
            p = _prob(_odds(r))
            ml = re.search(r"(\d+)\+", rn)
            name = re.split(r"\s*(?:-\s+)?(?:Over|Under|\d+\+)", rn)[0].strip(" -")
            if not name:                         # "21+" rows: "Cam Talbot Alt Saves"
                name = re.sub(r"\s+(Alt|Total)\b.*$", "", str(m.get("marketName") or "")).strip()
            if ml:
                add(name, stat, int(ml.group(1)), p ** ONE_SIDED_POW if p else None)
                continue
            hc = r.get("handicap")
            side = "O" if "Over" in rn else "U" if "Under" in rn else None
            if hc and side:
                sides.setdefault((name, float(hc)), {})[side] = p
        for (name, ln), s in sides.items():
            if s.get("O") and s.get("U"):
                add(name, stat, math.floor(ln) + 1, s["O"] / (s["O"] + s["U"]))
    return out


# --------------------------------------------------------------------- NBA
NBA_STAT = (("REBOUNDS", "R"), ("ASSISTS", "A"), ("THREE", "3"), ("MADE_3", "3"),
            ("STEALS", "S"), ("BLOCKS", "B"), ("POINTS", "P"))


def _nba_stat(text: str):
    t = text.upper().replace("POINTS", "PTS").replace("REBOUNDS", "REB").replace("ASSISTS", "AST")
    if "+" in t:                                 # combos: Bovada's PR / PA / RA / PRA
        has = [k for k in ("PTS", "REB", "AST") if k in t]
        return {("PTS", "REB", "AST"): "PRA", ("PTS", "REB"): "PR", ("PTS", "AST"): "PA",
                ("REB", "AST"): "RA"}.get(tuple(has))
    t = text.upper()
    return next((s for k, s in NBA_STAT if k in t), None)


def nba_parse(markets: dict) -> dict:
    """{"Name|": {stat: median}}"""
    import nba_props as NP
    rungs: dict = {}
    dd: dict = {}

    def add(name, stat, line, po, pu):
        if name and po is not None:
            rungs.setdefault(name.strip(), {}).setdefault(stat, []).append((float(line), po, pu))

    for m in markets.values():
        mt = str(m.get("marketType") or "")
        runners = m.get("runners") or []
        if mt in ("TO_RECORD_A_DOUBLE_DOUBLE", "TO_RECORD_A_TRIPLE_DOUBLE"):
            for r in runners:                    # bonus probabilities, kept as is
                p = _prob(_odds(r))
                if p and r.get("runnerName"):
                    dd.setdefault(r["runnerName"].strip(), {})[
                        "DD" if "DOUBLE_DOUBLE" in mt else "TD"] = round(p ** ONE_SIDED_POW, 4)
            continue
        if any(x in mt for x in SKIP):
            continue
        mt = re.sub(r"^(\d+)\+_MADE_THREES$", r"TO_RECORD_\1+_THREES", mt)
        mm = re.match(r"TO_(?:SCORE|RECORD)_(\d+)\+_(.+)$", mt)
        if mm:                                   # "25+ Points": runners are players
            stat = _nba_stat(mm.group(2))
            if stat:
                for r in runners:
                    p = _prob(_odds(r))
                    if p:
                        p = p ** ONE_SIDED_POW
                        add(r.get("runnerName"), stat, int(mm.group(1)) - 0.5, p, 1 - p)
            continue
        if not mt.startswith("PLAYER_"):
            continue
        stat = _nba_stat(mt)
        if not stat:
            continue
        name = str(m.get("marketName") or "").split(" - ")[0].strip()
        sides: dict = {}
        for r in runners:
            rn = str(r.get("runnerName") or "")
            p = _prob(_odds(r))
            alt = re.search(r"Over\s+(\d+(?:\.\d+)?)$", rn)
            if alt and not r.get("handicap"):      # alternate ladder, one-sided
                if p:
                    p = p ** ONE_SIDED_POW
                    add(name, stat, float(alt.group(1)), p, 1 - p)
                continue
            hc = r.get("handicap")
            side = "O" if rn.endswith("Over") else "U" if rn.endswith("Under") else None
            if hc and side:
                sides.setdefault(float(hc), {})[side] = p
        for ln, s in sides.items():
            if s.get("O") and s.get("U"):
                add(name, stat, ln, s["O"], s["U"])
    out = {}
    for name, st in rungs.items():
        med = {k: NP.ladder_median(v) for k, v in st.items()}
        med = {k: round(v, 2) for k, v in med.items() if v is not None}
        for k, v in st.items():                  # the ladder's spread too (ceilings)
            sd = _ladder_sd(v)
            if sd and k in med:
                med[f"sd_{k}"] = round(sd, 2)
        if med:
            out[f"{name}|"] = med
    for name, d in dd.items():
        out.setdefault(f"{name}|", {}).update(d)
    return out


def _ladder_sd(rungs) -> float | None:
    """Spread of a stat from its ladder: z = a + b*line on the de-vigged
    under-probabilities, sd = 1/b. Needs two or more distinct lines."""
    import numpy as np
    from statistics import NormalDist
    pts = []
    for line, po, pu in rungs:
        if po is None or pu is None or po + pu <= 0:
            continue
        p_over = po / (po + pu)
        if p_over < 0.25:                        # long shots: padded more
            p_over = p_over ** 1.10
        if 0.02 < p_over < 0.98:
            pts.append((line, NormalDist().inv_cdf(1 - p_over)))
    if len({x for x, _ in pts}) < 2:
        return None
    x = np.array([a for a, _ in pts]); z = np.array([b for _, b in pts])
    b, _a = np.polyfit(x, z, 1)
    return float(1 / b) if b > 0 else None


def fetch(sport: str) -> dict:
    parse = nhl_parse if sport == "nhl" else nba_parse
    out: dict = {}
    for mk in markets_for(sport):
        for k, v in parse(mk).items():
            if sport == "nhl":
                cur = out.setdefault(k, {})
                for s, pts in v.items():
                    cur.setdefault(s, []).extend(pts)
            else:
                out.setdefault(k, {}).update(v)
    log.info("fanduel: %s props for %d players", sport, len(out))
    return out


def _norm(s: str) -> str:
    import props_match as PM
    return PM.normalise(s)


def merge(a: dict, b: dict, ladders: bool) -> dict:
    """Bovada (a) + FanDuel (b). Ladders: rung by rung, same rung averaged.
    Medians: averaged where both books post the stat."""
    out = {k: (json.loads(json.dumps(v))) for k, v in a.items()}
    # one entry per player: a team-less copy (an earlier FanDuel-only fetch)
    # folds into the same player's Bovada entry, which carries his team
    idx: dict = {}
    for k in sorted(out, key=lambda k: k.endswith("|")):     # team keys first
        n = _norm(k.rsplit("|", 1)[0])
        if n in idx and k.endswith("|"):
            # fill only what the teamed entry lacks: its stats already hold
            # this book's lines, and averaging again would double-count them
            for stat, v in out.pop(k).items():
                out[idx[n]].setdefault(stat, v)
            continue
        idx.setdefault(n, k)
    for kb, sb in b.items():
        ka = idx.get(_norm(kb.rsplit("|", 1)[0]))
        tgt = out.setdefault(ka or kb, {})
        for stat, val in sb.items():
            if ladders:
                have: dict = {}
                for k, p in list(tgt.get(stat, [])) + list(val):
                    have.setdefault(int(k), []).append(p)
                tgt[stat] = sorted([k, sum(v) / len(v)] for k, v in have.items())
            else:
                tgt[stat] = round((tgt[stat] + val) / 2, 2) if stat in tgt else val
    return out
