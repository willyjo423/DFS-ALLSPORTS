"""NBA player props from Bovada, turned into a market DraftKings projection.

Bovada posts full-game over/under ladders (several alternate lines per
player) for points, rebounds, and the points/rebounds/assists combos. From
each ladder we take the de-vigged chance of the over at every line and fit a
normal distribution to it (probit on the line), which gives the market's
median for that stat. Assists come from the combos (PRA - P - R, PA - P,
RA - R). The rest of a DraftKings score - threes, steals, blocks, turnovers,
double-double bonuses - is not posted, so it comes from the player's own
recent per-minute rates.

Every fetch is saved under docs/data/props/nba/<date>.json, which both acts as
a fallback when Bovada is unreachable and builds a history to grade the blend
weight against once enough nights have been recorded.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import urllib.request
from datetime import datetime, timezone
from statistics import NormalDist

import numpy as np
import pandas as pd

log = logging.getLogger("nba_props")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "docs", "data", "props", "nba")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124 Safari/537.36")
LIST_URL = ("https://www.bovada.lv/services/sports/event/coupon/events/A/description/"
            "basketball/nba?marketFilterId=def&preMatchOnly=true&lang=en")
EVENT_URL = "https://www.bovada.lv/services/sports/event/v2/events/A/description{}?lang=en"

# "Total Points, Rebounds and Assists - Alperen Sengun (HOU)"
MARKET_RE = re.compile(r"^Total (?P<stat>.+?) - (?P<name>.+?) \((?P<team>[A-Z]{2,4})\)$")
STATS = {
    "points": "P", "rebounds": "R", "assists": "A",
    "points and rebounds": "PR", "points and assists": "PA",
    "rebounds and assists": "RA", "points, rebounds and assists": "PRA",
    "made threes": "3", "three pointers made": "3", "made 3 point shots": "3",
    "steals": "S", "blocks": "B", "turnovers": "T",
}
N = NormalDist()

# Graded 2026-11-18 (114 players with props who played): props MAE 7.74 vs
# model 8.31; regressing (actual - model) on (props - model) gave a weight of
# 0.98 on props, and the gap predicted the miss in BOTH directions (props 2-5
# below model -> actual 6.3 below; 5+ above -> 5.5 above). One night, so 0.85,
# not 1.0; regrade as nights accumulate.
# Weight on the market at full coverage. Sharp NBA shops lean on props more
# than on any public model; ours is a minutes x rate model with no news feed
# beyond injuries and starters, so the market gets the majority. Regrade this
# against docs/data/props/nba history once there are ~30 nights of it.
import props_weight as _PW
W_MARKET = _PW.weight("nba", 0.85)   # learned from nightly grades once 10+ nights
W_GAP_MAX = min(0.97, max(W_MARKET + 0.10, 0.95))          # market weight once model and props are 25%+ apart
DK_W = {"P": 1.0, "R": 1.25, "A": 1.5}


def _american_to_prob(a) -> float | None:
    if a in (None, ""):
        return None
    a = str(a).upper()
    if a == "EVEN":
        return 0.5
    v = float(a)
    return 100 / (v + 100) if v > 0 else -v / (-v + 100)


def ladder_median(rungs: list[tuple[float, float, float]]) -> float | None:
    """rungs: (line, p_over_raw, p_under_raw). De-vig each rung, then fit
    z = a + b*line where z = Phi^-1(P(under)); median = -a/b."""
    pts = []
    for line, po, pu in rungs:
        if po is None or pu is None or po + pu <= 0:
            continue
        p_over = po / (po + pu)
        if 0.02 < p_over < 0.98:
            pts.append((line, N.inv_cdf(1 - p_over)))
    if not pts:
        return None
    if len(pts) >= 2:
        x = np.array([p[0] for p in pts]); z = np.array([p[1] for p in pts])
        if x.std() > 0:
            b, a = np.polyfit(x, z, 1)
            if b > 0:
                med = -a / b
                # keep a wild fit inside the ladder's own range
                return float(np.clip(med, x.min() - 3, x.max() + 3))
    # One rung: sd from the usual count-stat spread.
    line, z = min(pts, key=lambda p: abs(p[1]))
    sd = max(1.0, 1.1 * math.sqrt(max(line, 1.0)))
    return float(line - z * sd)


def parse_event(ev: dict) -> dict:
    """{(name, team): {stat: median}} for one Bovada event."""
    out: dict = {}
    for g in ev.get("displayGroups", []):
        if "Player" not in g.get("description", ""):
            continue
        for m in g.get("markets", []):
            if (m.get("period") or {}).get("description") not in (None, "Game", "Live Game"):
                continue
            mm = MARKET_RE.match(m.get("description", "").strip())
            if not mm:
                continue
            stat = STATS.get(mm["stat"].lower())
            if not stat:
                continue
            by_line: dict = {}
            for o in m.get("outcomes", []):
                pr = o.get("price") or {}
                if pr.get("handicap") in (None, ""):
                    continue
                ln = float(pr["handicap"])
                side = (o.get("type") or o.get("description", "")[:1]).upper()[:1]
                by_line.setdefault(ln, {})[side] = _american_to_prob(pr.get("american"))
            rungs = [(ln, d.get("O"), d.get("U")) for ln, d in sorted(by_line.items())]
            med = ladder_median(rungs)
            if med is not None:
                out.setdefault((mm["name"].strip(), mm["team"]), {})[stat] = round(med, 2)
    return out


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def fetch() -> dict:
    """{ "Name|TEAM": {stat: median} } for every pre-match NBA game posted."""
    lst = _get(LIST_URL)
    events = [e for blk in lst for e in blk.get("events", [])]
    out = {}
    for e in events:
        try:
            full = _get(EVENT_URL.format(e["link"]))
            for ev in (ev for blk in full for ev in blk.get("events", [])):
                for (nm, tm), stats in parse_event(ev).items():
                    out[f"{nm}|{tm}"] = stats
        except Exception as exc:                               # noqa: BLE001
            log.warning("bovada event %s failed (%s)", e.get("description"), str(exc)[:60])
        time.sleep(0.3)
    return out


def load(day: str) -> dict:
    """Fresh fetch, saved for the day; falls back to the day's saved copy."""
    path = os.path.join(CACHE_DIR, f"{day}.json")
    try:
        props = fetch()
    except Exception as exc:                                   # noqa: BLE001
        log.warning("bovada unreachable (%s)", str(exc)[:80])
        props = {}
    # FanDuel merges into every Bovada line we hold today, fresh or saved,
    # by player name (fd_props.py) - never as a parallel entry.
    try:
        import fd_props
        fd = fd_props.fetch("nba")
        if fd:
            saved = {}
            if os.path.exists(path):
                with open(path) as f:
                    saved = json.load(f).get("players", {})
            saved.update(props)
            props = fd_props.merge(saved, fd, ladders=False)
    except Exception as exc:                                   # noqa: BLE001
        log.warning("fanduel NBA props unavailable (%s)", str(exc)[:80])
    try:
        if props:
            os.makedirs(CACHE_DIR, exist_ok=True)
            old = {}
            if os.path.exists(path):
                with open(path) as f:
                    old = json.load(f).get("players", {})
            old.update(props)           # keep players whose games went off the board
            try:                        # fold any team-less duplicate entries
                import fd_props
                old = fd_props.merge(old, {}, ladders=False)
            except Exception:                                  # noqa: BLE001
                pass
            with open(path, "w") as f:
                json.dump({"fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "players": old}, f, separators=(",", ":"), sort_keys=True)
            log.info("bovada: props for %d players", len(props))
            return old
    except Exception as exc:                                   # noqa: BLE001
        log.warning("bovada unreachable (%s)", str(exc)[:80])
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        log.info("bovada: using saved props from %s", d.get("fetched_at"))
        return d.get("players", {})
    return {}


def resolve(stats: dict) -> dict:
    """Median P, R, A from whatever ladders exist."""
    P, R = stats.get("P"), stats.get("R")
    if P is None and stats.get("PR") is not None and R is not None:
        P = stats["PR"] - R
    if R is None and stats.get("PR") is not None and P is not None:
        R = stats["PR"] - P
    a = []
    if stats.get("A") is not None:
        a.append(stats["A"])
    if stats.get("PRA") is not None and P is not None and R is not None:
        a.append(stats["PRA"] - P - R)
    if stats.get("PA") is not None and P is not None:
        a.append(stats["PA"] - P)
    if stats.get("RA") is not None and R is not None:
        a.append(stats["RA"] - R)
    A = max(0.0, float(np.mean(a))) if a else None
    return {k: v for k, v in (("P", P), ("R", R), ("A", A)) if v is not None}


def per_minute_rates(hist: pd.DataFrame, games: int = 20) -> pd.DataFrame:
    """Per player, over his last `games` played: per-minute P, R, A and the
    rest of the DraftKings score (threes, stocks, turnovers, bonuses)."""
    import nba_fetch as F
    h = hist[(hist["played"] == 1) & (hist["min"] > 0)].copy()
    h["fp"] = F.dk_points(h)
    h["other"] = h["fp"] - h["pts"] - 1.25 * h["reb"] - 1.5 * h["ast"]
    h = h.sort_values("date").groupby("player_id").tail(games)
    g = h.groupby("player_id")[["min", "pts", "reb", "ast", "other"]].sum()
    out = pd.DataFrame({"P": g["pts"] / g["min"], "R": g["reb"] / g["min"],
                        "A": g["ast"] / g["min"], "other": g["other"] / g["min"],
                        "min_avg": g["min"] / h.groupby("player_id").size()})
    return out


def market_dk(stats: dict, rates: pd.Series | None, minutes: float | None) -> tuple[float, float] | None:
    """(market DraftKings median, share of it that came from the market)."""
    got = resolve(stats)
    if "P" not in got:
        return None
    m = minutes if minutes else (float(rates["min_avg"]) if rates is not None else None)
    covered = sum(DK_W[k] * v for k, v in got.items())
    filled = 0.0
    for k in ("R", "A"):
        if k not in got:
            if rates is None or not m:
                return None
            filled += DK_W[k] * float(rates[k]) * m
    other = float(rates["other"]) * m if rates is not None and m else 0.0
    total = covered + filled + other
    return (total, covered / total if total > 0 else 0.0)


def blend(model_cond: float, mkt: tuple[float, float] | None) -> tuple[float, float]:
    """(blended conditional projection, market weight used)."""
    if not mkt or model_cond <= 0:
        return model_cond, 0.0
    total, share = mkt
    w = W_MARKET
    # When the model and the books disagree by a lot, it is nearly always our
    # usage bump for absent teammates running hot - 11/18: LeBron model 55.6
    # vs props 40.3 (actual 34.75), Quickley 50.5 vs 42.4 (38.75). The books
    # price absences well; trust them more the further apart we are.
    gap = abs(total - model_cond) / model_cond
    if gap > 0.10:
        w = W_MARKET + (W_GAP_MAX - W_MARKET) * min(1.0, (gap - 0.10) / 0.15)
    w *= min(1.0, share / 0.85)
    return (1 - w) * model_cond + w * total, w
