"""NHL player props from Bovada, blended into the skater projections.

Bovada posts, per skater: shots-on-goal over/unders and milestones (2+, 3+,
4+, 5+), points milestones (1+, 2+, 3+), assists milestones and, for many
defencemen, blocked-shot lines. Every one of those is a statement about
P(stat >= k). For each stat we fit the Poisson rate that best matches the
de-vigged probabilities, then build DraftKings points from the rates:

    8.5 G + 5 A + 1.5 SOG + 1.3 BLK
    + 3 P(SOG >= 5) + 3 P(BLK >= 3) + 3 P(points >= 3)

Goals are points minus assists; with no assists market they are the usual
share of points by position. A skater needs a points market and either a shots
market or a goal-scorer price to be blended; blocks fall back to the position average.

Goalies: the saves line replaces the model's expected saves (shots against
x save percentage); goals against and the win stay with the model.

Each day's props are saved under docs/data/props/nhl/<date>.json, both as a
fallback and as a record to grade the blend weight against.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import urllib.request
from datetime import datetime, timedelta, timezone

log = logging.getLogger("nhl_props")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "docs", "data", "props", "nhl")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124 Safari/537.36")
LIST_URL = ("https://www.bovada.lv/services/sports/event/coupon/events/A/description/"
            "hockey/nhl?marketFilterId=def&preMatchOnly=true&lang=en")
EVENT_URL = "https://www.bovada.lv/services/sports/event/v2/events/A/description{}?lang=en"

# Same logic as NBA: the books carry most of the weight, more when they
# disagree with us. NBA graded 0.98 on props over 114 players (2026-11-18);
# NHL is set to match and is regraded nightly from docs/data/props/nhl.
import props_weight as _PW
W_MARKET = _PW.weight("nhl", 0.85)   # learned from nightly grades once 10+ nights
W_GAP_MAX = min(0.97, max(W_MARKET + 0.10, 0.95))         # once model and props are 25%+ apart
ONE_SIDED_POW = 1.12     # a milestone posts only the "yes" price: p_true = p^1.12
                         # (power de-vig; shades long shots more, as books do)
SHOOTING_PCT = 0.105     # league goals per shot on goal
GOAL_SHARE = {"D": 0.27, "C": 0.42, "W": 0.42, "F": 0.42}
BLOCKS_AVG = {"D": 1.45, "C": 0.55, "W": 0.50, "F": 0.52}

_WHO = re.compile(r"^(?P<name>.+?)\s*\((?P<team>[A-Z]{2,4})\)\s*$")
_KIND = [("shots on goal", "SOG"), ("points", "PTS"), ("assists", "AST"),
         ("blocked shots", "BLK"), ("blocks", "BLK"), ("saves", "SV"),
         ("goals", "G")]
_CACHE: dict = {}


def _prob(a) -> float | None:
    if a in (None, ""):
        return None
    a = str(a).upper()
    if a == "EVEN":
        return 0.5
    v = float(a)
    return 100 / (v + 100) if v > 0 else -v / (-v + 100)


def _who(text: str):
    text = text.replace("\xa0", " ").strip()
    m = _WHO.match(text)
    return (m["name"].strip(), m["team"]) if m else (text, None)


def _kind(text: str):
    t = text.lower()
    if "period" in t:
        return None
    for key, k in _KIND:
        if key in t:
            return k
    return None


def parse_event(ev: dict) -> dict:
    """{(name, team): {stat: [(k, P(X >= k)), ...]}} for one game."""
    out: dict = {}

    def add(name, team, stat, k, p):
        if p is not None and 0.005 < p < 0.995:
            out.setdefault((name, team), {}).setdefault(stat, []).append((k, p))

    for g in ev.get("displayGroups", []):
        gname = g.get("description", "")
        if "period" in gname.lower() or gname in ("Game Lines", "Alternate Lines", "Game Props"):
            continue
        for m in g.get("markets", []):
            if (m.get("period") or {}).get("description") not in (None, "Game", "Regulation Time", "Live Game"):
                continue
            d = m.get("description", "").replace("\xa0", " ").strip()
            if " - " not in d:
                # "Anytime Goal Scorer": outcomes are the players
                if "anytime goal" in d.lower():
                    for o in m.get("outcomes", []):
                        nm, tm = _who(o.get("description", ""))
                        p = _prob((o.get("price") or {}).get("american"))
                        add(nm, tm, "G", 1, p ** ONE_SIDED_POW if p else None)
                continue
            left, right = d.split(" - ", 1)
            if _kind(left):
                stat, who = _kind(left), right
            elif _kind(right):
                stat, who = _kind(right), left
            else:
                continue
            name, team = _who(who)
            sides: dict = {}
            for o in m.get("outcomes", []):
                pr = o.get("price") or {}
                desc = (o.get("description") or "").strip()
                p = _prob(pr.get("american"))
                mm = re.search(r"(\d+)\+", desc)
                if mm and not desc.lower().startswith(("over", "under")):
                    add(name, team, stat, int(mm.group(1)), p ** ONE_SIDED_POW if p else None)
                    continue
                hc = pr.get("handicap")
                if hc in (None, ""):
                    mh = re.search(r"(\d+(?:\.\d+)?)", desc)
                    hc = mh.group(1) if mh else None
                if hc is None:
                    continue
                side = "O" if desc.lower().startswith("over") or o.get("type") == "O" else "U"
                sides.setdefault(float(hc), {})[side] = p
            for ln, s in sides.items():
                if s.get("O") and s.get("U"):
                    add(name, team, stat, int(math.floor(ln)) + 1, s["O"] / (s["O"] + s["U"]))
    return out


def poisson_tail(lam: float, k: int) -> float:
    """P(X >= k)."""
    if k <= 0:
        return 1.0
    term, cdf = math.exp(-lam), 0.0
    for i in range(k):
        cdf += term
        term *= lam / (i + 1)
    return max(0.0, 1.0 - cdf)


def fit_rate(points: list[tuple[int, float]]) -> float | None:
    """Poisson rate whose tail probabilities best match the posted ones
    (least squares on the logit, over a fine grid)."""
    pts = [(k, p) for k, p in points if 0.005 < p < 0.995]
    if not pts:
        return None

    def logit(x):
        x = min(max(x, 1e-6), 1 - 1e-6)
        return math.log(x / (1 - x))
    best, best_err = None, float("inf")
    lam = 0.01
    while lam < 60:
        err = sum((logit(poisson_tail(lam, k)) - logit(p)) ** 2 for k, p in pts)
        if err < best_err:
            best, best_err = lam, err
        lam *= 1.01
    return best


def expected(points: list[tuple[int, float]]) -> float | None:
    """E[X] = sum over k>=1 of P(X >= k): the posted tails where they exist
    (averaged across duplicate lines), the fitted Poisson elsewhere."""
    lam = fit_rate(points)
    if lam is None:
        return None
    obs: dict = {}
    for k, p in points:
        obs.setdefault(k, []).append(p)
    top = max(max(obs), int(lam * 3) + 6)
    e, prev = 0.0, 1.0
    for k in range(1, top + 1):
        p = sum(obs[k]) / len(obs[k]) if k in obs else poisson_tail(lam, k)
        p = min(p, prev)                      # tails never rise
        e += p
        prev = p
    return e


def market_dk(stats: dict, pos: str) -> dict | None:
    """Expected DraftKings points for a skater from his props, or None."""
    sog, pts = expected(stats.get("SOG", [])), expected(stats.get("PTS", []))
    ast = expected(stats.get("AST", []))
    g_only = expected(stats.get("G", []))
    if pts is None:
        return None
    sog_src = "market"
    if sog is None:
        # Bovada posts shots for about a third of skaters but a goal-scorer
        # price for nearly all. Shots from goals at the league shooting rate.
        if g_only is None:
            return None
        sog, sog_src = g_only / SHOOTING_PCT, "goals"
    if ast is not None:
        ast = min(ast, pts)
        goals = pts - ast
    elif g_only is not None:
        goals = min(g_only, pts)
        ast = pts - goals
    else:
        goals = GOAL_SHARE.get(pos, 0.4) * pts
        ast = pts - goals
    blk = expected(stats.get("BLK", []))
    blk_src = "market" if blk is not None else "avg"
    if blk is None:
        blk = BLOCKS_AVG.get(pos, 0.6)
    dk = (8.5 * goals + 5 * ast + 1.5 * sog + 1.3 * blk
          + 3 * poisson_tail(sog, 5) + 3 * poisson_tail(blk, 3) + 3 * poisson_tail(pts, 3))
    return {"dk": round(dk, 2), "sog_src": sog_src, "sog": round(sog, 2), "pts": round(pts, 2),
            "g": round(goals, 2), "a": round(ast, 2), "blk": round(blk, 2), "blk_src": blk_src}


# ------------------------------------------------------------------ fetching
def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def fetch() -> dict:
    """{"Name|TEAM": {stat: [[k, p], ...]}} for every pre-match game."""
    lst = _get(LIST_URL)
    out = {}
    for e in (e for blk in lst for e in blk.get("events", [])):
        if "@" not in e.get("description", "") or "Goals" in e.get("description", ""):
            continue
        try:
            full = _get(EVENT_URL.format(e["link"]))
            for ev in (ev for blk in full for ev in blk.get("events", [])):
                for (nm, tm), st in parse_event(ev).items():
                    out[f"{nm}|{tm or ''}"] = st
        except Exception as exc:                               # noqa: BLE001
            log.warning("bovada event %s failed (%s)", e.get("description"), str(exc)[:60])
        time.sleep(0.3)
    return out


def load(day: str) -> dict:
    if day in _CACHE:
        return _CACHE[day]
    path = os.path.join(CACHE_DIR, f"{day}.json")
    props = {}
    for attempt in range(3):            # Bovada now and then answers empty
        try:
            props = fetch()
        except Exception as exc:                               # noqa: BLE001
            log.warning("bovada unreachable (%s)", str(exc)[:80])
        if props:
            break
        time.sleep(5 * (attempt + 1))
    if not props:
        log.warning("bovada: no NHL props returned")
    old = {}
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f).get("players", {})
    old.update(props or {})
    # FanDuel merges into every Bovada line we hold today, fresh or saved,
    # by player name (fd_props.py) - never as a parallel entry.
    try:
        import fd_props
        fd = fd_props.fetch("nhl")
        if fd:
            old = fd_props.merge(old, fd, ladders=True)
            props = props or fd
    except Exception as exc:                                   # noqa: BLE001
        log.warning("fanduel NHL props unavailable (%s)", str(exc)[:80])
    if props:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "players": old}, f, separators=(",", ":"), sort_keys=True)
        log.info("bovada: NHL props for %d players", len(props))
    elif old:
        log.info("bovada: using saved NHL props (%d players)", len(old))
    _CACHE[day] = old
    return old


# ------------------------------------------------------------------ blending
def weight(model: float, market: float) -> float:
    if model <= 0:
        return W_MARKET
    gap = abs(market - model) / model
    if gap <= 0.10:
        return W_MARKET
    return W_MARKET + (W_GAP_MAX - W_MARKET) * min(1.0, (gap - 0.10) / 0.15)


def _apply_goalie(p: dict, st: dict) -> int:
    """Saves line against the model's expected saves. Only the saves part of
    a goalie's score moves (0.7 a save, and the 3-point bonus at 35); goals
    against and the win stay with the model, which builds them from the
    Vegas team totals already."""
    sv_mkt = expected(st.get("SV", []))
    sv_mod = p.get("sv_exp")
    if sv_mkt is None or not sv_mod:
        return 0
    sv_mod = float(sv_mod)
    model = float(p.get("proj") or 0)
    if model <= 0:
        return 0
    pp = min(float(p.get("pp") or 1.0), 1.0)
    delta = (0.7 * (sv_mkt - sv_mod)
             + 3 * (poisson_tail(sv_mkt, 35) - poisson_tail(sv_mod, 35)))
    w = weight(model, model + delta * pp)
    target = max(0.5, model + w * delta * pp)
    f = target / model
    p["proj"] = round(target, 2)
    for key in ("mean", "med", "ceil"):
        if p.get(key) is not None:
            p[key] = round(float(p[key]) * f, 2)
    if p.get("q"):
        p["q"] = [round(float(v) * f, 3) for v in p["q"]]
    p["props"] = round(model + delta * pp, 2)
    p["model"] = round(model, 2)
    p["sv_mkt"] = round(sv_mkt, 1)
    if abs(sv_mkt - sv_mod) >= 2:
        note = f"saves line {sv_mkt:.1f} vs model {sv_mod:.1f}"
        p["news"] = (p["news"] + "; " + note) if p.get("news") else note
    return 1



def apply(payload: dict) -> int:
    """Blend props into skater projections, after calibration. Returns the
    number of skaters blended."""
    if payload.get("sport") != "nhl":
        return 0
    from availability import name_keys
    day = None
    if payload.get("locks"):
        try:
            day = (datetime.fromisoformat(payload["locks"])
                   .astimezone(timezone(timedelta(hours=-5))).date().isoformat())
        except ValueError:
            pass
    day = day or datetime.now(timezone(timedelta(hours=-5))).date().isoformat()
    props = load(day)
    if not props:
        log.info("props: none available, projections unchanged")
        return 0
    import props_match as PM
    index = PM.build(props)
    n, gaps = 0, []
    ng = 0
    for p in payload.get("players", []):
        if p.get("status") == "out":
            continue
        st = PM.match(index, p["name"], p["team"])
        if not st:
            continue
        if p.get("pos") == "G":
            ng += _apply_goalie(p, st)
            continue
        mk = market_dk(st, p.get("pos", "F"))
        if not mk:
            continue
        model = float(p.get("proj") or 0)
        if model <= 0:
            continue
        pp = min(float(p.get("pp") or 1.0), 1.0)
        w = weight(model, mk["dk"] * pp)
        target = (1 - w) * model + w * mk["dk"] * pp
        f = target / model
        p["proj"] = round(target, 2)
        for key in ("mean", "med", "ceil"):
            if p.get(key) is not None:
                p[key] = round(float(p[key]) * f, 2)
        if p.get("q"):
            p["q"] = [round(float(v) * f, 3) for v in p["q"]]
        p["props"] = mk["dk"]
        p["model"] = round(model, 2)
        # The top of his range from the ladders too (engine/ladder_ceiling).
        try:
            from engine import ladder_ceiling as LC
            lc = LC.nhl_skater(st, p.get("pos", "F"), fit_rate,
                               BLOCKS_AVG.get(p.get("pos", "F"), 0.6),
                               GOAL_SHARE.get(p.get("pos", "F"), 0.42))
            qs = payload.get("quantiles") or [0.1, 0.25, 0.5, 0.75, 0.9, 0.97]
            if lc and p.get("q"):
                mkt97 = lc["q97"] * (target / lc["mean"]) if lc["mean"] > 0 else lc["q97"]
                p["ceil_props"] = round(mkt97, 2)
                p["ceil_model"] = round(float(p["q"][-1]), 2)
                LC.stretch_to(p, qs, LC.blend(float(p["q"][-1]), mkt97, "nhl"))
        except Exception as exc:                               # noqa: BLE001
            log.debug("ladder ceiling failed for %s (%s)", p.get("name"), exc)
        gaps.append(mk["dk"] * pp - model)
        if abs(mk["dk"] * pp - model) >= max(2.0, 0.2 * model):
            note = f"props {mk['dk']:.1f} vs model {model:.1f}"
            p["news"] = (p["news"] + "; " + note) if p.get("news") else note
        n += 1
    log.info("props: %d skaters matched with props, %d goalies with saves", n, ng)
    if n:
        payload["props"] = {"source": "Bovada player props", "players": n, "weight": W_MARKET}
        log.info("props: blended %d skaters, props - model mean %+.2f, mean |gap| %.2f",
                 n, sum(gaps) / n, sum(abs(g) for g in gaps) / n)
    return n
