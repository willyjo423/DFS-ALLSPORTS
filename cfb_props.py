"""College football player props from FanDuel (public, no login), blended
into the CFB projections. DraftKings college scoring is the NFL's, and
FanDuel's college markets use the NFL formats (passing / rushing / receiving
yards and receptions with alternate ladders, passing TDs, anytime / 2+ / 3+
TD scorer), so the reading and the maths are the NFL board's (NFL-DFS
fd_props.py and nfl_props.py), copied here.

Saved daily under docs/data/props/cfb/<date>.json, as a fallback and as the
record the blend weight is graded against.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from statistics import NormalDist

log = logging.getLogger("cfb_props")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "docs", "data", "props", "cfb")
AK = "FhMFpcPWXMeyZxOx"
BASE = "https://sbapi.nj.sportsbook.fanduel.com/api"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0 Safari/537.36")
TABS = ("passing-props", "receiving-props", "rushing-props", "td-scorer-props")
ONE_SIDED_POW = 1.12
TD_MARKETS = {"ANY_TIME_TOUCHDOWN_SCORER": 1, "TO_SCORE_2+_TOUCHDOWNS": 2, "TO_SCORE_3+_TOUCHDOWNS": 3}
N = NormalDist()
W_MARKET = 0.85          # as the NFL board; regraded once CFB grades accumulate
W_GAP_MAX = 0.95


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
    return (((r.get("winRunnerOdds") or {}).get("americanDisplayOdds") or {}).get("americanOdds"))


def _stat(mt: str) -> str | None:
    if any(w in mt for w in ("MOST_", "PLAYERS_WITH", "DRIVE", "HALF", "QUARTER", "+_RECEIVING", "LONGEST")):
        return None
    if "PASSING_YARDS" in mt:
        return "PY"
    if "PASSING_TOUCHDOWNS" in mt:
        return "PTD"
    if "RUSHING_YARDS" in mt:
        return "RY"
    if "RECEIVING_YARDS" in mt:
        return "RECY"
    if "RECEPTIONS" in mt:
        return "REC"
    if "INTERCEPTION" in mt:
        return "INT"
    return None


def parse_markets(markets: dict) -> dict:
    out: dict = {}

    def add(name, stat, k, p):
        if name and p is not None and 0.003 < p < 0.997:
            out.setdefault(f"{name.strip()}|", {}).setdefault(stat, []).append((float(k), p))

    for m in markets.values():
        mt = str(m.get("marketType") or "")
        runners = m.get("runners") or []
        if mt in TD_MARKETS:
            for r in runners:
                p = _prob(_odds(r))
                add(r.get("runnerName"), "TD", TD_MARKETS[mt], p ** ONE_SIDED_POW if p else None)
            continue
        stat = _stat(mt)
        if not stat:
            continue
        name = str(m.get("marketName") or "").split(" - ")[0].strip()
        sides: dict = {}
        for r in runners:
            rn = str(r.get("runnerName") or "")
            p = _prob(_odds(r))
            mm = re.search(r"(\d+)\+", rn)
            if mm:                                   # alternate ladder: "X 125+ Yards"
                add(name, stat, int(mm.group(1)), p ** ONE_SIDED_POW if p else None)
                continue
            hc = r.get("handicap")
            if not hc:
                continue
            side = "O" if rn.endswith("Over") else "U" if rn.endswith("Under") else None
            if side:
                sides.setdefault(float(hc), {})[side] = p
        for ln, s in sides.items():
            if s.get("O") and s.get("U"):
                add(name, stat, math.floor(ln) + 1, s["O"] / (s["O"] + s["U"]))
    return out




# ------------------------------------------------------------ the maths (NFL)
def poisson_tail(lam: float, k: int) -> float:
    if k <= 0:
        return 1.0
    term, cdf = math.exp(-lam), 0.0
    for i in range(int(k)):
        cdf += term
        term *= lam / (i + 1)
    return max(0.0, 1.0 - cdf)


def count_expected(points) -> float | None:
    """E[X] = sum_k P(X >= k): posted tails where they exist, a fitted
    Poisson elsewhere."""
    pts = [(int(k), p) for k, p in points if 0.003 < p < 0.997]
    if not pts:
        return None

    def logit(x):
        x = min(max(x, 1e-6), 1 - 1e-6)
        return math.log(x / (1 - x))
    best, err_best, lam = 0.5, float("inf"), 0.01
    while lam < 40:
        err = sum((logit(poisson_tail(lam, k)) - logit(p)) ** 2 for k, p in pts)
        if err < err_best:
            best, err_best = lam, err
        lam *= 1.02
    obs: dict = {}
    for k, p in pts:
        obs.setdefault(k, []).append(p)
    e, prev = 0.0, 1.0
    for k in range(1, max(max(obs), int(best * 3) + 6) + 1):
        p = sum(obs[k]) / len(obs[k]) if k in obs else poisson_tail(best, k)
        p = min(p, prev)
        e += p
        prev = p
    return e


DEFAULT_SD = {"PY": lambda m: 0.22 * m + 25, "RY": lambda m: 0.50 * m + 8,
              "RECY": lambda m: 0.55 * m + 8}


def yards_fit(stat: str, points) -> tuple[float, float] | None:
    """(mean, sd) of a normal fitted to P(X >= k) at each posted yardage."""
    pts = [(k, N.inv_cdf(1 - p)) for k, p in points if 0.01 < p < 0.99]
    if not pts:
        return None
    if len(pts) >= 2:
        xs = [k for k, _ in pts]
        zs = [z for _, z in pts]
        mx, mz = sum(xs) / len(xs), sum(zs) / len(zs)
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx > 0:
            b = sum((x - mx) * (z - mz) for x, z in zip(xs, zs)) / sxx
            if b > 0:
                a = mz - b * mx
                mu, sd = -a / b, 1 / b
                lo, hi = min(xs), max(xs)
                return max(0.0, min(max(mu, lo - 60), hi + 60)), max(sd, 3.0)
    k, z = min(pts, key=lambda t: abs(t[1]))
    mu = k
    for _ in range(4):                  # z = (k - mu) / sd(mu)
        mu = k - z * DEFAULT_SD[stat](max(mu, 1))
    return max(0.0, mu), DEFAULT_SD[stat](max(mu, 1))




def market_dk(stats: dict, pos: str) -> dict | None:
    """Expected DraftKings points from a player's props, and how complete the
    picture was (`share`, 0-1)."""
    pos = (pos or "").upper()
    py = yards_fit("PY", stats["PY"]) if "PY" in stats else None
    ry = yards_fit("RY", stats["RY"]) if "RY" in stats else None
    rcy = yards_fit("RECY", stats["RECY"]) if "RECY" in stats else None
    rec = count_expected(stats["REC"]) if "REC" in stats else None
    ptd = count_expected(stats["PTD"]) if "PTD" in stats else None
    it = count_expected(stats["INT"]) if "INT" in stats else None
    td = count_expected(stats["TD"]) if "TD" in stats else None

    def over(fit, line):
        return 0.0 if not fit else 1 - N.cdf((line - fit[0]) / fit[1])

    if pos == "QB":
        if py is None or ptd is None:
            return None
        need = [ry, it, td]
    elif pos == "RB":
        if ry is None or td is None:
            return None
        need = [rec, rcy]
    elif pos in ("WR", "TE"):
        if rcy is None or rec is None or td is None:
            return None
        need = [ry]                     # rarely posted; usually ~0 anyway
        need = [] if pos == "TE" else need
    else:
        return None
    dk = 0.0
    if py:
        dk += 0.04 * py[0] + 3 * over(py, 300)
    if ptd is not None:
        dk += 4 * ptd
    if it is not None:
        dk -= it
    elif pos == "QB":
        dk -= 0.7                       # league-ish INTs per start
    if ry:
        dk += 0.1 * ry[0] + 3 * over(ry, 100)
    if rcy:
        dk += 0.1 * rcy[0] + 3 * over(rcy, 100)
    if rec is not None:
        dk += rec
    if td is not None:
        dk += 6 * td
    missing = sum(1 for x in need if x is None)
    share = 1.0 if pos in ("WR", "TE") else max(0.5, 1.0 - 0.2 * missing)
    return {"dk": round(dk, 2), "share": share}




def fetch() -> dict:
    js = _get("content-managed-page", page="CUSTOM", customPageId="ncaaf")
    games = [e for e in ((js.get("attachments") or {}).get("events") or {}).values()
             if " @ " in str(e.get("name")) and "(" not in str(e.get("name"))]
    out: dict = {}
    for e in games:
        markets: dict = {}
        for tab in TABS:
            try:
                ev = _get("event-page", eventId=e["eventId"], tab=tab)
                markets.update((ev.get("attachments") or {}).get("markets") or {})
            except Exception as exc:                           # noqa: BLE001
                log.debug("fanduel %s %s: %s", e.get("name"), tab, str(exc)[:60])
            time.sleep(0.2)
        for key, st in parse_markets(markets).items():
            cur = out.setdefault(key, {})
            for stat, pts in st.items():
                cur.setdefault(stat, []).extend(pts)
    log.info("fanduel: college props for %d players from %d games", len(out), len(games))
    return out


def load(day: str | None = None) -> dict:
    day = day or datetime.now(timezone(timedelta(hours=-5))).date().isoformat()
    path = os.path.join(CACHE_DIR, f"{day}.json")
    old: dict = {}
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f).get("players", {})
    try:
        new = fetch()
    except Exception as exc:                                   # noqa: BLE001
        log.warning("fanduel college props unavailable (%s)", str(exc)[:80])
        new = {}
    if new:
        old.update(new)
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "players": old}, f, separators=(",", ":"), sort_keys=True)
    return old


def _weight(model: float, market: float) -> float:
    if model <= 0:
        return W_MARKET
    gap = abs(market - model) / model
    if gap <= 0.10:
        return W_MARKET
    return W_MARKET + (W_GAP_MAX - W_MARKET) * min(1.0, (gap - 0.10) / 0.15)


def apply(payload: dict, props: dict | None = None) -> int:
    """Blend props into a CFB board's players, in place (after calibration).
    The quantile curve, median, ceiling and projection scale together; the
    model's own number and the props number are kept in `model` / `props`."""
    import props_match as PM
    props = load() if props is None else props
    if not props:
        return 0
    index = PM.build(props)
    n, gaps = 0, []
    for p in payload.get("players", []):
        if p.get("status") == "out":
            continue
        st = PM.match(index, p["name"], p.get("team", ""))
        if not st:
            continue
        mk = market_dk(st, p.get("pos", ""))
        if not mk:
            continue
        model = float(p.get("proj") or p.get("med") or 0)
        if model <= 0:
            continue
        w = _weight(model, mk["dk"]) * min(1.0, mk["share"])
        target = (1 - w) * model + w * mk["dk"]
        f = target / model
        for k in ("proj", "med", "mean", "ceil"):
            if p.get(k) is not None:
                p[k] = round(float(p[k]) * f, 2)
        if p.get("q"):
            p["q"] = [round(float(v) * f, 3) for v in p["q"]]
        p["model"], p["props"] = round(model, 2), round(mk["dk"], 2)
        try:                                     # ceiling from the ladders
            from engine import ladder_ceiling as LC
            lc = LC.football(st, p.get("pos", ""), yards_fit, count_expected)
            qs = payload.get("quantiles") or [0.1, 0.25, 0.5, 0.75, 0.9, 0.97]
            if lc and lc["mean"] > 0 and p.get("q"):
                mkt97 = lc["q97"] * target / lc["mean"]
                p["ceil_model"], p["ceil_props"] = round(float(p["q"][-1]), 2), round(mkt97, 2)
                LC.stretch_to(p, qs, LC.blend(float(p["q"][-1]), mkt97, "cfb"))
        except Exception as exc:                               # noqa: BLE001
            log.debug("ladder ceiling failed for %s (%s)", p.get("name"), exc)
        gaps.append(mk["dk"] - model)
        n += 1
    if n:
        log.info("props: blended %d college players, props - model mean %+.2f, mean |gap| %.2f",
                 n, sum(gaps) / n, sum(abs(g) for g in gaps) / n)
    return n
