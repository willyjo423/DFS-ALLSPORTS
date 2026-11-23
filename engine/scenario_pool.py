"""Scenario lineups: the best lineup in each of N simulated versions of the slate.

Why (study_optimal.py, 2026-11-22): on most nights the hindsight-best lineup
was mostly made of players already somewhere in our pool, but the pool never
put them together. NBA: the pool's players could make 95% of the optimal
score, the pool's best lineup made 82%. CFB: 90% vs 63%. The projection and
ceiling families all chase one number per player, so they keep rebuilding
the same core and the combinations that win a night (two cheap players
booming together, a game shooting out) are never built.

So: draw N slates from the sim (with its correlations - stacks, games, the
'model was right about this team' scenario), solve the legal optimum of each
with the sport's own pool builder and roster rules, and add the distinct
results to the pool. The normal top-1% / ROI ranking then decides whether
any of them is worth playing.
"""
from __future__ import annotations

import copy
import logging
import math
import time

import numpy as np

log = logging.getLogger("scenario_pool")
N_SCENARIOS = 200     # simulated slates solved per board (time budget permitting)
KEEP = 60             # scenario lineups kept after the contest sim ranks them
# Sports where scenario lineups only appear on request (the SCENARIO-only
# box): 189 historical NBA nights, they lowered tournament ROI under the
# ROI and top-1% sorts without adding top-1% finishes (2026-11-23).
OPT_IN = {"nba"}
# Sports kept on the first version (60 slates, every optimum kept): NHL's
# 54-night backtest went from 37% to 22% 20-entry top-1% nights under v2.
V1_SPORTS = {"nhl"}
MODE = "v2"           # "v1" = 60 slates, every distinct optimum kept (2026-11-22)
MAX_SECONDS = 150     # per slate, so a publish run never times out on them


def draws(sport: str, payload: dict, n: int, rng) -> np.ndarray:
    if sport == "nba":
        from engine import nba_sim as SIM
        return SIM.simulate_scores(payload["players"], n, rng)
    SIM = __import__(f"engine.{sport}_sim", fromlist=["x"])
    return SIM.simulate_scores(payload, n, rng)


def add(payload: dict, sport: str, n: int = N_SCENARIOS, seed: int = 5,
        max_seconds: float | None = None) -> int:
    pool = payload.get("lineups")
    if not pool:
        return 0
    players, roster = payload["players"], payload["roster"]
    if MODE == "v1" or sport in V1_SPORTS:
        n = 60
    rng = np.random.default_rng(seed)
    S = draws(sport, payload, n, rng)
    have = {tuple(sorted(L["p"])) for L in pool}
    freq: dict = {}
    pcount = np.zeros(len(players))
    solved = 0
    proj = np.array([float(p.get("proj") or p.get("med") or 0) for p in players])
    ceil = np.array([float(p.get("ceil") or p.get("proj") or 0) * min(float(p.get("pp") or 1.0), 1.0)
                     for p in players])
    own = np.clip(np.array([float(p.get("own") or 0) for p in players]), 0, 1)
    size = len(roster["slots"])
    fs = int(payload.get("field_size") or 100_000)
    added = 0
    t0 = time.time()
    for k in range(len(S)):
        if max_seconds and time.time() - t0 > max_seconds:
            break
        pl = copy.deepcopy(players)
        for i, p in enumerate(pl):
            v = max(float(S[k, i]), 0.0)
            p["med"] = p["proj"] = p["mean"] = p["ceil"] = v if v > 0 else 0.0
        try:
            if sport == "nba":
                from engine import nba_pool as NP
                got = NP.build(pl, fs, per_objective=1, lambdas=(0.0,), bases=("median",))
            else:
                from engine import lineup_pool as LP
                got = LP.build(pl, roster, fs, lambdas=(0.0,), per_objective=1, stacks=False)
        except Exception as exc:                               # noqa: BLE001
            log.debug("scenario %d failed: %s", k, exc)
            continue
        L = next((x for x in got if len(x["p"]) == size and len(set(x["p"])) == size), None)
        if not L:
            continue
        key = tuple(sorted(L["p"]))
        solved += 1
        pcount[list(L["p"])] += 1
        freq[key] = freq.get(key, 0) + 1
        if key in have:
            for x in pool:
                if x.get("src") == "scenario" and tuple(sorted(x["p"])) == key:
                    x["freq"] = freq[key]
            continue
        have.add(key)
        idx = list(L["p"])
        o = own[idx]
        new = {"p": idx, "proj": round(float(proj[idx].sum()), 2),
               "ceil": round(float(ceil[idx].sum()), 2),
               "sal": int(sum(players[i]["salary"] for i in idx)),
               "own": round(float(100 * o.sum()), 1),
               "dup": round(float(fs * math.prod(max(x, 1e-4) for x in o)), 4),
               "src": "scenario", "freq": 1}
        if sport in OPT_IN:
            new["optin"] = True
        if "s" in L:
            new["s"] = L["s"]
        pool.append(new)
        added += 1
    # each player's share of the simulated slates' best lineups ("optimal %")
    if solved:
        for i, p in enumerate(players):
            p["sopt"] = round(float(100 * pcount[i] / solved), 1)
        payload["scenarios"] = {"solved": solved}
    log.info("scenario lineups: %d distinct from %d solved slates", added, solved)
    return added


def trim(payload: dict, keep: int = KEEP) -> int:
    """After the contest sim: keep the scenario lineups that rank in the top
    `keep` by large-tournament ROI or by top-1% rate, that the portfolio
    picked, or that were the best lineup in 2+ simulated slates."""
    if MODE == "v1" or payload.get("sport") in V1_SPORTS:
        return 0
    pool = payload.get("lineups") or []
    sc = [L for L in pool if L.get("src") == "scenario"]
    if len(sc) <= keep:
        return 0
    roi = lambda L: ((L.get("roit") or {}).get("large", L.get("roi")) or -1e9)
    ok = {id(L) for L in sorted(sc, key=lambda L: -roi(L))[:keep]}
    ok |= {id(L) for L in sorted(sc, key=lambda L: -(L.get("top1") or 0))[:keep]}
    ok |= {id(L) for L in sc if "port" in L or (L.get("freq") or 1) >= 2}
    before = len(pool)
    payload["lineups"] = [L for L in pool if L.get("src") != "scenario" or id(L) in ok]
    return before - len(payload["lineups"])


def safe_add(payload: dict, sport: str, log_=None) -> None:
    """For the publishers: scenario lineups, never at the cost of the slate."""
    try:
        k = add(payload, sport, N_SCENARIOS, max_seconds=MAX_SECONDS)
        (log_ or log).info("scenario lineups: %d added", k)
    except Exception as exc:                                   # noqa: BLE001
        (log_ or log).warning("scenario lineups failed (%s)", str(exc)[:80])


def safe_trim(payload: dict, log_=None) -> None:
    try:
        k = trim(payload)
        if k:
            (log_ or log).info("scenario lineups: %d trimmed after ranking", k)
    except Exception as exc:                                   # noqa: BLE001
        (log_ or log).warning("scenario trim failed (%s)", str(exc)[:80])
