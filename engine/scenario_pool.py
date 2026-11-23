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

import numpy as np

log = logging.getLogger("scenario_pool")
N_SCENARIOS = 60


def draws(sport: str, payload: dict, n: int, rng) -> np.ndarray:
    if sport == "nba":
        from engine import nba_sim as SIM
        return SIM.simulate_scores(payload["players"], n, rng)
    SIM = __import__(f"engine.{sport}_sim", fromlist=["x"])
    return SIM.simulate_scores(payload, n, rng)


def add(payload: dict, sport: str, n: int = N_SCENARIOS, seed: int = 5) -> int:
    pool = payload.get("lineups")
    if not pool:
        return 0
    players, roster = payload["players"], payload["roster"]
    rng = np.random.default_rng(seed)
    S = draws(sport, payload, n, rng)
    have = {tuple(sorted(L["p"])) for L in pool}
    proj = np.array([float(p.get("proj") or p.get("med") or 0) for p in players])
    ceil = np.array([float(p.get("ceil") or p.get("proj") or 0) * min(float(p.get("pp") or 1.0), 1.0)
                     for p in players])
    own = np.clip(np.array([float(p.get("own") or 0) for p in players]), 0, 1)
    size = len(roster["slots"])
    fs = int(payload.get("field_size") or 100_000)
    added = 0
    for k in range(len(S)):
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
        if key in have:
            continue
        have.add(key)
        idx = list(L["p"])
        o = own[idx]
        new = {"p": idx, "proj": round(float(proj[idx].sum()), 2),
               "ceil": round(float(ceil[idx].sum()), 2),
               "sal": int(sum(players[i]["salary"] for i in idx)),
               "own": round(float(100 * o.sum()), 1),
               "dup": round(float(fs * math.prod(max(x, 1e-4) for x in o)), 4),
               "src": "scenario"}
        if "s" in L:
            new["s"] = L["s"]
        pool.append(new)
        added += 1
    log.info("scenario lineups: %d of %d scenarios added a new lineup", added, len(S))
    return added
