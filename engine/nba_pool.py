"""Lineup pool for DraftKings NBA classic: PG SG SF PF C G F UTIL, $50,000.

The shared pool builder assumes one position per player. NBA players carry
several ("PG/SG", "SF/PF/C"), so here every (player, slot he can fill) pair
is its own binary and each slot takes exactly one player. No stacks: in
basketball the correlation that matters is game environment, which the
Vegas-adjusted projections already carry.

Same objective families as the other sports, so the page's ownership and
exposure filters behave identically: projection or ceiling, each discounted
by ownership at several strengths (lambda), with each new lineup forced to
differ from every earlier one by at least MIN_UNIQUE players.
"""
from __future__ import annotations

import logging
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

log = logging.getLogger("nba_pool")

SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
CAP = 50_000
LAMBDAS = (0.0, 0.5, 1.0, 1.5, 2.5)
PER_OBJECTIVE = 12
MIN_UNIQUE = 2
SHORTLIST = 90


def build(players: list[dict], field_size: int = 100_000,
          per_objective: int = PER_OBJECTIVE, time_limit: float = 1.0) -> list[dict]:
    keep = [i for i, p in enumerate(players)
            if (p.get("proj") or 0) > 0 and p.get("salary")
            and str(p.get("status") or "clear") != "out"]
    # Shortlist by projection and by value, so cheap punts survive.
    by_proj = sorted(keep, key=lambda i: -players[i]["proj"])[:SHORTLIST]
    by_val = sorted(keep, key=lambda i: -players[i]["proj"] / players[i]["salary"])[:SHORTLIST]
    keep = sorted(set(by_proj) | set(by_val))
    pl = [players[i] for i in keep]
    n = len(pl)
    if n < 8:
        return []
    pairs = [(j, s) for j, p in enumerate(pl) for s in SLOTS
             if s in (p.get("elig") or [])]
    m = len(pairs)
    proj = np.array([p["proj"] for p in pl], float)
    ceil = np.array([p.get("ceil") or p["proj"] for p in pl], float)
    own = np.array([p.get("own") or 0 for p in pl], float)
    sal = np.array([p["salary"] for p in pl], float)
    games = sorted({str(p.get("game") or p.get("team")) for p in pl})

    # Player-use matrix: y_j = sum over his pairs
    U = np.zeros((n, m))
    for k, (j, s) in enumerate(pairs):
        U[j, k] = 1
    A, lo, hi = [], [], []
    for s in SLOTS:                                  # each slot exactly once
        A.append([1.0 if ps == s else 0.0 for _, ps in pairs]); lo.append(1); hi.append(1)
    for j in range(n):                               # each player at most once
        A.append(U[j]); lo.append(0); hi.append(1)
    A.append(sal @ U); lo.append(0); hi.append(CAP)
    # at least two games, as DraftKings requires
    if len(games) >= 2:
        for g in games:
            row = np.array([1.0 if str(pl[j].get("game") or pl[j].get("team")) == g else 0.0
                            for j, _ in pairs])
            A.append(row); lo.append(0); hi.append(7)
    base = (np.array(A), np.array(lo, float), np.array(hi, float))

    pool, seen = [], []
    for name, value in (("median", proj), ("ceiling", ceil)):
        for lam in LAMBDAS:
            obj_p = value * (1.0 - lam * own)
            c = -(obj_p @ U)
            for _ in range(per_objective):
                A_, lo_, hi_ = base
                if seen:
                    extra = np.array([np.array([1.0 if j in L else 0.0 for j, _ in pairs])
                                      for L in seen])
                    A_ = np.vstack([A_, extra])
                    lo_ = np.concatenate([lo_, np.zeros(len(seen))])
                    hi_ = np.concatenate([hi_, np.full(len(seen), 8 - MIN_UNIQUE)])
                res = milp(c, constraints=LinearConstraint(A_, lo_, hi_),
                           integrality=np.ones(m), bounds=Bounds(0, 1),
                           options={"time_limit": time_limit})
                if res.x is None:
                    break
                x = res.x > 0.5
                pick = [pairs[k] for k in range(m) if x[k]]
                js = {j for j, _ in pick}
                seen.append(js)
                order = sorted(pick, key=lambda t: SLOTS.index(t[1]))
                o = own[[j for j, _ in order]]
                pool.append({
                    "p": [keep[j] for j, _ in order],
                    "s": [s for _, s in order],
                    "proj": round(float(proj[[j for j, _ in order]].sum()), 2),
                    "ceil": round(float(ceil[[j for j, _ in order]].sum()), 2),
                    "sal": int(sal[[j for j, _ in order]].sum()),
                    "own": round(float(100 * o.sum()), 1),
                    "dup": round(float(field_size * math.prod(max(v, 1e-4) for v in o)), 4),
                    "src": f"{name} lam={lam}",
                })
    log.info("nba pool: %d lineups", len(pool))
    return pool


def safe_build(payload: dict) -> list[dict]:
    try:
        return build(payload["players"], int(payload.get("field_size") or 100_000))
    except Exception as exc:                                   # noqa: BLE001
        log.warning("nba lineup pool failed (%s)", exc)
        return []
