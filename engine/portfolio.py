"""Choose several lineups as a SET, not one at a time.

Ranking each lineup by its own top-1% rate and taking the first 13 gives 13
versions of the same bet: on 2026-11-20 Ajay Mitchell was in 95 of 120 NBA
lineups and his bust sank nearly all of them. In a big tournament only your
best lineup matters each night, so the set worth entering is the one whose
BEST lineup does well across the most simulated nights.

Greedy selection on that objective: per simulated night each lineup earns a
payout proxy (WIN for beating the field's 99.9th percentile, TOP for its
99th, else 0); the set's value on a night is the max over its lineups; add
whichever lineup raises the average of that max the most. Greedy is within
1 - 1/e of the best possible set for this kind of objective and in practice
almost exactly optimal.
"""
from __future__ import annotations

import numpy as np

WIN, TOP = 10.0, 1.0
PICK = 40


def payoff(ls: np.ndarray, q999: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """sims x lineups payout proxy, as float32 to keep memory low."""
    return (np.where(ls > q999[:, None], WIN, 0.0)
            + np.where(ls > q99[:, None], TOP, 0.0)).astype(np.float32)


def greedy(V: np.ndarray, n: int = PICK) -> list[int]:
    """Indices of lineups in pick order."""
    sims, k = V.shape
    n = min(n, k)
    best = np.zeros(sims, np.float32)
    chosen: list[int] = []
    avail = np.ones(k, bool)
    for _ in range(n):
        gain = np.maximum(V, best[:, None]).mean(axis=0) - best.mean()
        gain[~avail] = -1.0
        j = int(np.argmax(gain))
        if gain[j] <= 0:                 # nothing adds anything: fill by own value
            rest = np.where(avail)[0]
            j = int(rest[np.argmax(V[:, rest].mean(axis=0))])
        chosen.append(j)
        avail[j] = False
        best = np.maximum(best, V[:, j])
    return chosen


def assign(pool: list[dict], ls: np.ndarray, q999: np.ndarray, q99: np.ndarray,
           n: int | None = None) -> list[int]:
    """Write L["port"] = 1..n (pick order) on the chosen lineups."""
    order = greedy(payoff(ls, q999, q99), n or PICK)
    for L in pool:
        L.pop("port", None)
    for r, j in enumerate(order, 1):
        pool[j]["port"] = r
    return order
