"""Per price tier, how wide the top of each player's simulated range should be.

tail_check.py (2026-11-22) placed every graded player's real score inside his
own simulated distribution. NHL depth skaters beat their simulated 90th / 97th
percentiles 12-15% / 4.5% of the time instead of 10% / 3% (12,900 skater
games); the most expensive quarter was right. A factor k per tier stretches
the part of each curve above the median: q -> med + k * (q - med).

Fitted by `python tail_check.py --fit` with the scale switched off, stored in
data/tail_scale.json; tiers are the salary quarter within the board, as in
tail_check. A tier keeps k = 1 unless its miss was statistically clear.
Set TAIL_SCALE_OFF=1 to simulate without it (the fit does).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parent.parent / "data" / "tail_scale.json"
TIERS = ("punt", "low", "mid", "high")
_CACHE = None


def table() -> dict:
    global _CACHE
    if _CACHE is None:
        try:
            _CACHE = json.loads(PATH.read_text())
        except Exception:                                      # noqa: BLE001
            _CACHE = {}
    return _CACHE


def tiers(players: list[dict]) -> list[str]:
    import pandas as pd
    sal = pd.Series([float(p.get("salary") or 0) for p in players])
    rk = sal.rank(pct=True, method="average").to_numpy()
    return [TIERS[min(int(r * 4 - 1e-9), 3)] for r in rk]


def factors(sport: str, players: list[dict]) -> np.ndarray:
    k = (table().get(sport) or {}).get("k") or {}
    if os.environ.get("TAIL_SCALE_OFF") or not k:
        return np.ones(len(players))
    return np.array([float(k.get(t, 1.0)) for t in tiers(players)])


def stretch(qs, q, k: float):
    """Stretch the curve above its median by k."""
    if k == 1.0 or not q:
        return q
    med = float(np.interp(0.5, qs, q))
    return [v if lv <= 0.5 else max(med + k * (v - med), 0.0) for lv, v in zip(qs, q)]
