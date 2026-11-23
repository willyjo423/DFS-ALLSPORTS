"""Ceilings from the books' ladders (2+ goals, 5+ shots, 30+ saves, 30+
points, 100+ yards, 2+ TDs ...), blended with the model's own ceiling.

The props blend (nhl_props / nba_props / cfb_props / NFL nfl_props) moves a
player's average to the market's; this moves the TOP of his range. Two
corrections before a ladder becomes a ceiling:
  * long shots: books pad the far rungs more than the near ones, so a
    one-sided price under 25% gets an extra power on top of the de-vig;
  * correlation: a player's stats rise and fall together (shots and goals,
    yards and TDs), so the simulated stats share a common factor.
How much of the market ceiling is used lives in data/ceiling_weight.json
per sport (0.5 to start); grade_nightly moves it toward whichever ceiling
the real scores agree with.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm, poisson

PATH = Path(__file__).resolve().parent.parent / "data" / "ceiling_weight.json"
LONGSHOT_POW = 1.10        # extra, on rungs under 25%
RHO = 0.5                  # shared factor between a player's stats
SIMS = 4000
_RNG = np.random.default_rng(23)


def weight(sport: str) -> float:
    try:
        return float(json.loads(PATH.read_text()).get(sport, {}).get("w", 0.5))
    except Exception:                                          # noqa: BLE001
        return 0.5


def longshot(points):
    return [(k, (p ** LONGSHOT_POW) if p < 0.25 else p) for k, p in points]


def _latent(n_stats: int):
    z0 = _RNG.standard_normal(SIMS)
    return [norm.cdf(math.sqrt(RHO) * z0 + math.sqrt(1 - RHO) * _RNG.standard_normal(SIMS))
            for _ in range(n_stats)]


def nhl_skater(stats: dict, pos: str, fit_rate, blocks_avg: float, goal_share: float) -> dict | None:
    """q90 / q97 of a skater's DraftKings points from his ladders."""
    lam = {k: fit_rate(longshot(stats.get(k, []))) for k in ("SOG", "PTS", "AST", "G")}
    if lam["PTS"] is None:
        return None
    g = lam["G"] if lam["G"] is not None else (
        lam["PTS"] - lam["AST"] if lam["AST"] is not None else goal_share * lam["PTS"])
    g = max(min(g, lam["PTS"]), 0.01)
    a = max(lam["PTS"] - g, 0.01)
    sog = lam["SOG"] if lam["SOG"] is not None else g / 0.105
    blk = fit_rate(stats["BLK"]) if stats.get("BLK") else blocks_avg
    u = _latent(4)
    G, A = poisson.ppf(u[0], g), poisson.ppf(u[1], a)
    S = np.maximum(poisson.ppf(u[2], sog), G)
    B = poisson.ppf(u[3], blk)
    dk = (8.5 * G + 5 * A + 1.5 * S + 1.3 * B + 3 * (G >= 3) + 3 * (S >= 5)
          + 3 * (B >= 3) + 3 * ((G + A) >= 3))
    return {"q90": float(np.quantile(dk, 0.90)), "q97": float(np.quantile(dk, 0.97)),
            "mean": float(dk.mean())}


def football(stats: dict, pos: str, yards_fit, count_expected) -> dict | None:
    """q90 / q97 of a QB / RB / WR / TE's DraftKings points (NFL and college
    scoring) from yards ladders and count ladders."""
    pos = (pos or "").upper()
    fits = {k: yards_fit(k, stats[k]) for k in ("PY", "RY", "RECY") if k in stats}
    cnt = {k: count_expected(longshot(stats[k])) for k in ("REC", "PTD", "TD", "INT") if k in stats}
    if not fits and not cnt:
        return None
    if pos == "QB" and "PY" not in fits:
        return None
    if pos in ("WR", "TE") and "RECY" not in fits:
        return None
    if pos == "RB" and "RY" not in fits:
        return None
    u = _latent(7)
    dk = np.zeros(SIMS)
    for i, (k, f) in enumerate(fits.items()):
        y = np.maximum(f[0] + f[1] * norm.ppf(u[i]), 0)
        if k == "PY":
            dk += 0.04 * y + 3 * (y >= 300)
        else:
            dk += 0.1 * y + 3 * (y >= 100)
    for j, (k, m) in enumerate(cnt.items()):
        if m is None:
            continue
        x = poisson.ppf(u[3 + j], max(m, 0.01))
        dk += {"REC": 1.0, "PTD": 4.0, "TD": 6.0, "INT": -1.0}[k] * x
    return {"q90": float(np.quantile(dk, 0.90)), "q97": float(np.quantile(dk, 0.97)),
            "mean": float(dk.mean())}


def stretch_to(p: dict, qs, target97: float) -> None:
    """Move a player's curve above the median so its 97th percentile is
    target97 (ceil follows)."""
    q = p.get("q")
    if not q or len(q) != len(qs):
        return
    med = float(np.interp(0.5, qs, q))
    top = float(q[-1])
    if top <= med + 1e-6 or target97 <= med:
        return
    k = (target97 - med) / (top - med)
    p["q"] = [round(v if lv <= 0.5 else med + k * (v - med), 3) for lv, v in zip(qs, q)]
    p["ceil"] = round(float(p["q"][-1]), 2)


def blend(model_ceil: float, market_ceil: float, sport: str) -> float:
    w = weight(sport)
    return (1 - w) * model_ceil + w * market_ceil
