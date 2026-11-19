"""Contest simulation: how often each pool lineup finishes in the top 1%,
top 10% and cash line of a field built the way the field actually builds.

A projection ranks players; a tournament pays lineups that beat everyone
else on the night. So:

1. Simulate the slate S times. Each player's score is his conditional
   projection with his own spread, plus a shared GAME shock (a fast,
   high-scoring game lifts everyone in it) and a TEAM shock, and a zero
   when he does not play (chance 1 - p_play).
2. Build a field of F lineups the way the public does: slot by slot,
   drawn in proportion to projected ownership, legal positions, spending
   close to the cap.
3. For every simulated night, find the field's top-1% / top-10% / cash
   scores, and count how often each of our lineups clears them.

The rates are relative, not promises: they rank our lineups against each
other on the thing a tournament pays for.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("nba_sim")

SLOTS_ORDER = ["C", "PF", "SF", "SG", "PG", "F", "G", "UTIL"]   # scarcest first
# Measured on ~100,000 player pairs (2026-01 to 2026-11), relative to the
# model's Vegas-adjusted projections: teammates -0.016, opponents -0.004.
# Once the projection has the game total in it, NBA players do not boom
# together - so no shared shocks. (Game stacks that won a night were luck.)
GAME_SD = 0.0
TEAM_SD = 0.0


def simulate_scores(players: list[dict], sims: int, rng) -> np.ndarray:
    n = len(players)
    med = np.array([p.get("med") or 0.0 for p in players])
    ceil = np.array([p.get("ceil") or 0.0 for p in players])
    sd = np.maximum((ceil - med) / 1.28, 0.5)
    pp = np.array([p.get("pp", 1.0) if p.get("status") != "out" else 0.0 for p in players])
    games = sorted({str(p.get("game")) for p in players})
    teams = sorted({str(p.get("team")) for p in players})
    gi = np.array([games.index(str(p.get("game"))) for p in players])
    ti = np.array([teams.index(str(p.get("team"))) for p in players])
    gs = rng.normal(0, GAME_SD, (sims, len(games)))
    ts = rng.normal(0, TEAM_SD, (sims, len(teams)))
    z = rng.normal(0, 1, (sims, n))
    score = med * (1 + gs[:, gi] + ts[:, ti]) + sd * z
    plays = rng.random((sims, n)) < pp
    return np.where(plays, np.maximum(score, 0), 0.0)


FIELD_STRENGTH = 0.90   # field lineups project at least this share of the best
# (projection floor as a share of the best, value chase): sharp half, casual half.
# Without the value chase a 300-player board almost never yields a sharp
# lineup by chance - the first live run built 8 of 1,250.
FIELD_MIX = ((0.80, 1.0), (0.65, 0.0))   # floors loosened 2026-11-19 so the
# field can match the ownership estimates (it could not at 0.93 / value^4)
FIELD_CALIBRATION_ROUNDS = 4


def build_field(players: list[dict], size: int, rng, cap: int = 50_000,
                min_spend: int = 47_000, tries: int = 40,
                best_proj: float | None = None, value_power: float = 0.0,
                adjust: np.ndarray | None = None) -> np.ndarray:
    own = np.array([max(p.get("own") or 0.0, 0.0) for p in players])
    if adjust is not None:
        own = own * adjust
    sal = np.array([p.get("salary") or 0 for p in players])
    if value_power:
        # Optimizer users chase points per dollar as well as the chalk.
        pr = np.array([p.get("proj") or 0.0 for p in players])
        own = own * (pr / np.maximum(sal, 1) * 200) ** value_power
    elig = [set(p.get("elig") or []) for p in players]
    ok = np.array([p.get("status") != "out" and (p.get("own") or 0) > 0.002 for p in players])
    by_slot = {s: np.array([i for i in range(len(players)) if ok[i] and s in elig[i]])
               for s in SLOTS_ORDER}
    proj = np.array([p.get("proj") or 0.0 for p in players])
    floor = FIELD_STRENGTH * best_proj if best_proj else 0.0
    out = []
    attempts = 0
    while len(out) < size and attempts < size * tries:
        attempts += 1
        chosen, spent = [], 0
        for k, s in enumerate(SLOTS_ORDER):
            cand = by_slot[s]
            cand = cand[~np.isin(cand, chosen)]
            left = 7 - k
            # leave room for the cheapest possible remaining picks
            cand = cand[spent + sal[cand] + left * 3000 <= cap]
            if not len(cand):
                break
            w = own[cand] + 1e-6
            j = int(rng.choice(cand, p=w / w.sum()))
            chosen.append(j)
            spent += sal[j]
        if (len(chosen) == 8 and min_spend <= spent <= cap
                and proj[chosen].sum() >= floor):
            out.append(chosen)
    return np.array(out, dtype=int)


def rank_pool(payload: dict, sims: int = 1500, field: int = 2500, seed: int = 7) -> None:
    """Adds top1 / top10 / cash (percent of simulated nights) to every pool
    lineup, in place."""
    pool = payload.get("lineups") or []
    players = payload.get("players") or []
    if not pool or not players:
        return
    rng = np.random.default_rng(seed)
    S = simulate_scores(players, sims, rng)
    # A real field is a mix: optimizer users close to the best projection,
    # and casual entries well below it. Half each.
    best = max(L["proj"] for L in pool)
    global FIELD_STRENGTH

    def make(size, adjust):
        global FIELD_STRENGTH
        parts = []
        for strength, vp in FIELD_MIX:
            FIELD_STRENGTH = strength
            parts.append(build_field(players, size // len(FIELD_MIX), rng, best_proj=best,
                                     tries=60, value_power=vp, adjust=adjust))
        parts = [x for x in parts if len(x)]
        return np.vstack(parts) if parts else np.zeros((0, 8), int)

    # The field has to roster players about as often as our ownership says
    # it will. Built raw - value chase plus a projection floor - it did not:
    # on 2026-11-18 it played Quickley in 79% of lineups against a 42%
    # estimate, LeBron 61% vs 30%, which made every fade look far better than
    # it was. The sampling weights are corrected until the two agree.
    target = np.array([max(p.get("own") or 0.0, 0.0) for p in players])
    adjust = np.ones(len(players))
    for _ in range(FIELD_CALIBRATION_ROUNDS):
        Fd = make(1000, adjust)
        if len(Fd) < 200:
            break
        freq = np.bincount(Fd.ravel(), minlength=len(players)) / len(Fd)
        adjust *= np.clip((target + 0.003) / (freq + 0.003), 0.25, 4.0) ** 0.8
    Fd = make(field, adjust)
    if len(Fd):
        freq = np.bincount(Fd.ravel(), minlength=len(players)) / len(Fd)
        top = np.argsort(-target)[:10]
        log.info("sim field ownership vs estimate (top 10): mean |gap| %.1f pts",
                 float(np.abs(freq[top] - target[top]).mean() * 100))
    if len(Fd) < 200:
        log.warning("sim: only %d field lineups could be built; skipped", len(Fd))
        return
    q99 = np.empty(sims); q90 = np.empty(sims); q50 = np.empty(sims)
    for a in range(0, sims, 250):
        fs = S[a:a + 250][:, Fd].sum(axis=2)
        q99[a:a + 250] = np.quantile(fs, 0.99, axis=1)
        q90[a:a + 250] = np.quantile(fs, 0.90, axis=1)
        q50[a:a + 250] = np.quantile(fs, 0.55, axis=1)     # double-up cash line
    P = np.array([L["p"] for L in pool])
    ls = S[:, P].sum(axis=2)
    for k, L in enumerate(pool):
        L["top1"] = round(float((ls[:, k] > q99).mean() * 100), 2)
        L["top10"] = round(float((ls[:, k] > q90).mean() * 100), 1)
        L["cash"] = round(float((ls[:, k] > q50).mean() * 100), 1)
    payload["sim"] = {"sims": sims, "field": int(len(Fd))}
    best = max(pool, key=lambda L: L["top1"])
    log.info("sim: %d nights x %d field lineups; best top-1%% rate %.2f%% (%s)",
             sims, len(Fd), best["top1"], best["src"])
