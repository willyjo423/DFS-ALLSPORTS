"""Contest simulation for DraftKings college football classic
(QB RB RB WR WR WR FLEX SFLEX).

Football's correlations, in the copula's shared shocks:
  * GAME:  a shootout lifts both sides (bring-backs work because of this)
  * TEAM:  an offence that moves the ball feeds everyone on it
  * PASS:  a quarterback and his receivers share the same completions -
           the strongest single correlation in DFS football
Each player's own spread comes from his published quantiles, scaled to the
calibrated projection the pool optimises. Weights are standard football
values, not yet fitted to college results.

The field: half sharp entries that build a QB + 2 stack and chase value,
half casual entries drawn by ownership alone.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.special import ndtr

from engine.nhl_sim import _inverse_cdf

log = logging.getLogger("cfb_sim")

W_GAME, W_TEAM, W_PASS = 0.08, 0.12, 0.25
FIELD_MIX = ((0.92, 3.0, True), (0.80, 0.0, False))


def simulate_scores(payload: dict, sims: int, rng) -> np.ndarray:
    players = payload["players"]
    qs = payload.get("quantiles") or [0.1, 0.25, 0.5, 0.75, 0.9, 0.97]
    teams = sorted({p["team"] for p in players})
    games = sorted({str(p.get("game")) for p in players})
    ti = {t: k for k, t in enumerate(teams)}
    gi = {g: k for k, g in enumerate(games)}
    zt = rng.normal(size=(sims, len(teams)))
    zp = rng.normal(size=(sims, len(teams)))
    zg = rng.normal(size=(sims, len(games)))
    out = np.zeros((sims, len(players)))
    for i, p in enumerate(players):
        if p.get("status") == "out":
            continue
        passing = p["pos"] in ("QB", "WR", "TE")
        wp = W_PASS if passing else 0.0
        z = (np.sqrt(W_GAME) * zg[:, gi[str(p.get("game"))]]
             + np.sqrt(W_TEAM) * zt[:, ti[p["team"]]]
             + np.sqrt(wp) * zp[:, ti[p["team"]]]
             + np.sqrt(1 - W_GAME - W_TEAM - wp) * rng.normal(size=sims))
        u = np.clip(ndtr(z), 1e-4, 1 - 1e-4)
        q = p.get("q") or [p.get("med") or 0.0] * len(qs)
        x = _inverse_cdf(qs, q, u)
        pp = float(p.get("pp") or 1.0)
        target = (p.get("proj") or p.get("med") or 0.0) / (pp if 0 < pp < 0.5 else 1.0)
        m = float(np.mean(x))
        x = x * (target / m) if m > 0 else x * 0
        out[:, i] = np.where(rng.random(sims) < pp, x, 0.0)
    return out


def build_field(payload: dict, size: int, rng, floor: float, value_power: float,
                stack: bool, best: float, tries: int = 60) -> np.ndarray:
    players = payload["players"]
    cap = payload["roster"].get("salary_cap", 50_000)
    n = len(players)
    own = np.array([max(p.get("own") or 0.0, 0.0) for p in players])
    sal = np.array([p.get("salary") or 0 for p in players], float)
    proj = np.array([p.get("proj") or p.get("med") or 0.0 for p in players])
    pos = [p["pos"] for p in players]
    team = [p["team"] for p in players]
    ok = np.array([p.get("status") != "out" and (p.get("own") or 0) > 0.001 for p in players])
    w = own * ((proj / np.maximum(sal, 1) * 1000 / 3.0) ** value_power if value_power else 1)
    idx = {s: np.array([i for i in range(n) if ok[i] and pos[i] == s]) for s in ("QB", "RB", "WR")}
    out, att = [], 0
    while len(out) < size and att < size * tries:
        att += 1
        chosen = []
        need = {"QB": 1, "RB": 2, "WR": 3}
        if not len(idx["QB"]):
            break
        qw = w[idx["QB"]] + 1e-9
        q = int(rng.choice(idx["QB"], p=qw / qw.sum()))
        chosen.append(q)
        need["QB"] = 0
        if stack:
            mates = np.array([i for i in idx["WR"] if team[i] == team[q]])
            if len(mates) >= 2:
                mw = w[mates] + 1e-9
                pick = rng.choice(mates, size=2, replace=False, p=mw / mw.sum())
                chosen += [int(x) for x in pick]
                need["WR"] -= 2
        bad = False
        for s in ("RB", "WR"):
            while need[s] > 0:
                cand = idx[s][~np.isin(idx[s], chosen)]
                if not len(cand):
                    bad = True
                    break
                cw = w[cand] + 1e-9
                chosen.append(int(rng.choice(cand, p=cw / cw.sum())))
                need[s] -= 1
            if bad:
                break
        if bad:
            continue
        # FLEX (RB/WR) then SFLEX (QB/RB/WR)
        for allowed in (("RB", "WR"), ("QB", "RB", "WR")):
            cand = np.concatenate([idx[s] for s in allowed])
            cand = cand[~np.isin(cand, chosen)]
            cw = w[cand] + 1e-9
            chosen.append(int(rng.choice(cand, p=cw / cw.sum())))
        spent = sal[chosen].sum()
        if (len(chosen) == 8 and 0.94 * cap <= spent <= cap
                and len({team[i] for i in chosen}) >= 2
                and proj[chosen].sum() >= floor * best):
            out.append(chosen)
    return np.array(out, dtype=int)


def rank_pool(payload: dict, sims: int = 1500, field: int = 2500, seed: int = 13) -> None:
    pool = payload.get("lineups") or []
    if not pool or not payload.get("players"):
        return
    rng = np.random.default_rng(seed)
    S = simulate_scores(payload, sims, rng)
    best = max(L["proj"] for L in pool)
    parts = [build_field(payload, field // len(FIELD_MIX), rng, fl, vp, st, best)
             for fl, vp, st in FIELD_MIX]
    parts = [x for x in parts if len(x)]
    if not parts or sum(len(x) for x in parts) < 300:
        log.warning("cfb sim: field too small; lineups left unranked")
        return
    Fd = np.vstack(parts)
    q99, q90, q55 = (np.empty(sims) for _ in range(3))
    for a in range(0, sims, 250):
        fs = S[a:a + 250][:, Fd].sum(axis=2)
        q99[a:a + 250] = np.quantile(fs, 0.99, axis=1)
        q90[a:a + 250] = np.quantile(fs, 0.90, axis=1)
        q55[a:a + 250] = np.quantile(fs, 0.55, axis=1)
    ls = S[:, np.array([L["p"] for L in pool])].sum(axis=2)
    for k, L in enumerate(pool):
        L["top1"] = round(float((ls[:, k] > q99).mean() * 100), 2)
        L["top10"] = round(float((ls[:, k] > q90).mean() * 100), 1)
        L["cash"] = round(float((ls[:, k] > q55).mean() * 100), 1)
    payload["sim"] = {"sims": sims, "field": int(len(Fd))}
    top = max(pool, key=lambda L: L["top1"])
    log.info("cfb sim: %d nights x %d field lineups; best top-1%% %.2f%% (%s)",
             sims, len(Fd), top["top1"], top["src"])
