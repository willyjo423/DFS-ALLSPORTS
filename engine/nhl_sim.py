"""Contest simulation for DraftKings NHL classic, the hockey twin of nba_sim.

Hockey scoring is lumpy and CORRELATED in a way basketball is not: one
goal is a goal and up to two assists for the same line, often the same
power-play unit, and it is a goal against the other side's goalie. So each
simulated night draws, per player, from his own projected distribution
(the published quantiles) through a Gaussian copula with shared shocks:

    skater:  team shock + line shock (+ PP-unit shock for PP1/PP2)
    goalie:  his team's shock, minus the opponent's shock (a big night for
             the other side is a bad night for him)

The shock weights are reasonable, not yet fitted; they can be fitted once
archived boards are graded against finished games.

The field: half "sharp" entries that build a line stack (a full L1/L2 line
drawn by ownership, the field's favourite structure) and chase value; half
casual entries drawn by ownership alone. Both are legal DraftKings rosters
(C C W W W D D G UTIL, $50,000, skaters from 3+ teams).
"""
from __future__ import annotations

import logging
from statistics import NormalDist

import numpy as np

log = logging.getLogger("nhl_sim")

# Fitted on 2025-26/2026-27 MoneyPuck games (relative DK residuals):
#   teammates +0.048, power-play unitmates +0.079, opponents -0.009,
#   goalie vs own skaters +0.075, goalie vs opposing skaters -0.182.
# Latent weights are set a little above those because a zero-heavy score
# shows a weaker correlation than the factor that drives it. Linemates are
# not identifiable in the history; +0.06 on top of team is an assumption.
W_TEAM, W_LINE, W_PP = 0.07, 0.06, 0.03
W_GOALIE_OWN, W_GOALIE_OPP = 0.15, 0.69
FIELD_MIX = ((0.92, 3.0, True), (0.80, 0.0, False))   # (floor, value chase, stack)
MODEL_SCENARIO = 0.15   # per team per night; ~1 - the props weight
_ND = NormalDist()


def _inverse_cdf(qs: list[float], vals: list[float], u: np.ndarray) -> np.ndarray:
    """Piecewise-linear quantile function through the published quantiles,
    with the tails extended to 0 and to 1.6x the top quantile."""
    xs = [0.0] + list(qs) + [1.0]
    ys = [0.0] + list(vals) + [vals[-1] * 1.6 if vals[-1] > 0 else 0.0]
    ys = np.maximum.accumulate(np.maximum(ys, 0.0))
    return np.interp(u, xs, ys)


def simulate_scores(payload: dict, sims: int, rng) -> np.ndarray:
    players = payload["players"]
    qs = payload.get("quantiles") or [0.1, 0.25, 0.5, 0.75, 0.9, 0.97]
    n = len(players)
    teams = sorted({p["team"] for p in players})
    t_idx = {t: k for k, t in enumerate(teams)}
    lines = sorted({p.get("lg") for p in players if p.get("lg")})
    l_idx = {g: k for k, g in enumerate(lines)}
    units = sorted({f"{p['team']}-PP{p['ppu']}" for p in players if p.get("ppu")})
    u_idx = {g: k for k, g in enumerate(units)}
    zt = rng.normal(size=(sims, len(teams)))
    zl = rng.normal(size=(sims, max(len(lines), 1)))
    zu = rng.normal(size=(sims, max(len(units), 1)))
    from engine import tail_scale as TS
    kf = TS.factors("nhl", players)
    out = np.zeros((sims, n))
    # The books or our model, per team (see nba_sim): in MODEL_SCENARIO of
    # simulated nights the model's number, not the props blend, is the truth
    # for a whole team.
    flip = rng.random((sims, len(teams))) < MODEL_SCENARIO
    for i, p in enumerate(players):
        if p.get("status") == "out":
            continue
        e = rng.normal(size=sims)
        if p["pos"] == "G":
            own_t = zt[:, t_idx[p["team"]]]
            opp = t_idx.get(p.get("opp"))
            opp_t = zt[:, opp] if opp is not None else 0.0
            w_own, w_opp = W_GOALIE_OWN, W_GOALIE_OPP
            z = (np.sqrt(w_own) * own_t - np.sqrt(w_opp) * opp_t
                 + np.sqrt(1 - w_own - w_opp) * e)
        else:
            wt, wl, wu = W_TEAM, 0.0, 0.0
            z = np.sqrt(W_TEAM) * zt[:, t_idx[p["team"]]]
            if p.get("lg") in l_idx:
                wl = W_LINE
                z = z + np.sqrt(W_LINE) * zl[:, l_idx[p["lg"]]]
            key = f"{p['team']}-PP{p['ppu']}" if p.get("ppu") else None
            if key in u_idx:
                wu = W_PP
                z = z + np.sqrt(W_PP) * zu[:, u_idx[key]]
            z = z + np.sqrt(max(1 - wt - wl - wu, 0.05)) * e
        from scipy.special import ndtr
        u = np.clip(ndtr(z), 1e-4, 1 - 1e-4)
        q = p.get("q") or [p.get("med") or 0.0] * len(qs)
        q = TS.stretch(qs, q, float(kf[i]))
        x = _inverse_cdf(qs, q, u)
        # Published quantiles are the model's; `proj` is the calibrated
        # number the pool optimises - "if he plays" for anyone likely to
        # play (see projection_calibration). Scale the draw to it, then zero
        # the nights he sits.
        pp = float(p.get("pp") or 1.0)
        target = (p.get("proj") or 0.0) / (pp if pp < 0.5 and pp > 0 else 1.0)
        m = float(np.mean(x))
        x = x * (target / m) if m > 0 else x * 0
        if p.get("model") and p.get("proj"):
            ratio = float(p["model"]) / float(p["proj"])
            if abs(ratio - 1) > 0.02:
                x = np.where(flip[:, t_idx[p["team"]]], x * ratio, x)
        plays = rng.random(sims) < pp
        out[:, i] = np.where(plays, x, 0.0)
    return out


def rake_field(F: np.ndarray, target: np.ndarray, rng, iters: int = 40) -> np.ndarray:
    """Redraw the field so each player appears about as often as his
    ownership estimate: lineup weights are raked (multiplied by the geometric
    mean of their players' target/actual ratios) until the weighted
    frequencies agree, then the field is resampled from those weights."""
    n = len(target)
    w = np.ones(len(F))
    for _ in range(iters):
        freq = np.bincount(F.ravel(), weights=np.repeat(w, F.shape[1]), minlength=n) / w.sum()
        r = np.log(np.clip((target + 0.003) / (freq + 0.003), 0.2, 5.0))
        w *= np.exp(r[F].mean(axis=1))
        w = np.clip(w / w.mean(), 0.02, 50.0)
    return F[rng.choice(len(F), size=len(F), p=w / w.sum())]


def build_field(payload: dict, size: int, rng, floor: float, value_power: float,
                stack: bool, best: float, tries: int = 60,
                adjust: np.ndarray | None = None) -> np.ndarray:
    players = payload["players"]
    roster = payload["roster"]
    cap = roster.get("salary_cap", 50_000)
    n = len(players)
    own = np.array([max(p.get("own") or 0.0, 0.0) for p in players])
    sal = np.array([p.get("salary") or 0 for p in players], float)
    proj = np.array([p.get("proj") or 0.0 for p in players])
    pos = [p["pos"] for p in players]
    ok = np.array([p.get("status") != "out" and (p.get("own") or 0) > 0.001 for p in players])
    w = own * ((proj / np.maximum(sal, 1) * 1000 / 2.0) ** value_power if value_power else 1)
    if adjust is not None:
        w = w * adjust
    by_pos = {s: np.array([i for i in range(n) if ok[i] and pos[i] == s]) for s in ("C", "W", "D", "G")}
    lines = {}
    for i, p in enumerate(players):
        if ok[i] and p.get("lg") and "-L" in p["lg"] and p["lg"][-1] in "12":
            lines.setdefault(p["lg"], []).append(i)
    lines = {k: v for k, v in lines.items() if len(v) >= 3}
    lkeys = list(lines)
    lw = np.array([own[lines[k]].sum() for k in lkeys]) if lkeys else np.array([])
    need = {"C": 2, "W": 3, "D": 2, "G": 1}
    out = []
    att = 0
    while len(out) < size and att < size * tries:
        att += 1
        chosen = []
        have = {"C": 0, "W": 0, "D": 0, "G": 0}
        util = False
        if stack and lkeys:
            L = lines[lkeys[int(rng.choice(len(lkeys), p=lw / lw.sum()))]]
            for i in L[:3]:
                s = pos[i]
                if have[s] < need[s]:
                    have[s] += 1
                elif not util:
                    util = True
                else:
                    continue
                chosen.append(i)
        bad = False
        for s in ("G", "D", "C", "W"):
            while have[s] < need[s]:
                cand = by_pos[s]
                cand = cand[~np.isin(cand, chosen)]
                if not len(cand):
                    bad = True
                    break
                ww = w[cand] + 1e-9
                j = int(rng.choice(cand, p=ww / ww.sum()))
                chosen.append(j)
                have[s] += 1
            if bad:
                break
        if bad:
            continue
        if not util:
            cand = np.concatenate([by_pos["C"], by_pos["W"], by_pos["D"]])
            cand = cand[~np.isin(cand, chosen)]
            ww = w[cand] + 1e-9
            chosen.append(int(rng.choice(cand, p=ww / ww.sum())))
        spent = sal[chosen].sum()
        skater_teams = {players[i]["team"] for i in chosen if pos[i] != "G"}
        if (len(chosen) == 9 and 0.94 * cap <= spent <= cap and len(skater_teams) >= 3
                and proj[chosen].sum() >= floor * best):
            out.append(chosen)
    return np.array(out, dtype=int)


def rank_pool(payload: dict, sims: int = 30000, field: int = 5000, seed: int = 11) -> None:
    pool = payload.get("lineups") or []
    if not pool or not payload.get("players"):
        return
    rng = np.random.default_rng(seed)
    S = simulate_scores(payload, sims, rng)
    best = max(L["proj"] for L in pool)
    parts = [build_field(payload, field // len(FIELD_MIX), rng, fl, vp, st, best)
             for fl, vp, st in FIELD_MIX]
    parts = [x for x in parts if len(x)]
    if parts:
        # Match the field's rostering to the ownership estimates. Built raw,
        # the 2026-11-18 field had Jack Hughes in 78% of lineups against a 42%
        # estimate. Rebuilding with corrected weights starved the salary
        # rules, so instead the lineups are reweighted (raking on players)
        # and the field redrawn from those weights.
        F0 = np.vstack(parts)
        target = np.array([max(p.get("own") or 0.0, 0.0) for p in payload["players"]])
        F1 = rake_field(F0, target, rng)
        freq0 = np.bincount(F0.ravel(), minlength=len(target)) / len(F0)
        freq1 = np.bincount(F1.ravel(), minlength=len(target)) / len(F1)
        top = np.argsort(-target)[:10]
        log.info("nhl sim field ownership vs estimate (top 10): mean |gap| %.1f -> %.1f pts",
                 float(np.abs(freq0[top] - target[top]).mean() * 100),
                 float(np.abs(freq1[top] - target[top]).mean() * 100))
        parts = [F1]
    if not parts or sum(len(x) for x in parts) < 300:
        log.warning("nhl sim: field too small (%d); lineups left unranked",
                    sum(len(x) for x in parts))
        return
    Fd = np.vstack(parts)
    q99 = np.empty(sims); q90 = np.empty(sims); q55 = np.empty(sims)
    q999 = np.empty(sims)
    for a in range(0, sims, 250):
        fs = S[a:a + 250][:, Fd].sum(axis=2)
        q99[a:a + 250] = np.quantile(fs, 0.99, axis=1)
        q999[a:a + 250] = np.quantile(fs, 0.999, axis=1)
        q90[a:a + 250] = np.quantile(fs, 0.90, axis=1)
        q55[a:a + 250] = np.quantile(fs, 0.55, axis=1)
    size = len(payload.get("roster", {}).get("slots", [])) or max(len(L["p"]) for L in pool)
    bad = [L for L in pool if len(L["p"]) != size]
    if bad:
        log.warning("nhl sim: dropping %d malformed lineup(s)", len(bad))
        pool[:] = [L for L in pool if len(L["p"]) == size]
    P = np.array([L["p"] for L in pool])
    ls = S[:, P].sum(axis=2)
    for k, L in enumerate(pool):
        L["top1"] = round(float((ls[:, k] > q99).mean() * 100), 2)
        L["top10"] = round(float((ls[:, k] > q90).mean() * 100), 1)
        L["cash"] = round(float((ls[:, k] > q55).mean() * 100), 1)
    payload["sim"] = {"sims": sims, "field": int(len(Fd))}
    from engine import portfolio as PF
    PF.assign(pool, ls, q999, q99)
    top = max(pool, key=lambda L: L["top1"])
    log.info("nhl sim: %d nights x %d field lineups; best top-1%% %.2f%% (%s)",
             sims, len(Fd), top["top1"], top["src"])
    return {"S": S, "Fd": Fd, "ls": ls}
