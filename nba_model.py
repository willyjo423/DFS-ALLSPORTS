"""NBA DraftKings projections: minutes x points-per-minute, adjusted for
tonight's absences, the Vegas total and blowout risk.

What the research says matters, in order, and how each is built here:

1. MINUTES. Everything is minutes x rate. A player's recent minutes
   (exponentially weighted over games he played) set his baseline.
2. WHO IS OUT. A team plays 240 minutes however many men it has, so when
   players are out the remaining baselines are scaled up to fill them - with
   a fitted damping, because coaches spread lost minutes unevenly and the
   bench soaks up part of it. Measured on every past game using who actually
   did not play.
3. USAGE. When a high-usage teammate is out, the rest do more per minute.
   `vacated` is the out players' share of the team's recent possessions used
   (FGA + 0.44 FTA + TOV), net of how often the player's own sample already
   had them missing; its coefficient is fitted.
4. VEGAS. Team implied points (total/2 - spread/2) against the team's recent
   scoring scales the rate - pace and matchup in one number the market has
   already sharpened. Exponent fitted.
5. BLOWOUTS. A big spread cuts the regulars' minutes and gives them to the
   bench. Fitted on closing spreads.

Points per minute is shrunk toward the player's longer-run rate, so a hot
week does not pass for a new player. Every coefficient comes from
`fit()` on history, walk-forward, and is saved to data/nba_model.json.
"""
from __future__ import annotations

import json
import logging
import math
import os

import numpy as np
import pandas as pd

import nba_fetch as F

log = logging.getLogger("nba_model")
HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(HERE, "data", "nba_model.json")

DEFAULTS = {
    "min_halflife": 5.0,      # games
    "rate_halflife": 12.0,
    "rate_prior_games": 6.0,  # shrinkage toward the long-run rate
    "team_minutes": 240.0,
    "fill_gamma": 0.7,        # 1 = fill lost minutes fully pro rata
    "usage_k": 0.6,           # fp/min gain per unit of vacated usage share
    "usage_floor": 0.10,      # ... above this share
    "vegas_a": 0.8,           # exponent on implied / recent team points
    "blowout_b": 0.010,       # minutes lost per point of spread beyond 6, regulars
    "max_minutes": 42.0,
    "role_w": 0.0,            # weight on start/bench-specific minutes
    "pair_on": 0,             # teammate-specific on/off adjustments
    "pair_kmin": 4.0, "pair_krate": 8.0,
    "sd_a": 2.5, "sd_b": 0.30,  # residual sd = a + b * mean
}


def params() -> dict:
    p = dict(DEFAULTS)
    if os.path.exists(PARAMS_PATH):
        try:
            p.update(json.load(open(PARAMS_PATH)).get("params", {}))
        except Exception:                                      # noqa: BLE001
            pass
    return p


# ---------------------------------------------------------------- features
def prepare(hist: pd.DataFrame) -> pd.DataFrame:
    """Per player-game features from games BEFORE that game only."""
    h = hist.copy()
    # All-Star and exhibition sides are not NBA teams.
    real = h.groupby("team")["game_id"].nunique()
    h = h[h["team"].isin(real[real >= 20].index) & h["opp"].isin(real[real >= 20].index)]
    h["date"] = pd.to_datetime(h["date"])
    h["fp"] = F.dk_points(h)
    h["use"] = h["fga"] + 0.44 * h["fta"] + h["tov"]
    h = h.sort_values(["player_id", "date"]).reset_index(drop=True)
    return h


def _ewm_prior(g: pd.Series, played: pd.Series, hl: float) -> pd.Series:
    """EWM over games played, shifted so a row only sees earlier games."""
    v = g.where(played == 1)
    m = v.ewm(halflife=hl, ignore_na=True).mean()
    return m.shift(1).ffill()


def player_history_features(h: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Columns base_min, rate, rate_long, use_rate, gp_prior per row,
    computed only from that player's earlier games."""
    out = []
    for pid, g in h.groupby("player_id", sort=False):
        played = g["played"]
        mins = g["min"].where(played == 1)
        fpm = (g["fp"] / g["min"]).where((played == 1) & (g["min"] > 0))
        usem = (g["use"] / g["min"]).where((played == 1) & (g["min"] > 0))
        w = mins.fillna(0)
        base_min = _ewm_prior(g["min"], played, p["min_halflife"])
        # minutes-weighted rates: weight each game by its minutes
        num = (fpm * w).ewm(halflife=p["rate_halflife"], ignore_na=True).mean()
        den = w.where(played == 1).ewm(halflife=p["rate_halflife"], ignore_na=True).mean()
        rate = (num / den).shift(1)
        long_num = (fpm.fillna(0) * w).cumsum().shift(1)
        long_den = w.cumsum().shift(1)
        rate_long = long_num / long_den
        use_rate = ((usem * w).ewm(halflife=p["rate_halflife"], ignore_na=True).mean()
                    / den).shift(1)
        st = g["starter"].fillna(0) if "starter" in g else pd.Series(0, index=g.index)
        start_min = g["min"].where((played == 1) & (st == 1)).ewm(
            halflife=p["min_halflife"], ignore_na=True).mean().shift(1).ffill()
        bench_min = g["min"].where((played == 1) & (st == 0)).ewm(
            halflife=p["min_halflife"], ignore_na=True).mean().shift(1).ffill()
        n_start = ((played == 1) & (st == 1)).cumsum().shift(1).fillna(0)
        n_bench = ((played == 1) & (st == 0)).cumsum().shift(1).fillna(0)
        gp = played.cumsum().shift(1).fillna(0)
        # How much of his team's recent stretch he had ALREADY missed. An
        # absence that is weeks old is in every teammate's recent numbers
        # already; only a fresh one moves anything.
        absent_prev = (1 - played).rolling(8, min_periods=1).mean().shift(1).fillna(0)
        out.append(pd.DataFrame({
            "base_min": base_min.values, "rate": rate.values,
            "rate_long": rate_long.values, "use_rate": use_rate.values,
            "gp_prior": gp.values, "absent_prev": absent_prev.values,
            "start_min": start_min.values, "bench_min": bench_min.values,
            "n_start": n_start.values, "n_bench": n_bench.values},
            index=g.index))
    f = pd.concat(out).sort_index()
    return h.join(f)


def team_scoring_prior(h: pd.DataFrame) -> pd.DataFrame:
    """Each team's recent points per game before each game."""
    g = (h.drop_duplicates(["game_id", "team"])[["game_id", "date", "team", "team_pts"]]
         .sort_values("date"))
    g["team_avg"] = (g.groupby("team")["team_pts"]
                     .transform(lambda s: s.ewm(halflife=10).mean().shift(1)))
    return g[["game_id", "team", "team_avg"]]


def project_rows(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Given rows of one or more team-games with features and an `avail`
    flag (1 = will play), return projected minutes and DK points for the
    available rows. Used by both the backtest and tonight's board."""
    df = df.copy()
    shrink_w = df["gp_prior"] / (df["gp_prior"] + p["rate_prior_games"])
    rate = (shrink_w * df["rate"].fillna(df["rate_long"])
            + (1 - shrink_w) * df["rate_long"].fillna(df["rate"]))
    df["rate_s"] = rate.fillna(0.8)
    df["base_min"] = df["base_min"].fillna(0.0)
    # Tonight's role, where known (1 starts, 0 comes off the bench): his own
    # minutes in that role, blended in. Unknown -> the role-blind baseline.
    if "starts" in df and p.get("role_w", 0) > 0:
        rs = df["starts"]
        role = np.where((rs == 1) & (df.get("n_start", 0) >= 2), df.get("start_min"),
                        np.where((rs == 0) & (df.get("n_bench", 0) >= 2), df.get("bench_min"),
                                 np.nan))
        role = pd.Series(role, index=df.index, dtype=float)
        df["base_min"] = np.where(role.notna(),
                                  (1 - p["role_w"]) * df["base_min"] + p["role_w"] * role,
                                  df["base_min"])
    # This player's own history without tonight's missing teammates.
    if "pair_dmin" in df and p.get("pair_on", 0):
        df["base_min"] = (df["base_min"] + df["pair_dmin"].fillna(0)).clip(lower=0)
        df["rate_s"] = df["rate_s"] * (1 + df["pair_drate"].fillna(0))
    keys = ["game_id", "team"]

    # usage share of each player within his team's available + out players
    df["use_tot"] = df["use_rate"].fillna(0) * df["base_min"]
    team_use = df.groupby(keys)["use_tot"].transform("sum")
    fresh = 1 - df.get("absent_prev", pd.Series(0.0, index=df.index)).fillna(0)
    out_use = (df["use_tot"] * fresh).where(df["avail"] == 0, 0).groupby(
        [df[k] for k in keys]).transform("sum")
    df["vacated"] = (out_use / team_use.replace(0, np.nan)).fillna(0)

    a = df[df["avail"] == 1].copy()
    tot = a.groupby(keys)["base_min"].transform("sum").replace(0, np.nan)
    scale = (p["team_minutes"] / tot).fillna(1.0)
    a["min_p"] = a["base_min"] * scale.clip(0.6, 2.0) ** p["fill_gamma"]
    # blowout: regulars lose, bench gains (zero-sum within the team)
    sp = a["spread"].abs().fillna(0)
    reg = (a["base_min"] >= 26).astype(float)
    loss = p["blowout_b"] * (sp - 6).clip(lower=0) * a["min_p"] * reg
    gain_pool = loss.groupby([a[k] for k in keys]).transform("sum")
    bench_w = (1 - reg) * a["min_p"]
    bench_share = bench_w / bench_w.groupby([a[k] for k in keys]).transform("sum").replace(0, np.nan)
    a["min_p"] = (a["min_p"] - loss + gain_pool * bench_share.fillna(0)).clip(0, p["max_minutes"])

    implied = a["total"] / 2 - a["spread"] / 2
    vegas = (implied / a["team_avg"]).where(implied.notna() & a["team_avg"].notna(), 1.0)
    a["vegas"] = vegas.clip(0.8, 1.25) ** p["vegas_a"]
    # Measured: no effect until about a tenth of the team's possessions are
    # missing, then roughly linear (20-40% vacated -> ~+12% per minute,
    # 40%+ -> ~+30%). A linear term from zero fitted to nothing.
    bump = p["usage_k"] * (a["vacated"] - p["usage_floor"]).clip(lower=0)
    a["proj"] = a["min_p"] * a["rate_s"] * (1 + bump) * a["vegas"]
    a["sd"] = p["sd_a"] + p["sd_b"] * a["proj"]
    return a


# ----------------------------------------------------------------- backtest
def build_frame(hist: pd.DataFrame, p: dict) -> pd.DataFrame:
    h = prepare(hist)
    h = player_history_features(h, p)
    h = h.merge(team_scoring_prior(h), on=["game_id", "team"], how="left")
    h["avail"] = h["played"]
    # Who started is announced before tip, so the backtest may use it; the
    # live board uses the NBA's official expected/confirmed lineups.
    h["starts"] = h["starter"]
    return h


def evaluate(frame: pd.DataFrame, p: dict, start: str) -> dict:
    sub = frame[(frame["date"] >= start)]
    pr = project_rows(sub, p)
    pr = pr[(pr["gp_prior"] >= 3) & pr["base_min"].gt(0)]
    if not len(pr):
        return {}
    err = pr["proj"] - pr["fp"]
    # per-night rank correlation, which is what a lineup decision uses
    rhos = []
    for _, g in pr.groupby("date"):
        if len(g) > 20:
            rhos.append(g["proj"].rank().corr(g["fp"].rank()))
    naive = pr["rate_long"].fillna(0) * pr["base_min"]
    return {"n": int(len(pr)), "mae": float(err.abs().mean()),
            "bias": float(err.mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "rho": float(np.mean(rhos)) if rhos else float("nan"),
            "naive_mae": float((naive - pr["fp"]).abs().mean()),
            "min_mae": float((pr["min_p"] - pr["min"]).abs().mean())}


def fit(hist: pd.DataFrame, start: str, grid: dict | None = None) -> tuple[dict, dict]:
    """Coordinate search of the fitted coefficients on MAE from `start`."""
    p = dict(DEFAULTS)
    best_hl = None
    for mh in (3.0, 5.0, 8.0):
        for rh in (8.0, 12.0, 20.0, 30.0):
            q = dict(p, min_halflife=mh, rate_halflife=rh)
            r = evaluate(build_frame(hist, q), q, start)
            if best_hl is None or r["mae"] < best_hl[0]:
                best_hl = (r["mae"], mh, rh)
    p.update(min_halflife=best_hl[1], rate_halflife=best_hl[2])
    frame = build_frame(hist, p)
    best = evaluate(frame, p, start)
    grid = grid or {
        "fill_gamma": [0.0, 0.3, 0.5, 0.7, 0.85, 1.0],
        "usage_k": [0.0, 0.2, 0.4, 0.6, 0.9, 1.2, 1.6],
        "usage_floor": [0.0, 0.05, 0.10, 0.15, 0.20],
        "vegas_a": [0.0, 0.4, 0.8, 1.2, 1.6],
        "blowout_b": [0.0, 0.005, 0.01, 0.015, 0.02, 0.03],
        "rate_prior_games": [4.0, 10.0, 16.0, 25.0, 40.0, 60.0],
        "role_w": [0.0, 0.3, 0.6, 0.8, 1.0],
    }
    # The usage pair is searched jointly: the effect only exists above the
    # floor, so either number alone looks like noise.
    for k_ in (0.0, 0.3, 0.6, 0.9, 1.2):
        for fl in (0.1, 0.15, 0.2, 0.25):
            q = dict(p, usage_k=k_, usage_floor=fl)
            r = evaluate(frame, q, start)
            if r and r["mae"] < best["mae"] - 1e-4:
                p, best = q, r
    grid.pop("usage_k", None)
    grid.pop("usage_floor", None)
    for _round in range(2):
        for k, vals in grid.items():
            for v in vals:
                q = dict(p, **{k: v})
                r = evaluate(frame, q, start)
                if r and r["mae"] < best["mae"] - 1e-4:
                    p, best = q, r
    # residual spread for the quantiles
    pr = project_rows(frame[frame["date"] >= start], p)
    pr = pr[(pr["gp_prior"] >= 3) & pr["base_min"].gt(0)]
    resid = (pr["fp"] - pr["proj"]).abs()
    A = np.vstack([np.ones(len(pr)), pr["proj"]]).T
    coef, *_ = np.linalg.lstsq(A, resid * math.sqrt(math.pi / 2), rcond=None)
    p["sd_a"], p["sd_b"] = float(max(coef[0], 1.0)), float(max(coef[1], 0.1))
    best = evaluate(frame, p, start)
    return p, best


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    hist = F.load()
    dates = sorted(hist["date"].unique())
    start = dates[int(len(dates) * 0.45)]
    p, best = fit(hist, start)
    base = evaluate(build_frame(hist, DEFAULTS), DEFAULTS, start)
    log.info("fitted from %s: %s", start, json.dumps(best))
    log.info("defaults:       %s", json.dumps(base))
    os.makedirs(os.path.dirname(PARAMS_PATH), exist_ok=True)
    json.dump({"fitted_from": str(start), "through": str(dates[-1]),
               "metrics": best, "params": p}, open(PARAMS_PATH, "w"), indent=1)


if __name__ == "__main__":
    main()


# ------------------------------------------------- teammate on/off + roles
def pair_table(h: pd.DataFrame, min_base: float = 15.0) -> pd.DataFrame:
    """For every (teammate, absent player) pair on the same team: how the
    teammate's minutes and points-per-minute changed in games that player
    missed, against games he played. Built only from the games in `h`.

    Absent games count only when the absent man was a rotation player
    (base_min >= min_base going in) - a two-way player sitting tells
    nobody anything."""
    p = h[["game_id", "team", "player_id", "played", "min", "fp", "base_min"]].copy()
    rot = p[(p["base_min"] >= min_base)]
    absent = rot[rot["played"] == 0][["game_id", "team", "player_id"]].rename(
        columns={"player_id": "x"})
    present = rot[rot["played"] == 1][["game_id", "team", "player_id"]].rename(
        columns={"player_id": "x"})
    me = p[(p["played"] == 1) & (p["min"] > 0)][["game_id", "team", "player_id", "min", "fp"]]
    wo = me.merge(absent, on=["game_id", "team"])
    wo = wo[wo["player_id"] != wo["x"]]
    wi = me.merge(present, on=["game_id", "team"])
    wi = wi[wi["player_id"] != wi["x"]]
    def agg(d, tag):
        g = d.groupby(["player_id", "x"]).agg(n=("min", "size"), m=("min", "mean"),
                                              fp=("fp", "sum"), mins=("min", "sum"))
        g["r"] = g["fp"] / g["mins"]
        return g[["n", "m", "r"]].add_suffix("_" + tag)
    t = agg(wo, "wo").join(agg(wi, "wi"), how="inner")
    t = t[t["n_wi"] >= 3]
    return t


def pair_effects(t: pd.DataFrame, k_min: float, k_rate: float) -> pd.DataFrame:
    w_m = t["n_wo"] / (t["n_wo"] + k_min)
    w_r = t["n_wo"] / (t["n_wo"] + k_rate)
    out = pd.DataFrame(index=t.index)
    out["dmin"] = (t["m_wo"] - t["m_wi"]) * w_m
    out["drate"] = ((t["r_wo"] / t["r_wi"]) - 1).clip(-0.5, 0.8) * w_r
    return out


def add_pair_columns(frame: pd.DataFrame, start, p: dict, freq: str = "MS") -> pd.DataFrame:
    """Walk-forward: each month's rows get pair effects fitted only on games
    before that month."""
    f = frame.copy()
    f["pair_dmin"] = 0.0
    f["pair_drate"] = 0.0
    start = pd.Timestamp(start)
    edges = list(pd.date_range(start, f["date"].max() + pd.Timedelta(days=31), freq=freq))
    edges = [start] + [e for e in edges if e > start]
    for lo, hi in zip(edges[:-1], edges[1:]):
        eff = pair_effects(pair_table(f[f["date"] < lo]), p["pair_kmin"], p["pair_krate"])
        blk = f[(f["date"] >= lo) & (f["date"] < hi)]
        outs = blk[(blk["played"] == 0) & (blk["base_min"] >= 15)
                   & (blk["absent_prev"] < 0.5)][["game_id", "team", "player_id"]]
        outs = outs.rename(columns={"player_id": "x"})
        me = blk[["game_id", "team", "player_id"]].reset_index()
        m = me.merge(outs, on=["game_id", "team"])
        m = m[m["player_id"] != m["x"]].join(eff, on=["player_id", "x"]).fillna(
            {"dmin": 0.0, "drate": 0.0})
        agg = m.groupby("index")[["dmin", "drate"]].sum()
        f.loc[agg.index, "pair_dmin"] = agg["dmin"].values
        f.loc[agg.index, "pair_drate"] = agg["drate"].values
    return f
