"""Are the simulations' upper tails wide enough - for cheap players especially?

For every archived lock-time board (backtest.boards) and its real box scores,
each player's own simulated distribution is drawn with the sport's simulation
(nba_sim / nhl_sim / cfb_sim; MLB from its published quantile curve), and the
actual score is placed in it. If the sims are honest, a player beats his
simulated 90th / 97th / 99th percentile 10% / 3% / 1% of the time. Reported by
salary tier within each board (cheapest quarter = "punt").

    python tail_check.py --sports nba nhl cfb mlb
Writes docs/data/backtest/tails.json.
"""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime

import numpy as np
import pandas as pd

import backtest as B
import grade_nightly as G

LEVELS = (0.90, 0.97, 0.99)
TIERS = ("punt", "low", "mid", "high")


def draws(sport: str, b: dict, sims: int, rng) -> np.ndarray:
    p = copy.deepcopy(b)
    if sport == "nba":
        from engine import nba_sim as SIM
        return SIM.simulate_scores(p["players"], sims, rng)
    if sport in ("nhl", "cfb"):
        import projection_calibration as PC
        try:
            PC.apply(p)
        except Exception:                                      # noqa: BLE001
            pass
        SIM = __import__(f"engine.{sport}_sim", fromlist=["x"])
        return SIM.simulate_scores(p, sims, rng)
    # MLB: the published curve, as engine.simulate reads it (margins only)
    from engine import simulate as S
    qs = np.asarray(p.get("quantiles") or [0.1, 0.25, 0.5, 0.75, 0.9, 0.97])
    zq = S._z_of(qs)
    out = np.zeros((sims, len(p["players"])))
    for i, pl in enumerate(p["players"]):
        q = pl.get("q")
        if not q or pl.get("status") == "out":
            continue
        x = np.clip(S._quantile_curve(zq, np.asarray(q, float), rng.standard_normal(sims)), 0, None)
        out[:, i] = x * (rng.random(sims) < float(pl.get("pp") or 1.0))
    return out


_MLB = None


def mlb_actuals(day: str) -> dict:
    global _MLB
    if _MLB is None:
        import mlb_data as MD
        _MLB = pd.read_csv(B.ROOT / "data" / "mlb_2026.csv.gz", usecols=["name", "date", "points"])
        _MLB["k"] = _MLB["name"].map(MD.normalise)
    d = _MLB[_MLB["date"] == day]
    return {k: float(v) for k, v in zip(d["k"], d["points"])}


def actual_for(sport: str, pl: dict, act: dict):
    if sport == "cfb":
        import cfb_data as D
        return act.get(D.primary_key(pl["name"]))
    if sport == "mlb":
        import mlb_data as MD
        return act.get(MD.normalise(pl["name"]))
    a = G.match(sport, pl, act)
    return None if a is None else a["dk"]


def run(sport: str, sims: int) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for day, b in B.boards(sport).items():
        d = day.split("|")[0]
        try:
            if sport == "cfb":
                act = B.cfb_actuals(b, d)
            elif sport == "mlb":
                act = mlb_actuals(d)
            else:
                act = B.actuals(sport, d)
        except Exception as exc:                               # noqa: BLE001
            print(f"  {sport} {day}: actuals failed ({str(exc)[:60]})")
            continue
        if not act:
            continue
        P = b["players"]
        X = draws(sport, b, sims, rng)
        sal = np.array([float(p.get("salary") or 0) for p in P])
        # salary rank within the board (ties share a rank: a pile of
        # minimum-salary players all lands in the bottom tier)
        rk = pd.Series(sal).rank(pct=True, method="average").to_numpy()
        got = 0
        for i, p in enumerate(P):
            if p.get("status") == "out" or float(p.get("pp") or 1.0) <= 0:
                continue
            a = actual_for(sport, p, act)
            if a is None:
                continue
            got += 1
            x = X[:, i]
            r = {"sport": sport, "day": d, "name": p["name"], "pos": p.get("pos"),
                 "salary": sal[i], "tier": TIERS[min(int(rk[i] * 4 - 1e-9), 3)],
                 "actual": float(a), "pit": float((x < a).mean() + 0.5 * (x == a).mean()),
                 "proj": float(p.get("proj") or p.get("med") or 0)}
            for lv in LEVELS:
                r[f"p{int(lv * 100)}"] = float(np.quantile(x, lv))
            rows.append(r)
        print(f"  {sport} {day}: {got}/{len(P)} players matched")
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> list[dict]:
    out = []
    for (sport, tier), g in [((s, t), df[(df.sport == s) & (df.tier == t)])
                             for s in df.sport.unique() for t in TIERS + ("all",)]:
        if tier == "all":
            g = df[df.sport == sport]
        if g.empty:
            continue
        r = {"sport": sport, "tier": tier, "n": len(g), "nights": g["day"].nunique(),
             "avg_salary": round(float(g.salary.mean())),
             "proj": round(float(g.proj.mean()), 2), "actual": round(float(g.actual.mean()), 2)}
        for lv in LEVELS:
            k = f"p{int(lv * 100)}"
            hit = (g.actual > g[k]) & (g[k] > 0)
            r[f"over_{k}"] = round(float(hit.mean()), 4)
            r[f"expect_{k}"] = round(1 - lv, 4)
            r[f"ratio_{k}"] = round(float(hit.mean()) / (1 - lv), 2)
            # one-sided binomial z against the nominal rate
            n, e = len(g), 1 - lv
            r[f"z_{k}"] = round((hit.sum() - n * e) / np.sqrt(n * e * (1 - e)), 2)
        out.append(r)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sports", nargs="+", default=["nba", "nhl", "cfb", "mlb"])
    ap.add_argument("--sims", type=int, default=4000)
    a = ap.parse_args(argv)
    frames = []
    for s in a.sports:
        try:
            frames.append(run(s, a.sims))
        except Exception as exc:                               # noqa: BLE001
            print(f"{s}: failed ({exc})")
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    summ = summary(df)
    B.OUT.mkdir(parents=True, exist_ok=True)
    (B.OUT / "tails.json").write_text(json.dumps({"generated": datetime.utcnow().isoformat(),
                                                  "summary": summ}, indent=1))
    df.to_csv(B.ROOT / "data" / "tails_rows.csv.gz", index=False)
    cols = ["sport", "tier", "n", "nights", "avg_salary", "proj", "actual",
            "over_p90", "over_p97", "over_p99", "z_p90", "z_p97", "z_p99"]
    print(pd.DataFrame(summ)[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
