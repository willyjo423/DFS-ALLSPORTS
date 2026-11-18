"""Sharpen the published projection with what DraftKings' price already knows.

Why
---
Graded on every archived NHL board (Sep 29 - Nov 15 2026, ~10,800 skater
games, scored from the box scores with DraftKings' rules), the model's median
ranked players about as well as salary alone did (rank correlation 0.40 vs
0.39), and the two carry DIFFERENT information. Salary encodes things the
stat model cannot see - line and power-play deployment, news, the pricing
team's judgement - while the model sees rates and matchups the price lags.
A blend of the two beats either one out of sample:

    trained Sep 29 - Oct 31, tested Nov 1 - Nov 15 (3,264 skater games)
                          rank corr     correlation
    model median            0.396          0.420
    salary alone            0.410          0.424
    blend (this file)       0.415          0.436

The blend is also an EXPECTATION rather than a median. A DFS lineup's total
is a sum, and a sum is maximised in expectation by summing means; the median
sits about 1.4 points under the mean for a skater because his scoring is
skewed, and it under-rates exactly the high-upside players a tournament
wants.

What is published
-----------------
Every player row gains `proj`: the calibrated expected DraftKings points.
`med`, `ceil` and the quantile curve are left exactly as the model made
them. Where a position has no calibration (goalies; any sport not yet fitted)
`proj` is the model's mean if published, else its median.

Refit with `python projection_calibration.py --fit nhl` (weekly in Actions).
Coefficients live in data/proj_calibration_<sport>.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("projection_calibration")
ROOT = Path(__file__).resolve().parent
NHL_HISTORY = ("https://raw.githubusercontent.com/willyjo423/nhlprojections/"
               "main/data/skaters_{season}.csv.gz")
MIN_GAMES = 300            # per position, before a fit is trusted
import datetime as _dt
CFB_WEEK1 = _dt.date(2026, 9, 5)


def path_for(sport: str) -> Path:
    return ROOT / "data" / f"proj_calibration_{sport}.json"


def load(sport: str) -> dict | None:
    p = path_for(sport)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        log.warning("%s unreadable (%s) - projections published uncalibrated",
                    p.name, exc)
        return None


def apply(payload: dict) -> None:
    """Add `proj` to every player row of one published slate, in place."""
    cal = load(str(payload.get("sport", "")))
    coefs = (cal or {}).get("positions", {})
    n_cal = 0
    for p in payload.get("players", []):
        base = p.get("mean")
        if base is None:
            base = p.get("med") or 0.0
        c = coefs.get(str(p.get("pos")))
        if c and p.get("salary"):
            pp = p.get("pp")
            if pp is not None and float(pp) < 0.5:
                # A player the model thinks will probably not play keeps that
                # view: the salary term must not resurrect a scratch.
                p["proj"] = round(float(base), 2)
                continue
            v = c["a"] + c["b"] * p["salary"] / 1000.0 + c["c"] * (p.get("med") or 0.0)
            p["proj"] = round(max(0.0, v), 2)
            n_cal += 1
        else:
            p["proj"] = round(float(base), 2)
    if cal:
        log.info("projections: %d of %d players calibrated (%s fit, %s)",
                 n_cal, len(payload.get("players", [])), payload.get("sport"),
                 cal.get("fitted_through"))


# ----------------------------------------------------------------- fitting
def _nhl_actuals(seasons=(2026,)) -> pd.DataFrame:
    import nhl_projections as NP
    from nhl_data import normalise
    frames = []
    for s in seasons:
        local = Path(f"/home/claude/nhlp/data/skaters_{s}.csv.gz")
        src = local if local.exists() else NHL_HISTORY.format(season=s)
        frames.append(pd.read_csv(src))
    h = pd.concat(frames, ignore_index=True)
    if "played" in h.columns:
        h = h[h["played"] == 1]
    shp = (h.get("pk_goals", 0) + h.get("pk_assists", 0)).to_numpy()
    h["dk"] = NP.score_skater_lines(h["goals"].to_numpy(),
                                    h["assists"].to_numpy(),
                                    h["sog"].to_numpy(),
                                    h["blocks"].to_numpy(), shp)
    h["date"] = pd.to_datetime(h["date"])
    h["norm"] = h["name"].map(normalise)
    return h[["norm", "date", "dk"]]


def _nhl_boards() -> pd.DataFrame:
    from nhl_data import normalise
    rows = []
    for f in sorted(glob.glob(str(ROOT / "docs/data/archive/*/nhl_dk_classic_*.json"))):
        day = os.path.basename(os.path.dirname(f))
        d = json.load(open(f))
        for p in d["players"]:
            if p.get("salary") and p.get("med") is not None:
                rows.append({"day": day, "norm": normalise(p["name"]),
                             "pos": p["pos"], "sal": p["salary"],
                             "med": p["med"]})
    b = pd.DataFrame(rows).drop_duplicates(["day", "norm"])
    b["date"] = pd.to_datetime(b["day"])
    return b


def fit_nhl() -> dict:
    from scipy.stats import spearmanr
    m = _nhl_boards().merge(_nhl_actuals(), on=["norm", "date"])
    days = sorted(m["day"].unique())
    cut = days[int(len(days) * 0.7)]
    out = {"sport": "nhl", "fitted_through": days[-1], "games": int(len(m)),
           "features": "proj = a + b * salary/1000 + c * model median",
           "positions": {}, "holdout": {}}
    for pos, g in m.groupby("pos"):
        if pos == "G" or len(g) < MIN_GAMES:
            continue           # goalie wins are not in the skater history
        X = np.column_stack([np.ones(len(g)), g["sal"] / 1000.0, g["med"]])
        a, b, c = np.linalg.lstsq(X, g["dk"], rcond=None)[0]
        out["positions"][pos] = {"a": round(float(a), 4), "b": round(float(b), 4),
                                 "c": round(float(c), 4), "n": int(len(g))}
        tr, te = g[g["day"] < cut], g[g["day"] >= cut]
        Xtr = np.column_stack([np.ones(len(tr)), tr["sal"] / 1000.0, tr["med"]])
        Xte = np.column_stack([np.ones(len(te)), te["sal"] / 1000.0, te["med"]])
        k = np.linalg.lstsq(Xtr, tr["dk"], rcond=None)[0]
        out["holdout"][pos] = {
            "from": cut, "n": int(len(te)),
            "rho_model": round(float(spearmanr(te["med"], te["dk"])[0]), 3),
            "rho_calibrated": round(float(spearmanr(Xte @ k, te["dk"])[0]), 3),
            "bias_model": round(float((te["dk"] - te["med"]).mean()), 2),
            "bias_calibrated": round(float((te["dk"] - Xte @ k).mean()), 2)}
    return out


def _cfb_matched() -> pd.DataFrame:
    """Archived CFB boards joined to what each player actually scored.

    CFBD numbers weeks from the first full Saturday (2026-09-05 is week 1;
    the week-0 games are folded into it), so a board's date gives its week.
    Matched by name within the week; anyone whose name maps to two players
    that week is dropped rather than guessed."""
    import datetime as dt
    import cfb_data as D
    frames = [pd.read_csv(f) for f in sorted((ROOT / "data").glob("history_20*.csv.gz"))]
    h = pd.concat(frames, ignore_index=True)
    h = h[h["season"] == h["season"].max()]
    h["k"] = h["name"].map(D.primary_key)
    hw = h.groupby(["k", "week"]).agg(dk=("points", "sum"), n=("points", "size")).reset_index()
    hw = hw[hw["n"] == 1]
    rows = []
    for f in sorted((ROOT / "docs" / "data" / "archive").glob("*/cfb_dk_classic_*.json")):
        d = json.loads(f.read_text())
        day = f.parent.name
        wk = (dt.date.fromisoformat(day) - CFB_WEEK1).days // 7 + 1
        for p in d.get("players", []):
            if p.get("salary") and p.get("med") is not None and (p.get("pp") or 1) >= 0.5:
                rows.append({"day": day, "week": wk, "name": p["name"], "pos": p["pos"],
                             "team": p.get("team"), "sal": p["salary"], "med": p["med"]})
    b = pd.DataFrame(rows).drop_duplicates(["week", "name", "team"])
    b["k"] = b["name"].map(D.primary_key)
    return b.merge(hw, on=["k", "week"])


def fit_cfb() -> dict:
    from scipy.stats import spearmanr
    m = _cfb_matched()
    weeks = sorted(m["week"].unique())
    cut = weeks[-2] if len(weeks) > 3 else weeks[-1]
    out = {"sport": "cfb", "fitted_through": f"week {weeks[-1]}", "games": int(len(m)),
           "features": "proj = a + b * salary/1000 + c * model median",
           "positions": {}, "holdout": {}}
    for pos, g in m.groupby("pos"):
        if len(g) < 150:
            continue
        X = lambda d: np.column_stack([np.ones(len(d)), d["sal"] / 1000.0, d["med"]])
        a, b, c = np.linalg.lstsq(X(g), g["dk"], rcond=None)[0]
        out["positions"][pos] = {"a": round(float(a), 4), "b": round(float(b), 4),
                                 "c": round(float(c), 4), "n": int(len(g))}
        tr, te = g[g["week"] < cut], g[g["week"] >= cut]
        k = np.linalg.lstsq(X(tr), tr["dk"], rcond=None)[0]
        out["holdout"][pos] = {
            "from_week": int(cut), "n": int(len(te)),
            "rho_model": round(float(spearmanr(te["med"], te["dk"])[0]), 3),
            "rho_salary": round(float(spearmanr(te["sal"], te["dk"])[0]), 3),
            "rho_calibrated": round(float(spearmanr(X(te) @ k, te["dk"])[0]), 3)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fit", choices=["nhl", "cfb"], required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fit = fit_nhl() if args.fit == "nhl" else fit_cfb()
    if not fit["positions"]:
        print("not enough graded games to fit anything - nothing written")
        return 1
    path_for(args.fit).parent.mkdir(exist_ok=True)
    path_for(args.fit).write_text(json.dumps(fit, indent=1) + "\n")
    print(json.dumps(fit, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
