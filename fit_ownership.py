"""Fit the classic ownership formula to real contests, one sport at a time.

Ownership is the one number on the board you can grade the next morning: a
finished DraftKings contest's standings export carries every player's
%Drafted. This script joins one or more of those exports to the slate file
that was published for the same draft group (docs/data/archive/<day>/...)
and grid-searches the four numbers the classic formula uses:

    appeal = points * z(projection) + salary * z(salary) + value * z(pts/$1k)
    ownership = softmax(appeal * concentration) per position, scaled so each
                position sums to what a lineup must field

    python fit_ownership.py --sport nhl \\
        --pair contest-standings-123.csv docs/data/archive/2026-11-17/nhl_dk_classic_156632.json \\
        --pair contest-standings-456.csv docs/data/archive/2026-11-18/nhl_dk_classic_156700.json

Writes data/ownership_fit_<sport>.json, which every publish of that sport
reads on start. Players the board priced but nobody drafted count as 0%.
Fit on as many slates as you have; one slate is a starting point, not an
answer.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from engine import ownership as OWN


def contest_ownership(path: str) -> dict[str, float]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "Player" not in df.columns or "%Drafted" not in df.columns:
        sys.exit(f"{path}: not a DraftKings standings export (no Player / "
                 f"%Drafted columns)")
    df = df.dropna(subset=["Player", "%Drafted"])
    return {str(n).strip(): float(str(v).rstrip("%")) / 100.0
            for n, v in zip(df["Player"], df["%Drafted"])}


def slate_pool(path: str) -> tuple[pd.DataFrame, dict]:
    d = json.load(open(path))
    pool = pd.DataFrame(d["players"]).rename(
        columns={"pos": "position", "med": "median", "ceil": "ceiling"})
    return pool, d["roster"]


def score(model: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    p = np.clip(model / model.sum(), 1e-9, None)
    q = np.clip(actual / actual.sum(), 1e-9, None)
    kl = float((q * np.log(q / p)).sum())
    rho = float(pd.Series(model).rank().corr(pd.Series(actual).rank()))
    return kl, rho


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sport", required=True, choices=["nhl", "mlb", "cfb"])
    ap.add_argument("--pair", nargs=2, action="append", required=True,
                    metavar=("STANDINGS_CSV", "SLATE_JSON"))
    args = ap.parse_args(argv)

    slates = []
    for csv_path, json_path in args.pair:
        own = contest_ownership(csv_path)
        pool, roster = slate_pool(json_path)
        if "CPT" in roster["slots"]:
            sys.exit(f"{json_path} is a showdown board; this fits classic")
        actual = pool["name"].map(own).fillna(0.0).to_numpy(float)
        matched = int(pool["name"].isin(own).sum())
        missing = sorted(set(own) - set(pool["name"]))
        print(f"{Path(json_path).name}: {matched} of {len(own)} drafted "
              f"players matched to the board"
              + (f"; NOT on the board: {', '.join(missing[:8])}" if missing
                 else ""))
        if matched < 0.6 * len(own):
            sys.exit("under 60% of drafted players matched - is this the "
                     "right slate file for that contest?")
        slates.append((pool, roster, actual))

    grid = itertools.product(np.arange(0.0, 1.01, 0.25),      # salary
                             np.arange(0.0, 2.01, 0.25),      # value
                             np.arange(0.3, 1.51, 0.1))       # concentration
    best = None
    for sal, val, conc in grid:
        OWN.CLASSIC_POINTS_WEIGHT, OWN.CLASSIC_SALARY_WEIGHT = 1.0, sal
        OWN.CLASSIC_VALUE_WEIGHT = val
        kls, rhos = [], []
        for pool, roster, actual in slates:
            m = OWN.project(pool, roster, concentration=conc).to_numpy(float)
            kl, rho = score(m, actual)
            kls.append(kl)
            rhos.append(rho)
        k = float(np.mean(kls))
        if best is None or k < best[0]:
            best = (k, sal, val, conc, float(np.mean(rhos)))

    kl, sal, val, conc, rho = best
    fit = {"points": 1.0, "salary": round(sal, 2), "value": round(val, 2),
           "concentration": round(conc, 2), "kl": round(kl, 4),
           "rho": round(rho, 3), "slates": len(slates),
           "contests": [p[0] for p in args.pair]}
    out = Path(__file__).resolve().parent / "data" / f"ownership_fit_{args.sport}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(fit, indent=1) + "\n")
    print(f"\nbest fit over {len(slates)} slate(s): salary {sal:.2f}, value "
          f"{val:.2f}, concentration {conc:.2f}  ->  KL {kl:.3f}, rank "
          f"correlation {rho:.2f}\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
