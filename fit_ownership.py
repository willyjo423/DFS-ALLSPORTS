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


CONTESTS = Path(__file__).resolve().parent / "data" / "contests"
DOCS = Path(__file__).resolve().parent / "docs" / "data"


def all_boards(sport: str) -> list[Path]:
    return sorted(DOCS.glob(f"{sport}_dk_classic_*.json")) + sorted(
        DOCS.glob(f"archive/*/{sport}_dk_classic_*.json"))


def best_board(own: dict, sport: str):
    """The published board this contest was played on: the one holding the
    largest share of the contest's drafted players (latest copy wins ties)."""
    from availability import name_keys
    want = {k for n in own for k in name_keys(n)}
    best = None
    for f in all_boards(sport):
        try:
            d = json.loads(f.read_text())
        except Exception:                                      # noqa: BLE001
            continue
        have = {k for p in d.get("players", []) for k in name_keys(p["name"])}
        share = len(want & have) / max(len(want), 1)
        if best is None or share >= best[0]:
            best = (share, f, d)
    return best


def board_pool(d: dict, sport: str) -> pd.DataFrame:
    pool = pd.DataFrame(d["players"]).rename(
        columns={"pos": "position", "med": "median", "ceil": "ceiling", "pp": "p_play"})
    if "ppg" in pool and "dkppg" not in pool:
        pool["dkppg"] = pool["ppg"]
    if "itt" in pool and "implied" not in pool:
        pool["implied"] = pool["itt"]
    pool["playing"] = np.where(pool.get("status", "clear") == "out", "out", "yes")
    return pool


def model_own(pool: pd.DataFrame, roster: dict, sport: str, conc: float) -> np.ndarray:
    if sport == "nba":                      # multi-position: one pool, eight slots
        sc = OWN.appeal(pool.assign(position="ALL"), classic=True).to_numpy(float)
        return OWN._allocate(sc, float(len(roster["slots"])), conc)
    return OWN.project(pool, roster, concentration=conc).to_numpy(float)


WEIGHTS = ("salary", "value", "dkppg", "implied")
GRID = {"salary": [0.0, 0.25, 0.5, 0.75, 1.0],
        "value": [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
        "dkppg": [0.0, 0.25, 0.5, 0.75, 1.0, 1.5],
        "implied": [0.0, 0.25, 0.5, 0.75, 1.0],
        "concentration": [round(x, 2) for x in np.arange(0.3, 1.61, 0.1)]}


def evaluate(w: dict, slates, sport) -> tuple[float, float, float]:
    OWN.CLASSIC_POINTS_WEIGHT = 1.0
    OWN.CLASSIC_SALARY_WEIGHT, OWN.CLASSIC_VALUE_WEIGHT = w["salary"], w["value"]
    OWN.CLASSIC_DKPPG_WEIGHT, OWN.CLASSIC_IMPLIED_WEIGHT = w["dkppg"], w["implied"]
    kls, rhos, maes = [], [], []
    for pool, roster, actual in slates:
        m = model_own(pool, roster, sport, w["concentration"])
        kl, rho = score(m, actual)
        kls.append(kl); rhos.append(rho)
        maes.append(float(np.abs(m - actual).mean() * 100))
    return float(np.mean(kls)), float(np.mean(rhos)), float(np.mean(maes))


def fit(sport: str, files: list[Path]) -> dict | None:
    slates, used = [], []
    for csv_path in files:
        own = contest_ownership(str(csv_path))
        hit = best_board(own, sport)
        if not hit or hit[0] < 0.6:
            print(f"{csv_path.name}: no published {sport} board holds 60% of its "
                  f"players (best {0 if not hit else hit[0]:.0%}) - skipped")
            continue
        share, path, d = hit
        if "CPT" in d["roster"]["slots"]:
            print(f"{csv_path.name}: showdown board - skipped")
            continue
        from availability import name_keys
        lookup = {k: v for n, v in own.items() for k in name_keys(n)}
        pool = board_pool(d, sport)
        actual = np.array([next((lookup[k] for k in name_keys(n) if k in lookup), 0.0)
                           for n in pool["name"]])
        print(f"{csv_path.name}: matched {path.name} ({share:.0%} of drafted players)")
        slates.append((pool, d["roster"], actual))
        used.append(csv_path.name)
    if not slates:
        return None
    w = {"salary": OWN.CLASSIC_SALARY_WEIGHT, "value": OWN.CLASSIC_VALUE_WEIGHT,
         "dkppg": 0.0, "implied": 0.0, "concentration": OWN.CLASSIC_CONCENTRATION}
    before = evaluate(w, slates, sport)
    best = before
    for _round in range(3):
        for k, vals in GRID.items():
            for v in vals:
                q = dict(w, **{k: v})
                r = evaluate(q, slates, sport)
                if r[0] < best[0] - 1e-5:
                    w, best = q, r
    return {"points": 1.0, **{k: round(float(v), 2) for k, v in w.items()},
            "kl": round(best[0], 4), "rho": round(best[1], 3),
            "mae_pct": round(best[2], 2),
            "before": {"rho": round(before[1], 3), "mae_pct": round(before[2], 2)},
            "slates": len(slates), "contests": used}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sport", choices=["nhl", "mlb", "cfb", "nba"])
    ap.add_argument("--auto", action="store_true",
                    help="fit every sport from data/contests/<sport>/*.csv")
    args = ap.parse_args(argv)
    sports = [args.sport] if args.sport else ["nba", "nhl", "cfb", "mlb"]
    wrote = 0
    for sp in sports:
        files = sorted((CONTESTS / sp).glob("*.csv"))
        if not files:
            continue
        import importlib
        importlib.reload(OWN)                # each sport starts from the defaults,
        OWN.use_fit(sp)                      # not the last sport's grid search
        res = fit(sp, files)
        if not res:
            continue
        out = Path(__file__).resolve().parent / "data" / f"ownership_fit_{sp}.json"
        # Only replace the formula when the refit is clearly better than what
        # the board used (a fit that matches it is churn, not information).
        gain = res["rho"] - res["before"]["rho"]
        drop = res["before"]["mae_pct"] - res["mae_pct"]
        if gain < 0.02 and drop < 0.1:
            print(f"{sp.upper()}: refit no better (rank corr {res['before']['rho']:.2f} -> "
                  f"{res['rho']:.2f}) - current formula kept")
            continue
        try:                                 # keep hand-fitted per-position settings
            old = json.loads(out.read_text())
            if old.get("position_concentration") and "position_concentration" not in res:
                res["position_concentration"] = old["position_concentration"]
        except (OSError, ValueError):
            pass
        out.write_text(json.dumps(res, indent=1) + "\n")
        wrote += 1
        print(f"\n{sp.upper()}: {res['slates']} slate(s). rank corr "
              f"{res['before']['rho']:.2f} -> {res['rho']:.2f}, average miss "
              f"{res['before']['mae_pct']:.2f} -> {res['mae_pct']:.2f} pts of ownership. "
              f"Weights: salary {res['salary']}, value {res['value']}, DK avg "
              f"{res['dkppg']}, team total {res['implied']}, concentration "
              f"{res['concentration']}")
    if not wrote:
        print("no contest exports found under data/contests/<sport>/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
