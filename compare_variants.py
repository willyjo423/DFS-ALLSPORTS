"""Paired comparison of two backtest dumps (same nights, different code).

    python compare_variants.py BASE.json [BASE2.json ...] -- VARIANT.json [...]

For each contest size (small / mid / large tournaments) and entry count, picks
lineups the way the page does (by highest ROI and by top-1% rate, default
exposure caps) on every night both dumps share, and reports the average
return per entry, how often a pick reached the top 1%, and the best pick's
percentile - with a paired bootstrap of whether the variant is better.
"""
from __future__ import annotations

import json
import sys

import numpy as np

import optimize_settings as O

ENTRIES = (1, 5, 20)
TIERS = ("small", "mid", "large")


def load(files):
    out = {}
    for f in files:
        out.update({k: v for k, v in json.load(open(f)).items() if v.get("lineups")})
    return out


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    k = argv.index("--")
    base, var = load(argv[:k]), load(argv[k + 1:])
    keys = sorted(set(base) & set(var))
    print(f"{len(keys)} nights in both")
    pb = np.mean([max(x["pct"] for x in base[d]["lineups"]) for d in keys])
    pv = np.mean([max(x["pct"] for x in var[d]["lineups"]) for d in keys])
    ob = np.mean([max(x["actual"] for x in base[d]["lineups"]) for d in keys])
    ov = np.mean([max(x["actual"] for x in var[d]["lineups"]) for d in keys])
    print(f"best lineup in pool: pct {pb:.2f} -> {pv:.2f}, points {ob:.1f} -> {ov:.1f}")
    print(f"pool size: {np.mean([len(base[d]['lineups']) for d in keys]):.0f} -> "
          f"{np.mean([len(var[d]['lineups']) for d in keys]):.0f}")
    print(f"{'tier':6}{'n':>3} {'sort':5} {'ROI base':>9} {'ROI var':>8} {'top1 b':>7} {'top1 v':>7} "
          f"{'best b':>7} {'best v':>7} {'P(var>base)':>12}")
    rows = []
    for tier in TIERS:
        for n in ENTRIES:
            for sort in ("roi", "top1"):
                rule = (sort, "none", 1.0 if n == 1 else O.DEFAULT[n][2], 0)
                a = np.array([O.night_value(base[d], n, rule, tier) for d in keys])
                b = np.array([O.night_value(var[d], n, rule, tier) for d in keys])
                ok = ~np.isnan(a[:, 0]) & ~np.isnan(b[:, 0])
                a, b = a[ok], b[ok]
                if not len(a):
                    continue
                p = O.bootstrap_better(b[:, 0], a[:, 0])
                rows.append({"tier": tier, "n": n, "sort": sort, "nights": int(ok.sum()),
                             "roi_base": float(a[:, 0].mean()), "roi_var": float(b[:, 0].mean()),
                             "p": p})
                print(f"{tier:6}{n:>3} {sort:5} {a[:, 0].mean():>+9.3f} {b[:, 0].mean():>+8.3f} "
                      f"{a[:, 1].mean():>7.1%} {b[:, 1].mean():>7.1%} {a[:, 2].mean():>7.1f} "
                      f"{b[:, 2].mean():>7.1f} {p:>12.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
