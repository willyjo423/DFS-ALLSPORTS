"""Pick lineup settings per sport and entry count from the replay dumps.

Each dump night holds every pool lineup with its sim numbers (top-1%, cash,
set-pick order, ownership, stack shape) and how it actually scored as a
percentile of that night's ownership-matched field. This tries every rule in
a small grid - sort x ownership filter x exposure cap x stack floor - at each
entry count, and scores it by money:

* 1 entry: a double-up (beat the 55th percentile -> 1.8x).
* more:    a top-heavy tournament payout table (below), per entry.

A rule replaces the current default only when it beats it on the bootstrap
over nights at least 90% of the time and by a real margin. Keeping the grid
small is deliberate: with a few dozen nights, a big grid would find a
"winner" that is only noise.

    python optimize_settings.py            # every dump in data/backtest_dumps
Writes docs/data/auto_settings.json -> "rules", which the page's Auto reads.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DUMPS = ROOT / "data" / "backtest_dumps"
OUT = ROOT / "docs" / "data" / "auto_settings.json"

# percentile floor -> multiple of entry fee (a big DraftKings GPP, ~20% paid)
GPP = [(99.9, 50.0), (99.5, 15.0), (99.0, 8.0), (98.0, 4.0), (95.0, 2.5),
       (90.0, 1.8), (80.0, 1.3)]
GPP = [(p, m * 1.75) for p, m in GPP]          # scaled so a random entry returns ~0.85
ENTRIES = [1, 3, 5, 10, 20, 40]
SORTS = ["cash", "top1", "port", "proj", "ceil"]
OWNF = ["none", "low", "mid", "high"]
EXPOS = [1.0, 0.6, 0.5, 0.4]
STACKS = {"nhl": [0, 3, 4], "nfl": [0, 3, 4], "cfb": [0, 3], "nba": [0]}
DEFAULT = {1: ("cash", "none", 1.0, 0), 3: ("port", "none", 1.0, 0),
           5: ("port", "none", 0.6, 0), 10: ("port", "none", 0.6, 0),
           20: ("port", "none", 0.5, 0), 40: ("port", "none", 0.4, 0)}


def payout(pct: float, n: int) -> float:
    if n == 1:
        return 1.8 if pct >= 55 else 0.0
    for floor, m in GPP:
        if pct >= floor:
            return m
    return 0.0


def order(L: list[dict], sort: str) -> list[dict]:
    if sort == "port":
        a = sorted([x for x in L if x.get("port")], key=lambda x: x["port"])
        b = sorted([x for x in L if not x.get("port")], key=lambda x: -(x.get("top1") or 0))
        return a + b
    return sorted(L, key=lambda x: -(x.get(sort) or 0))


def own_ok(L: list[dict], f: str):
    if f == "none":
        return lambda x: True
    o = np.array([x.get("own") or 0 for x in L])
    q20, q50, q80 = np.percentile(o, [20, 50, 80])
    return {"low": lambda x: (x.get("own") or 0) <= q50,
            "mid": lambda x: q20 <= (x.get("own") or 0) <= q80,
            "high": lambda x: (x.get("own") or 0) >= q50}[f]


def pick(L: list[dict], n: int, rule) -> list[dict]:
    sort, f, expo, st = rule
    ok = own_ok(L, f)
    cand = [x for x in order(L, sort) if ok(x) and (x.get("st1") or 0) >= st]
    if len(cand) < n:                       # never leave entries empty
        rest = [x for x in order(L, sort) if x not in cand]
        cand += rest
    c = expo
    while True:
        cap, cnt, out = max(1, int(c * n)), {}, []
        for x in cand:
            if len(out) >= n:
                break
            if any(cnt.get(i, 0) >= cap for i in x["p"]):
                continue
            out.append(x)
            for i in x["p"]:
                cnt[i] = cnt.get(i, 0) + 1
        if len(out) >= min(n, len(cand)) or c >= 1:
            return out
        c = min(1.0, c + 0.1)


def night_value(night: dict, n: int, rule) -> tuple[float, float, float]:
    L = night["lineups"]
    if not L:
        return np.nan, np.nan, np.nan
    got = pick(L, min(n, len(L)), rule)
    pays = [payout(x["pct"], n) for x in got]
    return (float(np.mean(pays)) - 1.0,                       # ROI per entry
            float(max(x["pct"] for x in got) >= 99.0),        # any top-1%
            float(max(x["pct"] for x in got)))                # best percentile


def evaluate(nights: dict, sport: str, n: int) -> dict:
    rules = list(itertools.product(SORTS, OWNF, EXPOS, STACKS.get(sport, [0])))
    if n == 1:
        rules = [r for r in rules if r[2] == 1.0]
    keys = sorted(nights)
    res = {}
    for r in rules:
        v = np.array([night_value(nights[k], n, r) for k in keys])
        res[r] = v
    return res


def bootstrap_better(a: np.ndarray, b: np.ndarray, iters: int = 4000, seed: int = 1) -> float:
    """P(mean(a) > mean(b)) over nights resampled with replacement."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), (iters, len(d)))
    return float((d[idx].mean(axis=1) > 0).mean())


def load() -> dict:
    out = {}
    for f in sorted(DUMPS.glob("*.json")):
        name = f.stem                      # e.g. nhl, nhl-bigpool, cfb-blowout
        sport, _, variant = name.partition("-")
        out.setdefault(sport, {})[variant or "base"] = json.loads(f.read_text())
    return out


def main() -> int:
    data = load()
    report = {}
    rules_out = {}
    for sport, variants in data.items():
        report[sport] = {}
        base = variants.get("base") or next(iter(variants.values()))
        nights = {k: v for k, v in base.items() if v.get("lineups")}
        report[sport]["nights"] = len(nights)
        print(f"\n=== {sport.upper()}  {len(nights)} nights ===")
        rules_out[sport] = {}
        for n in ENTRIES:
            if not nights:
                break
            res = evaluate(nights, sport, n)
            dflt = DEFAULT[n] if DEFAULT[n] in res else next(r for r in res if r[0] == DEFAULT[n][0])
            mean = {r: float(np.nanmean(v[:, 0])) for r, v in res.items()}
            best = max(mean, key=mean.get)
            p = bootstrap_better(res[best][:, 0], res[dflt][:, 0]) if best != dflt else 0.5
            gain = mean[best] - mean[dflt]
            # Out-of-sample guard. Taking the best of ~100 rules on the same
            # nights it is judged on overfits: on 27 CFB slates the in-sample
            # winners lost to the default on the later half they were not
            # picked from. A rule must also beat the default in the earlier
            # AND the later half of the nights separately.
            half = len(nights) // 2
            halves = []
            for part in (slice(0, half), slice(half, None)):
                a, b = res[best][part, 0], res[dflt][part, 0]
                halves.append(bootstrap_better(a, b) if len(a) >= 5 else 0.0)
            holds = min(halves) >= 0.75
            adopt = best != dflt and p >= 0.9 and gain >= 0.05 and len(nights) >= 10 and holds
            chosen = best if adopt else dflt
            top = sorted(mean.items(), key=lambda kv: -kv[1])[:5]
            print(f"  {n:>2} entries: current {dflt} ROI {mean[dflt]:+.2f} | best {best} "
                  f"ROI {mean[best]:+.2f} (P better {p:.2f}; early/late halves {halves[0]:.2f}/{halves[1]:.2f}) "
                  f"-> {'ADOPT' if adopt else 'keep'}")
            for r, m in top:
                v = res[r]
                print(f"        {str(r):38} ROI {m:+.2f}  top1 {np.nanmean(v[:, 1]):.0%}  "
                      f"best pct {np.nanmean(v[:, 2]):.1f}")
            report[sport][n] = {
                "current": {"rule": dflt, "roi": round(mean[dflt], 3),
                            "top1_rate": round(float(np.nanmean(res[dflt][:, 1])), 3)},
                "best": {"rule": best, "roi": round(mean[best], 3), "p_better": round(p, 3),
                         "p_halves": [round(h, 3) for h in halves],
                         "top1_rate": round(float(np.nanmean(res[best][:, 1])), 3)},
                "adopted": adopt}
            rules_out[sport][str(n)] = {"sort": chosen[0], "own": chosen[1],
                                        "expo": chosen[2], "stack": chosen[3],
                                        "evidence": f"{len(nights)} nights, ROI {mean[chosen]:+.2f}"}
        # pool variants: best rule's ROI at 5 and 20 entries per variant
        if len(variants) > 1:
            report[sport]["variants"] = {}
            for vname, vn in variants.items():
                vn = {k: v for k, v in vn.items() if v.get("lineups")}
                common = sorted(set(vn) & set(nights))
                if len(common) < 5:
                    continue
                row = {}
                for n in (1, 5, 20):
                    r = rules_out[sport][str(n)]
                    rule = (r["sort"], r["own"], r["expo"], r["stack"])
                    a = np.array([night_value(vn[k], n, rule)[0] for k in common])
                    b = np.array([night_value(nights[k], n, rule)[0] for k in common])
                    row[n] = {"roi": round(float(np.nanmean(a)), 3),
                              "vs_base": round(float(np.nanmean(a) - np.nanmean(b)), 3),
                              "p_better": round(bootstrap_better(a, b), 3)}
                report[sport]["variants"][vname] = {"nights": len(common), **row}
                print(f"  variant {vname:12} ({len(common)} nights): " +
                      "  ".join(f"{n}e ROI {row[n]['roi']:+.2f} ({row[n]['vs_base']:+.2f} vs base, "
                                f"P {row[n]['p_better']:.2f})" for n in row))
    conf = json.loads(OUT.read_text()) if OUT.exists() else {}
    for sport, r in rules_out.items():
        if report[sport].get("nights", 0) >= 10:      # too few nights: page keeps its defaults
            conf.setdefault(sport, {})["rules"] = r
        else:
            conf.get(sport, {}).pop("rules", None)
    conf["_optimizer"] = report
    OUT.write_text(json.dumps(conf, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
