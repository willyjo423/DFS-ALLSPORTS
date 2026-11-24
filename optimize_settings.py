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

try:
    from engine import contest_roi as CR
except ImportError:                       # NFL repo keeps it at the top level
    import contest_roi as CR

ROOT = Path(__file__).resolve().parent
DUMPS = ROOT / "data" / "backtest_dumps"
OUT = ROOT / "docs" / "data" / "auto_settings.json"

# percentile floor -> multiple of entry fee (a big DraftKings GPP, ~20% paid)
GPP = [(99.9, 50.0), (99.5, 15.0), (99.0, 8.0), (98.0, 4.0), (95.0, 2.5),
       (90.0, 1.8), (80.0, 1.3)]
GPP = [(p, m * 1.75) for p, m in GPP]          # scaled so a random entry returns ~0.85
ENTRIES = [1, 3, 5, 10, 20, 40]
# Contest types the page lets you pick; each is scored with its own payouts.
TIERS = ["cash", "small", "mid", "large"]
SORTS = ["cash", "top1", "port", "proj", "ceil", "roi", "roilow"]
OWNF = ["none", "low", "mid", "high"]
EXPOS = [1.0, 0.6, 0.5, 0.4]
STACKS = {"nhl": [0, 3, 4], "nfl": [0, 3, 4], "cfb": [0, 3], "nba": [0]}
DEFAULT = {1: ("cash", "none", 1.0, 0), 3: ("port", "none", 1.0, 0),
           5: ("port", "none", 0.6, 0), 10: ("port", "none", 0.6, 0),
           20: ("port", "none", 0.5, 0), 40: ("port", "none", 0.4, 0)}


def payout(pct: float, n: int, tier: str | None = None, row: dict | None = None,
           lines: dict | None = None) -> float:
    if tier == "cash":
        return 1.8 if pct >= 55 else 0.0
    if tier is not None:
        return CR.tier_multiple(pct, tier, lines, (row or {}).get("actual"))
    if n == 1:
        return 1.8 if pct >= 55 else 0.0
    for floor, m in GPP:
        if pct >= floor:
            return m
    return 0.0


def roi_of(x: dict, tier: str | None):
    """A lineup's simulated ROI in this contest type (large = the old single ROI)."""
    if tier in (None, "large"):
        return (x.get("roit") or {}).get("large", x.get("roi"))
    return (x.get("roit") or {}).get(tier)


def order(L: list[dict], sort: str, tier: str | None = None) -> list[dict]:
    # lineups built on a player under 50% to play go last (sanity.cap_unlikely)
    r = _order(L, sort, tier)
    return [x for x in r if not x.get("unl")] + [x for x in r if x.get("unl")]


def _order(L: list[dict], sort: str, tier: str | None = None) -> list[dict]:
    # as the page: opt-in scenario lineups never, and the ROI sorts skip
    # scenario lineups (their simulated ROI outruns their real results)
    L = [x for x in L if not x.get("optin")]
    if sort in ("roi", "roilow"):
        L = [x for x in L if x.get("src") != "scenario"]
        has = [x for x in L if roi_of(x, tier) is not None]
        if not has:
            return []                        # night replayed before ROI existed: no answer
        return sorted(has, key=lambda x: (roi_of(x, tier) if sort == "roilow" else -roi_of(x, tier)))
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


def pick(L: list[dict], n: int, rule, tier: str | None = None) -> list[dict]:
    sort, f, expo, st = rule
    ok = own_ok(L, f)
    o = order(L, sort, tier)
    cand = [x for x in o if ok(x) and (x.get("st1") or 0) >= st]
    if len(cand) < n:                       # never leave entries empty
        rest = [x for x in o if x not in cand]
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


def night_value(night: dict, n: int, rule, tier: str | None = None) -> tuple[float, float, float]:
    L = night["lineups"]
    if not L:
        return np.nan, np.nan, np.nan
    if rule[0] in ("roi", "roilow") and not any(roi_of(x, tier) is not None for x in L):
        return np.nan, np.nan, np.nan
    got = pick(L, min(n, len(L)), rule, tier)
    pays = [payout(x["pct"], n, tier, x, night.get("lines")) for x in got]
    return (float(np.mean(pays)) - 1.0,                       # ROI per entry
            float(max(x["pct"] for x in got) >= 99.0),        # any top-1%
            float(max(x["pct"] for x in got)))                # best percentile


def evaluate(nights: dict, sport: str, n: int, tier: str | None = None) -> dict:
    rules = list(itertools.product(SORTS, OWNF, EXPOS, STACKS.get(sport, [0])))
    if n == 1:
        rules = [r for r in rules if r[2] == 1.0]
    if tier == "cash":                     # no ROI for cash games
        rules = [r for r in rules if r[0] not in ("roi", "roilow")]
    keys = sorted(nights)
    res = {}
    for r in rules:
        v = np.array([night_value(nights[k], n, r, tier) for k in keys])
        res[r] = v
    return res


def bootstrap_better(a: np.ndarray, b: np.ndarray, iters: int = 4000, seed: int = 1) -> float:
    """P(mean(a) > mean(b)) over nights resampled with replacement."""
    rng = np.random.default_rng(seed)
    d = a - b
    d = d[~np.isnan(d)]
    if len(d) < 5:
        return 0.0
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
    tier_out = {}
    for sport, variants in data.items():
        report[sport] = {}
        base = variants.get("base") or next(iter(variants.values()))
        nights = {k: v for k, v in base.items() if v.get("lineups")}
        # Past seasons replayed with real DraftKings salaries (nba-hist-<season>,
        # nfl-hist) count as more nights of the same thing. Keys are dates, so
        # the both-halves check splits old seasons from newer ones.
        for vname, vn in variants.items():
            if vname.startswith("hist"):
                nights.update({f"{k}": v for k, v in vn.items() if v.get("lineups")})
        report[sport]["nights"] = len(nights)
        print(f"\n=== {sport.upper()}  {len(nights)} nights ===")
        rules_out[sport] = {}
        tier_out[sport] = {}
        RES = {}
        for tier, n in itertools.product(TIERS, ENTRIES):
            if not nights:
                break
            if n == 1 and tier == "cash":
                print(f"\n  -- {sport.upper()} by contest type --")
            res = evaluate(nights, sport, n, tier)
            RES[(tier, n)] = res
            d0 = DEFAULT[n] if tier != "cash" else ("cash", "none", DEFAULT[n][2], 0)
            dflt = d0 if d0 in res else next(r for r in res if r[0] == d0[0])
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
            adopt = (best != dflt and p >= 0.9 and gain >= 0.05 and holds
                     and int(np.sum(~np.isnan(res[best][:, 0] - res[dflt][:, 0]))) >= 10)
            chosen = best if adopt else dflt
            top = sorted(mean.items(), key=lambda kv: -kv[1])[:5]
            print(f"  {tier:5} {n:>2} entries: current {dflt} ROI {mean[dflt]:+.2f} | best {best} "
                  f"ROI {mean[best]:+.2f} (P better {p:.2f}; early/late halves {halves[0]:.2f}/{halves[1]:.2f}) "
                  f"-> {'ADOPT' if adopt else 'keep'}")
            for r, m in top:
                v = res[r]
                print(f"        {str(r):38} ROI {m:+.2f}  top1 {np.nanmean(v[:, 1]):.0%}  "
                      f"best pct {np.nanmean(v[:, 2]):.1f}")
            report[sport][f"{tier}-{n}"] = {
                "current": {"rule": dflt, "roi": round(mean[dflt], 3),
                            "top1_rate": round(float(np.nanmean(res[dflt][:, 1])), 3)},
                "best": {"rule": best, "roi": round(mean[best], 3), "p_better": round(p, 3),
                         "p_halves": [round(h, 3) for h in halves],
                         "top1_rate": round(float(np.nanmean(res[best][:, 1])), 3)},
                "adopted": adopt}
            nn = int(np.sum(~np.isnan(res[chosen][:, 0])))
            rule = {"sort": chosen[0], "own": chosen[1], "expo": chosen[2], "stack": chosen[3],
                    "evidence": f"{nn} nights, ROI {mean[chosen]:+.2f}"}
            tier_out[sport].setdefault(tier, {})[str(n)] = rule
            # entries-only rules (older pages): 1 entry = cash game, more = large GPP
            if (n == 1 and tier == "cash") or (n > 1 and tier == "large"):
                rules_out[sport][str(n)] = rule
        # Contests near a tier line (e.g. 555 entries, just over "mid"): the
        # rule that does best averaged over BOTH neighbouring tiers' payout
        # curves, under the same guard against the default as a single tier.
        for a_t, b_t in (("small", "mid"), ("mid", "large")):
            for n in ENTRIES:
                ra, rb = RES.get((a_t, n)), RES.get((b_t, n))
                if not ra or not rb:
                    continue
                both = {r: (ra[r][:, 0] + rb[r][:, 0]) / 2 for r in ra if r in rb}
                if not both:
                    continue
                mean = {r: float(np.nanmean(v)) for r, v in both.items()}
                dflt = DEFAULT[n] if DEFAULT[n] in both else next(iter(both))
                best = max(mean, key=mean.get)
                half = len(nights) // 2
                halves = [bootstrap_better(both[best][pt], both[dflt][pt]) if len(both[best][pt]) >= 5 else 0.0
                          for pt in (slice(0, half), slice(half, None))]
                p = bootstrap_better(both[best], both[dflt]) if best != dflt else 0.5
                adopt = (best != dflt and p >= 0.9 and mean[best] - mean[dflt] >= 0.05
                         and min(halves) >= 0.75)
                ch = best if adopt else dflt
                nn = int(np.sum(~np.isnan(both[ch])))
                tier_out[sport].setdefault(f"{a_t}-{b_t}", {})[str(n)] = {
                    "sort": ch[0], "own": ch[1], "expo": ch[2], "stack": ch[3],
                    "evidence": f"{nn} nights, ROI {mean[ch]:+.2f} (avg of {a_t} and {b_t})"}
        # pool variants: best rule's ROI at 5 and 20 entries per variant
        if len([v for v in variants if not v.startswith("hist")]) > 1:
            report[sport]["variants"] = {}
            for vname, vn in variants.items():
                if vname.startswith("hist"):
                    continue
                vn = {k: v for k, v in vn.items() if v.get("lineups")}
                common = sorted(set(vn) & set(nights))
                if len(common) < 5:
                    continue
                row = {}
                for n in (1, 5, 20):
                    r = rules_out[sport].get(str(n))
                    if not r:
                        continue
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
            conf[sport]["tier_rules"] = tier_out[sport]
            conf[sport]["tier_labels"] = CR.TIER_LABEL
        else:
            conf.get(sport, {}).pop("rules", None)
            conf.get(sport, {}).pop("tier_rules", None)
    conf["_optimizer"] = report
    OUT.write_text(json.dumps(conf, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
