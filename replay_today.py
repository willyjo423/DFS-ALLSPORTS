"""Would today's code (scenario lineups included) have found the best lineup?

    python replay_today.py --sport nba --date 2026-11-22

For every archived classic board locking that day: the published pool as it
was, plus scenario lineups added now from the same board; each lineup scored
with the real box scores; the hindsight optimum; where the best lineup sits
under each sort after the contest sim re-ranks the enlarged pool.
"""
import argparse, copy, glob, json
from datetime import datetime
import numpy as np
import backtest as B, grade_nightly as G, study_optimal as SO
from engine import scenario_pool as SP

ap = argparse.ArgumentParser()
ap.add_argument("--sport", required=True)
ap.add_argument("--date", required=True)
a = ap.parse_args()
act = B.actuals(a.sport, a.date)
seen = {}
for f in sorted(glob.glob(f"docs/data/archive/*/{a.sport}_dk_classic_*.json")):
    b = json.load(open(f))
    if not b.get("locks") or not b.get("lineups"):
        continue
    if datetime.fromisoformat(b["locks"]).astimezone(B.ET).date().isoformat() != a.date:
        continue
    seen[b["draft_group"]] = b          # later archive folders overwrite earlier
for dg, b in seen.items():
    P = b["players"]
    pts, hit = np.zeros(len(P)), 0
    for i, pl in enumerate(P):
        m = G.match(a.sport, pl, act)
        if m is not None:
            pts[i] = m["dk"]; hit += 1
    if hit < 0.5 * len(P):
        print(dg, b.get("label"), "- box scores not complete yet"); continue
    p = copy.deepcopy(b)
    base = [L for L in p["lineups"] if len(L["p"]) == len(p["roster"]["slots"])]
    p["lineups"] = copy.deepcopy(base)
    n = SP.add(p, a.sport, 60)
    SIM = __import__("engine.nba_sim" if a.sport == "nba" else f"engine.{a.sport}_sim", fromlist=["x"])
    SIM.rank_pool(p, sims=8000)
    from engine import contest_roi as CR
    try:
        CR.attach(p, p.pop("_inner", None) or {}, CR.STANDARD_GPP)
    except Exception:
        pass
    L = p["lineups"]
    sc = lambda x: float(pts[x["p"]].sum())
    opt = SO.best_lineup(a.sport, P, p["roster"], pts) or []
    best = max(L, key=sc)
    scen = [x for x in L if x.get("src") == "scenario"]
    ranks = {k: sorted(L, key=lambda x: -((x.get("roit") or {}).get("large", x.get("roi")) or -1e9)
                       if k == "roi" else -(x.get(k) or 0)).index(best) + 1 for k in ("top1", "roi", "proj")}
    print(f"{b.get('label','')[:40]} ({dg}): optimal {pts[opt].sum():.1f} | published pool best "
          f"{max(map(sc, base)):.1f} | scenario best {max(map(sc, scen)) if scen else 0:.1f} ({n} added) | "
          f"best is {'SCENARIO' if best.get('src') == 'scenario' else 'published'}, rank top1 #{ranks['top1']} "
          f"roi #{ranks['roi']} of {len(L)}")
    if scen:
        bs = max(scen, key=sc)
        print("   best scenario:", round(sc(bs), 1), [P[i]["name"] for i in bs["p"]],
              f"shares {len(set(bs['p']) & set(opt))}/{len(opt)} with optimal")
    print("   optimal:", [f"{P[i]['name']} {pts[i]:.1f}" for i in opt])
