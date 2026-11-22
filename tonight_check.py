"""Tonight's boards, exactly as published: how would each sort have done?

Scores the lineups the page showed (with their published simulated ROI)
against tonight's box scores so far, and compares picking by highest ROI,
lowest ROI, best set, cash rate, top-1% and projection at a few entry counts.
Lines (cash / top 10% / top 1%) come from a simulated field rebuilt on the
same board. Games still in progress count their points so far.

    python tonight_check.py docs/data/archive/2026-11-22/nba_dk_classic_157174.json ...
Writes data/tonight/<file>.json.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

import grade_nightly as G
import replay_night as R

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "tonight"
NS = [1, 3, 5, 10, 20]
SORTS = ["roi", "roilow", "port", "cash", "top1", "proj"]


def order(pool, s):
    if s in ("roi", "roilow"):
        has = [L for L in pool if L.get("roi") is not None]
        return sorted(has, key=lambda L: L["roi"] if s == "roilow" else -L["roi"])
    if s == "port":
        a = sorted([L for L in pool if "port" in L], key=lambda L: L["port"])
        return a + sorted([L for L in pool if "port" not in L], key=lambda L: -(L.get("top1") or 0))
    return sorted(pool, key=lambda L: -(L.get(s) or 0))


def run(path: str) -> dict:
    b = json.loads(Path(path).read_text())
    sport = Path(path).name.split("_")[0]
    if sport == "nba":
        from engine import nba_sim as SIM
        day = (b.get("lock") or "")[:10]
        # lock is UTC; US evening slates belong to the previous calendar day
        import datetime as dt
        lk = dt.datetime.fromisoformat(b["lock"]).astimezone(dt.timezone(dt.timedelta(hours=-5)))
        day = lk.date().isoformat()
        act = G.nba_actuals(day)
        pts = np.array([(G.match("nba", pl, act) or {}).get("dk", 0.0) for pl in b["players"]])
        hit = sum(G.match("nba", pl, act) is not None for pl in b["players"])
    else:
        import cfb_data as D
        from engine import cfb_sim as SIM
        a = R.cfb_actuals(b)
        pts = np.array([a.get(D.primary_key(pl["name"]), 0.0) for pl in b["players"]])
        hit = sum(D.primary_key(pl["name"]) in a for pl in b["players"])
    sim = copy.deepcopy(b)
    inner = SIM.rank_pool(sim)
    field = np.sort(pts[inner["Fd"]].sum(axis=1))
    lines = {k: float(np.quantile(field, q)) for k, q in
             (("top1", 0.99), ("top10", 0.90), ("cash", 0.55))}
    pool = b["lineups"]
    for L in pool:
        L["_act"] = float(pts[L["p"]].sum())
    res = {"board": b.get("label"), "contest": (b.get("contest") or {}).get("name"),
           "players_with_stats": f"{hit}/{len(b['players'])}",
           "lines": {k: round(v, 1) for k, v in lines.items()},
           "pool_best": round(max(L["_act"] for L in pool), 1), "sorts": {}}
    for s in SORTS:
        o = order(pool, s)
        if not o:
            continue
        res["sorts"][s] = {}
        for n in NS:
            expo = 1.0 if n <= 4 else 0.6 if n <= 10 else 0.5
            got = R.greedy(o, n, expo) or o[:n]
            sc = [L["_act"] for L in got]
            res["sorts"][s][n] = {"best": round(max(sc), 1), "avg": round(float(np.mean(sc)), 1),
                                  "cashed": sum(x > lines["cash"] for x in sc),
                                  "top10": sum(x > lines["top10"] for x in sc),
                                  "top1": sum(x > lines["top1"] for x in sc)}
    # rank correlation of published ROI with actual score
    r = [(L["roi"], L["_act"]) for L in pool if L.get("roi") is not None]
    if len(r) > 5:
        x, y = np.argsort(np.argsort([a for a, _ in r])), np.argsort(np.argsort([c for _, c in r]))
        res["roi_rank_corr"] = round(float(np.corrcoef(x, y)[0, 1]), 3)
    return res


def main(argv):
    OUT.mkdir(parents=True, exist_ok=True)
    for path in argv:
        try:
            res = run(path)
        except Exception as exc:                                  # noqa: BLE001
            print(path, "failed:", str(exc)[:200], flush=True)
            continue
        (OUT / Path(path).name).write_text(json.dumps(res, indent=1))
        print(f"\n== {res['board']} | {res['contest']} | stats {res['players_with_stats']} "
              f"| lines {res['lines']} | pool best {res['pool_best']} | roi rank corr {res.get('roi_rank_corr')}")
        for s, byn in res["sorts"].items():
            print(f"  {s:7s} " + "  ".join(
                f"{n}e: best {v['best']} cash {v['cashed']}/{n} t10 {v['top10']} t1 {v['top1']}"
                for n, v in byn.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
