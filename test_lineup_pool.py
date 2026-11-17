"""engine/lineup_pool.py: every lineup in the pool is one DraftKings accepts.

Runs as a script: `python test_lineup_pool.py`. Uses the committed slate files,
no network. A small pool (2 objectives x 3 lineups) to keep it fast.
"""
from __future__ import annotations

import collections
import glob
import json
import sys

from engine import lineup_pool as LP

FAILS = []


def check(cond, what):
    print(("  ok    " if cond else "  FAIL  ") + what)
    if not cond:
        FAILS.append(what)


files = (sorted(glob.glob("docs/data/nhl_dk_classic_*.json"))[:1]
         + sorted(glob.glob("docs/data/cfb_dk_classic_*.json"))[:1]
         + sorted(glob.glob("docs/data/archive/*/mlb_dk_classic_*.json"))[-3:-2])
for f in files:
    d = json.load(open(f))
    P, R = d["players"], d["roster"]
    pool = LP.build(P, R, lambdas=(0.0, 1.5), per_objective=3)
    print(f)
    check(len(pool) >= 4, f"{len(pool)} lineups built")
    flex = {"UTIL", "FLEX", "SFLEX"}
    need = collections.Counter(s for s in R["slots"] if s not in flex)
    legal = True
    for L in pool:
        ps = [P[i] for i in L["p"]]
        c = collections.Counter(p["pos"] for p in ps)
        legal &= len(ps) == len(R["slots"]) and len(set(L["p"])) == len(ps)
        legal &= sum(p["salary"] for p in ps) <= R["salary_cap"]
        legal &= all(c[k] >= v for k, v in need.items())
        for k, v in (R.get("max_position") or {}).items():
            legal &= c[k] <= v
        if R.get("max_per_team"):
            t = collections.Counter(p["team"] for p in ps
                                    if p["pos"] not in ("G", "P"))
            legal &= max(t.values()) <= R["max_per_team"]
        if R.get("min_skater_teams"):
            legal &= len({p["team"] for p in ps if p["pos"] != "G"}) >= R["min_skater_teams"]
        if R.get("min_games"):
            legal &= len({p["game"] for p in ps}) >= R["min_games"]
    check(legal, "every lineup legal: size, cap, positions, team/game rules")
    first = [L for L in pool if L["src"] == "median lam=0.0"][0]
    contrarian = [L for L in pool if L["src"] == "median lam=1.5"][0]
    check(first["proj"] >= contrarian["proj"] - 1e-6,
          "pure-projection lineup projects at least as high as the penalised one")
    check(contrarian["own"] <= first["own"] + 1e-6,
          "ownership-penalised lineup is no more owned than the pure one")
    check(all(len(set(a["p"]) & set(b["p"])) <= len(R["slots"]) - LP.MIN_UNIQUE
              for i, a in enumerate(pool) for b in pool[:i]),
          f"lineups differ by at least {LP.MIN_UNIQUE} players")

print()
if FAILS:
    print(f"{len(FAILS)} FAILED")
    sys.exit(1)
print("all passed")
