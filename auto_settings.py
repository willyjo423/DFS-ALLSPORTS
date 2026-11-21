"""Lineup settings learned from history, by slate size, for the page's Auto mode.

For every archived lock-time board that has box scores, this solves the ten
lineups that ACTUALLY scored best that night (the projection swapped for the
real points) and records their total ownership as we estimated it before
lock, along with the slate's number of games. Grouped by slate size, that
says which ownership range the nights' best lineups really came from: on a
small slate the winners are usually chalky, on a big one they are not.

It also reads the replay backtest (docs/data/backtest/<sport>.json) for which
sort found the best lineup at each entry count.

    python auto_settings.py            # every sport with archived boards
Writes docs/data/auto_settings.json, which the page reads.
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import numpy as np

import backtest as B
import grade_nightly as G

log = logging.getLogger("auto_settings")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "data" / "auto_settings.json"
BUCKETS = ((2, 3), (4, 6), (7, 9), (10, 99))
TOP_K = 10
MIN_SLATES = 5


def games_on(board: dict) -> int:
    return len({str(p.get("game") or p.get("team")) for p in board["players"]})


def hindsight_owns(sport: str, board: dict, act: dict) -> list[float] | None:
    pl = copy.deepcopy(board["players"])
    hit = 0
    for p in pl:
        a = G.match(sport, p, act)
        pts = float(a["dk"]) if a else 0.0
        hit += a is not None
        p["proj"] = p["med"] = p["ceil"] = max(pts, 0.01) if a else 0.0
        p["pp"] = 1.0
        p["status"] = "clear" if a else "out"
    if hit < 0.5 * len(pl):
        return None
    if sport == "nba":
        from engine import nba_pool as NP
        pool = NP.build(pl, per_objective=TOP_K, lambdas=(0.0,), bases=("median",))
    else:
        from engine import lineup_pool as LP
        pool = LP.build(pl, board["roster"], lambdas=(0.0,), per_objective=TOP_K, stacks=False)
    own = [100 * sum(board["players"][i].get("own") or 0 for i in L["p"]) for L in pool]
    stacks = [stack_shape(board["players"], L["p"]) for L in pool] if sport == "nhl" else []
    return (own, stacks) if own else None


def stack_shape(players, idx) -> dict:
    """Biggest team stack, second stack and biggest same-line group among
    the skaters of one lineup (hockey)."""
    from collections import Counter
    sk = [players[i] for i in idx if players[i].get("pos") != "G"]
    t = sorted(Counter(p.get("team") for p in sk).values(), reverse=True) + [0, 0]
    ln = Counter(p.get("lg") for p in sk if p.get("lg"))
    return {"first": t[0], "second": t[1], "line": max(ln.values()) if ln else 0}


def learn(sport: str) -> dict:
    rows = []
    for day, b in B.boards(sport).items():
        act = B.actuals(sport, day)
        if not act:
            continue
        try:
            got = hindsight_owns(sport, b, act)
            owns, shapes = got if got else (None, [])
        except Exception as exc:                               # noqa: BLE001
            log.warning("%s %s: %s", sport, day, str(exc)[:100])
            continue
        if owns:
            rows.append({"day": day, "games": games_on(b), "owns": owns, "shapes": shapes})
            print(f"{sport} {day} games {games_on(b)}: best lineups' total own "
                  f"{np.median(owns):.0f}% (range {min(owns):.0f}-{max(owns):.0f})", flush=True)

    def summarise(rs):
        allo = [o for r in rs for o in r["owns"]]
        out = {"slates": len(rs),
               "own_lo": round(float(np.percentile(allo, 15))),
               "own_hi": round(float(np.percentile(allo, 85))),
               "own_median": round(float(np.median(allo)))}
        sh = [x for r in rs for x in r.get("shapes") or []]
        if sh:
            share = lambda k, n: float(np.mean([x[k] >= n for x in sh]))
            # The biggest stack that at least half of the nights' best
            # lineups had - a floor the page can require without excluding
            # the shape that usually wins.
            out["stack"] = max([n for n in (2, 3, 4, 5) if share("first", n) >= 0.5], default=0)
            out["stack2"] = 2 if share("second", 2) >= 0.5 else 0
            out["line"] = max([n for n in (2, 3) if share("line", n) >= 0.5], default=0)
            out["stack_shares"] = {f"first>={n}": round(share("first", n), 2) for n in (2, 3, 4, 5)}
            out["stack_shares"].update({"second>=2": round(share("second", 2), 2),
                                        "line>=2": round(share("line", 2), 2),
                                        "line>=3": round(share("line", 3), 2)})
        return out

    out = {"slates": len(rows), "buckets": []}
    if rows:
        out["all"] = summarise(rows)
    for lo, hi in BUCKETS:
        rs = [r for r in rows if lo <= r["games"] <= hi]
        if len(rs) >= MIN_SLATES:
            out["buckets"].append({"games": [lo, hi], **summarise(rs)})

    bt = ROOT / "docs" / "data" / "backtest" / f"{sport}.json"
    if bt.exists():
        s = json.loads(bt.read_text())
        summ, n_n = s["summary"], len(s.get("nights") or {})
        if n_n >= 10:
            best = {}
            for n in ("1", "5", "13"):
                cands = {m: summ[m][n] for m in ("cash", "top1", "port") if n in summ.get(m, {})}
                if cands:
                    key = (lambda r: r["cash_rate"]) if n == "1" else (lambda r: (r["top1"], r["best_pct"]))
                    best[n] = max(cands, key=lambda m: key(cands[m]))
            out["sort_by_entries"] = best
            out["backtest_nights"] = n_n
    return out


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    res = {}
    for sport in ("nba", "nhl"):
        res[sport] = learn(sport)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
