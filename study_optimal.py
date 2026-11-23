"""How far was our lineup pool from the best possible lineup, and why?

For every night we can replay (archived lock boards for NBA / NHL / CFB, and
NBA 2018-20 from RotoGuru with real salaries and points), with today's pool
builder and sim:

  optimal      the hindsight-best legal lineup
  pool_best    the best lineup our pool contained
  reachable    the best legal lineup using only players that appear somewhere
               in our pool (pool_best < reachable  -> a COMBINATION problem;
               reachable < optimal -> a PLAYER-SELECTION problem)
  overlap      most players any pool lineup shared with the optimal one
  traits       the optimal lineup vs the average pool lineup: projected
               points, ownership, salary spent, cheap players, the largest
               same-team and same-game groups, players from the night's
               highest-projected games

    python study_optimal.py --sources nhl cfb nba nba_hist --sims 3000
Writes docs/data/backtest/optimal_study.json.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

import backtest as B


# ------------------------------------------------------------- eligibility
def eligible(sport: str, p: dict, slot: str) -> bool:
    pos = str(p.get("pos") or "")
    if sport == "nba":
        return slot in (p.get("elig") or [])
    if sport == "nhl":
        if slot == "UTIL":
            return pos != "G"
        return pos == slot or (slot == "W" and pos in ("LW", "RW"))
    if sport == "cfb":
        if slot == "FLEX":
            return pos in ("RB", "WR", "TE")
        if slot == "SFLEX":
            return pos in ("QB", "RB", "WR", "TE")
        return pos == slot or (slot == "WR" and pos == "TE")
    return pos == slot


def best_lineup(sport, players, roster, pts, allowed=None):
    """Hindsight-best legal lineup (scipy MILP: no solver install needed)."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    slots = roster["slots"]
    idx = [i for i in range(len(players)) if (allowed is None or i in allowed)
           and players[i].get("status") != "out"]
    var = [(i, s) for i in idx for s in range(len(slots)) if eligible(sport, players[i], slots[s])]
    if not var:
        return None
    nv = len(var)
    c = -np.array([pts[i] for i, s in var])
    A, lo, hi = [], [], []

    def row(mask, l, h):
        A.append(np.asarray(mask, float)); lo.append(l); hi.append(h)
    for s in range(len(slots)):
        row([ss == s for i, ss in var], 1, 1)
    for i in idx:
        m = [ii == i for ii, s in var]
        if sum(m) > 1:
            row(m, 0, 1)
    row([players[i]["salary"] for i, s in var], 0, roster["salary_cap"])
    if roster.get("max_per_team"):
        for t in {players[i]["team"] for i in idx}:
            row([players[i]["team"] == t and players[i].get("pos") != "G" for i, s in var],
                0, roster["max_per_team"])
    if roster.get("max_per_game"):
        for g in {str(players[i].get("game")) for i in idx}:
            row([str(players[i].get("game")) == g for i, s in var], 0, roster["max_per_game"])
    for pos, m in (roster.get("max_position") or {}).items():
        row([players[i].get("pos") == pos for i, s in var], 0, m)
    res = milp(c, constraints=LinearConstraint(np.vstack(A), lo, hi),
               integrality=np.ones(nv), bounds=Bounds(0, 1), options={"time_limit": 30})
    if res.x is None:
        return None
    return sorted({var[k][0] for k in range(nv) if res.x[k] > 0.5})


# ------------------------------------------------------------------ traits
def traits(sport, players, lineup, ctx):
    P = [players[i] for i in lineup]
    team = Counter(p["team"] for p in P if p.get("pos") != "G")
    game = Counter(str(p.get("game")) for p in P if p.get("pos") != "G")
    return {
        "proj": sum(ctx["proj"][i] for i in lineup),
        "own": sum(float(players[i].get("own") or 0) for i in lineup) * 100,
        "salary": sum(p["salary"] for p in P),
        "cheap": sum(ctx["sal_rank"][i] <= 0.25 for i in lineup),
        "proj_rank": float(np.mean([ctx["pos_rank"][i] for i in lineup])),
        "value_rank": float(np.mean([ctx["val_rank"][i] for i in lineup])),
        "max_team": max(team.values()) if team else 0,
        "max_game": max(game.values()) if game else 0,
        "top_games": sum(str(players[i].get("game")) in ctx["top_games"] for i in lineup),
        "line_mates": max(Counter(p.get("lg") for p in P if p.get("lg")).values() or [0])
        if sport == "nhl" else None,
    }


def context(players):
    proj = np.array([float(p.get("proj") or p.get("med") or 0) for p in players])
    sal = np.array([float(p["salary"]) for p in players])
    df = pd.DataFrame({"pos": [p.get("pos") for p in players], "proj": proj, "sal": sal})
    pos_rank = df.groupby("pos")["proj"].rank(pct=True).to_numpy()
    val_rank = (df["proj"] / (df["sal"] / 1000)).rank(pct=True).to_numpy()
    sal_rank = df["sal"].rank(pct=True, method="average").to_numpy()
    g = pd.Series(proj).groupby([str(p.get("game")) for p in players]).sum().sort_values(ascending=False)
    top = set(g.index[:max(1, len(g) // 4)])
    return {"proj": proj, "pos_rank": pos_rank, "val_rank": val_rank, "sal_rank": sal_rank,
            "top_games": top}


def study_night(sport, p, pts, inner=None):
    players, roster = p["players"], p["roster"]
    opt = best_lineup(sport, players, roster, pts)
    if not opt:
        return None
    pool = p["lineups"]
    in_pool = set(j for L in pool for j in L["p"])
    reach = best_lineup(sport, players, roster, pts, allowed=in_pool) or []
    scores = np.array([pts[L["p"]].sum() for L in pool])
    ov = [len(set(L["p"]) & set(opt)) for L in pool]
    ctx = context(players)
    o_t = traits(sport, players, opt, ctx)
    pool_t = pd.DataFrame([traits(sport, players, L["p"], ctx) for L in pool]).mean().to_dict()
    best_i = int(scores.argmax())
    rank_top1 = sorted(range(len(pool)), key=lambda i: -pool[i].get("top1", 0)).index(best_i) + 1
    rank_roi = sorted(range(len(pool)), key=lambda i: -(pool[i].get("roi") or 0)).index(best_i) + 1
    out = {"optimal": float(pts[opt].sum()), "pool_best": float(scores.max()),
           "reachable": float(pts[reach].sum()) if reach else None,
           "overlap_max": max(ov), "slots": len(roster["slots"]),
           "opt_in_pool": sum(i in in_pool for i in opt), "pool_players": len(in_pool),
           "best_rank_top1": rank_top1, "best_rank_roi": rank_roi, "pool_n": len(pool),
           "opt": o_t, "pool": pool_t}
    if inner and "Fd" in inner:
        field = np.sort(pts[inner["Fd"]].sum(axis=1))
        out["win_line"] = float(np.quantile(field, 0.999))
        out["top1_line"] = float(np.quantile(field, 0.99))
        # the best lineup's percentile in the simulated field
        out["pool_best_pct"] = float(np.searchsorted(field, scores.max()) / len(field) * 100)
    return out


# ------------------------------------------------------------------ sources
def live(sport, sims, max_nights):
    import grade_nightly as G
    res = {}
    for day, b in list(B.boards(sport).items())[-max_nights:]:
        d = day.split("|")[0]
        b = copy.deepcopy(b)
        b.pop("lineups", None)
        act = B.cfb_actuals(b, d) if sport == "cfb" else B.actuals(sport, d)
        if not act:
            continue
        try:
            p = B.rebuild(sport, b, sims)
        except Exception as exc:                               # noqa: BLE001
            print(day, "rebuild failed", str(exc)[:80], flush=True)
            continue
        if not p:
            continue
        inner = p.pop("_inner", None)
        pts, hit = np.zeros(len(p["players"])), 0
        for i, pl in enumerate(p["players"]):
            if sport == "cfb":
                import cfb_data as D
                v = act.get(D.primary_key(pl["name"]))
            else:
                a = G.match(sport, pl, act)
                v = None if a is None else a["dk"]
            if v is not None:
                pts[i] = v
                hit += 1
        if hit < (0.25 if sport == "cfb" else 0.5) * len(p["players"]):
            continue
        r = study_night(sport, p, pts, inner)
        if r:
            res[day] = r
            print(f"{sport} {day}: opt {r['optimal']:.1f} reach {r['reachable']:.1f} "
                  f"pool {r['pool_best']:.1f} overlap {r['overlap_max']}/{r['slots']}", flush=True)
    return res


def nba_hist(sims, seasons, every):
    import nba_hist_backtest as H
    from engine import nba_pool as POOL, nba_sim as SIM, ownership as OWN
    OWN.use_fit("nba")
    res = {}
    for season in seasons:
        prev = H.features(H.scrape(season - 1))
        cur = H.features(H.scrape(season))
        for k, day in enumerate(sorted(cur["date"].unique())):
            if k % every:
                continue
            rows = cur[cur["date"] == day]
            if rows["team"].nunique() < 8:
                continue
            beta = H.fit(pd.concat([prev, cur[cur["date"] < day]]))
            p = H.board(rows, beta)
            p["lineups"] = POOL.safe_build(p)
            if not p["lineups"]:
                continue
            inner = SIM.rank_pool(p, sims=sims)
            pts = np.array([pl["actual"] for pl in p["players"]])
            r = study_night("nba", p, pts, inner)
            if r:
                res[str(day)] = r
                print(f"nba_hist {day}: opt {r['optimal']:.1f} reach {r['reachable']:.1f} "
                      f"pool {r['pool_best']:.1f} overlap {r['overlap_max']}/8", flush=True)
    return res


def summarise(res):
    rows = list(res.values())
    if not rows:
        return {}
    f = lambda k: round(float(np.nanmean([r[k] for r in rows if r.get(k) is not None])), 2)
    out = {"nights": len(rows), "optimal": f("optimal"), "reachable": f("reachable"),
           "pool_best": f("pool_best"),
           "pool_best_share": round(float(np.mean([r["pool_best"] / r["optimal"] for r in rows])), 3),
           "reach_share": round(float(np.mean([r["reachable"] / r["optimal"] for r in rows])), 3),
           "overlap_max": f("overlap_max"), "opt_in_pool": f("opt_in_pool"),
           "best_rank_top1": f("best_rank_top1"), "best_rank_roi": f("best_rank_roi"),
           "pool_n": f("pool_n")}
    if any("win_line" in r for r in rows):
        out["pool_best_beats_win"] = round(float(np.mean([r["pool_best"] > r["win_line"] for r in rows if "win_line" in r])), 3)
        out["optimal_over_win"] = round(float(np.mean([r["optimal"] - r["win_line"] for r in rows if "win_line" in r])), 1)
    keys = [k for k in rows[0]["opt"] if rows[0]["opt"][k] is not None]
    out["traits"] = {k: {"optimal": round(float(np.mean([r["opt"][k] for r in rows])), 2),
                         "pool": round(float(np.mean([r["pool"][k] for r in rows])), 2)} for k in keys}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["nhl", "cfb", "nba", "nba_hist"])
    ap.add_argument("--sims", type=int, default=3000)
    ap.add_argument("--max-nights", type=int, default=60)
    ap.add_argument("--hist-seasons", type=int, nargs="+", default=[2018, 2019])
    ap.add_argument("--every", type=int, default=4)
    a = ap.parse_args(argv)
    out = {"generated": datetime.utcnow().isoformat(), "sources": {}}
    for s in a.sources:
        try:
            res = nba_hist(a.sims, a.hist_seasons, a.every) if s == "nba_hist" else live(s, a.sims, a.max_nights)
        except Exception as exc:                               # noqa: BLE001
            print(s, "failed:", exc, flush=True)
            continue
        out["sources"][s] = {"summary": summarise(res), "nights": res}
        print(s, json.dumps(out["sources"][s]["summary"], indent=1), flush=True)
        B.OUT.mkdir(parents=True, exist_ok=True)
        (B.OUT / f"optimal_study_{'_'.join(a.sources)}.json").write_text(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
