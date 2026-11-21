"""Would today's site have done better last night?

Takes a night's archived lock-time boards, applies the fixes made since
(props re-matched with props_match, the CFB blowout discount), rebuilds the
lineup pool and contest sim with today's code, picks N lineups the way the
page's Auto mode now would, and scores them - and what the page showed that
night - against the real box scores.

    python replay_night.py --date 2026-11-20 --nba 6 --cfb 4
Writes docs/data/replay/<date>.json and prints a summary.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

import backtest as B
import grade_nightly as G

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "data" / "replay"


# ------------------------------------------------------------ selection
def auto_pick(sport: str, payload: dict, n: int, conf: dict) -> tuple[list[dict], str]:
    """The page's Auto mode, in Python (keep in step with applyAuto())."""
    pool = payload["lineups"]
    if n == 1:
        order, why = sorted(pool, key=lambda L: -L.get("cash", 0)), "cash"
    elif any("port" in L for L in pool):
        ported = sorted([L for L in pool if "port" in L], key=lambda L: L["port"])
        rest = sorted([L for L in pool if "port" not in L], key=lambda L: -L["top1"])
        order, why = ported + rest, "best set"
    else:
        order, why = sorted(pool, key=lambda L: -L["top1"]), "top1"
    expo = 1.0 if n <= 4 else 0.6 if n <= 10 else 0.5 if n <= 20 else 0.4
    games = len({p.get("game") or p.get("team") for p in payload["players"]})
    c = conf.get(sport) or {}
    bucket = next((b for b in c.get("buckets", []) if b["games"][0] <= games <= b["games"][1]),
                  c.get("all") if (c.get("all") or {}).get("slates", 0) >= 3 else None)
    lo, hi = 0, 1e9
    if n > 1 and bucket:
        lo, hi = bucket["own_lo"], bucket["own_hi"]
        need = max(n + 5, 2 * n)
        while sum(lo <= L["own"] <= hi for L in pool) < need and (lo > 0 or hi < 400):
            lo, hi = max(0, lo - 10), hi + 10
        why += f", own {lo}-{hi}%"
    ok = [L for L in order if lo <= L["own"] <= hi]
    picked = greedy(ok, n, expo)
    while len(picked) < min(n, len(ok)) and expo < 1:
        expo = min(1.0, expo + 0.1)
        picked = greedy(ok, n, expo)
    return picked, why + f", max exposure {round(expo * 100)}%"


def greedy(order, n, expo):
    cap = max(1, int(expo * n))
    cnt, out = {}, []
    for L in order:
        if len(out) >= n:
            break
        if any(cnt.get(i, 0) >= cap for i in L["p"]):
            continue
        out.append(L)
        for i in L["p"]:
            cnt[i] = cnt.get(i, 0) + 1
    return out


def old_pick(payload: dict, n: int, key: str) -> list[dict]:
    """What the page showed that night: its default 60% exposure cap."""
    order = sorted(payload["lineups"], key=lambda L: -L.get(key, 0))
    return greedy(order, n, 0.6)


# ------------------------------------------------------------- scoring
def score(payload, picks, pts, field):
    out = []
    for L in picks:
        s = float(pts[L["p"]].sum())
        out.append({"players": [payload["players"][i]["name"] for i in L["p"]],
                    "actual": round(s, 2), "proj": L.get("proj"),
                    "own": L.get("own"),
                    "pct": round(float(np.searchsorted(field, s) / len(field) * 100), 1)})
    return out


def summary(rows, lines):
    if not rows:
        return {}
    best = max(r["actual"] for r in rows)
    return {"best": best, "avg": round(float(np.mean([r["actual"] for r in rows])), 1),
            "cashed": sum(r["actual"] > lines["cash"] for r in rows),
            "top10": sum(r["actual"] > lines["top10"] for r in rows),
            "top1": sum(r["actual"] > lines["top1"] for r in rows),
            "best_pct": max(r["pct"] for r in rows)}


# ------------------------------------------------------------------ NBA
def nba(day: str, n: int, conf: dict) -> dict:
    import nba_props as NPR
    import props_match as PM
    import nba_fetch as F
    from engine import nba_pool as POOL, nba_sim as SIM
    old = B.boards("nba")[day]
    act = B.actuals("nba", day)
    p = copy.deepcopy(old)
    p.pop("lineups", None)
    fixed = []
    try:                                     # re-match props the way we do now
        props = NPR.load(day)
        idx = PM.build(props)
        for pl in p["players"]:
            if pl.get("props") is not None and pl.get("model"):
                if PM.match(idx, pl["name"], pl["team"], canon=F.canon) is None:
                    f = pl["model"] / max(pl["proj"], 1e-6)
                    for k in ("proj", "med", "mean", "ceil"):
                        if pl.get(k) is not None:
                            pl[k] = round(pl[k] * f, 2)
                    pl["props"] = None
                    fixed.append(pl["name"])
    except Exception as exc:                                   # noqa: BLE001
        fixed.append(f"(props re-match unavailable: {str(exc)[:60]})")
    p["lineups"] = POOL.safe_build(p)
    inner = SIM.rank_pool(p)
    return finish("nba", day, n, conf, old, p, inner, act, fixed)


# ------------------------------------------------------------------ CFB
def cfb_actuals(board: dict) -> dict:
    import cfb_data as D
    key = os.environ.get("CFBD_API_KEY", "").strip()
    if not key:
        raise SystemExit("CFBD_API_KEY is not set")
    names = {D.primary_key(pl["name"]) for pl in board["players"]}
    week = D.live_week(key, 2026)
    best = {}
    for w in (week, week - 1):
        df = D.flatten_player_games(D.cfbd("games/players", key, year=2026, week=w,
                                           seasonType="regular"))
        if df.empty:
            continue
        df["pts"] = D.fantasy_points(df)
        df["k"] = df["name"].map(D.primary_key)
        got = {r.k: float(r.pts) for r in df.itertuples() if r.k in names}
        if len(got) > len(best):
            best = got
    return best


def cfb(day: str, n: int, conf: dict, path: str) -> dict:
    import cfb_data as D
    import cfb_publish as CP
    from engine import lineup_pool as LP, cfb_sim as SIM
    old = json.loads(Path(path).read_text())
    act_by_key = cfb_actuals(old)
    act = {pl["name"]: act_by_key.get(D.primary_key(pl["name"])) for pl in old["players"]}
    p = copy.deepcopy(old)
    p.pop("lineups", None)
    fixed = []
    for pl in p["players"]:                  # blowout discount, as published now
        loss = CP.blowout_loss(pl.get("spr")) if pl.get("spr") is not None else 0.0
        if loss > 0.02:
            pl["q"] = [v * (1 - loss * (0.5 + q)) for v, q in zip(pl["q"], p["quantiles"])]
            for k, m in (("med", 1.0), ("proj", 1.0), ("mean", 1.0), ("ceil", 1.4)):
                if pl.get(k) is not None:
                    pl[k] = round(pl[k] * (1 - loss * m), 2)
            pl["blow"] = round(loss, 3)
    fixed = sorted({f"{pl['team']} -{round(100 * pl['blow'])}%" for pl in p["players"]
                    if pl.get("blow")})
    p["lineups"] = LP.safe_build(p)
    inner = SIM.rank_pool(p)
    return finish("cfb", day, n, conf, old, p, inner, act, fixed)


# ------------------------------------------------------------ shared
def finish(sport, day, n, conf, old, new, inner, act, fixed):
    def pts_for(payload):
        out = np.zeros(len(payload["players"]))
        for i, pl in enumerate(payload["players"]):
            a = act.get(pl["name"]) if sport == "cfb" else G.match(sport, pl, act)
            v = a if sport == "cfb" else (a["dk"] if a else None)
            out[i] = v or 0.0
        return out
    pts_new = pts_for(new)
    field = np.sort(pts_new[inner["Fd"]].sum(axis=1))
    lines = {k: float(np.quantile(field, v)) for k, v in
             (("top1", 0.99), ("top10", 0.90), ("cash", 0.55))}
    picks, why = auto_pick(sport, new, n, conf)
    pts_old = pts_for(old)
    res = {"sport": sport, "date": day, "entries": n, "label": old.get("label"),
           "fixes_applied": fixed, "lines": {k: round(v, 1) for k, v in lines.items()},
           "now": {"how": "Auto: " + why, "lineups": score(new, picks, pts_new, field)},
           "then_cash": {"how": "sorted by cash, 60% cap",
                         "lineups": score(old, old_pick(old, n, "cash"), pts_old, field)},
           "then_top1": {"how": "sorted by top-1%, 60% cap",
                         "lineups": score(old, old_pick(old, n, "top1"), pts_old, field)},
           "pool_best_now": round(float(max(pts_new[L["p"]].sum() for L in new["lineups"])), 2),
           "pool_best_then": round(float(max(pts_old[L["p"]].sum() for L in old["lineups"])), 2)}
    for k in ("now", "then_cash", "then_top1"):
        res[k]["summary"] = summary(res[k]["lineups"], lines)
    res["chalk_actual"] = sorted(
        [{"name": pl["name"], "team": pl["team"], "own": round(100 * (pl.get("own") or 0)),
          "proj_then": pl.get("proj"), "proj_now": new["players"][i].get("proj"),
          "actual": round(float(pts_old[i]), 1)}
         for i, pl in enumerate(old["players"]) if (pl.get("own") or 0) >= 0.15],
        key=lambda r: -r["own"])
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--nba", type=int, default=0)
    ap.add_argument("--cfb", type=int, default=0)
    ap.add_argument("--cfb-board", default="")
    a = ap.parse_args(argv)
    conf = json.loads((ROOT / "docs/data/auto_settings.json").read_text()) \
        if (ROOT / "docs/data/auto_settings.json").exists() else {}
    out = {}
    if a.nba:
        out["nba"] = nba(a.date, a.nba, conf)
    if a.cfb:
        path = a.cfb_board or next(iter(sorted((ROOT / "docs/data/archive" / a.date)
                                               .glob("cfb_dk_classic_*.json"))))
        out["cfb"] = cfb(a.date, a.cfb, conf, str(path))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.date}.json").write_text(json.dumps(out, indent=1))
    for sport, r in out.items():
        print(f"\n## {sport.upper()} {r['date']} - {r['label']} - {r['entries']} entries")
        print("fixes:", r["fixes_applied"])
        print("lines (sim field, actual points):", r["lines"])
        for k in ("then_cash", "then_top1", "now"):
            s = r[k]["summary"]
            print(f"{k:10} {r[k]['how']}: best {s['best']} (pct {s['best_pct']}), avg {s['avg']}, "
                  f"cashed {s['cashed']}/{r['entries']}, top10 {s['top10']}, top1 {s['top1']}")
            for L in r[k]["lineups"]:
                print(f"     {L['actual']:6.1f}  proj {L['proj']}  own {L['own']}  " + ", ".join(L["players"]))
        print("best lineup in pool: then", r["pool_best_then"], "now", r["pool_best_now"])
        print("chalk (15%+ owned): name team own proj_then proj_now actual")
        for c in r["chalk_actual"]:
            print(f"     {c['name']:24} {c['team']:5} {c['own']:3}%  {c['proj_then']}  {c['proj_now']}  {c['actual']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
