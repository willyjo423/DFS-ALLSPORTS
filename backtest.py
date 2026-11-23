"""Replay archived boards against what actually happened.

One night proves nothing: 2026-11-20 alone said a pool-diversity cap was
worse, which may just have been Harden's 72. This replays every archived
lock-time board for a sport with the CURRENT code, scores each way of picking
lineups against the real box scores, and averages over all the nights.

How a night is scored, without the real contest: the sim's own field (built
to match our ownership estimates) is re-scored with the ACTUAL points. That
gives the night's real winning line (99.9th pct), top-1% line, top-10% and
cash line. A lineup's percentile in that field is what we compare.

    python backtest.py --sport nhl --sims 10000
    python backtest.py --sport nba --variant diverse

Writes docs/data/backtest/<sport>[-<variant>].json and prints a table.
Actual points are cached in data/actuals/<sport>/<date>.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import grade_nightly as G

log = logging.getLogger("backtest")
ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "docs" / "data" / "archive"
ACT = ROOT / "data" / "actuals"
OUT = ROOT / "docs" / "data" / "backtest"
NS = (1, 5, 13)
METHODS = ("proj", "cash", "top1", "port")


# --------------------------------------------------------------- boards
def boards(sport: str) -> dict[str, dict]:
    """{lock day: biggest board locking that day} from the archive. College
    football keeps every slate of 4+ games (keyed day|draft group): a
    Saturday's early, afternoon and night slates are different games."""
    best: dict[str, tuple[int, Path]] = {}
    for f in ARCHIVE.glob(f"*/{sport}_dk_classic_*.json"):
        try:
            b = json.loads(f.read_text())
            day = (datetime.fromisoformat(b["locks"]).astimezone(ET).date().isoformat()
                   if b.get("locks") else f.parent.name)       # CFB boards carry no lock time
        except Exception:                                      # noqa: BLE001
            continue
        if sport == "cfb":
            games = len({p.get("game") for p in b.get("players") or []})
            if games < 4:
                continue
            day = f"{day}|{b.get('draft_group')}"
        n = len(b.get("players") or [])
        # the latest write of the biggest slate; later folders win ties
        if day not in best or (n, str(f)) > (best[day][0], str(best[day][1])):
            best[day] = (n, f)
    today = date.today().isoformat()
    return {d: json.loads(f.read_text()) for d, (n, f) in sorted(best.items())
            if d.split("|")[0] < today}


def cfb_week(day: str, key: str) -> int | None:
    """The CFB week a slate dated `day` belongs to (CFBD calendar), or None if
    that week has not finished yet. A slate published on a Wednesday belongs
    to the coming weekend, never the last one."""
    import cfb_data as D
    path = ACT / "cfb" / f"calendar_{day[:4]}.json"
    if path.exists():
        cal = json.loads(path.read_text())
    else:
        cal = D.cfbd("calendar", key, year=int(day[:4]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cal))
    d = date.fromisoformat(day)
    for w in cal:
        if (w.get("seasonType") or w.get("season_type")) != "regular":
            continue
        end = str(w.get("endDate") or w.get("lastGameStart") or "")[:10]
        if end and date.fromisoformat(end) >= d:
            if date.fromisoformat(end) >= date.today():
                return None
            return int(w["week"])
    return None


def cfb_actuals(board: dict, day: str = "") -> dict:
    """{primary name key: DK points} for the week this board was played, from
    CFBD box scores (cached per week)."""
    import os
    import cfb_data as D
    key = os.environ.get("CFBD_API_KEY", "").strip()
    season = int(day[:4]) if day else 2026
    names = {D.primary_key(p["name"]) for p in board["players"]}
    week = cfb_week(day, key) if day else None
    if week is None:
        return {}
    best = {}
    for w in (week,):
        path = ACT / "cfb" / f"{season}_w{w:02d}.json"
        if path.exists():
            wk = json.loads(path.read_text())
        else:
            if not key:
                continue
            try:
                df = D.flatten_player_games(D.cfbd("games/players", key, year=season,
                                                   week=w, seasonType="regular"))
            except Exception as exc:                           # noqa: BLE001
                log.warning("cfbd week %d: %s", w, str(exc)[:80])
                continue
            if df.empty:
                continue
            df["pts"] = D.fantasy_points(df)
            wk = {}
            for r in df.itertuples():
                wk[D.primary_key(r.name)] = float(r.pts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(wk))
        got = {k: v for k, v in wk.items() if k in names}
        if len(got) > len(best):
            best = got
    return best


def actuals(sport: str, day: str) -> dict:
    path = ACT / sport / f"{day}.json"
    if path.exists():
        return json.loads(path.read_text())
    fn = G.nba_actuals if sport == "nba" else G.nhl_actuals
    for i in range(3):
        try:
            act = fn(day)
            break
        except Exception as exc:                               # noqa: BLE001
            log.warning("actuals %s %s failed (%s)", sport, day, str(exc)[:80])
            time.sleep(5 * (i + 1))
    else:
        return {}
    # Only cache a finished day: a mid-day fetch holds just the early games and,
    # cached, hid the rest of the night from every later grade (2026-11-22).
    if act and day < datetime.now(ET).date().isoformat():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(act))
    return act


# ------------------------------------------------------------ rebuild
def rebuild(sport: str, b: dict, sims: int) -> dict | None:
    """Pool (if the archive has none) and sim ranking, with today's code."""
    p = copy.deepcopy(b)
    if sport == "nba":
        from engine import nba_pool as POOL, nba_sim as SIM
        if not p.get("lineups"):
            p["lineups"] = POOL.safe_build(p)
        if SCENARIOS:
            from engine import scenario_pool as SP
            SP.add(p, sport, SP.N_SCENARIOS, max_seconds=SP.MAX_SECONDS)
        inner = SIM.rank_pool(p, sims=sims)
        from engine import contest_roi as CR
        CR.attach(p, inner, CR.STANDARD_GPP)
        if SCENARIOS:
            SP.trim(p)
        p["_inner"] = inner
        return p if inner else None
    import projection_calibration as PC
    from engine import lineup_pool as LP
    if sport == "cfb":
        from engine import cfb_sim as SIM
        if BLOWOUT:
            apply_blowout(p)
    else:
        from engine import nhl_sim as SIM
    if not p.get("lineups"):
        try:
            PC.apply(p)
        except Exception:                                      # noqa: BLE001
            pass
        p["lineups"] = LP.safe_build(p)
    if not p.get("lineups"):
        return None
    if SCENARIOS:
        from engine import scenario_pool as SP
        SP.add(p, sport, SP.N_SCENARIOS, max_seconds=SP.MAX_SECONDS)
    inner = SIM.rank_pool(p, sims=sims)
    from engine import contest_roi as CR
    CR.attach(p, inner, CR.STANDARD_GPP)
    if SCENARIOS:
        SP.trim(p)
    p["_inner"] = inner
    return p if inner else None


# -------------------------------------------------------------- score
def score_night(sport: str, p: dict, inner: dict, act: dict) -> dict | None:
    pts = np.zeros(len(p["players"]))
    hit = 0
    if sport == "cfb":
        import cfb_data as D
    for i, pl in enumerate(p["players"]):
        if sport == "cfb":
            v = act.get(D.primary_key(pl["name"]))
            a = None if v is None else {"dk": v}
        else:
            a = G.match(sport, pl, act)
        if a is not None:
            pts[i] = a["dk"]
            hit += 1
    if hit < (0.25 if sport == "cfb" else 0.5) * len(p["players"]):
        return None
    field = np.sort(pts[inner["Fd"]].sum(axis=1))
    q = {k: float(np.quantile(field, v)) for k, v in
         (("win", 0.999), ("top1", 0.99), ("top10", 0.90), ("cash", 0.55))}
    pool = p["lineups"]
    score = np.array([pts[L["p"]].sum() for L in pool])
    pct = np.searchsorted(field, score) / len(field) * 100

    def order(m):
        if m == "port":
            k = [i for i, L in enumerate(pool) if "port" in L]
            k.sort(key=lambda i: pool[i]["port"])
            rest = sorted(set(range(len(pool))) - set(k), key=lambda i: -pool[i]["top1"])
            return k + rest
        return sorted(range(len(pool)), key=lambda i: -pool[i].get(m, 0))

    out = {"lines": q, "pool_best_pct": float(pct.max()),
           "pool_best": float(score.max()), "methods": {},
           "games": len({str(pl.get("game") or pl.get("team")) for pl in p["players"]}),
           "dump": [dump_row(p, L, float(sc), float(pc)) for L, sc, pc in zip(pool, score, pct)]}
    for m in METHODS:
        idx = order(m)
        out["methods"][m] = {}
        for n in NS:
            s = score[idx[:n]]
            out["methods"][m][n] = {
                "best_pct": float(pct[idx[:n]].max()),
                "won": bool(s.max() > q["win"]),
                "top1": bool(s.max() > q["top1"]),
                "top10": bool(s.max() > q["top10"]),
                "cash_rate": float((s > q["cash"]).mean()),
            }
    return out


def summarise(nights: dict) -> dict:
    s = {}
    for m in METHODS:
        s[m] = {}
        for n in NS:
            rows = [v["methods"][m][n] for v in nights.values()]
            if not rows:
                continue
            s[m][n] = {k: round(float(np.mean([r[k] for r in rows])) * (100 if k in ("won", "top1", "top10", "cash_rate") else 1), 1)
                       for k in ("best_pct", "won", "top1", "top10", "cash_rate")}
    s["pool_best_pct"] = round(float(np.mean([v["pool_best_pct"] for v in nights.values()])), 1) if nights else None
    return s


def table(sport: str, summ: dict, n_nights: int) -> str:
    lines = [f"### {sport.upper()} replay — {n_nights} nights",
             "| pick by | entries | avg best percentile | top-1% nights | top-10% nights | lineups cashing |",
             "|---|---|---|---|---|---|"]
    for m in METHODS:
        for n, r in summ.get(m, {}).items():
            lines.append(f"| {m} | {n} | {r['best_pct']} | {r['top1']}% | {r['top10']}% | {r['cash_rate']}% |")
    lines.append(f"\nBest lineup in the whole pool, average percentile: {summ.get('pool_best_pct')}")
    return "\n".join(lines)


def dump_row(p: dict, L: dict, actual: float, pct: float) -> dict:
    from collections import Counter
    pl = p["players"]
    sk = [pl[i] for i in L["p"] if pl[i].get("pos") not in ("G", "DST")]
    teams = sorted(Counter(x.get("team") for x in sk).values(), reverse=True) + [0, 0]
    return {"p": L["p"], "proj": L.get("proj"), "ceil": L.get("ceil"), "own": L.get("own"),
            "top1": L.get("top1"), "top10": L.get("top10"), "cash": L.get("cash"),
            "port": L.get("port"), "dup": L.get("dup"), "roi": L.get("roi"),
            "roit": L.get("roit"), "st1": teams[0], "st2": teams[1],
            "actual": round(actual, 2), "pct": round(pct, 2)}


BLOWOUT = False


def apply_blowout(p: dict) -> None:
    """The CFB blowout discount, applied to an archived board's players."""
    import cfb_publish as CP
    for pl in p["players"]:
        loss = CP.blowout_loss(pl.get("spr")) if pl.get("spr") is not None else 0.0
        if loss <= 0.005 or pl.get("blow"):
            continue
        pl["q"] = [v * (1 - loss * (0.5 + q)) for v, q in zip(pl["q"], p["quantiles"])]
        for k, m in (("med", 1.0), ("proj", 1.0), ("mean", 1.0), ("ceil", 1.4)):
            if pl.get(k) is not None:
                pl[k] = pl[k] * (1 - loss * m)
        pl["blow"] = loss


def strip_blowout(p: dict) -> None:
    """Undo the discount on boards published with it."""
    for pl in p["players"]:
        loss = pl.get("blow") or 0.0
        if loss <= 0:
            continue
        pl["q"] = [v / (1 - loss * (0.5 + q)) for v, q in zip(pl["q"], p["quantiles"])]
        for k, m in (("med", 1.0), ("proj", 1.0), ("mean", 1.0), ("ceil", 1.4)):
            if pl.get(k) is not None:
                pl[k] = pl[k] / (1 - loss * m)
        pl["blow"] = 0.0


VARIANTS = {}


def variant(name):
    def deco(fn):
        VARIANTS[name] = fn
        return fn
    return deco


@variant("base")
def _base():
    pass


def _set_default(fn, name, value):
    """Change a keyword default bound at def time (per_objective=PER_OBJECTIVE)."""
    import inspect
    params = [q for q in inspect.signature(fn).parameters.values()
              if q.default is not inspect.Parameter.empty]
    d = list(fn.__defaults__)
    d[[q.name for q in params].index(name)] = value
    fn.__defaults__ = tuple(d)


@variant("bigpool")
def _bigpool():
    """Twice the lineups per objective: more for the set picker to choose from."""
    from engine import lineup_pool as LP, nba_pool as NP
    LP.STACKED_PER_OBJECTIVE *= 2
    LP.PER_TEAM_STACK *= 2
    _set_default(LP.build, "per_objective", LP.PER_OBJECTIVE * 2)
    _set_default(NP.build, "per_objective", NP.PER_OBJECTIVE * 2)


@variant("diverse")
def _diverse():
    """Each new lineup must differ by 3 players instead of 2."""
    from engine import lineup_pool as LP, nba_pool as NP
    LP.MIN_UNIQUE = 3
    NP.MIN_UNIQUE = 3


@variant("blowout")
def _blowout():
    """CFB: every board with the blowout discount (older boards had none)."""
    global BLOWOUT
    BLOWOUT = True


@variant("noblowout")
def _noblowout():
    """CFB: no board with the blowout discount."""
    global STRIP
    STRIP = True


STRIP = False
SCENARIOS = 60      # live since 2026-11-22: replays use what the boards publish


@variant("scenario")
def _scenario():
    """Kept for the 2026-11-22 comparison dumps; same as base now."""


@variant("notail")
def _notail():
    """Simulated ranges without the per-tier stretch (engine/tail_scale)."""
    import os
    os.environ["TAIL_SCALE_OFF"] = "1"


@variant("v2")
def _v2():
    """Same as base (named run, so it does not overwrite the base dump)."""


@variant("scen1")
def _scen1():
    """Scenario lineups as first shipped (60 slates, all kept)."""
    from engine import scenario_pool as SP_
    SP_.MODE = "v1"


@variant("noscenario")
def _noscenario():
    """The pool without scenario lineups (before 2026-11-22)."""
    global SCENARIOS
    SCENARIOS = 0


@variant("bigdiverse")
def _bigdiverse():
    _bigpool()
    _diverse()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", choices=("nba", "nhl", "cfb"), required=True)
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--variant", default="base")
    ap.add_argument("--rebuild-pools", action="store_true",
                    help="rebuild every pool with today's code, not only missing ones")
    ap.add_argument("--max-nights", type=int, default=60)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    VARIANTS[a.variant]()
    nights, t0 = {}, time.time()
    bs = boards(a.sport)
    days = list(bs)[-a.max_nights:]
    for day in days:
        b = copy.deepcopy(bs[day])
        b.pop("lineups", None)           # always today's pool builder and sim
        if STRIP:
            strip_blowout(b)
        act = cfb_actuals(b, day.split('|')[0]) if a.sport == "cfb" else actuals(a.sport, day)
        if not act:
            print(day, "no actuals", flush=True)
            continue
        try:
            p = rebuild(a.sport, b, a.sims)
        except Exception as exc:                               # noqa: BLE001
            print(day, "rebuild failed:", str(exc)[:120], flush=True)
            continue
        if not p:
            print(day, "no pool", flush=True)
            continue
        inner = p.pop("_inner", None) or {}
        if not inner:
            print(day, "no sim", flush=True)
            continue
        r = score_night(a.sport, p, inner, act)
        if r is None:
            print(day, "too few players matched", flush=True)
            continue
        nights[day] = r
        _save(a, nights)                 # every night, so a timeout keeps the work
        m = r["methods"]
        print(f"{day} pool best {r['pool_best_pct']:.1f}pct | top1@13 {m['top1'][13]['best_pct']:.1f} "
              f"port@13 {m['port'][13]['best_pct']:.1f} | {time.time() - t0:.0f}s", flush=True)
    summ = _save(a, nights)
    name = a.sport + ("" if a.variant == "base" else f"-{a.variant}")
    print(table(name, summ, len(nights)))
    return 0


DUMPS = ROOT / "data" / "backtest_dumps"


def _save(a, nights) -> dict:
    summ = summarise(nights)
    OUT.mkdir(parents=True, exist_ok=True)
    DUMPS.mkdir(parents=True, exist_ok=True)
    name = a.sport + ("" if a.variant == "base" else f"-{a.variant}")
    slim = {d: {k: v for k, v in r.items() if k != "dump"} for d, r in nights.items()}
    (OUT / f"{name}.json").write_text(json.dumps(
        {"sport": a.sport, "variant": a.variant, "sims": a.sims,
         "generated_at": datetime.now(ET).isoformat(timespec="seconds"),
         "summary": summ, "nights": slim}, indent=1))
    (DUMPS / f"{name}.json").write_text(json.dumps(
        {d: {"lines": r["lines"], "games": r.get("games"), "lineups": r.get("dump", [])}
         for d, r in nights.items()}, separators=(",", ":")))
    return summ


if __name__ == "__main__":
    raise SystemExit(main())
