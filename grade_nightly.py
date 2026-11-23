"""Grade last night's NBA and NHL boards against the box scores.

    python grade_nightly.py                 # yesterday (Eastern)
    python grade_nightly.py --date 2026-11-18

For each sport, the main slate's lock-time board (the last version written,
which docs/data/archive keeps) is scored against final box scores:

  players  - how far off our final projection, the model alone and the props
             alone were, and the regression of (actual - model) on
             (props - model): the weight the props deserved that night.
  minutes  - NBA only: how far off the minutes were.
  lineups  - our pool's average and best, the top 5 by Cash and by Top 1%,
             and the best legal lineup in hindsight.

Each night is written to docs/data/grades/<sport>/<date>.json, and
docs/data/grades/<sport>/summary.json pools every night graded so far. The
props blend reads its weight from that summary once there are enough nights
(see props_weight.py), so the model tunes itself.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import time
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger("grade_nightly")
HERE = Path(__file__).resolve().parent
DATA = HERE / "docs" / "data"
OUT = DATA / "grades"
ET = timezone(timedelta(hours=-5))


def key(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    for junk in (".", " jr", " sr", " iii", " ii", " iv", "'"):
        s = s.replace(junk, "")
    return " ".join(s.split())


# ------------------------------------------------------------------ boards
def lock_board(sport: str, day: str) -> dict | None:
    """The biggest slate of the night, as last written before lock."""
    seen = {}
    d0 = date.fromisoformat(day)
    for folder in (d0, d0 + timedelta(days=1)):
        for f in (DATA / "archive" / folder.isoformat()).glob(f"{sport}_dk_classic_*.json"):
            seen[f.name] = f
    for f in DATA.glob(f"{sport}_dk_classic_*.json"):
        seen.setdefault(f.name, f)
    best = None
    for f in seen.values():
        try:
            b = json.loads(f.read_text())
        except Exception:                                      # noqa: BLE001
            continue
        locks = b.get("locks")
        if not locks:
            continue
        try:
            on = datetime.fromisoformat(locks).astimezone(ET).date().isoformat()
        except ValueError:
            continue
        if on != day or not b.get("lineups"):
            continue
        if best is None or len(b["players"]) > len(best["players"]):
            best = b
    return best


# ----------------------------------------------------------------- actuals
def nba_actuals(day: str) -> dict:
    import nba_fetch as F
    import pandas as pd
    rows = []
    for g in F.games_on(date.fromisoformat(day)):
        rows += F.parse(F._get(F.SUMMARY, event=g["id"]), g["id"], g["date"])
        time.sleep(0.4)
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    df["dk"] = F.dk_points(df)
    return {key(r.name): {"dk": float(r.dk), "min": float(r.min or 0) if r.played else 0.0,
                          "team": F.canon(r.team)} for r in df.itertuples()}


def nhl_actuals(day: str) -> dict:
    import requests
    ua = {"User-Agent": "Mozilla/5.0"}
    sc = requests.get(f"https://api-web.nhle.com/v1/score/{day}", headers=ua, timeout=30).json()
    out = {}
    for g in sc.get("games", []):
        if g.get("gameState") not in ("OFF", "FINAL"):
            continue
        b = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{g['id']}/boxscore",
                         headers=ua, timeout=30).json()
        for side in ("awayTeam", "homeTeam"):
            team = b[side]["abbrev"]
            st = b["playerByGameStats"][side]
            for grp in ("forwards", "defense"):
                for p in st.get(grp, []):
                    gl, a, s, bl = (p.get("goals", 0), p.get("assists", 0),
                                    p.get("sog", 0), p.get("blockedShots", 0))
                    sh = p.get("shPoints", 0) or 0
                    dk = (8.5 * gl + 5 * a + 1.5 * s + 1.3 * bl + 2 * sh
                          + 3 * (gl >= 3) + 3 * (s >= 5) + 3 * (bl >= 3) + 3 * (gl + a >= 3))
                    out[_nhl_key(p, team)] = {"dk": dk, "team": team}
            for p in st.get("goalies", []):
                dec = p.get("decision")
                sv, ga = p.get("saves") or 0, p.get("goalsAgainst") or 0
                if not dec and not sv:
                    out[_nhl_key(p, team)] = {"dk": 0.0, "team": team}
                    continue
                dk = (0.7 * sv - 3.5 * ga + 6 * (dec == "W") + 2 * (dec == "O")
                      + 4 * (ga == 0 and dec == "W") + 3 * (sv >= 35))
                out[_nhl_key(p, team)] = {"dk": dk, "team": team}
        time.sleep(0.3)
    return out


def _nhl_key(p, team):
    nm = (p.get("name") or {}).get("default", "")       # "J. Hughes"
    first, _, last = nm.partition(". ")
    return f"{team}|{first[:1].lower()}|{key(last or nm)}"


def match(sport: str, p: dict, act: dict):
    if sport == "nba":
        return act.get(key(p["name"]))
    first, _, last = p["name"].partition(" ")
    return act.get(f"{p['team']}|{first[:1].lower()}|{key(last)}")


# ------------------------------------------------------------------- grade
def grade(sport: str, day: str) -> dict | None:
    board = lock_board(sport, day)
    if not board:
        log.info("%s %s: no locked board found", sport, day)
        return None
    act = nba_actuals(day) if sport == "nba" else nhl_actuals(day)
    if not act:
        log.info("%s %s: no final box scores", sport, day)
        return None
    P = board["players"]
    for p in P:
        a = match(sport, p, act)
        p["_act"] = float(a["dk"]) if a else 0.0
        p["_found"] = a is not None
        p["_min"] = (a or {}).get("min")
    played = [p for p in P if p["_found"] and p.get("status") != "out"
              and (sport != "nba" or (p["_min"] or 0) > 0)]

    def mae(rows, k):
        v = [abs(p["_act"] - float(p[k])) for p in rows if p.get(k) is not None]
        return round(float(np.mean(v)), 3) if v else None

    out = {"sport": sport, "date": day, "slate": board.get("label"),
           "draft_group": board.get("draft_group"), "players": len(P),
           "graded": len(played), "mae_final": mae(played, "proj")}
    pr = [p for p in played if p.get("props") is not None and p.get("model")]
    if pr:
        m = np.array([float(p["model"]) for p in pr])
        k = np.array([float(p["props"]) for p in pr])
        y = np.array([p["_act"] for p in pr])
        x, z = k - m, y - m
        out["props"] = {"n": len(pr), "mae_model": round(float(np.abs(y - m).mean()), 3),
                        "mae_props": round(float(np.abs(y - k).mean()), 3),
                        "mae_final": mae(pr, "proj"),
                        "sxy": float(x @ z), "sxx": float(x @ x),
                        "weight": round(float(x @ z / (x @ x)), 3) if x @ x > 0 else None}
    if sport == "nba":
        mm = [(p["min"], p["_min"]) for p in played if p.get("min")]
        if mm:
            a = np.array(mm, float)
            out["minutes"] = {"n": len(a), "mae": round(float(np.abs(a[:, 1] - a[:, 0]).mean()), 2),
                              "bias": round(float((a[:, 1] - a[:, 0]).mean()), 2)}
    late = [p["name"] for p in P if p["_found"] and sport == "nba" and not p["_min"]
            and p.get("status") != "out" and float(p.get("proj") or 0) > 10]
    if late:
        out["late_scratches"] = late
    L = board["lineups"]
    for x in L:
        x["_act"] = sum(P[i]["_act"] for i in x["p"])
    acts = np.array([x["_act"] for x in L])
    out["pool"] = {"n": len(L), "avg": round(float(acts.mean()), 1),
                   "best": round(float(acts.max()), 1)}
    for sort in ("cash", "top1"):
        if any(sort in x for x in L):
            top = sorted(L, key=lambda x: -x.get(sort, 0))[:5]
            out["pool"][f"top5_{sort}"] = [round(x["_act"], 1) for x in top]
    try:
        Q = copy.deepcopy(P)
        for q in Q:
            q["proj"] = q["med"] = q["mean"] = q["ceil"] = q["_act"]
            q["pp"] = 1.0
            q["status"] = "clear" if q["_found"] else "out"
        if sport == "nba":
            from engine import nba_pool as POOL
            o = POOL.optimal_lineup(Q)
        else:
            from engine import lineup_pool as LP
            o = LP.optimal_lineup(Q, board["roster"])
        if o:
            out["pool"]["hindsight"] = round(sum(Q[i]["_act"] for i in o), 1)
            out["hindsight_lineup"] = [P[i]["name"] for i in o]
    except Exception as exc:                                   # noqa: BLE001
        log.warning("hindsight optimal failed: %s", exc)
    # model vs ladder ceilings, for engine/ladder_ceiling's weight
    cr = [[float(p["ceil_model"]), float(p["ceil_props"]), p["_act"]] for p in played
          if p.get("ceil_model") is not None and p.get("ceil_props") is not None]
    if cr:
        out["ceil_rows"] = cr
    miss = sorted(played, key=lambda p: p["_act"] - float(p.get("proj") or 0))
    out["biggest_unders"] = [(p["name"], round(float(p["proj"]), 1), p["_act"]) for p in miss[:5]]
    out["biggest_overs"] = [(p["name"], round(float(p["proj"]), 1), p["_act"]) for p in miss[-5:][::-1]]
    return out


def summarise(sport: str) -> dict:
    nights = []
    for f in sorted((OUT / sport).glob("20*.json")):
        try:
            nights.append(json.loads(f.read_text()))
        except Exception:                                      # noqa: BLE001
            continue
    pr = [n["props"] for n in nights if n.get("props")]
    sxy, sxx = sum(p["sxy"] for p in pr), sum(p["sxx"] for p in pr)
    s = {"sport": sport, "nights": len(nights), "props_nights": len(pr),
         "props_players": sum(p["n"] for p in pr),
         "props_weight": round(sxy / sxx, 3) if sxx > 0 else None,
         "mae_model": _avg(pr, "mae_model"), "mae_props": _avg(pr, "mae_props"),
         "mae_final": _avg(pr, "mae_final"),
         "pool_avg": _avg([n["pool"] for n in nights if n.get("pool")], "avg"),
         "updated": datetime.now(timezone.utc).isoformat(timespec="minutes")}
    s["ceiling"] = tune_ceiling(sport, [r for n in nights for r in n.get("ceil_rows", [])])
    (OUT / sport / "summary.json").write_text(json.dumps(s, indent=1))
    return s


CEIL_LEVEL = {"nhl": 0.97, "nba": 0.90}     # what each board's `ceil` is
CEIL_MIN_ROWS = 300


def tune_ceiling(sport: str, rows: list) -> dict | None:
    """The share of the books' ladder ceiling (vs the model's) whose ceilings
    are beaten at the rate they should be, written to data/ceiling_weight.json
    once there are enough graded players."""
    if not rows or sport not in CEIL_LEVEL:
        return None
    a = np.array(rows, float)
    want = 1 - CEIL_LEVEL[sport]
    rate = lambda w: float((a[:, 2] > (1 - w) * a[:, 0] + w * a[:, 1]).mean())
    grid = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
    best = min(grid, key=lambda w: abs(rate(w) - want))
    out = {"n": len(a), "target": want, "model_rate": round(rate(0.0), 4),
           "props_rate": round(rate(1.0), 4), "best_w": best}
    if len(a) >= CEIL_MIN_ROWS:
        from engine import ladder_ceiling as LC
        try:
            cfg = json.loads(LC.PATH.read_text())
        except Exception:                                      # noqa: BLE001
            cfg = {}
        cfg.setdefault(sport, {})["w"] = best
        cfg[sport]["n"] = len(a)
        LC.PATH.write_text(json.dumps(cfg, indent=1))
        out["applied"] = True
    return out


def _avg(rows, k):
    v = [r[k] for r in rows if r.get(k) is not None]
    return round(float(np.mean(v)), 3) if v else None


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--sport", choices=("nba", "nhl"))
    a = ap.parse_args(argv)
    day = a.date or (datetime.now(ET).date() - timedelta(days=1)).isoformat()
    for sport in ([a.sport] if a.sport else ["nba", "nhl"]):
        try:
            g = grade(sport, day)
        except Exception as exc:                               # noqa: BLE001
            log.exception("%s %s: grading failed (%s)", sport, day, exc)
            continue
        if not g:
            continue
        (OUT / sport).mkdir(parents=True, exist_ok=True)
        (OUT / sport / f"{day}.json").write_text(json.dumps(g, indent=1))
        s = summarise(sport)
        pr = g.get("props") or {}
        print(f"### {sport.upper()} {day} - {g['slate']}")
        print(f"- pool avg {g['pool']['avg']}, best {g['pool']['best']}, "
              f"hindsight {g['pool'].get('hindsight')}; top-5 cash {g['pool'].get('top5_cash')}, "
              f"top-5 top-1% {g['pool'].get('top5_top1')}")
        if pr:
            print(f"- props: {pr['n']} players, MAE model {pr['mae_model']} / props "
                  f"{pr['mae_props']} / final {pr['mae_final']}, weight tonight {pr['weight']}")
        if g.get("minutes"):
            print(f"- minutes MAE {g['minutes']['mae']} (bias {g['minutes']['bias']:+})")
        print(f"- running: {s['props_nights']} nights, props weight {s['props_weight']} "
              f"over {s['props_players']} players")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
