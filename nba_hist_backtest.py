"""NBA lineup backtest over past seasons (2017-18 to 2020-21) with real
DraftKings salaries and points (RotoGuru daily pages).

Each day becomes a classic main slate (all games that day, 4+ games):
* players, positions, salaries and DK points are DraftKings' own; DNPs are on
  the page (0 points, no minutes), so there is no survivorship;
* a player who missed his team's previous game is treated as known-out (what
  an injury report would have said); a surprise scratch stays in and scores 0;
* projection walks forward: recent DK points per minute x recent minutes,
  blended with salary by a regression fitted only on earlier seasons;
* ownership, pool and contest sim are today's code (nba_publish's ownership
  formula, nba_pool, nba_sim with the set picker).

    python nba_hist_backtest.py --season 2018      # 2018-19
Writes data/hist/nba_dk_<season>.csv (cache) and
data/backtest_dumps/nba-hist-<season>.json.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import nba_data as ND
from engine import nba_pool as POOL, nba_sim as SIM, ownership as OWN

ROOT = Path(__file__).resolve().parent
HIST = ROOT / "data" / "hist"
DUMPS = ROOT / "data" / "backtest_dumps"
UA = {"User-Agent": "Mozilla/5.0 (compatible; dfs-model/1.0)"}
URL = "http://rotoguru1.com/cgi-bin/hyday.pl?game=dk&mon={m}&day={d}&year={y}"
ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)


# ---------------------------------------------------------------- scrape
def day_rows(d: date) -> list[dict]:
    t = requests.get(URL.format(m=d.month, d=d.day, y=d.year), headers=UA, timeout=40).text
    out = []
    for r in ROW.findall(t):
        if not re.search(r"\$\d{1,2},\d{3}", r):
            continue
        c = [re.sub(r"<[^>]+>", "", x).replace("&nbsp;", " ").strip() for x in CELL.findall(r)]
        if len(c) < 8:
            continue
        name = c[1].rstrip("^").strip()
        if "," in name:
            last, _, first = name.partition(",")
            name = f"{first.strip()} {last.strip()}"
        opp = c[5].split()
        mins = c[7]
        m = None
        if ":" in mins:
            a, b = mins.split(":")
            m = int(a) + int(b) / 60
        out.append({"date": d.isoformat(), "pos": c[0], "name": name,
                    "starter": c[1].endswith("^"), "pts": float(c[2] or 0),
                    "salary": int(c[3].replace("$", "").replace(",", "")),
                    "team": c[4].upper(), "home": opp[0] == "v" if opp else None,
                    "opp": opp[-1].upper() if opp else "", "min": m})
    return out


def scrape(season: int) -> pd.DataFrame:
    path = HIST / f"nba_dk_{season}.csv"
    if path.exists():
        return pd.read_csv(path)
    d, end = date(season, 10, 1), date(season + 1, 8, 31)
    rows, empty = [], 0
    while d <= end:
        try:
            r = day_rows(d)
        except Exception as exc:                               # noqa: BLE001
            print(d, "failed", str(exc)[:60], flush=True)
            r = []
        if r:
            rows += r
            print(d, len(r), "players", flush=True)
        d += timedelta(days=1)
        time.sleep(1.2)
    df = pd.DataFrame(rows)
    HIST.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


# ------------------------------------------------------------ projection
def features(df: pd.DataFrame) -> pd.DataFrame:
    """Per player-day features from STRICTLY earlier games."""
    df = df.sort_values(["name", "team", "date"]).copy()
    played = df["min"].notna() & (df["min"] > 0)
    df["ppm"] = np.where(played, df["pts"] / df["min"].clip(lower=1), np.nan)
    g = df.groupby(["name"], sort=False)
    df["f_ppm"] = g["ppm"].transform(lambda s: s.shift(1).ewm(halflife=8, ignore_na=True).mean())
    df["f_min"] = g["min"].transform(lambda s: s.shift(1).ewm(halflife=3, ignore_na=True).mean())
    df["f_gp"] = g["min"].transform(lambda s: s.shift(1).notna().cumsum())
    df["f_avg"] = g["pts"].transform(lambda s: s.shift(1).where(s.shift(1) > 0).expanding().mean())
    df["form"] = df["f_ppm"] * df["f_min"]
    # did he play his team's previous game?
    team_days = df.groupby("team")["date"].apply(lambda s: sorted(set(s))).to_dict()
    prev = {}
    for t, days in team_days.items():
        for a, b in zip(days[1:], days[:-1]):
            prev[(t, a)] = b
    played_set = set(zip(df.loc[played, "name"], df.loc[played, "date"]))
    df["prev_played"] = [((n, prev.get((t, d))) in played_set) if (t, d) in prev else True
                         for n, t, d in zip(df["name"], df["team"], df["date"])]
    df["played"] = played
    return df


def fit(train: pd.DataFrame) -> np.ndarray:
    t = train[train["played"] & train["form"].notna() & (train["f_gp"] >= 3)]
    X = np.c_[np.ones(len(t)), t["form"], t["salary"] / 1000]
    beta, *_ = np.linalg.lstsq(X, t["pts"].to_numpy(float), rcond=None)
    return beta


def project(rows: pd.DataFrame, beta: np.ndarray) -> np.ndarray:
    form = rows["form"].fillna(rows["salary"] / 1000 * 4.5)
    return np.maximum(beta[0] + beta[1] * form + beta[2] * rows["salary"] / 1000, 0.5)


# ----------------------------------------------------------------- night
def board(rows: pd.DataFrame, beta: np.ndarray) -> dict:
    rows = rows.copy()
    rows["proj"] = project(rows, beta)
    rows["pp"] = np.where(rows["f_gp"].fillna(0) >= 1, 0.97, 0.85)
    out_known = ~rows["prev_played"]
    players = []
    for r in rows.itertuples():
        med = float(r.proj)
        sd = 0.30 * med + 4.0
        game = f"{r.opp}@{r.team}" if r.home else f"{r.team}@{r.opp}"
        players.append({"name": r.name, "pos": r.pos, "elig": ND.eligibility(r.pos),
                        "team": r.team, "opp": r.opp, "game": game, "salary": int(r.salary),
                        "proj": round(med * r.pp, 2), "med": round(med, 2), "mean": round(med * r.pp, 2),
                        "ceil": round(med + 1.88 * sd, 2), "min": r.f_min, "pp": float(r.pp),
                        "status": "out" if out_known.loc[r.Index] else "clear",
                        "dkppg": None if pd.isna(r.f_avg) else float(r.f_avg),
                        "model": None, "actual": float(r.pts)})
    pl = pd.DataFrame(players)
    pool = pd.DataFrame({"position": "ALL", "salary": pl["salary"], "mean": pl["mean"],
                         "median": pl["med"], "ceiling": pl["ceil"], "p_play": pl["pp"],
                         "playing": np.where(pl["status"] == "out", "out", "yes"),
                         "dkppg": pd.to_numeric(pl["dkppg"], errors="coerce"),
                         "implied": np.nan})
    own = OWN._allocate(OWN.appeal(pool, classic=True).to_numpy(float), 8.0,
                        OWN.CLASSIC_CONCENTRATION)
    for d, o in zip(players, own):
        d["own"] = round(float(o), 5)
    return {"sport": "nba", "roster": {"slots": POOL.SLOTS, "salary_cap": 50_000},
            "field_size": 100_000, "players": players}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True, help="start year, 2018 = 2018-19")
    ap.add_argument("--sims", type=int, default=8000)
    ap.add_argument("--every", type=int, default=1, help="use every Nth slate day")
    a = ap.parse_args(argv)
    OWN.use_fit("nba")
    prev = features(scrape(a.season - 1))
    cur = features(scrape(a.season))
    out_path = DUMPS / f"nba-hist-{a.season}.json"
    dumps = json.loads(out_path.read_text()) if out_path.exists() else {}
    days = sorted(cur["date"].unique())
    for k, day in enumerate(days):
        if k % a.every or day in dumps:
            continue
        rows = cur[cur["date"] == day]
        if rows["team"].nunique() < 8:                    # 4+ games
            continue
        train = pd.concat([prev, cur[cur["date"] < day]])
        beta = fit(train)
        p = board(rows, beta)
        p["lineups"] = POOL.safe_build(p)
        if not p["lineups"]:
            continue
        inner = SIM.rank_pool(p, sims=a.sims)
        if not inner:
            continue
        from engine import contest_roi as CR
        CR.attach(p, inner, CR.STANDARD_GPP)
        pts = np.array([pl["actual"] for pl in p["players"]])
        field = np.sort(pts[inner["Fd"]].sum(axis=1))
        lines = {kk: float(np.quantile(field, v)) for kk, v in
                 (("win", 0.999), ("top1", 0.99), ("top10", 0.90), ("cash", 0.55))}
        import backtest as B
        rows_out = []
        for L in p["lineups"]:
            sc = float(pts[L["p"]].sum())
            rows_out.append(B.dump_row(p, L, sc, float(np.searchsorted(field, sc) / len(field) * 100)))
        dumps[day] = {"lines": lines, "games": rows["team"].nunique() // 2, "lineups": rows_out}
        DUMPS.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(dumps, separators=(",", ":")))
        best = max(rows_out, key=lambda r: r["actual"])
        print(f"{day}: {rows['team'].nunique() // 2} games, pool best {best['actual']:.1f} "
              f"({best['pct']:.1f} pct), beta {np.round(beta, 2).tolist()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
