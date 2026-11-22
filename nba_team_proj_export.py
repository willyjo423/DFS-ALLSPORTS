"""NBA team strength from the DFS player projections, for nfl-forecast's NBA tab.

Per team per game day:
  dfs_avail   projected DK points of the top 8 players expected to play
  dfs_full    the same over the whole roster that day, known-outs included
  dfs_missing dfs_full - dfs_avail: who is out, in DK points
  n_out       players flagged out

Pre-game knowledge only. History (2017-18 to 2020-21, RotoGuru): the walk-
forward projection nba_hist_backtest uses (recent DK points per minute x
recent minutes, blended with salary by a regression fitted on earlier days),
and "out" = missed his team's previous game, as an injury report would say.
This season: the boards the site actually published (docs/data/archive), with
their own statuses.

    python nba_team_proj_export.py
Writes data/nba_team_proj.csv.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

import nba_hist_backtest as H

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "nba_team_proj.csv"
TOP = 8


def _team_rows(day, team, opp, home, proj, out, source):
    proj = np.asarray(proj, float)
    out = np.asarray(out, bool)
    avail = np.sort(proj[~out])[::-1][:TOP].sum()
    full = np.sort(proj)[::-1][:TOP].sum()
    return {"date": day, "team": team, "opp": opp, "home": home,
            "dfs_avail": round(float(avail), 2), "dfs_full": round(float(full), 2),
            "dfs_missing": round(float(full - avail), 2), "n_out": int(out.sum()),
            "source": source}


def history(seasons=(2017, 2018, 2019, 2020)) -> list[dict]:
    rows = []
    for season in seasons:
        prev = H.features(pd.read_csv(H.HIST / f"nba_dk_{season - 1}.csv"))
        cur = H.features(pd.read_csv(H.HIST / f"nba_dk_{season}.csv"))
        cur = cur[cur["team"].notna()]
        for day in sorted(cur["date"].unique()):
            today = cur[cur["date"] == day]
            beta = H.fit(pd.concat([prev, cur[cur["date"] < day]]))
            proj = H.project(today, beta)
            today = today.assign(proj=np.asarray(proj), out=~today["prev_played"].astype(bool))
            for team, g in today.groupby("team"):
                rows.append(_team_rows(day, team, g["opp"].iloc[0], bool(g["home"].iloc[0]),
                                       g["proj"], g["out"], "rotoguru"))
        print(season, "done", flush=True)
    return rows


def boards() -> list[dict]:
    """This season's published boards: per (ET date, team) the biggest board."""
    best: dict = {}
    for f in glob.glob(str(ROOT / "docs" / "data" / "archive" / "*" / "nba_dk_classic_*.json")):
        try:
            b = json.loads(Path(f).read_text())
        except Exception:                                      # noqa: BLE001
            continue
        lock = b.get("locks") or b.get("lock")
        if isinstance(lock, dict):
            lock = min(lock.values())
        if not lock or not b.get("players"):
            continue
        day = (dt.datetime.fromisoformat(lock) - dt.timedelta(hours=5)).date().isoformat()
        n = len(b["players"])
        for team in {p["team"] for p in b["players"]}:
            k = (day, team)
            # latest copy of the biggest board wins (closest to lock)
            if k not in best or (n, f) > best[k][0]:
                best[k] = ((n, f), b)
    rows = []
    for (day, team), (_, b) in best.items():
        ps = [p for p in b["players"] if p["team"] == team]
        opp = ps[0].get("opp", "")
        game = ps[0].get("game", "")
        home = game.endswith(f"@{team}") or game.endswith(f"@ {team}")
        out = [str(p.get("status", "")).lower() in ("out", "o", "inactive") for p in ps]
        # an out player's projection is zeroed on the board; value him from salary
        proj = [float(p.get("salary") or 0) / 1000 * 5.0 if o else float(p.get("mean") or p.get("proj") or 0)
                for p, o in zip(ps, out)]
        rows.append(_team_rows(day, team, opp, home, proj, out, "board"))
    return rows


def main() -> int:
    rows = history() + boards()
    df = pd.DataFrame(rows).sort_values(["date", "team"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(len(df), "team-days;", df["source"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
