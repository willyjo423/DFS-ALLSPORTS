"""NHL team skater strength from the DFS projections, for the forecast site.

Per team per game day: sk_avail = projected DK points of the top 18 skaters
dressing (12 forwards + 6 D is what a lineup ices), so lineup changes and
scratches move it.

History (2025-26, data/nhl_2025.csv.gz): each skater's walk-forward projection
- his DK points per game decayed over earlier games (8.5 goal, 5 assist,
1.5 shot, 1.3 block), scaled by recent ice time - for the skaters who dressed.
This season: the boards the site published (docs/data/archive), with their
statuses.

    python nhl_team_proj_export.py
Writes data/nhl_team_proj.csv.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "nhl_team_proj.csv"
TOP = 18


def history() -> list[dict]:
    path = ROOT / "data" / "nhl_2025.csv.gz"
    if not path.exists():
        return []
    d = pd.read_csv(path)
    d = d[d["position"] != "G"].sort_values(["player_id", "date"]).copy()
    d["dk"] = (8.5 * d["goals"].fillna(0) + 5 * d["assists"].fillna(0)
               + 1.5 * d["shots_on_goal"].fillna(0) + 1.3 * d["blocked_shots"].fillna(0))
    d["rate"] = d["dk"] / d["time_on_ice"].clip(lower=5)
    g = d.groupby("player_id", sort=False)
    d["f_rate"] = g["rate"].transform(lambda s: s.shift(1).ewm(halflife=12, ignore_na=True).mean())
    d["f_toi"] = g["time_on_ice"].transform(lambda s: s.shift(1).ewm(halflife=4, ignore_na=True).mean())
    lg = float(d["rate"].mean())
    d["proj"] = d["f_rate"].fillna(lg) * d["f_toi"].fillna(d["time_on_ice"].median())
    rows = []
    for (day, team), t in d.groupby(["date", "team"]):
        rows.append({"date": day, "team": team,
                     "sk_avail": round(float(np.sort(t["proj"].to_numpy())[::-1][:TOP].sum()), 2),
                     "source": "nhl_2025"})
    return rows


def boards() -> list[dict]:
    best: dict = {}
    for f in glob.glob(str(ROOT / "docs" / "data" / "archive" / "*" / "nhl_dk_classic_*.json")):
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
            if k not in best or (n, f) > best[k][0]:
                best[k] = ((n, f), b)
    rows = []
    for (day, team), (_, b) in best.items():
        ps = [p for p in b["players"] if p["team"] == team and p.get("pos") != "G"
              and str(p.get("status", "")).lower() not in ("out", "o", "ir", "inactive")]
        proj = np.array([float(p.get("mean") or p.get("proj") or p.get("med") or 0) for p in ps])
        rows.append({"date": day, "team": team,
                     "sk_avail": round(float(np.sort(proj)[::-1][:TOP].sum()), 2), "source": "board"})
    return rows


def main() -> int:
    rows = history() + boards()
    df = pd.DataFrame(rows).sort_values(["date", "team"])
    df.to_csv(OUT, index=False)
    print(len(df), "team-days;", df["source"].value_counts().to_dict())
    print(df.groupby("source")["sk_avail"].describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
