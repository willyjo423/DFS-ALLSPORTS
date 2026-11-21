"""Do heavy CFB favourites' starters really score less than expected?

The blowout discount in cfb_publish.py was set from a rule of thumb. This
measures it on 2021-2025: for every fantasy-relevant player-game, compare
actual DK points with his trailing average, and ask how that ratio moves with
the pregame spread AFTER accounting for the implied team total (which the
model already uses - a favourite is expected to score more, and the question
is only whether its starters fall short of that because they sit late).

    ratio = actual / trailing average
    fit:  ratio ~ a + b * (implied team total / trailing team points) + spread buckets

Writes data/cfb_blowout_fit.json with the bucket effects (relative to a
pick'em game, with standard errors) and the discount the data supports.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import cfb_data as D

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "cfb_lines"
BUCKETS = [(-99, -24.5), (-24.5, -17.5), (-17.5, -10.5), (-10.5, -3.5), (-3.5, 3.5),
           (3.5, 10.5), (10.5, 17.5), (17.5, 24.5), (24.5, 31.5), (31.5, 99)]
BASE = (-3.5, 3.5)


def lines(season: int, key: str) -> pd.DataFrame:
    path = CACHE / f"{season}.json"
    if path.exists():
        raw = json.loads(path.read_text())
    else:
        raw = D.cfbd("lines", key, year=season, seasonType="regular")
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw))
    rows = []
    for g in raw:
        ls = [l for l in (g.get("lines") or []) if l.get("spread") is not None]
        if not ls:
            continue
        sp = float(np.median([float(l["spread"]) for l in ls]))
        ous = [float(l["overUnder"]) for l in ls if l.get("overUnder") is not None]
        tot = float(np.median(ous)) if ous else np.nan
        home, away = g.get("homeTeam"), g.get("awayTeam")
        # CFBD spread is the home side's line: -7 = home favoured by 7.
        rows.append({"game_id": g["id"], "school": home, "fav_by": -sp, "total": tot})
        rows.append({"game_id": g["id"], "school": away, "fav_by": sp, "total": tot})
    df = pd.DataFrame(rows)
    df["implied"] = df["total"] / 2 + df["fav_by"] / 2
    return df


def main() -> int:
    key = os.environ.get("CFBD_API_KEY", "").strip()
    frames = []
    for season in range(2021, 2026):
        h = pd.read_csv(ROOT / "data" / f"history_{season}.csv.gz")
        h = h.sort_values(["athlete_id", "week"])
        g = h.groupby("athlete_id")["points"]
        h["trail"] = g.transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean())
        team_pts = (h.groupby(["school", "week"])["points"].sum().rename("team_pts")
                    .reset_index().sort_values(["school", "week"]))
        team_pts["team_trail"] = team_pts.groupby("school")["team_pts"].transform(
            lambda s: s.shift(1).rolling(6, min_periods=3).mean())
        h = h.merge(team_pts[["school", "week", "team_trail"]], on=["school", "week"], how="left")
        ln = lines(season, key)
        h = h.merge(ln, on=["game_id", "school"], how="inner")
        h["season"] = season
        frames.append(h)
        print(season, "rows with lines", len(h), flush=True)
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["trail"] >= 8) & df["implied"].notna() & df["team_trail"].notna()].copy()
    df["ratio"] = (df["points"] / df["trail"]).clip(upper=5)
    # The model's own offence term: implied total over the team's usual output,
    # normalised per season so it is about 1 on average.
    df["mkt"] = df["implied"] / df.groupby("season")["implied"].transform("mean")
    df["form"] = df["team_trail"] / df.groupby("season")["team_trail"].transform("mean")
    df["mkt_rel"] = df["mkt"] / df["form"].clip(lower=0.3)
    df["bucket"] = pd.cut(df["fav_by"], [b[0] for b in BUCKETS] + [99],
                          labels=[f"{a}..{b}" for a, b in BUCKETS])
    base_label = f"{BASE[0]}..{BASE[1]}"
    X = pd.get_dummies(df["bucket"]).drop(columns=[base_label]).astype(float)
    X.insert(0, "mkt_rel", df["mkt_rel"].clip(0.3, 3))
    X.insert(0, "const", 1.0)
    y = df["ratio"].to_numpy()
    A = X.to_numpy()
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    s2 = resid @ resid / (len(y) - A.shape[1])
    cov = s2 * np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    out = {"rows": int(len(df)), "coef": {}, "by_position": {}}
    mean_ratio = float(y.mean())
    for name, b, e in zip(X.columns, beta, se):
        out["coef"][name] = {"effect": round(float(b), 4), "se": round(float(e), 4),
                             "as_share": round(float(b) / mean_ratio, 4) if name not in ("const", "mkt_rel") else None,
                             "n": int((df["bucket"] == name).sum()) if name not in ("const", "mkt_rel") else None}
    # Same thing per position for the favourite side, raw (no model), as a check
    for pos in ("QB", "RB", "WR", "TE"):
        d = df[df["position"] == pos]
        if len(d) < 200:
            continue
        g = d.groupby("bucket", observed=True)["ratio"].agg(["mean", "count"])
        base = float(g.loc[base_label, "mean"]) if base_label in g.index else float(d["ratio"].mean())
        out["by_position"][pos] = {str(k): {"rel": round(float(r["mean"]) / base, 3), "n": int(r["count"])}
                                   for k, r in g.iterrows()}
    path = ROOT / "data" / "cfb_blowout_fit.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
