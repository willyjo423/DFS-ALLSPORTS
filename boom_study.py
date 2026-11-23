"""Which signals, known before lock, predict a player beating his own
simulated 97th / 90th percentile more often than the sim already allows?

    python boom_study.py --sports nhl nba

Rows come from tail_check.run (every archived lock board, real box scores,
each player's own simulated distribution with today's sims). Signals:
  NHL  line promotion / demotion vs his previous board, PP1 now, PP unit gained,
       on the top line, goalie confirmed against, team implied total high
  NBA  'teammates out' / 'starting' in the news note, projected minutes up
       vs his previous board, props above model
  all  props above model by 25%+, low ownership for his salary tier
A signal counts if its boom rate is well above the rate for the same price
tier without it (z >= 2) in BOTH halves of the nights.
Writes docs/data/backtest/boom_study.json.
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

import backtest as B
import tail_check as T


def line_no(lg):
    m = re.search(r"L(\d)", str(lg or ""))
    return int(m.group(1)) if m else None


def signals(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    df = df.sort_values("day").copy()
    key = df["name"].astype(str) + "|" + df["team"].astype(str)
    df["boom97"] = df["actual"] > df["p97"]
    df["boom90"] = df["actual"] > df["p90"]
    m, pr = pd.to_numeric(df["model"], errors="coerce"), pd.to_numeric(df["props"], errors="coerce")
    df["props_up"] = (pr > 1.25 * m) & m.gt(1)
    df["props_down"] = (pr < 0.8 * m) & m.gt(1)
    own = pd.to_numeric(df["own"], errors="coerce").fillna(0)
    df["low_own_for_tier"] = own < df.groupby(["day", "tier"])["own"].transform(
        lambda x: pd.to_numeric(x, errors="coerce").quantile(0.25))
    if sport == "nhl":
        ln = df["lg"].map(line_no)
        prev_ln = ln.groupby(key).shift(1)
        ppu = pd.to_numeric(df["ppu"], errors="coerce")
        prev_ppu = ppu.groupby(key).shift(1)
        df["top_line"] = ln.eq(1)
        df["line_promoted"] = ln.notna() & prev_ln.notna() & (ln < prev_ln)
        df["line_demoted"] = ln.notna() & prev_ln.notna() & (ln > prev_ln)
        df["pp1"] = ppu.eq(1)
        df["pp_gained"] = ppu.notna() & (prev_ppu.isna() | (ppu < prev_ppu))
        imp = pd.to_numeric(df["implied"], errors="coerce")
        df["team_total_high"] = imp > imp.groupby(df["day"]).transform(lambda x: x.quantile(0.75))
    if sport == "nba":
        n = df["news"].astype(str).str.lower()
        df["teammates_out"] = n.str.contains("teammates out")
        df["starting"] = n.str.contains("starting")
        mins = pd.to_numeric(df["min"], errors="coerce")
        prev = mins.groupby(key).shift(1)
        df["minutes_up"] = mins.notna() & prev.notna() & (mins >= prev + 4)
    return df


def test(df: pd.DataFrame, sig: str) -> dict | None:
    s = df[sig].fillna(False).astype(bool)
    if s.sum() < 40:
        return None
    out = {"signal": sig, "n": int(s.sum())}
    for b in ("boom97", "boom90"):
        on = df[s]
        # expected: same tier, signal off
        base = df[~s].groupby("tier")[b].mean()
        exp = on["tier"].map(base).fillna(df[b].mean())
        obs, e = float(on[b].mean()), float(exp.mean())
        var = float((exp * (1 - exp)).sum())
        z = (on[b].sum() - exp.sum()) / np.sqrt(var) if var > 0 else 0.0
        out[f"{b}_rate"], out[f"{b}_expected"], out[f"{b}_z"] = round(obs, 4), round(e, 4), round(float(z), 2)
    days = sorted(df["day"].unique())
    h = days[len(days) // 2]
    zs = []
    for part in (df[df["day"] < h], df[df["day"] >= h]):
        s2 = part[sig].fillna(False).astype(bool)
        if s2.sum() < 15:
            zs.append(None); continue
        base = part[~s2].groupby("tier")["boom97"].mean()
        exp = part[s2]["tier"].map(base).fillna(part["boom97"].mean())
        var = float((exp * (1 - exp)).sum())
        zs.append(round(float((part[s2]["boom97"].sum() - exp.sum()) / np.sqrt(var)), 2) if var > 0 else None)
    out["z97_halves"] = zs
    out["holds"] = bool(out["boom97_z"] >= 2 and all(z is not None and z > 0 for z in zs))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sports", nargs="+", default=["nhl", "nba"])
    ap.add_argument("--sims", type=int, default=3000)
    a = ap.parse_args(argv)
    res = {}
    for sport in a.sports:
        df = T.run(sport, a.sims)
        if df.empty:
            continue
        df = signals(df, sport)
        sigs = [c for c in df.columns if df[c].dtype == bool and not c.startswith("boom")]
        rows = [r for r in (test(df, c) for c in sigs) if r]
        res[sport] = {"rows": len(df), "nights": int(df["day"].nunique()),
                      "boom97_all": round(float(df["boom97"].mean()), 4), "signals": rows}
        print(f"\n== {sport}: {len(df)} player-nights, {df['day'].nunique()} nights, "
              f"beat 97th {df['boom97'].mean():.1%}")
        print(pd.DataFrame(rows).to_string(index=False))
    B.OUT.mkdir(parents=True, exist_ok=True)
    (B.OUT / "boom_study.json").write_text(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
