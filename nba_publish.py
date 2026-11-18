"""Tonight's NBA DraftKings classic boards, published for the page.

    python nba_publish.py                 # every unlocked classic slate
    python nba_publish.py --draft-group N # one

Order of work for one slate:
  board (DraftKings) -> injuries (ESPN + DraftKings tags) -> lines (ESPN /
  DraftKings book) -> projection (nba_model, fitted coefficients) ->
  ownership (the classic formula fitted on real NBA contests) -> lineup pool.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import nba_data as D
import nba_fetch as F
import nba_model as M
from availability import name_keys
from engine import nba_pool as POOL
from engine import ownership as OWN

log = logging.getLogger("nba_publish")
SPORT = "nba"
EASTERN = "America/New_York"
ROSTER = {"slots": POOL.SLOTS, "salary_cap": 50_000}
OUT_DK = {"O", "OUT", "IR", "SUSP", "NA"}
Q_DK = {"Q", "GTD", "QUESTIONABLE"}
P_PLAY = {"Out": 0.0, "Doubtful": 0.15, "Questionable": 0.6, "Day-To-Day": 0.75}


def _status(row, inj: dict) -> tuple[float, str]:
    """(chance he plays, note) from the DraftKings tag and ESPN's report."""
    tag = str(row.get("dk_status") or "").upper()
    rep = next((inj[k] for k in name_keys(row["name"]) if k in inj), None)
    if tag in OUT_DK:
        return 0.0, f"out ({tag})"
    if rep:
        pp = P_PLAY.get(rep["status"], 1.0)
        if tag in Q_DK:
            pp = min(pp, 0.6)
        return pp, rep["status"].lower()
    if tag in Q_DK:
        return 0.6, "questionable"
    return 1.0, ""


def current_features(hist: pd.DataFrame, p: dict, day: str) -> pd.DataFrame:
    """Each player's features as of tonight: a dummy row dated today is
    appended to every player so the walk-forward feature code, which only
    ever looks backwards, produces exactly what it produced in the backtest."""
    h = M.prepare(hist)
    last = h.sort_values("date").groupby("player_id").tail(1)
    dummy = last.copy()
    for c in ("min", "pts", "fgm", "fga", "tpm", "tpa", "ftm", "fta", "reb",
              "oreb", "dreb", "ast", "tov", "stl", "blk", "pf", "fp", "use"):
        if c in dummy:
            dummy[c] = 0.0
    dummy["played"] = 0
    dummy["date"] = pd.Timestamp(day)
    dummy["game_id"] = "tonight"
    allr = pd.concat([h, dummy], ignore_index=True).sort_values(
        ["player_id", "date"]).reset_index(drop=True)
    f = M.player_history_features(allr, p)
    cur = f[f["game_id"] == "tonight"].copy()
    cur["last_team"] = last.set_index("player_id").loc[cur["player_id"], "team"].values
    return cur.set_index("player_id")


def seen_out(hist: pd.DataFrame, pid: str, out_ids: list[str], n: int = 8) -> float:
    """How much of this player's recent sample was ALREADY played without the
    men out tonight - so their usage is not handed to him twice."""
    if not out_ids:
        return 0.0
    h = hist[hist["player_id"] == pid].sort_values("date").tail(n)
    if not len(h):
        return 0.0
    games = h[h["played"] == 1]["game_id"]
    if not len(games):
        return 0.0
    sub = hist[hist["game_id"].isin(games) & hist["player_id"].isin(out_ids)]
    played_with = sub.groupby("player_id")["played"].sum()
    frac_absent = [1 - played_with.get(o, 0) / len(games) for o in out_ids]
    return float(np.mean(frac_absent))


def team_avg(hist: pd.DataFrame) -> dict:
    g = (hist.drop_duplicates(["game_id", "team"]).sort_values("date"))
    return {t: float(s["team_pts"].ewm(halflife=10).mean().iloc[-1])
            for t, s in g.groupby("team")}


def build_slate(dg: int, starts, label: str, hist: pd.DataFrame, p: dict,
                inj: dict, lines: dict, day: str, feats: pd.DataFrame) -> dict | None:
    b = D.board(dg)
    if b.empty:
        return None
    # board -> history id, by name then team
    hist_names = (hist.sort_values("date").groupby("player_id").tail(1)
                  [["player_id", "name", "team"]])
    key_to_ids = {}
    for r in hist_names.itertuples():
        for k in name_keys(r.name):
            key_to_ids.setdefault(k, []).append((r.player_id, r.team))

    def hid(name, team):
        cands = [c for k in name_keys(name) for c in key_to_ids.get(k, [])]
        same = [c for c in cands if c[1] == team]
        return (same or cands or [(None, None)])[0][0]

    b["pid_h"] = [hid(n, t) for n, t in zip(b["name"], b["team"])]
    st = [_status(r, inj) for r in b.to_dict("records")]
    b["pp"] = [s[0] for s in st]
    b["note"] = [s[1] for s in st]

    # Rows for the projection: every board player, plus injured men DraftKings
    # has already removed, so their minutes and usage are redistributed.
    rows = []
    teams = set(b["team"])
    for r in b.itertuples():
        rows.append({"player_id": r.pid_h, "team": r.team, "avail": int(r.pp > 0.3),
                     "board": True, "name": r.name})
    on_board = set(b["pid_h"].dropna())
    for info in {v["name"]: v for v in inj.values()}.values():
        if info["status"] not in ("Out", "Doubtful") or info["team"] not in teams:
            continue
        pid_ = hid(info["name"], info["team"])
        if pid_ and pid_ not in on_board:
            rows.append({"player_id": pid_, "team": info["team"], "avail": 0,
                         "board": False, "name": info["name"]})
    df = pd.DataFrame(rows)
    df = df.join(feats[["base_min", "rate", "rate_long", "use_rate", "gp_prior"]],
                 on="player_id")
    tavg = team_avg(hist)
    df["game_id"] = "tonight"
    df["spread"] = df["team"].map(lambda t: (lines.get(t) or {}).get("spread"))
    df["total"] = df["team"].map(lambda t: (lines.get(t) or {}).get("total"))
    df["team_avg"] = df["team"].map(tavg)
    out_by_team = df[df["avail"] == 0].groupby("team")["player_id"].apply(
        lambda s: [x for x in s if x])
    df["seen_out"] = [seen_out(hist, pid_, out_by_team.get(t, []))
                      if pid_ and a else 0.0
                      for pid_, t, a in zip(df["player_id"], df["team"], df["avail"])]
    proj = M.project_rows(df, p)
    proj = proj[proj["board"]]
    by_name = proj.set_index("name")

    players = []
    for r in b.itertuples():
        row = by_name.loc[r.name] if r.name in by_name.index else None
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        has_hist = row is not None and pd.notna(row.get("gp_prior")) and row.get("gp_prior", 0) > 0
        if has_hist and pd.notna(row.get("proj")) and r.pp > 0.3:
            cond = float(row["proj"])
            mins = float(row["min_p"])
            sd = float(row["sd"])
            vac = float(row.get("vacated") or 0)
        elif r.pp > 0.3 and pd.notna(r.dk_ppg) and r.dk_ppg > 0:
            cond, mins, sd, vac = float(r.dk_ppg), None, p["sd_a"] + p["sd_b"] * float(r.dk_ppg), 0.0
        else:
            cond, mins, sd, vac = 0.0, 0.0, 0.0, 0.0
        mean = cond * r.pp
        note = r.note
        bump = p["usage_k"] * max(0.0, vac - p["usage_floor"])
        if bump >= 0.02 and cond > 0:
            note = (note + "; " if note else "") + f"+{bump * 100:.0f}%/min with teammates out"
        ln = lines.get(r.team) or {}
        players.append({
            "name": r.name, "pos": r.position, "elig": r.elig, "team": r.team,
            "opp": r.opponent, "pid": str(r.dk_player_id),
            "game": "@".join(sorted([r.team, r.opponent])),
            "salary": int(r.salary),
            "proj": round(mean, 2), "med": round(cond, 2), "mean": round(mean, 2),
            "ceil": round(cond + 1.28 * sd, 2) if cond else 0.0,
            "min": round(mins, 1) if mins else None,
            "pp": round(float(r.pp), 2),
            "status": "out" if r.pp == 0 else "clear",
            "doubtful": bool(0 < r.pp < 0.7),
            "news": note or None,
            "implied": (round(ln["total"] / 2 - ln["spread"] / 2, 1)
                        if ln.get("total") is not None and ln.get("spread") is not None else None),
            "spread": ln.get("spread"),
        })
    pl = pd.DataFrame(players)
    pool = pd.DataFrame({"position": "ALL", "salary": pl["salary"],
                         "mean": pl["mean"], "median": pl["med"],
                         "ceiling": pl["ceil"], "p_play": pl["pp"],
                         "playing": np.where(pl["status"] == "out", "out", "yes")})
    score = OWN.appeal(pool, classic=True).to_numpy(float)
    own = OWN._allocate(score, 8.0, OWN.CLASSIC_CONCENTRATION)
    lev = (pl["med"].rank(pct=True) - pd.Series(own).rank(pct=True)).round(3)
    for d, o, l in zip(players, own, lev):
        d["own"] = round(float(o), 5)
        d["lev"] = float(l)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sport": SPORT, "site": "dk", "kind": "classic", "draft_group": int(dg),
        "label": label,
        "locks": pd.Timestamp(starts).isoformat() if starts is not None else None,
        "field_size": 100_000, "roster": ROSTER, "players": players,
        "market": {"source": "ESPN (DraftKings lines)",
                   "teams": sum(1 for t in teams if (lines.get(t) or {}).get("total"))}
                  if any((lines.get(t) or {}).get("total") for t in teams) else None,
        "model": {k: p[k] for k in ("fill_gamma", "usage_k", "vegas_a", "blowout_b")},
    }
    payload["lineups"] = POOL.safe_build(payload)
    log.info("draft group %s: %d players, %d out, %d lineups", dg, len(players),
             sum(1 for x in players if x["status"] == "out"), len(payload["lineups"]))
    return payload


def write(payloads: list[dict]) -> None:
    from pathlib import Path
    data = Path(__file__).resolve().parent / "docs" / "data"
    data.mkdir(parents=True, exist_ok=True)
    keep = set()
    for p in payloads:
        name = f"{SPORT}_dk_{p['kind']}_{p['draft_group']}.json"
        (data / name).write_text(json.dumps(p, indent=1, allow_nan=False))
        keep.add(name)
    for old in data.glob(f"{SPORT}_dk_*.json"):
        if old.name in keep:
            continue
        when = json.loads(old.read_text()).get("generated_at", "")[:10] or "undated"
        dest = data / "archive" / when
        dest.mkdir(parents=True, exist_ok=True)
        old.replace(dest / old.name)
    listed = []
    for f in sorted(data.glob("*_dk_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:                                      # noqa: BLE001
            continue
        listed.append({"sport": d.get("sport", "?"), "site": d.get("site", "dk"),
                       "kind": d.get("kind", "classic"), "label": d.get("label", ""),
                       "file": f"data/{f.name}", "locks": d.get("locks")})
    (data / "manifest.json").write_text(json.dumps({
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "slates": listed}, indent=1))


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-group", type=int)
    a = ap.parse_args(argv)
    OWN.use_fit("nba")
    p = M.params()
    hist = F.load()
    if hist.empty:
        log.error("no NBA history cached - run the NBA fetch workflow first")
        return 1
    sl = D.classic_slates()
    if a.draft_group:
        sl = D.slates()
        sl = sl[sl["draft_group"] == a.draft_group]
    if sl.empty:
        log.error("no unlocked NBA classic slates")
        return 1
    first = pd.Timestamp(sl["starts"].min()).tz_convert(EASTERN)
    day = first.date().isoformat()
    try:
        inj = D.injuries()
    except Exception as exc:                                   # noqa: BLE001
        log.warning("ESPN injuries unavailable (%s); DraftKings tags only", exc)
        inj = {}
    try:
        lines = D.lines(day)
    except Exception as exc:                                   # noqa: BLE001
        log.error("no lines (%s) - projections without Vegas", exc)
        lines = {}
    feats = current_features(hist, p, day)
    payloads = []
    for r in sl.itertuples():
        when = pd.Timestamp(r.starts).tz_convert(EASTERN).strftime("%-I:%M%p").lower()
        label = f"{when}{' ' + r.suffix if r.suffix else ''} · {r.contests} contests"
        try:
            pay = build_slate(int(r.draft_group), r.starts, label, hist, p, inj,
                              lines, day, feats)
        except Exception as exc:                               # noqa: BLE001
            log.exception("draft group %s failed: %s", r.draft_group, exc)
            continue
        if pay:
            payloads.append(pay)
    if not payloads:
        return 1
    write(payloads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
