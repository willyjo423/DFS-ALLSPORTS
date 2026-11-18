"""Tonight's NBA inputs: DraftKings slates and boards, ESPN injuries and lines.

The DraftKings lobby endpoints are the same ones the NHL board reads (the
draft-group URLs are sport-agnostic), so nhl_data's tested fetch, lock-time
and board parsing are reused rather than copied.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os

import pandas as pd
import requests

import nba_fetch as F
import nhl_data as ND

log = logging.getLogger("nba_data")

DK_CONTESTS = "https://www.draftkings.com/lobby/getcontests?sport=NBA"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
CLASSIC = 70
SHOWDOWN = 81
UA = {"User-Agent": "Mozilla/5.0 (compatible; dfs-model/1.0)"}
HERE = os.path.dirname(os.path.abspath(__file__))
ODDS_CACHE = os.path.join(HERE, "docs", "data", "odds")


def slates() -> pd.DataFrame:
    payload = ND._get(DK_CONTESTS, as_json=True)
    rows = [{"draft_group": g.get("DraftGroupId"),
             "game_type": g.get("GameTypeId"),
             "starts": ND._lock_time(g.get("StartDateEst")),
             "suffix": (g.get("ContestStartTimeSuffix") or "").strip()}
            for g in payload.get("DraftGroups") or []]
    df = pd.DataFrame(rows).dropna(subset=["draft_group"])
    if df.empty:
        raise ND.DataUnavailable("DraftKings lists no NBA slates")
    counts = pd.Series([c.get("dg") for c in payload.get("Contests") or []]).value_counts()
    df["contests"] = df["draft_group"].map(counts).fillna(0).astype(int)
    return df.sort_values("contests", ascending=False).reset_index(drop=True)


def classic_slates(now=None, top: int = 3) -> pd.DataFrame:
    """Unlocked classic boards worth publishing: the main slate and the
    early/late/turbo splits DraftKings runs, busiest first."""
    now = now or pd.Timestamp.now(tz="UTC")
    s = slates()
    s = s[(s["game_type"] == CLASSIC) & s["starts"].notna() & (s["starts"] > now)]
    s = s[~s["suffix"].str.contains("Tiers|Points|Pick|Snake", case=False, na=False)]
    return s.head(top)


def board(draft_group: int) -> pd.DataFrame:
    b = ND.board(draft_group)
    b["team"] = b["team"].map(F.canon)
    b["opponent"] = b["opponent"].map(F.canon)
    b["elig"] = b["position"].map(eligibility)
    return b


def eligibility(pos: str) -> list[str]:
    ps = [p.strip() for p in str(pos or "").upper().split("/") if p.strip()]
    out = set(ps)
    if out & {"PG", "SG"}:
        out.add("G")
    if out & {"SF", "PF"}:
        out.add("F")
    out.add("UTIL")
    order = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    return [o for o in order if o in out]


# ------------------------------------------------------------------ injuries
def injuries() -> dict:
    """{(TEAM, name_key): status} from ESPN's league injury feed. Statuses:
    Out, Doubtful, Questionable, Day-To-Day."""
    from availability import name_keys
    r = requests.get(f"{ESPN}/injuries", headers=UA, timeout=30)
    r.raise_for_status()
    out = {}
    for t in r.json().get("injuries") or []:
        for i in t.get("injuries") or []:
            ath = i.get("athlete") or {}
            team = F.canon(((ath.get("team") or {}).get("abbreviation"))
                           or (t.get("abbreviation") or ""))
            status = str(i.get("status") or "")
            nm = ath.get("displayName") or ""
            for k in name_keys(nm):
                out[k] = {"status": status, "team": team, "name": nm,
                          "note": i.get("shortComment") or ""}
    log.info("injuries: %d players listed by ESPN", len({v["name"] for v in out.values()}))
    return out


# --------------------------------------------------------------------- lines
def lines(day: str) -> dict:
    """{TEAM: {"opp", "home", "spread", "total"}} from ESPN's scoreboard
    (DraftKings lines, free, no quota). Cached per day so a refused call
    reuses the last good read."""
    path = os.path.join(ODDS_CACHE, f"nba_{day}.json")
    try:
        r = requests.get(f"{ESPN}/scoreboard", params={"dates": day.replace("-", "")},
                         headers=UA, timeout=30)
        r.raise_for_status()
        out = {}
        for ev in r.json().get("events") or []:
            comp = (ev.get("competitions") or [{}])[0]
            side = {c.get("homeAway"): F.canon((c.get("team") or {}).get("abbreviation"))
                    for c in comp.get("competitors") or []}
            o = (comp.get("odds") or [None])[0] or {}
            total = o.get("overUnder")
            sp = o.get("spread")
            fav_home = (o.get("homeTeamOdds") or {}).get("favorite")
            spread_home = None
            if sp is not None:
                spread_home = -abs(float(sp)) if fav_home else abs(float(sp))
            h, a = side.get("home"), side.get("away")
            if not (h and a):
                continue
            out[h] = {"opp": a, "home": 1, "spread": spread_home,
                      "total": float(total) if total is not None else None}
            out[a] = {"opp": h, "home": 0,
                      "spread": -spread_home if spread_home is not None else None,
                      "total": float(total) if total is not None else None}
        if out:
            os.makedirs(ODDS_CACHE, exist_ok=True)
            json.dump({"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
                       "lines": out}, open(path, "w"))
        return out
    except Exception as exc:                                   # noqa: BLE001
        if os.path.exists(path):
            log.warning("lines: ESPN failed (%s); using the cached read", str(exc)[:60])
            return json.load(open(path))["lines"]
        raise
