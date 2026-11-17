"""Late news for the NHL board: starting goalies, tonight's lines, scratches.

Read from Daily Faceoff, the source the whole DFS field uses for this: the
NHL publishes no starting goalie and no lines before warmups. Two pages:

* /starting-goalies - each team's expected starter and how sure that is
  (Confirmed / Likely / Unconfirmed).
* /teams/<slug>/line-combinations - tonight's projected forward lines,
  defence pairs, power-play units, and injury / game-time tags.

What it changes on the board, in this order:

1. Goalies. A Confirmed starter plays (p_play 1) and his team's other goalies
   do not (p_play 0, status out, so the lineup builder cannot pick them). A
   Likely starter gets 0.85 and the others share 0.15. Unconfirmed: nothing
   changes. Each goalie's expected points are his conditional projection times
   his new p_play, so ownership and the pool see the real expectation.
2. Lines. Tonight's lines and PP units replace last game's, which is what
   the stacks are built from.
3. Scratches. A skater the lines page tags out / IR is marked out. A skater
   missing from fresh lines is probably a healthy scratch: p_play capped at
   0.25 and flagged on the page, not removed, because the page is a
   projection and can be a man short.
4. Promotions. A skater moved up or down from last game's deployment (a
   new line, onto or off PP1) has his projection moved part of the way to
   what that slot plays - ice time is the opportunity number in hockey, and
   power-play ice is worth about twice even-strength ice. Half way, and
   clipped to -15% / +20%, because a projected line is not a certain one.

Then ownership is recomputed, so the board the field is modelled on is the
one after the news.

Every step is optional: if Daily Faceoff cannot be read, the board publishes
exactly as before.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone

import requests

log = logging.getLogger("nhl_news")

DFO = "https://www.dailyfaceoff.com"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124 Safari/537.36")
TIMEOUT = 30
PAUSE = 0.5
FRESH_HOURS = 36            # lines older than this are last game's, not tonight's
LIKELY = 0.85
SCRATCH_PP = 0.25
PP_WEIGHT = 2.0             # a power-play minute is worth ~2 even-strength minutes
SHRINK = 0.5
CLIP = (0.85, 1.20)
OUT_TAGS = {"out", "ir", "ltir", "inj", "injured", "suspended", "sus", "o"}


_CACHE: dict = {}
HEADERS = {"User-Agent": UA,
           "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
           "Accept-Language": "en-US,en;q=0.9"}


def _fetch(url: str) -> str:
    """One page, fetched once per run. Retries with back-off: the site's bot
    screen refuses some requests from cloud machines and lets the next
    through."""
    if url in _CACHE:
        return _CACHE[url]
    last = None
    for wait in (0, 3, 8, 20):
        if wait:
            time.sleep(wait)
        try:
            r = requests.get(url, timeout=TIMEOUT, headers=HEADERS)
            if r.status_code == 200:
                _CACHE[url] = r.text
                return r.text
            last = f"HTTP {r.status_code}"
        except requests.RequestException as exc:
            last = str(exc)[:80]
    raise RuntimeError(f"{last} for {url}")


def _next_data(url: str) -> dict:
    text = _fetch(url)
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                  text, re.S)
    if not m:
        raise ValueError("no page data")
    return json.loads(m.group(1))["props"]["pageProps"]


# ------------------------------------------------------------------ fetching
def goalies(day: str | None = None) -> tuple[dict, dict]:
    """({team: {"name", "status"}}, {team: slug}) for one day's games."""
    url = f"{DFO}/starting-goalies" + (f"/{day}" if day else "")
    out, slugs = {}, {}
    for g in _next_data(url).get("data") or []:
        for side in ("home", "away"):
            team = g.get(f"{side}TeamAbbreviation")
            if not team:
                continue
            slugs[team] = g.get(f"{side}TeamSlug")
            name = g.get(f"{side}GoalieName")
            if name:
                out[team] = {"name": name,
                             "status": g.get(f"{side}NewsStrengthName")
                             or "Unconfirmed"}
    return out, slugs


def lines(slug: str) -> dict:
    """{"updated": datetime|None, "players": [{name, group, cat, inj, gtd}]}"""
    c = _next_data(f"{DFO}/teams/{slug}/line-combinations").get(
        "combinations") or {}
    upd = c.get("updatedAt")
    try:
        upd = datetime.fromisoformat(str(upd).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        upd = None
    pl = [{"name": p.get("name") or "",
           "group": str(p.get("groupIdentifier") or "").lower(),
           "cat": str(p.get("categoryIdentifier") or "").lower(),
           "inj": str(p.get("injuryStatus") or "").lower(),
           "gtd": bool(p.get("gameTimeDecision"))}
          for p in c.get("players") or []]
    return {"updated": upd, "players": pl}


# --------------------------------------------------------------- pure logic
def _lookup(players, name_keys):
    idx = {}
    for i, p in enumerate(players):
        for k in name_keys(p["name"]):
            idx.setdefault((p["team"], k), i)
    return idx


def _find(idx, team, name, name_keys):
    return next((idx[(team, k)] for k in name_keys(name) if (team, k) in idx),
                None)


def _scale(p: dict, f: float) -> None:
    for key in ("mean", "med", "ceil"):
        if p.get(key) is not None:
            p[key] = round(float(p[key]) * f, 2)
    if p.get("q"):
        p["q"] = [round(float(v) * f, 3) for v in p["q"]]


def apply_goalies(players: list[dict], starters: dict, name_keys) -> int:
    idx = _lookup(players, name_keys)
    changed = 0
    for team, info in starters.items():
        status = info["status"]
        if status not in ("Confirmed", "Likely"):
            continue
        s = _find(idx, team, info["name"], name_keys)
        if s is None:
            continue
        mates = [i for i, p in enumerate(players)
                 if p["team"] == team and p["pos"] == "G"]
        others = [i for i in mates if i != s]
        p_start = 1.0 if status == "Confirmed" else LIKELY
        p_other = 0.0 if status == "Confirmed" else (1 - LIKELY) / max(1, len(others))
        for i, pp in [(s, p_start)] + [(i, p_other) for i in others]:
            g = players[i]
            old = float(g.get("pp") or 1.0)
            cond = float(g.get("mean") or 0.0) / old if old > 0 else float(
                g.get("med") or 0.0)
            g["pp"] = round(pp, 4)
            g["mean"] = round(cond * pp, 2)
            g["gst"] = status if i == s else "backup"
            if pp == 0.0:
                g["status"] = "out"
            changed += 1
    return changed


def _slot(group: str, cat: str):
    m = re.fullmatch(r"([fd])(\d)", group)
    if m and cat in ("ev", ""):
        return ("L" if m.group(1) == "f" else "D") + m.group(2)
    return None


def apply_lines(players: list[dict], team_lines: dict, name_keys,
                now: datetime | None = None) -> dict:
    """Tonight's lines, scratches and promotions. Returns counts."""
    now = now or datetime.now(timezone.utc)
    idx = _lookup(players, name_keys)
    n = {"lines": 0, "out": 0, "scratch": 0, "moved": 0}
    fresh_teams = set()
    tonight = {}            # player index -> {"line", "ppu"}
    for team, tl in team_lines.items():
        upd = tl.get("updated")
        fresh = upd is not None and now - upd <= timedelta(hours=FRESH_HOURS)
        for q in tl["players"]:
            i = _find(idx, team, q["name"], name_keys)
            if i is None:
                continue
            p = players[i]
            if q["inj"] in OUT_TAGS and p["pos"] != "G":
                p["status"] = "out"
                p["news"] = "out (" + q["inj"] + ")"
                n["out"] += 1
                continue
            if q["gtd"]:
                p["doubtful"] = True
            if q["group"] == "ir":
                # On the injury list but only day-to-day: not in tonight's
                # lines, so he is treated like any man missing from them.
                p["doubtful"] = True
                continue
            if not fresh:
                continue
            fresh_teams.add(team)
            slot = _slot(q["group"], q["cat"])
            t = tonight.setdefault(i, {})
            if slot:
                t["line"] = slot
            m = re.fullmatch(r"pp(\d)", q["group"])
            if m and q["cat"] in ("pp", ""):
                t["ppu"] = int(m.group(1))

    # Slot norms from this slate's own skaters: what a man on L1 / PP1 plays.
    def med(vals):
        vals = sorted(v for v in vals if v)
        return vals[len(vals) // 2] if vals else None

    def eff(toi, pptoi):
        toi, pptoi = float(toi or 0), float(pptoi or 0)
        return (toi - pptoi) + PP_WEIGHT * pptoi

    by_slot, by_pp = {}, {}
    for i, t in tonight.items():
        p = players[i]
        if p.get("toi"):
            by_slot.setdefault(t.get("line"), []).append(
                float(p["toi"]) - float(p.get("pptoi") or 0))
            by_pp.setdefault(t.get("ppu"), []).append(float(p.get("pptoi") or 0))
    ev_norm = {k: med(v) for k, v in by_slot.items() if k}
    pp_norm = {k: med(v) for k, v in by_pp.items()}

    for i, p in enumerate(players):
        if p["team"] not in fresh_teams or p["pos"] == "G":
            continue
        if p.get("status") == "out":
            continue
        t = tonight.get(i)
        if t is None:
            # Not on tonight's sheet at all: almost always a healthy scratch.
            old = float(p.get("pp") or 1.0)
            if old > SCRATCH_PP:
                # Only the unconditional expectation moves: the quantiles are
                # "if he plays", and calibration keeps `mean` when pp < 0.5.
                if p.get("mean") is not None:
                    p["mean"] = round(float(p["mean"]) * SCRATCH_PP / old, 2)
                p["pp"] = SCRATCH_PP
            p["news"] = "not in tonight's lines"
            n["scratch"] += 1
            continue
        prev_line, prev_pp = p.get("line"), p.get("ppu")
        new_line, new_pp = t.get("line", prev_line), t.get("ppu")
        if new_line:
            p["line"], p["lg"] = new_line, f"{p['team']}-{new_line}"
        if new_pp:
            p["ppu"] = new_pp
        else:
            p.pop("ppu", None)
        n["lines"] += 1
        moved = (prev_line and new_line and prev_line != new_line) or (
            (prev_pp == 1) != (new_pp == 1))
        if not moved or not p.get("toi"):
            continue
        own_ev = float(p["toi"]) - float(p.get("pptoi") or 0)
        ev_new = ev_norm.get(new_line, own_ev) if new_line != prev_line else own_ev
        pp_new = pp_norm.get(new_pp, float(p.get("pptoi") or 0)) if (
            (prev_pp == 1) != (new_pp == 1)) else float(p.get("pptoi") or 0)
        before = eff(p["toi"], p.get("pptoi"))
        after = ev_new + PP_WEIGHT * pp_new
        if before <= 0:
            continue
        f = 1 + SHRINK * (after / before - 1)
        f = min(max(f, CLIP[0]), CLIP[1])
        if abs(f - 1) >= 0.02:
            _scale(p, f)
            p["news"] = (f"moved up ({prev_line or '-'}"
                         f"{'/PP' + str(prev_pp) if prev_pp else ''} -> "
                         f"{new_line}{'/PP' + str(new_pp) if new_pp else ''})"
                         if f > 1 else
                         f"moved down ({prev_line or '-'} -> {new_line}"
                         f"{'/PP' + str(new_pp) if new_pp else ''})")
            n["moved"] += 1
    return n


def reown(payload: dict) -> None:
    """Ownership and leverage again, on the post-news board."""
    import pandas as pd
    from engine import ownership as OWN
    pl = payload["players"]
    pool = pd.DataFrame({
        "position": [p["pos"] for p in pl],
        "salary": [p.get("salary") for p in pl],
        "mean": [0.0 if p.get("status") == "out"
                 else (p.get("proj") if p.get("proj") is not None else p.get("mean"))
                 for p in pl],
        "median": [p.get("proj") or p.get("med") or 0 for p in pl],
        "ceiling": [p.get("ceil") for p in pl],
        "p_play": [0.0 if p.get("status") == "out" else p.get("pp", 1.0)
                   for p in pl],
    })
    own = OWN.project(pool, payload["roster"])
    lev = OWN.leverage(pool, own)
    for p, o, l in zip(pl, own, lev):
        p["own"] = round(float(o), 5)
        p["lev"] = round(float(l), 3)


# ---------------------------------------------------------------- top level
def apply(payload: dict) -> None:
    """Goalies, lines, scratches and promotions onto one NHL payload."""
    if payload.get("sport") != "nhl":
        return
    from availability import name_keys
    players = payload.get("players") or []
    teams = {p["team"] for p in players}
    day = None
    if payload.get("locks"):
        try:
            day = (datetime.fromisoformat(payload["locks"])
                   .astimezone(timezone(timedelta(hours=-5))).date().isoformat())
        except ValueError:
            day = None
    starters, slugs = goalies(day)
    g = apply_goalies(players, starters, name_keys)
    team_lines = {}
    for t in sorted(teams):
        if not slugs.get(t):
            continue
        try:
            team_lines[t] = lines(slugs[t])
        except Exception as exc:                               # noqa: BLE001
            log.warning("lines %s: %s", t, str(exc)[:80])
        time.sleep(PAUSE)
    n = apply_lines(players, team_lines, name_keys)
    conf = sum(1 for t in teams if starters.get(t, {}).get("status") == "Confirmed")
    payload["news"] = {
        "source": "Daily Faceoff",
        "at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "goalies_confirmed": conf, "teams": len(teams), **n}
    log.info("news: %d/%d starting goalies confirmed (%d goalies re-set); "
             "tonight's lines for %d skaters, %d out, %d likely scratches, "
             "%d moved", conf, len(teams), g, n["lines"], n["out"],
             n["scratch"], n["moved"])
