"""Forward lines, defence pairs and power-play units, from the NHL's own data.

Built from each team's MOST RECENT completed game: the official shift chart
(who was on the ice, second by second) and the play-by-play (what the
strength was, second by second). From those two:

* Forward lines and defence pairs: the players who shared the most
  five-on-five ice. Greedy - the forward with the most 5v5 time picks the two
  forwards he spent most of it with, then the next free forward does the
  same - which reproduces a coach's lines whenever a coach actually ran
  lines, and degrades gracefully when he blended them.
* Power-play units: ranked by time on ice while the team had more skaters
  than the opponent. PP1 is the top five, PP2 the next five.
* The goalie who started that game.

Last game's deployment is what tonight's projected lines from every news site
are built on too, until the morning skate. It is not a confirmation, and the
page says so.

What it is for, measured
------------------------
Graded on ~10,800 archived skater games, last-game power-play and ice time
add almost nothing to the projection once salary and the model are in it
(rank correlation 0.412 -> 0.416), so this module DOES NOT change
projections. Its value is correlation: linemates score together - one goal
is a goal and up to two assists for the same line - so a tournament lineup
wants the same line or PP unit, not just the same team. The lineup pool uses
these groups to build line and power-play stacks.

    python nhl_lines.py EDM TOR       # print the lines for those teams
"""
from __future__ import annotations

import datetime as dt
import logging
import sys
import time

import numpy as np
import requests

log = logging.getLogger("nhl_lines")

WEB = "https://api-web.nhle.com/v1"
STATS = "https://api.nhle.com/stats/rest/en"
TIMEOUT = 30
PAUSE = 0.4                 # between requests; the NHL API 429s a burst
REG_PERIOD = 1200
OT_PERIOD = 300             # regular-season overtime
MIN_PP_SECONDS = 60         # less than a minute of PP is not a unit

_CACHE: dict = {}


def _get(url: str) -> dict:
    if url in _CACHE:
        return _CACHE[url]
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=TIMEOUT,
                             headers={"User-Agent": "dfs-allsports/1.0"})
            if r.status_code == 429:
                time.sleep(min(float(r.headers.get("Retry-After") or 5 * (attempt + 1)), 30))
                continue
            r.raise_for_status()
            time.sleep(PAUSE)
            _CACHE[url] = r.json()
            return _CACHE[url]
        except requests.RequestException as exc:
            last = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"{url}: {last}")


def _sec(mmss: str | None) -> int:
    if not mmss:
        return 0
    m, s = str(mmss).split(":")
    return int(m) * 60 + int(s)


def _t(period: int, mmss: str | None) -> int:
    return (period - 1) * REG_PERIOD + _sec(mmss)


def last_game(team: str, before: dt.date | None = None) -> int | None:
    """Id of the team's most recent finished regular-season or playoff game."""
    before = before or dt.date.today()
    js = _get(f"{WEB}/club-schedule-season/{team}/now")
    done = [g for g in js.get("games") or []
            if g.get("gameState") in ("OFF", "FINAL")
            and g.get("gameType") in (2, 3)
            and dt.date.fromisoformat(g["gameDate"]) < before]
    return max(done, key=lambda g: g["gameDate"])["id"] if done else None


def game_groups(game_id: int) -> dict[str, dict]:
    """{team abbrev: {lines, pairs, pp1, pp2, goalie, names}} for one game."""
    box = _get(f"{WEB}/gamecenter/{game_id}/boxscore")
    pbp = _get(f"{WEB}/gamecenter/{game_id}/play-by-play")
    shifts = _get(f"{STATS}/shiftcharts?cayenneExp=gameId={game_id}")["data"]

    away = box["awayTeam"]["abbrev"]
    home = box["homeTeam"]["abbrev"]

    # Strength, second by second. situationCode is
    # [away goalie][away skaters][home skaters][home goalie] and holds from
    # the event that set it until the next one.
    last_period = max((p["periodDescriptor"]["number"] for p in pbp["plays"]),
                      default=3)
    T = 3 * REG_PERIOD + max(0, last_period - 3) * OT_PERIOD + 1
    away_sk = np.full(T, 5, int)
    home_sk = np.full(T, 5, int)
    both_g = np.ones(T, bool)
    plays = sorted(pbp["plays"], key=lambda p: (p["periodDescriptor"]["number"],
                                                _sec(p.get("timeInPeriod")),
                                                p.get("sortOrder", 0)))
    for k, p in enumerate(plays):
        code = str(p.get("situationCode") or "1551")
        if len(code) != 4 or not code.isdigit():
            continue
        t0 = _t(p["periodDescriptor"]["number"], p.get("timeInPeriod"))
        t1 = (_t(plays[k + 1]["periodDescriptor"]["number"],
                 plays[k + 1].get("timeInPeriod")) if k + 1 < len(plays) else T)
        t1 = min(max(t1, t0), T)
        away_sk[t0:t1] = int(code[1])
        home_sk[t0:t1] = int(code[2])
        both_g[t0:t1] = code[0] == "1" and code[3] == "1"
    es = (away_sk == 5) & (home_sk == 5) & both_g
    pp = {away: (away_sk > home_sk) & both_g, home: (home_sk > away_sk) & both_g}

    # Who is a forward, a defenceman, a goalie, from the box score.
    role, names, starter = {}, {}, {}
    for side, abbrev in (("awayTeam", away), ("homeTeam", home)):
        stats = box["playerByGameStats"][side]
        for grp, r in (("forwards", "F"), ("defense", "D"), ("goalies", "G")):
            for pl in stats.get(grp) or []:
                role[pl["playerId"]] = (abbrev, r)
                if r == "G" and pl.get("starter"):
                    starter[abbrev] = pl["playerId"]

    on = {}
    for s in shifts:
        if s.get("typeCode") != 517 or not s.get("startTime"):
            continue
        pid = s["playerId"]
        names[pid] = f"{s.get('firstName', '')} {s.get('lastName', '')}".strip()
        a = _t(s["period"], s["startTime"])
        b = _t(s["period"], s["endTime"])
        if pid not in on:
            on[pid] = np.zeros(T, bool)
        on[pid][a:max(a, b)] = True

    out = {}
    for abbrev in (away, home):
        mine = [p for p in on if role.get(p, ("", ""))[0] == abbrev]
        fwd = [p for p in mine if role[p][1] == "F"]
        dmen = [p for p in mine if role[p][1] == "D"]

        def groups(ids, size):
            # Seeded by TOTAL ice time, so the star seeds first and his
            # linemates are found before a checking line can claim them;
            # partners chosen by shared 5v5 time. Then numbered by the group's
            # total ice time - a coach's L1 is the line he plays most, which
            # 5v5 time alone gets wrong for a top line that lives on the PP
            # (McDavid's line came out "L3" by 5v5 minutes).
            tot = {p: int(on[p].sum()) for p in ids}
            free = sorted(ids, key=lambda p: -tot[p])
            res = []
            while len(free) >= size:
                seed = free.pop(0)
                mates = sorted(free, key=lambda q: -int((on[seed] & on[q] & es).sum()))[:size - 1]
                for q in mates:
                    free.remove(q)
                res.append([seed] + mates)
            return sorted(res, key=lambda g: -sum(tot[p] for p in g))

        sk = [p for p in mine if role[p][1] != "G"]
        pp_time = {p: int((on[p] & pp[abbrev]).sum()) for p in sk}
        ranked = [p for p in sorted(sk, key=lambda p: -pp_time[p])
                  if pp_time[p] >= MIN_PP_SECONDS]
        out[abbrev] = {
            "game": game_id,
            "lines": groups(fwd, 3)[:4],
            "pairs": groups(dmen, 2)[:3],
            "pp1": ranked[:5],
            "pp2": ranked[5:10],
            "goalie": starter.get(abbrev),
            "names": {p: names.get(p, str(p)) for p in mine},
        }
    return out


def team_groups(teams, today: dt.date | None = None) -> dict[str, dict]:
    """Groups for every team asked for, from each one's last game. A team
    whose game cannot be read is simply absent - never a guess."""
    out = {}
    for team in sorted(set(teams)):
        try:
            gid = last_game(team, today)
            if gid is None:
                continue
            g = game_groups(gid).get(team)
            if g:
                out[team] = g
        except Exception as exc:                               # noqa: BLE001
            log.warning("lines for %s unavailable (%s)", team, str(exc)[:80])
    log.info("lines: %d of %d teams read", len(out), len(set(teams)))
    return out


def annotate(payload: dict) -> None:
    """Add `line` (L1-L4 / D1-D3), `pp` (1/2) and `lg` (a group key such as
    'EDM-L1') to the skaters of one published NHL slate, and `g_started` to
    goalies who started their team's last game. In place; silent no-op for
    other sports or when nothing could be read."""
    if payload.get("sport") != "nhl":
        return
    from availability import name_keys
    players = payload.get("players") or []
    groups = team_groups({p["team"] for p in players})
    if not groups:
        return
    lookup = {}
    for team, g in groups.items():
        nm = {pid: name_keys(n) for pid, n in g["names"].items()}
        for i, line in enumerate(g["lines"], 1):
            for pid in line:
                for k in nm[pid]:
                    lookup[(team, k)] = {"line": f"L{i}", "lg": f"{team}-L{i}"}
        for i, pair in enumerate(g["pairs"], 1):
            for pid in pair:
                for k in nm[pid]:
                    lookup[(team, k)] = {"line": f"D{i}", "lg": f"{team}-D{i}"}
        for unit, key in ((1, "pp1"), (2, "pp2")):
            for pid in g[key]:
                for k in nm[pid]:
                    lookup.setdefault((team, k), {})["pp"] = unit
        if g.get("goalie"):
            for k in nm.get(g["goalie"], set()):
                lookup.setdefault((team, k), {})["g_started"] = True
    hit = 0
    for p in players:
        info = next((lookup[(p["team"], k)] for k in name_keys(p["name"])
                     if (p["team"], k) in lookup), None)
        if info:
            p.update(info)
            hit += 1
    payload["lines_from"] = {t: g["game"] for t, g in groups.items()}
    log.info("lines: %d of %d players placed on a line, pair or PP unit",
             hit, len(players))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for team, g in team_groups(sys.argv[1:] or ["EDM"]).items():
        n = g["names"]
        print(f"\n{team} (game {g['game']}), goalie: {n.get(g['goalie'], '?')}")
        for i, line in enumerate(g["lines"], 1):
            print(f"  L{i}: " + " - ".join(n[p] for p in line))
        for i, pair in enumerate(g["pairs"], 1):
            print(f"  D{i}: " + " - ".join(n[p] for p in pair))
        print("  PP1: " + ", ".join(n[p] for p in g["pp1"]))
        print("  PP2: " + ", ".join(n[p] for p in g["pp2"]))
