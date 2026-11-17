"""A pool of legal lineups per slate, spread across ownership, for the page to
filter.

This is how the serious optimisers work: build a few hundred good, distinct
lineups once, then let the person narrow them - by total ownership, by
projection, by who is in or out - instead of re-solving every time a slider
moves. The page never has to run a solver, so it stays fast on a phone.

How the spread is made
----------------------
Every lineup maximises one of several objectives:

    median points  x (1 - lam * ownership)      lam in LAMBDAS
    ceiling points x (1 - lam * ownership)

lam = 0 is pure projection - the chalky, highest-median build. Larger lam
pays for each point less the more of the field already owns that player, so
the pool runs from the field's lineup to genuinely contrarian ones without
anyone having to pick the contrarian players by hand. Within each objective
lineups are forced apart (at most `len(slots) - MIN_UNIQUE` players shared
with any lineup already in the pool).

Rules come from the slate's own `roster` block, the same one the publishers
write: position counts, flex eligibility, salary cap and floor, max players
per team, min teams (NHL), max per game and min games (CFB), and no hitter
against your own pitcher / no skater against your own goalie.

Uses scipy's MILP (HiGHS), which is already a dependency, so this module has
no new requirements.
"""
from __future__ import annotations

import logging
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

log = logging.getLogger(__name__)

FLEX = {"UTIL", "FLEX", "SFLEX"}
LAMBDAS = (0.0, 0.5, 1.0, 1.5, 2.5)
PER_OBJECTIVE = 12          # lineups per objective; 2 x 5 x 12 = 120 max
MIN_UNIQUE = 2              # players each new lineup must not share
STACKED_PER_OBJECTIVE = 6   # hockey: 3 shapes x 3 lam x 6 = 54
TEAM_STACKS = 10            # hockey: a 4-stack of each of the top 10 teams
PER_TEAM_STACK = 3
GOALIES = {"G"}
PITCHERS = {"P", "SP", "RP"}


def _rules(players: list[dict], roster: dict):
    """Rows of (coefficients, lo, hi) for one slate's roster rules."""
    n = len(players)
    slots = list(roster["slots"])
    pos = [str(p["pos"]) for p in players]
    team = [str(p.get("team") or "") for p in players]
    game = [str(p.get("game") or "") for p in players]
    rows = []

    def row(mask, lo, hi):
        rows.append((np.asarray(mask, float), lo, hi))

    need = {}
    for s in slots:
        if s not in FLEX:
            need[s] = need.get(s, 0) + 1
    n_flex = sum(1 for s in slots if s in FLEX)
    flex_pos = set(roster.get("flex_positions") or [])
    maxpos = roster.get("max_position") or {}

    row(np.ones(n), len(slots), len(slots))
    for p_, k in need.items():
        mask = [x == p_ for x in pos]
        if p_ in flex_pos and n_flex:
            hi = min(k + n_flex, maxpos.get(p_, k + n_flex))
            row(mask, k, hi)
        else:
            row(mask, k, k)
    for p_, cap in maxpos.items():
        row([x == p_ for x in pos], 0, cap)
    # Nobody at a position the roster has no slot for.
    unusable = [x not in need and x not in flex_pos for x in pos]
    if any(unusable):
        row(unusable, 0, 0)

    sal = np.array([p["salary"] for p in players], float)
    cap = float(roster.get("salary_cap", 50_000))
    row(sal, 0, cap)

    # Max per team counts hitters/skaters only, as DraftKings does.
    mpt = roster.get("max_per_team")
    if mpt:
        for t in set(team):
            row([tm == t and x not in PITCHERS | GOALIES
                 for tm, x in zip(team, pos)], 0, mpt)
    mpg = roster.get("max_per_game")
    if mpg:
        for g in set(game):
            row([gm == g for gm in game], 0, mpg)

    # Batter against your own pitcher, skater against your own goalie: one
    # outcome bet in two directions. Off only if the roster says so.
    if roster.get("no_hitter_vs_pitcher", True):
        for i, p in enumerate(players):
            if pos[i] in PITCHERS | GOALIES:
                for j, q in enumerate(players):
                    if (j != i and pos[j] not in PITCHERS | GOALIES
                            and team[j] == str(p.get("opp") or "~")):
                        m = np.zeros(n)
                        m[i] = m[j] = 1
                        row(m, 0, 1)
    return rows, sal, cap


SHORTLIST_PER_SLOT = 14


def _shortlist(players, keep, roster):
    """Players who could plausibly be in an optimal lineup under SOME
    objective in the pool: the top few per position by median, by ceiling,
    by value, and by median among the low-owned. Everyone else is dominated
    at every lam. Cuts a 577-player hockey board to about 150 and the solve
    time with it, without changing which lineups come out at the top."""
    from collections import defaultdict
    per = defaultdict(int)
    for s in roster["slots"]:
        if s not in FLEX:
            per[s] += 1
    flex_pos = set(roster.get("flex_positions") or [])
    n_flex = sum(1 for s in roster["slots"] if s in FLEX)
    by_pos = defaultdict(list)
    for i in keep:
        by_pos[str(players[i]["pos"])].append(i)
    out = set()
    for pos_, idx in by_pos.items():
        k = SHORTLIST_PER_SLOT * (per.get(pos_, 0)
                                  + (n_flex if pos_ in flex_pos else 0))
        if k == 0:
            continue
        med = lambda i: players[i].get("proj", players[i]["med"])
        keys = (med,
                lambda i: players[i].get("ceil") or players[i]["med"],
                lambda i: med(i) / players[i]["salary"],
                lambda i: med(i) * (1 - min(players[i].get("own")
                                                       or 0, 1)))
        for key in keys:
            out.update(sorted(idx, key=key, reverse=True)[:k])
    return sorted(out)


def build(players: list[dict], roster: dict, field_size: int = 100_000,
          lambdas=LAMBDAS, per_objective: int = PER_OBJECTIVE,
          time_limit: float = 1.0, stacks: bool | None = None
          ) -> list[dict]:
    """The pool. Each lineup: player indices into `players`, and totals."""
    if not players or "CPT" in roster.get("slots", []):
        return []
    keep = [i for i, p in enumerate(players)
            if (p.get("med") or 0) > 0 and p.get("salary")
            and str(p.get("status") or "clear") not in ("out",)]
    keep = _shortlist(players, keep, roster)
    pl = [players[i] for i in keep]
    n = len(pl)
    rows, sal, cap = _rules(pl, roster)

    # Distinct teams / games, with one helper binary per group.
    pos = [str(p["pos"]) for p in pl]
    extra = []          # (group values per player, minimum, excluded pos)
    if roster.get("min_skater_teams"):
        extra.append(("team", int(roster["min_skater_teams"]),
                      GOALIES | PITCHERS))
    if roster.get("min_games"):
        extra.append(("game", int(roster["min_games"]), frozenset()))
    helpers = []
    for key, minimum, excl in extra:
        for g in sorted({str(p.get(key) or "") for p in pl} - {""}):
            helpers.append((key, g, excl, minimum))
    m = n + len(helpers)

    A, lo, hi = [], [], []
    for mask, a, b in rows:
        A.append(np.concatenate([mask, np.zeros(len(helpers))]))
        lo.append(a)
        hi.append(b)
    # helper_g <= sum of eligible players in g ; sum helpers >= minimum
    by_req = {}
    for h, (key, g, excl, minimum) in enumerate(helpers):
        r = np.zeros(m)
        r[n + h] = 1
        for i, p in enumerate(pl):
            if str(p.get(key) or "") == g and pos[i] not in excl:
                r[i] = -1
        A.append(r)
        lo.append(-np.inf)
        hi.append(0)
        by_req.setdefault((key, minimum), []).append(n + h)
    for (key, minimum), idx in by_req.items():
        r = np.zeros(m)
        r[idx] = 1
        A.append(r)
        lo.append(minimum)
        hi.append(np.inf)

    # Stack binaries: for every team, p_t (primary stack) and q_t (secondary).
    # Always present; only constrained when a stacked family asks for it.
    skater = np.array([x not in GOALIES | PITCHERS for x in pos])
    teams = sorted({str(p.get("team") or "") for p in pl} - {""})
    tmask = {t: np.array([str(p.get("team")) == t for p in pl]) & skater
             for t in teams}
    m_base = m
    m = m_base + 2 * len(teams)
    A = [np.concatenate([r, np.zeros(m - len(r))]) for r in A]
    for k_, t in enumerate(teams):
        r = np.zeros(m)
        r[m_base + k_] = 1
        r[m_base + len(teams) + k_] = 1
        A.append(r)              # a team is at most one of primary/secondary
        lo.append(0)
        hi.append(1)

    floor = float(roster.get("min_salary_pct") or 0) * cap
    # The calibrated expectation where the publisher added one (see
    # projection_calibration.py); the model median otherwise.
    med = np.array([p.get("proj", p["med"]) for p in pl], float)
    ceil = np.array([p.get("ceil") or p["med"] for p in pl], float)
    own = np.clip(np.array([p.get("own") or 0 for p in pl], float), 0, 1)

    pool, seen = [], []
    size = len(roster["slots"])

    def solve(obj, use_floor=True, stack=None):
        AA, LL, HH = list(A), list(lo), list(hi)
        if stack and stack[0] == "team":
            _, t, k = stack
            r = np.zeros(m)
            r[:n] = tmask[t]
            AA.append(r)
            LL.append(k)
            HH.append(np.inf)
        elif stack:
            k1, k2 = stack
            for off, k in ((0, k1), (len(teams), k2)):
                if not k:
                    continue
                for k_, t in enumerate(teams):
                    r = np.zeros(m)          # skaters of t >= k * flag_t
                    r[:n] = tmask[t]
                    r[m_base + off + k_] = -k
                    AA.append(r)
                    LL.append(0)
                    HH.append(np.inf)
                r = np.zeros(m)
                r[m_base + off: m_base + off + len(teams)] = 1
                AA.append(r)
                LL.append(1)
                HH.append(np.inf)
        if use_floor and floor > 0:
            r = np.zeros(m)
            r[:n] = sal
            AA.append(r)
            LL.append(floor)
            HH.append(np.inf)
        for chosen in seen:
            r = np.zeros(m)
            r[chosen] = 1
            AA.append(r)
            LL.append(0)
            HH.append(size - MIN_UNIQUE)
        c = np.zeros(m)
        c[:n] = -obj
        res = milp(c, constraints=LinearConstraint(np.array(AA), LL, HH),
                   integrality=np.ones(m), bounds=Bounds(0, 1),
                   options={"time_limit": time_limit, "disp": False})
        if res.x is None:
            return None
        return [i for i in range(n) if res.x[i] > 0.5]

    families = []
    for base_name, base in (("median", med), ("ceiling", ceil)):
        for lam in lambdas:
            families.append((f"{base_name} lam={lam}", base, lam, None,
                             per_objective))
    if stacks is None:
        stacks = "G" in roster.get("slots", [])     # hockey stacks by default
    if stacks and len(teams) >= 3:
        # Generic team stacks across the ownership range: 4-3, 4-2, 3-3.
        for shape in ((4, 3), (4, 2), (3, 3)):
            for lam in (0.0, 1.0, 2.0):
                families.append((f"stack {shape[0]}-{shape[1]} lam={lam}",
                                 ceil, lam, shape, STACKED_PER_OBJECTIVE))
        # A 4-man stack of each of the likeliest teams, so a page filter for
        # one team always has something to show. Teams ranked by the
        # ceiling of their best four skaters.
        def strength(t):
            return float(np.sort(ceil[tmask[t]])[-4:].sum())
        for t in sorted(teams, key=strength, reverse=True)[:TEAM_STACKS]:
            if tmask[t].sum() >= 4:
                families.append((f"team {t} lam=0.0", ceil, 0.0,
                                 ("team", t, 4), PER_TEAM_STACK))

    floor_ok = True
    for name, base, lam, stack, count in families:
            obj = base * (1.0 - lam * own)
            for _ in range(count):
                pick = solve(obj, floor_ok, stack)
                if pick is None and floor_ok:
                    floor_ok = False
                    log.info("lineup pool: salary floor $%d unreachable "
                             "with diversity rules - dropped", floor)
                    pick = solve(obj, False, stack)
                if pick is None:
                    break
                seen.append(pick)
                o = own[pick]
                pool.append({
                    "p": [keep[i] for i in pick],
                    "proj": round(float(med[pick].sum()), 2),
                    "ceil": round(float(ceil[pick].sum()), 2),
                    "sal": int(sal[pick].sum()),
                    "own": round(float(100 * o.sum()), 1),
                    # Expected copies of this exact lineup in the field, if
                    # entries were independent. Rough, but the right order of
                    # magnitude, and it ranks lineups correctly.
                    "dup": round(float(field_size * math.prod(
                        max(x, 1e-4) for x in o)), 4),
                    "src": name,
                })
    log.info("lineup pool: %d lineups, ownership %.0f%%-%.0f%%, projection "
             "%.1f-%.1f", len(pool),
             min((x["own"] for x in pool), default=0),
             max((x["own"] for x in pool), default=0),
             min((x["proj"] for x in pool), default=0),
             max((x["proj"] for x in pool), default=0))
    return pool


def safe_build(payload: dict) -> list[dict]:
    """For the publishers: a pool, or an empty list and a log line. A pool
    failure must never cost the slate itself."""
    try:
        return build(payload["players"], payload["roster"],
                     int(payload.get("field_size") or 100_000))
    except Exception as exc:                                   # noqa: BLE001
        log.warning("lineup pool for draft group %s failed (%s) - slate "
                    "published without one", payload.get("draft_group"), exc)
        return []
