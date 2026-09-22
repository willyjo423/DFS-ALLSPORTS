"""Lineups, chosen by what they are trying to win.

The architecture, and why it is two stages
------------------------------------------
An integer program maximises a linear objective. The things a DFS lineup is
actually trying to maximise are not linear in the player selections:

    cash  - the probability the lineup beats a cash line
    GPP   - the probability the lineup reaches the top fraction of a percent

Both depend on the joint distribution of the players chosen, which is exactly
what a linear objective cannot see. Maximising expected points instead is the
standard amateur substitute, and it is wrong in a specific, costly way: it
picks the same lineup for a double-up and for a million-entry tournament, when
those two contests reward opposite shapes.

So the solver is used for what it is good at - enumerating distinct lineups
that satisfy hard roster rules - and the simulation is used for what it is good
at: telling you which of them wins the contest you are actually entering. Fifty
thousand correlated slate simulations, each candidate scored against the same
draws so the comparison between two lineups is signal rather than noise.

Diversity
---------
For multi-entry, a second lineup that overlaps the first by five of six players
is not a second bet. Each additional lineup is constrained to share at most
`max_overlap` players with every lineup already chosen, which is enforced in
the solver rather than filtered afterwards.

The salary floor
----------------
`roster["min_salary_pct"]` is a FRACTION of the cap the lineup must spend, and
it exists because unspent salary is not free - it is points declined. In MLB
the marginal rate is about 1.16 projected points per $1,000, in NFL about 2.0,
so $1,500 left on the table is one to three points the lineup simply chose not
to buy.

The usual argument for leaving money - it makes the lineup unique - mostly does
not survive contact with a small player pool, because the cheap plays there are
what everybody else rosters in order to afford the expensive ones. Leaving
salary to reach a punt can RAISE duplication while costing points.

A floor that no legal lineup can reach would make the whole slate infeasible,
which is the failure mode this project has already paid for once: one broken
constraint discarded fourteen good boards. So the floor is attempted, and if
the very first lineup cannot be built under it the floor is dropped for the
run, loudly, and the slate still publishes. A board built without the floor is
worth far more than no board at all.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pulp

from . import ownership as OWN
from . import simulate as S

log = logging.getLogger(__name__)


class Infeasible(RuntimeError):
    """No legal lineup exists under these constraints."""


# ---------------------------------------------------------- the salary floor
def salary_floor(roster: dict) -> float:
    """The fewest dollars a lineup may spend, in dollars.

    Stored as a fraction of the cap rather than a dollar amount so the same
    number means the same thing on a $50,000 classic board and on any other
    cap a site might use.
    """
    pct = float(roster.get("min_salary_pct") or 0.0)
    if pct <= 0:
        return 0.0
    if pct > 1.0:                      # somebody passed 98 instead of 0.98
        pct = pct / 100.0
    # Values between 1 and 2 are genuinely ambiguous - 1.4 could mean 1.4% or
    # 140% - and they are read as a PERCENT, so 1.4 is $700 rather than an
    # impossible floor. That direction is deliberate: a floor set too low
    # merely fails to bind, while one set too high would be dropped entirely
    # and the setting would appear to do nothing at all. Nobody asks for a
    # 1.4% floor by accident, and if they do they get a no-op rather than a
    # silent override of the thing they were trying to control.
    pct = min(pct, 1.0)
    # Rounded to whole dollars because salaries are whole dollars, and
    # because the arithmetic otherwise leaves dust: 1.4/100 * 50000 is
    # 699.9999999999999, which every int() in the logs would print as 699.
    return float(round(pct * float(roster["salary_cap"])))


def reachable_salary(pool: pd.DataFrame, roster: dict) -> float:
    """An UPPER BOUND on what any legal lineup could spend.

    Position and team rules can only ever make this smaller, so a floor above
    this number is definitely impossible while a floor below it merely might
    be. That is the right shape for a diagnostic: it never wrongly accuses a
    reachable floor of being unreachable.
    """
    sal = pd.to_numeric(pool["salary"], errors="coerce").dropna()
    if sal.empty:
        return 0.0
    size = len(roster["slots"])
    top = sal.sort_values(ascending=False).head(size)
    total = float(top.sum())
    if "CPT" in roster["slots"]:
        # The captain is charged a multiple, so the priciest player counts
        # for more than his listed salary.
        mult = float(roster.get("captain_multiplier", 1.5))
        total += (mult - 1.0) * float(top.iloc[0])
    return total


# --------------------------------------------------------------- the solver
def _classic_problem(pool: pd.DataFrame, roster: dict,
                     banned: list[list[int]], max_overlap: int,
                     objective: np.ndarray, min_salary: float = 0.0):
    """A classic roster: fixed position counts plus one flex."""
    n = len(pool)
    prob = pulp.LpProblem("lineup", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(n)]

    slots = roster["slots"]
    size = len(slots)
    flex_pos = set(roster["flex_positions"])
    fixed = {}
    for s in slots:
        if s != "FLEX":
            fixed[s] = fixed.get(s, 0) + 1
    n_flex = slots.count("FLEX")

    prob += pulp.lpSum(objective[i] * x[i] for i in range(n))
    prob += pulp.lpSum(x) == size
    spend = pulp.lpSum(float(pool["salary"].iloc[i]) * x[i]
                       for i in range(n))
    prob += spend <= roster["salary_cap"]
    if min_salary > 0:
        prob += spend >= min_salary

    pos = pool["position"].astype(str).tolist()
    for p, count in fixed.items():
        idx = [i for i in range(n) if pos[i] == p]
        lo = count
        hi = count + (n_flex if p in flex_pos else 0)
        prob += pulp.lpSum(x[i] for i in idx) >= lo
        prob += pulp.lpSum(x[i] for i in idx) <= hi

    # Nothing outside the eligible positions may be rostered at all.
    allowed = set(fixed) | flex_pos
    for i in range(n):
        if pos[i] not in allowed:
            prob += x[i] == 0

    if roster.get("max_per_team"):
        for t in pool["team"].astype(str).unique():
            idx = [i for i in range(n) if str(pool["team"].iloc[i]) == t]
            prob += pulp.lpSum(x[i] for i in idx) <= roster["max_per_team"]

    for prev in banned:
        prob += pulp.lpSum(x[i] for i in prev) <= max_overlap

    return prob, x, None


def _showdown_problem(pool: pd.DataFrame, roster: dict,
                      banned: list[list[int]], max_overlap: int,
                      objective: np.ndarray, min_salary: float = 0.0):
    """Showdown: one captain at 1.5x salary and 1.5x points, five flex.

    The captain has to be its own decision variable. Treating it as "the best
    player in the lineup, marked up afterwards" gets the salary wrong - the
    markup has to be inside the cap constraint, not applied to the answer.
    """
    n = len(pool)
    mult = roster.get("captain_multiplier", 1.5)
    prob = pulp.LpProblem("lineup", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(n)]   # flex
    c = [pulp.LpVariable(f"c{i}", cat="Binary") for i in range(n)]   # captain

    size = len(roster["slots"])
    prob += pulp.lpSum(objective[i] * x[i] + mult * objective[i] * c[i]
                       for i in range(n))
    prob += pulp.lpSum(c) == 1
    prob += pulp.lpSum(x) + pulp.lpSum(c) == size
    for i in range(n):
        prob += x[i] + c[i] <= 1          # nobody is his own captain twice

    sal = pool["salary"].astype(float).tolist()
    spend = pulp.lpSum(sal[i] * x[i] + mult * sal[i] * c[i]
                       for i in range(n))
    prob += spend <= roster["salary_cap"]
    if min_salary > 0:
        prob += spend >= min_salary

    # Both teams must be represented: a showdown lineup drawn entirely from one
    # side is legal on some sites and catastrophic on all of them, and on
    # DraftKings the five-per-team cap makes it illegal outright.
    teams = [str(t) for t in pool["team"].astype(str).unique()]
    if roster.get("max_per_team"):
        for t in teams:
            idx = [i for i in range(n) if str(pool["team"].iloc[i]) == t]
            prob += pulp.lpSum(x[i] + c[i] for i in idx) \
                <= roster["max_per_team"]

    for prev in banned:
        prob += pulp.lpSum(x[i] + c[i] for i in prev) <= max_overlap

    return prob, x, c


def solve_one(pool: pd.DataFrame, roster: dict, objective: np.ndarray,
              banned: list[list[int]], max_overlap: int,
              min_salary: float = 0.0) -> tuple[list[int], int | None]:
    """One legal lineup. Returns the player rows and the captain row, if any."""
    showdown = "CPT" in roster["slots"]
    build = _showdown_problem if showdown else _classic_problem
    prob, x, c = build(pool, roster, banned, max_overlap, objective,
                       min_salary)
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        raise Infeasible(f"solver returned {pulp.LpStatus[prob.status]}")

    rows = [i for i in range(len(pool)) if x[i].value() and x[i].value() > 0.5]
    cap = None
    if c is not None:
        caps = [i for i in range(len(pool))
                if c[i].value() and c[i].value() > 0.5]
        cap = caps[0] if caps else None
        rows = rows + ([cap] if cap is not None else [])
    return sorted(rows), cap


def candidates(pool: pd.DataFrame, roster: dict, draws: np.ndarray,
               n_candidates: int = 120, max_overlap: int | None = None,
               rng: np.random.Generator | None = None) -> list[dict]:
    """A varied set of legal lineups to choose between.

    Each is solved on a jittered objective so the pool explores rather than
    returning the same chalk build with one player swapped, and each is cut off
    afterwards so it cannot be produced twice. The jitter is proportional to
    each player's own spread: a volatile player moves more, which is exactly
    where the alternatives worth considering live.
    """
    rng = rng or np.random.default_rng(1729 + 2)
    size = len(roster["slots"])
    max_overlap = size - 1 if max_overlap is None else max_overlap

    mean = draws.mean(axis=1)
    spread = draws.std(axis=1)

    floor = salary_floor(roster)
    if floor > 0:
        cap_total = float(roster["salary_cap"])
        log.info("salary floor $%s (%.1f%% of the $%s cap); an upper bound on "
                 "what any lineup here could spend is $%s",
                 f"{int(floor):,}", 100 * floor / cap_total,
                 f"{int(cap_total):,}",
                 f"{int(reachable_salary(pool, roster)):,}")

    out, banned = [], []
    for k in range(n_candidates):
        obj = mean if k == 0 else mean + rng.normal(0, 1, len(mean)) * spread
        try:
            rows, cap = solve_one(pool, roster, obj, banned, max_overlap,
                                  min_salary=floor)
        except Infeasible:
            if k == 0 and floor > 0:
                # Nothing at all could be built under the floor. Drop it and
                # keep the slate: a board without the floor is worth immensely
                # more than no board, and this is exactly the shape of failure
                # that once threw away fourteen good slates.
                log.error(
                    "NO legal lineup reaches the $%s salary floor, so the "
                    "floor is being IGNORED for this board. The richest "
                    "lineup the position and team rules permit spends about "
                    "$%s. Lower min_salary_pct below %.0f%% to make it bind.",
                    f"{int(floor):,}",
                    f"{int(reachable_salary(pool, roster)):,}",
                    100 * reachable_salary(pool, roster)
                    / float(roster["salary_cap"]))
                floor = 0.0
                continue
            break
        banned.append(rows)
        out.append({"rows": rows, "captain": cap})
    if not out:
        raise Infeasible("no legal lineup could be built from this pool")
    log.info("built %d distinct legal lineups", len(out))
    return out


# ------------------------------------------------------------- the choosing
def score_candidates(cands: list[dict], draws: np.ndarray, roster: dict,
                     cash_line: float, gpp_line: float,
                     own: pd.Series | None = None,
                     field_size: int = 100_000) -> pd.DataFrame:
    """Every candidate's full distribution, against the same simulations.

    `edge` is the number a tournament is actually played for. Reaching the
    prize is worth nothing on its own - it is worth the prize DIVIDED BY the
    people who reach it with the same six players. A lineup of chalk that gets
    there alongside seventeen hundred identical entries is worth a fraction of
    one that gets there alone, and ranking by p_gpp cannot see the difference.
    """
    mult = roster.get("captain_multiplier", 1.5)
    rec = []
    for k, cand in enumerate(cands):
        rows = cand["rows"]
        m = [mult if i == cand["captain"] else 1.0 for i in rows]
        total = S.lineup_scores(draws, rows, m)
        p_gpp = float((total >= gpp_line).mean())

        dupes = float("nan")
        edge = p_gpp
        if own is not None:
            o = own.to_numpy(dtype=float)[rows]
            dupes = OWN.duplication(o, field_size)
            # Split the prize with whoever else fielded it. The +1 is you.
            edge = p_gpp / (1.0 + dupes)

        rec.append({
            "candidate": k,
            "mean": float(total.mean()),
            "median": float(np.median(total)),
            "floor": float(np.quantile(total, 0.10)),
            "ceiling": float(np.quantile(total, 0.99)),
            "p_cash": float((total >= cash_line).mean()),
            "p_gpp": p_gpp,
            "duplicates": dupes,
            "edge": edge,
            "own_sum": (float(own.to_numpy(dtype=float)[rows].sum())
                        if own is not None else float("nan")),
        })
    return pd.DataFrame(rec)


def build(pool: pd.DataFrame, roster: dict, draws: np.ndarray,
          objective: str = "cash", entries: int = 1,
          max_overlap: int | None = None,
          n_candidates: int = 120,
          own: pd.Series | None = None,
          field_size: int = 100_000) -> pd.DataFrame:
    """Lineups for one contest type.

    `objective` is "cash" - maximise the chance of beating the cash line - or
    "gpp", maximise the chance of reaching the top fraction of a percent. They
    genuinely disagree, and a build that returns the same lineup for both is a
    build that has not understood the question.
    """
    if objective not in ("cash", "gpp"):
        raise ValueError(f"objective must be cash or gpp, got {objective!r}")

    size = len(roster["slots"])
    max_overlap = (size - 1 if entries == 1 else max(1, size - 3)) \
        if max_overlap is None else max_overlap

    line_cash = S.cash_line(draws, roster)
    line_gpp = _gpp_line(draws, roster)
    log.info("cash line %.1f, tournament line %.1f (top %.2f%% of a rough "
             "random field)", line_cash, line_gpp,
             100 * S.GPP_TARGET_FRACTION)

    cands = candidates(pool, roster, draws, n_candidates=n_candidates,
                       max_overlap=size - 1)
    scored = score_candidates(cands, draws, roster, line_cash, line_gpp,
                              own=own, field_size=field_size)

    # Cash pays everyone who clears the line, so duplication is irrelevant
    # there - beating half the field is not a prize anyone splits with you.
    # A tournament is the opposite, which is why the two objectives rank on
    # different columns rather than on the same one with a different threshold.
    key = "p_cash" if objective == "cash" else (
        "edge" if own is not None else "p_gpp")
    chosen, used = [], []
    order = scored.sort_values(key, ascending=False)["candidate"].tolist()
    for k in order:
        rows = set(cands[k]["rows"])
        if any(len(rows & set(cands[j]["rows"])) > max_overlap for j in used):
            continue
        used.append(k)
        chosen.append(k)
        if len(chosen) >= entries:
            break

    if not chosen:
        raise Infeasible("no candidate survived the diversity constraint")

    frames = []
    for rank, k in enumerate(chosen, start=1):
        cand = cands[k]
        row = scored[scored["candidate"] == k].iloc[0]
        lu = pool.iloc[cand["rows"]].copy()
        lu["slot"] = ["CPT" if i == cand["captain"] else "FLEX"
                      for i in cand["rows"]]
        mult = roster.get("captain_multiplier", 1.5)
        lu["charged"] = [float(s) * (mult if i == cand["captain"] else 1.0)
                         for s, i in zip(lu["salary"], cand["rows"])]
        lu["entry"] = rank
        lu["objective"] = objective
        for c in ("mean", "median", "floor", "ceiling", "p_cash", "p_gpp",
                  "duplicates", "edge", "own_sum"):
            lu[c] = row[c]
        frames.append(lu)
    return pd.concat(frames, ignore_index=True)


def _gpp_line(draws: np.ndarray, roster: dict) -> float:
    """What it takes to reach the top of a tournament, roughly.

    Estimated the same way as the cash line and with the same honest caveat: a
    field of randomly assembled lineups is softer than a real one, so this
    understates the bar. It is used to RANK candidates, and a bar that is
    uniformly too low still ranks them correctly - which is why an approximate
    line is worth having and an assumed one is not.
    """
    return S.cash_line(draws, roster,
                       fraction=S.GPP_TARGET_FRACTION)


def report(lineups: pd.DataFrame, roster: dict) -> str:
    """One block per entry."""
    lines = []
    floor = salary_floor(roster)
    for entry, g in lineups.groupby("entry"):
        r = g.iloc[0]
        spent = int(g["charged"].sum())
        left = int(roster["salary_cap"]) - spent
        tail = f"  (${left:,} left"
        if floor > 0:
            tail += f", floor ${int(floor):,}"
            if spent < floor:
                tail += " NOT MET - the floor was dropped, see the log"
        tail += ")"
        lines += [
            "",
            f"{r['objective'].upper()}  entry {entry}"
            f"    salary ${spent:,} of "
            f"${roster['salary_cap']:,}{tail}",
            "-" * 70,
            f"{'slot':<6}{'player':<24}{'pos':<5}{'team':<5}"
            f"{'charged':>9}{'median':>8}",
        ]
        order = g.sort_values("slot")          # CPT sorts before FLEX
        for p in order.itertuples(index=False):
            proj = getattr(p, "q50", float("nan"))
            lines.append(
                f"{str(p.slot):<6}{str(p.name)[:23]:<24}{str(p.position):<5}"
                f"{str(p.team):<5}{int(p.charged):>9,}{proj:>8.1f}")
        lines += [
            f"  projected  mean {r['mean']:.1f}   median {r['median']:.1f}   "
            f"floor {r['floor']:.1f}   ceiling {r['ceiling']:.1f}",
            f"  beats the cash line {r['p_cash'] * 100:.1f}% of sims;  "
            f"reaches the tournament bar {r['p_gpp'] * 100:.2f}%",
            (f"  projected ownership {r['own_sum'] * 100:.0f}% across six;  "
             f"about {r['duplicates']:,.0f} other entries field this exact "
             f"lineup" if r.get('duplicates') == r.get('duplicates') else ""),
        ]
    return "\n".join(lines)
