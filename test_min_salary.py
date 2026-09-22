"""The lineup salary floor: its arithmetic, and its refusal to kill a board.

Runs as a script: `python test_min_salary.py`.

Unspent salary is not free. At about 1.16 projected points per $1,000 in MLB
and 2.0 in NFL, $1,500 left on the table is one to three points the lineup
declined to buy. The usual defence - that leaving money makes you unique -
inverts in a small player pool, because the cheap plays you free up salary for
are exactly the ones everybody rosters in order to afford the expensive ones.

So the floor is worth having. What it must never do is empty a board. A
constraint no lineup can satisfy turns a publishable slate into nothing, and
this project has already paid that bill once: a stale injury join produced a
board with no legal lineups, the failure escaped its handler, and fourteen
good slates were discarded with it.

These tests do not need a solver, which is the point - they cover the two
places the floor is decided (what the percentage means in dollars, and whether
it is reachable at all) rather than the place it is enforced.
"""
from __future__ import annotations

import sys

import pandas as pd

from engine.optimise import reachable_salary, salary_floor

CLASSIC = {"slots": ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"],
           "flex_positions": [], "salary_cap": 50_000, "max_per_team": 5}
SHOWDOWN = {"slots": ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"],
            "flex_positions": [], "salary_cap": 50_000,
            "captain_multiplier": 1.5, "max_per_team": 5}


def board(salaries):
    return pd.DataFrame({
        "name": [f"P{i}" for i in range(len(salaries))],
        "position": ["OF"] * len(salaries),
        "team": ["AAA"] * len(salaries),
        "salary": salaries,
    })


# ----------------------------------------------------- what the percent means
def a_fraction_becomes_dollars():
    assert salary_floor({**CLASSIC, "min_salary_pct": 0.98}) == 49_000


def zero_and_missing_both_mean_off():
    assert salary_floor(CLASSIC) == 0.0
    assert salary_floor({**CLASSIC, "min_salary_pct": 0}) == 0.0
    assert salary_floor({**CLASSIC, "min_salary_pct": None}) == 0.0


def somebody_passing_98_instead_of_point_98_is_understood():
    """A percent where a fraction belongs would otherwise set the floor to
    forty-nine TIMES the cap, which no lineup can reach - so the floor would
    silently drop and the setting would appear to do nothing at all."""
    assert salary_floor({**CLASSIC, "min_salary_pct": 98}) == 49_000


def the_floor_can_never_exceed_the_cap():
    assert salary_floor({**CLASSIC, "min_salary_pct": 140}) == 50_000
    assert salary_floor({**CLASSIC, "min_salary_pct": 1.0}) == 50_000


def the_ambiguous_range_is_read_as_a_percent():
    """1.4 could mean 1.4% or 140%, and it is read as 1.4% - $700.

    The direction is the point. A floor set too LOW just fails to bind. A
    floor set too HIGH gets dropped as unreachable, and then the setting
    silently does nothing while appearing to be on.
    """
    assert salary_floor({**CLASSIC, "min_salary_pct": 1.4}) == 700


def the_floor_is_whole_dollars():
    """1.4/100 * 50000 is 699.9999999999999 in binary floating point, and
    every int() that formats it for a log would print 699. Salaries are whole
    dollars, so the floor is too."""
    f = salary_floor({**CLASSIC, "min_salary_pct": 1.4})
    assert f == int(f), f"{f!r} is not a whole number of dollars"
    assert int(f) == 700


# ------------------------------------------------- is the floor even possible
def the_bound_is_the_priciest_legal_roster():
    """Ten slots, so the ten most expensive salaries and nothing else."""
    sal = [6000] * 10 + [3000] * 20
    assert reachable_salary(board(sal), CLASSIC) == 60_000


def a_board_that_cannot_reach_the_cap_says_so():
    """Every player at $3,000 and ten slots is $30,000 - a 98% floor of
    $49,000 is impossible, and the caller needs to know that BEFORE the
    solver returns an unhelpful 'Infeasible'."""
    pool = board([3000] * 30)
    assert reachable_salary(pool, CLASSIC) == 30_000
    assert reachable_salary(pool, CLASSIC) < salary_floor(
        {**CLASSIC, "min_salary_pct": 0.98})


def the_captain_multiplier_counts_toward_the_bound():
    """Six slots at $10,000 is $60,000 flat, but the captain is charged 1.5x,
    so the richest showdown lineup spends $65,000. A bound that missed this
    would call a reachable floor unreachable."""
    assert reachable_salary(board([10_000] * 12), SHOWDOWN) == 65_000


def a_bound_is_an_upper_bound_not_a_promise():
    """Position and team rules can only ever make the real ceiling smaller.
    That asymmetry is the whole point: the bound may fail to catch an
    impossible floor, but it never accuses a possible one."""
    pool = board([9000] * 3 + [2000] * 20)
    # Ignoring positions entirely, the top ten are 9+9+9 then sevens of 2000.
    assert reachable_salary(pool, CLASSIC) == 3 * 9000 + 7 * 2000


def an_empty_board_does_not_explode():
    assert reachable_salary(board([]), CLASSIC) == 0.0


def unreadable_salaries_are_skipped_not_guessed():
    """DraftKings has shipped a blank salary before. A row that cannot be read
    must not become a zero, because a zero would quietly LOWER the bound and
    make a reachable floor look unreachable."""
    pool = board([5000, 5000, 5000])
    pool["salary"] = pool["salary"].astype(object)
    pool.loc[1, "salary"] = "n/a"
    assert reachable_salary(pool, CLASSIC) == 10_000


# --------------------------------------------------- the point of all of this
def a_reachable_floor_and_an_unreachable_one_are_distinguishable():
    """The single decision the publisher makes with these two functions."""
    rich = board([6000] * 15)
    poor = board([3000] * 15)
    floor = salary_floor({**CLASSIC, "min_salary_pct": 0.98})
    assert reachable_salary(rich, CLASSIC) >= floor, "should bind, not drop"
    assert reachable_salary(poor, CLASSIC) < floor, "should drop, not empty"


SUITES = [
    ("WHAT THE PERCENTAGE MEANS", [
        ("a fraction becomes dollars", a_fraction_becomes_dollars),
        ("zero and missing both mean off", zero_and_missing_both_mean_off),
        ("98 is read as 98%, not 9800%",
         somebody_passing_98_instead_of_point_98_is_understood),
        ("the floor never exceeds the cap", the_floor_can_never_exceed_the_cap),
        ("the ambiguous 1-to-2 range is read as a percent",
         the_ambiguous_range_is_read_as_a_percent),
        ("the floor is whole dollars, not float dust",
         the_floor_is_whole_dollars),
    ]),
    ("IS THE FLOOR EVEN POSSIBLE", [
        ("the bound is the priciest legal roster",
         the_bound_is_the_priciest_legal_roster),
        ("a board too cheap to reach the floor is detectable",
         a_board_that_cannot_reach_the_cap_says_so),
        ("the captain multiplier counts toward the bound",
         the_captain_multiplier_counts_toward_the_bound),
        ("it is an upper bound, and deliberately loose",
         a_bound_is_an_upper_bound_not_a_promise),
        ("an empty board does not explode", an_empty_board_does_not_explode),
        ("unreadable salaries are skipped",
         unreadable_salaries_are_skipped_not_guessed),
    ]),
    ("BIND OR DROP, NEVER EMPTY", [
        ("the two cases are distinguishable before the solver runs",
         a_reachable_floor_and_an_unreachable_one_are_distinguishable),
    ]),
]


def main() -> int:
    passed = failed = 0
    for title, cases in SUITES:
        print(f"\n{title}")
        print("-" * 68)
        for name, fn in cases:
            try:
                fn()
                print(f"  ok    {name}")
                passed += 1
            except Exception as exc:                           # noqa: BLE001
                print(f"  FAIL  {name}\n        {exc}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
