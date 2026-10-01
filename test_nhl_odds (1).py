"""Checks for the market-implied team totals. No network.

The parsing is checked against a payload shaped the way The Odds API
documents it; the arithmetic is checked against numbers you can verify by
hand or by intuition. The parse is the part that could be wrong about the
real feed, which is why `--probe` exists and why these fixtures are the
documented shape rather than an invented one.

    python test_nhl_odds.py
"""
from __future__ import annotations

import sys

import nhl_odds as O

FAIL = []


def ok(cond, msg):
    print(("  ok  " if cond else "FAIL  ") + msg)
    if not cond:
        FAIL.append(msg)


def near(a, b, tol=0.02):
    return abs(float(a) - float(b)) <= tol


# ------------------------------------------------------------- the maths
def test_implied_and_devig():
    ok(near(O.implied(-110), 0.5238, 1e-3), "-110 is a 52.4% chance")
    ok(near(O.implied(100), 0.5, 1e-9), "+100 is a coin flip")
    ok(near(O.implied(150), 0.4, 1e-9), "+150 is 40%")
    ok(near(O.implied(-200), 0.6667, 1e-3), "-200 is 66.7%")
    ok(O.implied(None) is None and O.implied("x") is None,
       "a missing price is None, not zero")
    # A -110 / -110 market is 104.8% before the vig comes out, 50/50 after.
    ok(near(O.devig(O.implied(-110), O.implied(-110)), 0.5, 1e-9),
       "a balanced market de-vigs to exactly 50%")
    p = O.devig(O.implied(-200), O.implied(170))
    ok(0.62 < p < 0.66, f"a -200 favourite de-vigs to about 64% ({p:.3f})")
    ok(O.devig(0.9, 0.9) is None,
       "a market summing to 180% is refused rather than normalised")


def test_split_is_consistent():
    """The solve must reproduce the inputs it was given - that is the whole
    claim being made."""
    for total, p in ((6.5, 0.50), (6.0, 0.60), (5.5, 0.70), (7.0, 0.42),
                     (6.5, 0.35)):
        h, a = O.split_total(total, p)
        ok(near(h + a, total, 1e-6),
           f"total {total} p {p:.2f}: the halves add back to the total")
        got = O.home_win_prob(h, a)
        ok(near(got, p, 0.01),
           f"total {total} p {p:.2f}: implied win prob recovered ({got:.3f})")


def test_split_is_sane():
    even = O.split_total(6.5, 0.5)
    ok(near(even[0], even[1], 1e-6), "a coin-flip game splits evenly (3.25 each)")
    fav, dog = O.split_total(6.0, 0.65)
    ok(fav > dog, "the favourite gets the larger share")
    ok(3.0 < fav < 3.8 and 2.2 < dog < 3.0,
       f"and the numbers are hockey numbers ({fav:.2f} / {dog:.2f})")
    # The thing a stack decision actually turns on: the spread between the
    # best and worst spot on a slate.
    hi = O.split_total(7.5, 0.68)[0]
    lo = O.split_total(5.5, 0.38)[0]
    ok(hi / lo > 1.6,
       f"best to worst spot spans {hi/lo:.2f}x, against the 1.35x the "
       f"projection model alone can express")
    ok(O.home_win_prob(3.25, 3.25) > 0.49
       and O.home_win_prob(3.25, 3.25) < 0.51,
       "equal means give an even game once ties are split")


def test_team_codes():
    ok(O.code_for("Colorado Avalanche") == "COL", "full names map")
    ok(O.code_for("Montréal Canadiens") == "MTL", "and accented ones")
    ok(O.code_for("St. Louis Blues") == "STL" and O.code_for("St Louis Blues") == "STL",
       "both St Louis spellings")
    ok(O.code_for("Utah Mammoth") == "UTA" and O.code_for("Arizona Coyotes") == "UTA",
       "the relocated franchise maps either way")
    ok(O.code_for("Not A Team") is None, "an unknown name is None, not a guess")
    ok(len(set(O.TEAM_CODES.values())) == 32,
       f"all 32 clubs are covered ({len(set(O.TEAM_CODES.values()))})")


# ------------------------------------------------------------- the parsing
def payload():
    """The documented v4 shape."""
    def game(gid, home, away, total, hprice, aprice):
        return {
            "id": gid, "sport_key": "icehockey_nhl", "sport_title": "NHL",
            "commence_time": "2026-10-01T23:00:00Z",
            "home_team": home, "away_team": away,
            "bookmakers": [
                {"key": "draftkings", "title": "DraftKings",
                 "last_update": "2026-10-01T18:00:00Z",
                 "markets": [
                     {"key": "totals", "last_update": "2026-10-01T18:00:00Z",
                      "outcomes": [{"name": "Over", "price": -110, "point": total},
                                   {"name": "Under", "price": -110, "point": total}]},
                     {"key": "h2h", "last_update": "2026-10-01T18:00:00Z",
                      "outcomes": [{"name": home, "price": hprice},
                                   {"name": away, "price": aprice}]}]},
                {"key": "fanduel", "title": "FanDuel",
                 "last_update": "2026-10-01T18:00:00Z",
                 "markets": [
                     {"key": "totals", "last_update": "2026-10-01T18:00:00Z",
                      "outcomes": [{"name": "Over", "price": -105, "point": total + 0.5},
                                   {"name": "Under", "price": -115, "point": total + 0.5}]},
                     {"key": "h2h", "last_update": "2026-10-01T18:00:00Z",
                      "outcomes": [{"name": home, "price": hprice - 5},
                                   {"name": away, "price": aprice + 5}]}]}],
        }
    return [
        # A shootout spot: high total, Colorado a solid favourite.
        game("g1", "Colorado Avalanche", "Chicago Blackhawks", 6.5, -190, 160),
        # A grind: low total, near coin flip.
        game("g2", "Los Angeles Kings", "Minnesota Wild", 5.0, -105, -115),
        game("g3", "Toronto Maple Leafs", "Montreal Canadiens", 6.0, -140, 120),
    ]


def test_parse():
    df = O.team_totals(payload())
    ok(len(df) == 6, f"two rows per game ({len(df)})")
    ok(set(df["team"]) == {"COL", "CHI", "LAK", "MIN", "TOR", "MTL"},
       "every team is coded")
    col = df[df["team"] == "COL"].iloc[0]
    chi = df[df["team"] == "CHI"].iloc[0]
    ok(near(col["implied_goals"] + chi["implied_goals"], col["game_total"], 0.01),
       "a game's two team totals add to its game total")
    ok(col["implied_goals"] > chi["implied_goals"],
       f"Colorado, the favourite in the high-total game, has the bigger "
       f"number ({col['implied_goals']:.2f} vs {chi['implied_goals']:.2f})")
    ok(near(col["game_total"], 6.75, 0.01),
       f"the total is the MEDIAN across books ({col['game_total']})")

    # The point of the whole exercise: does the market separate the shootout
    # spot from the grind by more than the projection model can?
    lak = df[df["team"] == "LAK"].iloc[0]
    spread = col["implied_goals"] / lak["implied_goals"]
    ok(spread > 1.25,
       f"the best spot projects {spread:.2f}x the goals of the worst, which "
       f"the 1.35x-capped opponent factor could never have said")
    ok(col["factor"] > 1.0 > lak["factor"],
       f"and the multipliers straddle one (COL x{col['factor']:.3f}, "
       f"LAK x{lak['factor']:.3f})")
    ok(df["factor"].between(*O.TEAM_TOTAL_CLIP).all(), "every factor is clipped")
    ok(near(df["is_home"].sum(), 3), "three home teams in three games")


def test_parse_survives_damage():
    """A slate where one game has no moneyline and one team is unknown."""
    p = payload()
    for bk in p[1]["bookmakers"]:
        bk["markets"] = [m for m in bk["markets"] if m["key"] != "h2h"]
    p[2]["home_team"] = "Toronto Maple Leaves"          # misspelt
    df = O.team_totals(p)
    ok(len(df) == 4, f"the unmappable game is skipped, the rest survive ({len(df)})")
    lak = df[df["team"] == "LAK"].iloc[0]
    min_ = df[df["team"] == "MIN"].iloc[0]
    ok(near(lak["implied_goals"], min_["implied_goals"], 0.01),
       "with no moneyline the total splits evenly rather than failing")
    ok("TOR" not in set(df["team"]), "and the misspelt club is absent, not wrong")


def test_no_games():
    try:
        O.team_totals([])
        ok(False, "an empty slate raises")
    except O.OddsUnavailable:
        ok(True, "an empty slate raises OddsUnavailable rather than returning junk")


def test_key_is_required():
    import os
    saved = os.environ.pop("ODDS_API_KEY", None)
    try:
        O.fetch()
        ok(False, "a missing key raises")
    except O.OddsUnavailable as exc:
        ok("ODDS_API_KEY" in str(exc),
           "a missing key explains itself and names the variable")
    except Exception:                                          # noqa: BLE001
        ok(False, "a missing key raises OddsUnavailable, not something else")
    finally:
        if saved is not None:
            os.environ["ODDS_API_KEY"] = saved


if __name__ == "__main__":
    for fn in (test_implied_and_devig, test_split_is_consistent,
               test_split_is_sane, test_team_codes, test_parse,
               test_parse_survives_damage, test_no_games, test_key_is_required):
        print("\n" + fn.__name__)
        fn()
    print("\n" + (f"{len(FAIL)} FAILURES" if FAIL else "all checks passed"))
    sys.exit(1 if FAIL else 0)
