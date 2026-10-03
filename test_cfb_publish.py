"""The CFB publisher, run end to end against synthetic data.

Why this exists in this shape
-----------------------------
Every other check in this project tests a function. This one runs `main()` -
the whole thing: slates, boards, the team map, the name join, the projection
merge, the market, ownership, the simulation, both integer programs, the
payload, the JSON gate and the manifest. The reason is that four of the five
worst bugs in this project were in the SEAMS rather than in any function: a
column name that collided on a merge, a position vocabulary that disagreed
between the frame a model was fitted on and the board it was applied to, a
slot name one module knew and another did not.

A unit test cannot see a seam. So the network calls are stubbed and
everything else is the real code, including the real optimiser.

What it deliberately does NOT claim
-----------------------------------
It says nothing about whether the model is any good. That is
`cfb_run_grade.py`'s job, walk-forward, against baselines. This says the
pipeline is wired correctly and that a payload the page can read comes out of
the far end.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import cfb_cache as C
import cfb_data as D
import cfb_odds as CO
import cfb_publish as P
import cfb_sport as S

PASS, FAIL = [], []


def ok(label, cond, detail=""):
    (PASS if cond else FAIL).append(label)
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}"
          + (f"   {detail}" if detail else ""))


def head(t):
    print()
    print("-" * 70)
    print(t)
    print("-" * 70)


# --------------------------------------------------------------- the fixture
SCHOOLS = ["Alabama", "Auburn", "Ohio State", "Michigan", "Texas",
           "Oklahoma", "Georgia", "Florida"]
CODES = {"Alabama": "ALA", "Auburn": "AUB", "Ohio State": "OSU",
         "Michigan": "MICH", "Texas": "TEX", "Oklahoma": "OU",
         "Georgia": "UGA", "Florida": "FLA"}
FIXTURES = [("Auburn", "Alabama"), ("Michigan", "Ohio State"),
            ("Oklahoma", "Texas"), ("Florida", "Georgia")]
MASCOT = {"Alabama": "Alabama Crimson Tide", "Auburn": "Auburn Tigers",
          "Ohio State": "Ohio State Buckeyes",
          "Michigan": "Michigan Wolverines", "Texas": "Texas Longhorns",
          "Oklahoma": "Oklahoma Sooners", "Georgia": "Georgia Bulldogs",
          "Florida": "Florida Gators"}

# Two quarterbacks, four backs and seven receivers per school: enough depth
# that the competition term has a group to work on, and enough that the
# per-position ceiling has something to bind against.
SQUAD = [("QB", 2), ("RB", 4), ("WR", 7)]
SEASONS = [2024, 2025, 2026]
WEEKS = 9
SEASON = 2026


def roster_rows():
    rows = []
    for school in SCHOOLS:
        for pos, n in SQUAD:
            for i in range(n):
                rows.append({"school": school, "position": pos,
                             "athlete_id": f"{CODES[school]}{pos}{i}",
                             "name": f"{CODES[school]} {pos}man{i}",
                             "depth": i})
    return pd.DataFrame(rows)


def synth_history(rng) -> pd.DataFrame:
    """Player-games shaped exactly like `cfb_cache.load` returns them.

    The production is deliberately NOT uniform: a first-stringer gets most of
    the work and a fourth gets scraps, so the fitted model has something to
    find and the projections come out spread rather than flat. A flat fixture
    makes every downstream assertion vacuous - "the stack is better than no
    stack" cannot fail if every player is the same player.
    """
    roster = roster_rows()
    # Team strength, so the market has something to agree and disagree with.
    strength = {s: 0.75 + 0.5 * i / (len(SCHOOLS) - 1)
                for i, s in enumerate(SCHOOLS)}
    rows = []
    for season in SEASONS:
        for week in range(1, WEEKS + 1):
            for away, home in FIXTURES:
                gid = f"{season}{week:02d}{CODES[home]}"
                for school, opp, is_home in ((home, away, 1), (away, home, 0)):
                    sq = roster[roster["school"] == school]
                    for r in sq.itertuples(index=False):
                        share = {0: 1.0, 1: 0.45, 2: 0.22, 3: 0.10,
                                 4: 0.07, 5: 0.05, 6: 0.04}[r.depth]
                        lam = strength[school] * share
                        row = {"game_id": gid, "school": school,
                               "opponent": opp, "is_home": is_home,
                               "athlete_id": r.athlete_id, "name": r.name,
                               "position": r.position,
                               "season": season, "week": week}
                        for f in D.STAT_FIELDS:
                            row[f] = 0.0
                        if r.position == "QB":
                            row["pass_yards"] = max(
                                0.0, rng.normal(300 * lam, 70 * lam))
                            row["pass_td"] = rng.poisson(2.6 * lam)
                            row["interception"] = rng.poisson(0.8 * lam)
                            row["completions"] = rng.poisson(22 * lam)
                            row["carries"] = rng.poisson(5 * lam)
                            row["rush_yards"] = max(
                                0.0, rng.normal(25 * lam, 20))
                        elif r.position == "RB":
                            row["carries"] = rng.poisson(18 * lam)
                            row["rush_yards"] = max(
                                0.0, rng.normal(90 * lam, 40 * lam))
                            row["rush_td"] = rng.poisson(0.9 * lam)
                            row["rec"] = rng.poisson(2 * lam)
                            row["rec_yards"] = max(
                                0.0, rng.normal(15 * lam, 12))
                        else:
                            row["rec"] = rng.poisson(6 * lam)
                            row["rec_yards"] = max(
                                0.0, rng.normal(80 * lam, 40 * lam))
                            row["rec_td"] = rng.poisson(0.7 * lam)
                        rows.append(row)
    df = pd.DataFrame(rows)
    df["points"] = D.fantasy_points(df)
    df["athlete_id"] = df["athlete_id"].astype(str)
    df["game_id"] = df["game_id"].astype(str)
    return df


def synth_board(hist, rng, dg=11111, games=FIXTURES) -> pd.DataFrame:
    """A board shaped exactly like `cfb_data.board` returns one.

    Priced off the history so the model's own view of each team is coherent
    with its salaries, and with the names spelled slightly DIFFERENTLY from
    the history on a couple of rows - a middle initial on the board that CFBD
    does not carry - because that is the join this build actually has to
    survive and a fixture where every spelling is identical tests nothing.
    """
    per = hist.groupby("athlete_id")["points"].mean()
    roster = roster_rows()
    live = {s for fx in games for s in fx}
    opp = {}
    for away, home in games:
        opp[home], opp[away] = away, home

    rows = []
    for r in roster.itertuples(index=False):
        if r.school not in live:
            continue
        ppg = float(per.get(r.athlete_id, 0.0))
        name = r.name
        if r.depth == 0 and r.position == "WR":
            # "ALA WRman0" on the board, "ALA WRman0" in CFBD plus a middle
            # token: the one difference no exact key reduction can bridge.
            name = name.replace(" ", " J. ", 1)
        a, h = opp[r.school], r.school
        home_side = any(h == home for _, home in games if home == r.school)
        rows.append({
            "dk_player_id": f"dk-{r.athlete_id}",
            "name": name, "position": r.position, "team": CODES[r.school],
            "opponent": CODES[opp[r.school]],
            "is_home": 1 if home_side else 0,
            "salary": int(np.clip(2500 + 420 * ppg, 2500, 11500)),
            "dk_points_per_game": round(ppg, 1),
            "disabled": False, "roster_slot": 1,
            "game": (f"{CODES[opp[r.school]]} @ {CODES[r.school]}"
                     if home_side
                     else f"{CODES[r.school]} @ {CODES[opp[r.school]]}"),
            "is_captain": 0,
        })
    # Three tight ends and a kicker, which a real board does not price - here
    # to prove `roster_position` drops them rather than finding them a slot.
    for extra in ("TE", "TE", "TE", "K"):
        rows.append({"dk_player_id": f"dk-x{extra}{len(rows)}",
                     "name": f"ALA {extra}man", "position": extra,
                     "team": "ALA", "opponent": "AUB", "is_home": 0,
                     "salary": 3000, "dk_points_per_game": 4.0,
                     "disabled": False, "roster_slot": 1,
                     "game": "ALA @ AUB", "is_captain": 0})
    df = pd.DataFrame(rows)
    df["charged_salary"] = df["salary"]
    df["keys"] = df["name"].map(D.name_keys)
    df["key"] = df["name"].map(D.primary_key)
    return df.reset_index(drop=True)


def synth_cfbd_games(games=FIXTURES):
    return [{"id": 9000 + i, "home_team": h, "away_team": a,
             "week": 10, "season": SEASON}
            for i, (a, h) in enumerate(games)]


def synth_teams():
    return [{"school": s, "abbreviation": CODES[s],
             "alternateNames": [s]} for s in SCHOOLS]


def synth_odds(games=FIXTURES, totals=None):
    """The Odds API payload shape, with mascots on the names."""
    totals = totals or {}
    out = []
    for i, (a, h) in enumerate(games):
        total, point_home = totals.get(h, (56.5, -10.5))
        out.append({
            "id": f"g{i}",
            "commence_time": (pd.Timestamp.now("UTC")
                              + pd.Timedelta(days=2)).strftime(
                                  "%Y-%m-%dT%H:%M:%SZ"),
            "home_team": MASCOT[h], "away_team": MASCOT[a],
            "bookmakers": [{
                "key": f"book{b}",
                "markets": [
                    {"key": "totals",
                     "outcomes": [{"name": "Over", "point": total},
                                  {"name": "Under", "point": total}]},
                    {"key": "spreads",
                     "outcomes": [{"name": MASCOT[h], "point": point_home},
                                  {"name": MASCOT[a], "point": -point_home}]},
                ]} for b in range(3)],
        })
    return out


# ------------------------------------------------------------------- harness
class Stub:
    """Everything that would touch the network, replaced."""

    def __init__(self, board, odds=None, games=None):
        self.board, self.odds = board, odds
        self.games = games if games is not None else synth_cfbd_games()
        self.calls = []

    def install(self):
        self.old = {}
        for mod, name, fn in (
            (D, "slates", self.slates),
            (D, "board", self.board_fn),
            (D, "live_week", lambda key, season: 9),
            (D, "upcoming_games", self.upcoming),
            (D, "cfbd", self.cfbd),
            (CO, "fetch", self.fetch_odds),
        ):
            self.old[(mod, name)] = getattr(mod, name)
            setattr(mod, name, fn)

    def restore(self):
        for (mod, name), fn in self.old.items():
            setattr(mod, name, fn)

    def slates(self):
        return pd.DataFrame([
            {"draft_group": 11111, "contests": 40, "game_type": "Classic",
             "starts_text": "Sat 12:00PM ET", "biggest_prize": 100000,
             "biggest_field": 50000, "example": "CFB $100K Saturday"},
            {"draft_group": 22222, "contests": 900,
             "game_type": "Showdown Captain Mode",
             "starts_text": "Sat 3:30PM ET", "biggest_prize": 5000,
             "biggest_field": 1000, "example": "ALA vs AUB Showdown"},
        ])

    def board_fn(self, dg, captain_multiplier=None):
        self.calls.append(("board", dg))
        if int(dg) != 11111:
            raise D.Unavailable(f"draft group {dg} is not the test board")
        return self.board.copy()

    def upcoming(self, key, season, after_week, span=2):
        return self.games

    def cfbd(self, path, key, **params):
        self.calls.append(("cfbd", path))
        if path == "teams":
            return synth_teams()
        raise D.Unavailable(f"no stub for {path}")

    def fetch_odds(self, key=None, markets=None):
        if self.odds is None:
            raise CO.OddsUnavailable("no odds in this test")
        return self.odds


def run_publish(stub, tmp, argv):
    """Run main() with docs/data pointed at a temporary directory.

    A placeholder API key is set because `main` refuses to start without one -
    correctly, since without it the team map cannot be solved. Every call that
    would use it is stubbed, so the value is never sent anywhere. It is a
    literal nonsense string on purpose: a test that reads a real key out of the
    environment is a test that passes on one machine.
    """
    import os
    old_docs, old_data = P.DOCS, P.DATA
    old_key = os.environ.get("CFBD_API_KEY")
    P.DOCS, P.DATA = Path(tmp), Path(tmp) / "data"
    os.environ["CFBD_API_KEY"] = "not-a-real-key-every-call-is-stubbed"
    stub.install()
    try:
        rc = P.main(argv)
    finally:
        stub.restore()
        P.DOCS, P.DATA = old_docs, old_data
        if old_key is None:
            os.environ.pop("CFBD_API_KEY", None)
        else:
            os.environ["CFBD_API_KEY"] = old_key
    return rc


def payload_of(tmp):
    files = sorted((Path(tmp) / "data").glob("cfb_dk_classic_*.json"))
    assert files, "no payload written"
    return json.loads(files[0].read_text())


# --------------------------------------------------------------------- tests
def test_roster_encoding():
    head("THE ROSTER, AND WHY TWO FLEXES ARE ONE CONSTRAINT")
    R = S.DK_CLASSIC
    ok("eight slots, $50,000", len(R["slots"]) == 8
       and R["salary_cap"] == 50_000, f"{'/'.join(R['slots'])}")
    ok("quarterbacks capped at two", R["max_position"]["QB"] == 2)
    ok("at most seven from one game, which IS 'at least two games'",
       R["max_per_game"] == len(R["slots"]) - 1)

    # Enumerate every composition the encoding admits and check each one maps
    # onto a legal assignment of the eight NAMED slots. This is the claim the
    # whole simplification rests on, so it is enumerated rather than argued.
    admitted, legal = [], []
    for q in range(0, 4):
        for rb in range(0, 9):
            for wr in range(0, 9):
                if q + rb + wr != 8:
                    continue
                if q < 1 or rb < 2 or wr < 3:
                    continue
                if q > R["max_position"]["QB"]:
                    continue
                admitted.append((q, rb, wr))
                # FLEX takes RB/WR, SFLEX takes QB/RB/WR. One spare QB can
                # only go to SFLEX; the other spare must be an RB or a WR.
                spare_q, spare_rb, spare_wr = q - 1, rb - 2, wr - 3
                if spare_q + spare_rb + spare_wr != 2:
                    continue
                if spare_q <= 1 and (spare_rb + spare_wr) == 2 - spare_q:
                    legal.append((q, rb, wr))
    ok("every admitted composition is enterable",
       admitted and admitted == legal,
       f"{len(admitted)} compositions: {admitted}")
    ok("three quarterbacks is not admitted",
       not any(q > 2 for q, _, _ in admitted))


def test_check_entry_can_fail():
    head("check_entry CAN FAIL")
    good = pd.DataFrame({
        "name": list("abcdefgh"),
        "position": ["QB", "QB", "RB", "RB", "WR", "WR", "WR", "WR"],
        "team": ["ALA", "AUB", "ALA", "OSU", "ALA", "AUB", "OSU", "MICH"],
        "game": ["ALA v AUB"] * 2 + ["ALA v AUB", "MICH v OSU"]
                + ["ALA v AUB", "ALA v AUB", "MICH v OSU", "MICH v OSU"],
        "salary": [6000] * 8})
    ok("a legal lineup reports nothing", S.check_entry(good, S.DK_CLASSIC) == [],
       str(S.check_entry(good, S.DK_CLASSIC)))

    three_qb = good.copy()
    three_qb.loc[2, "position"] = "QB"
    ok("three quarterbacks is caught",
       any("QB" in m for m in S.check_entry(three_qb, S.DK_CLASSIC)))

    one_game = good.copy()
    one_game["game"] = "ALA v AUB"
    ok("one game is caught",
       any("game" in m for m in S.check_entry(one_game, S.DK_CLASSIC)))

    over = good.copy()
    over["salary"] = 7000
    ok("over the cap is caught",
       any("cap" in m for m in S.check_entry(over, S.DK_CLASSIC)))

    short = good.iloc[:7].copy()
    ok("seven players is caught",
       any("not 8" in m for m in S.check_entry(short, S.DK_CLASSIC)))

    te = good.copy()
    te.loc[7, "position"] = "TE"
    ok("a tight end has no slot and is caught",
       any("TE" in m for m in S.check_entry(te, S.DK_CLASSIC)))

    nogame = good.drop(columns=["game"])
    ok("a lineup with no game column says the rule could not be checked",
       any("could not be checked" in m
           for m in S.check_entry(nogame, S.DK_CLASSIC)))


def test_odds_sign():
    head("THE ODDS SIGN, AGAINST A WORKED EXAMPLE")
    # Alabama at home, favoured by 10.5, in a 56.5-point game.
    #   spread_line = +10.5, home_implied = (56.5 + 10.5) / 2 = 33.5
    games = synth_odds([("Auburn", "Alabama")], {"Alabama": (56.5, -10.5)})
    lines = CO.week_lines(games, SCHOOLS)
    by = lines.set_index("school")
    ok("the home favourite gets the bigger number",
       abs(by.loc["Alabama", "implied_total"] - 33.5) < 1e-6,
       f"Alabama {by.loc['Alabama', 'implied_total']:.2f}, "
       f"Auburn {by.loc['Auburn', 'implied_total']:.2f}")
    ok("the two implied totals sum to the game total",
       abs(by.loc["Alabama", "implied_total"]
           + by.loc["Auburn", "implied_total"] - 56.5) < 1e-6)
    ok("the home favourite's spread is POSITIVE here",
       by.loc["Alabama", "team_spread"] > 0,
       f"{by.loc['Alabama', 'team_spread']:+.1f}")

    # And the mirror: a ROAD favourite must also get the bigger number. This
    # is the case a sign error passes, because flipping the sign still
    # produces a plausible-looking split.
    games = synth_odds([("Auburn", "Alabama")], {"Alabama": (56.5, +10.5)})
    by = CO.week_lines(games, SCHOOLS).set_index("school")
    ok("a ROAD favourite gets the bigger number too",
       by.loc["Auburn", "implied_total"] > by.loc["Alabama", "implied_total"],
       f"Auburn {by.loc['Auburn', 'implied_total']:.2f} vs "
       f"Alabama {by.loc['Alabama', 'implied_total']:.2f}")


def test_name_matching():
    head("MASCOT NAMES, MATCHED WITHOUT A 134-ROW TABLE")
    feed = ["Alabama Crimson Tide", "Ohio State Buckeyes",
            "Michigan Wolverines", "Texas A&M Aggies", "Ole Miss Rebels",
            "San Jose State Spartans", "Hawai'i Rainbow Warriors",
            "Miami (OH) RedHawks", "Michigan State Spartans",
            "Louisiana Ragin' Cajuns"]
    schools = ["Alabama", "Ohio State", "Michigan", "Michigan State",
               "Texas A&M", "Ole Miss", "San José State", "Hawai'i",
               "Miami", "Miami (OH)", "Louisiana", "Louisiana Monroe"]
    got = CO.match_schools(feed, schools)
    want = {"Alabama Crimson Tide": "Alabama",
            "Ohio State Buckeyes": "Ohio State",
            "Michigan Wolverines": "Michigan",
            "Michigan State Spartans": "Michigan State",
            "Texas A&M Aggies": "Texas A&M",
            "Ole Miss Rebels": "Ole Miss",
            "San Jose State Spartans": "San José State",
            "Hawai'i Rainbow Warriors": "Hawai'i",
            "Miami (OH) RedHawks": "Miami (OH)",
            "Louisiana Ragin' Cajuns": "Louisiana"}
    for k, v in want.items():
        ok(f"{k[:34]:<34} -> {v}", got.get(k) == v, f"got {got.get(k)!r}")
    ok("Michigan State did not swallow Michigan",
       got.get("Michigan Wolverines") == "Michigan")
    ok("Louisiana Monroe did not swallow Louisiana",
       got.get("Louisiana Ragin' Cajuns") == "Louisiana")


def test_market_factor_does_not_double_count():
    head("THE MARKET FACTOR, WHICH MUST NOT COUNT A GOOD OFFENCE TWICE")
    # A pool where the model already rates ALA 1.5x AUB, and a market that
    # says exactly the same thing. Nothing should move.
    rows = []
    for team, scale in (("ALA", 1.5), ("AUB", 1.0)):
        for i in range(8):
            rows.append({"team": team, "mean": 10.0 * scale,
                         "median": 10.0 * scale})
    pool = pd.DataFrame(rows)
    agree = pd.DataFrame({"team": ["ALA", "AUB"],
                          "implied_total": [36.0, 24.0]})
    f = P.market_factor(pool, agree)
    ok("a market that AGREES with the model moves nothing",
       bool(np.allclose(f.to_numpy(), 1.0, atol=1e-6)),
       f"factors {sorted(set(f.round(4)))}")

    # Same model, a market that rates ALA higher still.
    disagree = pd.DataFrame({"team": ["ALA", "AUB"],
                             "implied_total": [40.0, 20.0]})
    f2 = P.market_factor(pool, disagree)
    ala = float(f2[pool["team"] == "ALA"].iloc[0])
    aub = float(f2[pool["team"] == "AUB"].iloc[0])
    ok("a market that likes ALA MORE than the model lifts ALA", ala > 1.001,
       f"ALA x{ala:.3f}")
    ok("and fades AUB", aub < 0.999, f"AUB x{aub:.3f}")
    ok("the factors are clipped", P.MARKET_CLIP[0] <= min(ala, aub)
       and max(ala, aub) <= P.MARKET_CLIP[1])

    naive = pd.Series([40.0 / 30.0, 20.0 / 30.0])
    ok("and it is NOT the naive implied-total ratio, which would be the "
       "double count", abs(ala - float(naive.iloc[0])) > 0.05,
       f"ratio-to-ratio {ala:.3f} vs naive {float(naive.iloc[0]):.3f}")

    ok("no market at all leaves everything alone",
       bool(np.allclose(P.market_factor(pool, None).to_numpy(), 1.0)))
    ok("one team with a line is not enough to scale a board",
       bool(np.allclose(P.market_factor(
           pool, pd.DataFrame({"team": ["ALA"], "implied_total": [36.0]})
       ).to_numpy(), 1.0)))


def test_regressions():
    """Defects an adversarial review found after everything above passed.

    Each one was silent: a wrong number published under a log line saying the
    step had worked.
    """
    head("REGRESSIONS")

    # 1. A SHORT CANDIDATE LIST HANDS A TEAM ANOTHER GAME'S LINE.
    #
    # The matcher resolves a feed name by longest prefix, which keeps
    # "Michigan State Spartans" off Michigan ONLY IF Michigan State is in the
    # candidate list. Given just a board's two dozen schools it matched the
    # nearest school that happened to be present - so Michigan was published
    # in Michigan State's game, implied 12.0 instead of 28.0, under a log line
    # reading "market names resolved: 4 of 4".
    short = ["Michigan", "Indiana"]
    full = ["Michigan", "Michigan State", "Indiana", "Ohio State"]
    feed = ["Michigan State Spartans", "Ohio State Buckeyes",
            "Michigan Wolverines", "Indiana Hoosiers"]
    got_short = CO.match_schools(feed, short)
    got_full = CO.match_schools(feed, full)
    ok("a SHORT school list is what produced the wrong answer",
       got_short.get("Michigan State Spartans") == "Michigan",
       f"got {got_short.get('Michigan State Spartans')!r} - this is the bug "
       f"being guarded against, not a pass of the matcher")
    ok("the FULL list resolves it correctly",
       got_full.get("Michigan State Spartans") == "Michigan State"
       and got_full.get("Michigan Wolverines") == "Michigan")
    ok("for_board now REQUIRES the school list as an argument",
       "schools" in CO.for_board.__code__.co_varnames
       and CO.for_board.__defaults__ is not None
       and len(CO.for_board.__defaults__) == 2,
       "so it cannot silently default to the board's own two dozen")

    # The prefix traps a real FBS list is full of.
    pairs = [
        ("Florida State Seminoles", "Florida State"),
        ("Texas Tech Red Raiders", "Texas Tech"),
        ("Texas A&M Aggies", "Texas A&M"),
        ("Washington State Cougars", "Washington State"),
        ("Georgia Tech Yellow Jackets", "Georgia Tech"),
        ("West Virginia Mountaineers", "West Virginia"),
        ("Northern Illinois Huskies", "Northern Illinois"),
        ("Middle Tennessee Blue Raiders", "Middle Tennessee"),
        ("Miami (OH) RedHawks", "Miami (OH)"),
        ("Louisiana Monroe Warhawks", "Louisiana Monroe"),
        ("Oregon State Beavers", "Oregon State"),
        ("San Diego State Aztecs", "San Diego State"),
        ("Appalachian State Mountaineers", "Appalachian State"),
        ("Coastal Carolina Chanticleers", "Coastal Carolina"),
    ]
    schools = sorted({s for _, s in pairs} | {
        "Florida", "Texas", "Washington", "Georgia", "Virginia", "Illinois",
        "Tennessee", "Miami", "Louisiana", "Oregon", "San Diego",
        "Appalachian", "Carolina", "Michigan", "Michigan State", "Ohio",
        "Ohio State", "Houston", "Sam Houston", "Ole Miss",
        "Southern Mississippi", "North Carolina", "NC State"})
    res = CO.match_schools([f for f, _ in pairs], schools)
    wrong = [(f, res.get(f), want) for f, want in pairs if res.get(f) != want]
    ok("no shorter school swallows a longer one", not wrong,
       "; ".join(f"{f} -> {g!r} not {w!r}" for f, g, w in wrong))

    # 2. THE SANITY FILTER USED TO KEEP HALF A GAME.
    #
    # Written as a row filter it dropped the dog and published the favourite,
    # which is exactly the half-garbage it exists to catch: a real
    # FBS-versus-FCS line (total 62, home -55) gives 58.5 and 3.5, and the
    # board kept 58.5 plus an opponent_implied of 3.5 for a team with no row.
    silly = synth_odds([("Auburn", "Alabama")], {"Alabama": (62.0, -55.0)})
    try:
        CO.week_lines(silly, SCHOOLS)
        kept = True
    except CO.OddsUnavailable:
        kept = False
    ok("a fixture with one implausible side is dropped WHOLE", not kept,
       "the favourite survived alone" if kept else "")

    # And a normal game alongside it still publishes.
    mixed = (synth_odds([("Auburn", "Alabama")], {"Alabama": (62.0, -55.0)})
             + synth_odds([("Michigan", "Ohio State")],
                          {"Ohio State": (54.0, -7.0)}))
    lines = CO.week_lines(mixed, SCHOOLS + ["Michigan", "Ohio State"])
    ok("and one silly fixture does not take the believable ones down",
       set(lines["school"]) == {"Michigan", "Ohio State"},
       str(sorted(lines["school"])))

    # 3. THE SPREAD HAS TO BE MATCHED THE SAME WAY THE NAMES ARE.
    #
    # Keyed on exact string equality against the game's `home_team`, a book
    # that spells the side differently contributed its TOTAL and not its
    # SPREAD - so implied_total mixed a median over three books with a median
    # over one, which is not a quantity.
    games = synth_odds([("Auburn", "Alabama")], {"Alabama": (56.0, -6.0)})
    games[0]["bookmakers"].append({
        "key": "oddly-named",
        "markets": [
            {"key": "totals", "outcomes": [{"name": "Over", "point": 56.0}]},
            {"key": "spreads", "outcomes": [
                {"name": "ALABAMA CRIMSON TIDE", "point": -6.0},
                {"name": "auburn tigers", "point": 6.0}]},
        ]})
    by = CO.week_lines(games, SCHOOLS).set_index("school")
    ok("a book that spells the side differently still contributes its spread",
       abs(by.loc["Alabama", "implied_total"] - 31.0) < 1e-6,
       f"Alabama {by.loc['Alabama', 'implied_total']:.2f}, want 31.00")

    # 4. THE MARKET FACTOR MUST NOT BE A FUNCTION OF ROW COUNTS.
    #
    # MARKET_TOP_N was 8 and MARKET_MIN_PRICED 5, so a five-player team's
    # "offence" was a sum of five against a nine-player team's sum of eight.
    # With every player identical and every team implied identically - a board
    # where the only correct factor is 1.00 - a five-player team came out at
    # x1.350 and an eight-player team at x0.875.
    ok("the market denominator is the same length for every team",
       P.MARKET_MIN_PRICED >= P.MARKET_TOP_N,
       f"min {P.MARKET_MIN_PRICED}, top-n {P.MARKET_TOP_N}")
    rows, lines2 = [], []
    for team, k in (("ALA", 5), ("AUB", 8), ("OSU", 14)):
        for _ in range(k):
            rows.append({"team": team, "mean": 5.0, "median": 5.0})
        lines2.append({"team": team, "implied_total": 27.0})
    f = P.market_factor(pd.DataFrame(rows), pd.DataFrame(lines2))
    ok("identical players and identical lines produce no movement at all",
       bool(np.allclose(f.to_numpy(), 1.0, atol=1e-9)),
       f"factors {sorted(set(np.round(f.to_numpy(), 4)))}")

    # 5. TWO PRICED PLAYERS, ONE ATHLETE: BOTH LOSE THE PROJECTION.
    board_df = pd.DataFrame({
        "dk_player_id": ["1", "2", "3"],
        "name": ["Michael Smith", "Michael Smith", "Other Guy"],
        "position": ["WR", "WR", "RB"],
        "team": ["ALA", "AUB", "ALA"],
        "opponent": ["AUB", "ALA", "AUB"],
        "salary": [7000, 3000, 5000],
        "dk_points_per_game": [12.0, 1.0, 8.0],
        "athlete_id": ["A1", "A1", "A2"],
    })
    proj = pd.DataFrame({"player_id": ["A1", "A2"], "median": [12.0, 8.0],
                         "ceiling": [30.0, 20.0], "mean": [11.0, 7.0],
                         "games_seen": [12, 12], "last_season": [2026, 2026]})

    class _D:
        Unavailable = D.Unavailable

        @staticmethod
        def attach_history(b, h, team_map=None):
            return b.copy()

        @staticmethod
        def join_quality(j):
            return "join: (stubbed)"

    old_attach, old_quality = D.attach_history, D.join_quality
    D.attach_history, D.join_quality = _D.attach_history, _D.join_quality
    try:
        out = P.join_board(board_df, proj, pd.DataFrame(), {})
    finally:
        D.attach_history, D.join_quality = old_attach, old_quality
    smiths = out[out["name"] == "Michael Smith"]
    ok("two priced players who resolved to one athlete BOTH lose the "
       "projection", smiths["median"].isna().all(),
       str(smiths["median"].tolist()))
    ok("and the player who matched cleanly keeps his",
       float(out[out["name"] == "Other Guy"]["median"].iloc[0]) == 8.0)

    # 6. A NaN OPPONENT IS "?" AND NOT THE STRING "nan".
    ok("a missing opponent does not become a team called nan",
       P.game_label("ALA", float("nan")) == "? v ALA",
       P.game_label("ALA", float("nan")))
    ok("and two players in one game with a missing opponent share a label",
       P.game_label("ALA", None) == P.game_label("ALA", float("nan")))


def test_readiness_findings():
    """The two things a live readiness probe turned up.

    It confirmed the free CFBD tier DOES return current-season player stats -
    weeks 1-4 complete with 8,700 to 11,900 athletes each, "weeks with
    completed games but NO stats: 0". It also turned up two facts the build
    has to handle rather than assume.
    """
    head("WHAT THE READINESS PROBE FOUND")

    # 1. A POSITION MISSING FROM ONE SEASON'S ROSTER, PRESENT IN ANOTHER.
    #    The probe measured 97.6% of scorers matched to a position. The 2.4%
    #    were being dropped from the fit and therefore from the board.
    hist = pd.DataFrame({
        "athlete_id": ["A", "A", "A", "B", "B", "C", "C"],
        "position": ["WR", "WR", None, "", None, None, None],
        "season": [2024, 2025, 2026, 2025, 2026, 2025, 2026],
        "week": [1, 1, 1, 1, 1, 1, 1],
    })
    out = P.fill_positions(hist)
    got = dict(zip(out["athlete_id"] + out["season"].astype(str),
                   out["position"]))
    ok("a position missing in 2026 is filled from the same athlete's 2025",
       got["A2026"] == "WR", f"got {got['A2026']!r}")
    ok("an athlete with no position in ANY season still has none",
       pd.isna(got["C2026"]) or got["C2026"] in (None, ""),
       f"got {got['C2026']!r}")
    ok("and nothing already positioned is changed",
       got["A2024"] == "WR" and got["A2025"] == "WR")

    # The modal spelling, not the first one seen.
    mixed = pd.DataFrame({
        "athlete_id": ["D"] * 5,
        "position": ["ATH", "WR", "WR", "WR", None],
        "season": [2022, 2023, 2024, 2025, 2026],
        "week": [1] * 5,
    })
    ok("the filled position is his MOST COMMON one, not his earliest",
       P.fill_positions(mixed)["position"].iloc[-1] == "WR",
       str(P.fill_positions(mixed)["position"].iloc[-1]))

    # And a fullback is a running back on a DraftKings board, which is the one
    # skill position the probe listed that needs a mapping.
    ok("a fullback maps to the RB slot", S.roster_position("FB") == "RB")
    ok("a tight end maps to nothing, because no college board prices one",
       S.roster_position("TE") is None)

    # 2. THE NEWEST WEEK IS PARTLY INGESTED, AND THE PAGE SAYS SO.
    #    The probe found week 5 holding 9 of its 304 games - the midweek card
    #    only - while weeks 1-4 were complete.
    full = P.caveats_for(True, 4, SEASON, 0, 100, partial=False)
    half = P.caveats_for(True, 5, SEASON, 0, 100, partial=True)
    ok("a complete week is described as a complete week",
       any("through week 4" in c and "PARTLY" not in c for c in full))
    ok("a partly ingested week is described as one",
       any("PARTLY" in c for c in half),
       next((c[:60] for c in half if "PARTLY" in c), "(not said)"))

    # And the detection itself, against the real shape the probe reported:
    # weeks 1-4 with a full card and week 5 with just the midweek games.
    def season_frame(per_week_games):
        rows = []
        for wk, n in per_week_games.items():
            for g in range(n):
                rows.append({"week": wk, "game_id": f"{wk}-{g}",
                             "athlete_id": f"a{g}", "points": 5.0})
        return pd.DataFrame(rows)

    probe_shape = season_frame({1: 204, 2: 131, 3: 128, 4: 123, 5: 6})
    ok("the probe's own week counts are detected as a partial newest week",
       P.newest_week_is_partial(probe_shape, 5) is True)
    ok("and the same season without week 5 is not",
       P.newest_week_is_partial(
           season_frame({1: 204, 2: 131, 3: 128, 4: 123}), 4) is False)
    ok("a week merely a bit lighter than its neighbours is NOT called partial",
       P.newest_week_is_partial(
           season_frame({1: 204, 2: 131, 3: 128, 4: 100}), 4) is False,
       "100 of a 129 median is a bye-heavy week, not an unfinished one")
    ok("one cached week on its own cannot be judged either way",
       P.newest_week_is_partial(season_frame({1: 200}), 1) is False)
    ok("and an empty season does not explode",
       P.newest_week_is_partial(pd.DataFrame(), 0) is False)


def test_fixture_window():
    """The schedule window has to contain the week being published.

    `live_week` is the newest week with ANY completed game, and college
    football plays a Thursday game in essentially every week - so by Saturday
    morning, which is when the main slate publishes, `live_week` has already
    advanced to the current week. A window of week+1..week+3 then held NEXT
    week's pairings, no team code resolved, every board was skipped, and the
    run exited "no board could be built". Every Saturday.
    """
    head("THE FIXTURE WINDOW CONTAINS THE WEEK BEING BUILT")
    asked = []

    def fake_upcoming(key, season, after_week, span=2):
        asked.append(list(range(after_week + 1, after_week + 1 + span)))
        return synth_cfbd_games()

    src = open("cfb_publish.py").read()
    ok("the window starts BEFORE the live week",
       "max(week - 1, 0)" in src,
       "so week N is inside it whichever side of its Thursday game it is on")
    # And the arithmetic, stated as the assertion it is: with live_week = N,
    # the weeks fetched must include N.
    for live in (1, 5, 9, 14):
        weeks = list(range(max(live - 1, 0) + 1, max(live - 1, 0) + 1 + 4))
        ok(f"live_week {live} -> fetches weeks {weeks}", live in weeks)


def test_end_to_end(tmp):
    head("END TO END: main(), real optimiser, stubbed network")
    rng = np.random.default_rng(11)
    hist = synth_history(rng)
    board = synth_board(hist, rng)
    stub = Stub(board, odds=synth_odds())

    season_dir = Path(tmp) / "cache"
    season_dir.mkdir(parents=True, exist_ok=True)
    old_cache = C.CACHE
    C.CACHE = season_dir
    try:
        for s in SEASONS:
            C.save(hist[hist["season"] == s], s)
        rc = run_publish(stub, tmp, ["--sims", "1500", "--slates", "1",
                                     "--first-season", str(SEASONS[0]),
                                     "--season", str(SEASON)])
    finally:
        C.CACHE = old_cache

    ok("main() returned 0", rc == 0, f"rc={rc}")
    pay = payload_of(tmp)

    ok("the payload is the shape the page reads",
       all(k in pay for k in ("sport", "roster", "players", "games",
                              "quantiles", "loadings", "server_lineups",
                              "caveats")))
    ok("it is valid JSON with no NaN",
       "NaN" not in json.dumps(pay), "")
    ok("sport is cfb", pay["sport"] == "cfb")

    players = pay["players"]
    ok("every priced position is one this roster has",
       set(p["pos"] for p in players) <= {"QB", "RB", "WR"},
       str(sorted(set(p["pos"] for p in players))))
    ok("the tight ends and the kicker were dropped, not slotted",
       not any(p["pos"] in ("TE", "K") for p in players))
    ok("every player has a canonical game label",
       all(" v " in p["game"] for p in players))
    ok("one label per fixture, not two",
       len({p["game"] for p in players}) == len(FIXTURES),
       f"{sorted({p['game'] for p in players})}")
    ok("the games list is built from the BOARD and names every fixture",
       len(pay["games"]) == len(FIXTURES))
    ok("every game reports how many projected players are in it",
       all(g["players"] > 0 for g in pay["games"]))

    ok("ownership sums to the roster size",
       abs(sum(p["own"] for p in players) - len(pay["roster"]["slots"])) < 0.05,
       f"{sum(p['own'] for p in players):.3f} vs "
       f"{len(pay['roster']['slots'])}")
    qb_own = sum(p["own"] for p in players if p["pos"] == "QB")
    ok("quarterback ownership matches the stated superflex demand",
       abs(qb_own - S.QB_SUPERFLEX_DEMAND) < 0.05,
       f"{qb_own:.3f} vs {S.QB_SUPERFLEX_DEMAND}")

    ok("leverage is signed both ways",
       min(p["lev"] for p in players) < 0 < max(p["lev"] for p in players))
    ok("market lines reached the rows",
       sum(1 for p in players if p["itt"] is not None) > 0.8 * len(players),
       f"{sum(1 for p in players if p['itt'] is not None)} of {len(players)}")

    sl = pay["server_lineups"]
    ok("both integer programs solved", set(sl) == {"cash", "gpp"},
       f"got {sorted(sl)}")
    for kind, got in sl.items():
        ok(f"the {kind} lineup is enterable", got["illegal"] == [],
           "; ".join(got["illegal"]))
        ok(f"the {kind} lineup is eight players",
           len(got["players"]) == 8, str(len(got["players"])))
        ok(f"the {kind} lineup spends within the cap",
           got["salary"] <= 50_000, f"${got['salary']:,}")

    ok("the manifest lists the slate",
       len(json.loads((Path(tmp) / "data" / "manifest.json").read_text())
           ["slates"]) >= 1)

    # The name that differs by a middle initial. This is the join that no
    # exact key reduction can bridge, and the one `attach_history` has its
    # school-restricted pass for.
    middles = [p for p in players if " J. " in p["name"]]
    ok("the board's middle-initial spellings still joined",
       len(middles) == len([s for s in SCHOOLS if s in
                            {x for fx in FIXTURES for x in fx}]),
       f"{len(middles)} matched")
    return pay


def test_no_market_still_publishes(tmp):
    head("A BOARD WITH NO MARKET STILL PUBLISHES, AND SAYS SO")
    rng = np.random.default_rng(12)
    hist = synth_history(rng)
    stub = Stub(synth_board(hist, rng), odds=None)

    season_dir = Path(tmp) / "cache"
    season_dir.mkdir(parents=True, exist_ok=True)
    old_cache = C.CACHE
    C.CACHE = season_dir
    try:
        for s in SEASONS:
            C.save(hist[hist["season"] == s], s)
        rc = run_publish(stub, tmp, ["--sims", "800", "--slates", "1",
                                     "--first-season", str(SEASONS[0]),
                                     "--season", str(SEASON)])
    finally:
        C.CACHE = old_cache
    ok("main() returned 0 without any market", rc == 0, f"rc={rc}")
    pay = payload_of(tmp)
    ok("the page is told the market was missing",
       any("NO MARKET" in c for c in pay["caveats"]),
       pay["caveats"][0][:70])
    ok("the implied totals are null rather than zero",
       all(p["itt"] is None for p in pay["players"]))
    ok("and the lineups still solved", set(pay["server_lineups"]) ==
       {"cash", "gpp"})


def test_showdown_is_skipped(tmp):
    head("A ONE-GAME BOARD IS NOT PUBLISHED WITH THE CLASSIC ROSTER")
    rng = np.random.default_rng(13)
    hist = synth_history(rng)
    one = synth_board(hist, rng, games=[("Auburn", "Alabama")])
    stub = Stub(one, odds=synth_odds([("Auburn", "Alabama")]))

    season_dir = Path(tmp) / "cache"
    season_dir.mkdir(parents=True, exist_ok=True)
    old_cache = C.CACHE
    C.CACHE = season_dir
    try:
        for s in SEASONS:
            C.save(hist[hist["season"] == s], s)
        try:
            rc = run_publish(stub, tmp, ["--sims", "400", "--slates", "1",
                                         "--first-season", str(SEASONS[0]),
                                         "--season", str(SEASON)])
            died = False
        except SystemExit as exc:
            rc, died = exc.code, True
    finally:
        C.CACHE = old_cache
    ok("a single-game board is refused rather than published as a classic",
       died, f"rc={rc!r}")
    ok("and the reason says nothing published", died and "publish" in str(rc))


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="cfbpub-")
    try:
        test_roster_encoding()
        test_check_entry_can_fail()
        test_odds_sign()
        test_name_matching()
        test_market_factor_does_not_double_count()
        test_regressions()
        test_readiness_findings()
        test_fixture_window()
        test_end_to_end(Path(tmp) / "a")
        test_no_market_still_publishes(Path(tmp) / "b")
        test_showdown_is_skipped(Path(tmp) / "c")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("=" * 70)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
