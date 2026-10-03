"""College football, declared. Everything generic lives in engine/.

This is what a sport looks like once the engine exists: a spec, an adapter
from the sport's own vocabulary to the canonical frame, and whatever features
are genuinely peculiar to the sport. Roughly a hundred lines, most of them
explaining why.

What is peculiar to college football
------------------------------------
**No targets.** CFBD publishes receptions, not targets. In the NFL model
target share is the single most informative feature - it separates a receiver
who IS the offence from one who happens to play in a good one, and it does so
before the catches arrive. Receptions are the same signal after the fact,
contaminated by catch rate and by the quarterback. There is no substitute, so
this model is weaker at receiver than the NFL one, and that should be
expected rather than explained away later. Air yards and WOPR go with it.

**Blowouts.** College games are lopsided in a way professional ones are not,
and starters sit in the fourth quarter of a forty-point win. That is
frequent, learnable, and most of what has to replace the injury report
college football does not have. It is why `team_ewm_margin` and
`opp_ewm_margin` exist, and they are computed here rather than in the engine
because no other sport needs them in this form.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engine import SportSpec
from engine import features as EF
from engine.spec import FLEX_SLOTS

log = logging.getLogger(__name__)

# ------------------------------------------------------------- the contest
# DraftKings college football Classic: eight players, $50,000, and a superflex
# that is the whole character of the format.
#
#   QB  RB  RB  WR  WR  WR  FLEX  SFLEX
#
# FLEX takes a runner or a receiver. SFLEX takes a quarterback as well. There
# is NO tight end slot and tight ends are not flex-eligible, so DraftKings
# prices quarterbacks, running backs and receivers and nothing else - which is
# why `roster_position` below maps everything else to None rather than trying
# to find somewhere to put it.
#
# WHY THIS IS NOT MODELLED AS TWO DIFFERENT FLEX SLOTS
# ----------------------------------------------------
# It does not need to be, and the simpler encoding is exactly equivalent.
# Declare both flexes as taking QB/RB/WR and cap quarterbacks at two, and
# every composition that admits maps onto a legal assignment of the eight
# named slots:
#
#   QB 2, RB 2, WR 4   -> second QB at SFLEX, fourth WR at FLEX
#   QB 2, RB 3, WR 3   -> second QB at SFLEX, third RB at FLEX
#   QB 1, RB 3, WR 4   -> third RB at FLEX, fourth WR at SFLEX
#   QB 1, RB 2, WR 5   -> fifth WR fills both
#
# and the one composition it forbids - three quarterbacks - is the only
# illegal one. A per-slot eligibility model would cost the solver a dimension
# and buy nothing.
#
# `max_per_game` is how "players from at least two games" becomes something a
# solver can hold. On an eight-man roster, "at most seven from one game" says
# the same thing. A maximum per TEAM cannot: two teams playing each other are
# still one game, so four and four across a single fixture passes every
# per-team cap and is refused at the window.
#
# WHAT THE FIELD ACTUALLY PLAYS IN THE SUPERFLEX
# ----------------------------------------------
# Left to arithmetic, the ownership model splits the two flex slots by how
# much of the eligible pool each position represents, which hands quarterbacks
# about 1.33 per lineup. The field does not build that way: published
# winning-lineup composition puts a quarterback in the superflex around 86% of
# the time, because a second quarterback is the point of the format.
#
# 1.80 is therefore stated here rather than derived. Three things about it,
# plainly: winners are NOT the field, so this is an upper-ish estimate of what
# the field does; the honest range is roughly 1.6 to 1.9; and 1.33 is the one
# number that is certainly wrong, because it assumes a field that fills the
# superflex by drawing at random from everyone eligible. It is a single
# constant so it can be moved in one place when a real field distribution is
# available, and the publisher logs it on every run so it cannot drift out of
# sight.
QB_SUPERFLEX_DEMAND = 1.80

DK_CLASSIC = {
    "slots": ["QB", "RB", "RB", "WR", "WR", "WR", "FLEX", "SFLEX"],
    "flex_positions": ["QB", "RB", "WR"],
    "salary_cap": 50_000,
    "max_position": {"QB": 2},
    "max_per_game": 7,          # == "at least two games", on eight slots
    "min_games": 2,             # the same rule, as the minimum it really is
    "position_demand": {"QB": QB_SUPERFLEX_DEMAND},
    # Unspent salary is points declined. The marginal rate in football is
    # about 2.0 projected points per $1,000, so $1,500 left over is three
    # points the lineup chose not to buy.
    "min_salary_pct": 0.99,
}

# What DraftKings calls a position, mapped to a slot this roster has.
POSITION_MAP = {"QB": "QB", "RB": "RB", "WR": "WR", "FB": "RB",
                "HB": "RB", "TB": "RB", "SE": "WR", "FL": "WR",
                "ATH": None, "TE": None, "K": None, "DST": None,
                "DEF": None, "P": None}


def roster_position(raw) -> str | None:
    """One of QB, RB, WR - or None for something this roster cannot use.

    None rather than a best guess. A tight end has no slot on a DraftKings
    college board at all, and inventing one for him puts a player in a lineup
    the site will refuse.
    """
    s = str(raw or "").upper().strip()
    if not s:
        return None
    first = s.replace("\\", "/").split("/")[0].strip()
    return POSITION_MAP.get(first)


def check_entry(lineup: pd.DataFrame, roster: dict) -> list[str]:
    """Every reason DraftKings would reject this lineup. Empty means legal.

    Returned as a LIST rather than raised on the first problem, because a
    lineup can be wrong in several ways at once and finding out one reason per
    round is how a build turns into an afternoon.

    This exists because the solver's constraints are not the site's rules. It
    knows the cap, the slots, a per-position ceiling and a per-game ceiling; it
    does not know that "at least two games" is a MINIMUM, and a minimum cannot
    be written as a maximum. The maximum encoding happens to be exact on an
    eight-man roster - but it is exact by arithmetic coincidence, and the thing
    that must never be wrong deserves to be checked directly.
    """
    problems = []

    spent = pd.to_numeric(lineup.get("charged", lineup.get("salary")),
                          errors="coerce").sum()
    cap = float(roster["salary_cap"])
    if spent > cap:
        problems.append(f"over the cap: ${spent:,.0f} of ${cap:,.0f}")

    want = len(roster["slots"])
    if len(lineup) != want:
        problems.append(f"{len(lineup)} players, not {want}")

    pos = lineup["position"].astype(str)
    counts = pos.value_counts().to_dict()
    for p, ceiling in (roster.get("max_position") or {}).items():
        if counts.get(p, 0) > int(ceiling):
            problems.append(
                f"{counts.get(p, 0)} at {p}, more than the {int(ceiling)} "
                f"DraftKings allows. This lineup CANNOT BE ENTERED.")

    # THE SAME MAN TWICE IS NOT A LINEUP, and nothing else here was checking
    # it. DraftKings prices some players at two positions, and a board that
    # lists one athlete under both gives the solver two rows it treats as two
    # people - an entry the site refuses outright. Checked on the DraftKings
    # player id where the board carries one, and on name-and-team otherwise,
    # because a name alone collides across schools in college football.
    if "dk_player_id" in lineup.columns and lineup["dk_player_id"].notna().all():
        who = lineup["dk_player_id"].astype(str)
    else:
        who = (lineup["name"].astype(str) + "|"
               + lineup["team"].astype(str))
    dup = who[who.duplicated()]
    if len(dup):
        names = lineup.loc[dup.index, "name"].astype(str).tolist()
        problems.append(f"{', '.join(sorted(set(names)))} appear(s) more than "
                        f"once. This lineup CANNOT BE ENTERED.")

    # The required counts, checked as the minimums they are. A lineup one
    # receiver short is not a lineup, however well it scores in a simulation.
    #
    # FLEX_SLOTS, not a tuple written out here. The version this replaced
    # listed FLEX, SFLEX, UTIL and CPT and omitted UTIL/FLEX and G/UTIL, which
    # is the FOURTH hand-written copy of this list in the project and the
    # third one to be wrong - the whole reason the set was centralised in
    # engine/spec.py. Run against a hockey roster it reported "0 at UTIL/FLEX,
    # fewer than the 1 required" on a perfectly legal lineup.
    required: dict[str, int] = {}
    for s in roster["slots"]:
        if s in FLEX_SLOTS or s == "CPT":
            continue
        required[s] = required.get(s, 0) + 1
    for p, lo in required.items():
        if counts.get(p, 0) < lo:
            problems.append(f"{counts.get(p, 0)} at {p}, fewer than the "
                            f"{lo} required")

    eligible = set(required) | set(roster.get("flex_positions") or [])
    stray = sorted(set(counts) - eligible)
    if stray:
        problems.append(f"{', '.join(stray)} has no slot on this roster")

    need_games = int(roster.get("min_games", 0) or 0)
    if need_games > 1:
        if "game" not in lineup.columns:
            problems.append(
                f"this roster needs players from {need_games} games and the "
                f"lineup carries no `game` column, so the rule could not be "
                f"checked at all")
        else:
            seen = lineup["game"].astype(str).nunique()
            if seen < need_games:
                problems.append(
                    f"every player comes from {seen} game(s); DraftKings "
                    f"requires at least {need_games}. This lineup CANNOT BE "
                    f"ENTERED.")

    cap_team = roster.get("max_per_team")
    if cap_team:
        worst = lineup["team"].astype(str).value_counts()
        if len(worst) and int(worst.iloc[0]) > int(cap_team):
            problems.append(
                f"{int(worst.iloc[0])} players from {worst.index[0]}, more "
                f"than the {cap_team} allowed")

    return problems


# ------------------------------------------------------------------- the spec
SPEC = SportSpec(
    name="CFB",
    usage=["carries", "rec", "rush_yards", "rec_yards", "pass_yards",
           "completions"],
    shares=["carries", "rec", "rec_yards"],
    touch_columns=["carries", "rec"],
    positions=["QB", "RB", "WR", "TE"],
    min_prior_games=3,
    halflife=4.0,
    extra_features=["team_ewm_margin", "opp_ewm_margin",
                    "opp_ewm_pass_yards_allowed",
                    "opp_ewm_rush_yards_allowed",
                    "team_ewm_pass_yards", "team_ewm_rush_yards"],
    # WITHOUT THESE THE SIMULATION IS CLOSE TO INDEPENDENT, AND A STACK IS
    # WORTH NOTHING TO IT.
    #
    # That is the honest default for a sport nobody has studied, and it was
    # the right default while there was no CFB board to publish. It is the
    # wrong default now: a quarterback-and-receiver stack is the single bet
    # college football DFS is built around, and a tournament objective that
    # cannot see the joint tail will never prefer one.
    #
    # The numbers are the engine's NFL reference, used deliberately rather
    # than invented fresh. It is the same sport with the same causal
    # structure - one passer, several receivers splitting his throws, a game
    # shock both sides share - and making up college-specific loadings with
    # no study behind them would be a guess that looks like knowledge.
    #
    # ONE DEVIATION, and it has a reason rather than a number behind it: the
    # TEAM loading is raised for all three positions. College games are
    # lopsided in a way professional ones are not, and the fourth quarter of
    # a forty-point win removes a team's whole first-string offence AT ONCE.
    # That is a shared team shock, which is exactly what this factor is, and
    # it is the same effect `team_ewm_margin` exists to let the projection
    # see. Its size here is a judgement, so it is written where it can be
    # argued with instead of buried.
    #
    # There is no TE row and no slot for one: DraftKings does not price tight
    # ends on a college board. There is no DST row either, for the same
    # reason - which spares this sport the one sign in the NFL table that
    # really matters, the negative game loading that stops a model stacking a
    # quarterback with the defence trying to stop him.
    loadings={
        "game":    {"QB": 0.39, "WR": 0.39, "RB": 0.20},
        "team":    {"QB": 0.50, "WR": 0.50, "RB": 0.44},
        # The only negative term, and bounded: a player cannot spend more
        # than all of his variance on competing with his team-mates. Two
        # quarterbacks sit near the bound on purpose - only one of them plays
        # a meaningful number of snaps, so their outcomes really are close to
        # mutually exclusive, and a model that left them positively
        # correlated would happily roster both.
        "compete": {"QB": 0.98, "WR": 0.86, "RB": 0.92},
    },
    rosters={("dk", "classic"): DK_CLASSIC},
)


def to_canonical(hist: pd.DataFrame) -> pd.DataFrame:
    """CFBD's vocabulary, renamed to the engine's.

    Kept as an explicit step rather than renaming inside the data layer, so
    anything reading cfb_data still sees CFBD's own names and only the
    modelling side sees the engine's.
    """
    out = hist.rename(columns={"athlete_id": "player_id",
                               "school": "team",
                               "week": "period"})
    # Checked BEFORE touching any column. Casting first turned "this frame is
    # missing player_id" into a bare KeyError from the cast, which says
    # nothing about what the caller got wrong.
    missing = [c for c in ("player_id", "season", "period", "team", "points")
               if c not in out.columns]
    if missing:
        raise ValueError(f"CFB history is missing {missing}")
    # Text, always. Read back as an integer, "0041" becomes 41 and matches
    # nothing - silently, and the join simply gets worse.
    out["player_id"] = out["player_id"].astype(str)
    return out


def margins(df: pd.DataFrame) -> pd.DataFrame:
    """Each team-period's scoring margin, from the player rows themselves.

    Fantasy points are not the scoreboard, but a team's production against
    its opponent's is a serviceable proxy for how lopsided the game was, and
    it needs no extra endpoint.

    Joined through the REAL fixtures. A first version merged team totals
    against themselves on (season, period) and filtered out self-pairings,
    which is a cross join: with ten teams it produced nine rows per team-game
    instead of one, and the merge downstream fanned the feature frame out
    forty-fold. Nothing raised; the training set simply became duplicates.
    """
    tp = (df.groupby(["team", "season", "period"], as_index=False)["points"]
          .sum().rename(columns={"points": "_tp"}))
    fixtures = df[["team", "opponent", "season", "period"]].drop_duplicates()
    m = fixtures.merge(tp, on=["team", "season", "period"], how="left")
    m = m.merge(tp.rename(columns={"team": "opponent", "_tp": "_op"}),
                on=["opponent", "season", "period"], how="left")
    m["margin"] = m["_tp"] - m["_op"]
    m = m[["team", "season", "period", "margin"]]

    dupes = int(m.duplicated(["team", "season", "period"]).sum())
    if dupes:
        log.warning("%d team-periods have more than one opponent; keeping "
                    "the first. Usually duplicated history or a team map "
                    "that merged two schools", dupes)
        m = m.drop_duplicates(["team", "season", "period"])
    return m


def build(hist: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
    """Canonical features, plus the college-specific ones."""
    df = to_canonical(hist)
    n_in = len(df)
    out = EF.build(df, SPEC, validate=validate)

    # Team and opponent yardage conceded, in the two flavours football cares
    # about. The engine only carries points allowed, because that is the one
    # every sport has.
    yards = (df.groupby(["team", "season", "period"], as_index=False)
             .agg(pass_yards=("pass_yards", "sum"),
                  rush_yards=("rush_yards", "sum"))
             .sort_values(["team", "season", "period"]))
    g = yards.groupby("team", sort=False)
    yards["team_ewm_pass_yards"] = g["pass_yards"].transform(
        lambda s: EF.ewm(s, SPEC.halflife))
    yards["team_ewm_rush_yards"] = g["rush_yards"].transform(
        lambda s: EF.ewm(s, SPEC.halflife))
    out = out.merge(yards[["team", "season", "period", "team_ewm_pass_yards",
                           "team_ewm_rush_yards"]],
                    on=["team", "season", "period"], how="left")
    out = out.merge(
        yards[["team", "season", "period", "team_ewm_pass_yards",
               "team_ewm_rush_yards"]]
        .rename(columns={"team": "opponent",
                         "team_ewm_pass_yards": "opp_ewm_pass_yards_allowed",
                         "team_ewm_rush_yards": "opp_ewm_rush_yards_allowed"}),
        on=["opponent", "season", "period"], how="left")

    m = margins(df)
    mg = m.groupby("team", sort=False)
    m["team_ewm_margin"] = mg["margin"].transform(
        lambda s: EF.ewm(s, SPEC.halflife))
    out = out.merge(m[["team", "season", "period", "team_ewm_margin"]],
                    on=["team", "season", "period"], how="left")
    out = out.merge(
        m[["team", "season", "period", "team_ewm_margin"]]
        .rename(columns={"team": "opponent",
                         "team_ewm_margin": "opp_ewm_margin"}),
        on=["opponent", "season", "period"], how="left")

    for c in SPEC.extra_features:
        if c not in out.columns:
            out[c] = np.nan

    if len(out) != n_in:
        raise ValueError(
            f"CFB build changed the row count: {n_in} in, {len(out)} out. A "
            f"merge fanned out on duplicate keys.")
    missing = [c for c in SPEC.features if c not in out.columns]
    if missing:
        raise ValueError(f"CFB build did not produce {missing}")
    return out
