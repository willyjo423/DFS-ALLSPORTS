"""Hockey, declared. Two specs, because it is two sports.

Skaters and goalies share a slate and nothing else. A goalie's scoring line is
built from saves and goals allowed; a skater's from goals, assists and shots.
More importantly their correlation runs in OPPOSITE directions: a goalie's
good night is, by definition, the other side's skaters having a bad one. One
spec with a position flag would ask a single fitted model to learn that a goal
is worth +8.5 to the man who scores it and -3.5 to the man it goes past, from
the same column. So: two specs, two fitted models.

What hockey needs that baseball did not
---------------------------------------
**Much stronger team correlation.** One goal pays up to THREE players on the
same line at once - the scorer 8.5 and each assister 5. Nothing in baseball
does that; a home run pays exactly one batter. A line is not a queue of
independent chances like a batting order, it is a single event generator whose
output is split three ways. So the team loading here is the highest in the
project, and it is the single most important thing about building an NHL
lineup.

**Ice time is the opportunity metric, and it is deployment, not talent.** In
baseball a hitter's plate appearances are set by his slot and the slot moves
slowly. In hockey a third-liner promoted to the top power-play unit gains more
in one night than a hitter gains moving from ninth to leadoff, and it can be
reversed the next game on a coach's whim. That makes ice time both the most
predictive feature and the most volatile, which is why the memory here is
shorter than baseball's.

**The forward-looking feature is the line and the power-play unit.** The exact
analogue of baseball's batting order: published at the morning skate and
confirmed an hour or so before puck drop, so it is genuinely known before the
game it describes. PP1 versus PP2 is the largest single difference between two
otherwise identical forwards, and a model that cannot see it is guessing at the
biggest number on the board.

**The starting goalie is the whole roster slot.** A backup who does not dress
scores zero, exactly like a benched hitter, and goalies are confirmed later and
less reliably than probable pitchers. This is the availability problem in its
most expensive form: one slot, one player, no partial credit.

A rule this file CANNOT express, stated plainly
-----------------------------------------------
DraftKings requires skaters from at least THREE different teams. That is a
minimum-teams rule, and the optimiser only understands a maximum-per-team one,
so it cannot be enforced as a hard constraint the way the cap and the slots
are. `max_per_team` below is set to the tightest value that makes the rule
impossible to break by accident, and the publisher checks the finished lineup
and refuses it otherwise. A lineup that violates it cannot be entered at all,
which makes it exactly as fatal as going over the cap and far easier to miss.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engine import SportSpec
from engine import features as EF

log = logging.getLogger(__name__)

# One game per team per day. Hockey has no doubleheaders, so a day identifies
# a game and the packing baseball needed is not required here. Kept as a named
# constant anyway, because `period` arithmetic that assumes 1 without saying so
# is how the baseball build discovered doubleheaders the hard way.
SLOTS = 1

# 82 games, and deployment moves faster than talent does. Baseball uses thirty
# games over a 162-game season because a hitter's true line moves slowly. A
# skater's does too - but his ICE TIME does not, and ice time is most of what
# this model predicts. A line shuffle or a power-play promotion changes his
# output immediately and permanently, so the memory has to be short enough to
# notice. Twelve games is roughly a month of hockey.
HALFLIFE = 12.0

# Goalies play perhaps 50 of 82 games and their workload is set by the team in
# front of them rather than by anything they control, so there is both less to
# remember and less that is theirs.
GOALIE_HALFLIFE = 8.0

# Whether the skater model is told tonight's line and power-play unit.
#
# ON, because unlike the opposing-starter experiment in baseball this is not a
# guess about somebody else - it is the player's own deployment, announced
# before the game, and it is the largest single driver of his night. The MLB
# equivalent (`bat_slot`) is the most valuable feature in that model.
#
# If the line data cannot be joined the columns arrive as NaN and the model
# degrades to one that cannot see deployment, which is where it started. It
# does not fail.
LINE_FEATURES = ["line_number", "pp_unit", "ewm_line_number", "ewm_pp_unit"]

# --------------------------------------------------------------- DraftKings
# Scoring, as DraftKings pays it. Written out rather than folded into the
# projection because a scoring change is a one-line edit here and an
# archaeological dig anywhere else.
#
# Note what is NOT here: DraftKings pays no power-play bonus (FanDuel does),
# and no points for a shootout goal by a goalie. Both are easy to assume by
# analogy and both would be wrong.
DK_SKATER_SCORING = {
    "goal": 8.5,
    "assist": 5.0,
    "shot_on_goal": 1.5,
    "blocked_shot": 1.3,
    "short_handed_point": 2.0,      # bonus, on top of the goal or assist
    "shootout_goal": 1.5,
}
# Thresholds pay a flat bonus once crossed. They are why a skater's
# distribution has a fatter right tail than his per-event scoring suggests:
# the night he takes six shots pays three extra points for nothing new.
DK_SKATER_BONUSES = {
    "shots_on_goal": (5, 3.0),
    "blocked_shots": (3, 3.0),
    "points": (3, 3.0),             # goals + assists
    "goals": (3, 3.0),              # the hat trick, stacked on the points one
}
DK_GOALIE_SCORING = {
    "win": 6.0,
    "save": 0.7,
    "goal_against": -3.5,
    "shutout": 4.0,
    "overtime_loss": 2.0,
}
DK_GOALIE_BONUSES = {"saves": (35, 3.0)}

# The roster, and the rule it cannot hold.
#
# UTIL takes any skater, which is why every skater position is a flex
# position. The goalie slot takes only G, so it is not.
DK_CLASSIC = {
    "slots": ["C", "C", "W", "W", "W", "D", "D", "UTIL", "G"],
    "flex_positions": ["C", "W", "D"],
    "salary_cap": 50_000,
    # DraftKings' real rule is "skaters from at least three teams", which is a
    # MINIMUM on distinct teams and cannot be written as a maximum. Six is the
    # tightest maximum that leaves the rule satisfiable and forbids the
    # obvious violation (seven or eight skaters off one bench). It does NOT
    # guarantee three teams on its own - eight skaters split 6/2 is legal by
    # this number and illegal at DraftKings - so `check_entry` below is the
    # thing that actually enforces it, and the publisher must call it.
    "max_per_team": 6,
    "min_skater_teams": 3,
    "min_salary_pct": 0.99,         # at most $500 unspent; see the MLB notes
}
DK_SHOWDOWN = {
    "slots": ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"],
    "flex_positions": ["C", "W", "D", "G"],
    "salary_cap": 50_000,
    "captain_multiplier": 1.5,
    "max_per_team": 5,
    "min_skater_teams": 1,          # one game, so the rule cannot apply
    "min_salary_pct": 0.98,
}

# The vocabulary everything downstream uses: three skater slots and a goalie.
#
# MoneyPuck records L and R; DraftKings prices a single W; the league writes
# LW and RW. All of them mean the same roster slot, and leaving them apart
# split every winger across three position dummies that were each nearly
# empty - the model reported "dropping 5 empty or constant features:
# line_number, pp_unit, is_W, is_LW, is_RW", which is to say it could not see
# that a winger was a winger at all.
SKATER_POSITIONS = ["C", "W", "D"]
GOALIE_POSITIONS = ["G"]

# Every spelling either source uses, collapsed to that vocabulary. Applied to
# the HISTORY here and to the board in the publisher, so the two agree.
POSITION_MAP = {
    "C": "C", "CENTER": "C", "CENTRE": "C",
    "W": "W", "LW": "W", "RW": "W", "L": "W", "R": "W",
    "LEFT WING": "W", "RIGHT WING": "W", "F": "W", "FORWARD": "W",
    "D": "D", "DEFENSE": "D", "DEFENCE": "D", "LD": "D", "RD": "D",
    "G": "G", "GOALIE": "G", "GOALTENDER": "G",
}


def roster_position(raw) -> str | None:
    """One of C, W, D, G - or None for something this roster cannot use.

    A multi-position player arrives as "C/W". The first is the one the source
    lists as primary, and pinning him to it means nothing downstream uses an
    eligibility it might have read wrong.
    """
    s = str(raw or "").upper().strip()
    if not s:
        return None
    first = s.replace("\\", "/").split("/")[0].strip()
    return POSITION_MAP.get(first)


def check_entry(lineup: pd.DataFrame, roster: dict) -> list[str]:
    """Every reason DraftKings would reject this lineup. Empty means legal.

    Returned rather than raised, and returned as a LIST, because a lineup can
    be wrong in more than one way and finding out one reason per round is how
    a build loop turns into an afternoon.

    This exists because the optimiser's constraints are not the site's rules.
    It knows the cap, the slots and a maximum per team; it does not know that
    hockey needs skaters from three different teams. A lineup that breaks that
    looks perfectly legal all the way to the moment the entry is refused.
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
    skaters = lineup[pos.isin(SKATER_POSITIONS)]
    need_teams = int(roster.get("min_skater_teams", 0) or 0)
    if need_teams > 1:
        teams = skaters["team"].astype(str).nunique()
        if teams < need_teams:
            problems.append(
                f"skaters come from only {teams} team(s); DraftKings requires "
                f"at least {need_teams}. This lineup CANNOT BE ENTERED.")

    cap_team = roster.get("max_per_team")
    if cap_team:
        worst = lineup["team"].astype(str).value_counts()
        if len(worst) and int(worst.iloc[0]) > int(cap_team):
            problems.append(
                f"{int(worst.iloc[0])} players from {worst.index[0]}, more "
                f"than the {cap_team} allowed")

    return problems


# ------------------------------------------------------------------- specs
SKATERS = SportSpec(
    name="NHL-skaters",
    usage=["time_on_ice", "pp_time_on_ice", "shots_on_goal", "blocked_shots",
           "goals", "assists", "shifts", "faceoffs_taken"],
    # Ice time as a SHARE of the team's, which is what deployment actually
    # is. A skater whose raw minutes fell because the game went to overtime
    # has not been demoted; one whose share fell has.
    shares=["time_on_ice", "pp_time_on_ice"],
    touch_columns=["time_on_ice"],
    positions=SKATER_POSITIONS,
    min_prior_games=8,
    halflife=HALFLIFE,
    min_train_rows=2000,
    min_played_rows=1500,
    # `line_number` and `pp_unit` are deliberately NOT shifted, and that is
    # not a leak. Every other feature describes games already played, because
    # the thing being predicted has not happened. Lines are announced at the
    # morning skate, so tonight's deployment is genuinely known before
    # tonight's game - the same class of feature as a market line, or as
    # baseball's batting order.
    #
    # The `ewm_` versions come along so the model can see the DIFFERENCE
    # between where a man usually plays and where he is playing tonight,
    # which is precisely what a promotion carries.
    extra_features=LINE_FEATURES,
    loadings={
        # The strongest team term in this project, and it is not a preference.
        # One goal pays the scorer and up to two assisters at once, so three
        # linemates share a single event rather than three independent ones.
        # Baseball's 0.42 reflects hitters queueing for chances off the same
        # pitcher; hockey's reflects them being paid for the SAME chance.
        "game":    {p: 0.20 for p in SKATER_POSITIONS},
        "team":    {p: 0.55 for p in SKATER_POSITIONS},
        # Defencemen compete for the same power-play minutes far more than
        # forwards compete for even-strength ones, because there are only two
        # D slots on a unit and six of them dressed.
        "compete": {"C": 0.08, "W": 0.08, "D": 0.22},
    },
)

GOALIES = SportSpec(
    name="NHL-goalies",
    # No `wins` and no `decision`, because MoneyPuck publishes neither. That
    # absence is not cosmetic: a win pays 6 points, roughly a quarter of a
    # typical goalie's score, and `score()` has to INFER it from ice time and
    # goals allowed. Listing a column the feed does not have would fail the
    # frame contract; pretending it exists downstream would be worse.
    usage=["saves", "shots_against", "goals_against", "time_on_ice"],
    shares=[],
    touch_columns=["shots_against"],
    positions=GOALIE_POSITIONS,
    min_prior_games=5,
    halflife=GOALIE_HALFLIFE,
    min_train_rows=800,
    min_played_rows=600,
    extra_features=[],
    loadings={
        # A goalie is the game, from the other side. His save total is driven
        # by how much the opposition shoots and his goals-against by how well
        # they finish, so the GAME term dominates and the team term - his own
        # side's scoring - barely moves him at all. DraftKings pays him 6 for
        # a win, which is the one place his own offence matters.
        #
        # `compete` is high for the same reason baseball's starters compete:
        # two goalies on one team in one game is not a thing that happens, and
        # where it does (a pull) the minutes one takes are minutes the other
        # does not.
        "game":    {"G": 0.45},
        "team":    {"G": 0.15},
        "compete": {"G": 0.60},
    },
)

SPECS = {"skaters": SKATERS, "goalies": GOALIES}


# -------------------------------------------------------------- conversion
def score(hist: pd.DataFrame) -> pd.Series:
    """What DraftKings would have paid for each of these games.

    This is the TARGET - the number the model is fitted to predict - and it
    has to be computed from the box score rather than read from anywhere,
    because MoneyPuck publishes hockey and not fantasy scoring.

    The threshold bonuses are the interesting part and the reason a skater's
    distribution has a fatter right tail than his per-event scoring implies.
    Five shots pays 7.5 and six pays 12.0 - the sixth shot is worth 4.5 on its
    own. A model fitted on per-event scoring alone would systematically
    understate exactly the nights that win tournaments.

    A row with no ice time scores zero rather than NaN: a scratch genuinely
    scored nothing, and the availability half of the model needs to see the
    difference between "did not play" and "played and was quiet".
    """
    def col(name, default=0.0):
        if name not in hist.columns:
            return pd.Series(default, index=hist.index, dtype=float)
        return pd.to_numeric(hist[name], errors="coerce").fillna(default)

    is_goalie = hist.get("position", pd.Series("", index=hist.index)) \
        .astype(str).str.upper().str.strip().eq("G")

    # ---- skaters
    goals = col("goals")
    assists = col("assists")
    shots = col("shots_on_goal")
    blocks = col("blocked_shots")
    points = goals + assists

    sk = (goals * DK_SKATER_SCORING["goal"]
          + assists * DK_SKATER_SCORING["assist"]
          + shots * DK_SKATER_SCORING["shot_on_goal"]
          + blocks * DK_SKATER_SCORING["blocked_shot"])
    for stat, series in (("shots_on_goal", shots), ("blocked_shots", blocks),
                         ("points", points), ("goals", goals)):
        need, bonus = DK_SKATER_BONUSES[stat]
        sk = sk + (series >= need) * bonus

    # ---- goalies
    saves = col("saves")
    against = col("goals_against")
    shots_faced = col("shots_against")

    # A win is not in this feed, so it is INFERRED and the inference is stated
    # rather than hidden: a goalie who played most of a game and allowed fewer
    # than the league's average is treated as likely to have won. That is a
    # proxy, it is wrong on a real share of games, and it is better than
    # dropping the six points a win pays - which is a quarter of a typical
    # goalie's score. Replace it the moment a decision column is available.
    toi = col("time_on_ice")
    likely_win = (toi >= 40) & (against <= 2)
    shutout = (toi >= 55) & (against <= 0) & (shots_faced > 0)

    go = (saves * DK_GOALIE_SCORING["save"]
          + against * DK_GOALIE_SCORING["goal_against"]
          + likely_win * DK_GOALIE_SCORING["win"]
          + shutout * DK_GOALIE_SCORING["shutout"])
    need, bonus = DK_GOALIE_BONUSES["saves"]
    go = go + (saves >= need) * bonus

    out = sk.where(~is_goalie, go)
    # Nobody on the ice scores nothing by accident; a scratch scores zero.
    return out.where(col("time_on_ice") > 0, 0.0).astype(float)


def to_canonical(hist: pd.DataFrame) -> pd.DataFrame:
    """Box-score rows, renamed to the engine's vocabulary.

    `period` is a day index. Hockey plays one game per team per day, so unlike
    baseball a day identifies a game and no packing is needed. The arithmetic
    is still routed through `day_of` and `periods_of` so that if this ever
    stops being true - an outdoor doubleheader, a rescheduled pair - there is
    one place to change rather than twelve.
    """
    out = hist.copy()

    if "date" in out.columns:
        day = pd.to_datetime(out["date"], errors="coerce")
        base = day.min()
        out["period"] = ((day - base).dt.days * SLOTS).astype("Int64")
    elif "period" not in out.columns:
        raise ValueError(
            "history carries neither `date` nor `period`, so games cannot be "
            "put in order. Columns were: " + ", ".join(sorted(out.columns)[:20]))

    # Time on ice arrives from the league as "18:42". Minutes as a float is
    # what every downstream mean, share and ewm needs, and a string that
    # silently becomes NaN would make ice time - the most important column
    # here - quietly empty.
    for col in ("time_on_ice", "pp_time_on_ice", "sh_time_on_ice"):
        if col in out.columns:
            out[col] = _minutes(out[col])

    # Positions into the one vocabulary, BEFORE anything is fitted on them.
    #
    # MoneyPuck writes L and R. The spec knows C, W, D and G. Left alone, the
    # model drops `is_W` as constant - it is looking for a value the history
    # never contains - and no winger is ever identified as one. It fits
    # cleanly and quietly loses a whole position.
    if "position" in out.columns:
        before = out["position"].astype(str).str.upper().str.strip()
        out["position"] = before.map(lambda p: roster_position(p) or "")
        lost = int((out["position"] == "").sum())
        if lost:
            log.warning("%d rows carry a position this roster has no slot "
                        "for and will not be fitted: %s", lost,
                        ", ".join(sorted(before[out["position"] == ""]
                                         .unique())[:8]))
        counts = out["position"].value_counts().to_dict()
        log.info("positions after mapping: %s", counts)

    if "played" not in out.columns:
        # A dressed skater who took no shift is not the same as one who was
        # scratched, and only ice time can tell them apart.
        toi = pd.to_numeric(out.get("time_on_ice"), errors="coerce")
        out["played"] = (toi.fillna(0) > 0).astype(int)

    # The target. Computed here rather than at fetch time so a scoring change
    # is a one-line edit and a re-publish, not a twelve-minute re-fetch of
    # every career MoneyPuck holds.
    if "points" not in out.columns:
        out["points"] = score(out)
        played = out[out["played"] == 1]["points"]
        if len(played):
            log.info("DK points: %d games scored, mean %.2f, median %.2f, "
                     "99th %.1f, max %.1f", len(played), played.mean(),
                     played.median(), played.quantile(0.99), played.max())
        if (out["points"] == 0).all():
            log.error("EVERY game scored zero. The stat columns are missing "
                      "or empty, and a model fitted on this would learn that "
                      "nothing ever happens.")

    return out


def _minutes(s: pd.Series) -> pd.Series:
    """"18:42" -> 18.7. Numbers pass through; anything else becomes NaN."""
    num = pd.to_numeric(s, errors="coerce")
    text = s.astype(str).str.strip()
    mmss = text.str.match(r"^\d{1,3}:\d{2}$", na=False)
    if mmss.any():
        parts = text[mmss].str.split(":", expand=True).astype(float)
        num.loc[mmss] = parts[0] + parts[1] / 60.0
    return num


def day_of(period: int) -> int:
    return int(period) // SLOTS


def periods_of(day: int) -> tuple[int, ...]:
    return tuple(int(day) * SLOTS + i for i in range(SLOTS))


def split(hist: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Skaters and goalies, by position rather than by guessing at columns."""
    pos = hist.get("position")
    if pos is None:
        raise ValueError("history has no `position` column, so skaters and "
                         "goalies cannot be told apart")
    is_g = pos.astype(str).str.upper().str.strip().isin(GOALIE_POSITIONS)
    return hist[~is_g].copy(), hist[is_g].copy()


def _with_lines(built: pd.DataFrame, lines: pd.DataFrame | None
                ) -> pd.DataFrame:
    """Tonight's line and power-play unit, joined on.

    Non-fatal by design. If the join finds nothing the columns arrive as NaN
    and the model degrades to one that cannot see deployment - which is where
    every hockey model without this data already is. A board without lines is
    worth far more than no board.
    """
    out = built.copy()
    for c in ("line_number", "pp_unit"):
        if c not in out.columns:
            out[c] = np.nan
    if lines is None or not len(lines):
        log.warning("no line data - `line_number` and `pp_unit` will be empty, "
                    "so the model cannot see tonight's deployment")
        return out

    try:
        want = lines[["player_id", "line_number", "pp_unit"]].copy()
        want["player_id"] = want["player_id"].astype(str)
        before = len(out)
        out["player_id"] = out["player_id"].astype(str)
        out = out.drop(columns=["line_number", "pp_unit"]).merge(
            want.drop_duplicates("player_id"), on="player_id", how="left")
        if len(out) != before:
            raise ValueError(f"the line join changed the row count from "
                             f"{before} to {len(out)} - it fanned out")
        hit = int(out["line_number"].notna().sum())
        log.info("lines joined to %d of %d skaters (%.0f%%)",
                 hit, len(out), 100 * hit / max(len(out), 1))
    except Exception as exc:                                   # noqa: BLE001
        log.error("line data could not be joined (%s: %s); the board is "
                  "unaffected but deployment is invisible to the model",
                  type(exc).__name__, str(exc)[:100])
        for c in ("line_number", "pp_unit"):
            if c not in out.columns:
                out[c] = np.nan
    return out


def build(hist: pd.DataFrame, which: str, lines: pd.DataFrame | None = None
          ) -> pd.DataFrame:
    """The feature frame for one side of the sport.

    THE SPLIT IS THE POINT, and it was missing.

    `split` existed and was never called, so both specs were built on the
    whole history: the log read "NHL-goalies features: 46664 rows, 973
    players" when there are only 2,665 goalie-games and 93 goalies. Every
    goalie's rolling ice time and save totals were computed through a queue
    of skaters, every skater's through goalies, and both models then
    projected all 973 players - which is why the concatenated projections
    held two rows per man and the board join had to guess between them.

    It fits cleanly and it is wrong, which is the expensive kind.
    """
    if which not in SPECS:
        raise ValueError(f"which must be one of {sorted(SPECS)}, got {which!r}")
    spec = SPECS[which]

    canon = to_canonical(hist)
    skaters, goalies = split(canon)
    side = skaters if which == "skaters" else goalies
    log.info("%s: %d of %d player-games, %d players",
             spec.name, len(side), len(canon), side["player_id"].nunique())
    if not len(side):
        raise ValueError(
            f"no {which} rows in this history. Positions present: "
            + ", ".join(sorted(canon.get('position', pd.Series(dtype=str))
                               .astype(str).unique())[:10]))

    built = EF.build(side, spec)
    if which == "skaters":
        built = _with_lines(built, lines)
    return built
