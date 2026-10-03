"""How much of the field will own each player.

The measurement used to be closed
---------------------------------
DraftKings' standings endpoint returns a login page to a script - identically
139,263 bytes for a 182,000-entry contest and a 52,000-entry one, which is how
you know it is one generic wall rather than data. So for the whole life of this
module ownership was modelled against nothing, and the docstring said so.

That changed. A finished contest's standings CSV, exported from the browser,
carries a `%Drafted` column per player. One real board is now in hand:

    DK NFL Showdown, DET @ BUF, contest 195775024, 784 entries,
    58 priced players, 29 of them also on our published board.

Everything below is fitted against it. The numbers in CALIBRATION are the
actual field ownership from that contest and `check_calibration` re-scores the
model against them, so the claims in this docstring are executable rather than
editorial.

What the real board proved wrong
--------------------------------
The old version led with points and then leaned hard on points-per-dollar
(VALUE_WEIGHT 0.80, second-heaviest term). Measured against the field, value
is not weakly predictive - it is predictive with the WRONG SIGN:

    Spearman correlation with actual ownership, 29 players
        projected points alone      +0.86
        salary alone                +0.81
        points per $1,000           -0.42      <-- the old second-heaviest term
        the old blended appeal      +0.46

The field does not hunt value on a showdown slate. It rosters the expensive,
obvious players, and it rosters them roughly in projection order. Value is
what a model thinks a lineup should be built from; it is not what the person
clicking through the lobby in four minutes is looking at. Feeding it in with a
positive coefficient dragged the cheap end of the board up and the good players
down, and cost about half of the available rank correlation.

Two consequences, both visible in the contest:

  * Ownership was spread far too flat. The old model put 20% of all ownership
    on the cheapest fifteen players; the field put 8%. It had Ty Johnson at
    17.1% (actual 0.1%), Tyler Conklin 14.6% (actual 0.3%), Greg Dortch 15.6%
    (actual 1.1%). Nobody rosters the fourth tight end.

  * The genuinely popular players were called as contrarian. Dalton Kincaid
    came out at 6.4% against an actual 50.9%; DJ Moore 8.3% against 43.4%;
    Sam LaPorta 27.2% against 50.9%.

Retuning CONCENTRATION does not fix any of this, and that is worth stating
plainly because the old docstring named it as "the first thing to calibrate if
a real ownership file ever arrives". It was the wrong thing. Sweeping the
temperature from 0.5 to 6.0 moves mean absolute error from 12.6pp to 12.5pp -
temperature changes how steep a ranking is, and this ranking was not steep in
the wrong way, it was in the wrong ORDER. No temperature repairs an ordering.

What replaced it
----------------
Projected points, plus salary as its own positive term, plus a punt bonus that
now only reaches players with a real projection. Fitted by grid search on the
board above, minimising KL(actual || model) - not mean absolute error, because
MAE on a board where half the pool is owned near zero is minimised by staying
flat, which is the exact failure being repaired.

                          old        new     actual
    KL(actual||model)     0.427      0.179      -
    mean abs error        12.0pp      9.9pp     -
    Spearman rho           0.46       0.87      -
    share on top 8          62%        75%      74%
    share on bottom 15      20%        11%       8%

What this fit does NOT cover
----------------------------
One slate, one sport, one game type, 29 players. Specifically untested:

  * Classic boards. The scale constraint (`slot_demand`) is arithmetic and
    still holds, but the shape coefficients are fitted on showdown only.
  * MLB. The same contest export exists for MLB and has not been joined yet.
  * The ceiling term. The published board carries no ceiling column, so
    ceiling could not be separated from the median. It is folded into the
    points term at a fixed blend rather than given a free coefficient, which
    keeps the fitted points:salary ratio intact instead of quietly changing it.

Why it matters even as an estimate
----------------------------------
A tournament is not won by a good lineup; it is won by a good lineup nobody
else had. The same contest shows what the old errors cost: Jared Goff was
modelled at 59.2% owned, came in at 23.5%, and scored 32.78 - the largest
leverage play on the slate, and the model had it marked as chalk. He was in
four of the top five lineups.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .spec import FLEX_SLOTS as O_FLEX_SLOTS

log = logging.getLogger(__name__)

# How sharply the field concentrates. Fitted, not eyeballed: grid search over
# 0.20-4.00 against the DET @ BUF contest, minimising KL(actual || model).
# The surface is shallow - anything from 0.7 to 1.1 scores within 0.01 KL - so
# this is not a knife-edge value, which is the honest thing to say about a
# parameter fitted on a single board.
CONCENTRATION = 0.90

# The field's preferences. Three terms now, where there were five.
#
# POINTS leads, as it always did, and on the real board it is the single best
# predictor available (rho +0.86 on its own).
#
# SALARY is new and replaces VALUE outright. Ownership rises with price, near
# monotonically (rho +0.81), because expensive players are the ones the field
# has heard of and the ones the projections it reads also like. The old model
# had no term for this at all - it had a small STUD_BONUS at the very top and
# a points-per-dollar term pulling the opposite way everywhere else.
#
# VALUE is gone. Not down-weighted, gone. See the docstring: measured against a
# real field it correlates -0.42 with ownership. A term that points backwards
# is worse than a missing one.
POINTS_WEIGHT = 1.00
SALARY_WEIGHT = 0.80
PUNT_BONUS = 0.80

# Ceiling, folded into the points term rather than given its own coefficient.
# Tournament players do chase upside, but the published board carries no
# ceiling column, so a free coefficient for it could not be fitted and adding
# an unfitted one would silently change the points:salary ratio that WAS
# fitted. This blend keeps that ratio and still lets upside move a player.
CEILING_BLEND = 0.20

# A punt is played for salary relief, but only when the cheap player might
# actually score. The old gate was the 35th percentile of projection, which on
# a 35-man showdown board admits most of the bench; the fit wanted it at the
# 60th. Frank Gore Jr. at $400 was 21.3% owned on the real board, so genuine
# punts exist - they are just much rarer than the old gate allowed.
PUNT_PROJECTION_GATE = 0.60
PUNT_SALARY_PCT = 0.15


def slot_demand(roster: dict) -> dict[str, float]:
    """How many of each position a single entry must field.

    This is where the scale comes from, and it is the one part of this module
    that is arithmetic rather than fitted. A classic lineup holds exactly one
    quarterback, so quarterback ownership across the slate sums to exactly 1.0
    - and any model that produces something else is wrong in a way that can be
    checked rather than argued about.
    """
    slots = roster["slots"]
    flex_positions = list(roster.get("flex_positions") or [])
    demand: dict[str, float] = {}
    # FLEX_SLOTS rather than a tuple of names written out here. The literal
    # tuple this replaced listed FLEX and UTIL and not SFLEX, so college
    # football's superflex was counted as a required position called "SFLEX" -
    # a position no player holds, so its demand went to a phantom while the
    # real flex demand was understated by half. That is not a crash; it is
    # every CFB ownership number being quietly wrong.
    for s in slots:
        if s in O_FLEX_SLOTS or s == "CPT":
            continue
        demand[s] = demand.get(s, 0.0) + 1.0

    # The HARD minimum per position, before any flex is shared out: how many
    # of each a lineup must field whatever else it does. Kept because an
    # explicit override has to respect it.
    required = dict(demand)

    spare = sum(1 for s in slots if s in O_FLEX_SLOTS)
    if spare and not flex_positions:
        # Spare slots with nothing declared eligible for them. There is no
        # position to attribute that demand to, so the total cannot reach the
        # roster size and the check at the bottom will say so loudly.
        #
        # The one roster shaped like this is SHOWDOWN - every slot drawn from
        # the whole board - and `project` never calls this function for one: it
        # short-circuits on "CPT" in slots and scales the whole board to the
        # roster size directly. So this branch means a roster has flex slots
        # and forgot to say who may fill them, which is worth reporting rather
        # than returning a silent zero for.
        log.error("this roster has %d flex slot(s) and an EMPTY "
                  "flex_positions, so there is no position to attribute their "
                  "demand to. Showdown is handled separately; any other "
                  "roster shaped like this needs flex_positions filled in.",
                  spare)
    if spare and flex_positions:
        # A flex is shared. Split it by how much of the flex-eligible pool each
        # position represents, which is closer to how a field actually fills it
        # than splitting evenly would be.
        #
        # THE EVEN-SPLIT FALLBACK WAS UNREACHABLE AND THE EFFECT WAS SILENT.
        # Written as `total = sum(...) or float(len(flex_positions))`, total
        # was never zero, so `if total else 1/len(...)` never fired - and when
        # no flex-eligible position is ALSO a required slot, every `share`
        # came out as 0/len = 0 and the spare slots were handed no demand at
        # all. A roster whose every slot is flex totalled 0.0, which is every
        # ownership number on that board divided by nothing.
        base = {p: demand.get(p, 0.0) for p in flex_positions}
        total = sum(base.values())
        for p in flex_positions:
            share = (base[p] / total) if total > 0 else 1.0 / len(flex_positions)
            demand[p] = demand.get(p, 0.0) + spare * share

    # The total is load-bearing: it is what every ownership number is scaled
    # to, so whatever happens below has to leave it equal to the roster size.
    want = float(sum(1 for s in slots if s != "CPT"))

    def _rebalance(d: dict[str, float], pinned: set[str]) -> dict[str, float]:
        """Scale everything not pinned so the total comes back to `want`."""
        left = want - sum(d[p] for p in pinned if p in d)
        free = {p: v for p, v in d.items() if p not in pinned}
        have = sum(free.values())
        if left < -1e-9:
            log.error("the pinned position demands already total %.2f on a "
                      "%d-man roster, so they cannot all be satisfied; "
                      "leaving the arithmetic alone", want - left, int(want))
            return d
        if have <= 0:
            if left > 1e-9:
                log.error("nothing is left to carry %.2f roster spots after "
                          "the pinned demands; leaving the arithmetic alone",
                          left)
                return d
            return d
        scale = left / have
        out = dict(d)
        for p in free:
            out[p] = free[p] * scale
        return out

    # A sport may state the demand it has MEASURED, instead of leaving it to
    # the flex-pool-share split above.
    #
    # College football is why this exists. The superflex takes a quarterback,
    # a runner or a receiver, so the share split hands quarterbacks about 1.33
    # per lineup - they are roughly a third of the eligible pool. The field
    # does not play it that way at all: two quarterbacks IS the shape of the
    # format, and a quarterback fills the superflex in the large majority of
    # entries. Scaling quarterback ownership to 1.33 when the field rosters
    # nearer 1.9 understates every quarterback on the board by a third and
    # overstates every running back and receiver to pay for it - which lands
    # directly on leverage, since leverage is a projection rank minus an
    # ownership rank.
    #
    # Only a measured number belongs here. A guess wearing a measurement's
    # clothes is worse than the arithmetic it replaces, because the arithmetic
    # is at least checkable from the roster rules alone.
    override = {str(p): float(v)
                for p, v in (roster.get("position_demand") or {}).items()
                if str(p) in demand}
    # A negative or impossible override is refused rather than applied.
    # `_rebalance` only guarded its own total, so a `position_demand` of -1
    # passed slot_demand's self-check - the total still came to 8.000 - and
    # then `_allocate(want=-1)` returned zeros for that position, leaving the
    # ownership actually produced summing to 9.0 on an eight-man roster. A
    # wrong number that the self-check calls right is worse than no check.
    # The bound is not 0-to-roster-size: it is "at least the required minimum
    # for this position, and no more than the roster size minus every OTHER
    # position's required minimum". A demand of 8 quarterbacks on an eight-man
    # roster that also requires two backs and three receivers is arithmetically
    # impossible, and the looser bound let it through - `_rebalance` then had
    # nothing left to distribute, logged, and returned a total of 2.0 on an
    # eight-man roster.
    bad = {}
    for p, v in override.items():
        lo = required.get(p, 0.0)
        hi = want - sum(c for q, c in required.items() if q != p)
        if not (lo - 1e-9 <= v <= hi + 1e-9):
            bad[p] = (v, lo, hi)
    if bad:
        log.error("these position_demand values are arithmetically impossible "
                  "on a %g-man roster and are IGNORED: %s", want,
                  ", ".join(f"{p}={v:g} (must be {lo:g} to {hi:g})"
                            for p, (v, lo, hi) in sorted(bad.items())))
        override = {p: v for p, v in override.items() if p not in bad}
    if override:
        demand.update(override)
        demand = _rebalance(demand, set(override))

    # A position with a hard ceiling cannot be demanded above it, and whatever
    # the clip takes away has to go somewhere or the total stops matching the
    # roster. A roster that capped quarterbacks at one while leaving them
    # flex-eligible would otherwise have been handed an impossible 1.33.
    # ITERATED, because one pass is not enough. Rebalancing to restore the
    # total pushes demand ONTO the uncapped positions, and a ceiling that did
    # not bind before the rebalance can bind after it: a QB cap of 1.4 against
    # a share-split demand of 1.5 came out of a single pass at 1.5, over its
    # own ceiling. Three passes is ample for the handful of positions any
    # roster has, and the loop reports if it has not settled.
    caps = {str(p): float(v)
            for p, v in (roster.get("max_position") or {}).items()
            if str(p) in demand}
    pinned = set(override)
    for _ in range(3):
        clipped = {p: min(demand[p], c) for p, c in caps.items()
                   if demand[p] > c + 1e-9}
        if not clipped:
            break
        log.info("per-position ceilings bind: %s", ", ".join(
            f"{p} {demand[p]:.2f}->{c:.2f}" for p, c in clipped.items()))
        demand.update(clipped)
        pinned |= set(clipped)
        demand = _rebalance(demand, pinned)
    over = {p: (demand[p], c) for p, c in caps.items()
            if demand[p] > c + 1e-6}
    if over:
        log.error("position demand is still above its ceiling after three "
                  "rebalances, so ownership at %s will be scaled to more of "
                  "them than a lineup may hold: %s", ", ".join(sorted(over)),
                  ", ".join(f"{p} {d:.2f} > {c:.2f}"
                            for p, (d, c) in sorted(over.items())))

    total = sum(demand.values())
    if abs(total - want) > 1e-6:
        log.error("position demand totals %.3f on a %d-man roster. Every "
                  "ownership number is scaled to this, so it being wrong is "
                  "every ownership number being wrong: %s",
                  total, int(want),
                  ", ".join(f"{p} {v:.2f}" for p, v in sorted(demand.items())))
    return demand


def appeal(pool: pd.DataFrame) -> pd.Series:
    """How attractive each player looks to somebody building a lineup fast.

    Built from what the field demonstrably responds to - projected points and
    price - rather than from what a model thinks is efficient. That distinction
    used to be stated as a principle and implemented backwards; it is now
    implemented the way the measured board says.
    """
    salary = pd.to_numeric(pool["salary"], errors="coerce")

    def column(name: str) -> pd.Series:
        """A numeric Series aligned to the pool, whether or not the column
        exists. `pool.get(name)` returns None for a missing column and
        `pd.to_numeric(None)` returns a bare scalar, not a Series - which is
        how the first version of this turned a missing column into an
        AttributeError two calls later instead of an empty column here."""
        if name not in pool.columns:
            return pd.Series(np.nan, index=pool.index, dtype=float)
        return pd.to_numeric(pool[name], errors="coerce")

    # The AVAILABILITY-ADJUSTED projection, not the conditional one. The field
    # does not roster a player who is not playing, however good he is when he
    # does - so ownership has to be driven by the unconditional expectation.
    # Reading the conditional median here would have made a doubtful star look
    # like the chalk of the slate.
    median = column("mean")
    if median.isna().all():
        median = column("median")
    median = median.fillna(0.0)

    ceiling = column("ceiling")
    if "p_play" in pool.columns:
        ceiling = ceiling * column("p_play").fillna(1.0)
    ceiling = ceiling.fillna(median)

    # One points term, with upside blended in at a fixed weight. See
    # CEILING_BLEND for why this is not a separate coefficient.
    points = (1.0 - CEILING_BLEND) * median + CEILING_BLEND * ceiling

    def z(s: pd.Series) -> pd.Series:
        sd = float(s.std(ddof=0))
        return (s - float(s.mean())) / sd if sd > 1e-9 else s * 0.0

    score = POINTS_WEIGHT * z(points) + SALARY_WEIGHT * z(salary.fillna(0.0))

    # The punt bonus, gated on a real projection. Without the gate the bonus
    # lands hardest on exactly the players who will not take a snap, which is
    # how an early version of this module arrived at a board led by long
    # snappers at 65% ownership.
    rank = salary.rank(pct=True)
    usable = median >= float(median.quantile(PUNT_PROJECTION_GATE))
    score = score + PUNT_BONUS * ((rank <= PUNT_SALARY_PCT) & usable).astype(float)

    # A player nobody can use is owned by nobody, whatever his price implies.
    if "playing" in pool.columns:
        score = score.mask(pool["playing"].isin(("out", "doubtful")), -1e9)
    return score


# No single player is ever owned by two-thirds of a field at one roster slot.
# On the DET @ BUF board Josh Allen was 62.5% in the flex and 24.0% at captain,
# so 86.5% across the six slots - which is why this cap is applied to the
# allocation and not to a per-slot share, and why the fit still preferred 0.65
# over 0.80 or 0.95 when it was allowed to choose.
MAX_OWNERSHIP = 0.65


def _allocate(scores: np.ndarray, want: float,
              concentration: float) -> np.ndarray:
    """Shape from the softmax, scale from the roster, cap applied honestly.

    The cap is the awkward part. Clipping a player at 65% removes ownership
    that the roster constraint says must exist somewhere - every entry still
    fields that slot - so the excess is pushed onto the players who are not at
    the cap, repeatedly until it settles. An earlier version redistributed
    across the WHOLE board instead of within the constrained group, which broke
    the very sums that make this model more than a guess: quarterback ownership
    came out at 128% when exactly one quarterback is in every lineup.
    """
    n = len(scores)
    if n == 0 or want <= 0:
        return np.zeros(n)
    out = _softmax(scores, concentration) * want

    # The cap has to be feasible or the redistribution below never settles.
    # If the pool is smaller than the roster demands - six showdown slots
    # against eight priced players, say - then every player must be owned
    # above 65% by arithmetic, and holding the cap there would push the excess
    # round the board fifty times and then return values ABOVE the cap anyway,
    # which is what the old `min(MAX_OWNERSHIP, ...)` line silently did. It
    # read as if it lowered the cap on a tiny slate; its condition was
    # `want < n * 1e-9`, which is never true for a real board, so it was dead
    # code that evaluated to MAX_OWNERSHIP every time.
    if want > n * MAX_OWNERSHIP:
        # Infeasible: every player must sit above the cap for the roster to
        # fill, so there is exactly one answer and it is uniform. Returning it
        # directly also avoids the iteration below converging only
        # geometrically onto it and leaving a value a hair over the cap.
        log.warning("ownership cap raised from %.0f%% to %.0f%%: %d players "
                    "cannot fill %.1f roster slots without it",
                    100 * MAX_OWNERSHIP, 100 * want / n, n, want)
        return np.full(n, want / n)
    cap = MAX_OWNERSHIP

    for _ in range(50):
        over = out > cap + 1e-12
        if not over.any():
            break
        excess = float((out[over] - cap).sum())
        out[over] = cap
        room = ~over
        if not room.any() or out[room].sum() <= 0:
            # Everyone is at the cap; the demand cannot be met without
            # exceeding it, which means the slate is smaller than the roster.
            break
        out[room] += excess * (out[room] / out[room].sum())
    return out


def project(pool: pd.DataFrame, roster: dict,
            concentration: float = CONCENTRATION) -> pd.Series:
    """Projected ownership per player, as a fraction of the field.

    Softmax of appeal within each position, scaled so the position sums to
    what a lineup actually demands. The softmax supplies the shape; the roster
    supplies the scale.
    """
    score = appeal(pool)
    out = pd.Series(0.0, index=pool.index, dtype=float)

    if "CPT" in roster["slots"]:
        # Showdown has no position requirements - six slots drawn from the
        # whole board - so the board is one pool and the demand is six. The
        # number this produces is therefore CPT + FLEX ownership combined,
        # which is what a standings export sums to as well.
        out.loc[:] = _allocate(score.to_numpy(dtype=float),
                               float(len(roster["slots"])), concentration)
    else:
        demand = slot_demand(roster)
        for position, idx in pool.groupby(
                pool["position"].astype(str)).groups.items():
            idx = list(idx)
            out.loc[idx] = _allocate(score.loc[idx].to_numpy(dtype=float),
                                     float(demand.get(position, 0.0)),
                                     concentration)

    log.info("ownership: %d players, total %.0f%% against %d roster slots",
             len(out), 100 * out.sum(), len(roster["slots"]))
    return out


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    x = np.where(np.isfinite(x), x, -1e9)
    z = x * temperature
    z = z - z.max()
    e = np.exp(z)
    total = e.sum()
    return e / total if total > 0 else np.full_like(e, 1.0 / len(e))


def duplication(own: np.ndarray, field_size: int) -> float:
    """Roughly how many other entries will field this exact lineup.

    The number that turns a good lineup into a good BET. A lineup whose six
    players are each owned by 30% is played by about 0.3^6 of the field - which
    in a 200,000-entry tournament is 146 other people splitting first place
    with you. The same score from six 8%-owned players is yours alone.

    Approximate on purpose: it ignores that roster rules make some combinations
    impossible, which overstates duplication slightly and uniformly. Since it
    is used to RANK lineups against each other, a uniform bias is harmless in a
    way that a wrong shape would not be.
    """
    p = np.clip(np.asarray(own, dtype=float), 1e-9, 1.0)
    return float(field_size * np.prod(p))


def leverage(pool: pd.DataFrame, own: pd.Series) -> pd.Series:
    """Where the model likes a player more than the field will.

    Expressed as the gap between two ranks rather than two raw numbers,
    because ownership and projected points are not in the same units and
    subtracting them directly would mean nothing.
    """
    median = pd.to_numeric(pool.get("median"), errors="coerce").fillna(0.0)
    return (median.rank(pct=True) - own.rank(pct=True)).round(3)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
# The real board. name -> (salary, our published median, actual CPT+FLEX
# ownership as a fraction). Taken from the DraftKings contest-standings export
# for contest 195775024 (DET @ BUF showdown, 784 entries) joined to our own
# published dk-showdown-153434.json. Only the 29 players that appear in both
# are here; the three the field rostered that our board did not carry at all
# (Tyler Bass 25.1%, Joshua Palmer 19.1%, Jake Bates 16.2%) are a pool bug,
# not an ownership bug, and are recorded in MISSING_FROM_BOARD instead.
CALIBRATION: dict[str, tuple[int, float, float]] = {
    "Josh Allen":          (11400, 24.06, 0.8648),
    "Jahmyr Gibbs":        (12000, 20.93, 0.6632),
    "Sam LaPorta":         ( 5400, 11.35, 0.5089),
    "Dalton Kincaid":      ( 6800,  8.75, 0.5089),
    "DJ Moore":            ( 7800, 10.21, 0.4337),
    "James Cook III":      ( 9400, 14.67, 0.3699),
    "Amon-Ra St. Brown":   (10400, 18.35, 0.3520),
    "Jameson Williams":    ( 7400, 13.46, 0.2832),
    "Jared Goff":          ( 9600, 19.00, 0.2347),
    "Frank Gore Jr.":      (  400,  8.02, 0.2130),
    "Khalil Shakir":       ( 6400, 11.33, 0.1595),
    "Bills":               ( 3600,  6.00, 0.1289),
    "Dawson Knox":         ( 2800,  5.93, 0.1276),
    "Isaac TeSlaa":        ( 3800,  5.37, 0.1135),
    "Jackson Hawes":       (  600,  3.35, 0.0804),
    "Keon Coleman":        ( 4000,  6.43, 0.0753),
    "Ray Davis":           ( 2400,  3.96, 0.0664),
    "Sione Vaki":          ( 3000,  1.72, 0.0523),
    "Brock Wright":        ( 1800,  3.96, 0.0510),
    "Lions":               ( 3400,  6.00, 0.0370),
    "Greg Dortch":         ( 1000,  4.76, 0.0115),
    "Tay Martin":          (  800,  3.90, 0.0089),
    "Keleki Latu":         (  200,  2.73, 0.0089),
    "Tom Kennedy":         ( 1200,  2.59, 0.0051),
    "Jacob Saylors":       ( 1400,  1.61, 0.0051),
    "Trent Sherfield Sr.": (  200,  2.41, 0.0038),
    "Tyler Conklin":       ( 1600,  4.25, 0.0026),
    "Ty Johnson":          ( 2000,  5.36, 0.0013),
    "Dominic Lovett":      (  200,  2.41, 0.0013),
}

# Players the field rostered that our published board did not contain at all.
# Not an ownership failure - a pool failure, upstream in the DraftKings
# draftables join. Joshua Palmer was in the winning lineup. Kickers are a
# legal showdown position and were simply absent.
MISSING_FROM_BOARD: dict[str, float] = {
    "Tyler Bass": 0.2513,
    "Joshua Palmer": 0.1913,
    "Jake Bates": 0.1620,
}

# What the model scored on that board BEFORE this rewrite, and what it has to
# beat to count as an improvement. These are hard numbers from the join, not
# targets chosen to be easy.
BASELINE = {"kl": 0.427, "mae": 0.120, "rho": 0.46}


def check_calibration(concentration: float = CONCENTRATION) -> dict[str, float]:
    """Re-score this module against the one real board there is.

    Returns KL(actual || model), mean absolute error in ownership fraction,
    and Spearman rank correlation. A test asserts these beat BASELINE, so a
    future edit that quietly reintroduces a value term fails rather than
    shipping.
    """
    names = list(CALIBRATION)
    pool = pd.DataFrame({
        "name": names,
        "position": ["FLEX"] * len(names),
        "salary": [CALIBRATION[n][0] for n in names],
        "median": [CALIBRATION[n][1] for n in names],
    })
    roster = {"slots": ["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX"],
              "flex_positions": []}
    model = project(pool, roster, concentration=concentration).to_numpy(float)
    actual = np.array([CALIBRATION[n][2] for n in names], dtype=float)

    p = np.clip(model / model.sum(), 1e-9, None)
    q = actual / actual.sum()
    kl = float((q[q > 0] * np.log(q[q > 0] / p[q > 0])).sum())

    n = len(names)
    rm = np.argsort(np.argsort(-model))
    ra = np.argsort(np.argsort(-actual))
    rho = 1.0 - 6.0 * float(((rm - ra) ** 2).sum()) / (n * (n * n - 1))

    return {"kl": kl,
            "mae": float(np.abs(model - actual).mean()),
            "rho": rho,
            "n": float(n)}


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    got = check_calibration()
    print(f"{'':16}{'now':>9}{'before':>9}")
    print(f"{'KL(actual||fit)':16}{got['kl']:9.3f}{BASELINE['kl']:9.3f}")
    print(f"{'mean abs error':16}{100*got['mae']:8.1f}%{100*BASELINE['mae']:8.1f}%")
    print(f"{'spearman rho':16}{got['rho']:9.2f}{BASELINE['rho']:9.2f}")
    print(f"{'players':16}{int(got['n']):9d}")
    print()
    print("missing from the published board entirely:")
    for who, own in MISSING_FROM_BOARD.items():
        print(f"   {who:<16}{100*own:5.1f}% owned")
