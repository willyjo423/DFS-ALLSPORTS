"""What a sport has to tell the engine about itself.

The engine - the model, the grading, the simulator, the optimiser - turned
out to be almost entirely sport-agnostic. Porting it from NFL to college
football touched exactly four things: which columns are features, which
positions count, what "enough history to fit on" means, and how points are
scored. Everything else worked unchanged.

So those four things are what a sport declares here, and nothing else in the
engine is allowed to know which sport it is running on. That rule is what
makes this worth doing: a bug fixed in the engine is fixed for every sport,
and a sport that needs the engine changed has found either a real gap in this
spec or a reason its data does not belong in the canonical frame.

What a spec deliberately does NOT carry
---------------------------------------
**Where the data comes from.** Every sport's ingest is different and always
will be - CFBD is not the MLB StatsAPI is not nflverse - and pretending
otherwise produced nothing but leaky abstractions. A sport's data layer is
its own code. Its only obligation is to emit the canonical frame.

**The correlation structure.** A QB-to-WR stack and a batting-order stack and
an NBA usage tradeoff are not the same object with different numbers; one is
positive within a drive, one is positive within an inning, and one is
NEGATIVE within a team. That belongs with the sport, not in a table of
coefficients pretending they are comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# WHAT COUNTS AS A FLEX SLOT IS A NAME, AND THE NAME IS NOT THE SAME IN EVERY
# SPORT. Football calls it FLEX and SFLEX; hockey and baseball call it UTIL.
#
# It lives here, in the module with no imports, because two modules had each
# written their own list out by hand and both lists were wrong - in different
# ways, for different sports, silently. `optimise` matched only the literal
# "FLEX", so hockey's UTIL became a required position no player holds, the
# constraint read 0 >= 1, and the integer program was INFEASIBLE on every NHL
# slate ever published. `ownership` listed FLEX and UTIL but not SFLEX, so
# college football's superflex became a phantom position and the real flex
# demand was understated by half.
#
# Neither crashed. One shipped empty server lineups behind a page that rendered
# perfectly; the other would have shipped ownership numbers quietly wrong in
# every row. So: one definition, imported by both, and a copy in
# docs/index.html that has to match it.
FLEX_SLOTS = frozenset({"FLEX", "SFLEX", "UTIL", "UTIL/FLEX", "G/UTIL"})


@dataclass(frozen=True)
class SportSpec:
    """Everything the engine needs, and nothing it does not."""

    name: str

    # --- the target ------------------------------------------------------
    # Raw volume columns the canonical frame must carry. These drive the
    # rolling features; a sport with different ones simply names them.
    usage: list[str]

    # Columns whose SHARE OF THE TEAM is worth knowing, named by their raw
    # column. A share is derived into `share_<col>` rather than replacing
    # `<col>`: listing a column in both places once made `ewm_carries` the
    # rolling mean of a RATIO instead of a volume, which raises nothing and
    # is close enough to plausible that grading would not obviously catch it.
    shares: list[str] = field(default_factory=list)

    # What counts as "he took part". Football sums to a player's own touches;
    # baseball would sum plate appearances and batters faced.
    touch_columns: list[str] = field(default_factory=list)

    # --- who gets projected ---------------------------------------------
    positions: list[str] = field(default_factory=list)

    # A player's first games carry no usable signal, and training on them
    # teaches the model that its features are noise. They are still
    # PREDICTED - that is the honest answer for a debut - just not fitted on.
    min_prior_games: int = 3

    # --- the distribution ------------------------------------------------
    quantiles: list[float] = field(
        default_factory=lambda: [0.10, 0.25, 0.50, 0.75, 0.90, 0.97])

    # How fast the past stops mattering. Four games is roughly where a role
    # change stops being noise and becomes the new normal in football; a
    # baseball season of 162 wants a much longer memory.
    halflife: float = 4.0

    # --- guards ----------------------------------------------------------
    min_train_rows: int = 500
    min_played_rows: int = 400

    # Extra feature columns the sport's own builder adds beyond the generic
    # ones - blowout margin in college football, park factor in baseball.
    extra_features: list[str] = field(default_factory=list)

    # --- correlation -----------------------------------------------------
    # How much of a player's variance is shared with his game, his team, and
    # the team-mates he competes with, by position. The SIGN matters more
    # than the size: a defence is negatively loaded on its own game because
    # it scores when the offence fails, and a model with that backwards will
    # happily stack a quarterback with the defence trying to stop him.
    #
    # Baseball inverts this again - a pitcher is strongly NEGATIVE with every
    # opposing hitter - which is exactly why these live with the sport and
    # not in a shared table of numbers that look comparable and are not.
    #
    # Left empty, every position falls back to weak defaults, and the
    # simulation behaves close to independent. That understates stacking
    # rather than inventing it, which is the safe direction to be wrong in.
    loadings: dict = field(default_factory=dict)

    # --- contests --------------------------------------------------------
    # Roster rules per (site, game type). Hard constraints, not preferences:
    # getting one wrong produces a lineup that cannot be entered.
    rosters: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.usage:
            raise ValueError(f"{self.name}: a sport with no usage columns "
                             f"has nothing to build features from")
        if not self.positions:
            raise ValueError(f"{self.name}: no positions declared")
        qs = list(self.quantiles)
        if qs != sorted(qs):
            raise ValueError(f"{self.name}: quantiles must be ascending, "
                             f"got {qs}")
        if not all(0.0 < q < 1.0 for q in qs):
            raise ValueError(f"{self.name}: quantiles must lie in (0, 1)")
        if len(set(qs)) != len(qs):
            raise ValueError(f"{self.name}: duplicate quantiles")

    # ------------------------------------------------------------------
    @property
    def features(self) -> list[str]:
        """Every feature column, in a fixed order.

        Order is fixed because a fitted model stores its column list and a
        reordering between fit and predict would silently hand the model the
        wrong numbers - no error, just worse projections.
        """
        out = [f"ewm_{c}" for c in self.usage]
        out += [f"ewm_share_{c}" for c in self.shares]
        out += ["ewm_points", "sd_points", "games_played", "ewm_touches",
                "team_ewm_points", "opp_ewm_points_allowed",
                "share_of_team_touches", "is_home",
                "played_last", "ewm_played", "played_rate"]
        out += list(self.extra_features)
        seen, uniq = set(), []
        for c in out:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq

    def qcol(self, q: float) -> str:
        return f"q{int(round(q * 100))}"

    @property
    def qcols(self) -> list[str]:
        return [self.qcol(q) for q in self.quantiles]
