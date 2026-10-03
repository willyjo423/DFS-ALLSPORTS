"""Saturday's college football board, published as the page's own data file.

    docs/data/cfb_dk_classic_<draft group>.json
    docs/data/manifest.json

The payload is the same shape the baseball and hockey builds emit, which is
the whole point: the page reads the roster, the players, the games and the
correlation loadings out of the slate rather than knowing anything about a
sport, so one page renders all three and there is one place to fix a
rendering bug.

Why the join here is BETTER than the other two sports'
------------------------------------------------------
Hockey joins MoneyPuck to DraftKings by name, because the two sources key on
different player ids with no crosswalk between them, and that is the weakest
link in that build. College football does not have that problem - and not by
luck. `cfb_data.attach_history` already resolves DraftKings' spelling to a
CFBD athlete id through four passes, the last two of them restricted to the
school a player is actually on, because "Ryan Williams" is ambiguous across
four seasons of college football and completely unambiguous at Alabama.

So the projections join on an ID, not a name, and the name matching happens
once in the place that was built to do it properly. What is measured here is
whether the resulting coverage is good enough to publish, which is a
different question and still worth asking out loud.

What the model cannot see, stated plainly
-----------------------------------------
**No targets.** CFBD publishes receptions, not targets. In football, target
share separates a receiver who IS the offence from one who happens to play in
a good one, and it does so BEFORE the catches arrive. Receptions are the same
signal after the fact, contaminated by catch rate and by the quarterback.
There is no substitute on this feed, so this model is weaker at receiver than
an NFL one, and that is a property of the data rather than something tuning
will fix.

**No depth chart and no injury report.** College football has no equivalent of
the NFL's Wednesday practice report. DraftKings removes players it knows are
out; everybody else is priced as if available. What partly replaces it is the
blowout feature - starters sit in the fourth quarter of a forty-point win,
which is frequent and learnable - and the market, which prices a game the
backups will finish.

**The portal.** A projection built on last season is a projection of a player
who may now be at a different school with a different role. The history is
keyed on athlete id, so a transfer keeps his own record and the team features
correctly follow his new school; what no feature can know is that his role
changed. Early in a season this is the largest error in the file.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import cfb_cache as C
import cfb_data as D
import cfb_odds as CO
import cfb_sport as S
from engine import model as EM
from engine import optimise as O
from engine import ownership as OWN
from engine import simulate as SIM
from engine.spec import FLEX_SLOTS

log = logging.getLogger("cfb_publish")

SPORT = "cfb"
EASTERN = ZoneInfo("America/New_York")
DOCS = Path("docs")
DATA = DOCS / "data"
ARCHIVE_DAYS = 120

ROSTER = S.DK_CLASSIC
roster_position = S.roster_position

# A board must cover at least this many games before it is treated as a
# classic slate. DraftKings requires players from two different games, so a
# one-game board is a showdown however it is labelled, and the classic roster
# is INFEASIBLE on it - which would be reported as "the integer program did
# not solve" and look like a solver problem rather than the wrong roster.
MIN_GAMES_FOR_CLASSIC = 2

# The least a board may price before it counts as a salary-cap game at all.
# The DraftKings lobby also lists pick'em and tiers contests, whose rows carry
# the same field names with completely different meanings - a hockey run came
# back reading "salaries $1-$272", which is a tier index, not money. A board
# like that builds a lineup that cannot be entered anywhere. The cheapest real
# college skill player is about $2,000.
MIN_TOP_SALARY = 1000

# ----------------------------------------------------------- market scaling
# How hard the market is allowed to move a projection, as a multiplier on
# every point of a player's distribution.
MARKET_CLIP = (0.75, 1.35)
# How many of a team's players count toward "what this model thinks the
# offence will do". Capped so a team with twenty-eight priced players is not
# measured as a bigger offence than one with nine.
MARKET_TOP_N = 8
# And the same number as the MINIMUM, which is the fix for a defect that made
# the market factor a function of ROW COUNTS rather than of the market.
#
# These were 8 and 5. `model[team]` is the SUM of a team's top-N projections,
# so a team with five priced players contributed a sum of five and a team with
# nine contributed a sum of eight - and the ratio-to-ratio comparison then
# read the shorter team as a far worse offence than the market did and lifted
# every one of its players to compensate.
#
# Measured, with every player projected identically and every team implied at
# exactly 27.0 - a board where the correct factor is 1.00 everywhere:
#
#       5 priced players -> x1.350   (pinned at the clip)
#       6 priced players -> x1.167
#       7 priced players -> x1.000
#       8 or more        -> x0.875
#
# A 35% lift from nothing but a row count, applied to every quantile. Equal
# to MARKET_TOP_N, the denominator is the same length for every team or the
# team gets no factor at all, which is the only version of this that is a
# comparison rather than an artefact. A real college board prices twenty-odd
# skill players per school, so this excludes nobody in practice - and when it
# does exclude a team, that team is left at 1.00 and the log says so.
MARKET_MIN_PRICED = MARKET_TOP_N


def market_factor(pool: pd.DataFrame, lines: pd.DataFrame) -> pd.Series:
    """How much to move each player, from what the market knows and the model
    does not.

    THE NAIVE VERSION DOUBLE COUNTS, and this is the whole reason this
    function is more than one line. Scaling every player by his team's implied
    total over the slate average says "this is a good offence" twice: the
    projection already knows it, because `team_ewm_points` and
    `opp_ewm_points_allowed` are features it was fitted on. Applying the
    market ratio on top of that stretches a board that was already stretched,
    and the direction of the error is exactly the one that matters - it
    exaggerates the chalk.

    So the comparison is RATIO TO RATIO. The model's own view of a team's
    offence is readable directly off its output: add up the projected means of
    that team's priced players. If the model already has this team at 1.3
    times the slate average and the market also has it at 1.3 times, the
    factor is 1.0 and nothing moves. If the market has it at 1.5 and the model
    at 1.3, the factor is about 1.15 - and that 15% is the part of the market
    that is genuinely new information rather than a restatement of the
    features.

    That framing is what makes the elasticity 1.0 rather than a guess. With
    the double counting removed there is no reason to damp the remainder: it
    is a disagreement between the best public estimate of a game and a model
    that has not seen the game.
    """
    if lines is None or not len(lines):
        return pd.Series(1.0, index=pool.index)

    mean_col = "mean" if "mean" in pool.columns else "median"
    teams = pool["team"].astype(str)

    model, thin = {}, []
    for team, grp in pool.groupby(teams):
        vals = pd.to_numeric(grp[mean_col], errors="coerce").dropna()
        if len(vals) < MARKET_MIN_PRICED:
            thin.append(f"{team} ({len(vals)})")
            continue
        # Exactly MARKET_TOP_N for every team, always - see the note on the
        # constants. A sum over a variable number of players is not a
        # comparison between teams.
        model[str(team)] = float(vals.nlargest(MARKET_TOP_N).sum())
    if thin:
        log.info("%d team(s) have fewer than %d projected players and are "
                 "left at a market factor of 1.00 rather than compared on a "
                 "shorter denominator: %s", len(thin), MARKET_MIN_PRICED,
                 ", ".join(sorted(thin)))
    if not model:
        log.warning("no team has %d priced players with a projection, so the "
                    "market cannot be compared to the model and is not "
                    "applied", MARKET_MIN_PRICED)
        return pd.Series(1.0, index=pool.index)

    mkt = dict(zip(lines["team"].astype(str),
                   pd.to_numeric(lines["implied_total"], errors="coerce")))
    shared = [t for t in model if t in mkt and pd.notna(mkt[t])]
    if len(shared) < 2:
        log.warning("only %d team(s) have both a market line and enough "
                    "priced players; the market is not applied", len(shared))
        return pd.Series(1.0, index=pool.index)

    model_mean = float(np.mean([model[t] for t in shared]))
    mkt_mean = float(np.mean([mkt[t] for t in shared]))
    if model_mean <= 0 or mkt_mean <= 0:
        return pd.Series(1.0, index=pool.index)

    lo, hi = MARKET_CLIP
    factor = {}
    for t in shared:
        model_rel = model[t] / model_mean
        mkt_rel = float(mkt[t]) / mkt_mean
        if model_rel <= 0:
            continue
        factor[t] = float(np.clip(mkt_rel / model_rel, lo, hi))

    pinned = int(sum(1 for f in factor.values() if f in (lo, hi)))
    if pinned:
        log.info("%d team factor(s) hit the %.2f-%.2f clip. A clip that binds "
                 "often is either a market this model badly disagrees with or "
                 "a team whose priced depth is unrepresentative - worth a "
                 "look, not an automatic problem.", pinned, lo, hi)
    if factor:
        # A factor of 1.35 is a 35% LIFT and 0.75 is a 25% FADE, so what gets
        # printed is f - 1. The first version printed `f:+.0%` through a
        # `.replace("+1", "+")`, which turned a 25% fade into "+75%" and a
        # no-change into "+00%" - and this is the only human-readable record
        # of what the market did to the board. It also printed every team
        # twice whenever fewer than six had factors, because `ranked[:3] +
        # ranked[-3:]` overlaps on a short list.
        ranked = sorted(factor.items(), key=lambda kv: -kv[1])
        show = ranked[:3] + [kv for kv in ranked[-3:] if kv not in ranked[:3]]
        log.info("market vs model, biggest disagreements: %s", ", ".join(
            f"{t} {(f - 1.0):+.0%}" for t, f in show))
        log.info("market factors: %.2f to %.2f over %d team(s)",
                 min(factor.values()), max(factor.values()), len(factor))
    return teams.map(lambda t: factor.get(str(t), 1.0)).astype(float)


# Every column that is measured in DraftKings points and therefore has to move
# with the market. Listed explicitly rather than inferred from a name pattern:
# `p_play` is a probability and `salary` is money, and scaling either of them
# would be a different kind of wrong that nothing downstream would notice.
def scaled_columns(quantiles: list[float]) -> list[str]:
    return ([f"q{int(round(q * 100))}" for q in quantiles]
            + ["median", "ceiling", "mean", "cond_mean", "spread"])


def apply_market(pool: pd.DataFrame, lines: pd.DataFrame,
                 quantiles: list[float]) -> pd.DataFrame:
    out = pool.copy()
    f = market_factor(out, lines)
    for c in scaled_columns(quantiles):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") * f
    out["market_factor"] = f
    if lines is not None and len(lines):
        by_team = lines.set_index(lines["team"].astype(str))
        for src, dest in (("implied_total", "implied_total"),
                          ("game_total", "game_total"),
                          ("team_spread", "team_spread"),
                          ("opponent_implied", "opponent_implied")):
            if src in by_team.columns:
                out[dest] = out["team"].astype(str).map(by_team[src])
    return out


# ------------------------------------------------------------- which boards
def candidate_slates(draft_group: int | None, look: int) -> list[tuple]:
    """Which boards to publish, biggest first.

    Ranked by how many GAMES a board covers rather than by contest count.
    Cheap single-game contests are numerous, so counting contests picks a
    showdown over the main slate - the mistake the baseball build made and
    fixed, and college football has more single-game contests than any other
    sport on the site.
    """
    def usable(dg, b, kind: str) -> str | None:
        """Why this board is not a classic cap slate, or None if it is.

        APPLIED TO AN EXPLICITLY NAMED DRAFT GROUP TOO, which it was not. The
        `--draft-group` branch used to return before any of these checks, and
        `D.slates()` lists every draft group the lobby is selling - including
        tiers and pick'em contests, whose rows carry the same field names with
        completely different meanings. A tiers board has `salary` holding a
        tier INDEX, so naming one published a lineup spending $24 against a
        $50,000 cap, and `check_entry` called it legal because $24 is under
        the cap. Naming a draft group says which board; it does not say the
        board is a cap game.
        """
        if "showdown" in kind.lower() or "single game" in kind.lower():
            return (f"the lobby calls it {kind!r}. The classic roster is "
                    f"infeasible on a one-game board and the failure would "
                    f"be reported as a solver problem rather than as the "
                    f"wrong roster")
        if len(b) < 40:
            return f"it prices only {len(b)} players"
        top = pd.to_numeric(b["salary"], errors="coerce").max()
        if not (top and top >= MIN_TOP_SALARY):
            return (f"its dearest player costs "
                    f"${0 if pd.isna(top) else int(top):,}, which is a tier "
                    f"index or a rank, not money. This is not a salary-cap "
                    f"slate")
        if b["game"].nunique() < MIN_GAMES_FOR_CLASSIC:
            return (f"it covers {b['game'].nunique()} game(s), so it is a "
                    f"showdown by shape whatever the lobby calls it")
        return None

    listed = D.slates()
    if draft_group:
        row = listed[listed["draft_group"] == int(draft_group)]
        label = str(row["example"].iloc[0]) if len(row) else "(given)"
        starts = str(row["starts_text"].iloc[0]) if len(row) else ""
        kind = str(row["game_type"].iloc[0]) if len(row) else ""
        b = D.board(int(draft_group))
        why = usable(draft_group, b, kind)
        if why:
            log.error("draft group %s was named explicitly, but %s. It is "
                      "NOT published: a board that is not a cap game produces "
                      "a lineup that cannot be entered anywhere.",
                      draft_group, why)
            return []
        return [(int(draft_group), label, b, starts)]

    out = []
    for r in listed.head(look).to_dict("records"):
        kind = str(r.get("game_type") or "")
        if "showdown" in kind.lower() or "single game" in kind.lower():
            # Checked before the board is fetched, to save the call.
            log.info("draft group %s is %s - skipped", r["draft_group"], kind)
            continue
        try:
            b = D.board(int(r["draft_group"]))
        except D.Unavailable as exc:
            log.warning("draft group %s unreadable: %s", r["draft_group"],
                        str(exc)[:90])
            continue
        why = usable(r["draft_group"], b, kind)
        if why:
            log.info("draft group %s skipped: %s", r["draft_group"], why)
            continue
        out.append((int(r["draft_group"]), str(r["example"]), b,
                    str(r.get("starts_text") or "")))
    out.sort(key=lambda t: t[2]["game"].nunique(), reverse=True)
    return out


# --------------------------------------------------------------- projecting
def fill_positions(hist: pd.DataFrame) -> pd.DataFrame:
    """A missing position, filled from the SAME ATHLETE's other rows.

    CFBD's games/players endpoint carries no position; the roster endpoint
    does, and the join between them is by athlete id and lands about 98.9% of
    the time. The one to three percent it misses are not noise. `project`
    below drops unpositioned rows before fitting - correctly, because position
    is a feature - so those athletes get no feature row, no projection, and
    then drop out of the pool entirely, even though DraftKings prices them and
    publishes a position for them. A readiness probe against a real week found
    71 of 2,910 scorers unpositioned.

    THIS IS NOT NAME MATCHING. The probe's own note was that an unmatched
    scorer "needs a name-based fallback and that fallback needs measuring, not
    assuming" - and it is right, which is why this does something else: it
    looks the SAME ATHLETE ID up in his other rows, where a different season's
    roster did carry a position. Nothing is guessed. An athlete with no
    position in any row keeps none, and the count is logged either way.
    """
    if "position" not in hist.columns:
        return hist
    out = hist.copy()
    pos = out["position"].astype("object").where(
        out["position"].notna() & (out["position"].astype(str).str.strip() != ""))
    missing = pos.isna()
    if not missing.any():
        return out

    known = out.loc[~missing, ["athlete_id", "position"]]
    if not len(known):
        return out
    # His most common spelling, not his first: a roster that lists him as WR
    # in three seasons and ATH in one is a receiver.
    modal = (known.groupby("athlete_id")["position"]
             .agg(lambda s: s.value_counts().idxmax()))
    filled = out.loc[missing, "athlete_id"].map(modal)
    got = filled.notna()
    if got.any():
        out.loc[missing & got.reindex(out.index, fill_value=False),
                "position"] = filled[got]
        log.info("recovered a position for %d row(s) across %d athlete(s) "
                 "from their OWN other seasons - these would otherwise have "
                 "been dropped from the fit and then from the board",
                 int(got.sum()), int(filled[got].groupby(
                     out.loc[missing & got.reindex(out.index, fill_value=False),
                             "athlete_id"]).size().shape[0]))
    still = int((missing & ~got.reindex(out.index, fill_value=False)).sum())
    if still:
        log.info("%d row(s) still carry no position in any season and stay "
                 "out of the fit", still)
    return out


PARTIAL_WEEK_SHARE = 0.6


def newest_week_is_partial(this_year: pd.DataFrame, newest: int) -> bool:
    """Is the newest cached week complete, or only half in?

    MEASURED AGAINST THE WEEKS BEHIND IT, rather than assumed either way. On a
    Saturday morning the current week holds only its midweek card: a readiness
    probe against a live season found week 5 with 9 of its 304 games played
    while weeks 1-4 were complete. CFBD also takes a day or two to ingest a
    week. Those games are real and worth fitting on - what must not happen is
    the page claiming a complete week it does not have.

    Counted in GAMES, not in rows. A week with few games has few rows for the
    ordinary reason; a week where four hundred athletes have rows from six
    games is exactly the state being detected.
    """
    if not newest or not len(this_year) or "game_id" not in this_year.columns:
        return False
    weeks = pd.to_numeric(this_year["week"], errors="coerce")
    per_week = this_year.groupby(weeks)["game_id"].nunique()
    older = per_week[per_week.index < newest]
    if not len(older):
        return False
    have = int(per_week.get(newest, 0))
    typical = float(older.median())
    if typical <= 0 or have >= PARTIAL_WEEK_SHARE * typical:
        return False
    log.warning("week %d holds %d game(s) against a median of %d in the weeks "
                "before it, so it is PARTLY ingested - the midweek card is in "
                "and the weekend's is not. That is normal, it is fitted on, "
                "and the page says so.", newest, have, int(typical))
    return True


def project(hist: pd.DataFrame) -> pd.DataFrame:
    """Fit the sport and project every athlete's next game.

    `latest_rows` is the last feature row per athlete, which describes games
    already played. That is the point: Saturday has not happened, so the
    projection is made from what he has done and never from a row about the
    game being predicted.
    """
    before = len(hist)
    hist = fill_positions(hist)
    hist = hist[hist["position"].notna() & (hist["position"].astype(str) != "")]
    log.info("fitting on %d of %d player-games (%d dropped for having no "
             "position - needed to fit, not needed to join)",
             len(hist), before, before - len(hist))

    built = S.build(hist)
    fitted = EM.Projections(S.SPEC).fit(built)
    latest = EM.latest_rows(built)
    proj = fitted.predict(latest)
    for c in ("player_id", "name", "team", "position", "season", "period"):
        if c in latest.columns and c not in proj.columns:
            proj[c] = latest[c].to_numpy()
    proj["player_id"] = proj["player_id"].astype(str)

    # How many games each athlete's projection is actually standing on, and
    # when he was last seen. A projection built on two games from 2024 is a
    # real number with no evidence behind it, and the page should be able to
    # say so on the row rather than leaving it to look like the others.
    seen = (built.sort_values(["season", "period"])
            .groupby("player_id")
            .agg(games_seen=("points", "size"),
                 last_season=("season", "max")))
    seen.index = seen.index.astype(str)
    proj = proj.merge(seen, left_on="player_id", right_index=True, how="left")
    log.info("projected %d athletes, fitted on %d rows",
             len(proj), getattr(fitted, "trained_rows", len(built)))
    return proj


def join_board(board: pd.DataFrame, proj: pd.DataFrame,
               hist: pd.DataFrame, team_map: dict) -> pd.DataFrame:
    """The slate's players with their projections, joined on ATHLETE ID.

    Two steps, and they fail in different ways, so they are reported
    separately. `attach_history` turns a DraftKings spelling into a CFBD
    athlete id and can leave a player unmatched. The merge then turns that id
    into a projection and can leave him unprojected - a different thing,
    meaning he was found in the history but had too little of it to model.
    """
    joined = D.attach_history(board, hist, team_map=team_map)
    log.info("%s", D.join_quality(joined))

    left = joined.copy()
    left["athlete_id"] = left["athlete_id"].astype("object")

    # TWO PRICED PLAYERS RESOLVED TO ONE ATHLETE IS NOT A JOIN, IT IS A
    # COLLAPSE, AND IT WAS INVISIBLE.
    #
    # `attach_history`'s strict pass builds a global "name key -> first
    # athlete" index with no school restriction, so two priced players who
    # share a name key both come back with the SAME athlete id. On a college
    # board that is not exotic: there were two Michael Smiths, one at Alabama
    # priced $7,000 and one at Auburn priced $3,000, and the Auburn man
    # received the Alabama man's 12.0 median - a four-times value play that is
    # the wrong human being.
    #
    # The guard that was here checked for a FAN-OUT, which the
    # `drop_duplicates` below makes impossible: the merge is many-to-one by
    # construction, so the row count can never change and the check was dead
    # code. The risk is on the left, and it is checked on the left.
    #
    # Both players lose the projection rather than one of them keeping it. We
    # do not know which man the history belongs to, and a wrong projection
    # attached to a real salary is worse than no projection: an unprojected
    # player is dropped from the pool, which is the honest outcome.
    ids = left["athlete_id"]
    dupe = ids.notna() & ids.duplicated(keep=False)
    if dupe.any():
        groups = left[dupe].groupby(ids[dupe])
        log.error("%d priced player(s) resolved to only %d athlete id(s) - "
                  "two different people matched the same history. ALL of them "
                  "lose their projection and drop out of the pool, because "
                  "guessing which man it is attaches a real salary to the "
                  "wrong production: %s", int(dupe.sum()), groups.ngroups,
                  "; ".join(
                      f"{aid} <- " + ", ".join(
                          f"{r['name']} ({r['team']}, ${int(r['salary']):,})"
                          for _, r in g.iterrows())
                      for aid, g in list(groups)[:6]))
        left.loc[dupe, "athlete_id"] = None

    keep = [c for c in proj.columns if c not in left.columns
            or c == "player_id"]
    dropped = [c for c in proj.columns if c not in keep]
    # A projection column the board also names is silently discarded by the
    # line above, and the three it discards today - name, team, position - are
    # exactly the ones that should be. There is no guard, though, so a future
    # DraftKings field called `median` or `p_play` would replace the
    # projection with DraftKings' own number and nothing downstream would
    # notice: `merged["median"].notna()` would still pass. Named columns are
    # checked explicitly.
    fatal = [c for c in ("median", "ceiling", "mean", "cond_mean", "p_play")
             if c in dropped]
    if fatal:
        raise RuntimeError(
            f"the DraftKings board now carries a column named {fatal}, which "
            f"collides with the projection's own. The board wins every "
            f"collision, so the projection would be silently replaced by "
            f"DraftKings' number - which still passes every downstream "
            f"not-null check. Rename one side before publishing.")
    log.info("the board already names %d projection column(s) and keeps its "
             "own: %s", len(dropped), ", ".join(sorted(dropped)) or "none")
    right = proj[keep].drop_duplicates("player_id", keep="first")

    before = len(left)
    out = left.merge(right, left_on="athlete_id", right_on="player_id",
                     how="left")
    if len(out) != before:
        raise RuntimeError(f"the projection join changed the row count from "
                           f"{before} to {len(out)}, which should be "
                           f"impossible after the dedup above")

    have = out["median"].notna() if "median" in out.columns else pd.Series(
        False, index=out.index)
    salary = pd.to_numeric(out["salary"], errors="coerce").fillna(0)
    ppg = pd.to_numeric(out.get("dk_points_per_game"),
                        errors="coerce").fillna(0)
    log.info("projected %d of %d priced players (%.0f%%), %.0f%% of slate "
             "salary, %.0f%% of the production DraftKings has published",
             int(have.sum()), len(out), 100 * have.mean(),
             100 * salary[have].sum() / max(salary.sum(), 1),
             100 * ppg[have].sum() / max(ppg.sum(), 1))

    # The misses that matter, which is NOT the misses that are expensive.
    # DraftKings prices a true freshman third-string quarterback at $4,500
    # precisely BECAUSE it has no data on him, so a dear miss is not evidence
    # of anything by itself. A miss with positive published points per game
    # is: DraftKings found his production and this did not.
    real = out[~have & (ppg > 0)].sort_values("dk_points_per_game",
                                              ascending=False)
    if len(real):
        log.warning("%d player(s) with published production carry no "
                    "projection and are dropped from the pool rather than "
                    "guessed at. The biggest are: %s", len(real),
                    ", ".join(f"{r['name']} ({r['team']}, "
                              f"{float(r['dk_points_per_game']):.1f} ppg)"
                              for _, r in real.head(8).iterrows()))
    return out


# ---------------------------------------------------------------- the lineups
def server_lineups(pool: pd.DataFrame, draws: np.ndarray,
                   own: pd.Series, field_size: int) -> dict:
    """The exact integer program's answer, checked against the site's rules.

    `check_entry` is not decoration. The solver understands the cap, the
    slots, a per-position ceiling and a per-game ceiling; it does not
    understand that "players from at least two games" is a MINIMUM, and a
    minimum cannot be written as a maximum. The maximum encoding happens to be
    exact on an eight-man roster - but exact by arithmetic coincidence is not
    the same as checked, and a lineup that breaks the rule looks perfectly
    legal right up to the moment the entry is refused.
    """
    out = {}
    for objective in ("cash", "gpp"):
        try:
            built = O.build(pool, ROSTER, draws, objective=objective,
                            entries=1, own=own, field_size=field_size)
        except Exception as exc:                               # noqa: BLE001
            log.error("the %s integer program did not solve (%s: %s)",
                      objective, type(exc).__name__, str(exc)[:140])
            continue
        if not len(built):
            continue

        problems = S.check_entry(built, ROSTER)
        if problems:
            log.error("the %s lineup CANNOT BE ENTERED: %s", objective,
                      "; ".join(problems))
        qb = built[built["position"].astype(str) == "QB"]
        stack = 0
        if len(qb):
            team = str(qb.iloc[0]["team"])
            stack = int((built["team"].astype(str) == team).sum()) - 1
        out[objective] = {
            "players": [str(n) for n in built["name"]],
            "salary": int(pd.to_numeric(
                built.get("charged", built["salary"])).sum()),
            "ceiling": round(float(pd.to_numeric(
                built["ceiling"], errors="coerce").sum()), 1),
            "qb_stack": stack,
            "illegal": problems,
        }
    return out


# ------------------------------------------------------------------ payload
def num(v, places: int = 2, default=None):
    """A JSON-safe number, or `default`.

    NaN IS NOT VALID JSON. Python's json.dumps writes a bare `NaN` quite
    happily; JavaScript's JSON.parse throws a SyntaxError on it. The fetch
    resolves, the parse dies, nothing catches it, and the page sits on
    "Loading data/..." for ever with no error anywhere on screen.

    On a board where some teams have a market line and some do not, this is
    guaranteed rather than hypothetical.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f or f in (float("inf"), float("-inf")):
        return default
    return round(f, places)


def _code(v) -> str:
    """A team code, or "?" - and NaN counts as missing.

    `str(v or "?")` looks like it does this and does not: `float("nan")` is
    TRUTHY, so a missing opponent came through as the string "nan". That is
    not merely ugly on screen - "ALA v nan" is a distinct label, so two
    players in the same game with a missing opponent counted as two different
    games and the max-per-game rule stopped meaning anything.
    """
    if v is None or (isinstance(v, float) and v != v) or pd.isna(v):
        return "?"
    s = str(v).strip()
    return s if s and s.lower() != "nan" else "?"


def game_label(team, opponent) -> str:
    """One label per game, whichever side asks for it.

    CANONICAL, so one fixture produces ONE label. Built directionally this was
    the bug that listed a hockey game twice - a Vancouver player gave
    "EDM v VAN" and an Edmonton player "VAN v EDM" - and counted it twice.
    Sorting the pair makes the label a property of the GAME rather than of
    whichever side you happened to be looking from.
    """
    return " v ".join(sorted([_code(team), _code(opponent)]))


def slate_games(board: pd.DataFrame, pool: pd.DataFrame,
                lines: pd.DataFrame | None) -> list[dict]:
    """Every game on the draft group, and whether anyone in it is projected.

    Derived from the BOARD, not the pool. Built from the pool, a game whose
    players all failed to project simply vanishes - which is how a sixteen-
    team hockey slate came out looking like a five-game one, with nothing on
    screen to say that ten clubs had been dropped on the way through.
    """
    have = pool.groupby(pool["team"].astype(str)).size().to_dict() if len(pool) else {}
    totals = {}
    if lines is not None and len(lines):
        totals = dict(zip(lines["team"].astype(str),
                          pd.to_numeric(lines["game_total"],
                                        errors="coerce")))
    seen, out = set(), []
    for _, r in board.iterrows():
        t, o = _code(r["team"]), _code(r.get("opponent"))
        key = game_label(t, o)
        if key in seen:
            continue
        seen.add(key)
        out.append({"label": key, "teams": sorted([t, o]),
                    "players": int(have.get(t, 0)) + int(have.get(o, 0)),
                    "total": num(totals.get(t, totals.get(o)), 1)})
    return sorted(out, key=lambda g: g["label"])


def slate_players(pool: pd.DataFrame, quantiles: list[float],
                  own: pd.Series, lev: pd.Series) -> list[dict]:
    qcols = [f"q{int(round(q * 100))}" for q in quantiles]
    rows = []
    for i, (_, r) in enumerate(pool.iterrows()):
        rows.append({
            "name": str(r["name"]),
            "pos": str(r["position"]),
            "team": _code(r["team"]),
            "opp": _code(r.get("opponent")),
            # DraftKings' own player id, shipped so the page can tell one man
            # listed at two positions from two different men. Keyed on
            # name-plus-position, the browser's duplicate check cannot: it
            # rostered the same player twice off a board that priced him at
            # 1B and at 3B, which DraftKings refuses outright.
            "pid": (None if pd.isna(r.get("dk_player_id"))
                    else str(r.get("dk_player_id"))),
            "game": game_label(r["team"], r.get("opponent")),
            "salary": int(r["salary"]),
            # The quantile curve must never carry a hole - the simulator reads
            # it directly - so a missing one falls back to the median rather
            # than to null.
            "q": [num(r[c], 3, num(r["median"], 3, 0.0)) for c in qcols],
            "med": num(r["median"], 2, 0.0),
            "ceil": num(r["ceiling"], 2, 0.0),
            "own": num(own.iloc[i], 5, 0.0),
            "lev": num(lev.iloc[i], 3, 0.0),
            "pp": num(r.get("p_play", 1.0), 4, 1.0),
            # Football's opportunity number is the game, the way hockey's is
            # ice time and baseball's is the batting slot. Shipped so the
            # board can be sorted by it, and null where no line was found -
            # which the page can render and a zero cannot.
            "itt": num(r.get("implied_total"), 1),
            "gt": num(r.get("game_total"), 1),
            "spr": num(r.get("team_spread"), 1),
            "mkt": num(r.get("market_factor"), 3),
            # How much evidence is under the projection, and how old it is.
            # A number built on two games from last season is a real number
            # with nothing behind it, and in a sport with a transfer portal
            # that is a routine case rather than an edge one.
            "gp": int(r["games_seen"]) if pd.notna(r.get("games_seen")) else 0,
            "yr": int(r["last_season"]) if pd.notna(r.get("last_season")) else None,
            "ppg": num(r.get("dk_points_per_game"), 1),
            "doubtful": bool(r.get("doubtful", False)),
            "status": "clear",
        })
    return rows


def thin_evidence(pool: pd.DataFrame, season: int) -> pd.DataFrame:
    """Mark the projections that are standing on last season's player.

    Not a filter. A back with four games this year and a new role is still
    worth projecting, and the page says so on his row rather than deciding
    for you. The reason this is flagged at all rather than ignored is the
    portal: a projection keyed on an athlete id follows the athlete to his new
    school correctly, and knows nothing whatever about his new role.
    """
    out = pool.copy()
    if "last_season" not in out.columns:
        out["doubtful"] = False
        return out
    stale = pd.to_numeric(out["last_season"], errors="coerce") < season
    thin = pd.to_numeric(out.get("games_seen"), errors="coerce").fillna(0) < 2
    out["doubtful"] = (stale | thin).fillna(False)
    n = int(out["doubtful"].sum())
    if n:
        log.info("%d of %d pooled players are flagged: no %d games yet, or "
                 "fewer than two games of history anywhere", n, len(out),
                 season)
    return out


def caveats_for(has_market: bool, fitted_week: int, season: int,
                thin: int, total: int, partial: bool = False) -> list[str]:
    out = [
        "CFBD publishes receptions, not targets. Target share is what "
        "separates a receiver who IS the offence from one who plays in a good "
        "one, and it does so before the catches arrive - so this model is "
        "weaker at receiver than an NFL one, and no amount of tuning fixes a "
        "column the feed does not have.",
        "There is no college injury report and no depth chart here. "
        "DraftKings drops the players it knows are out; everyone else is "
        "priced as available. What partly replaces it is the blowout feature "
        "- first-stringers sit in the fourth quarter of a forty-point win, "
        "which is frequent and learnable.",
        "Ownership is modelled, not observed, and has never been graded "
        "against a real college field. Quarterback demand is set at "
        f"{S.QB_SUPERFLEX_DEMAND:.2f} per lineup from published winning-lineup "
        "composition rather than from field data, so leverage on quarterbacks "
        "is the least trustworthy number on this page.",
    ]
    if has_market:
        out.insert(0, (
            "Market totals are live and are applied as the DISAGREEMENT "
            "between the market and this model, not as a raw multiplier - the "
            "projection already knows which offences are good, so scaling by "
            "the implied total on top of that would count it twice and "
            "exaggerate the chalk."))
    else:
        out.insert(0, (
            "NO MARKET LINES WERE AVAILABLE for this board. Without them "
            "'which team to stack' collapses into 'which team has the best "
            "players', which is the same team every week. Treat the stack "
            "board as a projection ranking only."))
    if total and thin / max(total, 1) > 0.25:
        out.insert(0, (
            f"{thin} of {total} players in this pool ({100 * thin / total:.0f}%) "
            f"have no {season} games yet or fewer than two games anywhere. "
            f"Early in a season, and after a transfer, the projection is "
            f"last year's player."))
    # The week the CACHE reaches, not the week that has been played. The first
    # version printed `live_week`, which is the newest week with a completed
    # game - so on a board whose history was a week behind the games, the page
    # claimed a fit it had not made, three lines below the code that logs a
    # warning about exactly that gap.
    if fitted_week and partial:
        # The newest week in the cache is PARTLY ingested, which is the normal
        # state on a Saturday morning: a readiness probe against a live season
        # found week 5 holding 9 of its 304 games - the Thursday card - while
        # the Saturday card had not been played. Those nine games are real and
        # worth having; what would be wrong is implying the week is complete.
        out.append(f"Fitted on history through week {fitted_week} of {season}, "
                   f"and week {fitted_week} is only PARTLY in - the midweek "
                   f"games are counted and the weekend's are not. CFBD "
                   f"ingests a week over a day or two, so the newest week is "
                   f"always thinner than the ones behind it.")
    elif fitted_week:
        out.append(f"Fitted on history through week {fitted_week} of "
                   f"{season}.")
    else:
        out.append(f"THERE IS NO {season} HISTORY IN THIS BUILD AT ALL. Every "
                   f"projection on this board is last season's player, which "
                   f"in a sport with a transfer portal is a serious statement "
                   f"rather than a caveat.")
    return out


def loadings_for_page() -> dict:
    return {kind: dict(S.SPEC.loadings.get(kind, {}))
            for kind in ("game", "team", "compete")}


def slate_label(board: pd.DataFrame, pool: pd.DataFrame, raw: str) -> str:
    bits = [f"{board['game'].nunique()} games",
            f"{pool['team'].nunique()} teams"]
    if raw:
        bits.append(raw.strip())
    return " · ".join(bits)


# ------------------------------------------------------------------- output
def write(payloads: list[dict]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    archive = DATA / "archive"

    keep, slates = set(), []
    for p in payloads:
        name = f"{SPORT}_dk_{p['kind']}_{p['draft_group']}.json"
        # allow_nan=False, and that is the whole point of writing it out here.
        #
        # By default Python emits a bare `NaN`, which is not valid JSON and
        # which JavaScript's JSON.parse refuses. The file writes, the commit
        # succeeds, the run goes green, and the page sits on "Loading ..." for
        # ever with nothing on screen to say why. Refusing here turns a silent
        # dead page into a loud build failure.
        try:
            body = json.dumps(p, indent=1, allow_nan=False)
        except ValueError as exc:
            log.error("draft group %s produced a payload that is not valid "
                      "JSON (%s). It is NOT being written - a file the page "
                      "cannot parse is worse than no file, because the page "
                      "gives no sign of why it is stuck.",
                      p["draft_group"], exc)
            continue
        (DATA / name).write_text(body)
        keep.add(name)
        slates.append({"sport": SPORT, "site": "dk", "kind": p["kind"],
                       "label": p["label"], "file": f"data/{name}",
                       "locks": p["locks"]})
        log.info("wrote %s (%d players)", name, len(p["players"]))

    if not keep:
        sys.exit("every payload failed to serialise, so nothing is published "
                 "and the page keeps what it had.")

    # Archived, never deleted. A board that is gone cannot be graded against a
    # finished contest, and grading published boards against real results is
    # the only feedback loop this project has.
    for old in DATA.glob(f"{SPORT}_dk_*.json"):
        if old.name in keep:
            continue
        try:
            when = json.loads(old.read_text()).get("generated_at", "")[:10]
        except Exception:                                      # noqa: BLE001
            when = ""
        when = when or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        dest = archive / when
        dest.mkdir(parents=True, exist_ok=True)
        old.replace(dest / old.name)
        log.info("archived %s -> archive/%s/", old.name, when)

    # THE MANIFEST IS WRITTEN BEFORE THE HOUSEKEEPING, NOT AFTER.
    #
    # It used to be last, after an expiry sweep that ended in `day.rmdir()` -
    # and `rmdir` raises on a directory holding anything that is not a
    # `*.json`. One stray file in an expired archive day therefore wrote the
    # new slates, MOVED the old ones, and then died before updating the
    # manifest, leaving a manifest that pointed at files which had just been
    # relocated. The publish step commits `docs/` whatever happened, so that
    # half-updated state would have shipped.
    #
    # Rebuilt from every slate file present, not just the ones this run
    # produced - otherwise publishing football would hide hockey.
    listed = []
    for f in sorted(DATA.glob("*_dk_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:                                      # noqa: BLE001
            continue
        listed.append({"sport": d.get("sport", "?"),
                       "site": d.get("site", "dk"),
                       "kind": d.get("kind", "classic"),
                       "label": d.get("label", ""),
                       "file": f"data/{f.name}",
                       "locks": d.get("locks")})
    (DATA / "manifest.json").write_text(json.dumps({
        "updated_at": payloads[0]["generated_at"],
        "slates": listed,
    }, indent=1))
    log.info("manifest lists %d slate(s) across %d sport(s)", len(listed),
             len({s["sport"] for s in listed}))

    # Housekeeping last, and it cannot take the run down. An archive that
    # fails to expire is a slowly growing directory; a publish that fails
    # because of one is a board nobody can use tonight.
    cutoff = datetime.now(timezone.utc) - timedelta(days=ARCHIVE_DAYS)
    for day in sorted(archive.glob("[0-9]" * 4 + "-*")):
        try:
            on = datetime.strptime(day.name, "%Y-%m-%d").replace(
                tzinfo=timezone.utc)
        except ValueError:
            continue
        if on >= cutoff:
            continue
        try:
            for f in day.glob("*.json"):
                f.unlink()
            # Only if it is now genuinely empty. Anything else in there was
            # put there by a person and is not this function's to delete.
            if not any(day.iterdir()):
                day.rmdir()
            else:
                log.info("archive/%s still holds files that are not slates, "
                         "so the directory is left alone", day.name)
        except OSError as exc:
            log.warning("could not expire archive/%s (%s) - the slate is "
                        "published either way", day.name, exc)


# --------------------------------------------------------------------- main
def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draft-group", type=int, default=None)
    p.add_argument("--season", type=int, default=None,
                   help="defaults to the current football season, which is "
                        "named for the year it starts")
    p.add_argument("--first-season", type=int, default=2021)
    p.add_argument("--field", type=int, default=100_000)
    p.add_argument("--sims", type=int, default=20_000)
    p.add_argument("--slates", type=int, default=4)
    p.add_argument("--look", type=int, default=12)
    p.add_argument("--min-salary-pct", type=float, default=99.0,
                   help="the least a LINEUP may spend, %% of the $50k cap. "
                        "Unspent salary is points declined; 0 turns it off.")
    p.add_argument("--market", default="on", choices=("on", "off"),
                   help="read live NCAAF totals and spreads and let them move "
                        "the projections. ON by default: without the market, "
                        "'which team to stack' is just 'which team has the "
                        "best players', and that is the same team every week.")
    args = p.parse_args(argv)

    pct = max(0.0, min(100.0, float(args.min_salary_pct))) / 100.0
    ROSTER["min_salary_pct"] = pct
    log.info("roster %s, cap $%s, lineup salary floor $%s, quarterbacks "
             "capped at %s, at least %s games",
             "/".join(ROSTER["slots"]), f"{ROSTER['salary_cap']:,}",
             f"{int(pct * ROSTER['salary_cap']):,}",
             ROSTER["max_position"]["QB"], ROSTER["min_games"])
    log.info("quarterback demand for ownership is %.2f per lineup (see "
             "cfb_sport.QB_SUPERFLEX_DEMAND - it is stated, not measured)",
             S.QB_SUPERFLEX_DEMAND)

    key = os.environ.get("CFBD_API_KEY", "").strip()
    if not key:
        sys.exit("CFBD_API_KEY is not set. Add it under Settings -> Secrets "
                 "and variables -> Actions and pass it into the job's env. It "
                 "is needed for the week number, the fixtures and the team "
                 "map - not for the history, which comes from data/.")

    now = datetime.now(timezone.utc)
    # A football season is named for the year it STARTS, and the postseason
    # runs into January. Before August, the live season is last year's.
    season = args.season or (now.year if now.month >= 8 else now.year - 1)

    # HISTORY FROM THE CACHE, NEVER THE API.
    #
    # Re-downloading the archive on every run is what exhausted the monthly
    # quota mid-grade once already, leaving 2024 and 2025 unfetched and the
    # walk-forward with nothing to walk over. Keeping the current season fresh
    # is the FETCH workflow's job, on its own schedule. Publishing reads what
    # is there and says how old it is.
    seasons = [s for s in range(args.first_season, season + 1)
               if s != D.COVID_SEASON]
    cached = C.cached_seasons()
    missing = [s for s in seasons if s not in cached]
    if missing:
        log.warning("seasons %s are not in data/ and are simply absent from "
                    "the fit. Run the fetch workflow for them.", missing)
    usable = [s for s in seasons if s in cached]
    # Checked BEFORE the call. `C.load([])` raises FileNotFoundError, whose
    # message is about a cache that needs fetching - true, but it buries the
    # thing that is actually wrong when the requested range and the cached
    # range simply do not overlap.
    if not usable:
        sys.exit(f"data/ holds {cached or 'nothing'} and this run asked for "
                 f"{seasons}, so there is no history to fit on. Run the CFB "
                 f"fetch workflow, or widen --first-season.")
    hist = C.load(usable)
    if not len(hist):
        sys.exit("every cached season read back empty, so there is nothing to "
                 "fit. The files in data/ are the thing to look at.")

    have_seasons = sorted(int(s) for s in hist["season"].unique())
    log.info("history: %d player-games, %d athletes, seasons %s",
             len(hist), hist["athlete_id"].nunique(), have_seasons)
    this_year = hist[hist["season"].astype(int) == season]
    weeks_cached = int(this_year["week"].astype(int).max()) if len(this_year) else 0
    if not len(this_year):
        log.error("THE CACHE HOLDS NO %d GAMES AT ALL. Every projection on "
                  "this board is therefore last season's player, which in a "
                  "sport with a transfer portal is a serious statement rather "
                  "than a caveat. If the fetch workflow is running and this "
                  "is still empty, the free CFBD tier is not returning "
                  "current-season player stats - run cfb_ready.py, which was "
                  "written to answer exactly that.", season)
    else:
        log.info("%d: %d player-games cached through week %d",
                 season, len(this_year), weeks_cached)

    partial_week = newest_week_is_partial(this_year, weeks_cached)

    try:
        week = D.live_week(key, season)
    except Exception as exc:                                   # noqa: BLE001
        week = weeks_cached
        log.warning("could not read the live week (%s: %s); falling back to "
                    "the newest week in the cache, %d", type(exc).__name__,
                    str(exc)[:90], week)
    if weeks_cached and week > weeks_cached:
        log.warning("week %d has been played but the cache stops at week %d. "
                    "This board is %d week(s) behind the games, which is "
                    "exactly the form the model is supposed to be reading. "
                    "Run the fetch workflow with --refresh %d.",
                    week, weeks_cached, week - weeks_cached, season)

    proj = project(hist)

    # The fixtures and the team list, fetched ONCE for every board. The teams
    # endpoint carries abbreviations and alternate names; without it the team
    # map resolves about a quarter of DraftKings' codes, because Georgia's
    # abbreviation IS "UGA" and no rule derives that from the string.
    #
    # THE WINDOW STARTS AT week - 1, AND THAT OFF-BY-ONE WAS A DEAD SATURDAY.
    #
    # `upcoming_games(after_week, span)` fetches weeks after_week+1 through
    # after_week+span, and `live_week` returns the newest week with ANY
    # completed game. College football plays a Thursday game in essentially
    # every week of the season, so by Friday evening - and certainly by
    # Saturday morning, which is when the main slate publishes - week N's
    # Thursday game has finished, `live_week` returns N, and a window of
    # N+1..N+3 does not contain the board being built.
    #
    # The consequence was total: the fixture list held next week's pairings,
    # so NONE of the board's team codes resolved, every draft group was
    # skipped, and the run exited "no board could be built". Reproduced end to
    # end. The test suite could not see it because it stubs both `live_week`
    # and `upcoming_games`.
    #
    # Starting one week earlier costs one extra API call and makes the window
    # weeks N-1..N+2, which contains this week whichever side of its Thursday
    # game the clock is on.
    try:
        games = D.upcoming_games(key, season, max(week - 1, 0), span=4)
        if not games:
            sys.exit(f"the schedule window around week {week} of {season} "
                     f"came back empty, so there are no fixtures to solve the "
                     f"team map against and nothing can be published safely.")
        teams = D.cfbd("teams", key, year=season)
    except SystemExit:
        raise
    except Exception as exc:                                   # noqa: BLE001
        sys.exit(f"the fixtures or team list could not be read ("
                 f"{type(exc).__name__}: {str(exc)[:140]}). Without them the "
                 f"team map cannot be solved, and a board whose team codes "
                 f"are guessed hands players the wrong opponent, the wrong "
                 f"implied total and the wrong correlation group - a full "
                 f"page of confident numbers that are wrong in every row.")

    # Every school CFBD knows about this season, for the odds matcher. The
    # `teams` endpoint is the whole division, which is exactly what that
    # matcher needs: given only a board's schools it resolves a feed name to
    # the nearest school that happens to be present, which is a wrong answer
    # wearing a resolved answer's clothes.
    all_schools = sorted({str(t.get("school")) for t in (teams or [])
                          if t.get("school")})
    log.info("%d schools in the %d CFBD team list, for matching market names",
             len(all_schools), season)
    if len(all_schools) < 100:
        log.warning("that is fewer schools than college football has at this "
                    "level. The market matcher needs the full list to tell "
                    "Michigan State from Michigan.")

    try:
        boards = candidate_slates(args.draft_group, args.look)
    except Exception as exc:                                   # noqa: BLE001
        # The DraftKings lobby, unlike everything else here, is not wrapped
        # inside the per-board handler - `candidate_slates` is what produces
        # the boards. An outage was arriving as a raw traceback rather than as
        # one of this file's own exit messages.
        sys.exit(f"the DraftKings lobby could not be read "
                 f"({type(exc).__name__}: {str(exc)[:160]}). Nothing is "
                 f"published and the page keeps what it had.")
    if not boards:
        sys.exit("no readable college football classic draft group is on sale "
                 "right now, so nothing is published and the page keeps what "
                 "it had.")

    quantiles = list(S.SPEC.quantiles)
    payloads = []
    for dg, label, board, starts_text in boards[:args.slates]:
        try:
            board = board.copy()
            board["position"] = board["position"].map(roster_position)
            unknown = int(board["position"].isna().sum())
            if unknown:
                log.info("draft group %s: dropped %d player(s) whose position "
                         "is not a slot on this roster (a college board "
                         "prices no tight ends, kickers or defences)",
                         dg, unknown)
            board = board[board["position"].notna()]

            mapping = D.fixture_team_map(board, games, teams)
            codes = sorted(set(board["team"].dropna().astype(str)))
            unsolved = [c for c in codes if c not in mapping]
            if unsolved:
                log.error("draft group %s: %d of %d team codes are UNSOLVED "
                          "(%s). Those players keep their DraftKings code and "
                          "lose their school-restricted name match and their "
                          "market line; they are not given a guessed school.",
                          dg, len(unsolved), len(codes), ", ".join(unsolved))
            if len(mapping) < 0.6 * max(len(codes), 1):
                log.error("draft group %s: only %d of %d team codes resolved, "
                          "which is too few to trust the join. Skipped.",
                          dg, len(mapping), len(codes))
                continue

            merged = join_board(board, proj, hist, mapping)
            pool = merged[merged["median"].notna()].copy()
            pool = pool[pd.to_numeric(pool["salary"],
                                      errors="coerce").notna()]
            pool["salary"] = pd.to_numeric(pool["salary"]).astype(int)
            pool["game"] = [game_label(t, o) for t, o in
                            zip(pool["team"], pool.get("opponent"))]

            if len(pool) < 60:
                log.error("draft group %s: only %d players survived the join, "
                          "which is too thin to build eight positions from. "
                          "Skipped.", dg, len(pool))
                continue

            # A COUNT IS NOT COVERAGE. On a hockey slate, 109 players out of
            # 413 cleared a forty-player bar comfortably and was still a
            # broken board: it was six clubs of sixteen, because the
            # projections were for a different night and only the overlapping
            # teams joined. The page then showed a sixteen-team slate as five
            # games with nothing on screen to say ten clubs had gone.
            #
            # Football's version of this is a team whose whole two-deep is
            # unmatched - a school with a new roster, or a code the map failed
            # on - and the symptom is identical.
            board_teams = set(board["team"].astype(str))
            per = pool.groupby(pool["team"].astype(str)).size()
            full = {t for t in board_teams if int(per.get(t, 0)) >= 6}
            share = len(full) / max(len(board_teams), 1)
            if share < 0.75:
                log.error("draft group %s: only %d of %d schools have a "
                          "usable set of projections (%.0f%%). Missing or "
                          "nearly empty: %s. Skipped rather than published "
                          "two thirds full.", dg, len(full), len(board_teams),
                          100 * share, ", ".join(sorted(board_teams - full)))
                continue
            if share < 1.0:
                log.warning("draft group %s: %s are thin on this board",
                            dg, ", ".join(sorted(board_teams - full)))

            # FLEX_SLOTS rather than a local tuple: a flex name this file
            # does not know about becomes a required position no player holds,
            # and the board is skipped for a rule that does not exist.
            for slot in sorted(set(ROSTER["slots"])):
                if slot in FLEX_SLOTS or slot == "CPT":
                    continue
                need = ROSTER["slots"].count(slot)
                got = int((pool["position"] == slot).sum())
                if got < need:
                    raise RuntimeError(
                        f"only {got} player(s) at {slot} and the roster needs "
                        f"{need}, so no legal lineup exists")

            if pool["game"].nunique() < MIN_GAMES_FOR_CLASSIC:
                log.error("draft group %s: the projected pool covers %d "
                          "game(s) and DraftKings requires %d. Skipped.",
                          dg, pool["game"].nunique(), MIN_GAMES_FOR_CLASSIC)
                continue

            lines = None
            if args.market == "on":
                try:
                    # EVERY school in the league, not the board's two dozen.
                    # See `cfb_odds.for_board` - handing it the short list is
                    # how Michigan State's game got published as Michigan's.
                    lines = CO.for_board(mapping, schools=all_schools)
                except Exception as exc:                       # noqa: BLE001
                    log.error("no market lines for draft group %s (%s: %s). "
                              "The board still publishes, and says on the "
                              "page that it is projection-only.",
                              dg, type(exc).__name__, str(exc)[:140])
            pool = apply_market(pool, lines, quantiles)
            pool = thin_evidence(pool, season)

            own = OWN.project(pool, ROSTER)
            lev = OWN.leverage(pool, own)
            draws = SIM.simulate(pool, quantiles, args.sims, spec=S.SPEC)

            payloads.append({
                "generated_at": now.isoformat(timespec="seconds"),
                "sport": SPORT,
                "site": "dk",
                "kind": "classic",
                "draft_group": dg,
                "label": slate_label(board, pool, label or starts_text),
                "locks": None,
                "field_size": args.field,
                "quantiles": quantiles,
                "loadings": loadings_for_page(),
                "roster": ROSTER,
                "players": slate_players(pool, quantiles, own, lev),
                "games": slate_games(board, pool, lines),
                "server_lineups": server_lineups(pool, draws, own, args.field),
                # The page prints these verbatim. A model that cannot see
                # targets or an injury report should say so where somebody
                # will read it, not in a log nobody opens.
                "caveats": caveats_for(lines is not None and len(lines) > 0,
                                       weeks_cached, season,
                                       int(pool["doubtful"].sum()), len(pool),
                                       partial=partial_week),
            })
        except Exception as exc:                               # noqa: BLE001
            # One bad board must not take the others down. The baseball build
            # once discarded fourteen good slates because a single dead one
            # raised past the handler.
            log.error("draft group %s failed (%s: %s) - the other boards are "
                      "unaffected", dg, type(exc).__name__, str(exc)[:200])
            continue

    if not payloads:
        sys.exit("no board could be built, so nothing is published and the "
                 "page keeps what it had.")
    write(payloads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
