"""Turn the stat projections into a DraftKings points distribution.

The stat model lives in its own repo (willyjo423/nhlprojections) and publishes
a dated CSV every morning: ice time, power-play time, goals, assists, shots,
blocks and the rest, per player, per slate. This module reads that file and
turns it into the frame the DFS publisher already consumes - `q10` through
`q97`, `median`, `ceiling`, `mean`, `p_play` - so nothing downstream changes.

Why simulate rather than project points directly
------------------------------------------------
DRAFTKINGS PAYS THRESHOLD BONUSES, AND A QUANTILE REGRESSION CANNOT SEE THEM.
Five shots on goal is worth +3, and four is worth nothing. Three blocks is
worth +3, and two is worth nothing. Those are step functions in a count, and
a model that fits quantiles of a continuous points total has no way to
represent a step - it will smear the jump across the distribution and
systematically misprice exactly the players the bonuses were designed to
reward: the high-volume shooter and the shot-blocking defenceman.

Simulating the counts and then applying the rules gets it exactly right. Draw
goals, assists, shots and blocks; check each threshold on the drawn line;
score it. The bonus either happened or it did not, in each of the fifty
thousand seasons, and the quantiles fall out of the result.

It also gets the within-player dependencies right for free. A goal IS a shot
on goal, so shots are drawn as goals plus the remainder rather than
independently - otherwise a two-goal game can come back with one shot, which
is not a thing that happens and which quietly understates every scorer's
ceiling.

What this does NOT do
---------------------
Cross-player correlation. That is the slate simulator's job, through the
factor model in `board_spec()`, and it works from these marginal curves.
Modelling it twice would double-count every line stack.
"""
from __future__ import annotations

import io
import logging
import datetime as dt

import numpy as np
import pandas as pd
import requests

import nhl_odds as NO
import nhl_sport as NS
from nhl_data import normalise

log = logging.getLogger("nhl_projections")

# The published stat model. Its Pages site is the contract; the repo is public
# and the files are dated, so this is a plain HTTPS GET with no key.
BASE = "https://willyjo423.github.io/nhlprojections/data"
TIMEOUT = 30

# How many days back to look if today's file is not there yet. A projection
# from yesterday is worth far more than no board at all - but the page is told
# how old it is, because a silently stale slate is the failure this project
# keeps paying for.
LOOK_BACK = 3

# Games in the last thirty days below which a player is treated as doubtful.
# DraftKings already drops anyone it knows is out; this catches the healthy
# scratch, the man on long-term injured reserve and the one who was sent down
# - none of which DraftKings always flags, and all of which look on a board
# exactly like a cheap player having a quiet month.
FRESH_GAMES = 3
# A man who has not dressed once in thirty days, while his league has been
# playing the whole time, is not a bargain. He is injured, in the minors, or
# out of the lineup, and the only honest number for him is a small one.
DOUBTFUL_P_PLAY = 0.10
MIN_P_PLAY = 0.60


class ProjectionsUnavailable(RuntimeError):
    """No published stat projection could be read for this slate."""


# ------------------------------------------------------------------ loading
def _get_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=TIMEOUT, headers={
        "User-Agent": "Mozilla/5.0 (compatible; dfs-model/1.0)"})
    if r.status_code != 200:
        raise ProjectionsUnavailable(f"{url} -> HTTP {r.status_code}")
    df = pd.read_csv(io.BytesIO(r.content))
    if not len(df):
        raise ProjectionsUnavailable(f"{url} -> empty")
    return df


def load(day: str, base: str = BASE,
         look_back: int = LOOK_BACK) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Tonight's published stat projections, or the freshest there are."""
    want = dt.date.fromisoformat(str(day)[:10])
    tried = []
    for back in range(look_back + 1):
        d = (want - dt.timedelta(days=back)).isoformat()
        try:
            sk = _get_csv(f"{base}/{d}_skaters.csv")
            go = _get_csv(f"{base}/{d}_goalies.csv")
        except Exception as exc:                               # noqa: BLE001
            tried.append(f"{d}: {str(exc)[:70]}")
            continue
        meta = {"date": d, "days_stale": back, "skaters": len(sk),
                "goalies": len(go)}
        if back:
            # Said loudly. Yesterday's ice time is a reasonable guess at
            # tonight's; yesterday's OPPONENT is not, and every opponent
            # adjustment in that file is for the wrong team.
            log.error("no stat projections for %s - falling back to %s, which "
                      "is %d day(s) old. The ice-time estimates are still "
                      "roughly right; the OPPONENT ADJUSTMENTS IN IT ARE FOR "
                      "THE WRONG GAMES.", want, d, back)
        else:
            log.info("stat projections for %s: %d skaters, %d goalies",
                     d, len(sk), len(go))
        return sk, go, meta
    raise ProjectionsUnavailable(
        "no published stat projections could be read:\n  " + "\n  ".join(tried))


# ------------------------------------------------------------- the scoring
def _bonus(counts: dict, table: dict, key: str, stat: str) -> np.ndarray:
    threshold, points = table[key]
    return np.where(counts[stat] >= threshold, points, 0.0)


def score_skater_lines(goals, assists, sog, blocks, shp=None) -> np.ndarray:
    """DraftKings points for drawn stat lines. Vectorised over draws.

    Kept as its own function so it can be checked against hand-computed
    lines, which is the only way to know a scoring table is right.
    """
    s = NS.DK_SKATER_SCORING
    b = NS.DK_SKATER_BONUSES
    pts = (s["goal"] * goals + s["assist"] * assists
           + s["shot_on_goal"] * sog + s["blocked_shot"] * blocks)
    if shp is not None:
        pts = pts + s["short_handed_point"] * shp
    counts = {"goals": goals, "assists": assists, "sog": sog,
              "blocks": blocks, "points": goals + assists}
    pts = pts + _bonus(counts, b, "shots_on_goal", "sog")
    pts = pts + _bonus(counts, b, "blocked_shots", "blocks")
    pts = pts + _bonus(counts, b, "points", "points")
    pts = pts + _bonus(counts, b, "goals", "goals")
    return pts


def score_goalie_lines(saves, goals_against, win, otl, shutout) -> np.ndarray:
    s = NS.DK_GOALIE_SCORING
    threshold, bonus = NS.DK_GOALIE_BONUSES["saves"]
    return (s["save"] * saves
            + s["goal_against"] * goals_against
            + s["win"] * win
            + s["overtime_loss"] * otl
            + s["shutout"] * shutout
            + np.where(saves >= threshold, bonus, 0.0))


# ----------------------------------------------------------- the simulation
def simulate_skaters(sk: pd.DataFrame, sims: int,
                     rng: np.random.Generator) -> np.ndarray:
    """One row per player, `sims` columns of DraftKings points."""
    n = len(sk)
    g_rate = _col(sk, "goals")
    a_rate = _col(sk, "assists")
    s_rate = _col(sk, "sog")
    b_rate = _col(sk, "blocks")
    pk = _col(sk, "pk_toi")

    goals = rng.poisson(np.tile(g_rate[:, None], (1, sims)))
    assists = rng.poisson(np.tile(a_rate[:, None], (1, sims)))
    blocks = rng.poisson(np.tile(b_rate[:, None], (1, sims)))

    # A GOAL IS A SHOT ON GOAL. Drawn independently, a two-goal game can come
    # back with one shot - impossible, and it quietly clips the top of every
    # scorer's distribution, which is the part a tournament is bought for. So
    # shots are goals plus a remainder.
    extra = np.maximum(s_rate - g_rate, 0.0)
    sog = goals + rng.poisson(np.tile(extra[:, None], (1, sims)))

    # Short-handed points, worth 2 apiece on top of the goal or assist. Rare,
    # and proportional to penalty-kill ice time. A crude rate rather than a
    # modelled one, and small enough that being crude is honest.
    shp = None
    if np.nanmax(pk) > 0:
        sh_rate = np.clip(pk, 0, None) / 60.0 * 0.35
        shp = rng.poisson(np.tile(sh_rate[:, None], (1, sims)))

    return score_skater_lines(goals, assists, sog, blocks, shp)


def win_probability(lam_for: np.ndarray, lam_against: np.ndarray,
                    sims: int, rng: np.random.Generator):
    """P(win) and P(overtime loss), from two Poisson goal counts.

    Derived rather than looked up: the published file carries no win
    probability, but it carries every skater's projected goals, so a team's
    expected goals is the sum over its own skaters and the goalie's expected
    goals against is already in his row. A tie at the end of regulation is
    settled in overtime or a shootout, which is close enough to a coin toss
    to model as one - and the loser of it collects DraftKings' overtime-loss
    points rather than nothing, which is the whole reason to bother.
    """
    f = rng.poisson(np.tile(lam_for[:, None], (1, sims)))
    a = rng.poisson(np.tile(lam_against[:, None], (1, sims)))
    reg_win = f > a
    tie = f == a
    coin = rng.random((len(lam_for), sims)) < 0.5
    win = reg_win | (tie & coin)
    otl = tie & ~coin
    return win, otl


def simulate_goalies(go: pd.DataFrame, team_goals: dict, sims: int,
                     rng: np.random.Generator) -> np.ndarray:
    shots = _col(go, "shots_against", default=29.0)
    sv = np.clip(_col(go, "save_pct", default=0.905), 0.80, 0.97)

    faced = rng.poisson(np.tile(shots[:, None], (1, sims)))
    # Saves are BINOMIAL on the shots actually faced, not an independent
    # count. Drawn apart, a goalie can be credited with more saves than shots.
    saves = rng.binomial(faced, np.tile(sv[:, None], (1, sims)))
    against = faced - saves

    lam_for = np.array([team_goals.get(t, 2.9) for t in go["team"]], dtype=float)
    win, otl = win_probability(lam_for, _col(go, "goals_against", default=2.9),
                               sims, rng)
    # A shutout is nought against over a full game. Nothing here knows whether
    # he is pulled, so this is the ceiling of a shutout chance, not a promise.
    shutout = (against == 0)
    return score_goalie_lines(saves, against, win.astype(float),
                              otl.astype(float), shutout.astype(float))


def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name not in df.columns:
        log.warning("the published projections carry no '%s' column, so it is "
                    "treated as %.2f for every player", name, default)
        return np.full(len(df), default, dtype=float)
    v = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
    return np.nan_to_num(v, nan=default, posinf=default, neginf=default)


# ---------------------------------------------------------------- assembly
def availability(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The chance he dresses, and whether he is doubtful enough to hide.

    Nothing published knows tonight's lineup, so this is recency and nothing
    more. But recency answers the question that actually matters on a board:
    a player who has not been in a lineup for a month is not a cheap source of
    value, he is unavailable, and the only reason he keeps turning up in
    optimised lineups is that his salary fell while his projection - built
    from games he played in October - did not.

    Returns (p_play, doubtful). `doubtful` is the flag the page hides on.
    """
    gp = _col(df, "gp_30d")
    n = len(df)
    if gp.max() <= 0:
        # Opening week: nobody has a thirty-day record, and scaling everyone
        # down by the same factor would be noise dressed as information.
        log.info("no player has a thirty-day game record yet (season just "
                 "started), so availability is left at 1.0 for everyone and "
                 "nobody is marked doubtful")
        return np.ones(n), np.zeros(n, dtype=bool)

    doubtful = gp <= 0
    p = np.where(
        doubtful, DOUBTFUL_P_PLAY,
        np.clip(MIN_P_PLAY + (1 - MIN_P_PLAY) * np.clip(gp / FRESH_GAMES, 0, 1),
                MIN_P_PLAY, 1.0))
    if doubtful.any():
        log.info("%d of %d players have not dressed in thirty days and are "
                 "marked doubtful", int(doubtful.sum()), n)
    return p, doubtful


# Which stats a team's game environment moves. Scoring and the shots that
# generate it scale with how many goals the game is expected to produce;
# blocks, hits and penalty minutes are closer to a function of ice time and
# role, and scaling them by the market would be inventing a relationship.
MARKET_STATS = ("goals", "assists", "points", "sog")


def apply_market(df: pd.DataFrame, totals: pd.DataFrame) -> pd.DataFrame:
    """Scale each team's scoring projections by the market's view of tonight.

    This is the fix for a stack that was always the same team. Without it the
    opponent adjustment is clipped to plus or minus fifteen percent, so the
    best matchup on the slate looks 1.35x the worst, and "which team to
    stack" collapses into "which team has the best players" - the same answer
    every night. Implied team totals span closer to 1.9x, and they already
    contain the goalie, the rest and the injuries this model cannot see.
    """
    out = df.copy()
    f = dict(zip(totals["team"], totals["factor"]))
    g = dict(zip(totals["team"], totals["implied_goals"]))
    fac = out["team"].map(f)
    missing = sorted(set(out.loc[fac.isna(), "team"]))
    if missing:
        log.warning("no market line for %s, so those teams keep their "
                    "unadjusted projections", ", ".join(missing))
    fac = fac.fillna(1.0).to_numpy(dtype=float)
    for c in MARKET_STATS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0) * fac
    out["market_factor"] = fac.round(3)
    out["implied_team_goals"] = out["team"].map(g)
    return out


def build(day: str, sims: int = 20_000, base: str = BASE,
          seed: int = 20260930, market: bool = True) -> tuple[pd.DataFrame, dict]:
    """The projection frame the DFS publisher expects, from published stats."""
    sk, go, meta = load(day, base=base)

    # The market, if it will answer. A failure here degrades the board to the
    # old behaviour rather than stopping the build - a slate without the
    # betting line is worth far more than no slate.
    totals = None
    if market:
        try:
            totals = NO.load()
            sk = apply_market(sk, totals)
            meta["market"] = {
                "teams": int(len(totals)),
                "best": totals.sort_values("implied_goals", ascending=False)
                              .head(3)[["team", "implied_goals"]]
                              .to_dict("records"),
            }
        except Exception as exc:                               # noqa: BLE001
            log.error("no betting market tonight (%s: %s). The board falls "
                      "back to projections alone, which means the stack "
                      "ranking is mostly 'who has the best players'.",
                      type(exc).__name__, str(exc)[:140])
            meta["market"] = None

    rng = np.random.default_rng(seed)
    quantiles = list(NS.SKATERS.quantiles)

    # A team's expected goals tonight, for the goalie win model. Summed over
    # the very same file, so the two halves of the board cannot disagree about
    # how much scoring is on offer.
    team_goals = (sk.assign(_g=_col(sk, "goals"))
                    .groupby("team")["_g"].sum().to_dict())
    if team_goals:
        avg = float(np.mean(list(team_goals.values())))
        if not 1.8 <= avg <= 4.5:
            log.error("team expected goals averages %.2f, which is not a "
                      "hockey number - the goalie win model is built on it",
                      avg)
        else:
            log.info("team expected goals tonight: %.2f average", avg)

    frames = []
    for side, df, draws in (("skaters", sk, None), ("goalies", go, None)):
        if not len(df):
            log.warning("no %s in the published projections", side)
            continue
        pts = (simulate_skaters(df, sims, rng) if side == "skaters"
               else simulate_goalies(df, team_goals, sims, rng))
        out = pd.DataFrame({
            "name": df["name"].astype(str),
            "team": df["team"].astype(str),
            "opponent": df.get("opponent", pd.Series([""] * len(df))).astype(str),
            "position": (df["position"].astype(str) if "position" in df
                         else pd.Series(["G"] * len(df))),
        })
        # The id the rest of the pipeline keys on. MoneyPuck's own id is not
        # in the published file, and the join to DraftKings is by name
        # anyway, so the normalised name IS the identity here.
        out["player_id"] = out["name"].map(normalise)
        for q in quantiles:
            out[NS.SKATERS.qcol(q)] = np.quantile(pts, q, axis=1)
        out["median"] = np.quantile(pts, 0.5, axis=1)
        out["ceiling"] = out[NS.SKATERS.qcol(quantiles[-1])]
        out["cond_mean"] = pts.mean(axis=1)
        out["spread"] = (out[NS.SKATERS.qcol(quantiles[-1])]
                         - out[NS.SKATERS.qcol(quantiles[0])])
        out["p_play"], out["doubtful"] = availability(df)
        # Carried through so the page can say WHY a player is hidden. A row
        # that vanishes with no reason given is indistinguishable from a bug.
        out["gp_30d"] = _col(df, "gp_30d").astype(int)
        out["last_played"] = (df["last_played"].astype(str)
                              if "last_played" in df.columns else "")
        out["mean"] = out["cond_mean"] * out["p_play"]
        out["ewm_time_on_ice"] = _col(df, "toi", default=np.nan)
        out["ewm_pp_time_on_ice"] = _col(df, "pp_toi", default=np.nan)
        log.info("%s: %d players, median DK points %.1f, ceiling %.1f",
                 side, len(out), out["median"].median(), out["ceiling"].median())
        frames.append(out)

    if not frames:
        raise ProjectionsUnavailable("the published files carried no players")
    proj = pd.concat(frames, ignore_index=True)
    meta["players"] = len(proj)
    return proj, meta
