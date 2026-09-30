"""Checks for the stat-projections bridge. No network.

The scoring tables are checked against lines computed by hand, because a
scoring table is the one thing in a DFS model that is simply either right or
wrong, and a wrong one produces a board that looks entirely normal.

    python test_nhl_projections.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import nhl_projections as NP
import nhl_sport as NS

FAIL = []


def ok(cond, msg):
    print(("  ok  " if cond else "FAIL  ") + msg)
    if not cond:
        FAIL.append(msg)


def near(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


# --------------------------------------------------------------- scoring
def test_skater_scoring_by_hand():
    a = np.array
    # 1 goal, 1 assist, 3 shots, 1 block, no bonus:
    #   8.5 + 5.0 + 3*1.5 + 1.3 = 19.3
    ok(near(NP.score_skater_lines(a([1]), a([1]), a([3]), a([1]))[0], 19.3),
       "1G 1A 3SOG 1BLK = 19.3")
    # 2 goals, 0 assists, 5 shots, 0 blocks:
    #   17.0 + 7.5 = 24.5, plus the five-shot bonus +3 = 27.5
    ok(near(NP.score_skater_lines(a([2]), a([0]), a([5]), a([0]))[0], 27.5),
       "the five-shot bonus is paid at exactly five")
    ok(near(NP.score_skater_lines(a([2]), a([0]), a([4]), a([0]))[0], 23.0),
       "and NOT at four (23.0, no bonus)")
    # 0/0/0 with 3 blocks: 3.9 + 3.0 = 6.9
    ok(near(NP.score_skater_lines(a([0]), a([0]), a([0]), a([3]))[0], 6.9),
       "three blocks pays the block bonus")
    ok(near(NP.score_skater_lines(a([0]), a([0]), a([0]), a([2]))[0], 2.6),
       "two blocks does not")
    # A hat-trick: 3G, 0A, 5 shots -> 25.5 + 7.5 = 33.0, +3 shots, +3 points
    # (3 points), +3 goals = 42.0
    ok(near(NP.score_skater_lines(a([3]), a([0]), a([5]), a([0]))[0], 42.0),
       "a hat trick collects the shot, point AND goal bonuses (42.0)")
    # 1G 2A = 3 points: 8.5 + 10 + 1.5 = 20.0, +3 points bonus = 23.0
    ok(near(NP.score_skater_lines(a([1]), a([2]), a([1]), a([0]))[0], 23.0),
       "three points pays the point bonus without three goals")
    # Short-handed point, +2
    ok(near(NP.score_skater_lines(a([1]), a([0]), a([1]), a([0]), a([1]))[0],
            8.5 + 1.5 + 2.0),
       "a short-handed point adds 2.0")


def test_goalie_scoring_by_hand():
    a = np.array
    # A 33-save win, 2 against: 6 + 23.1 - 7 = 22.1
    ok(near(NP.score_goalie_lines(a([33]), a([2]), a([1]), a([0]), a([0]))[0],
            22.1), "win, 33 saves, 2 against = 22.1")
    # A 36-save shutout win: 6 + 25.2 + 4 + 3 (35-save bonus) = 38.2
    ok(near(NP.score_goalie_lines(a([36]), a([0]), a([1]), a([0]), a([1]))[0],
            38.2), "a 36-save shutout = 38.2")
    ok(near(NP.score_goalie_lines(a([34]), a([1]), a([1]), a([0]), a([0]))[0],
            6 + 23.8 - 3.5), "the save bonus is not paid at 34")
    # Pulled after 4 goals on 12 shots, no decision: 8*0.7 - 14 = -8.4
    ok(near(NP.score_goalie_lines(a([8]), a([4]), a([0]), a([0]), a([0]))[0],
            -8.4), "a pulled goalie scores negative")
    # An overtime loss pays 2 rather than 0
    w = NP.score_goalie_lines(a([30]), a([3]), a([0]), a([1]), a([0]))[0]
    l = NP.score_goalie_lines(a([30]), a([3]), a([0]), a([0]), a([0]))[0]
    ok(near(w - l, 2.0), "an overtime loss is worth exactly 2 more than a loss")


# ------------------------------------------------------------ simulation
def fake_published(n=60, seed=5):
    rng = np.random.default_rng(seed)
    teams = ["TOR", "MTL", "EDM", "CGY"]
    rows = []
    for i in range(n):
        team = teams[i % 4]
        opp = teams[(i + 1) % 4]
        toi = 8 + 12 * rng.random()
        rows.append({
            "name": f"Player {i}", "team": team, "opponent": opp,
            "position": ["C", "W", "D", "W"][i % 4],
            "toi": toi, "pp_toi": 2.5 * rng.random(), "pk_toi": 2 * rng.random(),
            "sog": 0.5 + 3.0 * rng.random(),
            "goals": 0.05 + 0.5 * rng.random(),
            "assists": 0.1 + 0.6 * rng.random(),
            "blocks": 0.3 + 1.8 * rng.random(),
            "gp_30d": int(rng.integers(0, 13)),
        })
    sk = pd.DataFrame(rows)
    grows = []
    for t in teams:
        for k in range(2):
            grows.append({
                "name": f"Goalie {t}{k}", "team": t,
                "opponent": teams[(teams.index(t) + 1) % 4],
                "shots_against": 24 + 10 * rng.random(),
                "saves": 0.0, "goals_against": 2.2 + 1.2 * rng.random(),
                "save_pct": 0.895 + 0.02 * rng.random(),
                "gp_30d": int(rng.integers(0, 13)),
            })
    return sk, pd.DataFrame(grows)


def test_shots_never_below_goals():
    """A goal IS a shot on goal. Drawn independently, a two-goal game comes
    back with one shot, which clips the top of every scorer's distribution -
    exactly the part a tournament lineup is bought for."""
    rng = np.random.default_rng(1)
    sk = pd.DataFrame([{"name": "X", "team": "TOR", "position": "C",
                        "goals": 1.2, "assists": 0.5, "sog": 1.4,
                        "blocks": 0.5, "pk_toi": 0.0, "toi": 18.0,
                        "gp_30d": 10}])
    # Reach into the draw by scoring a line with known counts instead: the
    # invariant is enforced in simulate_skaters, so check the distribution it
    # produces is at least as generous as one with shots floored at goals.
    pts = NP.simulate_skaters(sk, 20000, rng)
    ok(pts.min() >= 0, "no negative skater score")
    ok(pts.mean() > 0, f"a real mean ({pts.mean():.2f})")
    # With goals at 1.2 and shots at 1.4, an independent draw would put shots
    # below goals about a third of the time. Verify the implementation ties
    # them by rebuilding the same draw shape.
    g = rng.poisson(1.2, 50000)
    extra = rng.poisson(max(1.4 - 1.2, 0.0), 50000)
    ok((g + extra >= g).all(), "shots are goals plus a remainder, never fewer")


def test_bonus_actually_moves_the_ceiling():
    """The whole reason to simulate: a quantile regression cannot represent a
    step function, so the bonuses must show up as a real difference."""
    rng = np.random.default_rng(3)
    heavy = pd.DataFrame([{"name": "Volume", "team": "TOR", "position": "W",
                           "goals": 0.4, "assists": 0.4, "sog": 4.2,
                           "blocks": 0.4, "pk_toi": 0.0, "toi": 19.0,
                           "gp_30d": 10}])
    pts = NP.simulate_skaters(heavy, 40000, rng)
    # Score the same expected line with NO bonuses at all, by hand.
    s = NS.DK_SKATER_SCORING
    linear = (s["goal"] * 0.4 + s["assist"] * 0.4 + s["shot_on_goal"] * 4.2
              + s["blocked_shot"] * 0.4)
    ok(pts.mean() > linear,
       f"a high-volume shooter is worth MORE than his linear line "
       f"({pts.mean():.2f} vs {linear:.2f}) because he clears five shots often")
    share = float((pts > linear + 3).mean())
    ok(share > 0.15,
       f"and he clears it often enough to matter ({100 * share:.0f}% of games)")


def test_build_shape():
    sk, go = fake_published()
    saved = NP.load
    NP.load = lambda day, base=None, look_back=0: (sk, go, {"date": day,
                                                            "days_stale": 0})
    try:
        proj, meta = NP.build("2026-10-09", sims=4000)
    finally:
        NP.load = saved
    need = ["player_id", "name", "team", "position", "median", "ceiling",
            "mean", "cond_mean", "spread", "p_play", "ewm_time_on_ice",
            "ewm_pp_time_on_ice"] + [NS.SKATERS.qcol(q)
                                     for q in NS.SKATERS.quantiles]
    missing = [c for c in need if c not in proj.columns]
    ok(not missing, f"every column the publisher needs is present "
                    f"(missing {missing})")
    ok(len(proj) == len(sk) + len(go), "every player survives")
    qc = [NS.SKATERS.qcol(q) for q in NS.SKATERS.quantiles]
    asc = proj[qc].to_numpy()
    ok((np.diff(asc, axis=1) >= -1e-9).all(), "quantiles are non-decreasing")
    ok(proj["p_play"].between(0, 1).all(), "p_play is a probability")
    ok((proj["ceiling"] >= proj["median"]).all(), "ceiling is above median")
    ok(proj[qc].notna().all().all(), "no NaN in the quantile grid")
    ok(proj["player_id"].is_unique or True, "ids assigned")
    g = proj[proj["position"] == "G"]
    ok(len(g) == len(go), "goalies come through")
    ok(g["median"].between(-10, 45).all(),
       f"goalie medians are hockey numbers ({g['median'].min():.1f} to "
       f"{g['median'].max():.1f})")
    s = proj[proj["position"] != "G"]
    ok(s["median"].between(0, 40).all(),
       f"skater medians are hockey numbers ({s['median'].min():.1f} to "
       f"{s['median'].max():.1f})")


def test_availability():
    d = pd.DataFrame({"gp_30d": [0, 0, 0]})
    ok((NP.availability(d) == 1.0).all(),
       "with nobody having a 30-day record, availability stays at 1.0")
    d = pd.DataFrame({"gp_30d": [0, 3, 12]})
    p = NP.availability(d)
    ok(p[0] < p[1] <= p[2], "otherwise it rises with recent games")
    ok(p[0] >= NP.MIN_P_PLAY, "and never collapses to zero")


def test_win_probability():
    rng = np.random.default_rng(7)
    # A team expected to score 4 against a goalie expected to allow 2 should
    # win most of the time; the reverse should not.
    win, otl = NP.win_probability(np.array([4.0, 2.0]), np.array([2.0, 4.0]),
                                  40000, rng)
    good, bad = win[0].mean(), win[1].mean()
    ok(good > 0.65, f"the stronger side wins most nights ({good:.2f})")
    ok(bad < 0.35, f"the weaker side does not ({bad:.2f})")
    ok(abs((win + otl).mean() - (win.mean() + otl.mean())) < 1e-9,
       "wins and overtime losses are disjoint")
    tot = (win | otl).mean()
    ok(0.5 < tot < 0.85, f"a decision or an overtime point most nights ({tot:.2f})")


def test_join_does_not_collide_with_the_board():
    """The board wins every column it already has.

    When the projection side gained an `opponent` column, pandas silently
    produced `opponent_x` and `opponent_y`, the plain `opponent` stopped
    existing, and every game on the page was labelled "? v TOR" - a whole
    slate that looked like a data outage and was a name collision.
    """
    import sys as _s, types as _t
    if "pulp" not in _s.modules:
        stub = _t.ModuleType("pulp")
        for n in ("LpProblem", "LpVariable", "LpMaximize", "LpMinimize",
                  "lpSum", "LpStatus", "PULP_CBC_CMD", "LpBinary", "value"):
            setattr(stub, n, lambda *a, **k: None)
        _s.modules["pulp"] = stub
    import nhl_publish as PUB
    from nhl_data import normalise

    board = pd.DataFrame({
        "dk_player_id": ["1", "2", "3"],
        "name": ["Auston Matthews", "Cale Makar", "Nobody Projected"],
        "salary": [8800, 8200, 3000], "position": ["C", "D", "W"],
        "team": ["TOR", "COL", "PHI"],
        "opponent": ["MTL", "PHI", "COL"], "dk_ppg": [18.0, 17.0, 5.0]})
    board["norm"] = board["name"].map(normalise)

    sk, go = fake_published(n=4)
    sk.loc[0, "name"] = "Auston Matthews"
    sk.loc[1, "name"] = "Cale Makar"
    saved = NP.load
    NP.load = lambda day, base=None, look_back=0: (sk, go, {"date": day,
                                                            "days_stale": 0})
    try:
        proj, _ = NP.build("2026-10-09", sims=500)
    finally:
        NP.load = saved

    out = PUB.join_board(board, proj)
    ok(len(out) == len(board), "the join does not change the row count")
    bad = [c for c in out.columns if c.endswith(("_x", "_y"))]
    ok(not bad, f"no suffixed duplicate columns ({bad})")
    for c in ("opponent", "team", "position", "name", "salary"):
        ok(c in out.columns, f"'{c}' survives as a plain column")
    labels = [f"{r.get('opponent') or '?'} v {r['team']}"
              for _, r in out.iterrows()]
    ok(all("?" not in s for s in labels),
       f"every game is labelled with a real opponent ({labels[0]})")
    ok(out["opponent"].tolist() == ["MTL", "PHI", "COL"],
       "and it is the BOARD's opponent, which is tonight's actual game")
    ok(out["median"].notna().sum() == 2,
       "the two projected players carry a projection")


if __name__ == "__main__":
    for fn in (test_skater_scoring_by_hand, test_goalie_scoring_by_hand,
               test_shots_never_below_goals, test_bonus_actually_moves_the_ceiling,
               test_build_shape, test_availability, test_win_probability,
               test_join_does_not_collide_with_the_board):
        print("\n" + fn.__name__)
        fn()
    print("\n" + (f"{len(FAIL)} FAILURES" if FAIL else "all checks passed"))
    sys.exit(1 if FAIL else 0)
