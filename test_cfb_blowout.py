import sys, types
try:
    import pulp  # noqa: F401
except ImportError:            # only where the solver is not installed
    sys.modules["pulp"] = types.ModuleType("pulp")
import pandas as pd
import cfb_publish as P


def test_close_game_untouched_blowout_cut():
    assert P.blowout_loss(0) == 0
    assert P.blowout_loss(3) < 0.01
    assert 0.06 < P.blowout_loss(22.5) < 0.11
    assert P.blowout_loss(-22.5) < P.blowout_loss(22.5)


def test_ceiling_cut_more_than_median():
    df = pd.DataFrame({"team": ["AF"], "team_spread": [22.5], "median": [20.0],
                       "ceiling": [50.0], "q50": [20.0], "q90": [40.0]})
    o = P.apply_blowout(df, [0.5, 0.9])
    assert o["ceiling"][0] / 50 < o["median"][0] / 20 < 1
    assert o["q90"][0] / 40 < o["q50"][0] / 20
