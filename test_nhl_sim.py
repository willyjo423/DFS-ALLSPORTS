import json, os
import pytest
from engine import nhl_sim as SIM


def test_goalie_moves_against_opposing_skaters():
    import numpy as np
    qs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.97]
    pl = [{"name": "G1", "pos": "G", "team": "A", "opp": "B", "proj": 12, "pp": 1.0,
           "q": [0, 4, 12, 18, 24, 30]},
          {"name": "S1", "pos": "W", "team": "B", "opp": "A", "proj": 8, "pp": 1.0, "lg": "B-L1",
           "q": [1, 3, 7, 11, 16, 22]},
          {"name": "S2", "pos": "C", "team": "B", "opp": "A", "proj": 8, "pp": 1.0, "lg": "B-L1",
           "q": [1, 3, 7, 11, 16, 22]}]
    S = SIM.simulate_scores({"players": pl, "quantiles": qs}, 4000, np.random.default_rng(0))
    assert abs(S[:, 1].mean() - 8) < 0.6 and abs(S[:, 0].mean() - 12) < 0.8
    assert np.corrcoef(S[:, 1], S[:, 2])[0, 1] > 0.08     # linemates together (fitted ~0.14)
    assert np.corrcoef(S[:, 0], S[:, 1])[0, 1] < -0.05    # goalie vs opposing skater
