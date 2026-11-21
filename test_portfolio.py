import numpy as np
from engine import portfolio as PF


def test_set_covers_different_nights():
    # lineups 0 and 1 win the same nights; 2 wins different ones
    V = np.zeros((100, 3), np.float32)
    V[:30, 0] = 10; V[:28, 1] = 10; V[50:75, 2] = 10
    order = PF.greedy(V, 2)
    assert order == [0, 2]          # by own rate it would be [0, 1]


def test_assign_marks_pick_order():
    pool = [{} for _ in range(4)]
    ls = np.random.default_rng(0).normal(100, 10, (500, 4))
    q999 = np.full(500, 125.0); q99 = np.full(500, 115.0)
    PF.assign(pool, ls, q999, q99, n=3)
    assert sorted(L["port"] for L in pool if "port" in L) == [1, 2, 3]
