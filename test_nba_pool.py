import random
from engine import nba_pool as NP
import nba_data as ND


def test_pool_is_legal():
    random.seed(1)
    pos = ["PG", "SG", "SF", "PF", "C", "PG/SG", "SF/PF", "PF/C", "SG/SF"]
    pl = []
    for i in range(60):
        p = random.choice(pos)
        sal = random.randrange(3000, 11000, 100)
        pl.append({"name": f"p{i}", "pos": p, "elig": ND.eligibility(p), "salary": sal,
                   "proj": sal / 1000 * 5 + random.uniform(-8, 8), "own": random.random() * 0.3,
                   "team": f"T{i % 6}", "game": f"G{i % 3}"})
    pool = NP.build(pl, per_objective=2, time_limit=1.0)
    assert len(pool) >= 10
    for L in pool:
        assert len(set(L["p"])) == 8 and L["sal"] <= 50000
        assert sorted(L["s"]) == sorted(NP.SLOTS)
        for i, s in zip(L["p"], L["s"]):
            assert s in pl[i]["elig"]
        assert len({pl[i]["game"] for i in L["p"]}) >= 2


def test_eligibility():
    assert ND.eligibility("PG/SG") == ["PG", "SG", "G", "UTIL"]
    assert ND.eligibility("PF/C") == ["PF", "C", "F", "UTIL"]
