import numpy as np
from engine import contest_roi as CR


def test_payout_table_sums_to_pool_and_falls():
    pay = CR.payout_table(104000, 300000, 3.0, 50000)
    assert abs(pay.sum() - 300000) < 1
    assert pay[0] >= pay[1] >= pay[100] >= pay[-1] > 0
    assert len(pay) == round(0.22 * 104000)


def test_featured_reads_first_prize_from_name():
    c = CR.featured(7, [{"dg": 7, "n": "CFB $70K Night [$20K to 1st]", "po": 70000, "m": 8000, "a": 10},
                        {"dg": 7, "n": "CFB $5K Double Up", "po": 5000, "m": 600, "a": 10}])
    assert c["first"] == 20000 and c["pool"] == 70000


def test_field_average_roi_is_the_rake():
    rng = np.random.default_rng(0)
    S = rng.normal(10, 4, (1500, 40)).clip(0)
    Fd = np.array([rng.choice(40, 8, replace=False) for _ in range(3000)])
    c = {"entries": 20000, "pool": 50000, "fee": 3.0, "first": 8000}
    r = CR.lineup_roi(S, Fd, Fd[:800], c, None, sharp=0.0)
    rake = c["pool"] / (c["entries"] * c["fee"]) - 1
    assert abs(r.mean() - rake) < 0.08
