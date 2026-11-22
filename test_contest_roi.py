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


def test_tiers_pick_lobby_contest_per_size_and_fall_back():
    from engine import contest_roi as C
    lobby = [{"dg": 7, "n": "Big [$10K to 1st]", "po": 50000, "m": 17000, "a": 3},
             {"dg": 7, "n": "mini-MAX", "po": 3000, "m": 1200, "a": 3},
             {"dg": 7, "n": "Double Up", "po": 900, "m": 100, "a": 10}]
    t = C.tier_contests(7, lobby)
    assert t["large"]["name"].startswith("Big") and t["large"]["real"]
    assert t["mid"]["name"] == "mini-MAX"
    assert t["small"]["real"] is False          # double-ups are not tournaments


def test_tier_multiple_is_top_heavy_and_respects_tail():
    from engine import contest_roi as C
    assert C.tier_multiple(70, "large") == 0.0
    assert C.tier_multiple(99.9, "large") > C.tier_multiple(99, "large") > C.tier_multiple(90, "large")
    lines = {"win": 300.0, "top1": 280.0}
    assert C.tier_multiple(100, "large", lines, 301.0) < C.tier_multiple(100, "large", lines, 340.0)
