import pandas as pd
import nba_props as PR


def _mk(desc, rungs):
    outs = []
    for ln, o, u in rungs:
        outs += [{"description": "Over", "type": "O", "price": {"handicap": str(ln), "american": o}},
                 {"description": "Under", "type": "U", "price": {"handicap": str(ln), "american": u}}]
    return {"description": desc, "period": {"description": "Game"}, "outcomes": outs}


EV = {"displayGroups": [
    {"description": "Player Points", "markets": [
        _mk("Total Points - Alperen Sengun (HOU)", [(17.5, "-280", "+205"), (18.5, "-215", "+160"), (19.5, "-170", "+130")]),
        {"description": "Points Milestones - Alperen Sengun (HOU)", "period": {"description": "Game"},
         "outcomes": [{"description": "To Score 20+ Points", "price": {"american": "-180"}}]}]},
    {"description": "Player Rebounds", "markets": [
        _mk("Total Rebounds - Alperen Sengun (HOU)", [(8.5, "-200", "+150"), (9.5, "-120", "-110"), (10.5, "+130", "-170")])]},
    {"description": "Player Combos", "markets": [
        _mk("Total Points, Rebounds and Assists - Alperen Sengun (HOU)", [(35.5, "-115", "-115")])]},
    {"description": "Game Lines", "markets": [_mk("Total Points - Nope (HOU)", [(1.5, "-110", "-110")])]},
]}


def test_parse_and_resolve():
    out = PR.parse_event(EV)
    s = out[("Alperen Sengun", "HOU")]
    assert 20 < s["P"] < 23          # ladder still favours the over at 19.5
    assert 9.3 < s["R"] < 9.9        # 9.5 is close to even, slight under lean
    r = PR.resolve(s)
    assert 4 < r["A"] < 6.5          # PRA 35.5 - P - R
    assert len(out) == 1             # non-player groups ignored


def test_market_dk_and_blend():
    s = {"P": 21.0, "R": 9.6, "A": 5.0}
    rates = pd.Series({"P": 0.6, "R": 0.3, "A": 0.15, "other": 0.12, "min_avg": 33.0})
    tot, share = PR.market_dk(s, rates, 34.0)
    assert abs(tot - (21 + 1.25 * 9.6 + 1.5 * 5 + 0.12 * 34)) < 1e-6
    assert share > 0.85
    v, w = PR.blend(40.0, (tot, share))
    assert abs(w - PR.W_MARKET) < 1e-9 and 40 < v < tot
    # points only: rebounds/assists filled from his rates, lower weight
    tot2, share2 = PR.market_dk({"P": 21.0}, rates, 34.0)
    assert share2 < 0.6 and PR.blend(40.0, (tot2, share2))[1] < PR.W_MARKET
    assert PR.market_dk({"R": 9.0}, rates, 34.0) is None


def test_big_gap_leans_on_market():
    v, w = PR.blend(55.6, (40.3, 0.95))
    assert w > 0.8 and v < 43.5          # LeBron 11/18: actual 34.75
    v2, w2 = PR.blend(40.0, (41.0, 0.95))
    assert abs(w2 - PR.W_MARKET) < 1e-9
