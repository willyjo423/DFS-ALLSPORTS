import nhl_props as P


def _ou(desc, lines):
    outs = []
    for ln, o, u in lines:
        outs += [{"description": "Over", "type": "O", "price": {"handicap": str(ln), "american": o}},
                 {"description": "Under", "type": "U", "price": {"handicap": str(ln), "american": u}}]
    return {"description": desc, "period": {"description": "Game"}, "outcomes": outs}


def _ms(desc, items):
    return {"description": desc, "period": {"description": "Game"},
            "outcomes": [{"description": t, "price": {"american": a}} for t, a in items]}


EV = {"displayGroups": [
    {"description": "Shots on Goal", "markets": [
        _ou("Total Shots On Goal - Alex DeBrincat (DET)", [(3.5, "+105", "-135")]),
        _ms("Shots on Goal Milestones - Alex DeBrincat (DET)",
            [("2+ Shots on Goal", "-800"), ("3+ Shots on Goal", "-245"), ("4+ Shots on Goal", "+112"), ("5+ Shots on Goal", "+245")])]},
    {"description": "Points/Assists", "markets": [
        _ms("Points Milestones - Alex DeBrincat (DET)", [("To Score 1+ Points", "-193"), ("To Score 2+ Points", "+235"), ("To Score 3+ Points", "+700")]),
        _ms("Assists Milestones - Alex DeBrincat (DET)", [("To Record 1+ Assists", "+140"), ("To Record 2+ Assists", "+800")])]},
    {"description": "Blocked Shots", "markets": [
        {"description": "Ben Chiarot (DET) - Blocked Shots", "period": {"description": "Game"},
         "outcomes": [{"description": "Over 1.5", "price": {"american": "EVEN"}}, {"description": "Under 1.5", "price": {"american": "-130"}}]}]},
    {"description": "Goalie Props", "markets": [_ou("Total Saves - Daniil Tarasov (DET)", [(23.5, "+115", "-150")])]},
    {"description": "Game Props", "markets": [_ms("Total Powerplay Goals - Detroit Red Wings", [("Over 0.5", "-125")])]},
    {"description": "Period Player Props", "markets": [_ms("1st Period 1+ Points", [("Alex DeBrincat (DET)", "+305")])]},
]}


def test_parse_and_dk():
    out = P.parse_event(EV)
    d = out[("Alex DeBrincat", "DET")]
    assert set(d) == {"SOG", "PTS", "AST"}
    assert ("Ben Chiarot", "DET") in out and out[("Ben Chiarot", "DET")]["BLK"][0][0] == 2
    assert ("Daniil Tarasov", "DET") in out
    assert not any(k[0].startswith("Detroit") for k in out)
    mk = P.market_dk(d, "W")
    assert 3.0 < mk["sog"] < 4.0 and 0.75 < mk["pts"] < 1.1 and 0 < mk["g"] < mk["pts"]
    assert 9 < mk["dk"] < 16
    assert P.market_dk({"SOG": d["SOG"]}, "W") is None


def test_poisson_fit_recovers_rate():
    pts = [(k, P.poisson_tail(2.7, k)) for k in (2, 3, 4, 5)]
    assert abs(P.fit_rate(pts) - 2.7) < 0.05


def test_goalie_saves_line_moves_projection():
    p = {"name": "X", "pos": "G", "team": "DET", "proj": 10.0, "med": 10.0, "mean": 10.0,
         "ceil": 25.0, "q": [1, 5, 10, 15, 20, 25], "pp": 1.0, "sv_exp": 24.0}
    st = {"SV": [(28, 0.5)]}         # saves line ~28 vs model 24
    assert P._apply_goalie(p, st) == 1
    assert p["proj"] > 12 and p["sv_mkt"] > 27 and "saves line" in p["news"]


def test_gap_weight():
    assert P.weight(10, 10.5) == P.W_MARKET
    assert P.weight(10, 14) == P.W_GAP_MAX
