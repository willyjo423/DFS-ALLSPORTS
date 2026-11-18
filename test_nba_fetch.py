import pandas as pd
import nba_fetch as F


def test_parse_and_dk_points():
    s = {"header": {"competitions": [{"competitors": [
            {"id": "2", "homeAway": "home", "score": "105", "team": {"abbreviation": "BOS"}},
            {"id": "19", "homeAway": "away", "score": "115", "team": {"abbreviation": "ORL"}}]}]},
         "pickcenter": [{"overUnder": 212.5, "spread": -8.5,
                         "homeTeamOdds": {"favorite": True}}],
         "boxscore": {"players": [{"team": {"id": "19"}, "statistics": [{
             "names": ["MIN", "PTS", "FG", "3PT", "FT", "REB", "AST", "TO", "STL", "BLK", "OREB", "DREB", "PF", "+/-"],
             "athletes": [
                 {"starter": True, "didNotPlay": False, "athlete": {"id": "1", "displayName": "Franz Wagner", "position": {"abbreviation": "F"}},
                  "stats": ["38", "21", "8-21", "0-2", "5-5", "11", "10", "4", "0", "1", "0", "8", "0", "+14"]},
                 {"starter": False, "didNotPlay": True, "reason": "INJURY", "athlete": {"id": "2", "displayName": "X"}, "stats": []}]}]}]}}
    rows = F.parse(s, "g1", "2026-11-16")
    df = pd.DataFrame(rows)
    w = df.iloc[0]
    assert w.team == "ORL" and w.opp == "BOS" and w.spread == 8.5 and w.total == 212.5
    assert w["played"] == 1 and w["min"] == 38 and w.tpm == 0 and w.fga == 21
    assert df.iloc[1].played == 0 and df.iloc[1].reason == "INJURY"
    pts = F.dk_points(df)
    # 21 + 0 + 13.75 + 15 + 0 + 2 - 2 = 49.75, double-double +1.5
    assert abs(pts.iloc[0] - 51.25) < 1e-9


def test_canon():
    assert F.canon("GS") == "GSW" and F.canon("utah") == "UTA" and F.canon("LAL") == "LAL"
