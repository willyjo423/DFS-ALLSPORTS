import cfb_odds as CO


def test_espn_shape_reads_through_week_lines():
    class R:
        status_code = 200
        def json(self):
            return {"events": [{"date": "2026-11-21T17:00Z", "competitions": [{
                "date": "2026-11-21T17:00Z",
                "competitors": [
                    {"homeAway": "home", "team": {"displayName": "Alabama Crimson Tide"}},
                    {"homeAway": "away", "team": {"displayName": "Auburn Tigers"}}],
                "odds": [{"overUnder": 50.5, "spread": -10.5,
                          "homeTeamOdds": {"favorite": True}}]}]}]}
    real = CO.requests.get
    CO.requests.get = lambda *a, **k: R()
    try:
        games = CO.espn_fetch()
    finally:
        CO.requests.get = real
    import datetime as dt
    t = CO.week_lines(games, ["Alabama", "Auburn"], now=dt.datetime(2026, 11, 19, tzinfo=dt.timezone.utc))
    t = t.set_index(t.columns[0]) if "school" not in t.columns else t.set_index("school")
    cols = [c for c in t.columns if "implied" in c]
    assert t.loc["Alabama", cols[0]] > t.loc["Auburn", cols[0]]
    assert abs(t.loc["Alabama", cols[0]] - 30.5) < 0.01
