import news_watch as W


def test_scratch_is_news_note_change_is_not():
    old = {"A": 1, "B": 3}
    assert W.diff_nba(old, {"A": 3, "B": 3}) == ["A 1->3"]       # ruled out
    assert W.diff_nba(old, {"A": 1, "B": 3}) == []
    assert W.diff_nba(old, {"A": 1}) == ["B 3->0"]               # cleared
    assert W.diff_nba({"C": 1}, {"C": 0}) == []                  # minor


def test_goalie_confirmation_is_news():
    old = {"DAL": ("Oettinger", "Unconfirmed")}
    assert W.diff_nhl(old, {"DAL": ("Oettinger", "Confirmed")})
    assert W.diff_nhl(old, {"DAL": ("DeSmith", "Likely")})
    assert not W.diff_nhl(old, dict(old))


def test_first_look_is_baseline(monkeypatch):
    monkeypatch.setattr(W, "nba_snapshot", lambda: {"A": 3})
    monkeypatch.setattr(W, "nhl_snapshot", lambda: {})
    w = W.Watcher()
    assert w.changes() == {}
    monkeypatch.setattr(W, "nba_snapshot", lambda: {"A": 0})
    assert "nba" in w.changes()
