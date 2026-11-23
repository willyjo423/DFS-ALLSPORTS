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
    monkeypatch.setattr(W, "nba_lineups_snapshot", lambda: {})
    monkeypatch.setattr(W, "nhl_lines_snapshot", lambda: {})
    w = W.Watcher()
    assert w.changes() == {}
    monkeypatch.setattr(W, "nba_snapshot", lambda: {"A": 0})
    assert "nba" in w.changes()


def test_lineup_confirmation_and_swap_are_news():
    old = {"BOS": (False, ("A", "B", "C", "D", "E"), ())}
    assert W.diff_nba_lineups(old, {"BOS": (True, ("A", "B", "C", "D", "E"), ())}) == ["BOS lineup confirmed"]
    m = W.diff_nba_lineups(old, {"BOS": (False, ("A", "B", "C", "D", "F"), ("E",))})
    assert any("starters F in" in x for x in m) and any("inactive E" in x for x in m)
    assert W.diff_nba_lineups(old, dict(old)) == []
    assert W.diff_nba_lineups({}, old) == ["BOS lineup posted"]


def test_nhl_line_and_pp_changes_are_news():
    old = {"DAL": (("f1:A", "f1:B"), ("A",), ())}
    assert W.diff_nhl_lines(old, {"DAL": (("f1:A", "f1:C"), ("A",), ())}) == ["DAL lines changed"]
    assert W.diff_nhl_lines(old, {"DAL": (("f1:A", "f1:B"), ("C",), ("B",))}) == ["DAL PP1 changed", "DAL out B"]
    assert W.diff_nhl_lines(old, dict(old)) == []


def test_lines_read_every_other_look(monkeypatch):
    calls = []
    for n in ("nba_snapshot", "nba_lineups_snapshot", "nhl_snapshot"):
        monkeypatch.setattr(W, n, lambda: {})
    monkeypatch.setattr(W, "nhl_lines_snapshot", lambda: calls.append(1) or {})
    w = W.Watcher()
    for _ in range(4):
        w.changes()
    assert len(calls) == 2
