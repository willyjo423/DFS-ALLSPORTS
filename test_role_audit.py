import role_audit as RA


def _board(players, lineups):
    return {"players": players, "lineups": lineups, "checks": []}


def test_nba_dnp_and_minutes_cut():
    P = [{"name": "A", "team": "X", "min": 30, "_found": True, "_min": 0, "proj": 30, "_act": 0, "model": 30},
         {"name": "B", "team": "X", "min": 28, "_found": True, "_min": 10, "proj": 28, "_act": 9,
          "status": "clear", "props": 25, "model": 27},
         {"name": "C", "team": "X", "min": 30, "_found": True, "_min": 29, "proj": 30, "_act": 31},
         {"name": "D", "team": "X", "min": 30, "_found": False, "status": "out"}]
    m = RA.audit("nba", _board(P, [{"p": [0, 1, 2, 3]}]))
    assert [(x["name"], x["reason"]) for x in m] == [("A", "did not play"), ("B", "minutes cut")]
    assert m[1]["kind"] == "blind"
    assert "no props posted" in m[0]["signals"] and m[0]["kind"] == "warned"


def test_unused_non_chalk_is_ignored():
    P = [{"name": "A", "min": 30, "_found": True, "_min": 0, "own": 0.05}]
    assert RA.audit("nba", _board(P, [])) == []
    P[0]["own"] = 0.3
    assert RA.audit("nba", _board(P, []))[0]["lineups"] == 0


def test_nhl_scratch_toi_and_goalie():
    P = [{"name": "S", "pos": "W", "toi": 15, "_found": False, "pp": 0.25},
         {"name": "T", "pos": "C", "toi": 18, "_found": True, "_toi": 8},
         {"name": "G", "pos": "G", "pp": 0.85, "_found": True, "_started": False},
         {"name": "H", "pos": "G", "pp": 0.15, "_found": True, "_started": False}]
    m = {x["name"]: x for x in RA.audit("nhl", _board(P, [{"p": [0, 1, 2, 3]}]))}
    assert m["S"]["reason"] == "scratched" and "chance to play 25%" in m["S"]["signals"]
    assert m["T"]["reason"] == "ice time cut"
    assert m["G"]["reason"] == "goalie did not start" and "H" not in m


def test_running_summary(tmp_path):
    miss = [{"name": "A", "team": "X", "reason": "did not play", "lineups": 3,
             "signals": ["status questionable"], "kind": "warned"}]
    f = tmp_path / "r.json"
    RA.update_running(f, "2026-11-20", miss)
    s = RA.update_running(f, "2026-11-21", miss)
    assert s["nights"] == 2 and s["warned"] == 2 and s["repeat_players"] == ["A"]
    assert s["warning_ignored"] == {"status": 2}


def test_football_dnp_and_barely_played():
    P = [{"name": "Hudson", "proj": 9, "_found": False, "model": 9},
         {"name": "W", "proj": 12, "_found": True, "_snaps": 9, "_snap_pct": 0.14, "props": 11, "model": 12},
         {"name": "K", "proj": 5, "_found": True, "_snaps": 9, "_snap_pct": 0.14}]
    m = RA.audit("nfl", _board(P, [{"p": [0, 1, 2]}]))
    assert [(x["name"], x["reason"]) for x in m] == [("Hudson", "did not play"), ("W", "barely played")]


def test_cap_unlikely():
    import sanity as S
    P = [{"name": "Hagel", "pp": 0.25}, {"name": "A", "pp": 1.0}, {"name": "Out", "pp": 0.0, "status": "out"}]
    L = [{"p": [0, 1], "top1": 5 - k * 0.1} for k in range(30)] + [{"p": [1], "top1": 1} for _ in range(70)]
    pay = S.cap_unlikely({"players": P, "lineups": L})
    unl = [x for x in pay["lineups"] if x.get("unl")]
    assert len(pay["lineups"]) == 80 and len(unl) == 10 and unl[0]["top1"] == 5
    import optimize_settings as O
    assert not any(x.get("unl") for x in O.order(pay["lineups"], "top1")[:13])
