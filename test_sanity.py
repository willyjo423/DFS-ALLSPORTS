import json
from pathlib import Path

import sanity


def board(**kw):
    p = dict(name="A", pos="SF", team="DEN", own=0.3, chalk=True, status="clear",
             model=8.0, props=19.4, min=10)
    p.update(kw)
    return {"sport": "nba", "players": [p, dict(name="B", pos="PG", team="X", own=0.05)],
            "lineups": [{"p": [0, 1]}] * 10 + [{"p": [1]}] * 2}


def test_chaney_johnson_is_flagged():
    kinds = {c["kind"] for c in sanity.check(board())}
    assert {"gap", "role", "pool"} <= kinds


def test_quiet_board_has_no_checks():
    b = board(props=8.5, min=32)
    b["lineups"] = [{"p": [0]}, {"p": [1]}]
    assert sanity.check(b) == []


def test_attach_tags_player():
    b = sanity.attach(board())
    assert b["checks"] and "chk" in b["players"][0]


def test_real_11_20_board_catches_chaney():
    f = Path("docs/data/archive/2026-11-20/nba_dk_classic_156300.json")
    if not f.exists():
        return
    names = {c["name"] for c in sanity.check(json.loads(f.read_text()))}
    assert {"Chaney Johnson", "Ajay Mitchell"} <= names
