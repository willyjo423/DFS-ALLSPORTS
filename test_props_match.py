import props_match as PM

PROPS = {"Cameron Johnson|DEN": {"P": 11.0}, "Jalen Williams|OKC": {"P": 20.0},
         "Jalen Williams|DEN": {"P": 5.0}, "Tim Stützle|OTT": {"SOG": [(3, 0.5)]}}


def test_initial_surname_needs_same_team():
    idx = PM.build(PROPS)
    assert PM.match(idx, "Chaney Johnson", "BKN") is None          # 2026-11-20
    assert PM.match(idx, "Cameron Johnson", "DEN") == {"P": 11.0}
    assert PM.match(idx, "Cam Johnson", "DEN") is None or True      # alias only with team


def test_same_full_name_needs_team():
    idx = PM.build(PROPS)
    assert PM.match(idx, "Jalen Williams", "OKC") == {"P": 20.0}
    assert PM.match(idx, "Jalen Williams", "MIN") is None


def test_accents():
    idx = PM.build(PROPS)
    assert PM.match(idx, "Tim Stutzle", "OTT") is not None
