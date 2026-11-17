import nhl_lines as L


def test_unmangle_double_encoded_names():
    assert L._unmangle("Tim StÃ¼tzle") == "Tim Stützle"
    assert L._unmangle("Martin FehÃ©rvÃ¡ry") == "Martin Fehérváry"
    assert L._unmangle("Tim Stützle") == "Tim Stützle"
    assert L._unmangle("Connor McDavid") == "Connor McDavid"
