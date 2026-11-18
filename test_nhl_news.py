from datetime import datetime, timezone, timedelta

import nhl_news as N
from availability import name_keys


def P(name, team, pos, **kw):
    d = {"name": name, "team": team, "pos": pos, "salary": 5000, "pp": 1.0,
         "mean": 10.0, "med": 9.0, "ceil": 20.0, "q": [1.0, 9.0, 20.0],
         "toi": 18.0, "pptoi": 2.0, "status": "clear"}
    d.update(kw)
    return d


def test_confirmed_goalie_benches_the_backup():
    pl = [P("Jeremy Swayman", "BOS", "G", toi=None, pptoi=None),
          P("Michael DiPietro", "BOS", "G", toi=None, pptoi=None)]
    N.apply_goalies(pl, {"BOS": {"name": "Michael DiPietro",
                                 "status": "Confirmed"}}, name_keys)
    assert pl[0]["status"] == "out" and pl[0]["pp"] == 0 and pl[0]["mean"] == 0
    assert pl[1]["pp"] == 1.0 and pl[1]["mean"] == 10.0
    assert pl[1]["gst"] == "Confirmed"


def test_likely_goalie_splits_the_chance():
    pl = [P("A Starter", "TBL", "G"), P("B Backup", "TBL", "G")]
    N.apply_goalies(pl, {"TBL": {"name": "A Starter", "status": "Likely"}},
                    name_keys)
    assert pl[0]["pp"] == 0.85 and abs(pl[1]["pp"] - 0.15) < 1e-9
    assert pl[1]["status"] == "clear"


def test_unconfirmed_changes_nothing():
    pl = [P("A Starter", "TBL", "G"), P("B Backup", "TBL", "G")]
    N.apply_goalies(pl, {"TBL": {"name": "A Starter",
                                 "status": "Unconfirmed"}}, name_keys)
    assert pl[0]["pp"] == pl[1]["pp"] == 1.0


def test_lines_scratch_injury_and_promotion():
    now = datetime(2026, 11, 17, 20, tzinfo=timezone.utc)
    pl = [P("Top Guy", "OTT", "C", line="L1", ppu=1, toi=20, pptoi=3),
          P("Hurt Guy", "OTT", "W", line="L1", ppu=1, toi=19, pptoi=3),
          P("Riser", "OTT", "W", line="L3", toi=13, pptoi=0.2),
          P("Healthy Scratch", "OTT", "W", line="L4", toi=10, pptoi=0),
          P("Mid Guy", "OTT", "C", line="L2", toi=16, pptoi=1),
          P("Dtd Guy", "OTT", "C", line="L2", ppu=2, toi=17, pptoi=1.5)]
    tl = {"OTT": {"updated": now - timedelta(hours=3), "players": [
        {"name": "Top Guy", "group": "f1", "cat": "ev", "inj": "", "gtd": False},
        {"name": "Riser", "group": "f1", "cat": "ev", "inj": "", "gtd": False},
        {"name": "Top Guy", "group": "pp1", "cat": "pp", "inj": "", "gtd": False},
        {"name": "Riser", "group": "pp1", "cat": "pp", "inj": "", "gtd": False},
        {"name": "Mid Guy", "group": "f2", "cat": "ev", "inj": "", "gtd": True},
        {"name": "Hurt Guy", "group": "ir", "cat": "oi", "inj": "ir", "gtd": False},
        {"name": "Dtd Guy", "group": "ir", "cat": "oi", "inj": "dtd", "gtd": False},
        {"name": "Dtd Guy", "group": "pp2", "cat": "pp", "inj": "dtd", "gtd": False},
    ]}}
    n = N.apply_lines(pl, tl, name_keys, now=now)
    top, hurt, riser, scr, mid, dtd = pl
    assert dtd["pp"] == 0.25 and dtd["doubtful"] and dtd["status"] == "clear"
    assert hurt["status"] == "out"
    assert scr["pp"] == 0.25 and scr["mean"] == 2.5 and scr["med"] == 9.0
    assert riser["line"] == "L1" and riser["ppu"] == 1 and riser["mean"] > 10
    assert riser["mean"] <= 12.0          # clipped at +20%
    assert top["mean"] == 10.0            # unchanged deployment, unchanged
    assert mid["doubtful"] is True
    assert n["moved"] == 1 and n["out"] == 1 and n["scratch"] == 2


def test_stale_lines_only_carry_injuries():
    now = datetime(2026, 11, 17, 20, tzinfo=timezone.utc)
    pl = [P("Riser", "OTT", "W", line="L3", toi=13, pptoi=0.2),
          P("Other", "OTT", "W", line="L2")]
    tl = {"OTT": {"updated": now - timedelta(days=4), "players": [
        {"name": "Riser", "group": "f1", "cat": "ev", "inj": "", "gtd": False}]}}
    N.apply_lines(pl, tl, name_keys, now=now)
    assert pl[0]["line"] == "L3" and pl[1]["pp"] == 1.0


def test_promotion_never_lowers():
    now = datetime(2026, 11, 17, 20, tzinfo=timezone.utc)
    pl = [P("Star", "OTT", "C", line="L3", toi=21, pptoi=3),
          P("Norm A", "OTT", "C", line="L1", toi=15, pptoi=1),
          P("Norm B", "OTT", "W", line="L1", toi=15, pptoi=1)]
    tl = {"OTT": {"updated": now, "players": [
        {"name": n, "group": "f1", "cat": "ev", "inj": "", "gtd": False}
        for n in ("Star", "Norm A", "Norm B")]}}
    N.apply_lines(pl, tl, name_keys, now=now)
    assert pl[0]["mean"] >= 10.0


def test_cache_round_trip_and_fallback():
    import tempfile
    N.CACHE_DIR = tempfile.mkdtemp()
    now = datetime(2026, 11, 17, 20, tzinfo=timezone.utc)
    st = {"BOS": {"name": "Michael DiPietro", "status": "Confirmed"}}
    tl = {"BOS": {"updated": now, "players": [
        {"name": "X", "group": "f1", "cat": "ev", "inj": "", "gtd": False}]}}
    N._save("2026-11-17", st, {"BOS": "boston-bruins"}, tl)
    real = N.goalies
    def refuse(day=None):
        raise RuntimeError("HTTP 403")
    N.goalies = refuse
    try:
        s2, slugs, tl2 = N._pull("2026-11-17", {"BOS"})
    finally:
        N.goalies = real
    assert s2 == st and tl2["BOS"]["updated"] == now
