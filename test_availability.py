"""availability.py: who gets dropped, and who must never be.

Runs as a script: `python test_availability.py`. No network.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

import availability as A

FAILS = []


def check(cond, what):
    print(("  ok    " if cond else "  FAIL  ") + what)
    if not cond:
        FAILS.append(what)


def board(**status):
    names = list(status)
    return pd.DataFrame({"dk_player_id": [str(100 + i) for i in range(len(names))],
                         "name": names, "team": "EDM",
                         "dk_status": list(status.values())})


print("reading the DraftKings tag")
check(A.dk_status({"i": "O"}) == "O", "lobby key `i`")
check(A.dk_status({"status": "IR"}) == "IR", "draftables key `status`")
check(A.dk_status({"status": "None"}) == "", "'None' means healthy")
check(A.dk_status({"i": "", "status": "q"}) == "Q", "first non-empty wins, upper-cased")
check(A.dk_status({}) == "", "no key at all is healthy, not out")

print("live tags at publish time")
with tempfile.TemporaryDirectory() as d:
    A.DATA = Path(d)
    b = A.available(board(McDavid="", Hurt="O", Gone="IR", Maybe="Q"), "nhl")
    check(list(b["name"]) == ["McDavid", "Maybe"],
          "O and IR dropped, Q kept, healthy kept")

    print("the morning file")
    doc = {"sport": "nhl", "date": A.today_et(),
           "out": [{"dk_player_id": "101", "name": "Sent Down",
                    "reason": "not on an NHL roster"}]}
    (Path(d) / "availability_nhl.json").write_text(json.dumps(doc))
    b = A.available(board(McDavid="", **{"Sent Down": ""}), "nhl")
    check(list(b["name"]) == ["McDavid"], "today's file drops by DK id")

    doc["date"] = "2000-01-01"
    (Path(d) / "availability_nhl.json").write_text(json.dumps(doc))
    b = A.available(board(McDavid="", **{"Sent Down": ""}), "nhl")
    check(len(b) == 2, "a file from another day is ignored")

    (Path(d) / "availability_nhl.json").write_text("{not json")
    b = A.available(board(McDavid=""), "nhl")
    check(len(b) == 1, "a corrupt file is ignored, not fatal")

    b = A.available(pd.DataFrame({"dk_player_id": ["1"], "name": ["X"]}), "cfb")
    check(len(b) == 1, "a board with no dk_status column passes through")

print("names")
check(bool(A.name_keys("Tim Stuetzle") & A.name_keys("Tim Stützle")),
      "Stuetzle (DraftKings) matches Stützle (NHL)")
check(not (A.name_keys("Connor McDavid") & A.name_keys("Connor Bedard")),
      "different surnames never match")

print("fetch(): the NHL morning file, offline")
import nhl_data as ND  # noqa: E402
with tempfile.TemporaryDirectory() as d:
    A.DATA = Path(d)
    rows = [{"pid": 1, "fn": "Connor", "ln": "McDavid", "s": 9000, "pn": "C",
             "tid": 1, "htid": 1, "atid": 2, "htabbr": "EDM", "atabbr": "VAN",
             "IsDisabledFromDrafting": False, "ppg": 5, "i": ""},
            {"pid": 2, "fn": "Long", "ln": "Term", "s": 2500, "pn": "W",
             "tid": 1, "htid": 1, "atid": 2, "htabbr": "EDM", "atabbr": "VAN",
             "IsDisabledFromDrafting": False, "ppg": 1, "i": "IR"},
            {"pid": 3, "fn": "Sent", "ln": "Down", "s": 2500, "pn": "D",
             "tid": 2, "htid": 1, "atid": 2, "htabbr": "EDM", "atabbr": "VAN",
             "IsDisabledFromDrafting": False, "ppg": 1, "i": ""},
            {"pid": 4, "fn": "Maybe", "ln": "Plays", "s": 4000, "pn": "C",
             "tid": 2, "htid": 1, "atid": 2, "htabbr": "EDM", "atabbr": "VAN",
             "IsDisabledFromDrafting": False, "ppg": 3, "i": "Q"}]
    ND._draft_rows = lambda dg: (rows, "lobby")
    b = ND.board(1)
    check(list(b["dk_status"]) == ["", "IR", "", "Q"],
          "nhl_data.board() carries the DK tag through")
    ND.slates = lambda *a, **k: pd.DataFrame({"draft_group": [1]})
    A._nhl_roster_names = lambda: ({"connor mcdavid", "long term",
                                    "maybe plays"}, 32)
    doc = A.fetch("nhl")
    out = {r["name"]: r["reason"] for r in doc["out"]}
    check(out == {"Long Term": "DK IR", "Sent Down": "not on an NHL roster"},
          "IR from DraftKings, sent-down from the NHL roster, Q kept")
    check([q["name"] for q in doc["questionable"]] == ["Maybe Plays"],
          "questionable reported, not dropped")
    A._nhl_roster_names = lambda: (set(), 5)
    doc = A.fetch("nhl")
    check([r["name"] for r in doc["out"]] == ["Long Term"],
          "a roster check that mostly failed drops nobody")
    b = A.available(ND.board(1), "nhl")
    check(list(b["name"]) == ["Connor McDavid", "Sent Down", "Maybe Plays"],
          "publish-time filter: live IR dropped, unverified player kept")

print()
if FAILS:
    print(f"{len(FAILS)} FAILED")
    sys.exit(1)
print("all passed")
