"""projection_calibration.apply: calibrated where fitted, untouched elsewhere.

Runs as a script: `python test_projection_calibration.py`. No network.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import projection_calibration as PC

FAILS = []


def check(cond, what):
    print(("  ok    " if cond else "  FAIL  ") + what)
    if not cond:
        FAILS.append(what)


with tempfile.TemporaryDirectory() as d:
    PC.ROOT = Path(d)
    (Path(d) / "data").mkdir()
    (Path(d) / "data" / "proj_calibration_nhl.json").write_text(json.dumps(
        {"positions": {"W": {"a": -0.3, "b": 0.6, "c": 0.9}}}))
    pay = {"sport": "nhl", "players": [
        {"pos": "W", "salary": 8000, "med": 13.0, "pp": 0.99},
        {"pos": "W", "salary": 8000, "med": 13.0, "pp": 0.20, "mean": 3.0},
        {"pos": "G", "salary": 8000, "med": 15.0, "mean": 16.2},
        {"pos": "G", "salary": 8000, "med": 15.0}]}
    PC.apply(pay)
    p = [x["proj"] for x in pay["players"]]
    check(abs(p[0] - (-0.3 + 0.6 * 8 + 0.9 * 13)) < 0.01, f"winger calibrated ({p[0]})")
    check(p[1] == 3.0, "a probable scratch keeps the model's own expectation")
    check(p[2] == 16.2, "uncalibrated position uses the model mean")
    check(p[3] == 15.0, "and the median when no mean is published")
    pay = {"sport": "cfb", "players": [{"pos": "QB", "salary": 8000, "med": 20.0}]}
    PC.apply(pay)
    check(pay["players"][0]["proj"] == 20.0, "a sport with no fit is left as the model made it")

print()
if FAILS:
    print(f"{len(FAILS)} FAILED")
    sys.exit(1)
print("all passed")
