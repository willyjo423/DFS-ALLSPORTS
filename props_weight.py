"""The props blend weight, learned from the nightly grades.

grade_nightly.py records, every night, the weight the props deserved:
the regression of (actual - model) on (props - model). Pooled over nights in
docs/data/grades/<sport>/summary.json it is the best single weight so far.
It is used once there are MIN_NIGHTS of it, shrunk toward the default by a
prior worth PRIOR_PLAYERS players, and kept inside sensible bounds.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("props_weight")
HERE = Path(__file__).resolve().parent
MIN_NIGHTS = 10
PRIOR_PLAYERS = 300
BOUNDS = (0.4, 0.95)


def weight(sport: str, default: float) -> float:
    f = HERE / "docs" / "data" / "grades" / sport / "summary.json"
    try:
        s = json.loads(f.read_text())
    except Exception:                                          # noqa: BLE001
        return default
    w, n, nights = s.get("props_weight"), s.get("props_players") or 0, s.get("props_nights") or 0
    if w is None or nights < MIN_NIGHTS:
        return default
    learned = (n * float(w) + PRIOR_PLAYERS * default) / (n + PRIOR_PLAYERS)
    learned = min(max(learned, BOUNDS[0]), BOUNDS[1])
    log.info("%s props weight %.2f (learned from %d nights, %d players)", sport, learned, nights, n)
    return learned
