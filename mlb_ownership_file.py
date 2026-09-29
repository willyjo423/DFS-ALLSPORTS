"""Somebody else's projected ownership, read from a CSV.

Drop the export anywhere in the repo and the publisher uses it instead of the
modelled numbers. Nothing else changes.

Why bother
----------
Ownership is half of the number the page ranks lineups by:

    edge = P(reaches the bar) / (1 + field x product of ownerships)

The projection half of that has been graded against ten days of real results.
The ownership half has been graded against exactly one contest, in a different
sport, and never against a baseball field at all. It is the least tested input
in the whole system and it carries equal weight.

A vendor who sells ownership projections has a feedback loop we do not: they
see every contest's real ownership, every day, for years. Using their number
is not giving up on modelling - it is using the better measurement for the
input where ours is weakest, and keeping ours for the input where ours is
tested.

Which column
------------
This does NOT hardcode a column name, because guessing field names has cost
this project several rounds already - seven guesses at a player id found it on
zero of 652 rows before somebody printed the payload. So it looks for likely
names, and if it cannot find one it PRINTS EVERY HEADER IN THE FILE and stops.
One run then answers the question instead of another guess.

The scale
---------
Ten roster slots means ownership across a slate sums to 1000%. That is
arithmetic, not opinion, and the duplication estimate is wrong by exactly the
factor the sum is wrong by. So the file's numbers are rescaled to the roster
demand, and the factor is logged. If it is far from 1.0, the export probably
covers a different slate than the board.
"""
from __future__ import annotations

import csv
import logging
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("mlb_ownership_file")

# Header names seen on DFS ownership exports, lowercased and stripped of
# punctuation. Ordered: the most specific wins, so a file carrying both
# "ownership" and "projected ownership" uses the projection.
NAME_HEADERS = ["player", "player name", "name", "playername", "fullname"]
OWN_HEADERS = ["projected ownership", "proj ownership", "ownership",
               "own", "pown", "proj own", "ownership projection",
               "projownership", "drafted", "drafted"]
TEAM_HEADERS = ["team", "tm", "teamabbrev", "team abbrev"]

_SUFFIX = re.compile(r"\s+(jr|sr|ii|iii|iv|v)\.?$")
_PUNCT = re.compile(r"[^a-z0-9 ]+")


def norm(name: str) -> str:
    """The same normalisation the rest of this project uses.

    A period becomes a SPACE, not nothing - "St.Brown" and "St. Brown" have to
    meet, and deleting the period makes "stbrown" and "st brown", which never
    will.
    """
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace(".", " ").replace("-", " ").replace("'", "")
    s = _PUNCT.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return _SUFFIX.sub("", s).strip()


def _key(h: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", str(h or "").strip().lower())


def _pick(headers: list[str], wanted: list[str]) -> str | None:
    keys = {_key(h): h for h in headers}
    for w in wanted:
        if w in keys:
            return keys[w]
    # Then a contains-match, longest header first so "projected ownership"
    # beats "ownership" when both are present.
    for w in wanted:
        hits = sorted((h for k, h in keys.items() if w in k),
                      key=len, reverse=True)
        if hits:
            return hits[0]
    return None


def find_file(explicit: str | None = None) -> Path | None:
    """The ownership CSV, wherever it was dropped."""
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for pattern in ("ownership/*.csv", "ownership*.csv", "*ownership*.csv",
                    "*.csv"):
        hits = [p for p in sorted(Path(".").glob(pattern))
                if "contest-standings" not in p.name]
        if hits:
            return hits[0]
    return None


def read(path: Path) -> dict[str, float]:
    """name -> ownership as a FRACTION, or {} with a loud explanation."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        log.error("%s has no rows", path)
        return {}

    headers = list(rows[0].keys())
    name_col = _pick(headers, NAME_HEADERS)
    own_col = _pick(headers, OWN_HEADERS)
    team_col = _pick(headers, TEAM_HEADERS)

    if not name_col or not own_col:
        log.error(
            "could not find a %s column in %s.\n"
            "Every header in that file:\n  %s\n"
            "Tell me which one holds the ownership and I will add it - "
            "guessing field names has cost this project whole rounds before.",
            "player-name" if not name_col else "ownership", path,
            "\n  ".join(headers))
        return {}

    log.info("%s: using '%s' for the name and '%s' for the ownership%s",
             path.name, name_col, own_col,
             f" ('{team_col}' for team)" if team_col else "")

    out: dict[str, float] = {}
    bad = 0
    for r in rows:
        who = norm(r.get(name_col))
        raw = str(r.get(own_col) or "").strip().rstrip("%").replace(",", "")
        if not who or not raw:
            continue
        try:
            v = float(raw)
        except ValueError:
            bad += 1
            continue
        out[who] = v

    if not out:
        log.error("%s: no usable ownership values in column '%s'",
                  path, own_col)
        return {}

    # Percent or fraction, decided by the data rather than by assumption. A
    # slate of ten-slot lineups sums to 1000% - or 10.0 as fractions - so the
    # median tells the two apart without ambiguity.
    med = float(np.median(list(out.values())))
    if med > 1.5:
        out = {k: v / 100.0 for k, v in out.items()}
        log.info("%s: values look like percentages (median %.1f), divided "
                 "by 100", path.name, med)
    else:
        log.info("%s: values look like fractions (median %.3f)",
                 path.name, med)

    if bad:
        log.warning("%s: %d rows had an unreadable ownership value", path, bad)
    log.info("%s: %d players", path.name, len(out))
    return out


def apply(pool: pd.DataFrame, own: pd.Series, table: dict[str, float],
          slots: int) -> pd.Series:
    """Replace the modelled ownership with the file's, where it has an answer.

    Players the file does not mention keep the modelled number rather than
    dropping to zero. A missing row means "this vendor did not cover him",
    not "nobody will roster him", and zeroing him would make every lineup
    containing him look falsely contrarian - which is exactly the direction
    that produces a confident bad recommendation.
    """
    if not table:
        return own

    keys = pool["name"].map(norm)
    hit = keys.map(lambda k: table.get(k, np.nan))
    matched = int(hit.notna().sum())
    log.info("file ownership matched %d of %d players on the board (%.0f%%)",
             matched, len(pool), 100 * matched / max(len(pool), 1))

    if matched == 0:
        log.error("NOT ONE name matched. The board and the file are almost "
                  "certainly different slates, or the name column is wrong. "
                  "Keeping the modelled ownership.\n"
                  "  board:  %s\n  file:   %s",
                  ", ".join(sorted(pool['name'].head(4))),
                  ", ".join(sorted(list(table)[:4])))
        return own
    if matched < 0.5 * len(pool):
        log.warning("fewer than half the board matched. The missing players "
                    "keep their modelled ownership, so the two sources are "
                    "mixed - check the file covers this slate.")

    merged = own.copy()
    merged[hit.notna()] = hit[hit.notna()].astype(float)

    # Ten slots means the board sums to 1000%. Arithmetic, and the
    # duplication estimate is wrong by exactly the factor this is wrong by.
    total = float(merged.sum())
    want = float(slots)
    if total > 0:
        factor = want / total
        log.info("ownership sums to %.0f%% against the %d roster slots "
                 "(%.0f%%), a factor of %.3f",
                 100 * total, slots, 100 * want, factor)
        if 0.5 <= factor <= 2.0:
            merged = merged * factor
        else:
            # Refused, not applied. A factor this far out means the file and
            # the board disagree about what slate this is, and multiplying
            # every player by it would not fix that - it would spread one
            # bad assumption across the whole board while making the sum
            # look correct. Better a scale that is visibly wrong than one
            # that is invisibly wrong.
            log.error("NOT rescaling by %.3f - that is too far from 1.0 to "
                      "be a rounding difference. The file probably covers a "
                      "different slate. The vendor's numbers are used as "
                      "given, so the duplication estimate will be off by "
                      "roughly this factor.", factor)

    # Nobody is owned by more than everybody.
    return merged.clip(lower=0.0, upper=1.0)
