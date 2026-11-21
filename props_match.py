"""Which sportsbook prop line belongs to which DraftKings player.

2026-11-20: Chaney Johnson (BKN, $3,100) was matched to Cameron Johnson's
(DEN) props through the shared first-initial-plus-surname key "c johnson".
The props said 19.4 against our model's 8.2, the blend took the props, and a
bench player who did not play was a top value, 32% owned, and in 6 of the top
13 lineups. The same thing happened on 2026-11-18.

So a line is attached only on:
  - the same full name (normalised), and if two players share it, the same
    team; or
  - a looser spelling (initial + surname, transliteration) AND the same team.
Anything else is not attached.
"""
from __future__ import annotations

from availability import name_keys
from nhl_data import normalise


def build(props: dict) -> list[tuple]:
    out = []
    for key, stats in props.items():
        nm, tm = key.rsplit("|", 1)
        out.append((normalise(nm), name_keys(nm), tm, stats))
    return out


def match(index: list[tuple], name: str, team: str, canon=lambda t: t):
    full = normalise(name)
    t = canon(team)
    exact = [e for e in index if e[0] == full]
    if len(exact) == 1:
        e = exact[0]
        return e[3]
    if len(exact) > 1:
        same = [e for e in exact if canon(e[2]) == t]
        return same[0][3] if len(same) == 1 else None
    keys = name_keys(name)
    loose = [e for e in index if e[1] & keys and e[2] and canon(e[2]) == t]
    return loose[0][3] if len(loose) == 1 else None
