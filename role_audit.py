"""Role audit: who in our lineups barely played, and did we have a warning?

Run inside the nightly grade, on the board as it was at lock. For every player
our lineups used (or the field was expected to play heavily), it compares the
role we projected with the role he got:

* NBA   - did not play, or played under half his projected minutes
          (projected 12+).
* NHL   - skater missing from the box score (scratched), or under 60% of his
          projected ice time; a goalie we projected to start who did not.
* NFL / college football use the same record shape (snaps), from their own
          weekly grade.

Each miss records what the board knew at lock - status, chance of playing,
the news/role note, games in the last 30 days, line, whether props were
posted, any pre-lock flag - and is sorted into:

* "warned": the board carried a signal and the lineups used him anyway.
  The fix is to act on that signal (cap or drop), not to find a new one.
* "blind":  nothing on the board pointed at it. These are the cases that
  need a new filter, and the running file shows which kinds repeat.

The running file (docs/data/grades/role_audit_<sport>.json) keeps the last
60 nights so a pattern - the same team, the same reason, the same kind of
player - is visible instead of one bad night at a time.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

NBA_MIN_PROJ = 12.0       # minutes projection worth auditing
NBA_CUT = 0.5             # under half the projected minutes is a role miss
NHL_TOI_CUT = 0.6         # under 60% of projected ice time
CHALK_OWN = 0.20          # audit the field's chalk even if we avoided it
KEEP_NIGHTS = 60


def _f(v, d=None):
    try:
        v = float(v)
        return v if v == v else d
    except (TypeError, ValueError):
        return d


def signals(p: dict, flagged: set) -> list[str]:
    """Warnings the board carried for this player at lock."""
    s = []
    st = str(p.get("status") or "clear").lower()
    if st not in ("clear", ""):
        s.append(f"status {st}")
    pp = _f(p.get("pp"), 1.0)
    if pp < 0.95:
        s.append(f"chance to play {pp:.0%}")
    if p.get("doubtful"):
        s.append("doubtful tag")
    news = str(p.get("news") or "")
    for w in ("day-to-day", "questionable", "game-time", "likely scratch", "not in"):
        if w in news.lower():
            s.append(f"news: {w}")
            break
    gp30 = p.get("gp30")
    if gp30 is not None and _f(gp30, 99) < 6:
        s.append(f"{int(gp30)} games in 30 days")
    if p.get("props") is None and p.get("model") is not None:
        s.append("no props posted")
    if p.get("name") in flagged:
        s.append("pre-lock flag")
    return s


def _exposure(P, L):
    n = [0] * len(P)
    top = [0] * len(P)
    for k, lu in enumerate(L):
        for i in lu.get("p", []):
            if 0 <= i < len(P):
                n[i] += 1
                if k < 13:
                    top[i] += 1
    return n, top


def audit(sport: str, board: dict) -> list[dict]:
    """Players whose role collapsed. Each player in board['players'] must
    already carry `_found` (in the box score) and, for NBA, `_min`; for NHL,
    `_toi` (minutes) and `_started` (goalies)."""
    P, L = board.get("players") or [], board.get("lineups") or []
    flagged = set()
    for c in board.get("checks") or []:
        for nm in str(c.get("name") or "").split(", "):
            flagged.add(nm)
    n, top = _exposure(P, L)
    out = []
    for i, p in enumerate(P):
        own = _f(p.get("own"), 0.0)
        if n[i] == 0 and own < CHALK_OWN:
            continue
        if str(p.get("status") or "").lower() == "out":
            continue                    # the board already had him out
        reason, proj_role, act_role = None, None, None
        if sport == "nba":
            pm, am = _f(p.get("min")), _f(p.get("_min"), 0.0)
            if not p.get("_found") or am == 0:
                reason = "did not play"
            elif pm and pm >= NBA_MIN_PROJ and am < NBA_CUT * pm:
                reason = "minutes cut"
            proj_role, act_role = pm, am
        elif sport == "nhl":
            if p.get("pos") == "G":
                if _f(p.get("pp"), 1.0) >= 0.5 and not p.get("_started"):
                    reason = "goalie did not start"
                proj_role, act_role = _f(p.get("pp")), 1.0 if p.get("_started") else 0.0
            else:
                pt, at = _f(p.get("toi")), _f(p.get("_toi"), 0.0)
                if not p.get("_found"):
                    reason = "scratched"
                elif pt and at < NHL_TOI_CUT * pt:
                    reason = "ice time cut"
                proj_role, act_role = pt, at
        else:                           # football: snaps share
            ps, as_ = _f(p.get("snap_proj")), _f(p.get("_snaps"), 0.0)
            if not p.get("_found") or as_ == 0:
                reason = "did not play"
            elif ps and as_ < 0.5 * ps:
                reason = "snaps cut"
            proj_role, act_role = ps, as_
        if not reason:
            continue
        sig = signals(p, flagged)
        out.append({
            "name": p.get("name"), "team": p.get("team"), "pos": p.get("pos"),
            "salary": p.get("salary"), "proj": _f(p.get("proj")),
            "act": round(_f(p.get("_act"), 0.0), 2), "reason": reason,
            "role_proj": proj_role, "role_act": act_role,
            "lineups": n[i], "of": len(L), "top13": top[i], "own": own,
            "signals": sig, "kind": "warned" if sig else "blind",
            "line": p.get("line"), "news": p.get("news"),
        })
    out.sort(key=lambda m: (-m["top13"], -m["lineups"], -(m["own"] or 0)))
    return out


def update_running(path: Path, day: str, misses: list[dict]) -> dict:
    """Append tonight and re-summarise the last KEEP_NIGHTS nights."""
    try:
        doc = json.loads(path.read_text())
    except Exception:                                          # noqa: BLE001
        doc = {"nights": {}}
    doc["nights"][day] = misses
    keep = sorted(doc["nights"])[-KEEP_NIGHTS:]
    doc["nights"] = {d: doc["nights"][d] for d in keep}
    allm = [m for d in keep for m in doc["nights"][d]]
    used = [m for m in allm if m["lineups"] > 0]
    sig = Counter(s.split(" ")[0] if s.startswith(("status", "chance")) else s
                  for m in used for s in m["signals"])
    doc["summary"] = {
        "nights": len(keep),
        "misses_in_lineups": len(used),
        "blind": sum(m["kind"] == "blind" for m in used),
        "warned": sum(m["kind"] == "warned" for m in used),
        "by_reason": dict(Counter(m["reason"] for m in used).most_common()),
        "warning_ignored": dict(sig.most_common(10)),
        "repeat_players": [k for k, v in Counter(m["name"] for m in used).most_common(10) if v > 1],
        "repeat_teams": [k for k, v in Counter(m["team"] for m in used).most_common(8) if v > 2],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1))
    return doc["summary"]


def report(sport: str, day: str, misses: list[dict], summary: dict) -> list[str]:
    used = [m for m in misses if m["lineups"] > 0]
    lines = [f"- role audit: {len(used)} lineup player(s) lost their role"
             + (f" ({sum(m['kind'] == 'blind' for m in used)} with no warning)" if used else "")]
    for m in used[:8]:
        rp = "" if m["role_proj"] is None else f" {m['role_proj']:.0f}->{m['role_act']:.0f}"
        lines.append(f"  - {m['name']} ({m['team']}) {m['reason']}{rp}; in {m['lineups']}/{m['of']} "
                     f"lineups, top-13 {m['top13']}; "
                     + (("warned: " + "; ".join(m["signals"])) if m["signals"] else "BLIND"))
    s = summary
    lines.append(f"- running ({s['nights']} nights): {s['misses_in_lineups']} misses, "
                 f"{s['blind']} blind / {s['warned']} warned; reasons {s['by_reason']}; "
                 f"warnings we ignored {s['warning_ignored']}")
    return lines
