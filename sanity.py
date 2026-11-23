"""Pre-lock sanity checks, run on every board just before it is written.

Each of the last few bad nights had a warning sign that was visible on the
board and that nobody was looking for:

* 2026-11-18/20, NBA: Chaney Johnson carried Cameron Johnson's props (19.4
  against a model of about 8), was 32% owned in our own estimate and sat in 6
  of the top 13 lineups. He did not play. Props more than double the model
  is rare, and when it happens it is either a real role change or bad data.
* 2026-11-20, NBA: Ajay Mitchell was in 95 of 120 lineups and scored 17
  against 37.6. One player in most of the pool means one bust sinks it.
* 2026-11-20, CFB: Air Force (-22.5) were 63% Medders; starters sat in a 52-3.

The checks do not change any projection. They put a warning box above the
lineups and a CHECK tag on the player, so the five minutes before lock go to
the two or three names that actually need a look.
"""
from __future__ import annotations

import math

GAP_RATIO = 2.0          # props vs model, either direction
POOL_SHARE = 0.70        # share of the lineup pool one player is in
CHALK_OWN = 0.25
NBA_MIN_MINUTES = 20.0
CFB_BLOWOUT = 0.08
TOP_N = 13               # the lineups anyone actually plays


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _is_chalk(p) -> bool:
    own = _num(p.get("own")) or 0.0
    return bool(p.get("chalk")) or own >= CHALK_OWN


def check(payload: dict, today: str | None = None) -> list[dict]:
    sport = payload.get("sport", "")
    players = payload.get("players") or []
    lineups = payload.get("lineups") or []
    out: list[dict] = []

    counts = [0] * len(players)
    top = [0] * len(players)
    for i, lu in enumerate(lineups):
        for j in lu.get("p", []):
            if 0 <= j < len(players):
                counts[j] += 1
                if i < TOP_N:
                    top[j] += 1

    def add(j, kind, msg):
        out.append({"i": j, "name": players[j]["name"], "kind": kind,
                    "msg": msg, "top": top[j]})

    for j, p in enumerate(players):
        used = counts[j] > 0
        model, props = _num(p.get("model")), _num(p.get("props"))
        if model and props and model > 1 and props > 1:
            r = props / model
            if (r >= GAP_RATIO or r <= 1 / GAP_RATIO) and (used or _is_chalk(p)):
                add(j, "gap", f"props {props:.1f} vs model {model:.1f} "
                    f"({r:.1f}x) — confirm he is playing and it is the right player")

        if lineups and counts[j] / len(lineups) > POOL_SHARE:
            add(j, "pool", f"in {counts[j]} of {len(lineups)} lineups — "
                "one bust sinks most of the pool; cap his exposure")

        if not _is_chalk(p) and not used:
            continue
        if p.get("status") not in (None, "clear") and used:
            add(j, "status", f"status {p.get('status')} but in lineups")
        if sport == "nba":
            mins = _num(p.get("min"))
            if mins is not None and mins < NBA_MIN_MINUTES and _is_chalk(p):
                add(j, "role", f"chalk on {mins:.0f} projected minutes — "
                    "check he is in the rotation tonight")
        elif sport == "nhl":
            if p.get("pos") == "G" and top[j] and str(p.get("gst") or "").lower() not in ("confirmed", "backup"):
                add(j, "goalie", "starter not confirmed — check before lock")
        elif sport == "cfb":
            blow = _num(p.get("blow"))
            if blow and blow >= CFB_BLOWOUT and _is_chalk(p):
                add(j, "blowout", f"big favourite (spread {p.get('spr')}): "
                    f"projection cut {100 * blow:.0f}% for sitting in a blowout")

    # one row per player, the most important reason first
    order = {"gap": 0, "status": 1, "role": 2, "goalie": 3, "pool": 4, "blowout": 5}
    out.sort(key=lambda c: (order.get(c["kind"], 9), -c["top"]))
    return out


# ------------------------------------------------------------- legality
# 2026-11-22 had three bad lineups reach the page: Jordan Hudson (no role,
# NFL), and an NBA lineup holding Tre Jones twice after a line-level git merge
# spliced two pools. These checks run on every board before it is written:
# an illegal lineup, or one holding a player who cannot score, never publishes.
FLEX_SLOTS = {"UTIL", "FLEX", "SFLEX"}   # NBA G/F checked by eligibility instead


def _legal(L: dict, players: list, roster: dict) -> str | None:
    slots = list(roster.get("slots") or [])
    p = L.get("p")
    if not isinstance(p, list) or (slots and len(p) != len(slots)):
        return "wrong size"
    if len(set(p)) != len(p):
        return "player twice"
    if any(not isinstance(i, int) or i < 0 or i >= len(players) for i in p):
        return "unknown player"
    if len({players[i]["name"] for i in p}) != len(p):
        return "player twice"
    cap = roster.get("salary_cap")
    if cap and sum(int(players[i].get("salary") or 0) for i in p) > cap:
        return "over the cap"
    for i in p:
        q = players[i]
        if str(q.get("status") or "clear") == "out" or float(q.get("pp") if q.get("pp") is not None else 1) <= 0:
            return f"{q['name']} is out"
    s = L.get("s")
    if s and len(s) == len(p) and any(players[i].get("elig") for i in p):
        if any(players[i].get("elig") and sl not in players[i]["elig"] for i, sl in zip(p, s)):
            return "player in a slot he cannot fill"
    elif slots and not any(players[i].get("elig") for i in p):
        # position counts: every fixed slot filled, extras only where a flex allows
        need = {}
        for sl in slots:
            if sl not in FLEX_SLOTS:
                need[sl] = need.get(sl, 0) + 1
        have = {}
        for i in p:
            have[str(players[i].get("pos"))] = have.get(str(players[i].get("pos")), 0) + 1
        flex_pos = set(roster.get("flex_positions") or [])
        n_flex = sum(1 for sl in slots if sl in FLEX_SLOTS)
        extra = 0
        for pos, c in have.items():
            r = need.get(pos, 0)
            if c < r:
                return f"too few {pos}"
            if c > r:
                if pos not in flex_pos:
                    return f"too many {pos}"
                extra += c - r
        if any(have.get(pos, 0) < r for pos, r in need.items()):
            return "a slot left empty"
        if extra != n_flex:
            return "flex slots do not add up"
        for pos, m in (roster.get("max_position") or {}).items():
            if have.get(pos, 0) > m:
                return f"too many {pos}"
    return None


def enforce(payload: dict, log=None) -> dict:
    players = payload.get("players") or []
    roster = payload.get("roster") or {}
    keep, why = [], {}
    for L in payload.get("lineups") or []:
        r = _legal(L, players, roster) if isinstance(L, dict) else "not a lineup"
        if r:
            why[r] = why.get(r, 0) + 1
        else:
            keep.append(L)
    if why:
        payload["lineups"] = keep
        if log is not None:
            log.warning("lineup check dropped %d lineups from %s: %s", sum(why.values()),
                        payload.get("label", ""), why)
    # ownership estimates pinned at the cap: the estimate ran out of room
    own = [float(p.get("own") or 0) for p in players]
    top = max(own) if own else 0
    pinned = [p["name"] for p, o in zip(players, own) if top >= 0.5 and abs(o - top) < 1e-6]
    payload["integrity"] = {"dropped": sum(why.values()), "reasons": why,
                            "own_pinned": pinned if len(pinned) >= 3 else []}
    return payload


UNLIKELY_PP = 0.5        # under this chance to play is "probably not playing"
UNLIKELY_SHARE = 0.10    # most of the pool such lineups may take
TOP_CLEAR = 13           # and never inside the lineups anyone actually plays


def cap_unlikely(payload: dict, log=None) -> dict:
    """Lineups built on a player who probably will not play.

    The audit found a 25%-to-play skater (Hagel, 2026-11-17) in 115 of 258
    lineups: his conditional upside kept winning simulations he would not be
    in. Now: such lineups are marked `unl`, at most 10% of the pool keeps
    them (the best by top-1%), and the page and the settings optimiser sort
    them after every other lineup, so none reaches the first 13 picked.
    A confirmation flips his chance to play and the late-news rebuild
    brings him back in full."""
    P = payload.get("players") or []
    L = payload.get("lineups") or []
    if not L:
        return payload
    bad = {j for j, p in enumerate(P)
           if (_num(p.get("pp")) if p.get("pp") is not None else 1.0) < UNLIKELY_PP
           and str(p.get("status") or "clear").lower() != "out"}
    if not bad:
        return payload
    risky = [x for x in L if any(i in bad for i in x.get("p", []))]
    if not risky:
        return payload
    keep_n = int(UNLIKELY_SHARE * len(L))
    risky.sort(key=lambda x: -(_num(x.get("top1")) or 0))
    kept = set(map(id, risky[:keep_n]))
    out = []
    for x in L:
        if any(i in bad for i in x.get("p", [])):
            if id(x) not in kept:
                continue
            x["unl"] = True
        out.append(x)
    payload["lineups"] = out
    if log is not None:
        log.warning("unlikely players (%s): %d lineups used them, kept %d of %d",
                    ", ".join(P[j]["name"] for j in sorted(bad))[:200],
                    len(risky), min(keep_n, len(risky)), len(L))
    return payload


def attach(payload: dict, today: str | None = None, log=None) -> dict:
    enforce(payload, log)
    cap_unlikely(payload, log)
    checks = check(payload, today)
    pin = (payload.get("integrity") or {}).get("own_pinned") or []
    if pin:
        idx = {p["name"]: j for j, p in enumerate(payload.get("players") or [])}
        j = idx[pin[0]]
        checks.append({"i": j, "name": ", ".join(pin), "kind": "own",
                       "msg": f"ownership estimate hit its {100 * float(payload['players'][j].get('own') or 0):.0f}% ceiling for "
                              f"{len(pin)} players — their real ownership is likely different; don't lean on these numbers",
                       "top": 0})
    payload["checks"] = checks
    tags: dict[int, list[str]] = {}
    for c in checks:
        tags.setdefault(c["i"], []).append(c["msg"])
    for j, msgs in tags.items():
        payload["players"][j]["chk"] = "; ".join(msgs)
    if log is not None and checks:
        log.warning("pre-lock checks for %s: %s", payload.get("label", ""),
                    " | ".join(f"{c['name']}: {c['msg']}" for c in checks))
    return payload
