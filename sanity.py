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


def attach(payload: dict, today: str | None = None, log=None) -> dict:
    checks = check(payload, today)
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
