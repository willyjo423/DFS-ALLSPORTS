"""Who is actually available today, for every sport, from official sources.

Runs once a morning (see .github/workflows/availability.yml, 6am Eastern) and
writes data/availability_<sport>.json. Each publish then drops those players
from its board before projecting, so a man who is out for the year cannot be
the best points-per-dollar on the slate.

This file decides WHO IS ON THE BOARD. It never touches a projection: every
number the optimiser uses is still your own.

Sources, all free, no key, no login
-----------------------------------
* DraftKings' own injury tag on every priced player (O, IR, IL, SUSP ...).
  Read for all three sports, and read LIVE again at publish time, so a player
  ruled out after 6am is still caught by the evening run.
* NHL: every team's current roster (api-web.nhle.com). A priced player on no
  NHL roster has been sent down, released or traded out of the league.
* MLB: every team's 40-man roster status (statsapi.mlb.com), the same check
  mlb_publish already makes, done up front.
* CFB: college football has no official injury report, so DraftKings' tag is
  all there is.

Never drops a player it merely failed to hear about. A roster that did not
load proves nothing, and deleting a man on the strength of silence is how a
filter empties a board.

    python availability.py                 # all three sports
    python availability.py --sports nhl    # one
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

log = logging.getLogger("availability")

EASTERN = ZoneInfo("America/New_York")
DATA = Path(__file__).resolve().parent / "data"
SPORTS = ("nhl", "mlb", "cfb")

# DraftKings tags that mean "will not play". Anything else - Q, GTD, DTD, P,
# D - is uncertain, kept on the board, and only reported.
OUT_CODES = {"O", "OUT", "IR", "IL", "IL10", "IL15", "IL60", "LTIR", "NA",
             "SUSP", "SUS", "PUP", "NFI", "DNR", "EXP", "RET"}
DOUBT_CODES = {"Q", "GTD", "DTD", "D", "P", "DOUBTFUL", "QUESTIONABLE",
               "DAY-TO-DAY", "PROBABLE"}

# Where DraftKings has been seen to put the tag. The lobby feed abbreviates
# (`i`); the draftables API spells it out. First non-empty one wins. If the
# first live run logs "no injury field found", print a raw row with
# `python nhl_data.py --probe` and add the key here.
STATUS_KEYS = ("i", "injuryStatus", "status", "inj", "InjuryStatus")

# A roster check is only trusted if nearly every team answered. A morning
# where half the league timed out must not turn into half the league "out".
MIN_TEAMS_READ = {"nhl": 28, "mlb": 27}


def today_et() -> str:
    return dt.datetime.now(EASTERN).strftime("%Y-%m-%d")


def dk_status(row: dict) -> str:
    """The injury tag on one raw DraftKings player row, upper-cased, or ''."""
    for k in STATUS_KEYS:
        v = row.get(k) if isinstance(row, dict) else None
        if isinstance(v, str) and v.strip() and v.strip().upper() != "NONE":
            return v.strip().upper()
    return ""


def is_out(code) -> bool:
    return str(code or "").strip().upper() in OUT_CODES


# ------------------------------------------------------------- applying it
def path_for(sport: str) -> Path:
    return DATA / f"availability_{sport}.json"


def load(sport: str, date: str | None = None) -> dict | None:
    """Today's file, or None. A file from another day is ignored on purpose:
    yesterday's injury list applied to today's board is worse than none."""
    p = path_for(sport)
    if not p.exists():
        return None
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        log.warning("%s unreadable (%s) - not applied", p.name, exc)
        return None
    want = date or today_et()
    if doc.get("date") != want:
        log.info("%s is for %s, not %s - not applied", p.name,
                 doc.get("date"), want)
        return None
    return doc


def available(board: pd.DataFrame, sport: str,
              date: str | None = None) -> pd.DataFrame:
    """The board minus everyone known to be out. Logs every name it drops.

    Two reasons, either is enough:
      1. the board's own live DraftKings tag (column `dk_status`) is OUT/IR;
      2. this morning's file lists his DraftKings id.
    """
    if board is None or board.empty:
        return board
    b = board.copy()
    reason = pd.Series("", index=b.index, dtype=object)

    if "dk_status" in b.columns:
        live = b["dk_status"].map(is_out)
        reason[live] = "DK " + b.loc[live, "dk_status"].astype(str)
        doubt = b["dk_status"].astype(str).str.upper().isin(DOUBT_CODES)
        if doubt.any():
            log.info("%s: %d questionable, kept: %s", sport, int(doubt.sum()),
                     ", ".join(f"{n} ({s})" for n, s in
                               zip(b.loc[doubt, "name"],
                                   b.loc[doubt, "dk_status"]))[:400])

    doc = load(sport, date)
    if doc and "dk_player_id" in b.columns:
        why = {str(r["dk_player_id"]): r["reason"]
               for r in doc.get("out", []) if r.get("dk_player_id")}
        ids = b["dk_player_id"].astype(str)
        hit = ids.isin(why) & (reason == "")
        reason[hit] = ids[hit].map(why)

    gone = reason != ""
    if gone.any():
        log.info("%s availability: dropped %d of %d players: %s", sport,
                 int(gone.sum()), len(b),
                 "; ".join(f"{n} [{r}]" for n, r in
                           zip(b.loc[gone, "name"], reason[gone]))[:1500])
    else:
        log.info("%s availability: nobody dropped (%s)", sport,
                 "morning file applied" if doc else "no file for today; live "
                 "DraftKings tags only")
    return b[~gone].reset_index(drop=True)


# ------------------------------------------------------------- fetching it
def _boards(D, sport: str, look: int = 12) -> list[pd.DataFrame]:
    """Every live board DraftKings lists for the sport, as parsed by the
    sport's own data module. One bad board does not stop the rest."""
    listed = D.slates()
    out = []
    for dg in listed["draft_group"].head(look):
        try:
            b = D.board(int(dg))
        except Exception as exc:                               # noqa: BLE001
            log.info("%s draft group %s skipped (%s)", sport, dg,
                     str(exc)[:80])
            continue
        b = b.assign(draft_group=int(dg))
        out.append(b)
    return out


_TRANSLIT = str.maketrans({"ü": "ue", "ö": "oe", "ä": "ae", "Ü": "Ue",
                           "Ö": "Oe", "Ä": "Ae", "ß": "ss", "ø": "o", "Ø": "O",
                           "æ": "ae", "å": "aa"})


def name_keys(name: str) -> set[str]:
    """Every spelling a player's name is likely to arrive in. DraftKings
    writes 'Tim Stuetzle', the NHL 'Tim Stützle': stripping the accent gives
    'stutzle' and the two never meet, which dropped a first-line centre as
    'not on an NHL roster'. So a name is matched on any of: accents stripped,
    German/Nordic transliteration, and first initial + surname."""
    from nhl_data import normalise
    keys = {normalise(name), normalise(str(name).translate(_TRANSLIT))}
    for k in list(keys):
        parts = k.split()
        if len(parts) >= 2:
            keys.add(parts[0][0] + " " + " ".join(parts[1:]))
    return {k for k in keys if k}


def _nhl_roster_names() -> tuple[set, int]:
    import requests
    import nhl_data as ND
    teams_url = "https://api-web.nhle.com/v1/standings/now"
    try:
        st = requests.get(teams_url, timeout=30).json()
        codes = sorted({r["teamAbbrev"]["default"]
                        for r in st.get("standings", [])})
    except Exception as exc:                                   # noqa: BLE001
        log.warning("NHL team list unavailable (%s)", exc)
        return set(), 0
    import time
    names, read = set(), 0
    for c in codes:
        # The NHL's API answers 429 to 32 requests in a burst - the first run
        # read 24 teams and lost the rest. A short pause between teams, and a
        # wait-and-retry on 429, keeps all 32 for about 20 seconds total.
        js = None
        for attempt in range(4):
            try:
                r = requests.get(
                    f"https://api-web.nhle.com/v1/roster/{c}/current",
                    timeout=30)
                if r.status_code == 429:
                    wait = float(r.headers.get("Retry-After") or 5 * (attempt + 1))
                    time.sleep(min(wait, 30))
                    continue
                r.raise_for_status()
                js = r.json()
                break
            except Exception as exc:                           # noqa: BLE001
                log.warning("NHL roster %s unavailable (%s)", c,
                            str(exc)[:60])
                break
        time.sleep(0.6)
        if js is None:
            log.warning("NHL roster %s not read", c)
            continue
        read += 1
        for grp in ("forwards", "defensemen", "goalies"):
            for p in js.get(grp) or []:
                full = (f"{(p.get('firstName') or {}).get('default', '')} "
                        f"{(p.get('lastName') or {}).get('default', '')}")
                names |= name_keys(full)
    log.info("NHL rosters: %d of %d teams read, %d players", read,
             len(codes), len(names))
    return names, read


def _mlb_roster() -> dict | None:
    import mlb_data as MD
    try:
        js = MD._get(f"{MD.STATS}/teams", params={"sportId": 1})
        ids = [t["id"] for t in js.get("teams") or []]
    except Exception as exc:                                   # noqa: BLE001
        log.warning("MLB team list unavailable (%s)", exc)
        return None
    st = MD.roster_status(ids)
    return st


def fetch(sport: str) -> dict:
    """Build today's availability file for one sport and write it."""
    if sport == "nhl":
        import nhl_data as D
    elif sport == "mlb":
        import mlb_data as D
    elif sport == "cfb":
        import cfb_data as D
    else:
        raise ValueError(sport)

    doc = {"sport": sport, "date": today_et(),
           "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(
               timespec="seconds"),
           "sources": [], "out": [], "questionable": [], "notes": []}

    try:
        boards = _boards(D, sport)
    except Exception as exc:                                   # noqa: BLE001
        boards = []
        doc["notes"].append(f"DraftKings lobby unreadable: {str(exc)[:120]}")
    if not boards:
        doc["notes"].append("no live DraftKings boards for this sport today")
        return _write(sport, doc)

    pool = (pd.concat(boards, ignore_index=True)
            .drop_duplicates("dk_player_id"))
    doc["sources"].append(f"draftkings ({len(boards)} boards, "
                          f"{len(pool)} players)")
    out = {}

    # 1. DraftKings' own tag.
    if "dk_status" in pool.columns and pool["dk_status"].astype(bool).any():
        for r in pool.itertuples(index=False):
            code = str(getattr(r, "dk_status", "") or "")
            if is_out(code):
                out[str(r.dk_player_id)] = (r.name, r.team, f"DK {code}")
            elif code.upper() in DOUBT_CODES:
                doc["questionable"].append(
                    {"name": r.name, "team": r.team, "status": code})
    else:
        doc["notes"].append(
            "no DraftKings injury field found on any player - see "
            "availability.STATUS_KEYS")

    # 2. The league's own rosters.
    if sport == "nhl":
        import nhl_data as ND
        names, read = _nhl_roster_names()
        if read >= MIN_TEAMS_READ["nhl"]:
            doc["sources"].append(f"nhl rosters ({read} teams)")
            gone = ~pool["name"].map(lambda nm: bool(name_keys(nm) & names))
            for r in pool[gone].itertuples(index=False):
                out.setdefault(str(r.dk_player_id),
                               (r.name, r.team, "not on an NHL roster"))
        else:
            doc["notes"].append(f"only {read} NHL rosters read - roster "
                                f"check skipped, nobody dropped by it")
    elif sport == "mlb":
        import mlb_data as MD
        st = _mlb_roster()
        if st and int(st.get("teams_read") or 0) >= MIN_TEAMS_READ["mlb"]:
            doc["sources"].append(f"mlb 40-man ({st['teams_read']} teams)")
            ids = pool["mlb_id"].astype(str).str.replace(r"\.0$", "",
                                                          regex=True)
            nm = pool["name"].map(MD.normalise)
            seen = ids.isin(st["seen_ids"]) | nm.isin(st["seen_names"])
            act = ids.isin(st["active_ids"]) | nm.isin(st["active_names"])
            for r in pool[seen & ~act].itertuples(index=False):
                out.setdefault(str(r.dk_player_id),
                               (r.name, r.team, "not active on 40-man"))
        else:
            doc["notes"].append("MLB roster check skipped - too few teams "
                                "read")

    doc["out"] = [{"dk_player_id": k, "name": v[0], "team": v[1],
                   "reason": v[2]} for k, v in sorted(out.items(),
                                                       key=lambda x: x[1][0])]
    return _write(sport, doc)


def _write(sport: str, doc: dict) -> dict:
    DATA.mkdir(exist_ok=True)
    path_for(sport).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    log.info("%s: %d out, %d questionable -> %s %s", sport, len(doc["out"]),
             len(doc["questionable"]), path_for(sport).name,
             ("| " + " | ".join(doc["notes"])) if doc["notes"] else "")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sports", nargs="+", default=list(SPORTS),
                    choices=SPORTS)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    failed = []
    for s in args.sports:
        try:
            fetch(s)
        except Exception as exc:                               # noqa: BLE001
            log.error("%s availability failed: %s", s, exc)
            failed.append(s)
    # One sport failing must not stop the others, but it must not look green.
    return 1 if failed and len(failed) == len(args.sports) else 0


if __name__ == "__main__":
    sys.exit(main())
