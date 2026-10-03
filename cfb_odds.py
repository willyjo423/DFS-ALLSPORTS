"""Live market lines for college football, keyed the way the board is keyed.

What this is for
----------------
Without it, "which team do I stack" degenerates into "which team has the best
players", and that is the same team every week. The market is the only input
that knows tonight's game is a 72-point shootout in Lubbock and that one is a
38-point rock fight in the rain, and in college football that spread is
enormous: implied team totals run from about 10 to about 50, a five-fold
range. An NHL slate spans 2.3 to 4.3 goals. A model blind to the market is
making a much bigger mistake in this sport than in any other.

So the lines do two jobs here. They scale the projections, so a back on a team
implied for 45 is not priced like the same back on a team implied for 17. And
they go onto the page, so the stack board can be read by game environment
rather than by name recognition.

The sign, which is the only thing in this file that can be badly wrong
---------------------------------------------------------------------
The feed quotes a home favourite at a NEGATIVE point, because that is what you
lay. The arithmetic here wants a spread that is POSITIVE when the home side is
favoured, because it adds it to the total:

    implied_home = (total + spread_line) / 2        spread_line = -point_home

Get that backwards and every favourite's players are handed their opponent's
expectation. It is wrong in every single row and entirely plausible in
aggregate - the totals still average out, the distribution still looks like
football - which is exactly why it is checked against a worked example in
`test_cfb_odds.py` rather than against my confidence.

The 130-team name problem, and why there is no table here
---------------------------------------------------------
The feed names schools with their mascots attached - "Alabama Crimson Tide" -
and CFBD names them without - "Alabama". There are about 134 FBS teams and the
spellings disagree on accents, apostrophes, ampersands, "State" versus "St.",
and a dozen one-off abbreviations.

A hand-written table of 134 pairs is 134 guesses that look like knowledge, and
at least one of them is wrong in a way nobody can see. A wrong one does not
drop a team: it hands that team's players the OTHER game's total, which
produces a full board of confident numbers that are wrong.

So the names are matched against the schools CFBD actually scheduled this
week, by longest-prefix and then by token overlap, each match has to be
unique, and anything unresolved is REPORTED rather than guessed. That is the
same discipline `cfb_data.fixture_team_map` already applies to DraftKings'
team codes, and for the same reason.

    python cfb_odds.py --probe
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import re
import statistics
import unicodedata

import pandas as pd
import requests

log = logging.getLogger("cfb_odds")

API = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds"
TIMEOUT = 30
REGIONS = "us"
MARKETS = "spreads,totals"

# A college team plays once a week, so "this week" is the earliest game per
# team inside this horizon. Deliberately not a week-number lookup: the feed
# carries no week number, and deriving one is the bug that made the hockey
# build read a different night's slate.
HORIZON_DAYS = 8

# What a college implied team total can plausibly be. Wider than the NFL's
# because college football genuinely is: a 20-point team total and a 48-point
# team total are both ordinary Saturdays. The guard is there to catch a parse
# that has read the moneyline as a total, not to second-guess a real line.
SANE_TEAM_TOTAL = (6.0, 60.0)

# Words that carry no information about which school this is. "State" is NOT
# in here, and must not be: Michigan and Michigan State are different teams,
# and the single most expensive mistake this file could make is to collapse
# them.
_NOISE = {"the", "university", "of", "at"}

_MASCOT_SPLIT = re.compile(r"[^a-z0-9&' ]+")


class OddsUnavailable(RuntimeError):
    """The market could not be read. The caller decides whether that is fatal."""


# --------------------------------------------------------------- name shapes
def normalise(name) -> str:
    """One spelling of a school name, for comparison only.

    Accents folded (San Jose State and San Jose State), apostrophes dropped
    (Hawai'i), ampersands spelled out (Texas A&M survives either way), and the
    common abbreviations of "State" unified - because a feed that writes
    "Michigan St." and a schedule that writes "Michigan State" are talking
    about the same team, and a feed that writes "Michigan" is not.
    """
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().replace("&", " and ").replace("'", "").replace("’", "")
    s = _MASCOT_SPLIT.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Only as a whole word, and only these two. A blanket "st -> state" would
    # rewrite "St. John's" into "State Johns".
    s = re.sub(r"\bst\b", "state", s)
    s = re.sub(r"\buniv\b", "university", s)
    return s


def _tokens(name) -> list[str]:
    return [t for t in normalise(name).split() if t not in _NOISE]


def match_schools(feed_names: list[str], schools: list[str]) -> dict:
    """Feed name -> CFBD school, for the ones that can be resolved uniquely.

    Three passes, strictest first, and a pass only gets to answer if its
    answer is the ONLY one of its kind:

      1. the normalised names are equal;
      2. the school's tokens are a leading run of the feed name's tokens -
         "alabama crimson tide" starts with "alabama" - and if several schools
         qualify the LONGEST wins, which is what keeps "miami oh redhawks"
         off Miami and "michigan state spartans" off Michigan;
      3. every token of the school appears somewhere in the feed name, for the
         handful of feeds that lead with the mascot or interpose a word.

    Anything that two different schools can claim at the same strength is left
    unresolved on purpose. An unresolved team loses its market line, which the
    caller can see and report. A wrongly resolved one gets another game's
    total, which nobody can see.
    """
    by_norm: dict[str, list[str]] = {}
    for s in schools:
        by_norm.setdefault(normalise(s), []).append(s)

    out, ambiguous = {}, {}
    for raw in feed_names:
        n = normalise(raw)
        if not n:
            continue
        if n in by_norm and len(by_norm[n]) == 1:
            out[raw] = by_norm[n][0]
            continue

        toks = _tokens(raw)
        best, best_len, tie = None, 0, False
        for s in schools:
            st = _tokens(s)
            if not st or len(st) > len(toks):
                continue
            if toks[:len(st)] != st:
                continue
            if len(st) > best_len:
                best, best_len, tie = s, len(st), False
            elif len(st) == best_len and s != best:
                tie = True
        if best is not None and not tie:
            out[raw] = best
            continue
        if tie:
            ambiguous[raw] = best_len
            continue

        # Subset pass. Scored by how much of the feed name the school
        # explains, so a one-token school cannot outrank a two-token one.
        have = set(toks)
        scored = []
        for s in schools:
            st = _tokens(s)
            if st and set(st) <= have:
                scored.append((len(st), s))
        if scored:
            top = max(n for n, _ in scored)
            winners = sorted({s for n, s in scored if n == top})
            if len(winners) == 1:
                out[raw] = winners[0]
            else:
                ambiguous[raw] = top

    if ambiguous:
        log.error("these market team names match more than one scheduled "
                  "school equally well and were left UNRESOLVED rather than "
                  "guessed, so those games carry no live line: %s",
                  ", ".join(sorted(ambiguous)))
    missed = [r for r in feed_names if r and r not in out]
    if missed:
        log.warning("%d market team name(s) matched no scheduled school: %s",
                    len(missed), ", ".join(sorted(missed)[:12]))
    log.info("market names resolved: %d of %d", len(out),
             len({r for r in feed_names if r}))
    return out


# ------------------------------------------------------------------ fetching
def _redact(text: str, key: str) -> str:
    """The key, removed from anything that might be printed.

    NOT a truncation. `requests` puts the full request URL into its transport
    exceptions, and this API takes the key as a QUERY PARAMETER, so a DNS or
    proxy failure produces a message ending `...odds?apiKey=<the key>`. The
    first version of this relied on cutting the message at 120 characters,
    which happened to land 17 characters in front of the key on the
    production URL - a margin, not a guarantee, and the thing on the other
    side of it is a secret written into a log file that Actions then uploads
    as an artifact. Redacting the value is the only version of this that is
    true for every message.
    """
    out = str(text)
    if key:
        out = out.replace(key, "<redacted>")
        # Also any prefix long enough to be worth brute-forcing, in case the
        # message truncated the key itself mid-way.
        for n in (24, 16, 12, 8):
            if len(key) > n:
                out = out.replace(key[:n], "<redacted>")
    return out


def fetch(key: str | None = None, markets: str = MARKETS) -> list:
    key = key or os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise OddsUnavailable(
            "ODDS_API_KEY is not set. In Actions add it under Settings -> "
            "Secrets and variables -> Actions, and pass it into the job's env.")
    try:
        r = requests.get(API, timeout=TIMEOUT, params={
            "apiKey": key, "regions": REGIONS, "markets": markets,
            "oddsFormat": "american", "dateFormat": "iso"})
    except Exception as exc:                                   # noqa: BLE001
        raise OddsUnavailable(
            f"{type(exc).__name__}: "
            f"{_redact(exc, key)[:200]}") from None
    left = r.headers.get("x-requests-remaining")
    if left is not None:
        log.info("odds api quota: %s used, %s remaining this month",
                 r.headers.get("x-requests-used"), left)
    if r.status_code != 200:
        raise OddsUnavailable(
            f"HTTP {r.status_code}: {_redact(r.text, key)[:200]}")
    data = r.json()
    if not isinstance(data, list):
        raise OddsUnavailable(
            f"expected a list of games, got {type(data).__name__}")
    return data


# ------------------------------------------------------------------- parsing
def _median_across_books(game: dict, market: str, picker):
    """The median of every book that quoted it.

    A median rather than one book's number, and rather than a mean: one book
    posting a stale or mistaken line moves a mean and does not move a median.
    """
    vals = []
    for bk in game.get("bookmakers") or []:
        for mk in bk.get("markets") or []:
            if mk.get("key") != market:
                continue
            v = picker(mk.get("outcomes") or [])
            if v is not None:
                vals.append(float(v))
    return statistics.median(vals) if vals else None


def _start(game: dict):
    try:
        t = dt.datetime.fromisoformat(
            str(game.get("commence_time", "")).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def week_lines(games: list, schools: list[str],
               horizon_days: int = HORIZON_DAYS,
               now: dt.datetime | None = None) -> pd.DataFrame:
    """One row per SCHOOL for the coming week: total, spread, implied total."""
    now = now or dt.datetime.now(dt.timezone.utc)
    horizon = now + dt.timedelta(days=horizon_days)

    inside = []
    for g in games:
        t = _start(g)
        if t is None or t > horizon:
            continue
        inside.append((g, t))

    feed_names = []
    for g, _ in inside:
        feed_names += [g.get("home_team"), g.get("away_team")]
    mapping = match_schools([str(x) for x in feed_names if x], list(schools))

    rows = []
    for g, t in inside:
        home, away = g.get("home_team"), g.get("away_team")
        hs, aws = mapping.get(str(home)), mapping.get(str(away))
        if not hs or not aws:
            continue

        total = _median_across_books(
            g, "totals",
            lambda o: next((x.get("point") for x in o
                            if str(x.get("name", "")).lower() == "over"), None))
        # Matched through `normalise`, not by string equality.
        #
        # The totals picker keys on "over", which every book spells the same
        # way. The spreads picker has to identify WHICH SIDE a point belongs
        # to, and it was comparing the outcome name to the game-level
        # `home_team` string exactly. A book that writes "Miami (FL)" where
        # the game says "Miami Hurricanes" then contributed its TOTAL and not
        # its SPREAD - so `implied_total` became the median total over three
        # books combined with the median spread over two, which is not a
        # quantity. Reproduced: median of 55 and 59 for the total, median of
        # one value for the spread.
        hn = normalise(home)
        point_home = _median_across_books(
            g, "spreads",
            lambda o: next((x.get("point") for x in o
                            if normalise(x.get("name")) == hn), None))
        if total is None or point_home is None:
            log.info("%s at %s has no %s posted yet; skipped", aws, hs,
                     "total" if total is None else "spread")
            continue

        # THE SIGN. See the module docstring - this one line is the whole risk
        # in this file, and it is wrong in every row if it is wrong at all.
        spread_line = -float(point_home)
        home_implied = (float(total) + spread_line) / 2.0
        away_implied = float(total) - home_implied

        # SANITY CHECKED PER FIXTURE, NOT PER ROW.
        #
        # Written as a row filter this kept the favourite and dropped the dog
        # whenever the two implied totals straddled the bound - which is
        # exactly the half-garbage it exists to catch. A real FBS-versus-FCS
        # line (total 62, home -55) gives the home side 58.5 and the visitor
        # 3.5: the filter dropped the 3.5 and PUBLISHED the 58.5, leaving a
        # board with an opponent_implied of 3.5 for a team that has no row.
        #
        # Either both halves of a game are believable or neither is.
        lo, hi = SANE_TEAM_TOTAL
        if not (lo <= home_implied <= hi and lo <= away_implied <= hi):
            log.error("%s at %s parses to %.1f and %.1f implied points, and "
                      "at least one of those is not a football number. The "
                      "WHOLE fixture is dropped - half a game is worse than "
                      "none, because the half that survives looks fine.",
                      aws, hs, away_implied, home_implied)
            continue

        rows.append({"school": hs, "opponent_school": aws, "is_home": 1.0,
                     "game_total": float(total), "team_spread": spread_line,
                     "implied_total": home_implied,
                     "opponent_implied": away_implied, "start": t})
        rows.append({"school": aws, "opponent_school": hs, "is_home": 0.0,
                     "game_total": float(total), "team_spread": -spread_line,
                     "implied_total": away_implied,
                     "opponent_implied": home_implied, "start": t})

    if not rows:
        raise OddsUnavailable(
            "no game inside the horizon produced both a believable total and "
            "a spread for a school on this week's schedule")

    out = pd.DataFrame(rows)
    # A school plays once a week, so the earliest game inside the horizon IS
    # this week's. Anything later is next week, and averaging two weeks
    # together is a quiet way to be wrong about both.
    dup = int(out["school"].duplicated().sum())
    if dup:
        log.info("%d school(s) appear more than once inside %d days; keeping "
                 "each one's earliest game", dup, horizon_days)
    out = out.sort_values("start").drop_duplicates("school", keep="first")
    if not len(out):
        raise OddsUnavailable(
            "every fixture was dropped by the sanity check, so there is no "
            "market to apply. The parse, not the team map, is what to look "
            "at - the errors above say which fixtures and what they read.")
    log.info("live market: %d schools, implied totals %.1f to %.1f "
             "(mean %.1f)", len(out), out["implied_total"].min(),
             out["implied_total"].max(), out["implied_total"].mean())
    return out.drop(columns=["start"]).reset_index(drop=True)


def for_board(team_map: dict, schools: list[str],
              key: str | None = None, games: list | None = None
              ) -> pd.DataFrame:
    """The market, keyed by DRAFTKINGS team code.

    `team_map` is DraftKings code -> CFBD school, as `fixture_team_map` solves
    it. Inverting it is what lets a line found under "Ohio State" reach a
    board that prices players under "OSU".

    `schools` MUST BE EVERY SCHOOL IN THE LEAGUE, NOT JUST THE ONES ON THIS
    BOARD, and it is a required argument now because defaulting it to the
    board's own schools was the worst bug in this file.
    ----------------------------------------------------------------------
    The matcher resolves a feed name by finding the longest school name that
    is a leading run of it, which is what keeps "Michigan State Spartans" off
    Michigan. That only works if Michigan State is IN THE CANDIDATE LIST.
    Given only the twenty-four schools on a DraftKings slate, a feed name
    whose own school is absent matches the nearest one that is present - so
    with Michigan on the board and Michigan State not, Michigan State's game
    was resolved to "Michigan", `drop_duplicates(keep="first")` then kept
    whichever of Michigan's two candidate games kicked off earlier, and
    Michigan was published in the wrong game entirely: implied 12.0 instead
    of 28.0, a 2.3x error on every Michigan row, under a log line reading
    "market names resolved: 4 of 4".
    Measured against a real 136-school list, offering only one board school
    produced 37 wrong resolutions - Florida State to Florida, Texas Tech and
    Texas A&M to Texas, Miami (OH) to Miami, West Virginia to Virginia. Given
    the full list the matcher gets 135 of 136 right with none wrong.

    A school that two DraftKings codes both claim is dropped rather than
    assigned to one of them. That cannot happen from a solved fixture map, and
    if it ever does the map is broken, which is worth a loud nothing rather
    than a quiet guess.
    """
    schools = sorted({str(s) for s in schools if str(s).strip()})
    # The board's own schools must be in there, or the inversion below has
    # nothing to map back to.
    schools = sorted(set(schools) | {str(s) for s in team_map.values()})
    if len(schools) < 100:
        log.warning("only %d candidate schools were supplied. College "
                    "football has about 134 at this level, and a SHORT list "
                    "is how a team gets handed another game's line: a feed "
                    "name whose own school is missing matches the nearest one "
                    "that is present. Pass the full CFBD teams list.",
                    len(schools))
    lines = week_lines(games if games is not None else fetch(key), schools)

    back: dict[str, list[str]] = {}
    for code, school in team_map.items():
        back.setdefault(str(school), []).append(str(code))
    duped = {s: c for s, c in back.items() if len(c) > 1}
    if duped:
        log.error("these schools are claimed by more than one DraftKings code "
                  "and are therefore given NO line: %s",
                  ", ".join(f"{s} ({', '.join(sorted(c))})"
                            for s, c in sorted(duped.items())))

    out = lines.copy()
    out["team"] = out["school"].map(
        lambda s: back[s][0] if len(back.get(s, [])) == 1 else None)
    out["opponent"] = out["opponent_school"].map(
        lambda s: back[s][0] if len(back.get(s, [])) == 1 else None)
    # Expected to be most of them, and not interesting: the feed covers every
    # game in the country and a DraftKings slate is a dozen of them. Counted
    # rather than listed, so the log says the matcher worked without burying
    # the lines that matter under a hundred school names.
    unmapped = int(out["team"].isna().sum())
    if unmapped:
        log.info("%d school(s) have a line but are not on this board, which "
                 "is normal - the feed covers the whole country", unmapped)
    out = out[out["team"].notna()].copy()
    if not len(out):
        raise OddsUnavailable(
            "every school with a market line failed to map back to a "
            "DraftKings team code. The fixture team map is the thing to look "
            "at, not the odds feed.")
    log.info("market lines attached to %d of %d board team codes",
             len(out), len(team_map))
    return out.reset_index(drop=True)


# --------------------------------------------------------------------- probe
def probe(key: str | None = None) -> None:
    data = fetch(key)
    print(f"games returned: {len(data)}")
    if not data:
        print("No games. Out of season, or no lines posted yet.")
        return
    g = data[0]
    print("\nTOP-LEVEL KEYS:", sorted(g.keys()))
    for k in ("id", "commence_time", "home_team", "away_team"):
        print(f"  {k:16s} {g.get(k)!r}")
    for bk in (g.get("bookmakers") or [])[:2]:
        print(f"\n--- {bk.get('key')} ---")
        for mk in bk.get("markets") or []:
            print(f"  market {mk.get('key')!r}")
            for o in (mk.get("outcomes") or [])[:3]:
                print(f"    {o}")

    names = sorted({str(x.get(side)) for x in data
                    for side in ("home_team", "away_team") if x.get(side)})
    print(f"\n{len(names)} distinct team names in the feed, first 20:")
    for nm in names[:20]:
        print(f"  {nm!r}  ->  normalised {normalise(nm)!r}")
    print("\n--- WHAT THIS PARSES TO, matching the feed against ITSELF ---")
    print("(in a real run the schools come from the CFBD schedule)")
    try:
        df = week_lines(data, [normalise(n).title() for n in names])
        print(df.sort_values("implied_total", ascending=False)
              .head(20).to_string(index=False))
    except Exception as exc:                                   # noqa: BLE001
        print(f"PARSE FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    if a.probe:
        probe()
    else:
        print("cfb_odds is a library; --probe is the only thing it does "
              "alone, because keying lines to a board needs the board.")
