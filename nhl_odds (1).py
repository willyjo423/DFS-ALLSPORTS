"""Implied team goal totals from the betting market.

Why this exists
---------------
The stack was always Toronto. Not because the model liked Toronto's matchup -
it has almost no way to express a matchup - but because Toronto has the best
individual players, and with the opponent adjustment clipped to plus or minus
fifteen percent, "which team to stack" degenerates into "which team has the
best players". That is the same team most nights.

The betting market does not have that problem. A game total and a moneyline
are the sharpest public estimate of how many goals a team scores tonight, and
they already contain the things this model cannot see: who is in net, who is
hurt, who played last night, what the pace of the matchup is. Implied team
totals across an NHL slate run roughly 2.3 to 4.3 goals - a spread of about
1.9x, against the 1.35x the projection model can currently express.

How a team total is derived
---------------------------
NOT from the puck line. In football the spread carries the information; in
hockey the puck line is pinned at plus or minus 1.5 almost every night and
tells you almost nothing. The moneyline is where the information is.

So: take the game total T and the de-vigged home win probability p, and solve
for the pair of Poisson means that produce both -

    lambda_home + lambda_away = T
    P(home scores more) + half of P(tie) = p

The tie term matters: an NHL game tied after regulation is settled in
overtime or a shootout, which is close enough to a coin toss to model as one.
Dropping it biases every favourite's total upward.

Secrets
-------
The key is read from the ODDS_API_KEY environment variable and is never
written to a file, a log line or a committed artefact. In GitHub Actions put
it in repository Settings -> Secrets and variables -> Actions.

    python nhl_odds.py --probe      # dump the real payload shape, no building
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import statistics

import pandas as pd
import requests

log = logging.getLogger("nhl_odds")

API = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"
TIMEOUT = 30

# Bookmakers to prefer, in order. The median across whoever answers is used
# rather than any one book, so a single stale line cannot move a slate.
REGIONS = "us"
MARKETS = "totals,h2h"

# An NHL game averages about 6.1 total goals. Used only as the fallback when
# a game has no total, and as the yardstick the multiplier is measured
# against when the slate itself is too small to average.
LEAGUE_GAME_TOTAL = 6.1

# The multiplier handed to the projection model is clipped here. The market's
# own spread is about 1.9x best to worst; this allows a bit more than that in
# each direction and no more, because one mis-parsed line should degrade a
# slate rather than destroy it.
TEAM_TOTAL_CLIP = (0.70, 1.40)

# The Odds API names teams in full. Everything downstream speaks MoneyPuck's
# three-letter codes, and a name that fails to map silently removes a whole
# team's worth of signal - so this is exhaustive and the loader shouts about
# anything it cannot place.
TEAM_CODES = {
    "anaheim ducks": "ANA", "boston bruins": "BOS", "buffalo sabres": "BUF",
    "calgary flames": "CGY", "carolina hurricanes": "CAR",
    "chicago blackhawks": "CHI", "colorado avalanche": "COL",
    "columbus blue jackets": "CBJ", "dallas stars": "DAL",
    "detroit red wings": "DET", "edmonton oilers": "EDM",
    "florida panthers": "FLA", "los angeles kings": "LAK",
    "minnesota wild": "MIN", "montreal canadiens": "MTL",
    "montréal canadiens": "MTL", "nashville predators": "NSH",
    "new jersey devils": "NJD", "new york islanders": "NYI",
    "new york rangers": "NYR", "ottawa senators": "OTT",
    "philadelphia flyers": "PHI", "pittsburgh penguins": "PIT",
    "san jose sharks": "SJS", "seattle kraken": "SEA",
    "st louis blues": "STL", "st. louis blues": "STL",
    "tampa bay lightning": "TBL", "toronto maple leafs": "TOR",
    "utah hockey club": "UTA", "utah mammoth": "UTA",
    "arizona coyotes": "UTA", "vancouver canucks": "VAN",
    "vegas golden knights": "VGK", "washington capitals": "WSH",
    "winnipeg jets": "WPG",
}


class OddsUnavailable(RuntimeError):
    """The market could not be read. The caller decides whether that is fatal."""


def code_for(name: str) -> str | None:
    return TEAM_CODES.get(str(name or "").strip().lower())


# ------------------------------------------------------------------ fetching
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
        raise OddsUnavailable(f"{type(exc).__name__}: {str(exc)[:120]}") from exc

    used = r.headers.get("x-requests-used")
    left = r.headers.get("x-requests-remaining")
    if left is not None:
        log.info("odds api quota: %s used, %s remaining this month", used, left)
        try:
            if int(left) < 50:
                log.warning("only %s odds-api credits left this month", left)
        except ValueError:
            pass
    if r.status_code != 200:
        # The message is included because this API explains itself well -
        # an expired key and an exhausted quota are different problems and
        # the body says which.
        raise OddsUnavailable(f"HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    if not isinstance(data, list):
        raise OddsUnavailable(f"expected a list of games, got {type(data).__name__}")
    return data


# ------------------------------------------------------------- the arithmetic
def implied(american) -> float | None:
    """American odds to an implied probability, vig included."""
    try:
        p = float(american)
    except (TypeError, ValueError):
        return None
    if p == 0:
        return None
    return 100.0 / (p + 100.0) if p > 0 else (-p) / ((-p) + 100.0)


def devig(p_home: float, p_away: float) -> float | None:
    """Two outcomes that sum to more than one, normalised back to one.

    The book's margin is in both numbers; dividing by the sum removes it
    proportionally. Crude compared with a power or shin de-vig, and entirely
    adequate for splitting a total.
    """
    if p_home is None or p_away is None:
        return None
    s = p_home + p_away
    if not 0.8 < s < 1.5:
        return None
    return p_home / s


def _pois(k: int, lam: float) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


def home_win_prob(lam_home: float, lam_away: float, cap: int = 15) -> float:
    """P(home wins), counting half of regulation ties as home wins.

    A tie after sixty minutes goes to overtime or a shootout. Treating that as
    a coin toss is not exact, but omitting it entirely biases every
    favourite's implied total upward, which is the error that matters here.
    """
    ph = [_pois(i, lam_home) for i in range(cap + 1)]
    pa = [_pois(i, lam_away) for i in range(cap + 1)]
    win = tie = lose = 0.0
    for i, a in enumerate(ph):
        for j, b in enumerate(pa):
            if i > j:
                win += a * b
            elif i == j:
                tie += a * b
            else:
                lose += a * b
    # NORMALISED BY THE MASS ACTUALLY SUMMED. Stopping at fifteen goals drops
    # about a millionth of the distribution, and dropping it unevenly makes
    # two equal means come back as something fractionally off a coin flip -
    # which the bisection then "corrects" by handing the home side a few
    # thousandths of a goal it has not earned. Dividing by the mass summed
    # makes the function exact on its own support.
    total = win + tie + lose
    return (win + 0.5 * tie) / total if total > 0 else 0.5


def split_total(total: float, p_home: float) -> tuple[float, float]:
    """Split a game total into two team totals consistent with the moneyline.

    Solved rather than approximated. The usual shortcut - half the total plus
    some constant times the edge - needs a magic constant that is wrong at the
    ends of the range, and the ends are exactly where a stack decision gets
    made.
    """
    total = max(3.0, min(12.0, float(total)))
    lo, hi = 0.2, total - 0.2
    for _ in range(60):                       # bisection; 60 is far more than
        mid = (lo + hi) / 2.0                 # enough for four decimals
        if home_win_prob(mid, total - mid) < p_home:
            lo = mid
        else:
            hi = mid
    lam_home = (lo + hi) / 2.0
    return lam_home, total - lam_home


# ------------------------------------------------------------------ assembly
def _median_market(game: dict, market: str, picker) -> float | None:
    vals = []
    for bk in game.get("bookmakers") or []:
        for mk in bk.get("markets") or []:
            if mk.get("key") != market:
                continue
            v = picker(mk.get("outcomes") or [])
            if v is not None:
                vals.append(v)
    return statistics.median(vals) if vals else None


def team_totals(games: list) -> pd.DataFrame:
    """One row per team on the slate: implied goals, and the multiplier."""
    rows, unmapped = [], set()
    for g in games:
        home, away = g.get("home_team"), g.get("away_team")
        hc, ac = code_for(home), code_for(away)
        if not hc or not ac:
            if not hc:
                unmapped.add(str(home))
            if not ac:
                unmapped.add(str(away))
            continue

        total = _median_market(
            g, "totals",
            lambda outs: next((o.get("point") for o in outs
                               if str(o.get("name", "")).lower() == "over"), None))
        ph = _median_market(
            g, "h2h",
            lambda outs: devig(
                implied(next((o.get("price") for o in outs
                              if o.get("name") == home), None)),
                implied(next((o.get("price") for o in outs
                              if o.get("name") == away), None))))

        if total is None:
            log.warning("%s @ %s has no total; using the league average %.1f",
                        ac, hc, LEAGUE_GAME_TOTAL)
            total = LEAGUE_GAME_TOTAL
        if ph is None:
            log.warning("%s @ %s has no usable moneyline; splitting the total "
                        "evenly", ac, hc)
            ph = 0.5

        lam_h, lam_a = split_total(total, ph)
        start = g.get("commence_time")
        rows.append({"team": hc, "opponent": ac, "is_home": 1.0,
                     "game_total": round(float(total), 2),
                     "win_prob": round(float(ph), 4),
                     "implied_goals": round(lam_h, 3), "start": start})
        rows.append({"team": ac, "opponent": hc, "is_home": 0.0,
                     "game_total": round(float(total), 2),
                     "win_prob": round(1 - float(ph), 4),
                     "implied_goals": round(lam_a, 3), "start": start})

    if unmapped:
        # Loud, because a team that fails to map loses its whole signal and
        # the board simply reverts to the old behaviour for it.
        log.error("these team names are not in TEAM_CODES and were SKIPPED, "
                  "so those games carry no market signal: %s",
                  ", ".join(sorted(unmapped)))
    if not rows:
        raise OddsUnavailable("no game produced a usable total")

    out = pd.DataFrame(rows)
    mean = float(out["implied_goals"].mean())
    if not 2.0 <= mean <= 4.5:
        log.error("implied team goals average %.2f across the slate, which is "
                  "not a hockey number - check the parse before trusting it",
                  mean)
    out["factor"] = (out["implied_goals"] / max(mean, 1e-9)).clip(*TEAM_TOTAL_CLIP)
    log.info("market team totals, %d teams, average %.2f goals", len(out), mean)
    for _, r in out.sort_values("implied_goals", ascending=False).head(6).iterrows():
        log.info("  %-4s %.2f goals (total %.1f, win %.0f%%) -> x%.3f",
                 r["team"], r["implied_goals"], r["game_total"],
                 100 * r["win_prob"], r["factor"])
    return out


def load(key: str | None = None) -> pd.DataFrame:
    return team_totals(fetch(key))


# --------------------------------------------------------------------- probe
def probe(key: str | None = None) -> None:
    """Print what the API actually returns and build nothing.

    The same move that paid for itself five times over against MoneyPuck: one
    read of the real payload answers every field-name question at once, and
    the alternative is an evening of guessing.
    """
    data = fetch(key, markets="totals,h2h,spreads")
    print(f"games returned: {len(data)}")
    if not data:
        print("No games. Out of season, or no lines posted yet.")
        return
    g = data[0]
    print("\nTOP-LEVEL KEYS:", sorted(g.keys()))
    for k in ("id", "sport_key", "commence_time", "home_team", "away_team"):
        print(f"  {k:16s} {g.get(k)!r}")
    bks = g.get("bookmakers") or []
    print(f"\nbookmakers: {len(bks)} -> {[b.get('key') for b in bks[:8]]}")
    for bk in bks[:2]:
        print(f"\n--- {bk.get('key')} ---  keys: {sorted(bk.keys())}")
        for mk in bk.get("markets") or []:
            print(f"  market {mk.get('key')!r}  keys: {sorted(mk.keys())}")
            for o in (mk.get("outcomes") or [])[:3]:
                print(f"    outcome: {o}")
    print("\n--- WHAT THIS PARSES TO ---")
    try:
        print(team_totals(data).to_string(index=False))
    except Exception as exc:                                   # noqa: BLE001
        print(f"PARSE FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="print the live payload shape and build nothing")
    a = ap.parse_args()
    if a.probe:
        probe()
    else:
        print(load().to_string(index=False))
