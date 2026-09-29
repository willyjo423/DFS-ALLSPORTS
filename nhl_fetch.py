"""Season history, fetched once and cached.

MoneyPuck publishes CAREERS, not slates: one file per player holding every
game he has ever played. So a season is assembled by asking the season summary
who played, pulling each of their careers, and keeping the rows that belong to
the season asked for.

That is about nine hundred requests for skaters and a hundred for goalies, and
it is why this goes through the cache rather than running on every publish. A
finished season never changes, so it is fetched once and read from parquet
forever after. Only the current season is worth refreshing, and the publisher
does that explicitly.

Run directly to fill the cache:

    python nhl_fetch.py --seasons 2024 2025
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

import nhl_data as ND
from engine import cache as C

SPORT = "nhl"
log = logging.getLogger("nhl_fetch")


class NotPlayedYet(RuntimeError):
    """This season exists on the calendar and has produced no data.

    A normal state, not an error. On the night a season opens MoneyPuck has
    published nothing for it, and asking is the right thing to do - failing
    because the answer is "no games yet" is not.
    """


def fetch_season(season: int, limit: int = 0, pause: float = 0.05
                 ) -> pd.DataFrame:
    """Every player-game of one season, skaters and goalies together.

    Both sides live in one cached frame because the engine splits them by
    position anyway, and because two caches that can disagree about which
    seasons they hold is a bug waiting for a quiet night.
    """
    frames = []
    for side in ("skaters", "goalies"):
        try:
            who = ND.season_summary(season, side)
        except ND.DataUnavailable as exc:
            # An empty summary means the season has not produced data yet,
            # which on an opening night is simply true. Raised as its own type
            # so the caller can skip the season instead of dying on it.
            raise NotPlayedYet(
                f"MoneyPuck has no {side} data for the {season}-"
                f"{str(season + 1)[-2:]} season yet ({str(exc)[:90]})"
            ) from exc
        ids = who["player_id"].dropna().unique().tolist()
        if limit:
            ids = ids[:limit]
        log.info("%d %s: %d players listed, pulling their careers",
                 season, side, len(ids))
        games = ND.game_by_game(ids, side=side, pause=pause)

        # A career file holds every season. Keep the one asked for, by DATE
        # rather than by any season column - the file's own `season` is the
        # one MoneyPuck assigns and this has to agree with the cache's idea
        # of a season, which is the calendar the schedule uses.
        #
        # A hockey season spans two calendar years: 2025 means October 2025
        # through June 2026.
        start = pd.Timestamp(f"{season}-08-01")
        stop = pd.Timestamp(f"{season + 1}-07-31")
        keep = games[(games["date"] >= start) & (games["date"] <= stop)]
        log.info("%d %s: %d of %d career rows fall in this season",
                 season, side, len(keep), len(games))
        frames.append(keep)

    out = pd.concat(frames, ignore_index=True)
    if out.empty:
        raise RuntimeError(
            f"{season}: no player-games survived. Either the season has not "
            f"started or the date window is wrong - the careers held "
            f"{', '.join(str(x) for x in sorted({d.year for d in pd.concat(frames + [pd.DataFrame({'date': []})])['date'].dropna()})[:6])}")

    out["season"] = season
    log.info("%d: %d player-games, %d players, %s to %s",
             season, len(out), out["player_id"].nunique(),
             out["date"].min().date(), out["date"].max().date())
    return out


def load(seasons: list[int], refresh: list[int] | None = None) -> pd.DataFrame:
    """The cached history, fetching only what is missing.

    A season that has not been played yet is SKIPPED, not fatal. The alternative
    is what happened on the 2026-27 opening night: the publisher asked for a
    season whose first games were being played that evening, MoneyPuck
    correctly returned nothing, and the whole build died holding a complete and
    perfectly good 2025-26 cache.
    """
    usable, skipped = [], []
    for s in seasons:
        if C.have(SPORT, s) and s not in set(refresh or []):
            usable.append(s)
            continue
        try:
            C.save(fetch_season(s), SPORT, s)
            usable.append(s)
        except NotPlayedYet as exc:
            skipped.append(s)
            log.info("season %d skipped: %s", s, exc)

    if not usable:
        raise RuntimeError(
            f"no season in {seasons} has any data. Run the NHL fetch "
            f"workflow for a season that has actually been played - on an "
            f"opening night that means LAST season.")
    if skipped:
        log.warning("history is seasons %s; %s had no data yet. Until a few "
                    "games are in the books this board is last season's "
                    "model - the rosters and lines have changed since.",
                    usable, skipped)
    return C.load(SPORT, usable)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seasons", type=int, nargs="+", required=True)
    p.add_argument("--refresh", type=int, nargs="*", default=None,
                   help="seasons to re-fetch even if already cached")
    p.add_argument("--limit", type=int, default=0,
                   help="only this many players per side (for a smoke test)")
    args = p.parse_args(argv)

    if args.limit:
        # A SMOKE TEST WRITES NOTHING, and that is the whole point.
        #
        # Saving a 25-player frame to the real cache path would leave the
        # season looking fetched: `ensure` would see the file, skip the real
        # download, and every publish afterwards would fit on two dozen
        # players while reporting a full cache. The run that was meant to
        # prove the path works would be the run that silently broke it.
        for s in args.seasons:
            df = fetch_season(s, limit=args.limit)
            print(f"\nSMOKE TEST for {s}: {len(df):,} player-games from "
                  f"{df['player_id'].nunique()} players.")
            print(df.head(5).to_string(max_colwidth=14))
        print("\nNOTHING WAS CACHED - this was a smoke test. The whole path "
              "works: summary, careers, the season window, the columns. "
              "Re-run with the limit box empty for the real fetch.")
        return 0

    load(args.seasons, refresh=args.refresh)
    df = C.load(SPORT, args.seasons)
    print(f"\ncache holds {len(df):,} player-games across "
          f"{df['season'].nunique()} season(s)")
    print(df.groupby("season").agg(
        games=("game_id", "nunique"), players=("player_id", "nunique"),
        rows=("player_id", "size")).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
