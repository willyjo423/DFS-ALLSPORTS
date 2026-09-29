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


def fetch_season(season: int, limit: int = 0, pause: float = 0.05
                 ) -> pd.DataFrame:
    """Every player-game of one season, skaters and goalies together.

    Both sides live in one cached frame because the engine splits them by
    position anyway, and because two caches that can disagree about which
    seasons they hold is a bug waiting for a quiet night.
    """
    frames = []
    for side in ("skaters", "goalies"):
        who = ND.season_summary(season, side)
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
    """The cached history, fetching only what is missing."""
    return C.ensure(SPORT, seasons, fetch_season, refresh=refresh)


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
        for s in args.seasons:
            C.save(fetch_season(s, limit=args.limit), SPORT, s)
    else:
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
