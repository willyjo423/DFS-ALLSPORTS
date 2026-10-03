"""Download seasons into the cache, once, and stop.

Run this when a season is missing or the current one has new games. It is the
only file in the project permitted to spend API calls on the archive.

It is deliberately conservative about the quota that this project has already
exhausted once:

  * a season already on disk is skipped unless explicitly refreshed;
  * seasons are fetched oldest-first, so a run that dies partway still leaves
    the cache strictly better than it found it;
  * every season is written as soon as it is complete rather than at the end,
    so a quota error on season four does not discard seasons one to three.

That last point is the difference between a failed run costing nothing and
costing everything, and the first grading run lost five minutes of
downloading to exactly that.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import cfb_cache as C
import cfb_data as D


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", default="2021 2022 2023 2024 2025 2026")
    p.add_argument("--refresh", default="",
                   help="seasons to re-fetch even if cached (the live one)")
    args = p.parse_args()

    key = os.environ.get("CFBD_API_KEY", "").strip()
    if not key:
        sys.exit("CFBD_API_KEY is not set.")

    seasons = [int(x) for x in args.seasons.split()]
    refresh = {int(x) for x in args.refresh.split()} if args.refresh else set()

    print("cached before:", C.cached_seasons() or "nothing")
    print("requested    :", seasons)
    print("refreshing   :", sorted(refresh) or "nothing")
    print()

    done, skipped, failed = [], [], []
    for season in sorted(seasons):
        if C.have(season) and season not in refresh:
            skipped.append(season)
            print(f"{season}: already cached, skipping")
            continue
        try:
            df = C.fetch_season(key, season)
            C.save(df, season)
            done.append(season)
            print(f"{season}: {len(df):,} player-games cached")
        except Exception as exc:                   # noqa: BLE001
            failed.append((season, str(exc)[:160]))
            print(f"{season}: FAILED - {str(exc)[:160]}")
            if "quota" in str(exc).lower() or "429" in str(exc):
                print("\nAPI quota exhausted. Everything fetched so far is "
                      "already written to data/ and will not need "
                      "re-downloading.")
                break

    print()
    print("=" * 60)
    print(f"fetched : {done or 'nothing'}")
    print(f"skipped : {skipped or 'nothing'}")
    print(f"failed  : {[s for s, _ in failed] or 'nothing'}")
    print(f"cache now holds: {C.cached_seasons()}")
    if failed:
        print("\nCommit what succeeded, then re-run for the rest when the "
              "quota resets. Nothing already on disk will be fetched again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
