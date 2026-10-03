"""Fetch each season once, ever.

CFBD's free tier has a monthly call quota and this project spent it. Every
verification run re-downloaded three or four complete seasons a week at a
time - roughly sixteen calls per season - and the first grading run asked for
five more on top. The quota ran out mid-2023, so 2024 and 2025 never loaded,
the walk-forward had nothing to walk over, and the failure arrived as
"nothing was gradeable" rather than as anything about quotas.

A finished season never changes. Downloading 2021 more than once is not a
tradeoff or an optimisation question - it is waste that eventually costs the
ability to work at all, which is exactly what happened.

So: history lives in parquet under data/, one file per season, fetched once
and committed. Runs read from disk. The only season worth re-fetching is the
current one, because it grows every Saturday, and `refresh` exists for that
and only that.

Gzipped CSV rather than parquet, for one reason: parquet needs pyarrow, which
cannot be installed in the environment this code is written in, so a parquet
cache could only ever be tested somewhere other than where it was written.
The last three bugs in this project all hid in exactly that gap - a fixture
that differed from production in the dimension under test. A format that runs
identically in both places is worth more than a smaller file.

The cost is that CSV forgets types, and one of those types matters: an
athlete id is digits but is not a number, and read back as an integer it
stops matching anything. So the dtypes are stated explicitly on read rather
than inferred, and there is a test that a leading zero survives the trip.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import cfb_data as D

log = logging.getLogger(__name__)

CACHE = Path("data")
CURRENT_SEASON_MARKER = "_partial"


# Columns that look numeric and are not. An athlete id read back as int64
# stops matching the board, silently, and the join simply gets worse.
TEXT_COLUMNS = {"athlete_id": "string", "name": "string",
                "school": "string", "opponent": "string",
                "position": "string", "game_id": "string"}


def path_for(season: int) -> Path:
    return CACHE / f"history_{season}.csv.gz"


def have(season: int) -> bool:
    return path_for(season).exists()


def cached_seasons() -> list[int]:
    if not CACHE.exists():
        return []
    out = []
    for p in sorted(CACHE.glob("history_*.csv.gz")):
        try:
            out.append(int(p.name.split("_")[1].split(".")[0]))
        except (IndexError, ValueError):
            continue
    return out


def fetch_season(key: str, season: int, through_week: int | None = None
                 ) -> pd.DataFrame:
    """One season, from the API, with positions attached.

    Positions are kept rather than required: a row without one is still a
    player's own history and still joins to a board, and only the model fit
    needs the position. Filtering happens at fit time.
    """
    positions = D.roster_positions(key, season)
    weeks = through_week
    if weeks is None:
        try:
            weeks = D.live_week(key, season)
        except D.Unavailable:
            weeks = 15
    df = D.season_history(key, season, weeks, positions,
                          require_position=False)
    df["_through_week"] = weeks
    return df


def save(df: pd.DataFrame, season: int) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = path_for(season)
    # `keys` holds python sets, which parquet cannot represent. They are
    # derived from `name` in one line, so they are rebuilt on load rather
    # than stored - cheaper than serialising and impossible to get stale.
    out = df.drop(columns=[c for c in ("keys",) if c in df.columns])
    out.to_csv(p, index=False, compression="gzip")
    log.info("wrote %s (%d rows, %.1f MB)", p, len(out),
             p.stat().st_size / 1e6)
    return p


def load(seasons: list[int]) -> pd.DataFrame:
    """Read cached seasons and rebuild the derived columns."""
    frames = []
    missing = []
    for s in seasons:
        p = path_for(s)
        if not p.exists():
            missing.append(s)
            continue
        # dtype stated, never inferred: athlete_id "0041" must come back as
        # "0041" and not as 41.
        frames.append(pd.read_csv(p, compression="gzip", dtype=TEXT_COLUMNS))
    if missing:
        log.warning("no cached file for %s - run the Fetch history workflow "
                    "for those seasons", missing)
    if not frames:
        raise FileNotFoundError(
            f"no cached history for {seasons}. Run the Fetch history "
            f"workflow first; grading must not re-download the archive on "
            f"every run, which is what exhausted the API quota.")
    df = pd.concat(frames, ignore_index=True)
    df["key"] = df["name"].map(D.primary_key)
    df["keys"] = df["name"].map(D.name_keys)
    log.info("loaded %d player-games from cache, seasons %s",
             len(df), sorted(int(s) for s in df["season"].unique()))
    return df


def ensure(key: str | None, seasons: list[int],
           refresh: list[int] | None = None) -> pd.DataFrame:
    """Load from cache, fetching ONLY what is missing or explicitly refreshed.

    `refresh` is for the current season, which grows every week. Finished
    seasons are never re-fetched, because they cannot have changed.
    """
    refresh = set(refresh or [])
    todo = [s for s in seasons if not have(s) or s in refresh]
    if todo and not key:
        raise RuntimeError(
            f"seasons {todo} are not cached and no API key was given")
    for s in todo:
        why = "refresh" if have(s) else "missing"
        log.info("fetching %d (%s)", s, why)
        save(fetch_season(key, s), s)
    return load(seasons)
