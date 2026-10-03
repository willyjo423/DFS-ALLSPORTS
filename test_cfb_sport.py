"""What the engine cannot check, because it is specific to this sport.

The generic properties - no leak, stable row count, shares not overwriting
raw columns - are tested once in test_engine.py against an invented sport.
These are the college-football ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import cfb_sport as S
from engine import features as EF


def _history(periods=6, seed=0):
    """Five real fixtures a week, each team with exactly one opponent."""
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(1, periods + 1):
        for a, b in [(0, 1), (2, 3), (4, 5)]:
            for side, opp in ((a, b), (b, a)):
                for j in range(6):
                    p = side * 6 + j
                    rows.append({
                        "athlete_id": f"p{p}", "name": f"Player {p}",
                        "season": 2025, "week": w, "school": f"T{side}",
                        "opponent": f"T{opp}", "is_home": int(side < opp),
                        "position": ["QB", "RB", "WR", "TE"][j % 4],
                        "carries": float(rng.integers(0, 15)),
                        "rec": float(rng.integers(0, 8)),
                        "rush_yards": float(rng.integers(0, 90)),
                        "rec_yards": float(rng.integers(0, 110)),
                        "pass_yards": float(rng.integers(0, 260)),
                        "completions": float(rng.integers(0, 22)),
                        # Asymmetric on purpose. With both sides scoring
                        # identically every margin is exactly zero, and the
                        # margin tests below pass while measuring nothing.
                        "points": float(10 * j + w + 7 * side),
                    })
    return pd.DataFrame(rows)


def test_canonical_rename_and_string_ids():
    """athlete_id must survive as TEXT. Read back as an int, "0041" becomes
    41 and matches nothing - which has already cost this project once."""
    hist = _history(periods=2)
    hist.loc[0, "athlete_id"] = "0041"
    out = S.to_canonical(hist)
    for c in ("player_id", "team", "period"):
        assert c in out.columns, c
    assert out["player_id"].iloc[0] == "0041"
    assert isinstance(out["player_id"].iloc[0], str)


def test_to_canonical_refuses_a_frame_it_cannot_use():
    try:
        S.to_canonical(pd.DataFrame({"nonsense": [1]}))
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("should have refused")


def test_build_produces_every_declared_feature():
    out = S.build(_history())
    missing = [c for c in S.SPEC.features if c not in out.columns]
    assert not missing, missing


def test_build_preserves_row_count():
    hist = _history()
    assert len(S.build(hist)) == len(hist)


def test_margins_are_one_per_team_period_and_cancel():
    df = S.to_canonical(_history())
    m = S.margins(df)
    assert m.duplicated(["team", "season", "period"]).sum() == 0
    for (_, _), g in m.groupby(["season", "period"]):
        assert abs(g["margin"].sum()) < 1e-9, "margins must cancel"


def test_margin_is_a_real_signal_not_a_constant():
    """A constant column looks exactly like data and the model learns it."""
    out = S.build(_history(periods=8))
    assert out["team_ewm_margin"].notna().any()
    assert out["team_ewm_margin"].nunique(dropna=True) > 1


def test_opponent_features_differ_from_own_team_features():
    """opp_* must describe the OTHER side. Joining them on the wrong key
    produces a column that is real, plausible and about the wrong team."""
    out = S.build(_history(periods=8))
    both = out.dropna(subset=["team_ewm_margin", "opp_ewm_margin"])
    assert len(both) > 0
    assert not np.allclose(both["team_ewm_margin"].to_numpy(dtype=float),
                           both["opp_ewm_margin"].to_numpy(dtype=float))


def test_receptions_are_used_and_targets_are_not_pretended():
    """CFBD publishes no targets. The spec must not claim one."""
    assert "rec" in S.SPEC.usage
    assert not any("target" in c for c in S.SPEC.usage + S.SPEC.shares)


def test_share_columns_are_derived_not_overwritten():
    out = S.build(_history())
    assert out["carries"].max() > 1.0, "raw volume was replaced by a share"
    assert out["share_carries"].max() <= 1.0 + 1e-9
    assert "ewm_share_carries" in out.columns


def test_trainable_respects_the_spec():
    out = S.build(_history(periods=9))
    tr = EF.trainable(out, S.SPEC)
    assert tr["games_played"].min() >= S.SPEC.min_prior_games
    assert set(tr["position"]) <= set(S.SPEC.positions)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  pass  {t.__name__}")
        except Exception as exc:                   # noqa: BLE001
            failed.append((t.__name__, exc))
            print(f"  FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.ERROR)
    raise SystemExit(main())
