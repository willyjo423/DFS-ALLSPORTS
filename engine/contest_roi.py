"""Simulated ROI of each lineup in a real DraftKings tournament.

Top-1% and cash rates ignore what a finish is actually worth. A real GPP pays
a steep curve (first place is often 10-20% of the pool), cuts off around the
top 20-25%, and splits a prize with everyone who entered the same lineup. This
scores each lineup the way the money does:

  for every simulated night: where would this lineup finish among N entries
  (the field model's score distribution, scaled to the contest's size), what
  does that rank pay, divided by 1 + its expected duplicates;
  ROI = average payout / entry fee - 1.

The contest is the slate's featured tournament from DraftKings' lobby (the
biggest field). The lobby gives the pool, the fee, the field size and usually the
first prize in the name ("[$20K to 1st]"), but not the whole payout table, so
the table is rebuilt in DraftKings' usual shape: ~22% of entries paid, first
prize as listed, the rest falling off as a power curve to a min-cash of about
twice the fee, summing to the pool.
"""
from __future__ import annotations

import logging
import re

import numpy as np

log = logging.getLogger("contest_roi")
PAID_SHARE = 0.22
# Share of a big tournament's entries that are optimizer-quality - lineups as
# strong as our own pool. Without them the simulated field is only casual,
# ownership-weighted lineups and every pool lineup looks like +1000% ROI.
SHARP_SHARE = 0.20   # 0.20 conserves the prize pool across the field; 0.30 over-pays (duplicate ties)
_CONTESTS: list[dict] = []


def remember(contests: list[dict] | None) -> None:
    """Called by each sport's lobby read with the lobby's Contests list."""
    global _CONTESTS
    if contests:
        _CONTESTS = list(contests)


def _money(s: str) -> float | None:
    m = re.search(r"\$([\d.,]+)\s*([KkMm]?)", s or "")
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    return v * {"k": 1e3, "m": 1e6}.get(m.group(2).lower(), 1.0)


def _eligible(draft_group, contests):
    """The draft group's real tournaments (no double-ups, satellites, free
    promos or unlimited matchups), as {name, pool, fee, entries, first}."""
    out = []
    for c in contests if contests is not None else _CONTESTS:
        if str(c.get("dg")) != str(draft_group):
            continue
        name = str(c.get("n") or "")
        if re.search(r"double up|50/50|head|h2h|satellite|qualifier|beginner|tiers", name, re.I):
            continue
        pool, m = float(c.get("po") or 0), int(c.get("m") or 0)
        fee = float(c.get("a") or 0)
        # Free promos and unlimited "matchups" (max entries 999,999,999) are
        # not tournaments with a payout curve.
        if m < 20 or pool <= 0 or fee <= 0 or m > 2_000_000 or re.search(r"matchup", name, re.I):
            continue
        first = None
        fm = re.search(r"\[\s*(\$[\d.,]+\s*[KkMm]?)\s*to\s*1st", name)
        if fm:
            first = _money(fm.group(1))
        out.append({"name": name, "pool": pool, "fee": fee, "entries": m,
                    "first": first or 0.15 * pool, "first_from_name": bool(first)})
    return out


def featured(draft_group: int, contests: list[dict] | None = None) -> dict | None:
    """The draft group's biggest tournament: pool, fee, entries, first prize.
    The BIGGEST FIELD, not the biggest pool: the biggest pool is often a
    200-entry high-roller room, which is not the tournament played here."""
    got = [c for c in _eligible(draft_group, contests) if c["entries"] >= 200]
    return max(got, key=lambda c: (c["entries"], c["pool"])) if got else None


# Contest sizes the page lets you pick, each scored with its own payout curve.
# Small fields pay flatter and reward the safest strong lineup; huge fields
# pay almost everything to the top few hundred and reward being different.
TIER_RANGES = {"small": (20, 500), "mid": (500, 10_000), "large": (10_000, 2_000_001)}
TIER_LABEL = {"cash": "Double-up / cash game", "small": "Small tournament (under 500 entries)",
              "mid": "Mid tournament (500–10K entries)", "large": "Large tournament (10K+ entries)"}
TIERS = {   # standard contests, used for backtests and when the lobby has none in a tier
    "small": {"name": "standard small GPP", "pool": 2_640.0, "fee": 20.0, "entries": 150,
              "first": 600.0, "first_from_name": False},
    "mid": {"name": "standard mid GPP", "pool": 7_920.0, "fee": 3.0, "entries": 3_000,
            "first": 1_500.0, "first_from_name": False},
    "large": {"name": "standard GPP (backtest)", "pool": 100_000.0, "fee": 5.0,
              "entries": 23_000, "first": 15_000.0, "first_from_name": False},
}


def tier_of(entries: int) -> str:
    return next(t for t, (lo, hi) in TIER_RANGES.items() if lo <= entries < hi)


def tier_contests(draft_group, contests=None) -> dict:
    """{tier: the lobby's biggest-field contest in that size range, or the
    standard one if the lobby has none}."""
    got = _eligible(draft_group, contests)
    out = {}
    for t, (lo, hi) in TIER_RANGES.items():
        inside = [c for c in got if lo <= c["entries"] < hi]
        if inside:
            out[t] = dict(max(inside, key=lambda c: (c["entries"], c["pool"])), real=True)
        else:
            out[t] = dict(TIERS[t], real=False)
    return out


def tier_multiple(pct: float, tier: str, lines: dict | None = None,
                  score: float | None = None) -> float:
    """Prize / fee for finishing at field percentile `pct` in a tier's standard
    contest. Beyond the top 0.1% the field sample cannot resolve a rank, so
    when the night's lines are given the finish is extended on the same
    exponential tail the live ROI uses (fitted between the top 1% and 0.1%)."""
    c = TIERS[tier]
    pay = _table(tier)
    f = max(0.0, (100.0 - pct) / 100.0)
    if lines and score is not None and lines.get("win") and lines.get("top1") \
            and score > lines["win"] and lines["win"] > lines["top1"]:
        b = (lines["win"] - lines["top1"]) / np.log(10.0)
        f = 0.001 * float(np.exp(-(score - lines["win"]) / b))
    rank = max(1, int(np.ceil(f * c["entries"])))
    return float(pay[rank - 1] / c["fee"]) if rank <= len(pay) else 0.0


_TABLES: dict = {}


def _table(tier):
    if tier not in _TABLES:
        c = TIERS[tier]
        _TABLES[tier] = payout_table(c["entries"], c["pool"], c["fee"], c["first"])
    return _TABLES[tier]


def payout_table(entries: int, pool: float, fee: float, first: float) -> np.ndarray:
    """Prize by finishing rank (index 0 = 1st), in DraftKings' usual shape."""
    paid = max(1, int(round(PAID_SHARE * entries)))
    min_cash = min(2.0 * fee, 0.5 * pool / paid)
    first = min(first, 0.5 * pool)
    r = np.arange(1, paid + 1, dtype=float)

    def total(alpha):
        return np.maximum(first * r ** -alpha, min_cash).sum()

    lo, hi = 0.05, 4.0
    for _ in range(60):                     # total() falls as alpha rises
        mid = (lo + hi) / 2
        if total(mid) > pool:
            lo = mid
        else:
            hi = mid
    pay = np.maximum(first * r ** -hi, min_cash)
    return pay * (pool / pay.sum())          # exact pool


def lineup_roi(S: np.ndarray, Fd: np.ndarray, P: np.ndarray, contest: dict,
               dup: np.ndarray | None = None, chunk: int = 250,
               sharp: float | None = None) -> np.ndarray:
    """Simulated ROI per lineup. S: sims x players, Fd: field lineups x slots,
    P: pool lineups x slots."""
    sharp = SHARP_SHARE if sharp is None else sharp
    if sharp > 0 and len(P):
        k = max(1, int(round(sharp * len(Fd) / (1 - sharp) / len(P))))
        Fd = np.vstack([Fd, np.repeat(P, k, axis=0)])
    N = int(contest["entries"])
    pay = payout_table(N, contest["pool"], contest["fee"], contest["first"])
    paid = len(pay)
    ranks = np.unique(np.round(np.logspace(0, np.log10(paid), 80)).astype(int))
    f = ranks / N                                     # top fraction at each rank
    nF = len(Fd)
    emp = f >= 5.0 / nF                               # the sample resolves these
    f0 = 5.0 / nF
    sims = S.shape[0]
    total = np.zeros(len(P))
    for a in range(0, sims, chunk):
        s = S[a:a + chunk]
        fs = s[:, Fd].sum(axis=2)                       # chunk x field
        ls = s[:, P].sum(axis=2)                        # chunk x lineups
        thr = np.empty((len(s), len(ranks)))
        if emp.any():
            thr[:, emp] = np.quantile(fs, 1 - f[emp], axis=1).T
        if (~emp).any():
            # Beyond the sample: an exponential tail fitted between the top
            # 1% and the top 5-of-the-sample, extended in log(1/f).
            q0 = np.quantile(fs, 1 - f0, axis=1)
            q1 = np.quantile(fs, 0.99, axis=1)
            b = (q0 - q1) / np.log(0.01 / f0)
            thr[:, ~emp] = q0[:, None] + b[:, None] * np.log(f0 / f[~emp])[None, :]
        # The best rank whose threshold the lineup beats (thresholds fall as
        # rank grows), then interpolate between that rank and the one above
        # it in log-rank by where the score sits between the two thresholds -
        # paying the bracket's worse end would bias every ROI low.
        beat = ls[:, :, None] > thr[:, None, :]           # chunk x lineups x ranks
        any_beat = beat.any(axis=2)
        k = np.argmax(beat, axis=2)
        kk = np.clip(k - 1, 0, None)
        t_lo = thr[np.arange(len(s))[:, None], k]
        t_hi = thr[np.arange(len(s))[:, None], kk]
        span = np.where(t_hi > t_lo, t_hi - t_lo, 1.0)
        frac = np.clip((ls - t_lo) / span, 0, 1)
        lr = np.log(ranks[k]) - frac * (np.log(ranks[k]) - np.log(ranks[kk]))
        rank = np.where(k == 0, 1, np.clip(np.round(np.exp(lr)).astype(int), 1, paid))
        prize = np.where(any_beat, pay[rank - 1], 0.0)
        total += prize.sum(axis=0)
    mean_pay = total / sims
    if dup is not None:
        mean_pay = mean_pay / (1.0 + np.clip(dup, 0, None))
    return mean_pay / contest["fee"] - 1.0


# For backtests, where the night's real lobby is gone: a typical big GPP.
STANDARD_GPP = TIERS["large"]


def attach(payload: dict, inner: dict | None, contest: dict | None = None) -> None:
    """Add L["roi"] (simulated ROI %, featured contest) and payload["contest"]."""
    if not inner or not payload.get("lineups"):
        return
    c = contest or featured(payload.get("draft_group"))
    if not c:
        log.info("no featured tournament found for draft group %s; ROI not computed",
                 payload.get("draft_group"))
        return
    pool = payload["lineups"]
    P = np.array([L["p"] for L in pool])
    field_size = float(payload.get("field_size") or 100_000)
    dup = np.array([float(L.get("dup") or 0.0) for L in pool]) * (c["entries"] / field_size)
    try:
        roi = lineup_roi(inner["S"], inner["Fd"], P, c, dup)
    except Exception as exc:                                   # noqa: BLE001
        log.warning("ROI failed (%s)", str(exc)[:80])
        return
    for L, r in zip(pool, roi):
        L["roi"] = round(float(r) * 100, 1)
    payload["contest"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items()}
    best = max(pool, key=lambda L: L["roi"])
    log.info("ROI vs %s (%d entries, $%.0f): best lineup %+.0f%%",
             c["name"], c["entries"], c["fee"], best["roi"])
    tiers = {t: dict(v, real=False) for t, v in TIERS.items()} if contest is not None \
        else tier_contests(payload.get("draft_group"))
    attach_tiers(payload, inner, tiers, featured_contest=c, featured_roi=roi)


def attach_tiers(payload: dict, inner: dict, tiers: dict, featured_contest=None,
                 featured_roi=None) -> None:
    """L["roit"] = {tier: simulated ROI %} for the small / mid / large contest
    of the slate, and payload["tiers"] describing each."""
    pool = payload["lineups"]
    P = np.array([L["p"] for L in pool])
    field_size = float(payload.get("field_size") or 100_000)
    base_dup = np.array([float(L.get("dup") or 0.0) for L in pool])
    res = {}
    for t, c in tiers.items():
        if featured_contest is not None and featured_roi is not None and \
                c.get("name") == featured_contest.get("name") and c.get("entries") == featured_contest.get("entries"):
            res[t] = featured_roi
            continue
        try:
            res[t] = lineup_roi(inner["S"], inner["Fd"], P, c, base_dup * (c["entries"] / field_size))
        except Exception as exc:                               # noqa: BLE001
            log.warning("tier ROI %s failed (%s)", t, str(exc)[:80])
    for i, L in enumerate(pool):
        L["roit"] = {t: round(float(r[i]) * 100, 1) for t, r in res.items()}
    payload["tiers"] = {t: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items()}
                        for t, c in tiers.items() if t in res}
