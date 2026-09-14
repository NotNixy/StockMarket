"""Data integrity checks for the price panel.

Modelled on the accounting-identity checks used on the PAIP extract: a fixed
set of named assertions, each either passing or producing the rows that broke
it. The point is to be able to answer "how do I know this data is right?"
with something other than "it looked fine".

Every check here corresponds to a way a Bursa panel has actually been wrong,
and several of them produce fake trading signal rather than obvious garbage --
which is why they run before any strategy work, not after a disappointing
backtest.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Single-day move beyond this is implausible for a liquid stock and is
# reviewed by hand. Bursa's static limit is wider, but a real 35% day on a
# screened name is rare enough to be worth a look.
EXTREME_MOVE = 0.35

# Ratios a missed split would produce. Approximate -- checked with tolerance.
SPLIT_SIGNATURES = (-0.500, -0.667, -0.750, -0.800, 1.000, 2.000)
SPLIT_TOL = 0.02


@dataclass
class CheckResult:
    name: str
    passed: bool
    n_bad: int
    message: str
    sample: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name:<28} {self.message}"


def _result(name: str, bad: pd.DataFrame, message_ok: str,
            message_bad: str, cols: list[str] | None = None) -> CheckResult:
    n = len(bad)
    if n == 0:
        return CheckResult(name, True, 0, message_ok)
    sample = bad.head(5)
    if cols:
        sample = sample[[c for c in cols if c in sample.columns]]
    return CheckResult(name, False, n, message_bad.format(n=n), sample)


# --------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------
def check_ohlc_brackets(panel: pd.DataFrame) -> CheckResult:
    """High must be the highest and low the lowest of the four prices.

    A violation means the bar is corrupt, and any high/low-based feature
    computed from it is meaningless.
    """
    bad = panel[(panel["high"] < panel["low"])
                | (panel["high"] < panel[["open", "close"]].max(axis=1))
                | (panel["low"] > panel[["open", "close"]].min(axis=1))]
    return _result("ohlc brackets", bad,
                   "high/low bracket open and close on every bar",
                   "{n} bars where high/low do not bracket open/close",
                   ["date", "ticker", "open", "high", "low", "close"])


def check_positive_prices(panel: pd.DataFrame) -> CheckResult:
    """No zero or negative prices. A zero close makes returns infinite."""
    price_cols = ["open", "high", "low", "close", "adj_close"]
    bad = panel[(panel[price_cols] <= 0).any(axis=1)]
    return _result("positive prices", bad,
                   "all prices strictly positive",
                   "{n} bars with a zero or negative price",
                   ["date", "ticker"] + price_cols)


def check_no_duplicates(panel: pd.DataFrame) -> CheckResult:
    """One bar per ticker per day. Duplicates double-count in any groupby."""
    bad = panel[panel.duplicated(["ticker", "date"], keep=False)]
    return _result("no duplicate bars", bad,
                   "one bar per ticker per date",
                   "{n} duplicated (ticker, date) rows",
                   ["date", "ticker", "close"])


def check_no_missing(panel: pd.DataFrame) -> CheckResult:
    """Nulls in the columns everything downstream depends on."""
    cols = ["date", "ticker", "close", "adj_close", "volume"]
    bad = panel[panel[cols].isna().any(axis=1)]
    return _result("no missing values", bad,
                   "no nulls in date/ticker/close/adj_close/volume",
                   "{n} rows with a null in a required column",
                   cols)


def check_extreme_moves(panel: pd.DataFrame) -> CheckResult:
    """Single-day moves beyond +/-35%, flagged for human review.

    These are not automatically wrong -- small caps do move like this. But an
    unadjusted split looks *identical* to a genuine crash, and a ranking model
    will happily treat it as the strongest signal in the universe. Anything
    landing near a canonical split ratio is called out specifically.
    """
    out = panel.sort_values(["ticker", "date"]).copy()
    out["adj_ret"] = out.groupby("ticker", observed=True)["adj_close"].pct_change()
    bad = out[out["adj_ret"].abs() > EXTREME_MOVE].copy()

    if bad.empty:
        return CheckResult("extreme moves", True, 0,
                           f"no single-day move beyond {EXTREME_MOVE:.0%}")

    bad["near_split_ratio"] = [
        any(abs(r - s) < SPLIT_TOL for s in SPLIT_SIGNATURES)
        for r in bad["adj_ret"]
    ]
    n_split = int(bad["near_split_ratio"].sum())
    msg = (f"{len(bad)} moves beyond {EXTREME_MOVE:.0%}"
           + (f", {n_split} near a split ratio -- CHECK THESE" if n_split else ""))
    return CheckResult("extreme moves", False, len(bad), msg,
                       bad.head(8)[["date", "ticker", "close", "adj_close",
                                    "adj_ret", "near_split_ratio"]])


def check_adjustment_applied(panel: pd.DataFrame) -> CheckResult:
    """Raw and adjusted series should differ where corporate actions occurred.

    If adj_close is identical to close for every bar of every ticker across
    years, the adjustment almost certainly did not happen -- dividends alone
    would separate them. That silently turns every ex-dividend date into a
    fake negative return.
    """
    g = panel.groupby("ticker", observed=True)
    identical = g.apply(
        lambda d: np.allclose(d["close"], d["adj_close"], rtol=1e-9),
        include_groups=False)
    bad_tickers = identical[identical].index.tolist()

    if not bad_tickers:
        return CheckResult("adjustment applied", True, 0,
                           "adjusted series differs from raw for every ticker")
    return CheckResult(
        "adjustment applied", False, len(bad_tickers),
        f"{len(bad_tickers)} ticker(s) have adj_close == close throughout",
        pd.DataFrame({"ticker": bad_tickers[:8]}))


def check_calendar_gaps(panel: pd.DataFrame,
                        max_gap_days: int = 10) -> CheckResult:
    """Long unexplained gaps in a ticker's history.

    Short gaps are holidays. A multi-week hole is a suspension, a delisting
    and relisting, or a failed download -- and a feature with a 60-day lookback
    will straddle it without complaint.
    """
    out = panel.sort_values(["ticker", "date"])
    gaps = out.groupby("ticker", observed=True)["date"].diff().dt.days
    bad = out[gaps > max_gap_days].copy()
    bad["gap_days"] = gaps[gaps > max_gap_days]
    return _result("calendar gaps", bad,
                   f"no gap longer than {max_gap_days} calendar days",
                   f"{{n}} gaps longer than {max_gap_days} days",
                   ["date", "ticker", "gap_days"])


def check_zero_volume(panel: pd.DataFrame,
                      max_share: float = 0.20) -> CheckResult:
    """Tickers where too many bars have no volume.

    A zero-volume bar is a halt, a suspension, or a stock nobody trades. The
    close still prints, so it looks like a normal bar to everything downstream
    -- but you could not have traded it.
    """
    share = (panel.assign(z=panel["volume"].fillna(0) <= 0)
                  .groupby("ticker", observed=True)["z"].mean())
    bad = share[share > max_share]
    if bad.empty:
        return CheckResult("zero-volume bars", True, 0,
                           f"no ticker above {max_share:.0%} zero-volume bars")
    return CheckResult(
        "zero-volume bars", False, len(bad),
        f"{len(bad)} ticker(s) above {max_share:.0%} zero-volume bars",
        bad.sort_values(ascending=False).head(8)
           .rename("zero_volume_share").reset_index())


def check_history_depth(panel: pd.DataFrame,
                        min_bars: int = 120) -> CheckResult:
    """Tickers too short to compute a lookback feature on."""
    counts = panel.groupby("ticker", observed=True).size()
    bad = counts[counts < min_bars]
    if bad.empty:
        return CheckResult("history depth", True, 0,
                           f"every ticker has at least {min_bars} bars")
    return CheckResult(
        "history depth", False, len(bad),
        f"{len(bad)} ticker(s) with fewer than {min_bars} bars",
        bad.sort_values().head(8).rename("n_bars").reset_index())


def check_monotonic_dates(panel: pd.DataFrame) -> CheckResult:
    """Dates strictly increasing within each ticker.

    Out-of-order rows make every shift(), rolling() and pct_change() wrong in
    ways that do not raise.
    """
    out = panel.sort_index()
    bad_tickers = [
        t for t, d in out.groupby("ticker", observed=True)
        if not d["date"].is_monotonic_increasing
    ]
    if not bad_tickers:
        return CheckResult("monotonic dates", True, 0,
                           "dates increase within every ticker")
    return CheckResult(
        "monotonic dates", False, len(bad_tickers),
        f"{len(bad_tickers)} ticker(s) with out-of-order dates",
        pd.DataFrame({"ticker": bad_tickers[:8]}))


ALL_CHECKS = (
    check_ohlc_brackets,
    check_positive_prices,
    check_no_duplicates,
    check_no_missing,
    check_monotonic_dates,
    check_adjustment_applied,
    check_extreme_moves,
    check_calendar_gaps,
    check_zero_volume,
    check_history_depth,
)


def validate(panel: pd.DataFrame, verbose: bool = True) -> list[CheckResult]:
    """Run every check. Returns results; never raises on bad data.

    Failures are information, not exceptions -- `check_extreme_moves` in
    particular is expected to "fail" on a real panel and wants reading, not
    fixing.
    """
    results = [check(panel) for check in ALL_CHECKS]
    if verbose:
        print(f"Validating {len(panel):,} bars, "
              f"{panel['ticker'].nunique()} tickers, "
              f"{panel['date'].min().date()} to {panel['date'].max().date()}\n")
        for r in results:
            print(r)
        n_fail = sum(not r.passed for r in results)
        print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
        if n_fail:
            print("\nFailures in detail:")
            for r in results:
                if not r.passed and not r.sample.empty:
                    print(f"\n--- {r.name} ---")
                    print(r.sample.to_string(index=False))
    return results


def assert_clean(panel: pd.DataFrame,
                 allow: tuple[str, ...] = ("extreme moves", "calendar gaps")
                 ) -> None:
    """Raise if any check outside `allow` fails.

    For use at the top of a pipeline run. Extreme moves and calendar gaps are
    tolerated by default because a real panel legitimately contains both;
    everything else indicates the data is broken.
    """
    failures = [r for r in validate(panel, verbose=False)
                if not r.passed and r.name not in allow]
    if failures:
        lines = "\n".join(f"  - {r.name}: {r.message}" for r in failures)
        raise ValueError(f"Panel failed {len(failures)} integrity check(s):\n{lines}")
