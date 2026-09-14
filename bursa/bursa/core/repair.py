"""Repair and quarantine faults found by validate.py.

Written after running the checks against real Bursa data, which turned up
three distinct problems in a 58-ticker panel. They are not hypothetical:

* **VITROX (0097) fell exactly 50.1% on 2024-05-02** -- in BOTH the raw and
  the adjusted series, at a ratio of 1.995. That is an unadjusted 2-for-1
  split, not a crash. A momentum model reads it as the worst-performing stock
  in the universe; a reversal model buys it.

* **FRONTKEN (0208) has adj_close identical to close across 1,781 bars.**
  57 of the other 58 tickers show a median divergence of RM 1.38. Frontken's
  dividends were never adjusted, so every ex-dividend date is a fake negative
  return. Nothing about the series looks wrong.

* **One bar of CelcomDigi (6947) has close 4.49 above a high of 4.48.**
  A one-sen data error, but any high/low feature computed from it is nonsense.

The governing rule here: **never repair silently.** Every change is recorded
in a RepairReport, and a repair that cannot be made confidently quarantines
the ticker instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# A move this large in the ADJUSTED series is a split candidate.
SPLIT_MIN_MOVE = 0.30

# Ratios a corporate action produces. A detected move must land within
# tolerance of one of these to be treated as a split rather than a real move.
KNOWN_RATIOS = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0)

# Thresholds calibrated against real Bursa data, not chosen by feel. Measured
# on three candidates in a 58-ticker panel:
#
#   0097 2024-05-02  REAL SPLIT     0.7% off 2.0    6.5x volume   +0.0% after
#   5202 2020-04-20  news rally     3.6% off 1.5   16.3x volume   +2.5% after
#   5202 2021-02-25  speculation    2.7% off 1.5   58.6x volume  +53.2% after
#
# An earlier 4% tolerance accepted both 5202 moves and rewrote genuine price
# history -- the exact error this module warns about. All three gates below
# must pass, which is deliberately conservative: a MISSED split surfaces as a
# flagged extreme move for a human to look at, while a FALSE positive silently
# fabricates prices.
RATIO_TOL = 0.015                # 2.014 counts as 2.0; 1.541 does not
MAX_VOLUME_SPIKE = 10.0          # vs the prior 20-day median
MAX_DRIFT_AFTER = 0.10           # a split re-denominates; it does not trend
DRIFT_WINDOW = 3                 # bars

PRICE_COLS = ("open", "high", "low", "close", "adj_close")


@dataclass
class RepairReport:
    """Everything that was changed, and everything that was refused."""
    splits_fixed: pd.DataFrame = field(default_factory=pd.DataFrame)
    quarantined: dict[str, str] = field(default_factory=dict)
    bars_dropped: pd.DataFrame = field(default_factory=pd.DataFrame)
    suspect_moves: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __str__(self) -> str:
        lines = ["REPAIR REPORT"]
        if len(self.splits_fixed):
            lines.append(f"  splits back-adjusted: {len(self.splits_fixed)}")
            for _, r in self.splits_fixed.iterrows():
                lines.append(f"    {r.ticker} {r.date.date()}  "
                             f"ratio {r.ratio:.3f}  ({r.move:+.1%})")
        else:
            lines.append("  splits back-adjusted: none")

        if self.quarantined:
            lines.append(f"  quarantined tickers: {len(self.quarantined)}")
            for t, why in self.quarantined.items():
                lines.append(f"    {t:<12} {why}")
        else:
            lines.append("  quarantined tickers: none")

        lines.append(f"  bars dropped: {len(self.bars_dropped)}")
        if len(self.suspect_moves):
            lines.append(f"  large moves left alone (look real): "
                         f"{len(self.suspect_moves)}")
            for _, r in self.suspect_moves.iterrows():
                lines.append(f"    {r.ticker} {r.date.date()}  {r.move:+.1%}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
def detect_splits(panel: pd.DataFrame) -> pd.DataFrame:
    """Find large moves that look like unadjusted corporate actions.

    The test is deliberately narrow. A move qualifies only if it is large AND
    lands within tolerance of a known split ratio AND appears in the adjusted
    series too -- a genuine adjustment leaves adj_close smooth, so a jump that
    survives adjustment is a jump nobody adjusted for.

    Everything else is reported as a suspect move and left alone. Rewriting a
    real 45% crash as a split would be far worse than missing one.
    """
    out = panel.sort_values(["ticker", "date"]).copy()
    g = out.groupby("ticker", observed=True)
    out["adj_ret"] = g["adj_close"].pct_change()
    # Volume relative to its own recent normal, and where the price sits a few
    # bars later. Both are shifted/forward-looking only WITHIN the diagnostic --
    # this runs once over history, not inside a signal.
    out["vol_x"] = out["volume"] / g["volume"].transform(
        lambda v: v.shift(1).rolling(20, min_periods=5).median())
    out["drift_after"] = (g["adj_close"].shift(-DRIFT_WINDOW)
                          / out["adj_close"] - 1.0)

    cols = ["ticker", "date", "ratio", "move", "vol_x", "drift_after",
            "ratio_ok", "volume_ok", "stable_after", "is_split"]
    big = out[out["adj_ret"].abs() > SPLIT_MIN_MOVE].copy()
    if big.empty:
        return pd.DataFrame(columns=cols)

    # A -50% move implies a 2:1 split (ratio 2); a +100% move implies 1:2.
    big["ratio"] = np.where(big["adj_ret"] < 0,
                            1.0 / (1.0 + big["adj_ret"]),
                            1.0 + big["adj_ret"])

    big["ratio_ok"] = [
        any(abs(r - k) / k < RATIO_TOL for k in KNOWN_RATIOS)
        for r in big["ratio"]
    ]
    # A split roughly scales share volume by the ratio; it does not cause a
    # 50x day. Missing volume data fails open rather than blocking a repair.
    big["volume_ok"] = ~(big["vol_x"] > MAX_VOLUME_SPIKE)
    # A re-denomination goes nowhere afterwards. A rally keeps running.
    big["stable_after"] = ~(big["drift_after"].abs() > MAX_DRIFT_AFTER)

    big["is_split"] = big["ratio_ok"] & big["volume_ok"] & big["stable_after"]
    big = big.rename(columns={"adj_ret": "move"})
    return big[cols].reset_index(drop=True)


def apply_split_adjustment(panel: pd.DataFrame,
                           splits: pd.DataFrame) -> pd.DataFrame:
    """Back-adjust prices BEFORE each split date so the series is continuous.

    Prices before the split are divided by the ratio and volumes multiplied by
    it, which is the standard convention: the adjusted history reflects what
    the position would be worth in post-split shares.
    """
    out = panel.sort_values(["ticker", "date"]).copy()
    # Volume arrives as int64 and a split factor is fractional. Writing floats
    # into an int column raises under pandas 3, and silently truncates under
    # older versions -- which would be worse. Carry it as float throughout.
    out["volume"] = out["volume"].astype("float64")

    for _, s in splits[splits["is_split"]].iterrows():
        mask = (out["ticker"] == s.ticker) & (out["date"] < s.date)
        if not mask.any():
            continue
        # A negative move (price halved) means MORE shares: divide old prices.
        factor = s.ratio if s.move < 0 else 1.0 / s.ratio
        for c in PRICE_COLS:
            out.loc[mask, c] = out.loc[mask, c] / factor
        out.loc[mask, "volume"] = out.loc[mask, "volume"] * factor
    return out


# --------------------------------------------------------------------------
# Quarantine
# --------------------------------------------------------------------------
def find_unadjusted_tickers(panel: pd.DataFrame,
                            tol: float = 1e-9) -> list[str]:
    """Tickers whose adjusted series never differs from the raw one.

    Over several years of a dividend-paying stock the two must diverge. If
    they do not, the adjustment was never applied and every ex-dividend date
    is a fake loss. That cannot be repaired without a dividend feed, so the
    ticker is excluded rather than silently trusted.
    """
    div = panel.groupby("ticker", observed=True).apply(
        lambda d: float((d["close"] - d["adj_close"]).abs().max()),
        include_groups=False)
    return sorted(div[div <= tol].index.tolist())


# --------------------------------------------------------------------------
# Bad bars
# --------------------------------------------------------------------------
def find_bad_bars(panel: pd.DataFrame) -> pd.DataFrame:
    """Bars where high/low do not bracket open/close, or a price is <= 0."""
    bad = panel[
        (panel["high"] < panel["low"])
        | (panel["high"] < panel[["open", "close"]].max(axis=1))
        | (panel["low"] > panel[["open", "close"]].min(axis=1))
        | (panel[list(PRICE_COLS)] <= 0).any(axis=1)
    ]
    return bad[["date", "ticker", "open", "high", "low", "close"]]


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def repair(panel: pd.DataFrame,
           fix_splits: bool = True,
           quarantine_unadjusted: bool = True,
           drop_bad_bars: bool = True,
           verbose: bool = True) -> tuple[pd.DataFrame, RepairReport]:
    """Clean the panel and report every change.

    Returns (repaired_panel, report). The report is the point -- a silent
    cleaner is indistinguishable from a bug.
    """
    report = RepairReport()
    out = panel.copy()

    detected = detect_splits(out)
    if len(detected):
        report.splits_fixed = detected[detected["is_split"]].copy()
        report.suspect_moves = detected[~detected["is_split"]].copy()
        if fix_splits and len(report.splits_fixed):
            out = apply_split_adjustment(out, report.splits_fixed)

    if quarantine_unadjusted:
        for t in find_unadjusted_tickers(out):
            report.quarantined[t] = ("adj_close == close throughout; "
                                     "dividends never adjusted")
        if report.quarantined:
            out = out[~out["ticker"].isin(report.quarantined)]

    if drop_bad_bars:
        bad = find_bad_bars(out)
        report.bars_dropped = bad
        if len(bad):
            key = set(zip(bad["ticker"], bad["date"]))
            out = out[~pd.Series(list(zip(out["ticker"], out["date"])),
                                 index=out.index).isin(key)]

    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    if verbose:
        print(report)
        print(f"\n  {len(panel):,} bars in, {len(out):,} out "
              f"({panel['ticker'].nunique()} -> {out['ticker'].nunique()} tickers)")
    return out, report
