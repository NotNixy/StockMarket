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

# Minimum price for a split call. Found by running the heuristic over all 865
# Shariah-compliant tickers: it produced 1,104 "splits", including 30+ on one
# ticker within months, alternating +50% / -33%. Those are penny stocks where
# Bursa's half-sen tick dominates -- RM 0.02 to RM 0.03 IS exactly a 1.5 ratio.
# Below this floor, tick granularity manufactures split ratios out of noise.
MIN_SPLIT_PRICE = 0.50

# Bursa's tick below RM 1.00. A move of only a few ticks is quantisation, not
# a corporate action, however cleanly its ratio lands.
BURSA_TICK = 0.005
MIN_TICKS_MOVED = 20

# Yahoo dated VITROX's split 2024-06-10 while the price stepped 2024-05-02 --
# five weeks apart. So a recorded event confirms a nearby price step rather
# than having to fall on it. Wide enough to cover that, narrow enough that two
# unrelated actions do not confirm each other.
CONFIRM_WINDOW_DAYS = 45

PRICE_COLS = ("open", "high", "low", "close", "adj_close")


@dataclass
class RepairReport:
    """Everything that was changed, and everything that was refused."""
    splits_fixed: pd.DataFrame = field(default_factory=pd.DataFrame)
    splits_located: pd.DataFrame = field(default_factory=pd.DataFrame)
    unadjusted_noted: list[str] = field(default_factory=list)
    quarantined: dict[str, str] = field(default_factory=dict)
    bars_dropped: pd.DataFrame = field(default_factory=pd.DataFrame)
    suspect_moves: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __str__(self) -> str:
        lines = ["REPAIR REPORT"]
        if len(self.splits_fixed):
            lines.append(f"  splits back-adjusted: {len(self.splits_fixed)}")
            for _, r in self.splits_fixed.head(40).iterrows():
                lines.append(f"    {r.ticker} {r.date.date()}  "
                             f"ratio {r.ratio:.3f}  ({r.move:+.1%})")
            if len(self.splits_fixed) > 40:
                lines.append(f"    ... and {len(self.splits_fixed) - 40} more")
        else:
            lines.append("  splits back-adjusted: none")

        if len(self.splits_located):
            n = len(self.splits_located)
            already = int((~self.splits_located["needs_repair"]).sum())
            lines.append(f"  recorded actions checked: {n}  "
                         f"({already} already applied by the source)")

        if self.quarantined:
            lines.append(f"  quarantined tickers: {len(self.quarantined)}")
            for t, why in list(self.quarantined.items())[:20]:
                lines.append(f"    {t:<12} {why}")
            if len(self.quarantined) > 20:
                lines.append(f"    ... and {len(self.quarantined) - 20} more")
        else:
            lines.append("  quarantined tickers: none")

        if self.unadjusted_noted:
            lines.append(f"  NOTE: {len(self.unadjusted_noted)} tickers have "
                         f"adj_close == close throughout.")
            lines.append("        Either they never paid a dividend or the "
                         "source never adjusted one;")
            lines.append("        the data cannot say which. Kept, and "
                         "declared. See find_unadjusted_tickers.")

        lines.append(f"  bars dropped: {len(self.bars_dropped)}")
        if len(self.suspect_moves):
            lines.append(f"  large moves left alone (look real): "
                         f"{len(self.suspect_moves)}")
            # The near-misses are the interesting ones: a clean split ratio
            # with no corporate action on record. Either the record is
            # incomplete or the move is real, and both are worth a look.
            s = self.suspect_moves
            if "confirmed" in s.columns and "ratio_ok" in s.columns:
                near = s[s["ratio_ok"] & ~s["confirmed"].astype(bool)]
                lines.append(f"    of which clean split ratio but NO action "
                             f"on record: {len(near)}")
            for _, r in s.head(15).iterrows():
                lines.append(f"    {r.ticker} {r.date.date()}  {r.move:+.1%}")
            if len(s) > 15:
                lines.append(f"    ... and {len(s) - 15} more")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Splits, driven from the record
# --------------------------------------------------------------------------
def locate_recorded_splits(panel: pd.DataFrame,
                           known: pd.DataFrame,
                           tol: float = 0.35) -> pd.DataFrame:
    """For each recorded corporate action, find whether it needs repairing.

    This is the inverted, and correct, way round. `detect_splits` asks "does
    this price move look like a split?", which over 865 tickers answers yes
    1,104 times. This asks "the issuer declared a split -- is it already in
    the price series, and if not, where does it land?" There is no inference:
    the record says a split happened and at what ratio, and the only open
    question is whether Yahoo applied it.

    Measured over the Shariah universe: 315 recorded actions big enough to
    see a step, of which Yahoo had already adjusted 292. The 23 it missed are
    what this finds -- including ratios like 1.05 and 1.2 that the 30% move
    threshold in `detect_splits` cannot reach at all.

    `tol` is how close the observed step must be to the expected one, as a
    fraction of the expected step. Loose, because the price also moves on its
    own that day; the record has already established that a split occurred.
    """
    cols = ["ticker", "date", "recorded_date", "ratio", "observed_step",
            "expected_step", "needs_repair"]
    if known is None or not len(known):
        return pd.DataFrame(columns=cols)

    k = known.copy()
    k["date"] = pd.to_datetime(k["date"])
    k["value"] = pd.to_numeric(k["value"], errors="coerce")
    k = k[k["value"].notna() & (k["value"] > 0)]

    out = panel.sort_values(["ticker", "date"]).copy()
    out["adj_ret"] = out.groupby("ticker", observed=True)["adj_close"].pct_change()
    by_ticker = dict(tuple(out.groupby("ticker", observed=True)))

    window = pd.Timedelta(days=CONFIRM_WINDOW_DAYS)
    rows = []
    for _, e in k.iterrows():
        bars = by_ticker.get(e["ticker"])
        if bars is None or len(bars) < 5:
            continue
        # Yahoo value 2.0 = 2-for-1, price halves. value 0.2 = 1-for-5
        # consolidation, price quintuples.
        implied = e["value"] if e["value"] >= 1.0 else 1.0 / e["value"]
        expected = (-(1.0 - 1.0 / implied) if e["value"] > 1.0
                    else implied - 1.0)

        win = bars[(bars["date"] >= e["date"] - window)
                   & (bars["date"] <= e["date"] + window)]
        win = win[win["adj_ret"].notna()]
        if win.empty:
            continue
        # Yahoo dated VITROX's split five weeks after the price stepped, so
        # take the closest MATCHING bar in the window, not the nearest date.
        gap = (win["adj_ret"] - expected).abs()
        j = gap.idxmin()
        needs = bool(gap.loc[j] < abs(expected) * tol)
        rows.append({
            "ticker": e["ticker"], "date": win.loc[j, "date"],
            "recorded_date": e["date"], "ratio": implied,
            "observed_step": float(win.loc[j, "adj_ret"]),
            "expected_step": expected, "needs_repair": needs,
        })

    df = pd.DataFrame(rows, columns=cols)
    # A step big enough to matter. A 0.7% bonus issue left unadjusted is not
    # worth rewriting history over, and the match is unreliable at that size.
    if len(df):
        df.loc[df["expected_step"].abs() < 0.03, "needs_repair"] = False
    return df


# --------------------------------------------------------------------------
# Splits, inferred from price (detector of UNRECORDED anomalies)
# --------------------------------------------------------------------------
def _confirm_from_record(big: pd.DataFrame,
                         known: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Match each candidate price step against the recorded split events.

    Returns (confirmed, recorded_ratio). A step is confirmed only if the
    issuer actually recorded a split for that ticker near that date, at a
    ratio consistent with the step AND in the same direction: a forward split
    (Yahoo value > 1) must show the price falling, a consolidation (value < 1)
    must show it rising. A +50% move with a 2-for-1 split on record is not
    that split.
    """
    n = len(big)
    confirmed = pd.Series(False, index=big.index)
    matched = pd.Series(np.nan, index=big.index)
    if known is None or not len(known):
        return confirmed, matched

    k = known.copy()
    k["date"] = pd.to_datetime(k["date"])
    k["value"] = pd.to_numeric(k["value"], errors="coerce")
    k = k.dropna(subset=["value"])
    k = k[k["value"] > 0]
    by_ticker: dict[str, pd.DataFrame] = dict(tuple(k.groupby("ticker")))

    window = pd.Timedelta(days=CONFIRM_WINDOW_DAYS)
    for idx, row in big.iterrows():
        events = by_ticker.get(row["ticker"])
        if events is None:
            continue
        near = events[(events["date"] - row["date"]).abs() <= window]
        for v in near["value"]:
            # A forward split (value 2.0) halves the price; a consolidation
            # (value 0.2) multiplies it by five. Either way the observed
            # ratio is >= 1, so compare against the value's magnitude.
            implied = v if v >= 1.0 else 1.0 / v
            direction_ok = (row["adj_ret"] < 0) if v > 1.0 else (row["adj_ret"] > 0)
            if direction_ok and abs(row["ratio"] - implied) / implied < RATIO_TOL:
                confirmed.loc[idx] = True
                matched.loc[idx] = implied
                break
    return confirmed, matched


def detect_splits(panel: pd.DataFrame,
                  known_splits: pd.DataFrame | None = None) -> pd.DataFrame:
    """Find large moves that look like unadjusted corporate actions.

    The test is deliberately narrow. A move qualifies only if it is large AND
    lands within tolerance of a known split ratio AND appears in the adjusted
    series too -- a genuine adjustment leaves adj_close smooth, so a jump that
    survives adjustment is a jump nobody adjusted for.

    **Pass `known_splits` whenever you have it.** With a record of actual
    corporate actions in hand, the ratio pattern stops being evidence and
    becomes merely a way to LOCATE a split the issuer already told us about.
    Run without one over 865 tickers, the heuristic produced 1,104 "splits" --
    penny stocks where a single half-sen tick off RM 0.02 is exactly +50%,
    landing dead on the 1.5 ratio, with no volume spike and no drift because
    the price ticks straight back. Supplying the record cuts that to the
    events that actually happened.

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

    # The level the price stepped FROM. A split re-denominates a real price;
    # it cannot happen to a stock trading at two sen.
    out["prev_close"] = g["adj_close"].shift(1)

    cols = ["ticker", "date", "ratio", "move", "vol_x", "drift_after",
            "prev_close", "ratio_ok", "volume_ok", "stable_after",
            "price_ok", "confirmed", "recorded_ratio", "is_split"]
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
    # Both halves of the same idea: the price must be high enough that the
    # tick grid is not what produced the ratio. A 1.5 ratio off RM 0.02 is one
    # tick; off RM 3.00 it is two hundred.
    moved = (big["prev_close"] - big["prev_close"] / big["ratio"]).abs()
    big["price_ok"] = ((big["prev_close"] >= MIN_SPLIT_PRICE)
                       & (moved >= BURSA_TICK * MIN_TICKS_MOVED))

    big["confirmed"], big["recorded_ratio"] = _confirm_from_record(
        big, known_splits)

    heuristic = (big["ratio_ok"] & big["volume_ok"] & big["stable_after"]
                 & big["price_ok"])
    if known_splits is not None and len(known_splits):
        # With a record in hand, nothing is repaired on pattern alone. The
        # heuristic gates still apply -- a confirmed split whose price step
        # looks nothing like one is worth leaving for a human.
        big["is_split"] = heuristic & big["confirmed"]
    else:
        big["is_split"] = heuristic

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
    they do not, either the adjustment was never applied -- making every
    ex-dividend date a fake loss -- or the company has simply never paid a
    dividend, in which case the series is perfectly correct.

    **This function cannot tell those apart, and neither can the data.** The
    obvious test, "does Yahoo have dividends on record for it", returns zero
    for all 227 such names in the Shariah universe INCLUDING Frontken (0208),
    which pays dividends reliably and was the case that motivated writing
    this check. Yahoo simply has no dividend history for Bursa small caps.

    So this reports; it no longer decides. Excluding all 227 would drop 26%
    of the universe, and not at random -- it would drop the names Yahoo
    happens to hold least data on, which skews small and speculative. That is
    a worse and far less visible bias than the residual it avoids. See
    `quarantine_unadjusted` in `repair`.
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
           known_splits: pd.DataFrame | None = None,
           fix_splits: bool = True,
           quarantine_unadjusted: bool = False,
           drop_bad_bars: bool = True,
           verbose: bool = True) -> tuple[pd.DataFrame, RepairReport]:
    """Clean the panel and report every change.

    Returns (repaired_panel, report). The report is the point -- a silent
    cleaner is indistinguishable from a bug.

    `known_splits` is the recorded corporate actions (data/splits.csv, written
    by scripts.fetch_shariah_universe). Supply it. Without it the split fix
    runs on pattern alone, which over a wide universe rewrites real prices.

    `quarantine_unadjusted` defaults to **False** and should usually stay
    there. It drops every ticker whose adj_close never diverges from close,
    which on the Shariah universe is 227 of 865 names -- and the check cannot
    distinguish "dividends never adjusted" from "never paid a dividend". See
    `find_unadjusted_tickers`. The names are still listed in the report, and
    the residual effect (a fake one-day loss on ex-dividend dates for any
    payer among them) belongs in the writeup, not in a silent exclusion.
    """
    report = RepairReport()
    out = panel.copy()
    have_record = known_splits is not None and len(known_splits) > 0

    if have_record:
        # Preferred path: the issuer told us what happened. Repair exactly the
        # recorded actions Yahoo failed to apply, and nothing else.
        located = locate_recorded_splits(out, known_splits)
        report.splits_located = located
        todo = located[located["needs_repair"]].copy()
        if len(todo):
            todo = todo.rename(columns={"observed_step": "move"})
            todo["is_split"] = True
            report.splits_fixed = todo
            if fix_splits:
                out = apply_split_adjustment(out, todo)
        # The heuristic still runs, but only to SURFACE moves that look like
        # unrecorded corporate actions. It no longer rewrites anything.
        detected = detect_splits(out, known_splits)
        if len(detected):
            report.suspect_moves = detected[~detected["is_split"]].copy()
    else:
        detected = detect_splits(out, None)
        if len(detected):
            report.splits_fixed = detected[detected["is_split"]].copy()
            report.suspect_moves = detected[~detected["is_split"]].copy()
            if fix_splits and len(report.splits_fixed):
                out = apply_split_adjustment(out, report.splits_fixed)

    report.unadjusted_noted = find_unadjusted_tickers(out)
    if quarantine_unadjusted:
        for t in report.unadjusted_noted:
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
