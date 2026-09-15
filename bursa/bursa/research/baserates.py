"""What actually happened to stocks that looked like this one.

A prediction with error bars, built from history rather than from a model.

## The idea

You want to know whether to buy 0270. The honest answer is not a direction,
it is a distribution: *of every stock that has ever sat in the top momentum
decile of this universe, here is what the next 21 days did to them* -- the
median, the quartiles, the tails, and how often it went badly.

That is a real forecast. It just reports its own uncertainty instead of
hiding it behind a single number, which is what the user asked for: the
prediction can be wrong, it only has to be honest about how wrong it tends
to be.

## The four ways this could quietly lie, and what stops each

**Bucketing across time instead of across stocks.** A stock's momentum decile
has to be computed against the OTHER STOCKS ON THE SAME DAY, not against all
of history. Rank against history and in a bull market every name lands in the
top decile, so the "top decile" bucket is really "2021", and its forward
returns measure a regime rather than a signal. `bucket_cross_sectionally`
ranks within each date.

**Lookahead.** The factor may use bars up to and including t. The forward
return covers t+1 to t+1+horizon, and the fill is at t+1 -- the same
signal-then-fill rule the backtester enforces. A factor that peeked at t+1
would make every bucket look predictive.

**Overlapping windows.** Sampling daily with a 21-day forward return means
consecutive observations share 20 of their 21 days. Two hundred thousand rows
are NOT two hundred thousand independent facts; at best there are
`n_dates / horizon` independent periods, and even that overstates it because
names move together. Every result therefore reports `independent_periods`
alongside `n`, and this module never computes a t-statistic or a p-value --
the honest sample is small enough that one would be theatre.

**Survivorship.** The panel still cannot see roughly 52 companies that
delisted, because Yahoo drops their history. The left tail here is
understated, and the `p_loss_30` figures are floors.

## What it deliberately does not do

Recommend. It answers "what happened to names like this", and what to do
about that is a decision, not an output.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_HORIZON = 21          # trading days: the contest length
DEFAULT_LOOKBACK = 60         # bars the factor may look at
DEFAULT_BUCKETS = 10          # deciles

# A bucket thinner than this is an anecdote. Reporting quartiles on nine
# observations invites exactly the false confidence this module exists to
# avoid.
MIN_BUCKET_OBS = 100


@dataclass(frozen=True)
class BucketStats:
    """The forward-return distribution of one factor bucket."""
    bucket: int
    n: int
    independent_periods: float
    median: float
    mean: float
    p05: float
    p25: float
    p75: float
    p95: float
    p_positive: float
    p_gain_20: float
    p_loss_20: float
    p_loss_30: float

    @property
    def thin(self) -> bool:
        return self.n < MIN_BUCKET_OBS

    def __str__(self) -> str:
        warn = "  [THIN]" if self.thin else ""
        return (f"  decile {self.bucket:>2}  n={self.n:>6,}  "
                f"median {self.median:>+7.2%}  "
                f"[{self.p05:>+7.2%} .. {self.p95:>+7.2%}]  "
                f"up {self.p_positive:>5.1%}{warn}")


def forward_returns(panel: pd.DataFrame, horizon: int = DEFAULT_HORIZON,
                    price_col: str = "adj_close") -> pd.DataFrame:
    """Add the return from the NEXT bar to `horizon` bars after that.

    Entry at t+1, not t. A signal seen at the close of t is acted on the
    following bar, so the return that matters starts there -- the same rule
    `research.backtest` enforces. Measuring from t would hand the strategy
    the overnight gap it could never have captured, which is the single most
    common way a study like this invents an edge.
    """
    out = panel.sort_values(["ticker", "date"]).copy()
    g = out.groupby("ticker", observed=True)[price_col]
    entry = g.shift(-1)
    exit_ = g.shift(-(1 + horizon))
    out["fwd_return"] = exit_ / entry - 1.0
    return out


def add_factor(panel: pd.DataFrame, factor: str = "momentum",
               lookback: int = DEFAULT_LOOKBACK,
               price_col: str = "adj_close") -> pd.DataFrame:
    """Compute a factor using only bars up to and including each date."""
    out = panel.sort_values(["ticker", "date"]).copy()
    g = out.groupby("ticker", observed=True)[price_col]

    if factor == "momentum":
        out["factor"] = g.transform(lambda s: s / s.shift(lookback) - 1.0)
    elif factor in ("volatility", "high_vol"):
        out["factor"] = g.transform(
            lambda s: s.pct_change().rolling(lookback).std())
    elif factor == "reversal":
        out["factor"] = -g.transform(lambda s: s / s.shift(lookback) - 1.0)
    else:
        raise ValueError(f"unknown factor {factor!r}")
    return out


def bucket_cross_sectionally(panel: pd.DataFrame,
                             n_buckets: int = DEFAULT_BUCKETS,
                             eligible_col: str = "eligible") -> pd.DataFrame:
    """Rank each date's factor values against EACH OTHER, then bucket.

    Cross-sectional, not time-series. This is the load-bearing choice in the
    module: ranking against history would make "top decile" mean "a bull
    market" and the forward returns would measure the regime rather than the
    factor. Ranking within the date asks the only question that can be acted
    on -- is this name high-momentum RELATIVE TO WHAT ELSE I COULD BUY TODAY.
    """
    out = panel.copy()
    if eligible_col in out.columns:
        out = out[out[eligible_col]]
    out = out[out["factor"].notna() & out["fwd_return"].notna()]
    if out.empty:
        return out.assign(bucket=pd.Series(dtype=int))

    # A date with fewer names than buckets cannot be split into deciles
    # meaningfully; qcut would produce ragged, incomparable groups.
    counts = out.groupby("date", observed=True)["ticker"].transform("size")
    out = out[counts >= n_buckets * 2]
    if out.empty:
        return out.assign(bucket=pd.Series(dtype=int))

    out["bucket"] = (
        out.groupby("date", observed=True)["factor"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), n_buckets,
                                     labels=False, duplicates="drop") + 1)
    )
    return out.dropna(subset=["bucket"]).astype({"bucket": int})


def summarise_buckets(bucketed: pd.DataFrame,
                      horizon: int = DEFAULT_HORIZON) -> list[BucketStats]:
    """Forward-return distribution per bucket, with the sample size told
    honestly."""
    stats = []
    for b, g in bucketed.groupby("bucket", observed=True):
        r = g["fwd_return"].to_numpy()
        # Overlapping daily windows are not independent draws. The number of
        # non-overlapping periods the data actually spans is the honest
        # denominator, and it is far smaller than len(r).
        n_dates = g["date"].nunique()
        stats.append(BucketStats(
            bucket=int(b), n=len(r),
            independent_periods=n_dates / horizon,
            median=float(np.median(r)), mean=float(np.mean(r)),
            p05=float(np.percentile(r, 5)), p25=float(np.percentile(r, 25)),
            p75=float(np.percentile(r, 75)), p95=float(np.percentile(r, 95)),
            p_positive=float((r > 0).mean()),
            p_gain_20=float((r > 0.20).mean()),
            p_loss_20=float((r < -0.20).mean()),
            p_loss_30=float((r < -0.30).mean()),
        ))
    return sorted(stats, key=lambda s: s.bucket)


def base_rates(panel: pd.DataFrame, factor: str = "momentum",
               horizon: int = DEFAULT_HORIZON,
               lookback: int = DEFAULT_LOOKBACK,
               n_buckets: int = DEFAULT_BUCKETS) -> list[BucketStats]:
    """The whole pipeline: factor, forward return, cross-sectional bucket."""
    p = add_factor(panel, factor, lookback)
    p = forward_returns(p, horizon)
    return summarise_buckets(bucket_cross_sectionally(p, n_buckets), horizon)


def current_bucket(panel: pd.DataFrame, ticker: str,
                   factor: str = "momentum",
                   lookback: int = DEFAULT_LOOKBACK,
                   n_buckets: int = DEFAULT_BUCKETS,
                   as_of: pd.Timestamp | None = None) -> dict:
    """Where one ticker sits today, relative to everything else tradable.

    Returns the bucket, the raw factor value, and the rank -- so the report
    can say "1st of 195" rather than only "decile 10", which reads very
    differently when the decile holds twenty names.
    """
    p = add_factor(panel, factor, lookback)
    as_of = as_of or p["date"].max()
    today = p[p["date"] == as_of]
    if "eligible" in today.columns:
        today = today[today["eligible"]]
    today = today[today["factor"].notna()]
    if today.empty:
        return {"found": False, "reason": f"no eligible names on {as_of}"}

    today = today.sort_values("factor", ascending=False).reset_index(drop=True)
    hit = today.index[today["ticker"] == ticker]
    if len(hit) == 0:
        return {"found": False, "as_of": as_of, "n_universe": len(today),
                "reason": f"{ticker} is not eligible on {as_of.date()}"}

    rank = int(hit[0]) + 1
    n = len(today)
    # Bucket 1 is the WORST, n_buckets the best -- matching the ascending
    # qcut in bucket_cross_sectionally, so the two agree.
    bucket = int(np.ceil((n - rank + 1) / n * n_buckets))
    return {"found": True, "as_of": as_of, "ticker": ticker,
            "rank": rank, "n_universe": n, "bucket": max(1, bucket),
            "factor_value": float(today.loc[hit[0], "factor"]),
            "percentile": (n - rank + 1) / n}
