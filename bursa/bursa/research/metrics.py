"""Performance statistics, including the one that matters most.

Most of this file is standard: Sharpe, drawdown, Calmar. The part worth
reading is `deflated_sharpe`, which answers the question every backtest
silently begs -- "given how many strategies I tried, how surprised should I
be by the best one?"

The answer is usually "not very". Trying 50 variants and reporting the best
Sharpe without correction is the single most common way a retail backtest
lies, and it is why `results/` is committed to git: the trial count has to
come from somewhere honest.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
EULER_MASCHERONI = 0.5772156649015329


# --------------------------------------------------------------------------
# Basics
# --------------------------------------------------------------------------
def total_return(returns: pd.Series) -> float:
    """Compounded return over the whole period."""
    return float((1.0 + returns.fillna(0.0)).prod() - 1.0)


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised compound growth rate."""
    n = len(returns.dropna())
    if n == 0:
        return 0.0
    growth = 1.0 + total_return(returns)
    if growth <= 0:
        return -1.0
    return float(growth ** (periods_per_year / n) - 1.0)


def volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised standard deviation of returns."""
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe(returns: pd.Series,
           rf: float = 0.0,
           periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised Sharpe ratio. rf is an annual rate."""
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    excess = r - rf / periods_per_year
    sd = excess.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(excess.mean() / sd * math.sqrt(periods_per_year))


def sortino(returns: pd.Series,
            rf: float = 0.0,
            periods_per_year: int = TRADING_DAYS) -> float:
    """Like Sharpe, but only downside deviation is treated as risk."""
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    excess = r - rf / periods_per_year
    downside = excess[excess < 0]
    if len(downside) < 2 or downside.std(ddof=1) == 0:
        return 0.0
    return float(excess.mean() / downside.std(ddof=1) * math.sqrt(periods_per_year))


def equity_curve(returns: pd.Series, start: float = 1.0) -> pd.Series:
    """Cumulative wealth from a return series."""
    return start * (1.0 + returns.fillna(0.0)).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline, as a negative fraction."""
    eq = equity_curve(returns)
    if eq.empty:
        return 0.0
    return float((eq / eq.cummax() - 1.0).min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    eq = equity_curve(returns)
    return eq / eq.cummax() - 1.0


def calmar(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """CAGR divided by max drawdown. Return per unit of worst pain."""
    mdd = abs(max_drawdown(returns))
    return float(cagr(returns, periods_per_year) / mdd) if mdd > 0 else 0.0


def hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with a positive return.

    Deliberately reported alongside Sharpe, never instead of it: you can win
    70% of the time and still lose money if the 30% are the big moves.
    """
    r = returns.dropna()
    return float((r > 0).mean()) if len(r) else 0.0


# --------------------------------------------------------------------------
# Multiple testing: the correction nobody applies
# --------------------------------------------------------------------------
def probabilistic_sharpe(observed_sr: float,
                         benchmark_sr: float,
                         n_obs: int,
                         skew: float = 0.0,
                         kurtosis: float = 3.0) -> float:
    """P(true Sharpe > benchmark), given the sample and its shape.

    Returns are not normal -- they are skewed and fat-tailed -- and both
    inflate the uncertainty around a measured Sharpe. This accounts for it.
    """
    if n_obs < 2:
        return 0.0
    denom = 1.0 - skew * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr ** 2
    if denom <= 0:
        return 0.0
    z = (observed_sr - benchmark_sr) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_variance: float = 1.0) -> float:
    """The Sharpe you would expect from the BEST of n worthless strategies.

    This is the number that should terrify anyone who has run a parameter
    sweep. Test enough variants of nothing and one of them looks good --
    not because it is, but because you looked many times.
    """
    if n_trials < 2:
        return 0.0
    sd = math.sqrt(sr_variance)
    g = EULER_MASCHERONI
    return float(sd * ((1 - g) * stats.norm.ppf(1 - 1.0 / n_trials)
                       + g * stats.norm.ppf(1 - 1.0 / (n_trials * math.e))))


def deflated_sharpe(returns: pd.Series,
                    n_trials: int,
                    periods_per_year: int = TRADING_DAYS) -> float:
    """P(the strategy's true Sharpe > 0), after correcting for `n_trials`.

    Bailey & Lopez de Prado. Read it as a probability: 0.95 means the result
    survives the fact that you went looking. Below ~0.90 and the honest
    reading is that you found the best of several coin flips.

    `n_trials` must be every variant you actually tested, not the ones you
    kept. That is what the git history is for.
    """
    r = returns.dropna()
    if len(r) < 3:
        return 0.0
    sr = sharpe(r, periods_per_year=periods_per_year)
    # Convert to the per-period Sharpe the formula expects.
    sr_period = sr / math.sqrt(periods_per_year)
    sr_star = expected_max_sharpe(n_trials, sr_variance=1.0 / (len(r) - 1))

    # A (near) constant series has undefined higher moments -- scipy returns
    # NaN via catastrophic cancellation, which would propagate silently into
    # the result. Fall back to normal assumptions; with no variance there is
    # no shape to correct for anyway.
    if r.std(ddof=1) < 1e-12:
        skew_, kurt_ = 0.0, 3.0
    else:
        skew_ = float(stats.skew(r))
        kurt_ = float(stats.kurtosis(r, fisher=False))
        if not (np.isfinite(skew_) and np.isfinite(kurt_)):
            skew_, kurt_ = 0.0, 3.0

    return probabilistic_sharpe(sr_period, sr_star, len(r),
                                skew=skew_, kurtosis=kurt_)


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
@dataclass
class Performance:
    """Everything about one return stream, in one object."""
    name: str
    n_periods: int
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    turnover: float = 0.0
    n_trials: int = 1
    deflated_sharpe: float = float("nan")

    def as_row(self) -> dict:
        return {
            "strategy": self.name,
            "total_return": self.total_return,
            "cagr": self.cagr,
            "vol": self.volatility,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_dd": self.max_drawdown,
            "calmar": self.calmar,
            "hit_rate": self.hit_rate,
            "turnover": self.turnover,
            "dsr": self.deflated_sharpe,
        }

    def __str__(self) -> str:
        dsr = ("" if math.isnan(self.deflated_sharpe)
               else f"  DSR {self.deflated_sharpe:.3f} (n_trials={self.n_trials})")
        return (f"{self.name:<28} ret {self.total_return:+7.2%}  "
                f"CAGR {self.cagr:+6.2%}  vol {self.volatility:5.2%}  "
                f"SR {self.sharpe:+5.2f}  maxDD {self.max_drawdown:6.2%}{dsr}")


def summarise(returns: pd.Series,
              name: str = "strategy",
              turnover: float = 0.0,
              n_trials: int = 1,
              periods_per_year: int = TRADING_DAYS) -> Performance:
    """Compute every statistic for one return series."""
    r = returns.dropna()
    return Performance(
        name=name,
        n_periods=len(r),
        total_return=total_return(r),
        cagr=cagr(r, periods_per_year),
        volatility=volatility(r, periods_per_year),
        sharpe=sharpe(r, periods_per_year=periods_per_year),
        sortino=sortino(r, periods_per_year=periods_per_year),
        max_drawdown=max_drawdown(r),
        calmar=calmar(r, periods_per_year),
        hit_rate=hit_rate(r),
        turnover=turnover,
        n_trials=n_trials,
        deflated_sharpe=(deflated_sharpe(r, n_trials, periods_per_year)
                         if n_trials > 1 else float("nan")),
    )


def compare(performances: list[Performance]) -> pd.DataFrame:
    """Side-by-side table. Strategy first, then every baseline it must beat."""
    return pd.DataFrame([p.as_row() for p in performances]).set_index("strategy")
