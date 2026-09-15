"""The opponents. Build these before any strategy, and know their numbers.

A strategy is not "good" because it made money. It is good if it beats what
you would have got for free. Three opponents, in increasing order of how
uncomfortable they are:

* **buy_and_hold** -- own the universe, equally, and never trade. Zero skill,
  near-zero cost. Most strategies lose to this once costs are charged.

* **equal_weight_rebalanced** -- the same, but rebalanced on your schedule.
  This separates "my signal works" from "rebalancing works", which is a real
  and frequently overlooked effect.

* **random_entry** -- the honest one. Picks the same NUMBER of names, on the
  same DATES, held for the same DURATION, at random. If your clever signal
  cannot beat coin flips at matched exposure, the signal is doing nothing and
  every return you see is the market plus noise.

Run random_entry many times and compare your strategy against the resulting
distribution, not against one draw. A single random run that your strategy
beats proves nothing.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from core.costs import Broker
from research.backtest import BacktestConfig, BacktestResult, run_backtest
from research.metrics import Performance, summarise


def buy_and_hold_signal(history: pd.DataFrame,
                        date: pd.Timestamp,
                        eligible: list[str]) -> dict[str, float]:
    """Equal weight across everything eligible."""
    return {t: 1.0 for t in eligible}


def equal_weight_signal(history: pd.DataFrame,
                        date: pd.Timestamp,
                        eligible: list[str]) -> dict[str, float]:
    """Identical to buy-and-hold; distinct because the config differs.

    Kept as its own name so the comparison table reads honestly -- the only
    difference between the two rows is the rebalance rule and its cost.
    """
    return {t: 1.0 for t in eligible}


def make_random_signal(n_names: int, seed: int = 0):
    """A signal that picks `n_names` eligible tickers uniformly at random.

    The benchmark that actually hurts: same count, same dates, same holding
    period as the real strategy, zero information.
    """
    rng = np.random.default_rng(seed)

    def _signal(history: pd.DataFrame, date: pd.Timestamp,
                eligible: list[str]) -> dict[str, float]:
        if not eligible:
            return {}
        k = min(n_names, len(eligible))
        picks = rng.choice(np.asarray(eligible, dtype=object), size=k,
                           replace=False)
        return {str(t): 1.0 for t in picks}

    return _signal


# A baseline must be something you could actually have done. "Own all 643
# eligible names" is not: at a per-trade minimum fee, entering 643 positions
# on RM 10,000 costs a fifth of the account before a single price moves, and
# rebalancing monthly wipes it out inside two turns. Cap the breadth so that
# ENTERING the basket costs at most this fraction of capital.
MAX_ENTRY_COST_FRACTION = 0.01


def affordable_breadth(capital: float, broker: Broker,
                       hard_cap: int = 200) -> int:
    """How many names this capital can hold without fees dominating.

    Returns at least 1. The binding constraint on a small Bursa account is
    not the share price -- it is the minimum brokerage charged per contract,
    which does not shrink as the position does.
    """
    per_name = max(broker.min_fee, 0.01)
    n = int((capital * MAX_ENTRY_COST_FRACTION) / per_name)
    return max(1, min(n, hard_cap))


def run_buy_and_hold(panel: pd.DataFrame,
                     cfg: BacktestConfig | None = None,
                     max_names: int | None = None) -> BacktestResult:
    """Own a broad equal-weight basket, rebalance once a year.

    As close to free as it gets, and still capped at what the capital can
    actually carry -- see `affordable_breadth`.
    """
    base = cfg or BacktestConfig()
    n = max_names or affordable_breadth(base.capital, base.broker)
    bh = BacktestConfig(capital=base.capital, rebalance="YE",
                        broker=base.broker, slippage=base.slippage,
                        max_positions=n, price_col=base.price_col)
    return run_backtest(panel, buy_and_hold_signal, bh)


def run_equal_weight(panel: pd.DataFrame,
                     cfg: BacktestConfig | None = None,
                     max_names: int | None = None) -> BacktestResult:
    """The same basket, rebalanced on the strategy's own schedule.

    The gap between this and buy & hold is what the rebalance rhythm costs or
    earns; the gap between this and the strategy is what the RANKING adds.
    Both comparisons are meaningless if the two runs hold different numbers
    of names, so the breadth cap is shared.
    """
    base = cfg or BacktestConfig()
    n = max_names or affordable_breadth(base.capital, base.broker)
    ew = BacktestConfig(capital=base.capital, rebalance=base.rebalance,
                        broker=base.broker, slippage=base.slippage,
                        max_positions=n, price_col=base.price_col)
    return run_backtest(panel, equal_weight_signal, ew)


def run_random_trials(panel: pd.DataFrame,
                      n_names: int,
                      cfg: BacktestConfig | None = None,
                      n_trials: int = 200,
                      seed: int = 0,
                      measure_from: pd.Timestamp | None = None) -> pd.DataFrame:
    """Run `n_trials` random-entry backtests at matched exposure.

    Returns one row per trial. The distribution is the point: a strategy's
    Sharpe means something only relative to this spread.

    `measure_from` restricts the SCORING window without restricting the run,
    so the comparison is against the same dates the strategy was measured on.
    Without it the random trials would be scored over all history while the
    strategy is scored over its out-of-sample period, and the two Sharpes
    would not be comparable -- which is exactly the kind of quiet mismatch
    that makes a strategy look better than it is.
    """
    base = cfg or BacktestConfig()
    rows = []
    for i in range(n_trials):
        c = BacktestConfig(capital=base.capital, rebalance=base.rebalance,
                           broker=base.broker, slippage=base.slippage,
                           max_positions=n_names, price_col=base.price_col)
        res = run_backtest(panel, make_random_signal(n_names, seed=seed + i), c)
        if measure_from is not None:
            r = res.returns[res.returns.index >= pd.Timestamp(measure_from)]
            res = replace(res, returns=r)
        p = res.performance(name=f"random_{i}")
        rows.append({"trial": i, "total_return": p.total_return,
                     "cagr": p.cagr, "sharpe": p.sharpe,
                     "max_dd": p.max_drawdown, "vol": p.volatility})
    return pd.DataFrame(rows)


def percentile_vs_random(strategy_value: float,
                         random_values: pd.Series) -> float:
    """Where the strategy sits in the random distribution, 0-100.

    This is the number to report. A Sharpe of 0.8 sounds good; a Sharpe at
    the 55th percentile of random picking does not, and they can be the same
    result.
    """
    if random_values.empty:
        return float("nan")
    return float((random_values < strategy_value).mean() * 100.0)


def baseline_table(panel: pd.DataFrame,
                   strategy_result: BacktestResult,
                   strategy_name: str = "strategy",
                   n_names: int = 3,
                   cfg: BacktestConfig | None = None,
                   n_random: int = 100,
                   n_trials_tested: int = 1,
                   seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The full comparison: strategy against all three opponents.

    Returns (summary_table, random_distribution).
    """
    base = cfg or strategy_result.config

    perfs: list[Performance] = [
        strategy_result.performance(strategy_name, n_trials=n_trials_tested),
        run_buy_and_hold(panel, base).performance("buy & hold"),
        run_equal_weight(panel, base).performance("equal weight, rebalanced"),
    ]

    randoms = run_random_trials(panel, n_names, base, n_trials=n_random, seed=seed)
    median_random = summarise(
        pd.Series(dtype=float), name="random entry (median of "
                                     f"{n_random})")
    # Fill the median row from the trial distribution rather than a rerun.
    median_random.total_return = float(randoms["total_return"].median())
    median_random.cagr = float(randoms["cagr"].median())
    median_random.volatility = float(randoms["vol"].median())
    median_random.sharpe = float(randoms["sharpe"].median())
    median_random.max_drawdown = float(randoms["max_dd"].median())
    perfs.append(median_random)

    table = pd.DataFrame([p.as_row() for p in perfs]).set_index("strategy")
    table.loc[strategy_name, "pctile_vs_random"] = percentile_vs_random(
        perfs[0].sharpe, randoms["sharpe"])
    return table, randoms
