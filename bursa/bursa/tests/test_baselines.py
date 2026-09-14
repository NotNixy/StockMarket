"""Tests for the baseline opponents."""
import numpy as np
import pandas as pd
import pytest

from core.costs import SlippageModel
from research.backtest import BacktestConfig, run_backtest
from research.baselines import (
    baseline_table, buy_and_hold_signal, make_random_signal,
    percentile_vs_random, run_buy_and_hold, run_equal_weight, run_random_trials,
)

NO_SLIP = SlippageModel(0.0, 0.0)


def make_panel(n=400, k=8, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    rows = []
    for i in range(k):
        px = 5.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.018, n)))
        rows.append(pd.DataFrame({
            "date": dates, "ticker": f"T{i}.KL",
            "open": px, "high": px * 1.01, "low": px * 0.99,
            "close": px, "adj_close": px, "volume": 1_000_000,
            "eligible": True,
        }))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------- signals
def test_buy_and_hold_holds_everything_eligible():
    w = buy_and_hold_signal(None, None, ["A", "B", "C"])
    assert set(w) == {"A", "B", "C"}


def test_random_signal_picks_the_requested_count():
    sig = make_random_signal(3, seed=0)
    w = sig(None, None, [f"T{i}" for i in range(10)])
    assert len(w) == 3


def test_random_signal_never_exceeds_the_universe():
    sig = make_random_signal(10, seed=0)
    assert len(sig(None, None, ["A", "B"])) == 2


def test_random_signal_handles_an_empty_universe():
    assert make_random_signal(3)(None, None, []) == {}


def test_random_signal_varies_with_seed():
    universe = [f"T{i}" for i in range(20)]
    a = set(make_random_signal(3, seed=1)(None, None, universe))
    b = set(make_random_signal(3, seed=2)(None, None, universe))
    assert a != b


def test_random_signal_is_reproducible():
    universe = [f"T{i}" for i in range(20)]
    a = set(make_random_signal(3, seed=7)(None, None, universe))
    b = set(make_random_signal(3, seed=7)(None, None, universe))
    assert a == b


# ----------------------------------------------------------------- baselines
def test_buy_and_hold_trades_less_than_monthly_rebalancing():
    panel = make_panel()
    cfg = BacktestConfig(rebalance="ME", slippage=NO_SLIP)
    bh = run_buy_and_hold(panel, cfg)
    ew = run_equal_weight(panel, cfg)
    assert bh.n_rebalances < ew.n_rebalances
    assert bh.cost_drag <= ew.cost_drag


def test_equal_weight_holds_the_whole_universe():
    panel = make_panel(k=6)
    res = run_equal_weight(panel, BacktestConfig(rebalance="ME"))
    held = (res.weights > 0).sum(axis=1).max()
    assert held == 6


def test_random_trials_produce_a_distribution():
    panel = make_panel(n=300, k=8)
    cfg = BacktestConfig(rebalance="ME", max_positions=3, slippage=NO_SLIP)
    trials = run_random_trials(panel, 3, cfg, n_trials=15, seed=0)
    assert len(trials) == 15
    assert trials["sharpe"].std() > 0          # genuinely different outcomes
    assert set(trials.columns) >= {"total_return", "cagr", "sharpe", "max_dd"}


def test_random_trials_are_reproducible():
    panel = make_panel(n=250, k=6)
    cfg = BacktestConfig(rebalance="ME", max_positions=2, slippage=NO_SLIP)
    a = run_random_trials(panel, 2, cfg, n_trials=5, seed=42)
    b = run_random_trials(panel, 2, cfg, n_trials=5, seed=42)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------- percentile
def test_percentile_vs_random_locates_the_strategy():
    dist = pd.Series(np.arange(100, dtype=float))
    assert percentile_vs_random(50.0, dist) == pytest.approx(50.0)
    assert percentile_vs_random(-10.0, dist) == 0.0
    assert percentile_vs_random(1000.0, dist) == 100.0


def test_percentile_vs_random_handles_empty_distribution():
    assert np.isnan(percentile_vs_random(1.0, pd.Series(dtype=float)))


# --------------------------------------------------------------------- table
def test_baseline_table_includes_every_opponent():
    panel = make_panel(n=300, k=6)
    cfg = BacktestConfig(rebalance="ME", max_positions=3, slippage=NO_SLIP)
    strat = run_backtest(panel, make_random_signal(3, seed=0), cfg)
    table, randoms = baseline_table(panel, strat, "test strategy", n_names=3,
                                    cfg=cfg, n_random=10, seed=0)
    assert "test strategy" in table.index
    assert "buy & hold" in table.index
    assert "equal weight, rebalanced" in table.index
    assert any("random entry" in i for i in table.index)
    assert len(randoms) == 10
    assert not np.isnan(table.loc["test strategy", "pctile_vs_random"])
