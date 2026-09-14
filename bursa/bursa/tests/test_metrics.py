"""Tests for performance statistics, especially the multiple-testing correction."""
import numpy as np
import pandas as pd
import pytest

from research.metrics import (
    TRADING_DAYS, cagr, calmar, compare, deflated_sharpe, drawdown_series,
    equity_curve, expected_max_sharpe, hit_rate, max_drawdown,
    probabilistic_sharpe, sharpe, sortino, summarise, total_return, volatility,
)


def const_returns(r, n=252):
    return pd.Series([r] * n, index=pd.date_range("2024-01-01", periods=n, freq="B"))


# ------------------------------------------------------------------- basics
def test_total_return_compounds():
    # 1% a day for 3 days = 1.01^3 - 1
    assert total_return(const_returns(0.01, 3)) == pytest.approx(0.030301)


def test_cagr_annualises():
    # exactly 252 bars of 0.1% -> (1.001^252) - 1
    assert cagr(const_returns(0.001, 252)) == pytest.approx(1.001 ** 252 - 1,
                                                            rel=1e-6)


def test_zero_volatility_gives_zero_sharpe_not_infinity():
    assert sharpe(const_returns(0.01)) == 0.0
    assert volatility(const_returns(0.01)) == 0.0


def test_sharpe_scales_with_root_time():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 2520))
    daily = r.mean() / r.std(ddof=1)
    assert sharpe(r) == pytest.approx(daily * np.sqrt(TRADING_DAYS), rel=1e-9)


def test_max_drawdown_is_negative_and_bounded():
    r = pd.Series([0.1, -0.5, 0.2])
    dd = max_drawdown(r)
    assert -1.0 <= dd < 0
    # peak 1.10 -> trough 0.55 = -50%
    assert dd == pytest.approx(-0.5)


def test_no_drawdown_when_monotonic():
    assert max_drawdown(const_returns(0.01, 20)) == pytest.approx(0.0)


def test_equity_curve_matches_total_return():
    r = const_returns(0.01, 10)
    assert equity_curve(r).iloc[-1] == pytest.approx(1 + total_return(r))


def test_hit_rate_counts_positive_periods():
    assert hit_rate(pd.Series([1, -1, 1, -1, 1.0])) == pytest.approx(0.6)


def test_sortino_ignores_upside_volatility():
    # Big upside, small downside -> Sortino should exceed Sharpe.
    r = pd.Series([0.10, -0.01, 0.10, -0.01] * 30)
    assert sortino(r) > sharpe(r)


def test_calmar_is_zero_without_drawdown():
    assert calmar(const_returns(0.01, 20)) == 0.0


def test_empty_and_tiny_series_do_not_explode():
    empty = pd.Series(dtype=float)
    for fn in (total_return, cagr, volatility, sharpe, sortino, max_drawdown,
               calmar, hit_rate):
        assert np.isfinite(fn(empty))


# ------------------------------------------------------- multiple testing
def test_expected_max_sharpe_rises_with_trials():
    """The core intuition: try more worthless things, the best looks better."""
    a = expected_max_sharpe(10)
    b = expected_max_sharpe(100)
    c = expected_max_sharpe(1000)
    assert 0 < a < b < c


def test_single_trial_has_no_selection_bias():
    assert expected_max_sharpe(1) == 0.0


def test_probabilistic_sharpe_is_a_probability():
    p = probabilistic_sharpe(0.1, 0.0, n_obs=252)
    assert 0.0 <= p <= 1.0


def test_probabilistic_sharpe_rises_with_sample_size():
    # Same observed Sharpe, more evidence -> more confidence.
    small = probabilistic_sharpe(0.05, 0.0, n_obs=50)
    large = probabilistic_sharpe(0.05, 0.0, n_obs=2000)
    assert large > small


def test_fat_tails_reduce_confidence():
    thin = probabilistic_sharpe(0.1, 0.0, 252, skew=0.0, kurtosis=3.0)
    fat = probabilistic_sharpe(0.1, 0.0, 252, skew=0.0, kurtosis=12.0)
    assert fat < thin


def test_deflated_sharpe_falls_as_trials_rise():
    """The headline behaviour: the same backtest is worth less if you looked
    at fifty variants before choosing it."""
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0008, 0.01, 756))   # a decent-looking result
    one = deflated_sharpe(r, n_trials=1)
    fifty = deflated_sharpe(r, n_trials=50)
    thousand = deflated_sharpe(r, n_trials=1000)
    assert one > fifty > thousand


def test_deflated_sharpe_rejects_a_lucky_coin_flip():
    """A pure random walk selected as best-of-many must not pass."""
    rng = np.random.default_rng(3)
    # Best of 200 random strategies, by construction worthless.
    best, best_sr = None, -np.inf
    for i in range(200):
        cand = pd.Series(rng.normal(0.0, 0.01, 252))
        if sharpe(cand) > best_sr:
            best, best_sr = cand, sharpe(cand)
    assert best_sr > 0.5                      # looks good naively
    assert deflated_sharpe(best, n_trials=200) < 0.95   # but does not survive


def test_summarise_only_reports_dsr_when_trials_exceed_one():
    r = const_returns(0.001, 100)
    assert np.isnan(summarise(r, n_trials=1).deflated_sharpe)
    assert not np.isnan(summarise(r, n_trials=10).deflated_sharpe)


def test_compare_builds_a_table():
    rng = np.random.default_rng(1)
    perfs = [summarise(pd.Series(rng.normal(0, 0.01, 100)), name=f"s{i}")
             for i in range(3)]
    tbl = compare(perfs)
    assert list(tbl.index) == ["s0", "s1", "s2"]
    assert "sharpe" in tbl.columns
