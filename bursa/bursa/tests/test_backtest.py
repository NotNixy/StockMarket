"""Tests for the harness.

The lookahead tests are the reason this file exists. A harness that fills on
the signal bar produces beautiful, worthless backtests, and nothing about the
output looks wrong.
"""
import numpy as np
import pandas as pd
import pytest

from core.costs import BANK_STD, MOOMOO, SlippageModel
from research.backtest import (
    BacktestConfig, _normalise_weights, _rebalance_dates, run_backtest,
)

NO_SLIP = SlippageModel(0.0, 0.0)


def make_panel(n=300, tickers=("AAA.KL", "BBB.KL", "CCC.KL"), seed=0,
               drift=None):
    """Random-walk panel unless `drift` gives per-ticker daily drift."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    rows = []
    for i, t in enumerate(tickers):
        mu = 0.0 if drift is None else drift[i]
        px = 5.0 * np.exp(np.cumsum(rng.normal(mu, 0.015, n)))
        rows.append(pd.DataFrame({
            "date": dates, "ticker": t,
            "open": px * 0.999, "high": px * 1.01, "low": px * 0.99,
            "close": px, "adj_close": px, "volume": 1_000_000,
            "eligible": True,
        }))
    return pd.concat(rows, ignore_index=True)


def equal_all(history, date, eligible):
    return {t: 1.0 for t in eligible}


# ------------------------------------------------------------------ plumbing
def test_normalise_weights_keeps_top_n_and_sums_to_one():
    w = _normalise_weights({"a": 3, "b": 2, "c": 1}, max_positions=2)
    assert set(w) == {"a", "b"}
    assert sum(w.values()) == pytest.approx(1.0)


def test_normalise_weights_drops_non_positive_and_nan():
    w = _normalise_weights({"a": 1, "b": 0, "c": -1, "d": np.nan}, 10)
    assert set(w) == {"a"}


def test_rebalance_dates_use_the_panels_own_calendar():
    dates = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=90, freq="B"))
    monthly = _rebalance_dates(dates, "ME")
    assert all(d in dates for d in monthly)          # never invents a holiday


def test_missing_columns_are_rejected():
    with pytest.raises(ValueError, match="missing column"):
        run_backtest(pd.DataFrame({"date": [], "ticker": []}), equal_all)


# -------------------------------------------------------------- NO LOOKAHEAD
def test_signal_never_sees_future_bars():
    """The signal must receive history up to the decision date and no further."""
    seen = []

    def spy(history, date, eligible):
        seen.append((date, history["date"].max()))
        return {t: 1.0 for t in eligible}

    run_backtest(make_panel(), spy, BacktestConfig(rebalance="ME"))
    assert seen, "signal was never called"
    for decision_date, max_seen in seen:
        assert max_seen <= decision_date


def test_fills_happen_on_the_bar_after_the_signal():
    """A perfect-foresight signal must NOT be able to capture same-bar moves."""
    panel = make_panel(n=200, tickers=("AAA.KL", "BBB.KL"), seed=5)
    px = panel.pivot_table(index="date", columns="ticker", values="adj_close")
    fwd = px.pct_change().shift(-1)          # tomorrow's return, known to nobody

    def cheat_today(history, date, eligible):
        # Picks the stock that rose MOST TODAY -- information available at the
        # close, but useless unless you can trade at today's open.
        today = px.loc[date]
        yday = px.shift(1).loc[date]
        move = (today / yday).dropna()
        if move.empty:          # first bar: no previous close to compare to
            return {}
        return {move.idxmax(): 1.0}

    res = run_backtest(panel, cheat_today,
                       BacktestConfig(rebalance="W-FRI", slippage=NO_SLIP,
                                      max_positions=1))
    # If the harness leaked, this would show a large positive return.
    # It is allowed to be positive by luck, but not implausibly so.
    assert abs(res.returns.sum()) < 1.0


def test_a_genuine_oracle_does_make_money():
    """Sanity check on the opposite side: if a signal legitimately knows
    tomorrow, the harness should reward it. Otherwise the lookahead test above
    would pass for the wrong reason (a broken harness that earns nothing)."""
    panel = make_panel(n=300, tickers=("AAA.KL", "BBB.KL"), seed=9)
    px = panel.pivot_table(index="date", columns="ticker", values="adj_close")
    fwd_month = px.pct_change(21).shift(-21)

    def oracle(history, date, eligible):
        if date not in fwd_month.index:
            return {}
        row = fwd_month.loc[date].dropna()
        return {row.idxmax(): 1.0} if not row.empty else {}

    res = run_backtest(panel, oracle,
                       BacktestConfig(rebalance="ME", slippage=NO_SLIP,
                                      max_positions=1))
    assert res.returns.sum() > 0


# ------------------------------------------------------------------- costs
def test_costs_are_charged_and_reduce_returns():
    panel = make_panel()
    free = run_backtest(panel, equal_all,
                        BacktestConfig(broker=MOOMOO, slippage=NO_SLIP))
    dear = run_backtest(panel, equal_all,
                        BacktestConfig(broker=BANK_STD,
                                       slippage=SlippageModel(50.0, 0.0)))
    assert dear.cost_drag > free.cost_drag
    assert dear.returns.sum() < free.returns.sum()


def test_zero_turnover_after_the_first_rebalance_costs_nothing_extra():
    """Holding a fixed set should only pay on the initial build."""
    panel = make_panel(tickers=("AAA.KL",))
    res = run_backtest(panel, equal_all,
                       BacktestConfig(rebalance="ME", slippage=NO_SLIP))
    # First rebalance builds the position; the rest are no-ops.
    assert res.trades["date"].nunique() == 1


def test_higher_turnover_costs_more():
    panel = make_panel(n=400, seed=2)
    rng = np.random.default_rng(0)

    def churn(history, date, eligible):
        return {str(rng.choice(np.asarray(eligible, dtype=object))): 1.0}

    calm = run_backtest(panel, equal_all, BacktestConfig(rebalance="ME"))
    busy = run_backtest(panel, churn,
                        BacktestConfig(rebalance="W-FRI", max_positions=1))
    assert busy.avg_turnover > calm.avg_turnover
    assert busy.cost_drag > calm.cost_drag


# ------------------------------------------------------------------ outputs
def test_result_shapes_are_consistent():
    panel = make_panel()
    res = run_backtest(panel, equal_all, BacktestConfig(rebalance="ME"))
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    assert len(res.returns) == len(dates)
    assert len(res.equity) == len(dates)
    assert res.n_rebalances > 0
    assert res.weights.shape[1] == panel["ticker"].nunique()


def test_weights_never_exceed_max_positions():
    panel = make_panel(tickers=tuple(f"T{i}.KL" for i in range(10)))
    res = run_backtest(panel, equal_all, BacktestConfig(max_positions=3))
    held = (res.weights > 0).sum(axis=1)
    assert held.max() <= 3


def test_ineligible_tickers_are_never_held():
    panel = make_panel()
    panel.loc[panel["ticker"] == "BBB.KL", "eligible"] = False
    res = run_backtest(panel, equal_all, BacktestConfig())
    assert res.weights["BBB.KL"].abs().sum() == 0.0


def test_an_empty_universe_produces_a_flat_curve():
    panel = make_panel()
    panel["eligible"] = False
    res = run_backtest(panel, equal_all, BacktestConfig())
    assert res.returns.abs().sum() == pytest.approx(0.0)


def test_equity_curve_matches_the_return_series():
    panel = make_panel()
    res = run_backtest(panel, equal_all, BacktestConfig())
    expected = res.config.capital * (1 + res.returns).cumprod()
    assert np.allclose(res.equity.values, expected.values)


def test_position_is_established_on_the_first_bar():
    """A period-end rebalance rule must not leave the portfolio in cash until
    the first period closes. With an annual rule that meant sitting out almost
    a whole year, which made buy-and-hold look far worse than it is."""
    panel = make_panel(n=400, tickers=("AAA.KL",))
    res = run_backtest(panel, equal_all,
                       BacktestConfig(rebalance="YE", slippage=NO_SLIP))
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    # Invested within the first few bars, not months later.
    held_early = (res.weights.loc[dates[:5]] > 0).any(axis=1)
    assert held_early.any()


def test_first_date_is_always_a_rebalance_point():
    dates = pd.DatetimeIndex(pd.date_range("2024-01-10", periods=400, freq="B"))
    for rule in ("ME", "YE", "W-FRI", "QE"):
        assert _rebalance_dates(dates, rule)[0] == dates[0]
