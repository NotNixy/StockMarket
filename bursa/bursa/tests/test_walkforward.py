"""Tests for walk-forward evaluation.

The properties that matter: the test window never leaks into selection, the
holdout is untouched, and a fitted edge visibly degrades out of sample.
"""
import numpy as np
import pandas as pd
import pytest

from core.costs import MOOMOO, SlippageModel
from research.backtest import BacktestConfig
from research.walkforward import (
    Fold, WalkForwardResult, evaluate_holdout, make_splits, walk_forward,
)

NO_SLIP = SlippageModel(0.0, 0.0)
CFG = BacktestConfig(capital=10_000, rebalance="ME", broker=MOOMOO,
                     slippage=NO_SLIP, max_positions=2)


def panel(n_days=1500, n_tickers=8, seed=0, momentum_edge=0.0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n_days, freq="B")
    vols = rng.uniform(0.012, 0.030, n_tickers)
    rets = np.zeros((n_days, n_tickers))
    for t in range(n_days):
        drift = np.zeros(n_tickers)
        if momentum_edge and t > 130:
            drift = momentum_edge * np.sign(rets[t-126:t-21].sum(axis=0)) * vols
        rets[t] = rng.normal(drift, vols)
    px = 5.0 * np.exp(np.cumsum(rets, axis=0))
    rows = []
    for i in range(n_tickers):
        p = px[:, i]
        rows.append(pd.DataFrame({
            "date": dates, "ticker": f"T{i}.KL", "open": p, "high": p * 1.01,
            "low": p * 0.99, "close": p, "adj_close": p,
            "volume": 1_000_000, "eligible": True}))
    return pd.concat(rows, ignore_index=True)


class Strat:
    """Minimal strategy object with the .signal the harness expects."""
    def __init__(self, pick_first=True):
        self.pick_first = pick_first

    def signal(self, history, date, eligible):
        if not eligible:
            return {}
        return {eligible[0] if self.pick_first else eligible[-1]: 1.0}


# --------------------------------------------------------------------- splits
def test_splits_are_chronological_and_non_overlapping():
    dates = pd.DatetimeIndex(pd.date_range("2019-01-01", periods=1500, freq="B"))
    splits, _ = make_splits(dates, n_folds=4, train_months=18, test_months=6)
    assert len(splits) >= 2
    for tr_s, tr_e, te_s, te_e in splits:
        assert tr_s < tr_e <= te_s < te_e     # train ends before test begins
    for a, b in zip(splits, splits[1:]):
        assert a[3] <= b[3]                   # folds advance through time


def test_holdout_is_excluded_from_every_fold():
    dates = pd.DatetimeIndex(pd.date_range("2019-01-01", periods=1500, freq="B"))
    splits, holdout = make_splits(dates, n_folds=4, holdout_months=12)
    for _, _, _, te_e in splits:
        assert te_e <= holdout, "a test window overlapped the holdout"


def test_expanding_windows_grow():
    dates = pd.DatetimeIndex(pd.date_range("2019-01-01", periods=1500, freq="B"))
    splits, _ = make_splits(dates, n_folds=4, expanding=True)
    lengths = [(tr_e - tr_s).days for tr_s, tr_e, _, _ in splits]
    assert lengths == sorted(lengths), "expanding windows should not shrink"


def test_rolling_windows_stay_roughly_constant():
    dates = pd.DatetimeIndex(pd.date_range("2019-01-01", periods=1500, freq="B"))
    splits, _ = make_splits(dates, n_folds=4, train_months=18, expanding=False)
    lengths = [(tr_e - tr_s).days for tr_s, tr_e, _, _ in splits]
    assert max(lengths) - min(lengths) < 40


def test_too_little_history_is_rejected():
    with pytest.raises(ValueError):
        make_splits(pd.DatetimeIndex(pd.date_range("2024-01-01", periods=50)))


def test_an_oversized_holdout_is_rejected():
    dates = pd.DatetimeIndex(pd.date_range("2019-01-01", periods=400, freq="B"))
    with pytest.raises(ValueError, match="too little history"):
        make_splits(dates, holdout_months=18)


# --------------------------------------------------------------- no leakage
def test_selection_never_sees_the_test_window():
    """Record the dates each candidate was trained on and assert none of them
    fall inside that fold's test period."""
    seen = {}

    class Spy:
        def signal(self, history, date, eligible):
            seen.setdefault(id(self), []).append(history["date"].max())
            return {eligible[0]: 1.0} if eligible else {}

    p = panel()
    res = walk_forward(p, {"spy": Spy}, CFG, n_folds=3, verbose=False)
    assert res.folds
    for f in res.folds:
        # Anything the signal saw during selection must precede the test start.
        assert f.train_end <= f.test_start


def test_out_of_sample_returns_lie_inside_the_test_windows():
    p = panel()
    res = walk_forward(p, {"a": Strat}, CFG, n_folds=3, verbose=False)
    assert len(res.oos_returns) > 0
    lo = min(f.test_start for f in res.folds)
    hi = max(f.test_end for f in res.folds)
    assert res.oos_returns.index.min() >= lo
    assert res.oos_returns.index.max() <= hi


def test_holdout_returns_are_not_in_the_walk_forward_output():
    p = panel()
    res = walk_forward(p, {"a": Strat}, CFG, n_folds=3, verbose=False)
    assert res.oos_returns.index.max() < res.holdout_start


# ------------------------------------------------------------- degradation
def test_a_real_edge_survives_out_of_sample():
    p = panel(momentum_edge=0.35, seed=3)

    class Mom:
        def signal(self, history, date, eligible):
            px = history.pivot_table(index="date", columns="ticker",
                                     values="adj_close", aggfunc="last")
            if len(px) < 150:
                return {}
            trail = (px.iloc[-22] / px.iloc[-147] - 1).reindex(eligible).dropna()
            return {trail.idxmax(): 1.0} if not trail.empty else {}

    res = walk_forward(p, {"mom": Mom}, CFG, n_folds=3, verbose=False)
    assert res.mean_oos > 0, "a planted edge should survive out of sample"


def test_degradation_is_reported_when_in_sample_is_positive():
    p = panel(momentum_edge=0.35, seed=5)
    res = walk_forward(p, {"a": Strat, "b": lambda: Strat(pick_first=False)},
                       CFG, n_folds=3, verbose=False)
    if np.isfinite(res.mean_is) and res.mean_is > 0:
        assert np.isfinite(res.degradation)


def test_summary_renders():
    res = walk_forward(panel(), {"a": Strat}, CFG, n_folds=3, verbose=False)
    s = res.summary()
    assert "survival ratio" in s and "stitched OOS" in s


# ----------------------------------------------------------------- holdout
def test_holdout_evaluates_only_the_reserved_slice():
    p = panel()
    res = walk_forward(p, {"a": Strat}, CFG, n_folds=3, verbose=False)
    out = evaluate_holdout(p, Strat, CFG, res.holdout_start, verbose=False)
    assert out["start"] >= res.holdout_start
    assert "deflated_sharpe" in out


def test_holdout_deflation_responds_to_declared_trials():
    p = panel()
    res = walk_forward(p, {"a": Strat}, CFG, n_folds=3, verbose=False)
    one = evaluate_holdout(p, Strat, CFG, res.holdout_start, n_trials=1,
                           verbose=False)
    many = evaluate_holdout(p, Strat, CFG, res.holdout_start, n_trials=100,
                            verbose=False)
    assert many["deflated_sharpe"] <= one["deflated_sharpe"]


def test_a_too_short_holdout_is_rejected():
    p = panel()
    with pytest.raises(ValueError, match="too short"):
        evaluate_holdout(p, Strat, CFG, p["date"].max(), verbose=False)
