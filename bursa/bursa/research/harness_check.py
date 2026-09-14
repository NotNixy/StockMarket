"""Validate the harness before trusting anything it says.

A backtest engine that reports every strategy as profitable is worse than
useless, and so is one that reports every strategy as worthless. So this
script runs two controls:

  NEGATIVE -- synthetic random-walk prices with no edge by construction.
              The harness must NOT find one. If it does, it leaks.

  POSITIVE -- the same, but with a real, planted edge: high-momentum names
              genuinely continue. The harness must find it. Otherwise the
              negative result above proves nothing, because an engine that
              earns nothing on anything would also pass it.

  DEFLATION -- pick the best of 100 worthless strategies and confirm the
              deflated Sharpe refuses to be impressed.

Run:  python -m research.harness_check
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.costs import MOOMOO, SlippageModel
from research.backtest import BacktestConfig, run_backtest
from research.baselines import (
    make_random_signal, percentile_vs_random, run_buy_and_hold,
    run_equal_weight, run_random_trials,
)
from research.metrics import deflated_sharpe, sharpe, summarise
from research.strategies import CompositeStrategy

RULE = "=" * 74


def synth_panel(n_tickers=40, n_days=750, seed=0,
                momentum_edge=0.0) -> pd.DataFrame:
    """Synthetic Bursa-like panel.

    momentum_edge = 0 gives pure random walks: no strategy can work.
    momentum_edge > 0 makes past winners genuinely continue, so a momentum
    strategy SHOULD work -- the positive control.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n_days, freq="B")
    vols = rng.uniform(0.010, 0.035, n_tickers)     # cross-sectional vol spread

    rets = np.zeros((n_days, n_tickers))
    for t in range(n_days):
        drift = np.zeros(n_tickers)
        if momentum_edge > 0 and t > 130:
            past = rets[t - 126:t - 21].sum(axis=0)      # trailing 6m, skip 1m
            drift = momentum_edge * np.sign(past) * vols
        rets[t] = rng.normal(drift, vols)

    prices = 5.0 * np.exp(np.cumsum(rets, axis=0))
    rows = []
    for i in range(n_tickers):
        px = prices[:, i]
        rows.append(pd.DataFrame({
            "date": dates, "ticker": f"T{i:02d}.KL",
            "open": px * 0.999, "high": px * 1.012, "low": px * 0.988,
            "close": px, "adj_close": px,
            "volume": rng.integers(400_000, 3_000_000, n_days),
            "eligible": True,
        }))
    return pd.concat(rows, ignore_index=True)


def run_control(label: str, panel: pd.DataFrame, cfg: BacktestConfig,
                n_random: int = 60, seed: int = 0):
    """Run one strategy against every baseline, and report the percentile."""
    strat = CompositeStrategy(
        weights={"momentum": 70, "volume_surge": 15, "low_vol": 15},
        top_n=cfg.max_positions, name="momentum composite")

    res = run_backtest(panel, strat.signal, cfg)
    perf = res.performance(strat.name)

    bh = run_buy_and_hold(panel, cfg).performance("buy & hold")
    ew = run_equal_weight(panel, cfg).performance("equal weight, rebalanced")
    randoms = run_random_trials(panel, cfg.max_positions, cfg,
                                n_trials=n_random, seed=seed)

    pct = percentile_vs_random(perf.sharpe, randoms["sharpe"])

    print(f"\n{RULE}\n{label}\n{RULE}")
    for p in (perf, bh, ew):
        print(f"  {p}")
    print(f"  {'random entry, ' + str(n_random) + ' trials':<28} "
          f"median SR {randoms['sharpe'].median():+5.2f}  "
          f"5-95pct [{randoms['sharpe'].quantile(.05):+5.2f}, "
          f"{randoms['sharpe'].quantile(.95):+5.2f}]")
    print(f"\n  cost drag {res.cost_drag:.2%} of capital over "
          f"{res.n_rebalances} rebalances, avg turnover {res.avg_turnover:.2f}")
    print(f"  >> strategy sits at the {pct:.0f}th percentile of random picking")
    return perf, pct, randoms


def main() -> int:
    cfg = BacktestConfig(capital=10_000, rebalance="ME", broker=MOOMOO,
                         slippage=SlippageModel(15.0, 0.0), max_positions=3)
    failures = []

    # ---------------------------------------------------------- NEGATIVE
    neg_panel = synth_panel(seed=1, momentum_edge=0.0)
    neg_perf, neg_pct, neg_rand = run_control(
        "NEGATIVE CONTROL -- random walks, no edge exists", neg_panel, cfg,
        seed=100)
    print("\n  expectation: percentile near 50, no systematic outperformance.")
    if not (10 <= neg_pct <= 90):
        failures.append(
            f"negative control landed at the {neg_pct:.0f}th percentile "
            "-- the harness may be leaking information")

    # ---------------------------------------------------------- POSITIVE
    pos_panel = synth_panel(seed=1, momentum_edge=0.30)
    pos_perf, pos_pct, _ = run_control(
        "POSITIVE CONTROL -- momentum edge planted in the data",
        pos_panel, cfg, seed=200)
    print("\n  expectation: high percentile. If this fails, the negative "
          "result above\n  proves nothing -- an engine that earns nothing "
          "would also pass it.")
    if pos_pct < 70:
        failures.append(
            f"positive control only reached the {pos_pct:.0f}th percentile "
            "-- the harness cannot detect a real edge")

    # --------------------------------------------------------- DEFLATION
    print(f"\n{RULE}\nDEFLATION -- best of 100 worthless strategies\n{RULE}")
    best_sr, best_ret = -np.inf, None
    for i in range(100):
        r = run_backtest(neg_panel, make_random_signal(3, seed=500 + i),
                         cfg).returns
        if sharpe(r) > best_sr:
            best_sr, best_ret = sharpe(r), r

    naive = best_sr
    dsr_1 = deflated_sharpe(best_ret, n_trials=1)
    dsr_100 = deflated_sharpe(best_ret, n_trials=100)
    print(f"  best Sharpe found across 100 random strategies : {naive:+.2f}")
    print(f"  P(true SR > 0) if you pretend it was 1 trial   : {dsr_1:.3f}")
    print(f"  P(true SR > 0) accounting for all 100 trials   : {dsr_100:.3f}")
    print("\n  expectation: the naive figure looks respectable, the deflated "
          "one does not.")
    if dsr_100 >= dsr_1:
        failures.append("deflation did not reduce confidence with more trials")

    # ------------------------------------------------------------ verdict
    print(f"\n{RULE}")
    if failures:
        print("HARNESS CHECK FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("HARNESS CHECK PASSED")
    print("  The engine finds no edge where none exists, finds one where it")
    print("  does, and refuses to be impressed by the best of many trials.")
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
