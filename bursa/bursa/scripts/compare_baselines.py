"""Put the strategy next to doing nothing, over the same window.

    python -m scripts.compare_baselines
    python -m scripts.compare_baselines --start 2023-03-12 --trials 200

A strategy's own equity curve is not evidence. The question is never "did it
make money" -- over a rising market almost anything does -- but "did it make
more than the cheapest alternative, by enough to survive the number of things
I tried before finding it."

Four comparisons, in increasing order of how uncomfortable they are:

1. **Buy and hold.** Buy the eligible names on day one and never trade. Zero
   effort, near-zero cost.
2. **Equal weight, rebalanced.** Same names, same rebalance schedule as the
   strategy, no ranking at all. This isolates whether the RANKING adds
   anything, separately from the universe and the rebalance rhythm.
3. **Random entry at matched exposure.** Hold the same number of names for
   the same fraction of the time, chosen at random, many times. If the
   strategy sits at the 55th percentile of that distribution, it is a coin.
4. **Deflated Sharpe.** Discounts the result by how many strategies were
   tried to find it. Three presets across five folds is not one trial.

The window defaults to the stitched out-of-sample period from the
walk-forward, so the comparison is like-for-like: the same dates the strategy
was measured on, without hindsight.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from research.backtest import BacktestConfig, run_backtest
from research.baselines import (
    percentile_vs_random, run_buy_and_hold, run_equal_weight, run_random_trials,
)
from research.metrics import (
    deflated_sharpe, max_drawdown, sharpe, total_return,
)
from research.strategies import PRESETS


def _measure(returns: pd.Series, start: pd.Timestamp) -> dict:
    r = returns[returns.index >= start]
    return {"sharpe": sharpe(r), "return": total_return(r),
            "maxdd": max_drawdown(r), "n": len(r)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=Path("data/screened.parquet"))
    ap.add_argument("--strategy", default="momentum", choices=sorted(PRESETS))
    ap.add_argument("--start", default="2023-03-12",
                    help="start of the comparison window (default: the "
                         "stitched OOS period from the walk-forward)")
    ap.add_argument("--end", default=None,
                    help="end of the window; defaults to the holdout start, "
                         "so the untouched holdout stays untouched")
    ap.add_argument("--rebalance", default="ME")
    ap.add_argument("--max-positions", type=int, default=5)
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--trials", type=int, default=200,
                    help="random-entry trials")
    ap.add_argument("--searched", type=int, default=15,
                    help="strategies effectively tried, for the deflation. "
                         "3 presets x 5 folds = 15 is the honest floor.")
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    if "eligible" in panel.columns:
        panel = panel[panel["eligible"]].copy()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp("2025-09-14")
    panel = panel[panel["date"] < end]
    print(f"universe: {panel['ticker'].nunique()} names | "
          f"window {start.date()} -> {end.date()}\n")

    cfg = BacktestConfig(capital=args.capital, rebalance=args.rebalance,
                         max_positions=args.max_positions)

    # Every run covers all history so the signals have their lookback, and is
    # then MEASURED only over the comparison window. Same rule as the folds.
    strat = run_backtest(panel, PRESETS[args.strategy]().signal, cfg)
    bh = run_buy_and_hold(panel, cfg)
    ew = run_equal_weight(panel, cfg)

    rows = {
        f"{args.strategy} (strategy)": _measure(strat.returns, start),
        "buy & hold": _measure(bh.returns, start),
        "equal weight, rebalanced": _measure(ew.returns, start),
    }
    table = pd.DataFrame(rows).T
    table["return"] = table["return"].map(lambda v: f"{v:+.2%}")
    table["maxdd"] = table["maxdd"].map(lambda v: f"{v:.2%}")
    table["sharpe"] = table["sharpe"].map(lambda v: f"{v:+.3f}")
    print(table[["sharpe", "return", "maxdd"]].to_string())

    s_sharpe = sharpe(strat.returns[strat.returns.index >= start])

    print(f"\nrandom entry at matched exposure ({args.trials} trials)...")
    rnd = run_random_trials(panel, n_names=args.max_positions, cfg=cfg,
                            n_trials=args.trials, measure_from=start)
    rnd_sharpes = rnd["sharpe"].dropna()
    pct = percentile_vs_random(s_sharpe, rnd_sharpes)
    print(f"  random Sharpe: median {rnd_sharpes.median():+.3f}, "
          f"5th {rnd_sharpes.quantile(0.05):+.3f}, "
          f"95th {rnd_sharpes.quantile(0.95):+.3f}")
    print(f"  strategy {s_sharpe:+.3f} sits at the {pct:.0f}th percentile "
          f"of random")

    dsr = deflated_sharpe(strat.returns[strat.returns.index >= start],
                          n_trials=args.searched)
    print(f"\ndeflated Sharpe (declaring {args.searched} trials): {dsr:.3f}")
    print("  = P(the true Sharpe is above zero), after discounting the search.")
    print("  Below ~0.95 is not a finding. Below ~0.5 is noise.")

    verdict = ("BEATS" if s_sharpe > float(sharpe(bh.returns[bh.returns.index >= start]))
               else "DOES NOT BEAT")
    print(f"\n  Against buy & hold on the same dates: {verdict} it.")
    print("  Survivorship and backfilled compliance both flatter these "
          "numbers.\n  See scripts.run_walkforward for the full list.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
