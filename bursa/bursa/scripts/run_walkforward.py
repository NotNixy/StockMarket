"""Walk-forward the strategy presets over the screened Shariah universe.

    python -m scripts.run_walkforward
    python -m scripts.run_walkforward --rebalance W-FRI --max-positions 3

Reads `data/screened.parquet` (written by scripts.build_panel + the screen)
and reports, per fold: which strategy the training window chose, how it did
in-sample, and how the same choice did on data it had not seen.

## What the numbers mean, and what they do not

The headline is the **survival ratio**: out-of-sample Sharpe divided by
in-sample Sharpe. Above ~0.5 means some of the edge is real. Near zero means
the in-sample number was search, not skill.

Three declared biases sit on top of every figure here:

* **Survivorship.** The SC list is a November 2025 snapshot, so the universe
  is companies that existed and were compliant in November 2025. Anything
  that delisted in 2019 is invisible. This flatters every result, and by an
  unknown amount.
* **Backfilled compliance.** One SC release covers ten months; the backtest
  runs ten years. `backfill_shariah=True` applies the 2025 list backwards,
  which is lookahead. Shrinks with every extra release parsed.
* **Price return, not total return.** 227 of 865 tickers have adj_close ==
  close throughout and the data cannot say whether that is "never paid a
  dividend" or "never adjusted one". Dividends are therefore inconsistently
  represented across names.

None of these are fixed by running this script again. They belong in any
sentence that quotes a number it produced.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from research.backtest import BacktestConfig
from research.strategies import PRESETS
from research.walkforward import walk_forward


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=Path("data/screened.parquet"))
    ap.add_argument("--rebalance", default="ME")
    ap.add_argument("--max-positions", type=int, default=5)
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-months", type=int, default=24)
    ap.add_argument("--test-months", type=int, default=6)
    ap.add_argument("--holdout-months", type=int, default=12)
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    if "eligible" in panel.columns:
        keep = panel[panel["eligible"]].copy()
        print(f"screened panel: {len(panel):,} bars -> {len(keep):,} eligible "
              f"({keep['ticker'].nunique()} names)")
        panel = keep

    cfg = BacktestConfig(capital=args.capital, rebalance=args.rebalance,
                         max_positions=args.max_positions)
    print(f"config: {args.rebalance} rebalance, {args.max_positions} positions, "
          f"RM {args.capital:,.0f}\n")

    walk_forward(panel, PRESETS, cfg,
                 n_folds=args.folds, train_months=args.train_months,
                 test_months=args.test_months,
                 holdout_months=args.holdout_months, verbose=True)

    print("\n  Declared biases: survivorship (Nov-2025 SC snapshot), "
          "backfilled\n  compliance, and price-not-total return. See the "
          "module docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
