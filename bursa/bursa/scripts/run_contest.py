"""What actually maximises the chance of finishing first.

    python -m scripts.run_contest
    python -m scripts.run_contest --entrants 400 --horizon 21

Runs the contest simulator over the real screened Shariah panel and answers
three questions in order:

1. **How many names should you hold?** The concentration sweep.
2. **Does tilting the selection help?** The rule sweep -- high volatility,
   momentum, reversal, against random picking.
3. **What does it cost you?** The downside column, which is the part that
   gets left out of "go greedy".

Read the p_win column against 1/n_entrants. That is what you get for showing
up and holding a sensible portfolio. Anything a strategy adds is the
difference between its p_win and that baseline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from tournament.contest import (
    ContestConfig, concentration_sweep, run_contest, rule_sweep,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=Path("data/screened.parquet"))
    ap.add_argument("--entrants", type=int, default=400)
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--field-k", type=int, default=10)
    ap.add_argument("--trials", type=int, default=4000)
    ap.add_argument("--best-k", type=int, default=0,
                    help="concentration to use for the rule sweep; "
                         "0 = pick the best from the concentration sweep")
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    n_names = panel.loc[panel.get("eligible", True), "ticker"].nunique()
    base = ContestConfig(n_entrants=args.entrants, horizon=args.horizon,
                         field_k=args.field_k, n_trials=args.trials)

    fair = 1.0 / args.entrants
    print(f"universe {n_names} names | {args.entrants} entrants | "
          f"{args.horizon} trading days")
    print(f"fair share (hold anything sensible): {fair:.2%}\n")

    print("=" * 72)
    print("1. CONCENTRATION -- how many names to hold")
    print("=" * 72)
    conc = concentration_sweep(panel, base=base)
    best_k = int(conc["p_win"].idxmax())
    print(f"\n  best k = {best_k}  at P(win) {conc['p_win'].max():.2%}  "
          f"({conc['p_win'].max() / fair:.1f}x fair share)")

    use_k = args.best_k or best_k
    print("\n" + "=" * 72)
    print(f"2. SELECTION RULE -- at k = {use_k}")
    print("=" * 72)
    rules = rule_sweep(panel, base=ContestConfig(**{**base.__dict__,
                                                   "my_k": use_k}))

    print("\n" + "=" * 72)
    print("3. WHAT IT COSTS")
    print("=" * 72)
    print(conc[["p_win", "median", "mean", "p95", "p_lose_half"]]
          .rename(columns={"p_lose_half": "P(lose >50%)"})
          .to_string(float_format=lambda v: f"{v:+.3f}"))

    print("\n  'mean' should be roughly flat down the column. Concentration "
          "does not\n  raise expected return -- it buys win probability with "
          "variance, and the\n  P(lose >50%) column is the price.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
