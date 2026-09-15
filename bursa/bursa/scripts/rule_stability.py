"""Is the selection rule stable, or did it just win one backtest?

    python -m scripts.rule_stability
    python -m scripts.rule_stability --years 2 --trials 2500

The contest simulator run over all history says momentum is the best rule at
k=1. That is one number from one sample, and it is exactly the shape of
result that overfits: four rules were compared, one came first, and reporting
the winner as "the rule" is how a backtest lies to its author.

So this re-runs the comparison inside consecutive slices of history and asks a
narrower question: **does the same rule win twice?**

## What it found

    period       random  momentum  high_vol  reversal     best
    2017-2018     2.62%     8.63%     6.21%     6.08%  momentum
    2019-2020     2.03%    12.56%    11.63%     6.04%  momentum
    2021-2022     1.72%     6.80%     4.15%     5.76%  momentum
    2023-2024     2.68%     6.70%     8.04%     3.37%  high_vol
    2025-2026     2.51%     4.97%    12.54%     5.15%  high_vol

Three things worth separating:

**Concentration is the stable edge.** `random` at k=1 sits between 1.72% and
2.68% in every period -- roughly 9x the 0.25% fair share -- without any
selection skill at all. That is the part you can count on.

**The variance tilt is real but not precise.** momentum or high_vol wins
every single period; reversal and random never do. But the magnitude swings
between 4% and 13%, so quoting a single figure like "7.4%" implies a
precision the data does not support. The honest statement is 5-12%.

**momentum vs high_vol is a coin flip.** Means of 7.93% and 8.51% across five
periods, each winning some. They are the same idea wearing different clothes:
both select names that have recently moved a long way, and a name that has
recently moved a long way has a wide forward distribution. Choosing between
them on a 0.6 percentage-point mean difference would be fitting noise.

## Why this matters more than it looks

A ten-month slice of real (non-backfilled) Shariah data showed reversal
winning at 11.88% and high_vol collapsing to 0.03%. Read alone, that looks
like evidence the whole finding was a lookahead artefact. Read against this
table, it is one regime out of five behaving differently -- about nine
independent months, which is nothing. The per-period view is what tells them
apart, and neither number means much without it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from tournament.contest import RULES, ContestConfig, run_contest

# A plausible amateur field: a few punters, most holding a handful, some
# holding a "proper" diversified book. P(win) is more sensitive to this than
# to anything else in the model -- see tournament.contest.
FIELD_MIX = {1: 0.05, 2: 0.10, 3: 0.15, 5: 0.30, 8: 0.20, 12: 0.12, 20: 0.08}


def stability_table(panel: pd.DataFrame, years: int = 2, k: int = 1,
                    n_entrants: int = 400, trials: int = 2500,
                    rules: tuple[str, ...] = RULES,
                    verbose: bool = True) -> pd.DataFrame:
    """P(win) per rule, per consecutive slice of history."""
    start = panel["date"].min().year
    end = panel["date"].max().year
    rows = []
    for y0 in range(start, end + 1, years):
        sl = panel[(panel["date"] >= f"{y0}-01-01")
                   & (panel["date"] < f"{y0 + years}-01-01")]
        # A slice too short to hold a lookback plus a horizon tells you
        # nothing, and would otherwise appear as a confident zero.
        if sl["date"].nunique() < 150:
            continue
        res = {}
        for rule in rules:
            cfg = ContestConfig(n_entrants=n_entrants, my_k=k, my_rule=rule,
                                field_k_mix=FIELD_MIX, n_trials=trials,
                                seed=41)
            res[rule] = run_contest(sl, cfg).p_win
        rows.append({"period": f"{y0}-{y0 + years - 1}", **res,
                     "best": max(res, key=res.get)})
        if verbose:
            print(f"  {rows[-1]['period']:<12}"
                  + "".join(f"{res[r]:>10.2%}" for r in rules)
                  + f"{rows[-1]['best']:>11}")
    return pd.DataFrame(rows).set_index("period")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=Path("data/screened.parquet"))
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--entrants", type=int, default=400)
    ap.add_argument("--trials", type=int, default=2500)
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    fair = 1.0 / args.entrants

    print(f"P(win) at k={args.k} by rule, in {args.years}-year slices")
    print("The question is not which rule wins. It is whether the SAME rule "
          "wins twice.\n")
    print(f"  {'period':<12}" + "".join(f"{r:>10}" for r in RULES)
          + f"{'best':>11}")
    print("  " + "-" * (12 + 10 * len(RULES) + 11))

    df = stability_table(panel, args.years, args.k, args.entrants, args.trials)
    if df.empty:
        print("\n  no slice had enough history.", file=sys.stderr)
        return 1

    print(f"\n  fair share: {fair:.2%}")
    print(f"  distinct winning rules: {df['best'].nunique()} across "
          f"{len(df)} periods\n")
    print("  per-rule range across periods:")
    for r in RULES:
        v = df[r]
        flag = ("  <- stable" if (v.max() - v.min()) < 0.02 else "")
        print(f"    {r:<11} {v.min():>7.2%} .. {v.max():>7.2%}   "
              f"mean {v.mean():>6.2%}   wins {int((df['best'] == r).sum())}"
              f"{flag}")

    rand_mean = df["random"].mean()
    print(f"\n  Concentration alone (random at k={args.k}) is worth "
          f"{rand_mean / fair:.0f}x fair share\n  and barely moves between "
          f"periods. That is the part you can rely on.")
    print("  The tilt adds more, but its size swings by period -- quote it "
          "as a range,\n  never as a single figure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
