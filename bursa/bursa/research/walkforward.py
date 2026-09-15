"""Walk-forward evaluation. The difference between a number and a finding.

A backtest run once over all history tells you how a strategy would have done
if you had known, in 2018, which strategy to pick. You did not. Walk-forward
removes that advantage: choose on data you could have had, evaluate on data
you could not, and repeat.

Three things this produces that a single backtest cannot:

**In-sample versus out-of-sample.** The gap between them IS the overfitting.
A strategy with IS Sharpe 1.2 and OOS Sharpe 0.1 has told you nothing except
that you searched hard. Reporting only the first number is the most common
way a backtest misleads its own author.

**Selection under uncertainty.** At each step the best strategy on the
training window is chosen, then evaluated blind. That is what you would
actually do, so it is what should be measured -- not the performance of the
winner chosen with hindsight.

**A holdout nobody has touched.** The final slice is set aside and evaluated
exactly once, at the end. Every look at it costs a trial.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from research.backtest import BacktestConfig, run_backtest
from research.metrics import deflated_sharpe, max_drawdown, sharpe, total_return

# A strategy factory: name -> signal function. Kept as a factory rather than a
# built object so each fold gets a clean instance with no carried state.
StrategyFactory = Callable[[], object]


@dataclass
class Fold:
    """One train/test pair and what happened in it."""
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    chosen: str = ""
    is_sharpe: float = np.nan          # of the chosen strategy, on train
    oos_sharpe: float = np.nan         # of the same strategy, on test
    oos_return: float = np.nan
    oos_maxdd: float = np.nan
    candidates: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __str__(self) -> str:
        return (f"  fold {self.index}  train {self.train_start.date()}→"
                f"{self.train_end.date()}  test {self.test_start.date()}→"
                f"{self.test_end.date()}\n"
                f"    chose {self.chosen:<24} IS {self.is_sharpe:+5.2f}  "
                f"OOS {self.oos_sharpe:+5.2f}  ret {self.oos_return:+7.2%}")


@dataclass
class WalkForwardResult:
    folds: list[Fold]
    oos_returns: pd.Series
    holdout_start: pd.Timestamp | None = None

    @property
    def mean_is(self) -> float:
        v = [f.is_sharpe for f in self.folds if np.isfinite(f.is_sharpe)]
        return float(np.mean(v)) if v else np.nan

    @property
    def mean_oos(self) -> float:
        v = [f.oos_sharpe for f in self.folds if np.isfinite(f.oos_sharpe)]
        return float(np.mean(v)) if v else np.nan

    # Below this, the in-sample Sharpe is indistinguishable from zero and the
    # ratio below it means nothing. Measured: the Shariah universe produced a
    # mean IS Sharpe of 0.098 and an OOS of 0.680, giving a "survival ratio"
    # of 6.97 -- which reads as a spectacular result and is pure division by
    # almost-zero. There was no in-sample edge to survive.
    MIN_MEANINGFUL_IS = 0.20

    @property
    def degradation(self) -> float:
        """How much of the in-sample edge survived. 1.0 = all of it, 0 = none.

        NaN when there was no in-sample edge to begin with. A ratio is only
        interpretable if its denominator is real.
        """
        if not np.isfinite(self.mean_is) or self.mean_is < self.MIN_MEANINGFUL_IS:
            return np.nan
        return float(self.mean_oos / self.mean_is)

    @property
    def stitched_sharpe(self) -> float:
        """Sharpe of the concatenated out-of-sample returns.

        This is the honest headline: the equity curve you would actually have
        experienced, choosing as you went.
        """
        return sharpe(self.oos_returns)

    def summary(self) -> str:
        lines = ["WALK-FORWARD", ""]
        for f in self.folds:
            lines.append(str(f))
        lines += [
            "",
            f"  mean IS Sharpe   {self.mean_is:+.3f}",
            f"  mean OOS Sharpe  {self.mean_oos:+.3f}",
        ]
        if np.isfinite(self.degradation):
            lines.append(f"  survival ratio   {self.degradation:+.2f}   "
                         f"(OOS / IS -- below ~0.5 means most was fitted)")
        else:
            lines.append(f"  survival ratio   n/a  -- mean IS Sharpe "
                         f"{self.mean_is:+.3f} is too close to zero for the")
            lines.append(f"                   ratio to mean anything. There "
                         f"was no in-sample edge to survive;")
            lines.append(f"                   any OOS result here is a "
                         f"statement about the test period,")
            lines.append(f"                   not about the strategy.")
        lines += [
            f"  stitched OOS     SR {self.stitched_sharpe:+.3f}  "
            f"ret {total_return(self.oos_returns):+.2%}  "
            f"maxDD {max_drawdown(self.oos_returns):.2%}",
        ]
        return "\n".join(lines)


def make_splits(dates: pd.DatetimeIndex,
                n_folds: int = 5,
                train_months: int = 24,
                test_months: int = 6,
                expanding: bool = True,
                holdout_months: int = 12) -> tuple[list[tuple], pd.Timestamp]:
    """Generate (train_start, train_end, test_start, test_end) windows.

    `expanding=True` grows the training window each fold, which matches how
    you would really work -- you do not throw away history. `False` rolls a
    fixed window, which adapts faster to regime change.

    The last `holdout_months` are excluded entirely and returned separately.
    Nothing in the walk-forward may see them.
    """
    dates = pd.DatetimeIndex(sorted(dates))
    if len(dates) < 100:
        raise ValueError("not enough history to walk forward")

    holdout_start = dates[-1] - pd.DateOffset(months=holdout_months)
    usable = dates[dates < holdout_start]
    if len(usable) < 100:
        raise ValueError(f"holdout of {holdout_months} months leaves too "
                         f"little history to train on")

    splits = []
    first, last = usable[0], usable[-1]
    # Place the final test window flush against the holdout, then step back.
    for k in range(n_folds):
        test_end = last - pd.DateOffset(months=test_months * k)
        test_start = test_end - pd.DateOffset(months=test_months)
        train_end = test_start
        train_start = (first if expanding
                       else train_end - pd.DateOffset(months=train_months))
        if train_start < first:
            train_start = first
        # Need a meaningful training window.
        if (train_end - train_start).days < train_months * 30 * 0.6:
            continue
        splits.append((train_start, train_end, test_start, test_end))

    return list(reversed(splits)), holdout_start


def walk_forward(panel: pd.DataFrame,
                 strategies: dict[str, StrategyFactory],
                 cfg: BacktestConfig,
                 n_folds: int = 5,
                 train_months: int = 24,
                 test_months: int = 6,
                 expanding: bool = True,
                 holdout_months: int = 12,
                 verbose: bool = True) -> WalkForwardResult:
    """Select on train, evaluate on test, fold by fold.

    `strategies` maps a name to a factory. At each fold every candidate is run
    on the training window, the best by Sharpe is chosen, and THAT ONE is run
    on the test window. The choice is made with no knowledge of the test data,
    which is the entire point.
    """
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    splits, holdout_start = make_splits(dates, n_folds, train_months,
                                        test_months, expanding, holdout_months)
    if not splits:
        raise ValueError("no usable folds -- shorten train_months or "
                         "holdout_months")

    folds, oos_pieces = [], []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(splits, 1):
        train = panel[(panel["date"] >= tr_s) & (panel["date"] < tr_e)]
        test = panel[(panel["date"] >= te_s) & (panel["date"] < te_e)]
        if train["date"].nunique() < 60 or test["date"].nunique() < 10:
            continue

        rows = []
        for name, factory in strategies.items():
            try:
                r = run_backtest(train, factory().signal, cfg)
                rows.append({"strategy": name, "train_sharpe": sharpe(r.returns),
                             "train_return": total_return(r.returns)})
            except Exception as e:                 # a fold may starve a strategy
                rows.append({"strategy": name, "train_sharpe": np.nan,
                             "train_return": np.nan, "error": str(e)[:60]})
        cand = pd.DataFrame(rows).sort_values("train_sharpe", ascending=False)

        if cand["train_sharpe"].isna().all():
            continue
        best = cand.iloc[0]["strategy"]

        # Run over everything up to the end of the test window, then MEASURE
        # only the test period. Handing the signal the bare test slice would
        # starve it -- a 6-month window is ~130 bars and a momentum feature
        # needs 150, so every position would come back empty. In reality you
        # reach the test period holding all prior history, and that is what
        # this reproduces. Selection still used the training window alone.
        through_test = panel[panel["date"] < te_e]
        full = run_backtest(through_test, strategies[best]().signal, cfg)
        test_res_returns = full.returns[full.returns.index >= te_s]
        oos_pieces.append(test_res_returns)

        class _R:                       # thin holder so the fields below read
            returns = test_res_returns  # the same as before
        test_res = _R()

        f = Fold(index=i, train_start=tr_s, train_end=tr_e,
                 test_start=te_s, test_end=te_e, chosen=best,
                 is_sharpe=float(cand.iloc[0]["train_sharpe"]),
                 oos_sharpe=sharpe(test_res.returns),
                 oos_return=total_return(test_res.returns),
                 oos_maxdd=max_drawdown(test_res.returns),
                 candidates=cand)
        folds.append(f)
        if verbose:
            print(f)

    oos = (pd.concat(oos_pieces).sort_index() if oos_pieces
           else pd.Series(dtype=float))
    oos = oos[~oos.index.duplicated(keep="first")]

    result = WalkForwardResult(folds, oos, holdout_start)
    if verbose:
        print()
        print(result.summary())
        print(f"\n  holdout begins {holdout_start.date()} and has NOT been "
              f"touched.\n  Evaluate it once, at the end, and count that as a "
              f"trial.")
    return result


def evaluate_holdout(panel: pd.DataFrame,
                     strategy_factory: StrategyFactory,
                     cfg: BacktestConfig,
                     holdout_start: pd.Timestamp,
                     n_trials: int = 1,
                     verbose: bool = True) -> dict:
    """Run the chosen strategy on the untouched holdout. Once.

    Every call to this is a trial. Calling it repeatedly while adjusting the
    strategy turns the holdout into another training set, silently.
    """
    hold = panel[panel["date"] >= holdout_start]
    if hold["date"].nunique() < 20:
        raise ValueError("holdout too short to evaluate")

    # Same reasoning as the folds: the signal needs the history it would
    # really have had. Only the holdout period is measured.
    full = run_backtest(panel, strategy_factory().signal, cfg)
    r = full.returns[full.returns.index >= holdout_start]

    out = {
        "start": hold["date"].min(), "end": hold["date"].max(),
        "sharpe": sharpe(r),
        "total_return": total_return(r),
        "max_drawdown": max_drawdown(r),
        "deflated_sharpe": deflated_sharpe(r, n_trials),
        "n_trials_declared": n_trials,
    }
    if verbose:
        print(f"HOLDOUT  {out['start'].date()} → {out['end'].date()}")
        print(f"  Sharpe          {out['sharpe']:+.3f}")
        print(f"  return          {out['total_return']:+.2%}")
        print(f"  max drawdown    {out['max_drawdown']:.2%}")
        print(f"  deflated Sharpe {out['deflated_sharpe']:.3f} "
              f"(declaring {n_trials} trials)")
        print("\n  This was one trial. Adjusting the strategy and re-running "
              "makes it two,\n  and the holdout stops being a holdout.")
    return out
