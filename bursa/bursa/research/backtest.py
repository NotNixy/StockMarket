"""The harness. Signals in, equity curve out.

Two rules are enforced structurally rather than by discipline, because
discipline fails at 2am:

1. **A signal computed on date t is executed on date t+1.** Never the same
   bar. If you rank on today's close and fill at today's close, you have
   built a time machine, and the backtest will be excellent.

2. **Costs are charged on turnover, every rebalance.** Not optional, not a
   flag, not applied afterwards. `core.costs` owns the arithmetic and this
   module calls it.

The signal function is the only thing that varies. It receives the screened
panel up to and including date t, and returns target *weights* -- a mapping
from ticker to fraction of capital. Not orders, not buy/sell. That interface
is what lets the same strategy code drive a simulator, a paper account and a
live account without modification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from core.costs import MOOMOO, Broker, SlippageModel, trade_cost
from research.metrics import Performance, summarise

# A signal function: (history_up_to_t, date_t, eligible_tickers) -> weights
SignalFn = Callable[[pd.DataFrame, pd.Timestamp, list[str]], dict[str, float]]


@dataclass
class BacktestConfig:
    capital: float = 10_000.0
    rebalance: str = "ME"           # pandas offset alias: ME, W-FRI, etc.
    broker: Broker = MOOMOO
    slippage: SlippageModel = field(default_factory=SlippageModel)
    max_positions: int = 5
    allow_fractional: bool = True   # False enforces 100-share board lots
    price_col: str = "adj_close"


@dataclass
class BacktestResult:
    """Everything a run produced. Enough to audit, not just to admire."""
    returns: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    equity: pd.Series
    cost_drag: float                # total costs as a fraction of starting capital
    avg_turnover: float
    n_rebalances: int
    config: BacktestConfig

    def performance(self, name: str = "strategy", n_trials: int = 1) -> Performance:
        return summarise(self.returns, name=name,
                         turnover=self.avg_turnover, n_trials=n_trials)


def _rebalance_dates(dates: pd.DatetimeIndex, rule: str) -> list[pd.Timestamp]:
    """Last available trading day of each period, plus the very first day.

    Uses the panel's own dates rather than a generated calendar, so holidays
    and suspensions are handled by construction.

    The first date is included deliberately. Without it a period-end rule
    leaves the portfolio in cash until the first period closes -- which for an
    annual rebalance means sitting out nearly a year, making buy-and-hold look
    far worse than it is and flattering anything compared against it.
    """
    s = pd.Series(dates, index=dates)
    marks = list(s.resample(rule).last().dropna())
    first = dates[0]
    if not marks or marks[0] != first:
        marks.insert(0, first)
    return sorted(set(marks))


def _normalise_weights(weights: dict[str, float],
                       max_positions: int) -> dict[str, float]:
    """Keep the largest N, drop non-positive, rescale to sum to 1."""
    clean = {t: float(w) for t, w in weights.items()
             if w is not None and np.isfinite(w) and w > 0}
    if not clean:
        return {}
    top = dict(sorted(clean.items(), key=lambda kv: -kv[1])[:max_positions])
    total = sum(top.values())
    return {t: w / total for t, w in top.items()} if total > 0 else {}


def run_backtest(panel: pd.DataFrame,
                 signal_fn: SignalFn,
                 cfg: BacktestConfig = BacktestConfig(),
                 eligible_col: str = "eligible",
                 verbose: bool = False) -> BacktestResult:
    """Run `signal_fn` over `panel` and return the result.

    `panel` must be long format with date, ticker, the price column, and an
    `eligible` boolean from core.universe.screen().
    """
    required = {"date", "ticker", cfg.price_col}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing column(s): {sorted(missing)}")
    if eligible_col not in panel.columns:
        panel = panel.assign(**{eligible_col: True})

    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    if len(dates) < 3:
        raise ValueError("need at least 3 dates to backtest")

    prices = panel.pivot_table(index="date", columns="ticker",
                               values=cfg.price_col, aggfunc="last")
    daily_ret = prices.pct_change()

    rebal = [d for d in _rebalance_dates(dates, cfg.rebalance) if d < dates[-1]]
    if not rebal:
        raise ValueError(f"no rebalance dates under rule '{cfg.rebalance}'")

    equity = cfg.capital
    held: dict[str, float] = {}              # ticker -> weight
    rows_w, rows_t = [], []
    port_ret = pd.Series(0.0, index=dates)
    total_costs = 0.0
    turnovers = []

    # Map each rebalance date to the NEXT trading day -- the execution bar.
    next_day = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}

    for d in rebal:
        history = panel[panel["date"] <= d]
        elig = sorted(panel.loc[(panel["date"] == d)
                                & panel[eligible_col], "ticker"].unique())

        raw = signal_fn(history, d, elig) if elig else {}
        target = _normalise_weights(raw, cfg.max_positions)
        # Never hold something that was not eligible at the decision date.
        target = {t: w for t, w in target.items() if t in elig}
        target = _normalise_weights(target, cfg.max_positions)

        exec_date = next_day[d]

        # Turnover, and the cost of trading it.
        names = set(held) | set(target)
        turnover = sum(abs(target.get(t, 0.0) - held.get(t, 0.0)) for t in names)
        turnovers.append(turnover)

        cost = 0.0
        for t in names:
            delta = abs(target.get(t, 0.0) - held.get(t, 0.0))
            if delta > 1e-9:
                notional = delta * equity
                c = trade_cost(notional, cfg.broker, cfg.slippage).total
                cost += c
                rows_t.append({"date": exec_date, "ticker": t,
                               "from_w": held.get(t, 0.0),
                               "to_w": target.get(t, 0.0),
                               "notional": notional, "cost": c})
        total_costs += cost
        if equity > 0:
            port_ret[exec_date] -= cost / equity      # charged on the fill bar
            equity -= cost

        held = target
        rows_w.append({"date": exec_date, **target})
        if verbose:
            print(f"{d.date()} -> fill {exec_date.date()}  "
                  f"{len(target)} names  turnover {turnover:.2f}  cost {cost:,.2f}")

    # Accrue holding returns between rebalances.
    weights_df = (pd.DataFrame(rows_w).set_index("date")
                  if rows_w else pd.DataFrame(index=pd.DatetimeIndex([])))
    weights_df = weights_df.reindex(columns=prices.columns).fillna(0.0)
    held_daily = weights_df.reindex(dates).ffill().fillna(0.0)

    gross = (held_daily.shift(1) * daily_ret).sum(axis=1)
    port_ret = port_ret.add(gross.fillna(0.0), fill_value=0.0)
    port_ret.iloc[0] = 0.0

    eq = cfg.capital * (1.0 + port_ret).cumprod()
    return BacktestResult(
        returns=port_ret,
        weights=held_daily,
        trades=pd.DataFrame(rows_t),
        equity=eq,
        cost_drag=total_costs / cfg.capital,
        avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        n_rebalances=len(rebal),
        config=cfg,
    )
