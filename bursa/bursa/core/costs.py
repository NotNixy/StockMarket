"""Bursa Malaysia trading costs.

Pure functions: a trade goes in, a cost comes out. No I/O, no globals, no
dependency on the rest of the package -- so this file can be tested to
exhaustion, and so it is hard to quietly weaken when a backtest disappoints.

The statutory components (stamp duty, clearing fee) are set by Bursa and
apply to every broker. Brokerage varies, and for small trades the *minimum*
fee dominates everything else: on a RM 1,000 position a RM 30 minimum is a
6.26% round trip, against 0.26% with zero commission. Broker choice is worth
more than most strategies.

Rates verified September 2026 -- re-check before relying on them, especially
SST, which has moved more than once.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Statutory Bursa charges. Same for everyone.
# --------------------------------------------------------------------------
STAMP_DUTY_PER_1000 = 1.00      # RM 1 per RM 1,000 *or part thereof*
STAMP_DUTY_CAP = 200.00         # per contract
CLEARING_FEE_RATE = 0.0003      # 0.03% of contract value
CLEARING_FEE_CAP = 1000.00      # per contract

BOARD_LOT = 100                 # Bursa trades in lots of 100 shares


@dataclass(frozen=True)
class Broker:
    """A brokerage fee schedule.

    rate_pct : percentage of contract value, e.g. 0.1 for 0.1%
    min_fee  : the floor. This is what actually matters on small trades.
    sst_rate : service tax applied to the brokerage component only.
               Zero brokerage means zero SST, which is why it costs nothing
               to leave this on for the zero-commission profile.
    """
    name: str
    rate_pct: float
    min_fee: float
    sst_rate: float = 0.08

    def brokerage(self, value: float) -> float:
        gross = max(value * self.rate_pct / 100.0, self.min_fee)
        return gross * (1.0 + self.sst_rate)


# Profiles used in the cost comparison. MOOMOO is the working assumption;
# the others exist so a backtest can be re-run to show what the same strategy
# costs elsewhere -- a useful sanity check on how fragile an edge is.
MOOMOO = Broker("moomoo", rate_pct=0.0, min_fee=0.0)
BANK_LOW = Broker("bank/online (0.1%, min RM8)", rate_pct=0.1, min_fee=8.0)
BANK_STD = Broker("traditional (0.1%, min RM30)", rate_pct=0.1, min_fee=30.0)


@dataclass(frozen=True)
class SlippageModel:
    """What you lose to the spread and to moving the price.

    half_spread_bps: half the quoted bid-ask, in basis points. You cross it
        on entry and again on exit. Bursa large caps sit around 10-20 bps;
        thinly traded small caps are far worse, which is exactly why the
        liquidity screen exists upstream of this.
    impact_bps: extra cost from your own order moving the price. Negligible
        at retail size on liquid names; left here so it is explicit rather
        than silently assumed to be zero.
    """
    half_spread_bps: float = 15.0
    impact_bps: float = 0.0

    def cost(self, value: float) -> float:
        return value * (self.half_spread_bps + self.impact_bps) / 10_000.0


@dataclass(frozen=True)
class CostBreakdown:
    """Every component, kept separate. Summing early hides which part hurts."""
    value: float
    brokerage: float
    stamp_duty: float
    clearing_fee: float
    slippage: float

    @property
    def total(self) -> float:
        return self.brokerage + self.stamp_duty + self.clearing_fee + self.slippage

    @property
    def pct(self) -> float:
        """Cost as a percentage of contract value."""
        return self.total / self.value * 100.0 if self.value else 0.0

    def __str__(self) -> str:
        return (f"RM {self.value:,.2f}: brokerage {self.brokerage:.2f} + "
                f"stamp {self.stamp_duty:.2f} + clearing {self.clearing_fee:.2f} + "
                f"slippage {self.slippage:.2f} = {self.total:.2f} ({self.pct:.3f}%)")


def stamp_duty(value: float) -> float:
    """RM 1 per RM 1,000 *or part thereof*, capped at RM 200.

    The rounding-up is not a rounding error: a RM 1,001 contract pays RM 2,
    not RM 1.001. On small trades that is a real fraction of the cost.
    """
    if value <= 0:
        return 0.0
    duty = math.ceil(value / 1000.0) * STAMP_DUTY_PER_1000
    return min(duty, STAMP_DUTY_CAP)


def clearing_fee(value: float) -> float:
    """0.03% of contract value, capped at RM 1,000."""
    if value <= 0:
        return 0.0
    return min(value * CLEARING_FEE_RATE, CLEARING_FEE_CAP)


def trade_cost(value: float,
               broker: Broker = MOOMOO,
               slippage: SlippageModel | None = None) -> CostBreakdown:
    """Cost of ONE side of a trade (a buy, or a sell) on a contract of `value`.

    Buys and sells carry the same charges on Bursa, so side is not a parameter.
    """
    if value < 0:
        raise ValueError(f"contract value must be non-negative, got {value}")
    slip = slippage if slippage is not None else SlippageModel()
    return CostBreakdown(
        value=value,
        brokerage=broker.brokerage(value) if value > 0 else 0.0,
        stamp_duty=stamp_duty(value),
        clearing_fee=clearing_fee(value),
        slippage=slip.cost(value),
    )


def round_trip_pct(value: float,
                   broker: Broker = MOOMOO,
                   slippage: SlippageModel | None = None) -> float:
    """Buy and sell the same position: total cost as a % of position value.

    This is the number a strategy must beat before it has earned anything.
    """
    one_side = trade_cost(value, broker, slippage)
    return 2.0 * one_side.total / value * 100.0 if value else 0.0


def breakeven_move_pct(value: float,
                       broker: Broker = MOOMOO,
                       slippage: SlippageModel | None = None) -> float:
    """How far the price must move in your favour just to break even.

    Identical to round_trip_pct, but named for how it is used -- as the hurdle
    a signal has to clear, not as an accounting figure.
    """
    return round_trip_pct(value, broker, slippage)


def affordable_lots(price: float, budget: float) -> int:
    """Whole board lots of 100 shares affordable at `price` within `budget`.

    Board lots are why a small account cannot diversify: at RM 12 a share,
    one lot is RM 1,200, so a RM 10,000 account holds eight positions at most.
    """
    if price <= 0 or budget <= 0:
        return 0
    return int(budget // (price * BOARD_LOT))
