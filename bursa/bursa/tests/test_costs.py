"""Tests for the Bursa cost model.

Every expected value here is computed by hand in the comment above it. If a
test fails, one of the two is wrong -- and finding out which is the point.
"""
import math

import pytest

from core.costs import (
    BANK_LOW, BANK_STD, MOOMOO, Broker, CostBreakdown, SlippageModel,
    affordable_lots, clearing_fee, round_trip_pct, stamp_duty, trade_cost,
)

NO_SLIP = SlippageModel(half_spread_bps=0.0, impact_bps=0.0)


# ---------------------------------------------------------------- stamp duty
def test_stamp_duty_rounds_up_to_the_next_thousand():
    # RM 1 per RM 1,000 OR PART THEREOF.
    assert stamp_duty(1000) == 1.00      # exactly one block
    assert stamp_duty(1001) == 2.00      # a single ringgit over -> two blocks
    assert stamp_duty(500) == 1.00       # part of a block is still a block
    assert stamp_duty(1) == 1.00
    assert stamp_duty(0) == 0.00


def test_stamp_duty_caps_at_200():
    # cap bites at RM 200,000 (200 blocks)
    assert stamp_duty(200_000) == 200.00
    assert stamp_duty(500_000) == 200.00


# -------------------------------------------------------------- clearing fee
def test_clearing_fee_is_three_basis_points():
    assert clearing_fee(10_000) == pytest.approx(3.00)      # 10000 * 0.0003
    assert clearing_fee(1_000) == pytest.approx(0.30)


def test_clearing_fee_caps_at_1000():
    # cap bites above RM 3,333,333
    assert clearing_fee(10_000_000) == 1000.00


# ----------------------------------------------------------------- brokerage
def test_zero_commission_broker_charges_nothing():
    # 0% brokerage means SST on zero is also zero.
    assert MOOMOO.brokerage(10_000) == 0.0


def test_minimum_fee_dominates_small_trades():
    # RM 1,000 at 0.1% = RM 1, but the RM 8 floor applies. Plus 8% SST.
    assert BANK_LOW.brokerage(1_000) == pytest.approx(8.0 * 1.08)
    # RM 50,000 at 0.1% = RM 50, above the floor.
    assert BANK_LOW.brokerage(50_000) == pytest.approx(50.0 * 1.08)


# ------------------------------------------------------------ one-side total
def test_moomoo_one_side_on_1000():
    # brokerage 0 + stamp 1.00 + clearing 0.30 + slippage 0 = 1.30
    c = trade_cost(1_000, MOOMOO, NO_SLIP)
    assert c.total == pytest.approx(1.30)
    assert c.pct == pytest.approx(0.130, abs=1e-3)


def test_components_sum_to_total():
    c = trade_cost(7_531, BANK_LOW)
    assert c.total == pytest.approx(
        c.brokerage + c.stamp_duty + c.clearing_fee + c.slippage)


# ----------------------------------------------------------------- round trip
@pytest.mark.parametrize("value,broker,expected_pct", [
    # moomoo, RM 1,000: 2 * (0 + 1.00 + 0.30) / 1000 = 0.26%
    (1_000, MOOMOO, 0.260),
    # moomoo, RM 10,000: 2 * (0 + 10.00 + 3.00) / 10000 = 0.26%
    (10_000, MOOMOO, 0.260),
    # bank low, RM 1,000: 2 * (8.64 + 1.00 + 0.30) / 1000 = 1.988%
    (1_000, BANK_LOW, 1.988),
    # bank std, RM 1,000: 2 * (32.40 + 1.00 + 0.30) / 1000 = 6.740%
    (1_000, BANK_STD, 6.740),
])
def test_round_trip_matches_hand_computation(value, broker, expected_pct):
    assert round_trip_pct(value, broker, NO_SLIP) == pytest.approx(
        expected_pct, abs=0.002)


def test_slippage_adds_twice_the_half_spread():
    # 15 bps each way = 30 bps = 0.30 percentage points on the round trip.
    bare = round_trip_pct(10_000, MOOMOO, NO_SLIP)
    with_slip = round_trip_pct(10_000, MOOMOO, SlippageModel(15.0, 0.0))
    assert with_slip - bare == pytest.approx(0.30, abs=1e-6)


def test_zero_commission_round_trip_is_flat_in_size():
    # Statutory charges are proportional (below their caps), so the percentage
    # cost does not improve with size -- unlike a minimum-fee broker.
    small = round_trip_pct(5_000, MOOMOO, NO_SLIP)
    large = round_trip_pct(50_000, MOOMOO, NO_SLIP)
    assert small == pytest.approx(large, abs=0.01)


def test_minimum_fee_broker_gets_cheaper_with_size():
    assert round_trip_pct(1_000, BANK_STD, NO_SLIP) > \
           round_trip_pct(50_000, BANK_STD, NO_SLIP)


# --------------------------------------------------------------- board lots
def test_affordable_lots_respects_board_lot_of_100():
    # RM 12 a share -> RM 1,200 a lot -> 8 lots from RM 10,000.
    assert affordable_lots(12.0, 10_000) == 8
    # Cannot afford a single lot.
    assert affordable_lots(12.0, 500) == 0


def test_affordable_lots_handles_nonsense_input():
    assert affordable_lots(0, 10_000) == 0
    assert affordable_lots(-5, 10_000) == 0
    assert affordable_lots(12.0, 0) == 0


# ------------------------------------------------------------------- guards
def test_negative_contract_value_is_rejected():
    with pytest.raises(ValueError):
        trade_cost(-100)


def test_zero_value_does_not_divide_by_zero():
    assert trade_cost(0, MOOMOO, NO_SLIP).pct == 0.0
    assert round_trip_pct(0, MOOMOO, NO_SLIP) == 0.0
