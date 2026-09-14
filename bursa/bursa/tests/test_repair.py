"""Tests for the repair layer.

Built from faults found in real Bursa data, so the fixtures mirror what
actually happened rather than an invented failure.
"""
import numpy as np
import pandas as pd
import pytest

from core.repair import (
    KNOWN_RATIOS, RepairReport, apply_split_adjustment, detect_splits,
    find_bad_bars, find_unadjusted_tickers, repair,
)


def panel(n=300, ticker="AAA.KL", start=10.0, adj_factor=0.97):
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    px = np.linspace(start, start * 1.3, n)
    return pd.DataFrame({
        "date": dates, "ticker": ticker,
        "open": px * 0.998, "high": px * 1.01, "low": px * 0.99,
        "close": px, "adj_close": px * adj_factor, "volume": 1_000_000,
    })


def with_unadjusted_split(n=300, at=150, ratio=2.0):
    """The VITROX case: price halves in BOTH series and never recovers."""
    p = panel(n)
    p.loc[p.index >= at, ["open", "high", "low", "close", "adj_close"]] /= ratio
    return p


# ------------------------------------------------------------------ detection
def test_an_unadjusted_two_for_one_split_is_detected():
    d = detect_splits(with_unadjusted_split())
    assert len(d) == 1
    assert bool(d["is_split"].iloc[0])
    assert d["ratio"].iloc[0] == pytest.approx(2.0, abs=0.05)


@pytest.mark.parametrize("ratio", KNOWN_RATIOS)
def test_every_known_ratio_is_recognised(ratio):
    d = detect_splits(with_unadjusted_split(ratio=ratio))
    assert bool(d["is_split"].iloc[0]), f"missed a {ratio}:1 split"


def test_vitrox_actual_ratio_is_recognised():
    # The real one measured 1.995, not a clean 2.0.
    d = detect_splits(with_unadjusted_split(ratio=1.995))
    assert bool(d["is_split"].iloc[0])


def test_a_genuine_crash_is_not_called_a_split():
    """A 40% fall that matches no split ratio must be left alone. Rewriting a
    real crash would be far worse than missing a split."""
    p = panel()
    p.loc[p.index >= 150, ["open", "high", "low", "close", "adj_close"]] *= 0.60
    d = detect_splits(p)
    assert len(d) == 1
    assert not bool(d["is_split"].iloc[0])


def test_quiet_series_produces_no_detections():
    assert detect_splits(panel()).empty


def test_upward_split_is_detected():
    p = panel()
    p.loc[p.index >= 150, ["open", "high", "low", "close", "adj_close"]] *= 2.0
    d = detect_splits(p)
    assert bool(d["is_split"].iloc[0])


# ----------------------------------------------------------------- adjustment
def test_back_adjustment_removes_the_discontinuity():
    p = with_unadjusted_split()
    fixed = apply_split_adjustment(p, detect_splits(p))
    ret = fixed.sort_values("date")["adj_close"].pct_change()
    assert ret.abs().max() < 0.30, "discontinuity survived the repair"


def test_back_adjustment_scales_volume_the_other_way():
    p = with_unadjusted_split()
    fixed = apply_split_adjustment(p, detect_splits(p))
    pre = fixed[fixed["date"] < fixed["date"].iloc[150]]["volume"].iloc[0]
    post = fixed[fixed["date"] >= fixed["date"].iloc[150]]["volume"].iloc[-1]
    assert pre == pytest.approx(post * 2, rel=0.01)


def test_back_adjustment_leaves_post_split_prices_untouched():
    p = with_unadjusted_split()
    fixed = apply_split_adjustment(p, detect_splits(p))
    after = p["date"].iloc[200]
    assert fixed.loc[fixed.date == after, "close"].iloc[0] == pytest.approx(
        p.loc[p.date == after, "close"].iloc[0])


def test_a_non_split_move_is_not_adjusted():
    p = panel()
    p.loc[p.index >= 150, ["open", "high", "low", "close", "adj_close"]] *= 0.60
    fixed = apply_split_adjustment(p, detect_splits(p))
    expected = p.sort_values(["ticker", "date"]).reset_index(drop=True)
    expected["volume"] = expected["volume"].astype("float64")
    pd.testing.assert_frame_equal(fixed.reset_index(drop=True), expected)


# ----------------------------------------------------------------- quarantine
def test_a_ticker_with_no_adjustment_is_found():
    """The FRONTKEN case: adj_close identical to close for years."""
    good = panel(ticker="GOOD.KL", adj_factor=0.97)
    bad = panel(ticker="BAD.KL", adj_factor=1.0)
    assert find_unadjusted_tickers(pd.concat([good, bad])) == ["BAD.KL"]


def test_properly_adjusted_tickers_are_not_quarantined():
    assert find_unadjusted_tickers(panel()) == []


# ------------------------------------------------------------------ bad bars
def test_a_close_above_the_high_is_caught():
    """The CelcomDigi case: close 4.49 against a high of 4.48."""
    p = panel()
    p.loc[10, "close"] = p.loc[10, "high"] + 0.01
    assert len(find_bad_bars(p)) == 1


def test_non_positive_prices_are_caught():
    p = panel()
    p.loc[5, "low"] = 0.0
    assert len(find_bad_bars(p)) == 1


def test_clean_bars_are_left_alone():
    assert find_bad_bars(panel()).empty


# --------------------------------------------------------------- orchestration
def test_repair_handles_all_three_faults_at_once():
    split = with_unadjusted_split()                       # AAA.KL, split
    unadj = panel(ticker="UNADJ.KL", adj_factor=1.0)      # never adjusted
    broken = panel(ticker="BROKEN.KL")
    broken.loc[10, "close"] = broken.loc[10, "high"] + 1  # bad bar

    out, rep = repair(pd.concat([split, unadj, broken], ignore_index=True),
                      verbose=False)

    assert len(rep.splits_fixed) == 1
    assert "UNADJ.KL" in rep.quarantined
    assert len(rep.bars_dropped) == 1
    assert "UNADJ.KL" not in set(out["ticker"])
    assert len(out) == len(split) + len(broken) - 1


def test_repair_can_be_told_to_do_nothing():
    p = with_unadjusted_split()
    out, rep = repair(p, fix_splits=False, quarantine_unadjusted=False,
                      drop_bad_bars=False, verbose=False)
    assert len(out) == len(p)


def test_repair_report_renders():
    p = with_unadjusted_split()
    _, rep = repair(p, verbose=False)
    assert "splits back-adjusted" in str(rep)


def test_repair_is_idempotent():
    """Running it twice must not double-adjust."""
    p = with_unadjusted_split()
    once, _ = repair(p, verbose=False)
    twice, rep2 = repair(once, verbose=False)
    assert len(rep2.splits_fixed) == 0
    pd.testing.assert_frame_equal(once, twice)


# ---------------------------------------------------- false-positive defence
# These encode the real Bursa cases. An earlier 4% ratio tolerance accepted
# both 5202 moves and rewrote genuine price history; these stop that returning.

def rally(n=300, at=150, jump=1.541, vol_spike=59, keeps_running=True):
    """5202's 2021 move: a speculative run that resembles a 3-for-2 split."""
    p = panel(n)
    p.loc[p.index >= at, ["open", "high", "low", "close", "adj_close"]] *= jump
    if keeps_running:                       # and then keeps going
        for k in range(1, 4):
            p.loc[p.index >= at + k,
                  ["open", "high", "low", "close", "adj_close"]] *= 1.15
    p.loc[at, "volume"] = int(1_000_000 * vol_spike)
    return p


def test_a_speculative_rally_is_not_treated_as_a_split():
    d = detect_splits(rally())
    assert len(d) >= 1
    first = d.iloc[0]
    assert not bool(first["is_split"]), "rewrote a real rally as a split"


def test_a_ratio_slightly_off_a_known_one_is_rejected():
    # 1.541 is 2.7% from 1.5 -- real 5202 case, must fail the 1.5% tolerance.
    d = detect_splits(with_unadjusted_split(ratio=1.541))
    assert not bool(d["is_split"].iloc[0])


def test_a_huge_volume_spike_blocks_a_split_call():
    # Even at a clean 2.0 ratio, a 50x volume day is not a re-denomination.
    p = with_unadjusted_split(ratio=2.0)
    p.loc[150, "volume"] = 50_000_000
    d = detect_splits(p)
    assert not bool(d["is_split"].iloc[0])
    assert not bool(d["volume_ok"].iloc[0])


def test_price_that_keeps_moving_blocks_a_split_call():
    p = with_unadjusted_split(ratio=2.0)
    for k in range(1, 4):                    # drifts on after the step
        p.loc[p.index >= 150 + k,
              ["open", "high", "low", "close", "adj_close"]] *= 1.12
    d = detect_splits(p)
    assert not bool(d["stable_after"].iloc[0])
    assert not bool(d["is_split"].iloc[0])


def test_the_real_vitrox_case_still_passes_all_three_gates():
    # ratio 2.014, modest volume, dead flat afterwards.
    p = with_unadjusted_split(ratio=2.014)
    d = detect_splits(p)
    row = d.iloc[0]
    assert bool(row["ratio_ok"]) and bool(row["volume_ok"]) \
        and bool(row["stable_after"]) and bool(row["is_split"])
