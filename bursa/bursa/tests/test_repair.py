"""Tests for the repair layer.

Built from faults found in real Bursa data, so the fixtures mirror what
actually happened rather than an invented failure.
"""
import numpy as np
import pandas as pd
import pytest

from core.repair import (
    locate_recorded_splits,
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
    # Noted, but NOT dropped. The check cannot tell "dividends never
    # adjusted" from "never paid one", so it reports instead of deciding.
    assert "UNADJ.KL" in rep.unadjusted_noted
    assert "UNADJ.KL" not in rep.quarantined
    assert "UNADJ.KL" in set(out["ticker"])
    assert len(rep.bars_dropped) == 1
    assert len(out) == len(split) + len(unadj) + len(broken) - 1


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


# ------------------------------------------------- penny stocks and the record
# These cover the regression found by running the detector over all 865
# Shariah-compliant tickers instead of a 58-name large-cap sample. It reported
# 1,104 splits. Ticker 0034 alone showed ~30 alternating +50% / -33.3% steps
# inside five months -- a stock at RM 0.02 moving one half-sen tick.
def penny_oscillation(n=300, low=0.02, high=0.03):
    """A two-sen stock ticking between two adjacent prices.

    Every up move is +50% (ratio 1.5, a KNOWN_RATIO) and every down move is
    -33.3% (also ratio 1.5). Volume is flat, so no spike. It ticks back, so
    there is no drift. It passes all three of the original gates.
    """
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    px = np.where(np.arange(n) % 2 == 0, low, high).astype(float)
    return pd.DataFrame({
        "date": dates, "ticker": "PENNY.KL",
        "open": px, "high": px, "low": px, "close": px,
        "adj_close": px * 0.99, "volume": 50_000,
    })


def test_penny_tick_oscillation_is_not_a_split():
    """The regression itself. Without the price floor this returns hundreds."""
    d = detect_splits(penny_oscillation())
    assert len(d) > 100, "fixture should trip the ratio and volume gates"
    assert not d["is_split"].any(), (
        f"{int(d['is_split'].sum())} tick moves called splits")


def test_penny_moves_fail_specifically_on_price_not_by_accident():
    """Guard the reason, not just the outcome.

    If a later change makes these fail some other gate, the price floor could
    be removed without any test noticing.
    """
    d = detect_splits(penny_oscillation())
    assert d["ratio_ok"].any(), "a half-sen tick IS a clean 1.5 ratio"
    assert not d["price_ok"].any()


def test_a_real_split_still_passes_the_price_floor():
    d = detect_splits(with_unadjusted_split(ratio=2.0))   # RM 10 stock
    assert bool(d["price_ok"].iloc[0])
    assert bool(d["is_split"].iloc[0])


def recorded(ticker="AAA.KL", date="2023-08-01", value=2.0):
    return pd.DataFrame({"date": [pd.Timestamp(date)],
                         "ticker": [ticker], "value": [value]})


def test_a_split_on_record_is_repaired():
    p = with_unadjusted_split(at=150, ratio=2.0)
    step = p["date"].iloc[150]
    d = detect_splits(p, recorded(date=step))
    assert bool(d["confirmed"].iloc[0])
    assert bool(d["is_split"].iloc[0])


def test_a_price_step_with_no_action_on_record_is_left_alone():
    """The whole point of fetching the event list.

    The pattern is a textbook 2-for-1. Nobody recorded a split. It stays."""
    p = with_unadjusted_split(at=150, ratio=2.0)
    other = recorded(ticker="ZZZ.KL")           # a record, but not for AAA
    d = detect_splits(p, other)
    assert not bool(d["confirmed"].iloc[0])
    assert not bool(d["is_split"].iloc[0])


def test_confirmation_tolerates_yahoos_date_being_weeks_off():
    """VITROX: Yahoo dated the split 2024-06-10, the price stepped 2024-05-02."""
    p = with_unadjusted_split(at=150, ratio=2.0)
    step = p["date"].iloc[150]
    late = recorded(date=step + pd.Timedelta(days=35))
    assert bool(detect_splits(p, late)["is_split"].iloc[0])


def test_confirmation_rejects_an_action_from_a_different_quarter():
    p = with_unadjusted_split(at=150, ratio=2.0)
    step = p["date"].iloc[150]
    far = recorded(date=step + pd.Timedelta(days=200))
    assert not bool(detect_splits(p, far)["is_split"].iloc[0])


def test_direction_must_match_the_recorded_action():
    """A 2-for-1 split halves the price. It cannot explain a doubling.

    Matching on ratio magnitude alone would accept this, and back-adjust the
    wrong way -- turning one bad bar into a whole rewritten history.
    """
    p = panel()
    p.loc[p.index >= 150, ["open", "high", "low", "close", "adj_close"]] *= 2.0
    step = p["date"].iloc[150]
    d = detect_splits(p, recorded(date=step, value=2.0))   # forward split
    assert not bool(d["confirmed"].iloc[0])


def test_a_consolidation_is_confirmed_by_its_own_direction():
    """Yahoo writes a 1-for-5 reverse split as value 0.2; the price rises 5x."""
    p = panel()
    p.loc[p.index >= 150, ["open", "high", "low", "close", "adj_close"]] *= 5.0
    step = p["date"].iloc[150]
    d = detect_splits(p, recorded(date=step, value=0.2))
    assert bool(d["confirmed"].iloc[0])
    assert bool(d["is_split"].iloc[0])


def test_repair_passes_the_record_through():
    p = with_unadjusted_split(at=150, ratio=2.0)
    step = p["date"].iloc[150]
    _, with_rec = repair(p, recorded(date=step), verbose=False)
    _, no_rec = repair(p, pd.DataFrame(columns=["date", "ticker", "value"]),
                       verbose=False)
    assert len(with_rec.splits_fixed) == 1
    # An EMPTY record is treated as "no record available", so the heuristic
    # still runs. An empty file must not silently disable every repair.
    assert len(no_rec.splits_fixed) == 1


def test_quarantine_is_opt_in_and_still_works_when_asked():
    unadj = panel(ticker="UNADJ.KL", adj_factor=1.0)
    keep, _ = repair(unadj, verbose=False)
    drop, rep = repair(unadj, quarantine_unadjusted=True, verbose=False)
    assert "UNADJ.KL" in set(keep["ticker"])
    assert "UNADJ.KL" not in set(drop["ticker"])
    assert "UNADJ.KL" in rep.quarantined


# ------------------------------------------------- record-driven split repair
def test_a_recorded_split_already_applied_is_left_alone():
    """The common case. 292 of 315 recorded actions were already in the data.

    Repairing one twice would halve the history a second time.
    """
    p = panel()                                    # smooth, no step anywhere
    step = p["date"].iloc[150]
    loc = locate_recorded_splits(p, recorded(date=step, value=2.0))
    assert len(loc) == 1
    assert not bool(loc["needs_repair"].iloc[0])


def test_a_recorded_split_that_was_missed_is_located_at_the_price_step():
    p = with_unadjusted_split(at=150, ratio=2.0)
    step = p["date"].iloc[150]
    # Record it five weeks late, as Yahoo did for VITROX.
    loc = locate_recorded_splits(p, recorded(date=step + pd.Timedelta(days=35)))
    assert bool(loc["needs_repair"].iloc[0])
    assert loc["date"].iloc[0] == step, "must land on the price step, not the record date"


def test_record_driven_repair_catches_ratios_the_heuristic_cannot_see():
    """A 1.2 bonus issue is a -16.7% step: under SPLIT_MIN_MOVE entirely.

    detect_splits will never look at it. The record makes it repairable.
    """
    p = with_unadjusted_split(at=150, ratio=1.2)
    step = p["date"].iloc[150]
    assert detect_splits(p).empty, "fixture should be invisible to the heuristic"
    loc = locate_recorded_splits(p, recorded(date=step, value=1.2))
    assert bool(loc["needs_repair"].iloc[0])


def test_a_trivial_bonus_issue_is_not_repaired():
    """value 1.0142857 is a 1-for-70 bonus: a 1.4% step.

    Too small to locate reliably and not worth rewriting history for.
    """
    p = panel()
    step = p["date"].iloc[150]
    loc = locate_recorded_splits(p, recorded(date=step, value=1.0142857))
    assert not bool(loc["needs_repair"].iloc[0])


def test_repair_prefers_the_record_over_the_pattern():
    """With a record supplied, the heuristic must not rewrite anything.

    A clean 2:1 pattern with no action on record stays put, and the ticker
    that DOES have one on record gets fixed.
    """
    rec = with_unadjusted_split(at=150, ratio=2.0)                 # AAA.KL
    ghost = with_unadjusted_split(at=150, ratio=2.0).assign(ticker="GHOST.KL")
    step = rec["date"].iloc[150]
    out, rep = repair(pd.concat([rec, ghost], ignore_index=True),
                      known_splits=recorded(date=step), verbose=False)

    assert set(rep.splits_fixed["ticker"]) == {"AAA.KL"}
    # GHOST's step survives untouched -- it is reported, not rewritten.
    g = out[out["ticker"] == "GHOST.KL"].sort_values("date")
    assert g["close"].iloc[151] / g["close"].iloc[149] < 0.6


def test_a_corporate_action_recorded_twice_is_applied_only_once():
    """Found on real data: 8567 and 9318 had splits applied twice.

    The same event reached data/splits.csv with two float spellings --
    "0.0333333333333333" and "0.03333333333333333" -- so drop_duplicates at
    write time kept both rows, and the repair layer divided the pre-split
    history by the ratio twice. Those ratios were near 1.03, so the damage
    was a few percent. The identical mechanism on a 2-for-1 halves ten years
    of history twice, and nothing downstream would notice.
    """
    p = with_unadjusted_split(at=150, ratio=2.0)
    step = p["date"].iloc[150]
    # The same split, twice, at float spellings that are not textually equal.
    twice = pd.DataFrame({
        "date": [step, step], "ticker": ["AAA.KL", "AAA.KL"],
        "value": [2.0, 2.0000000001],
    })
    loc = locate_recorded_splits(p, twice)
    repairs = loc[loc["needs_repair"]]
    assert len(repairs) == 1, f"emitted {len(repairs)} repairs for one split"

    once = locate_recorded_splits(p, recorded(date=step, value=2.0))
    out_twice, _ = repair(p, known_splits=twice, verbose=False)
    out_once, _ = repair(p, known_splits=recorded(date=step), verbose=False)
    pd.testing.assert_frame_equal(
        out_twice.reset_index(drop=True), out_once.reset_index(drop=True))


def test_genuinely_different_actions_on_one_day_are_both_kept():
    """Dedup must key on the RATIO too. A bonus issue and a consolidation can
    legitimately land on the same date, and collapsing them would silently
    drop a real corporate action."""
    p = with_unadjusted_split(at=150, ratio=2.0)
    step = p["date"].iloc[150]
    two = pd.DataFrame({"date": [step, step], "ticker": ["AAA.KL", "AAA.KL"],
                        "value": [2.0, 1.1]})
    assert len(locate_recorded_splits(p, two)) == 2
