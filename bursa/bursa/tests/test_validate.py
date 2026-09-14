"""Tests for the integrity checks.

Each test breaks the panel in one specific way that has actually occurred in
Bursa data, and asserts the corresponding check notices.
"""
import numpy as np
import pandas as pd
import pytest

from core.validate import (
    ALL_CHECKS, assert_clean, check_adjustment_applied, check_calendar_gaps,
    check_extreme_moves, check_history_depth, check_monotonic_dates,
    check_no_duplicates, check_no_missing, check_ohlc_brackets,
    check_positive_prices, check_zero_volume, validate,
)


def clean_panel(n=200, tickers=("AAA.KL", "BBB.KL")):
    """A panel that should pass everything."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    rows = []
    for i, t in enumerate(tickers):
        close = np.linspace(2.0, 3.0, n) + i
        rows.append(pd.DataFrame({
            "date": dates, "ticker": t,
            "open": close * 0.995, "high": close * 1.02,
            "low": close * 0.98, "close": close,
            "adj_close": close * 0.97,        # differs: adjustment applied
            "volume": 1_000_000,
        }))
    return pd.concat(rows, ignore_index=True)


def test_a_clean_panel_passes_every_check():
    results = validate(clean_panel(), verbose=False)
    failed = [r.name for r in results if not r.passed]
    assert failed == [], f"unexpected failures: {failed}"


# ------------------------------------------------------------------- brackets
def test_high_below_close_is_caught():
    p = clean_panel()
    p.loc[10, "high"] = p.loc[10, "close"] - 0.5
    r = check_ohlc_brackets(p)
    assert not r.passed and r.n_bad == 1


def test_high_below_low_is_caught():
    p = clean_panel()
    p.loc[10, ["high", "low"]] = [1.0, 5.0]
    assert not check_ohlc_brackets(p).passed


# --------------------------------------------------------------------- prices
def test_zero_price_is_caught():
    p = clean_panel()
    p.loc[7, "close"] = 0.0
    r = check_positive_prices(p)
    assert not r.passed and r.n_bad == 1


# ----------------------------------------------------------------- duplicates
def test_duplicate_bars_are_caught():
    p = clean_panel()
    p = pd.concat([p, p.iloc[[5]]], ignore_index=True)
    r = check_no_duplicates(p)
    assert not r.passed and r.n_bad == 2      # both members of the pair


def test_missing_values_are_caught():
    p = clean_panel()
    p.loc[3, "adj_close"] = np.nan
    assert not check_no_missing(p).passed


# --------------------------------------------------------------------- splits
def test_an_unadjusted_halving_is_flagged_as_a_split_signature():
    p = clean_panel()
    mask = (p["ticker"] == "AAA.KL") & (p.index >= 100)
    p.loc[mask, "adj_close"] *= 0.5           # abrupt -50%, never recovers
    r = check_extreme_moves(p)
    assert not r.passed
    assert "split ratio" in r.message
    assert bool(r.sample["near_split_ratio"].any())


def test_an_ordinary_large_move_is_flagged_but_not_called_a_split():
    p = clean_panel()
    p.loc[100, "adj_close"] *= 1.42           # +42%, not a split ratio
    r = check_extreme_moves(p)
    assert not r.passed
    assert "split ratio" not in r.message


def test_a_quiet_panel_has_no_extreme_moves():
    assert check_extreme_moves(clean_panel()).passed


# ----------------------------------------------------------------- adjustment
def test_adjustment_never_applied_is_caught():
    # The silent one: every ex-dividend date becomes a fake negative return.
    p = clean_panel()
    p["adj_close"] = p["close"]
    r = check_adjustment_applied(p)
    assert not r.passed and r.n_bad == 2      # both tickers


def test_adjustment_applied_to_one_ticker_only_is_caught():
    p = clean_panel()
    p.loc[p["ticker"] == "BBB.KL", "adj_close"] = p.loc[
        p["ticker"] == "BBB.KL", "close"]
    r = check_adjustment_applied(p)
    assert not r.passed and r.n_bad == 1


# ------------------------------------------------------------------- calendar
def test_a_long_suspension_gap_is_caught():
    p = clean_panel()
    drop = (p["ticker"] == "AAA.KL") & p["date"].between("2024-03-01",
                                                          "2024-04-15")
    p = p[~drop].reset_index(drop=True)
    r = check_calendar_gaps(p)
    assert not r.passed and r.n_bad >= 1


def test_normal_weekends_are_not_flagged():
    assert check_calendar_gaps(clean_panel()).passed


# --------------------------------------------------------------------- volume
def test_a_mostly_untraded_ticker_is_caught():
    p = clean_panel()
    mask = (p["ticker"] == "BBB.KL") & (p.index % 2 == 0)
    p.loc[mask, "volume"] = 0
    r = check_zero_volume(p)
    assert not r.passed and r.n_bad == 1


def test_occasional_halts_are_tolerated():
    p = clean_panel()
    p.loc[[5, 6], "volume"] = 0               # ~1% of bars
    assert check_zero_volume(p).passed


# -------------------------------------------------------------------- history
def test_a_short_ticker_is_caught():
    p = clean_panel(n=200)
    short = clean_panel(n=40, tickers=("CCC.KL",))
    r = check_history_depth(pd.concat([p, short], ignore_index=True))
    assert not r.passed and r.n_bad == 1


def test_out_of_order_dates_are_caught():
    p = clean_panel()
    p.loc[[10, 11], "date"] = p.loc[[11, 10], "date"].values
    assert not check_monotonic_dates(p).passed


# ---------------------------------------------------------------- assert_clean
def test_assert_clean_passes_on_good_data():
    assert_clean(clean_panel())          # must not raise


def test_assert_clean_raises_on_structural_damage():
    p = clean_panel()
    p.loc[7, "close"] = -1.0
    with pytest.raises(ValueError, match="positive prices"):
        assert_clean(p)


def test_assert_clean_tolerates_extreme_moves_by_default():
    # Real panels contain genuine 40% days; those want reading, not blocking.
    p = clean_panel()
    p.loc[100, "adj_close"] *= 1.42
    assert_clean(p)                      # must not raise


def test_every_check_is_registered():
    # Guards against adding a check and forgetting to wire it into validate().
    assert len(ALL_CHECKS) == 10
    assert len(validate(clean_panel(), verbose=False)) == 10
