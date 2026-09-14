"""Tests for the two-source comparison.

The comparison logic is pure, so all of this runs offline. The broker
connection in scripts/compare_moomoo.py is deliberately thin and untested --
there is no OpenD here to test against.
"""
import numpy as np
import pandas as pd
import pytest

from core.compare import (
    ADJ_TOL, MATERIAL_GAP, compare_panels, compare_ticker, report,
    splits_missing_from,
)

DATES = pd.date_range("2024-01-01", periods=100, freq="B")


def src(ticker="1155.KL", close=10.0, adj_factor=0.97, n=100, offset=0.0):
    # Generate the FULL path then truncate, so a shorter source carries the
    # same prices on the dates it does have. Building linspace over n points
    # instead would give genuinely different prices on shared dates -- which
    # the comparison would correctly flag, and the test would be wrong.
    px = (np.linspace(close, close * 1.2, 100) + offset)[:n]
    return pd.DataFrame({"date": DATES[:n], "ticker": ticker,
                         "close": px, "adj_close": px * adj_factor})


# ------------------------------------------------------------------ agreement
def test_identical_sources_agree():
    c = compare_ticker(src(), src(), "1155.KL")
    assert c.clean and not c.material
    assert c.n_common == 100
    assert c.max_close_diff == pytest.approx(0.0)


def test_tiny_rounding_differences_are_tolerated():
    c = compare_ticker(src(), src(offset=0.001), "1155.KL")
    assert c.clean, "0.01% difference should not register as a mismatch"


# --------------------------------------------------------------- disagreement
def test_a_material_price_divergence_is_flagged():
    c = compare_ticker(src(close=10.0), src(close=12.0), "1155.KL")
    assert not c.clean and c.material
    assert c.max_close_diff > MATERIAL_GAP


def test_an_adjustment_convention_difference_is_a_warning_not_material():
    """Yahoo back-adjusts, moomoo's QFQ forward-adjusts. A small systematic
    gap in adj_close is expected and must not be reported as an error."""
    a = src(adj_factor=0.97)
    b = src(adj_factor=0.96)          # ~1% apart, inside ADJ_TOL
    c = compare_ticker(a, b, "1155.KL")
    assert c.max_adj_diff < ADJ_TOL
    assert not c.material


def test_an_unapplied_split_shows_as_a_material_adj_divergence():
    """The VITROX case seen from two sources: one halved the series, the
    other did not."""
    a = src()
    b = src()
    b.loc[b.index >= 50, "adj_close"] /= 2.0
    c = compare_ticker(a, b, "1155.KL")
    assert c.material
    assert len(c.adj_mismatches) == 50


# ------------------------------------------------------------------ alignment
def test_dates_in_only_one_source_are_counted_not_errors():
    a = src(n=100)
    b = src(n=80)                     # shorter history
    c = compare_ticker(a, b, "1155.KL")
    assert c.n_common == 80
    assert c.n_only_a == 20
    assert c.clean, "differing history depth is not a data error"


def test_no_overlap_gives_an_empty_comparison():
    a = src(n=10)
    b = a.copy(); b["date"] = b["date"] + pd.Timedelta(days=500)
    c = compare_ticker(a, b, "1155.KL")
    assert c.n_common == 0


# --------------------------------------------------------------------- panels
def test_compare_panels_covers_only_shared_tickers():
    a = pd.concat([src("A.KL"), src("B.KL")], ignore_index=True)
    b = pd.concat([src("B.KL"), src("C.KL")], ignore_index=True)
    out = compare_panels(a, b)
    assert [c.ticker for c in out] == ["B.KL"]


def test_report_builds_a_table():
    a = pd.concat([src("A.KL"), src("B.KL")], ignore_index=True)
    t = report(compare_panels(a, a), verbose=False)
    assert list(t["ticker"]) == ["A.KL", "B.KL"]
    assert not t["material"].any()


# --------------------------------------------------------------------- splits
def test_a_split_left_unapplied_is_detected():
    """Split on record, and the adjusted series steps by the ratio anyway."""
    p = src()
    p.loc[p.index >= 50, "adj_close"] /= 2.0
    splits = pd.DataFrame({"ticker": ["1155.KL"], "date": [DATES[50]],
                           "value": [2.0]})
    out = splits_missing_from(p, splits)
    assert len(out) == 1
    assert not bool(out["applied"].iloc[0])


def test_a_split_properly_applied_is_reported_as_applied():
    # Smooth adjusted series across the split date.
    p = src()
    splits = pd.DataFrame({"ticker": ["1155.KL"], "date": [DATES[50]],
                           "value": [2.0]})
    out = splits_missing_from(p, splits)
    assert bool(out["applied"].iloc[0])


def test_no_splits_gives_an_empty_frame():
    out = splits_missing_from(src(), pd.DataFrame(
        columns=["ticker", "date", "value"]))
    assert out.empty
