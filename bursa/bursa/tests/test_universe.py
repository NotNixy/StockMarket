"""Tests for the point-in-time universe screen.

The tests that matter most are the lookahead ones: a screen that quietly uses
future information produces a backtest that cannot be trusted and does not
look broken.
"""
import numpy as np
import pandas as pd
import pytest

from core.universe import (
    ScreenConfig, add_liquidity, add_shariah_flag, compliant_on,
    load_shariah_lists, max_affordable_names, rejection_reasons, screen,
    screen_summary, universe_on,
)

DATES = pd.date_range("2024-01-01", periods=200, freq="B")


def make_panel(tickers=("AAA", "BBB"), close=2.0, volume=1_000_000, n=200):
    rows = []
    for t in tickers:
        rows.append(pd.DataFrame({
            "date": DATES[:n], "ticker": f"{t}.KL",
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "adj_close": close, "volume": volume,
        }))
    return pd.concat(rows, ignore_index=True)


def make_lists(tmp_path, releases):
    """releases: {'2024-05-30': ['AAA'], ...}"""
    rows = [{"list_date": d, "ticker": t}
            for d, ts in releases.items() for t in ts]
    p = tmp_path / "shariah.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return load_shariah_lists(p)


# ------------------------------------------------------- shariah, point in time
def test_compliance_uses_the_most_recent_release_on_or_before_the_date(tmp_path):
    lists = make_lists(tmp_path, {
        "2024-05-30": ["AAA"],
        "2024-11-28": ["AAA", "BBB"],   # BBB added at the November revision
    })
    # Before BBB was added it is not compliant, even though it is today.
    assert compliant_on(lists, "2024-08-01") == {"AAA.KL"}
    # On and after the release it is.
    assert compliant_on(lists, "2024-11-28") == {"AAA.KL", "BBB.KL"}
    assert compliant_on(lists, "2025-03-01") == {"AAA.KL", "BBB.KL"}


def test_removal_at_a_revision_is_respected(tmp_path):
    lists = make_lists(tmp_path, {
        "2024-05-30": ["AAA", "BBB"],
        "2024-11-28": ["AAA"],          # BBB dropped
    })
    assert "BBB.KL" in compliant_on(lists, "2024-06-01")
    assert "BBB.KL" not in compliant_on(lists, "2024-12-01")


def test_dates_before_the_first_release_yield_an_empty_universe(tmp_path):
    # Deliberately loud: silently using the earliest list would be lookahead.
    lists = make_lists(tmp_path, {"2024-05-30": ["AAA"]})
    assert compliant_on(lists, "2024-01-01") == set()


def test_shariah_flag_changes_over_time_within_one_panel(tmp_path):
    lists = make_lists(tmp_path, {
        "2024-01-01": ["AAA"],
        "2024-06-03": ["AAA", "BBB"],
    })
    flagged = add_shariah_flag(make_panel(), lists)
    bbb = flagged[flagged["ticker"] == "BBB.KL"].sort_values("date")
    assert not bbb[bbb["date"] < "2024-06-03"]["shariah"].any()
    assert bbb[bbb["date"] >= "2024-06-03"]["shariah"].all()


def test_missing_column_in_the_list_file_is_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"ticker": ["AAA"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="list_date"):
        load_shariah_lists(p)


# ------------------------------------------------------------------ liquidity
def test_median_traded_value_excludes_the_current_bar():
    """The critical lookahead test.

    A stock dead for its whole history that trades once, hugely, on day 150
    must NOT be eligible on day 150 -- you could not have known.
    """
    panel = make_panel(("AAA",), volume=1_000)
    spike = panel["date"] == DATES[150]
    panel.loc[spike, "volume"] = 10_000_000_000

    out = add_liquidity(panel, ScreenConfig(lookback_days=20))
    on_spike_day = out.loc[spike, "median_dtv"].iloc[0]
    day_after = out.loc[out["date"] == DATES[151], "median_dtv"].iloc[0]

    assert on_spike_day < 1e6          # spike not yet visible
    assert day_after >= on_spike_day   # it enters the window afterwards


def test_liquidity_is_computed_per_ticker():
    panel = pd.concat([make_panel(("AAA",), volume=10_000_000),
                       make_panel(("BBB",), volume=100)], ignore_index=True)
    out = add_liquidity(panel, ScreenConfig(lookback_days=20))
    last = out.groupby("ticker").tail(1).set_index("ticker")["median_dtv"]
    assert last["AAA.KL"] > last["BBB.KL"] * 1000


# --------------------------------------------------------------------- screen
def test_screen_requires_shariah_lists_when_the_filter_is_on():
    with pytest.raises(ValueError, match="require_shariah"):
        screen(make_panel(), ScreenConfig(require_shariah=True))


def test_illiquid_names_are_excluded(tmp_path):
    lists = make_lists(tmp_path, {"2023-01-01": ["AAA", "BBB"]})
    panel = pd.concat([make_panel(("AAA",), volume=10_000_000),
                       make_panel(("BBB",), volume=10)], ignore_index=True)
    out = screen(panel, ScreenConfig(lookback_days=20, min_history=30), lists)
    assert universe_on(out, DATES[100]) == ["AAA.KL"]


def test_price_bounds_are_applied(tmp_path):
    lists = make_lists(tmp_path, {"2023-01-01": ["CHEAP", "DEAR", "OK"]})
    panel = pd.concat([
        make_panel(("CHEAP",), close=0.05),
        make_panel(("DEAR",), close=500.0),
        make_panel(("OK",), close=3.0),
    ], ignore_index=True)
    out = screen(panel, ScreenConfig(lookback_days=20, min_history=30), lists)
    assert universe_on(out, DATES[100]) == ["OK.KL"]


def test_unseasoned_tickers_are_excluded(tmp_path):
    lists = make_lists(tmp_path, {"2023-01-01": ["AAA"]})
    out = screen(make_panel(("AAA",)), ScreenConfig(min_history=150,
                                                    lookback_days=20), lists)
    assert universe_on(out, DATES[100]) == []      # only 100 bars of history
    assert universe_on(out, DATES[180]) == ["AAA.KL"]


def test_rejection_reasons_explain_an_empty_universe(tmp_path):
    lists = make_lists(tmp_path, {"2023-01-01": ["AAA"]})
    out = screen(make_panel(("AAA",), volume=1), ScreenConfig(
        lookback_days=20, min_history=30), lists)
    why = rejection_reasons(out, DATES[100])
    assert len(why) == 1
    assert not why["liquid"].iloc[0]        # names the failing condition
    assert why["shariah"].iloc[0]


def test_screen_summary_tracks_universe_size(tmp_path):
    lists = make_lists(tmp_path, {"2023-01-01": ["AAA", "BBB"]})
    out = screen(make_panel(), ScreenConfig(lookback_days=20,
                                            min_history=30), lists)
    summary = screen_summary(out)
    assert summary.loc[DATES[100], "n_eligible"] == 2
    assert summary.loc[DATES[5], "n_eligible"] == 0     # not yet seasoned


# ----------------------------------------------------------------- board lots
def test_board_lots_limit_how_many_names_fit():
    # median price RM 12 -> RM 1,200 a lot -> 8 names from RM 10,000
    assert max_affordable_names(pd.Series([10.0, 12.0, 14.0]), 10_000) == 8
    assert max_affordable_names(pd.Series([12.0]), 500) == 0
    assert max_affordable_names(pd.Series([], dtype=float), 10_000) == 0


# ------------------------------------------------------- backfill (lookahead)
def test_backfill_is_off_by_default(tmp_path):
    lists = make_lists(tmp_path, {"2024-05-30": ["AAA"]})
    assert compliant_on(lists, "2024-01-01") == set()


def test_backfill_extends_the_earliest_release_backwards(tmp_path):
    lists = make_lists(tmp_path, {"2024-05-30": ["AAA"], "2024-11-28": ["AAA", "BBB"]})
    # Before any release, backfill uses the EARLIEST list -- not the latest,
    # which would import even more future knowledge.
    assert compliant_on(lists, "2024-01-01", backfill=True) == {"AAA.KL"}


def test_backfill_does_not_change_dates_after_a_release(tmp_path):
    lists = make_lists(tmp_path, {"2024-05-30": ["AAA"], "2024-11-28": ["AAA", "BBB"]})
    for d in ("2024-06-01", "2024-12-01"):
        assert compliant_on(lists, d) == compliant_on(lists, d, backfill=True)


def test_screen_config_carries_the_backfill_flag(tmp_path):
    lists = make_lists(tmp_path, {"2025-01-01": ["AAA", "BBB"]})
    strict = screen(make_panel(), ScreenConfig(lookback_days=20, min_history=30), lists)
    loose = screen(make_panel(), ScreenConfig(lookback_days=20, min_history=30,
                                              backfill_shariah=True), lists)
    # The panel predates the release entirely.
    assert strict["eligible"].sum() == 0
    assert loose["eligible"].sum() > 0
