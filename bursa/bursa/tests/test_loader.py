"""Tests for the loader's pure logic.

Nothing here touches the network. The parts that talk to yfinance are thin;
the parts that can silently corrupt a panel -- column normalisation, dedup,
return calculation -- are what get tested.
"""
import numpy as np
import pandas as pd
import pytest

from core.loader import (
    COLUMNS, _cache_path, _normalise, add_returns, load_panel, to_yahoo,
)


# ------------------------------------------------------------ ticker mapping
def test_to_yahoo_appends_suffix_once():
    assert to_yahoo("1155") == "1155.KL"
    assert to_yahoo("1155.KL") == "1155.KL"       # idempotent
    assert to_yahoo(" maybank ") == "MAYBANK.KL"  # trims and upcases


def _yf_frame(n=5, multiindex=False, with_adj=True):
    """A frame shaped like yfinance's output."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B", name="Date")
    data = {
        "Open": np.linspace(1.00, 1.04, n),
        "High": np.linspace(1.02, 1.06, n),
        "Low": np.linspace(0.98, 1.02, n),
        "Close": np.linspace(1.01, 1.05, n),
        "Volume": np.full(n, 100_000),
    }
    if with_adj:
        data["Adj Close"] = np.linspace(1.01, 1.05, n) * 0.97
    df = pd.DataFrame(data, index=idx)
    if multiindex:
        df.columns = pd.MultiIndex.from_product([df.columns, ["X.KL"]])
    return df


# ------------------------------------------------------------- normalisation
def test_normalise_produces_the_agreed_schema():
    out = _normalise(_yf_frame(), "1155")
    assert list(out.columns) == COLUMNS
    assert (out["ticker"] == "1155.KL").all()
    assert out["date"].dt.tz is None          # tz-naive, normalised to midnight


def test_normalise_flattens_a_multiindex():
    # A single-ticker download can still come back two-level.
    out = _normalise(_yf_frame(multiindex=True), "X")
    assert list(out.columns) == COLUMNS
    assert len(out) == 5


def test_normalise_synthesises_adj_close_when_absent():
    # auto_adjust=True drops 'Adj Close'; schema must stay stable.
    out = _normalise(_yf_frame(with_adj=False), "X")
    assert (out["adj_close"] == out["close"]).all()


def test_normalise_keeps_raw_and_adjusted_apart():
    # validate.py needs both to detect an unadjusted corporate action.
    out = _normalise(_yf_frame(with_adj=True), "X")
    assert not np.allclose(out["close"], out["adj_close"])


def test_normalise_handles_empty_input():
    assert _normalise(pd.DataFrame(), "X").empty
    assert list(_normalise(None, "X").columns) == COLUMNS


def test_normalise_drops_duplicate_dates():
    df = _yf_frame(3)
    doubled = pd.concat([df, df])
    assert len(_normalise(doubled, "X")) == 3


def test_normalise_drops_rows_with_no_close():
    df = _yf_frame(5)
    df.iloc[2, df.columns.get_loc("Close")] = np.nan
    assert len(_normalise(df, "X")) == 4


# -------------------------------------------------------------- cache + panel
def _write_cache(tmp_path, ticker, n=6, start="2024-01-01"):
    idx = pd.date_range(start, periods=n, freq="B")
    df = pd.DataFrame({
        "date": idx,
        "ticker": to_yahoo(ticker),
        "open": 1.0, "high": 1.1, "low": 0.9,
        "close": np.linspace(1.0, 1.5, n),
        "adj_close": np.linspace(1.0, 1.5, n),
        "volume": 50_000,
    })[COLUMNS]
    df.to_parquet(_cache_path(ticker, tmp_path), index=False)
    return df


def test_load_panel_concatenates_cached_tickers(tmp_path):
    _write_cache(tmp_path, "AAA")
    _write_cache(tmp_path, "BBB")
    panel = load_panel(raw_dir=tmp_path)
    assert set(panel["ticker"]) == {"AAA.KL", "BBB.KL"}
    assert len(panel) == 12
    # sorted by ticker then date
    assert panel["ticker"].is_monotonic_increasing


def test_load_panel_raises_when_cache_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_panel(raw_dir=tmp_path / "nope")


def test_load_panel_ignores_tickers_with_no_cache_file(tmp_path):
    _write_cache(tmp_path, "AAA")
    panel = load_panel(["AAA", "MISSING"], raw_dir=tmp_path)
    assert set(panel["ticker"]) == {"AAA.KL"}


# ------------------------------------------------------------------- returns
def test_returns_are_computed_from_adjusted_prices():
    panel = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3, freq="B"),
        "ticker": "X.KL",
        "open": 1.0, "high": 1.0, "low": 1.0,
        "close": [10.0, 20.0, 20.0],        # raw: looks like a +100% jump
        "adj_close": [10.0, 10.0, 11.0],    # adjusted: flat, then +10%
        "volume": 1,
    })[COLUMNS]
    out = add_returns(panel)
    assert np.isnan(out["ret"].iloc[0])
    assert out["ret"].iloc[1] == pytest.approx(0.0)    # NOT +1.0
    assert out["ret"].iloc[2] == pytest.approx(0.1)


def test_returns_do_not_leak_across_tickers():
    # The first bar of each ticker must be NaN, not a return computed from
    # the previous ticker's last price.
    a = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=2, freq="B"),
                      "ticker": "A.KL", "open": 1.0, "high": 1.0, "low": 1.0,
                      "close": [1.0, 2.0], "adj_close": [1.0, 2.0], "volume": 1})
    b = a.copy(); b["ticker"] = "B.KL"; b["adj_close"] = [50.0, 55.0]
    out = add_returns(pd.concat([a, b], ignore_index=True)[COLUMNS])
    # .head(1) takes the literal first row per group; .first() would skip the
    # NaN we are specifically trying to assert on.
    assert out.groupby("ticker").head(1)["ret"].isna().all()
    # and B's second bar is its own +10%, not 55/2 from A's last price
    assert out.loc[out.ticker == "B.KL", "ret"].iloc[1] == pytest.approx(0.1)
