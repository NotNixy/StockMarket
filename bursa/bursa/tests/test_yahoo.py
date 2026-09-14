"""Tests for the direct Yahoo fetcher.

No network. The response shapes here mirror what the live API actually
returned for Bursa tickers, including the awkward cases.
"""
import datetime as dt
import pandas as pd
import pytest

from core.yahoo import (
    COLUMNS, Fetched, _bars_from, _events_from, _to_symbol, to_panel, to_splits,
)


def ts(datestr):
    return int(dt.datetime.strptime(datestr, "%Y-%m-%d").timestamp())


def result(n=5, with_adj=True, nulls=False, splits=None, divs=None):
    close = [10.0 + i for i in range(n)]
    if nulls:
        close[2] = None                       # Yahoo emits null on halted days
    r = {
        "timestamp": [ts("2024-01-0%d" % (i + 1)) for i in range(n)],
        "indicators": {
            "quote": [{"open": [9.9] * n, "high": [10.5] * n, "low": [9.5] * n,
                       "close": close, "volume": [1000] * n}],
        },
    }
    if with_adj:
        r["indicators"]["adjclose"] = [{"adjclose": [c * 0.97 if c else None
                                                     for c in close]}]
    ev = {}
    if splits:
        ev["splits"] = {str(ts(d)): {"numerator": num, "denominator": den,
                                     "splitRatio": f"{num}:{den}"}
                        for d, num, den in splits}
    if divs:
        ev["dividends"] = {str(ts(d)): {"amount": a} for d, a in divs}
    if ev:
        r["events"] = ev
    return r


# ------------------------------------------------------------------- symbols
def test_symbol_suffix_is_added_once():
    assert _to_symbol("1155") == "1155.KL"
    assert _to_symbol("1155.KL") == "1155.KL"
    assert _to_symbol(" 0097 ") == "0097.KL"


def test_five_digit_ace_codes_survive():
    # 03065 must not be truncated to four digits.
    assert _to_symbol("03065") == "03065.KL"


# ---------------------------------------------------------------------- bars
def test_bars_have_the_agreed_schema():
    df = _bars_from(result(), "1155.KL")
    assert list(df.columns) == COLUMNS
    assert len(df) == 5
    assert (df["ticker"] == "1155.KL").all()


def test_adjusted_close_is_kept_separate_from_raw():
    df = _bars_from(result(with_adj=True), "X.KL")
    assert not (df["close"] == df["adj_close"]).all()


def test_missing_adjclose_falls_back_to_close():
    df = _bars_from(result(with_adj=False), "X.KL")
    assert (df["close"] == df["adj_close"]).all()


def test_null_closes_are_dropped_not_carried():
    """A halted day arrives as a null close. Keeping it would put a NaN into
    every rolling feature downstream."""
    df = _bars_from(result(nulls=True), "X.KL")
    assert len(df) == 4
    assert df["close"].notna().all()


def test_empty_timestamps_produce_an_empty_frame():
    assert _bars_from({"timestamp": []}, "X.KL").empty
    assert list(_bars_from({}, "X.KL").columns) == COLUMNS


# -------------------------------------------------------------------- events
def test_split_ratio_is_numerator_over_denominator():
    ev = _events_from(result(splits=[("2024-06-10", 2, 1)]), "0097.KL", "splits")
    assert len(ev) == 1
    assert ev["value"].iloc[0] == pytest.approx(2.0)


def test_three_for_one_split_is_read_correctly():
    ev = _events_from(result(splits=[("2020-09-03", 3, 1)]), "7113.KL", "splits")
    assert ev["value"].iloc[0] == pytest.approx(3.0)


def test_multiple_splits_come_back_in_date_order():
    ev = _events_from(result(splits=[("2024-06-10", 2, 1), ("2017-07-19", 2, 1),
                                     ("2022-01-19", 2, 1)]), "0097.KL", "splits")
    assert ev["date"].is_monotonic_increasing
    assert len(ev) == 3


def test_dividends_carry_their_amount():
    ev = _events_from(result(divs=[("2024-03-01", 0.25)]), "X.KL", "dividends")
    assert ev["value"].iloc[0] == pytest.approx(0.25)


def test_absent_events_give_an_empty_frame():
    ev = _events_from(result(), "X.KL", "splits")
    assert ev.empty
    assert list(ev.columns) == ["date", "ticker", "value"]


# ------------------------------------------------------------------ assembly
def test_to_panel_skips_failed_fetches():
    good = Fetched("A.KL", _bars_from(result(), "A.KL"))
    bad = Fetched("B.KL", pd.DataFrame(columns=COLUMNS), error="HTTP 404")
    panel = to_panel([good, bad])
    assert set(panel["ticker"]) == {"A.KL"}


def test_to_panel_raises_when_everything_failed():
    bad = Fetched("B.KL", pd.DataFrame(columns=COLUMNS), error="HTTP 404")
    with pytest.raises(ValueError, match="no successful"):
        to_panel([bad])


def test_to_splits_gathers_across_tickers():
    a = Fetched("A.KL", _bars_from(result(), "A.KL"),
                splits=_events_from(result(splits=[("2024-06-10", 2, 1)]),
                                    "A.KL", "splits"))
    b = Fetched("B.KL", _bars_from(result(), "B.KL"),
                splits=_events_from(result(splits=[("2020-09-03", 3, 1)]),
                                    "B.KL", "splits"))
    s = to_splits([a, b])
    assert len(s) == 2
    assert set(s["ticker"]) == {"A.KL", "B.KL"}


def test_to_splits_is_empty_when_no_ticker_split():
    a = Fetched("A.KL", _bars_from(result(), "A.KL"))
    assert to_splits([a]).empty


def test_ok_is_false_for_an_empty_result():
    assert not Fetched("X.KL", pd.DataFrame(columns=COLUMNS)).ok
    assert Fetched("X.KL", _bars_from(result(), "X.KL")).ok
