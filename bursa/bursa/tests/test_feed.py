"""Tests for the pluggable quote feed.

The moomoo backend cannot be tested against a real gateway here -- there is
no OpenD and no account. So what IS tested is everything that does not need
one: the symbol mapping, the fallback behaviour, and above all that a delayed
feed never claims to be live. That last one is the property worth protecting;
a wrong price you know is 15 minutes old is useful, and the same price
believed to be live is dangerous.
"""
import socket

import pandas as pd
import pytest

from core.feed import (
    MoomooFeed, YahooFeed, bare_code, get_feed, opend_listening, to_moomoo,
    to_yahoo,
)

# A port nothing is listening on. Port 1 is privileged and never a gateway.
DEAD_PORT = 1


# -------------------------------------------------------------- symbol shapes
@pytest.mark.parametrize("given", ["0270", "0270.KL", "MY.0270", " 0270.kl "])
def test_every_spelling_reduces_to_the_bare_code(given):
    assert bare_code(given) == "0270"


def test_each_vendor_gets_the_shape_it_expects():
    """Yahoo wants 0270.KL, moomoo wants MY.0270. Neither convention may leak
    into the rest of the codebase."""
    assert to_yahoo("MY.0270") == "0270.KL"
    assert to_moomoo("0270.KL") == "MY.0270"


def test_mapping_is_idempotent():
    assert to_yahoo(to_yahoo("0270")) == "0270.KL"
    assert to_moomoo(to_moomoo("0270")) == "MY.0270"


# ------------------------------------------------------------------- fallback
def test_a_dead_port_is_detected_quickly_and_without_raising():
    assert opend_listening("127.0.0.1", DEAD_PORT) is False


def test_without_opend_the_feed_falls_back_to_yahoo():
    feed = get_feed(port=DEAD_PORT)
    assert isinstance(feed, YahooFeed)


def test_the_fallback_feed_never_claims_to_be_realtime():
    """THE property. Everything else here is plumbing.

    A delayed feed that reports realtime=True would let the dashboard print a
    quarter-hour-old price as live, which is how you send a limit order into
    a market that has already moved.
    """
    feed = get_feed(port=DEAD_PORT)
    assert feed.realtime is False
    assert "delay" in feed.describe().lower()


def test_prefer_realtime_false_does_not_even_probe():
    assert isinstance(get_feed(prefer_realtime=False), YahooFeed)


# -------------------------------------------------------------------- moomoo
def test_moomoo_without_a_gateway_returns_an_actionable_error():
    """It must not raise into the caller, and the message must say what to do."""
    q = MoomooFeed(port=DEAD_PORT).quote("0270")
    assert not q.ok
    assert q.error
    assert "OpenD" in q.error
    # The ticker stays in the app's own spelling even on failure, so a failed
    # quote can still be matched to the position it belongs to.
    assert q.ticker == "0270.KL"


def test_a_moomoo_failure_still_carries_the_right_ticker():
    assert MoomooFeed(port=DEAD_PORT).quote("MY.5317").ticker == "5317.KL"


class _FakeCtx:
    """Stands in for OpenQuoteContext. Shaped from moomoo's documented
    contract: (ret, DataFrame) on success, (non-zero, str) on failure."""

    def __init__(self, ret=0, data=None):
        self._ret, self._data = ret, data
        self.closed = False

    def get_market_snapshot(self, codes):
        return self._ret, self._data

    def close(self):
        self.closed = True


def test_a_successful_snapshot_is_parsed_into_a_quote():
    df = pd.DataFrame([{
        "code": "MY.0270", "last_price": 1.71, "prev_close_price": 1.66,
        "high_price": 1.73, "low_price": 1.66, "volume": 23_564_600,
        "update_time": "2026-09-15 12:29:39",
    }])
    feed = MoomooFeed(port=DEAD_PORT)
    feed._ctx = _FakeCtx(0, df)
    q = feed.quote("0270")
    assert q.ok
    assert q.price == pytest.approx(1.71)
    assert q.change_pct == pytest.approx(0.0301, abs=1e-4)
    assert q.quoted_at.hour == 12 and q.quoted_at.minute == 29


def test_a_nonzero_return_code_is_an_error_even_with_data():
    feed = MoomooFeed(port=DEAD_PORT)
    feed._ctx = _FakeCtx(-1, "quota exceeded")
    q = feed.quote("0270")
    assert not q.ok
    assert "quota exceeded" in q.error


def test_an_empty_frame_is_a_failure_not_a_blank_quote():
    """ret==0 with no rows is moomoo saying it found nothing. Reading that as
    a valid quote would put None into a position valuation."""
    feed = MoomooFeed(port=DEAD_PORT)
    feed._ctx = _FakeCtx(0, pd.DataFrame())
    assert not feed.quote("0270").ok


def test_missing_or_junk_fields_do_not_raise():
    df = pd.DataFrame([{"code": "MY.0270", "last_price": None,
                        "prev_close_price": "n/a"}])
    feed = MoomooFeed(port=DEAD_PORT)
    feed._ctx = _FakeCtx(0, df)
    q = feed.quote("0270")
    assert q.price is None
    assert q.prev_close is None


def test_closing_releases_the_context():
    feed = MoomooFeed(port=DEAD_PORT)
    ctx = _FakeCtx()
    feed._ctx = ctx
    feed.close()
    assert ctx.closed
    assert feed._ctx is None
