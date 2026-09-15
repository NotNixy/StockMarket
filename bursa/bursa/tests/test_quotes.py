"""Tests for live quotes and market-session logic.

No network. Every test builds its own Quote, because a test that depends on
Bursa being open passes or fails according to the time of day, which makes it
worse than no test.
"""
import datetime as dt

import pandas as pd
import pytest

from core.quotes import (
    MYT, Market, Position, Quote, market_status, session_start,
    standing,
)


def at(day: str, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {hhmm}", tz=MYT)


# ------------------------------------------------------------------ sessions
# 2026-09-15 is a Tuesday; 2026-09-19 a Saturday.
@pytest.mark.parametrize("hhmm,expected", [
    ("08:59", Market.CLOSED),
    ("09:00", Market.OPEN),
    ("12:29", Market.OPEN),
    ("12:30", Market.LUNCH),      # the boundary that caused the false alarm
    ("14:29", Market.LUNCH),
    ("14:30", Market.OPEN),
    ("16:59", Market.OPEN),
    ("17:00", Market.CLOSED),
    ("23:00", Market.CLOSED),
])
def test_bursa_sessions(hhmm, expected):
    assert market_status(at("2026-09-15", hhmm)) is expected


def test_the_weekend_is_not_merely_closed():
    assert market_status(at("2026-09-19", "11:00")) is Market.WEEKEND


def test_a_naive_timestamp_is_read_as_malaysia_time():
    """Bursa hours are Malaysian. A naive timestamp interpreted as UTC would
    put the whole trading day in the wrong session."""
    assert market_status(pd.Timestamp("2026-09-15 10:00")) is Market.OPEN


# ----------------------------------------------------------------- staleness
def q(price=1.71, prev=1.66, minutes_old=5, day="2026-09-15", now="10:00"):
    t = at(day, now) - pd.Timedelta(minutes=minutes_old)
    return Quote("0270.KL", price, prev, 1.73, 1.66, 1e6, t)


def test_a_lunch_break_quote_is_not_reported_as_stale():
    """The bug this module was written to avoid.

    At 14:00 the newest trade is from 12:30 -- 90 minutes old and completely
    correct. Flagging it daily would train you to ignore the warning on the
    day it actually means something.
    """
    verdict = q(minutes_old=90, now="14:00").staleness(at("2026-09-15", "14:00"))
    assert "STALE" not in verdict
    assert "lunch" in verdict.lower()


def test_a_frozen_quote_during_an_open_session_is_stale():
    verdict = q(minutes_old=90, now="11:00").staleness(at("2026-09-15", "11:00"))
    assert "STALE" in verdict


def test_a_normal_delay_is_described_as_delayed_not_live():
    """Yahoo KLSE is ~15 minutes behind. The wording must never imply live."""
    verdict = q(minutes_old=12, now="11:00").staleness(at("2026-09-15", "11:00"))
    assert "STALE" not in verdict
    assert "not real-time" in verdict


def test_an_overnight_quote_is_expected_not_alarming():
    verdict = q(minutes_old=16 * 60, now="08:00").staleness(at("2026-09-15", "08:00"))
    assert "STALE" not in verdict


# -------------------------------------------------------------------- quotes
def test_change_percent_uses_the_previous_close():
    assert q(price=1.71, prev=1.66).change_pct == pytest.approx(0.0301, abs=1e-4)


def test_a_failed_quote_carries_its_error_and_is_not_ok():
    bad = Quote("X.KL", None, None, None, None, None, None, error="HTTP 404")
    assert not bad.ok
    assert bad.change_pct is None
    assert bad.age_minutes() is None


def test_a_quote_with_no_previous_close_does_not_divide_by_zero():
    assert Quote("X.KL", 1.0, 0.0, None, None, None, None).change_pct is None


# ------------------------------------------------------------------ standing
def test_standing_measures_the_whole_account_not_just_the_position():
    """Cash left over after board-lot rounding is still your money.

    RM 9,960 of stock plus RM 40 cash on RM 10,000 is a 0% return when the
    stock is flat -- not 0% of 9,960 reported as the account.
    """
    pos = Position("0270.KL", shares=6000, entry_price=1.66)
    s = standing(pos, q(price=1.66, prev=1.66), capital=10_000.0)
    assert s["equity"] == pytest.approx(10_000.0)
    assert s["return_pct"] == pytest.approx(0.0)


def test_the_gap_to_the_winning_score_shrinks_as_you_gain():
    pos = Position("0270.KL", shares=6000, entry_price=1.66)
    flat = standing(pos, q(price=1.66), capital=10_000.0)
    up = standing(pos, q(price=1.90), capital=10_000.0)
    assert up["return_pct"] > flat["return_pct"]
    assert up["gap_pct"] < flat["gap_pct"]
    # +29% is the bar the field sets, not a forecast -- it must not move.
    assert flat["target_pct"] == up["target_pct"] == 0.29


def test_standing_survives_a_broken_quote():
    """A feed outage must not take the dashboard down with it."""
    pos = Position("0270.KL", shares=6000, entry_price=1.66)
    bad = Quote("0270.KL", None, None, None, None, None, None, error="timeout")
    s = standing(pos, bad, capital=10_000.0)
    assert s["position_value"] is None
    assert s["price"] is None


# ------------------------------------------------- session-relative staleness
def test_a_quote_from_the_morning_close_is_not_stale_just_after_reopen():
    """Caught by running the feed self-check at 14:31.

    The newest trade was the 12:29 morning close -- two hours old by the wall
    clock, and flagged STALE one minute into the afternoon session. Nothing
    was wrong: trading had barely resumed. A warning that fires at every
    reopen is a warning you learn to ignore.
    """
    reopen = at("2026-09-15", "14:31")
    morning_close = Quote("0270.KL", 1.71, 1.66, 1.73, 1.66, 1e6,
                          at("2026-09-15", "12:29"))
    verdict = morning_close.staleness(reopen)
    assert "STALE" not in verdict, verdict


def test_the_same_quote_does_go_stale_once_the_session_has_run_a_while():
    """The other half: silence an hour INTO the session is a real signal."""
    later = at("2026-09-15", "15:45")
    morning_close = Quote("0270.KL", 1.71, 1.66, 1.73, 1.66, 1e6,
                          at("2026-09-15", "12:29"))
    assert "STALE" in morning_close.staleness(later)


def test_session_start_tracks_whichever_session_is_running():
    assert session_start(at("2026-09-15", "10:00")).hour == 9
    assert session_start(at("2026-09-15", "15:00")).hour == 14
    assert session_start(at("2026-09-15", "13:00")) is None   # lunch
    assert session_start(at("2026-09-19", "11:00")) is None   # weekend
