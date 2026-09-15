"""Live-ish quotes, with the delay stated rather than hidden.

## What this is, honestly

Yahoo's KLSE data is **delayed, typically by about 15 minutes**. It is not a
real-time feed and this module does not pretend otherwise: every quote
carries its own timestamp and age, and the dashboard prints them. Treating a
15-minute-old price as live is how you send a limit order at a price that
stopped existing a quarter of an hour ago.

Real-time Bursa data is a licensed product. The two honest routes:

* **moomoo's app** — real-time to account holders. Free, already on your
  screen, and the right place to read the spread before sending an order.
* **moomoo OpenAPI via OpenD** — programmatic real-time, but OpenD is a
  gateway that runs on YOUR machine and listens on localhost. A dashboard on
  Streamlit Community Cloud cannot reach it -- "localhost" there means
  Streamlit's container, not your laptop, and no setting changes that. OpenD
  works only for a dashboard you run locally. It is an architecture choice:
  deployed-and-delayed, or local-and-real-time. Not both.

## Does the strategy actually need real-time?

No, and it is worth being clear about that before anyone builds a tick feed.
The rule is a 60-day momentum rank held for ~21 days; the decision is made
once, at the open. A price that is 15 minutes old changes a 60-day ranking
by nothing at all.

Where live prices DO earn their place:

1. **Reading the spread before you send.** Do this in moomoo, not here.
2. **Watching your position during the contest** -- where you stand against
   the score that wins it. That is what this module is for.

## Bursa's sessions

    09:00-12:30  morning
    12:30-14:30  lunch break -- the market is CLOSED, quotes go stale, and
                 that is normal. Reporting it as a data fault every lunchtime
                 trains you to ignore the warning that matters.
    14:30-17:00  afternoon

So staleness is judged against the session, not against the wall clock.
"""
from __future__ import annotations

import datetime as dt
import json
import time
import urllib.request
from dataclasses import dataclass
from enum import Enum

import pandas as pd

BASE = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; bursa-research/1.0)"}

MYT = dt.timezone(dt.timedelta(hours=8))
MORNING = (dt.time(9, 0), dt.time(12, 30))
AFTERNOON = (dt.time(14, 30), dt.time(17, 0))

# Yahoo's stated delay for most exchanges. Not a guarantee -- it is why the
# age is reported per quote rather than assumed.
ASSUMED_DELAY_MIN = 15

# While a session is running, a quote older than this suggests a real problem
# (halt, feed outage) rather than ordinary delay.
STALE_DURING_SESSION_MIN = 45


class Market(str, Enum):
    OPEN = "open"
    LUNCH = "lunch break"
    CLOSED = "closed"
    WEEKEND = "weekend"


@dataclass(frozen=True)
class Quote:
    ticker: str
    price: float | None
    prev_close: float | None
    day_high: float | None
    day_low: float | None
    volume: float | None
    quoted_at: pd.Timestamp | None     # in MYT
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.price is not None

    @property
    def change_pct(self) -> float | None:
        if not self.ok or not self.prev_close:
            return None
        return self.price / self.prev_close - 1.0

    def age_minutes(self, now: pd.Timestamp | None = None) -> float | None:
        if self.quoted_at is None:
            return None
        now = now or pd.Timestamp.now(tz=MYT)
        return (now - self.quoted_at).total_seconds() / 60.0

    def staleness(self, now: pd.Timestamp | None = None) -> str:
        """Plain-language verdict that accounts for the session.

        A quote frozen at 12:30 during the lunch break is correct, not stale.
        """
        now = _as_myt(now)
        age = self.age_minutes(now)
        if age is None:
            return "no timestamp"
        state = market_status(now)
        if state is not Market.OPEN:
            return f"{state.value} — last trade {age:.0f} min ago (expected)"

        # Age WITHIN the running session. A quote from the morning close is
        # not stale at 14:31; the afternoon is a minute old. Only silence
        # since trading actually resumed counts against the name.
        opened = session_start(now)
        effective_from = max(self.quoted_at, opened) if opened else self.quoted_at
        age = (now - effective_from).total_seconds() / 60.0
        if age > STALE_DURING_SESSION_MIN:
            return (f"STALE — {age:.0f} min old during an open session. "
                    f"Halted, or the feed is down.")
        return f"delayed ~{age:.0f} min (Yahoo is not real-time)"


def session_start(now: pd.Timestamp | None = None) -> pd.Timestamp | None:
    """Start of the session currently running, or None if none is.

    Needed because staleness has to be measured from whichever is later: the
    quote, or the moment trading resumed. At 14:31 the newest trade is from
    the 12:30 morning close -- two hours old and completely normal, because
    the afternoon session is sixty seconds into its life. Measuring from the
    wall clock alone reports STALE at every session reopen, which is exactly
    the kind of daily false alarm that teaches you to ignore the real one.
    """
    now = _as_myt(now)
    if market_status(now) is not Market.OPEN:
        return None
    t = now.time()
    start = MORNING[0] if MORNING[0] <= t < MORNING[1] else AFTERNOON[0]
    return now.normalize() + pd.Timedelta(hours=start.hour,
                                          minutes=start.minute)


def _as_myt(now: pd.Timestamp | None) -> pd.Timestamp:
    now = now if now is not None else pd.Timestamp.now(tz=MYT)
    return (now.tz_localize(MYT) if now.tzinfo is None
            else now.tz_convert(MYT))


def market_status(now: pd.Timestamp | None = None) -> Market:
    """Where Bursa is in its trading day, in Malaysia time."""
    now = _as_myt(now)
    if now.weekday() >= 5:
        return Market.WEEKEND
    t = now.time()
    if MORNING[0] <= t < MORNING[1] or AFTERNOON[0] <= t < AFTERNOON[1]:
        return Market.OPEN
    if MORNING[1] <= t < AFTERNOON[0]:
        return Market.LUNCH
    return Market.CLOSED


def _symbol(code: str) -> str:
    c = str(code).strip().upper()
    return c if c.endswith(".KL") else f"{c}.KL"


def quote(code: str, timeout: int = 20) -> Quote:
    """One ticker's latest quote. Never raises; failures ride on the object."""
    sym = _symbol(code)
    try:
        url = BASE.format(symbol=sym) + "?range=1d&interval=1m"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.load(r)
        results = (payload.get("chart") or {}).get("result") or []
        if not results:
            return Quote(sym, None, None, None, None, None, None,
                         error="no result")
        m = results[0].get("meta") or {}
        ts = m.get("regularMarketTime")
        return Quote(
            ticker=sym,
            price=m.get("regularMarketPrice"),
            prev_close=m.get("chartPreviousClose") or m.get("previousClose"),
            day_high=m.get("regularMarketDayHigh"),
            day_low=m.get("regularMarketDayLow"),
            volume=m.get("regularMarketVolume"),
            quoted_at=(pd.Timestamp(ts, unit="s", tz="UTC").tz_convert(MYT)
                       if ts else None),
        )
    except Exception as e:                       # noqa: BLE001
        return Quote(sym, None, None, None, None, None, None,
                     error=f"{type(e).__name__}: {e}")


def quotes(codes: list[str], pause: float = 0.4) -> dict[str, Quote]:
    """Several tickers, sequentially and politely.

    A contest position is one name, so this is for the watchlist, not a
    universe scan. Do not point it at 865 tickers.
    """
    out = {}
    for c in codes:
        q = quote(c)
        out[q.ticker] = q
        time.sleep(pause)
    return out


# --------------------------------------------------------------------------
# Standing
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Position:
    ticker: str
    shares: int
    entry_price: float

    def value(self, q: Quote) -> float | None:
        return self.shares * q.price if q.ok else None

    def pnl_pct(self, q: Quote) -> float | None:
        if not q.ok or not self.entry_price:
            return None
        return q.price / self.entry_price - 1.0


def standing(position: Position, q: Quote, capital: float,
             target_return: float = 0.29) -> dict:
    """Where you are against the score that wins the contest.

    `target_return` defaults to +29%, the median winning score the contest
    simulator produced over real Bursa windows. It is not a forecast of your
    return -- it is the bar the field sets.
    """
    val = position.value(q)
    equity = (capital - position.shares * position.entry_price) + (val or 0.0)
    ret = equity / capital - 1.0 if capital else None
    return {
        "ticker": position.ticker,
        "price": q.price,
        "quoted_at": q.quoted_at,
        "staleness": q.staleness(),
        "market": market_status().value,
        "position_value": val,
        "equity": equity,
        "return_pct": ret,
        "target_pct": target_return,
        "gap_pct": (target_return - ret) if ret is not None else None,
    }
