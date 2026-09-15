"""Where quotes come from. Two backends, one interface, no pretending.

    from core.feed import get_feed
    feed = get_feed()                 # real-time if available, else delayed
    print(feed.describe())            # says WHICH, and whether it is live
    q = feed.quote("0270")

## The two backends

**YahooFeed** — works anywhere, including a deployed dashboard. Delayed by
roughly 15 minutes. `realtime = False`.

**MoomooFeed** — real-time, via moomoo's OpenD gateway. OpenD runs on YOUR
machine and listens on 127.0.0.1:11111, so this backend only exists for a
dashboard you run locally. A Streamlit Cloud container's localhost is its
own container. `realtime = True`.

`get_feed()` tries moomoo, falls back to Yahoo, and tells you which it got.
The fallback is silent in the sense that nothing crashes, and loud in the
sense that `describe()` and the dashboard both state the delay. Falling back
without saying so is the failure mode worth avoiding here: a delayed price
presented as live is how you send a limit order against a market that moved
on a quarter of an hour ago.

## Status of the moomoo backend

**This code has never run against a live OpenD gateway.** There is no OpenD
and no moomoo account in the environment it was written in. It is built from
moomoo's published API docs -- `get_market_snapshot(code_list)` returning
`(ret, data)`, codes shaped `MY.<code>`, fields `last_price`,
`prev_close_price`, `high_price`, `low_price`, `volume`, `update_time` -- and
those docs could be stale or the Malaysia specifics could differ.

So it is written to fail loudly and fall back cleanly rather than to be
trusted. Run `python -m scripts.check_feed` on the machine with OpenD before
relying on it; that script is the verification step this file cannot perform
for itself.

## Symbol shapes

Yahoo wants `0270.KL`. moomoo wants `MY.0270`. Both are derived from the bare
Bursa code, so the rest of the codebase keeps using bare codes or `.KL` and
neither convention leaks outward.
"""
from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from core.quotes import MYT, Quote
from core.quotes import quote as _yahoo_quote

OPEND_HOST = "127.0.0.1"
OPEND_PORT = 11111
# A gateway that is not listening should be discovered in milliseconds, not
# after a 30-second SDK timeout while a dashboard sits blank.
PROBE_TIMEOUT_S = 0.4


def bare_code(code: str) -> str:
    """'0270.KL' or 'MY.0270' or '0270' -> '0270'."""
    c = str(code).strip().upper()
    if c.endswith(".KL"):
        c = c[:-3]
    if c.startswith("MY."):
        c = c[3:]
    return c


def to_moomoo(code: str) -> str:
    return f"MY.{bare_code(code)}"


def to_yahoo(code: str) -> str:
    return f"{bare_code(code)}.KL"


class Feed(Protocol):
    name: str
    realtime: bool

    def quote(self, code: str) -> Quote: ...
    def describe(self) -> str: ...


@dataclass
class YahooFeed:
    """Delayed quotes from Yahoo. Available everywhere."""
    name: str = "Yahoo (delayed)"
    realtime: bool = False

    def quote(self, code: str) -> Quote:
        return _yahoo_quote(bare_code(code))

    def quotes(self, codes: list[str]) -> dict[str, Quote]:
        return {q.ticker: q for q in (self.quote(c) for c in codes)}

    def describe(self) -> str:
        return ("Yahoo Finance, delayed ~15 minutes. Fine for tracking a "
                "position; do not price an order off it.")


def opend_listening(host: str = OPEND_HOST, port: int = OPEND_PORT,
                    timeout: float = PROBE_TIMEOUT_S) -> bool:
    """Is anything accepting connections on OpenD's port?

    A cheap TCP probe rather than an SDK call, so a missing gateway costs
    milliseconds instead of a long timeout. It proves something is listening,
    not that it is OpenD or that it is logged in -- `MoomooFeed.connect()`
    establishes that.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@dataclass
class MoomooFeed:
    """Real-time quotes via a locally running OpenD gateway.

    UNVERIFIED against a live gateway -- see the module docstring.
    """
    host: str = OPEND_HOST
    port: int = OPEND_PORT
    name: str = "moomoo OpenD (real-time)"
    realtime: bool = True
    _ctx: object | None = None

    def connect(self) -> None:
        """Open a quote context, or raise with something actionable."""
        if self._ctx is not None:
            return
        if not opend_listening(self.host, self.port):
            raise ConnectionError(
                f"Nothing is listening on {self.host}:{self.port}. Start the "
                f"OpenD gateway and log in, then retry. If this is a deployed "
                f"dashboard, OpenD cannot be reached at all -- it runs on your "
                f"machine, not on the server.")
        try:
            # moomoo renamed from futu; both SDKs expose the same class.
            try:
                from moomoo import OpenQuoteContext          # type: ignore
            except ImportError:
                from futu import OpenQuoteContext            # type: ignore
        except ImportError as e:
            raise ImportError(
                "moomoo's Python SDK is not installed. `pip install "
                "moomoo-api` (or `futu-api`). It is deliberately NOT in "
                "requirements.txt: the deployed dashboard cannot use it, and "
                "every package listed is one more that can fail to build."
            ) from e
        self._ctx = OpenQuoteContext(host=self.host, port=self.port)

    def close(self) -> None:
        if self._ctx is not None:
            try:
                self._ctx.close()
            finally:
                self._ctx = None

    def quote(self, code: str) -> Quote:
        sym = to_yahoo(code)          # keep one ticker spelling across the app
        try:
            self.connect()
            ret, data = self._ctx.get_market_snapshot([to_moomoo(code)])
            # moomoo returns (0, DataFrame) on success, (non-zero, str) on
            # failure. Checking the code rather than the type, because a
            # DataFrame with zero rows is also a failure.
            if ret != 0:
                return Quote(sym, None, None, None, None, None, None,
                             error=f"moomoo error {ret}: {data}")
            if not isinstance(data, pd.DataFrame) or data.empty:
                return Quote(sym, None, None, None, None, None, None,
                             error=f"moomoo returned no rows for {code}")
            r = data.iloc[0]
            ts = r.get("update_time")
            return Quote(
                ticker=sym,
                price=_f(r.get("last_price")),
                prev_close=_f(r.get("prev_close_price")),
                day_high=_f(r.get("high_price")),
                day_low=_f(r.get("low_price")),
                volume=_f(r.get("volume")),
                quoted_at=(pd.Timestamp(ts).tz_localize(MYT) if ts else None),
            )
        except Exception as e:                       # noqa: BLE001
            return Quote(sym, None, None, None, None, None, None,
                         error=f"{type(e).__name__}: {e}")

    def describe(self) -> str:
        return (f"moomoo OpenD at {self.host}:{self.port}, real-time. "
                f"Requires OpenD running and logged in on this machine.")


def _f(v) -> float | None:
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def get_feed(prefer_realtime: bool = True,
             host: str = OPEND_HOST, port: int = OPEND_PORT) -> Feed:
    """The best feed available here, without ever silently claiming live data.

    Order: moomoo if OpenD answers, Yahoo otherwise. The caller is expected to
    surface `feed.realtime` -- that flag is the whole reason this indirection
    exists rather than a hardcoded import.
    """
    if prefer_realtime and opend_listening(host, port):
        feed = MoomooFeed(host=host, port=port)
        try:
            feed.connect()
            return feed
        except Exception:                            # noqa: BLE001
            feed.close()
    return YahooFeed()
