"""Direct Yahoo chart fetcher, with corporate actions.

Exists for two reasons discovered the hard way against real Bursa data.

**Host.** yfinance talks to `query2.finance.yahoo.com`, which is unreachable
from some networks -- it fails as a connection reset mid-handshake, which
surfaces as an EMPTY frame rather than an error, and an empty frame is
indistinguishable from a delisted stock. `query1` answers normally. This
module talks to query1 and raises on failure instead of returning nothing.

**Corporate actions.** yfinance's adjusted close missed a 2-for-1 split in
VITROX (0097): the adjusted series showed the same -50.3% step as the raw one.
The split was in Yahoo's data all along, under `events`. Fetching it means the
repair layer can confirm a split from the authoritative record rather than
inferring one from a price pattern.

One wrinkle worth knowing: for VITROX, Yahoo dates that split **2024-06-10**
while the price actually steps on **2024-05-02**, five weeks earlier. So the
event record is authoritative for *whether a split happened and at what
ratio*, and the price series is authoritative for *when*. `core.repair` uses
the event list to confirm and the price step to locate.
"""
from __future__ import annotations

import datetime as dt
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import pandas as pd

BASE = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; bursa-research/1.0)"}
COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]

# Yahoo rate-limits. Sequential with a pause, because a throttled reply can
# come back as a 200 with no rows, which would look like a delisting.
DEFAULT_PAUSE = 2.0


@dataclass
class Fetched:
    """One ticker's bars plus the corporate actions Yahoo has on record."""
    ticker: str
    bars: pd.DataFrame
    splits: pd.DataFrame = field(default_factory=pd.DataFrame)
    dividends: pd.DataFrame = field(default_factory=pd.DataFrame)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and len(self.bars) > 0


def _to_symbol(code: str) -> str:
    c = str(code).strip().upper()
    return c if c.endswith(".KL") else f"{c}.KL"


def _request(symbol: str, rng: str, timeout: int) -> dict:
    url = BASE.format(symbol=symbol) + f"?range={rng}&interval=1d&events=div%2Csplit"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _bars_from(result: dict, symbol: str) -> pd.DataFrame:
    ts = result.get("timestamp") or []
    if not ts:
        return pd.DataFrame(columns=COLUMNS)
    q = result["indicators"]["quote"][0]
    adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose")

    df = pd.DataFrame({
        "date": [pd.Timestamp(dt.date.fromtimestamp(t)) for t in ts],
        "ticker": symbol,
        "open": q.get("open"), "high": q.get("high"), "low": q.get("low"),
        "close": q.get("close"),
        "adj_close": adj if adj is not None else q.get("close"),
        "volume": q.get("volume"),
    })
    # Yahoo emits nulls for halted days. A bar with no close is not a bar.
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    df["volume"] = df["volume"].fillna(0)
    return df[COLUMNS]


def _events_from(result: dict, symbol: str, kind: str) -> pd.DataFrame:
    ev = (result.get("events") or {}).get(kind) or {}
    if not ev:
        return pd.DataFrame(columns=["date", "ticker", "value"])
    rows = []
    for k, v in ev.items():
        value = (v.get("numerator", 0) / v.get("denominator", 1)
                 if kind == "splits" else v.get("amount"))
        rows.append({"date": pd.Timestamp(dt.date.fromtimestamp(int(k))),
                     "ticker": symbol, "value": value})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def fetch(code: str, rng: str = "10y", timeout: int = 40) -> Fetched:
    """Fetch one ticker. Raises nothing; failures come back on the object."""
    symbol = _to_symbol(code)
    try:
        payload = _request(symbol, rng, timeout)
    except urllib.error.HTTPError as e:
        return Fetched(symbol, pd.DataFrame(columns=COLUMNS),
                       error=f"HTTP {e.code}")
    except Exception as e:
        return Fetched(symbol, pd.DataFrame(columns=COLUMNS),
                       error=f"{type(e).__name__}: {e}")

    chart = payload.get("chart") or {}
    if chart.get("error"):
        return Fetched(symbol, pd.DataFrame(columns=COLUMNS),
                       error=str(chart["error"]))
    results = chart.get("result") or []
    if not results:
        return Fetched(symbol, pd.DataFrame(columns=COLUMNS),
                       error="no result in response")

    r = results[0]
    bars = _bars_from(r, symbol)
    if bars.empty:
        return Fetched(symbol, bars, error="no bars returned "
                                           "(delisted, bad code, or throttled)")
    return Fetched(symbol, bars,
                   splits=_events_from(r, symbol, "splits"),
                   dividends=_events_from(r, symbol, "dividends"))


def fetch_many(codes: list[str], rng: str = "10y",
               pause: float = DEFAULT_PAUSE,
               verbose: bool = True) -> list[Fetched]:
    """Fetch a list, politely and sequentially."""
    out = []
    for i, c in enumerate(codes, 1):
        f = fetch(c, rng)
        out.append(f)
        if verbose:
            state = (f"{len(f.bars):>5} bars, {len(f.splits)} splits"
                     if f.ok else f"FAILED {f.error}")
            print(f"[{i:>3}/{len(codes)}] {f.ticker:<12} {state}")
        time.sleep(pause)
    if verbose:
        bad = [f for f in out if not f.ok]
        print(f"\n{len(out) - len(bad)}/{len(out)} succeeded")
    return out


def to_panel(results: list[Fetched]) -> pd.DataFrame:
    """Concatenate the successful fetches into one long panel."""
    frames = [f.bars for f in results if f.ok]
    if not frames:
        raise ValueError("no successful fetches")
    return (pd.concat(frames, ignore_index=True)
              .sort_values(["ticker", "date"]).reset_index(drop=True))


def to_splits(results: list[Fetched]) -> pd.DataFrame:
    """All recorded split events across the fetched tickers.

    Feed this to core.repair as authoritative confirmation: a price step that
    matches a recorded ratio is a split, and one that matches nothing is a
    real move that must not be rewritten.
    """
    frames = [f.splits for f in results if f.ok and len(f.splits)]
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "value"])
    return (pd.concat(frames, ignore_index=True)
              .sort_values(["ticker", "date"]).reset_index(drop=True))
