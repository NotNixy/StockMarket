"""Fetch and cache Bursa Malaysia daily bars.

One job: get price history onto disk, once, and never fetch it twice. Every
other module reads from the cache, so a bug downstream costs a re-run rather
than a re-download.

Design notes that matter later:

* Prices are stored BOTH raw and adjusted. yfinance's adjustment is usually
  right, but an unadjusted split is indistinguishable from a 50% crash to a
  ranking model, so validate.py needs the raw series to check against.
* Storage is long format (ticker, date, ...) rather than a wide matrix.
  Cross-sectional ranking wants to group by date; a wide frame makes that
  awkward and makes incremental updates worse.
* Each ticker is its own Parquet file. Refreshing one name does not rewrite
  the whole panel, and a corrupt download damages one file.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
BURSA_SUFFIX = ".KL"

# Columns as stored. Keeping raw close alongside adj_close is what lets
# validate.py spot an unadjusted corporate action.
COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]


@dataclass
class FetchResult:
    """What happened for one ticker. Failures are data, not exceptions --
    a delisted or mistyped ticker should not abort a 100-name pull."""
    ticker: str
    rows: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    cached: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.rows > 0


def to_yahoo(ticker: str) -> str:
    """'MAYBANK' or '1155' -> '1155.KL'. Idempotent."""
    t = ticker.strip().upper()
    return t if t.endswith(BURSA_SUFFIX) else f"{t}{BURSA_SUFFIX}"


def _cache_path(ticker: str, raw_dir: Path) -> Path:
    return raw_dir / f"{to_yahoo(ticker).replace('.', '_')}.parquet"


def _normalise(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """yfinance's frame -> our schema. Tolerates its column naming drift."""
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS)

    # A single-ticker download can still come back with a MultiIndex.
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)

    df = df.reset_index()
    rename = {
        "Date": "date", "Datetime": "date",
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
    }
    df = df.rename(columns=rename)

    # auto_adjust=True collapses Close into the adjusted series and drops
    # 'Adj Close' entirely. Mirror it so the schema stays stable either way.
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    df["ticker"] = to_yahoo(ticker)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()

    for c in ("open", "high", "low", "close", "adj_close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[COLUMNS].dropna(subset=["close"])
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def fetch_one(ticker: str,
              start: str = "2015-01-01",
              end: str | None = None,
              raw_dir: Path = RAW_DIR,
              refresh: bool = False) -> FetchResult:
    """Download one ticker's daily bars, or read them from cache.

    Incremental: if the cache already ends after `start`, only the missing
    tail is requested and appended.
    """
    import yfinance as yf  # imported late so tests can run without the network

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker, raw_dir)

    existing = pd.DataFrame(columns=COLUMNS)
    fetch_from = start

    if path.exists() and not refresh:
        existing = pd.read_parquet(path)
        if not existing.empty:
            last = existing["date"].max()
            # Re-request the final day: the cached copy may have been written
            # mid-session with a provisional close.
            fetch_from = last.strftime("%Y-%m-%d")
            if end is None and last.date() >= pd.Timestamp.today().normalize().date():
                return FetchResult(to_yahoo(ticker), len(existing),
                                   existing["date"].min(), last, cached=True)

    try:
        raw = yf.download(to_yahoo(ticker), start=fetch_from, end=end,
                          auto_adjust=False, progress=False, threads=False)
        fresh = _normalise(raw, ticker)
    except Exception as exc:                      # network, parse, rate limit
        return FetchResult(to_yahoo(ticker), len(existing), None, None,
                           error=f"{type(exc).__name__}: {exc}")

    combined = (pd.concat([existing, fresh], ignore_index=True)
                  .sort_values("date")
                  .drop_duplicates("date", keep="last")   # fresh wins
                  .reset_index(drop=True))

    if combined.empty:
        return FetchResult(to_yahoo(ticker), 0, None, None,
                           error="no data returned (delisted or bad ticker?)")

    combined.to_parquet(path, index=False)
    return FetchResult(to_yahoo(ticker), len(combined),
                       combined["date"].min(), combined["date"].max())


def fetch_many(tickers: list[str],
               start: str = "2015-01-01",
               end: str | None = None,
               raw_dir: Path = RAW_DIR,
               refresh: bool = False,
               pause: float = 0.3,
               verbose: bool = True) -> list[FetchResult]:
    """Fetch a list of tickers, one at a time, politely.

    Sequential with a pause rather than parallel: yfinance throttles, and a
    throttled response can come back as an empty frame rather than an error,
    which would silently look like a delisted stock.
    """
    results: list[FetchResult] = []
    for i, t in enumerate(tickers, 1):
        r = fetch_one(t, start=start, end=end, raw_dir=raw_dir, refresh=refresh)
        results.append(r)
        if verbose:
            state = "cached" if r.cached else ("OK" if r.ok else f"FAIL {r.error}")
            print(f"[{i:>3}/{len(tickers)}] {r.ticker:<14} {r.rows:>5} rows  {state}")
        if not r.cached:
            time.sleep(pause)

    if verbose:
        bad = [r for r in results if not r.ok]
        print(f"\n{len(results) - len(bad)}/{len(results)} succeeded")
        for r in bad:
            print(f"  FAILED {r.ticker}: {r.error}")
    return results


def load_panel(tickers: list[str] | None = None,
               raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read the cache into one long frame: (date, ticker, ohlcv, adj_close).

    Reads only; never fetches. If a ticker has no cache file it is simply
    absent from the result -- call fetch_many first.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"No cache at {raw_dir.resolve()}. Run fetch_many() first.")

    paths = ([_cache_path(t, raw_dir) for t in tickers] if tickers
             else sorted(raw_dir.glob("*.parquet")))
    frames = [pd.read_parquet(p) for p in paths if p.exists()]

    if not frames:
        raise FileNotFoundError(f"No cached tickers found in {raw_dir.resolve()}")

    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["ticker", "date"]).reset_index(drop=True)


def add_returns(panel: pd.DataFrame, price_col: str = "adj_close") -> pd.DataFrame:
    """Daily simple returns per ticker, from the ADJUSTED series.

    Using raw close here would read every dividend and split as a real move,
    which is the most common way a Bursa backtest invents signal that is not
    there.
    """
    out = panel.sort_values(["ticker", "date"]).copy()
    out["ret"] = out.groupby("ticker", observed=True)[price_col].pct_change()
    return out
