"""Which stocks are tradable, and when.

Everything here is evaluated **as of a date**. That is the whole point of the
module. Screening once on today's data and applying the result backwards is
lookahead bias -- structurally the same error as survivorship bias, and just
as invisible in a backtest that reports a good number.

Two filters:

* **Shariah compliance.** The SC's Shariah Advisory Council publishes its list
  twice a year (May and November). Names are added and removed at each
  revision -- one recent update added 44 and removed 18 -- so "compliant" is a
  function of date, not a fixed property of a ticker.

* **Liquidity.** A high-volatility ranking will otherwise hand you exactly the
  thin small caps you cannot actually trade. The screen uses median daily
  traded value rather than mean, because a single block trade should not
  qualify a stock that is otherwise dead.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core.costs import BOARD_LOT
from core.loader import to_yahoo


@dataclass(frozen=True)
class ScreenConfig:
    """Universe rules. Defaults suit a small account on Bursa.

    min_median_dtv : median daily traded value (RM) over the lookback. The
        floor exists so a position can be entered and exited without moving
        the price. RM 500k is modest; raise it if trading size grows.
    lookback_days  : trailing window for the liquidity measure.
    min_price      : below this, the minimum tick is a large fraction of the
        price and spreads are punishing.
    max_price      : one board lot (100 shares) must be affordable. At RM 50
        a share that is RM 5,000 in a single position.
    min_history    : bars of history required before a stock is eligible, so
        features with a lookback have something to work with.
    backfill_shariah : apply the earliest SC release to all earlier dates.
        Lookahead bias, off by default. See compliant_on(). Turn it on only
        when you have too few releases to cover the backtest window, and
        record it as a limitation of any result produced with it.
    """
    min_median_dtv: float = 500_000.0
    lookback_days: int = 60
    min_price: float = 0.20
    max_price: float = 50.00
    min_history: int = 120
    require_shariah: bool = True
    backfill_shariah: bool = False


# --------------------------------------------------------------------------
# Shariah compliance, point in time
# --------------------------------------------------------------------------
def load_shariah_lists(path: str | Path) -> pd.DataFrame:
    """Read the parsed SC lists.

    Expected CSV schema -- one row per (release, ticker):

        list_date,ticker
        2025-05-30,1155
        2025-05-30,5225
        2025-11-28,1155
        ...

    Each release is a complete snapshot, not a diff. A ticker absent from a
    release was not compliant at that revision.
    """
    # dtype=str is load-bearing, not tidiness. Bursa codes are strings with
    # significant leading zeros: ACE-market names are 0xxx and LEAP are 03xxx.
    # Letting pandas infer reads "0001" as the integer 1, which becomes
    # "1.KL", which matches no ticker in the panel. That silently dropped 312
    # of 865 names -- every ACE and LEAP stock -- from the compliant set, with
    # no error anywhere: the screen simply reported them non-compliant.
    df = pd.read_csv(path, dtype={"ticker": str})
    missing = {"list_date", "ticker"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {sorted(missing)}")

    df["list_date"] = pd.to_datetime(df["list_date"]).dt.normalize()
    df["ticker"] = df["ticker"].astype(str).map(to_yahoo)
    return df.drop_duplicates().sort_values(["list_date", "ticker"])


def compliant_on(lists: pd.DataFrame, date: pd.Timestamp,
                 backfill: bool = False) -> set[str]:
    """Tickers compliant as of `date`: the most recent release on or before it.

    Returns an empty set if `date` precedes the first release -- deliberately.
    An empty universe is a loud failure; silently falling back to the earliest
    available list would be a quiet one.

    `backfill=True` applies the EARLIEST release to all prior dates. This is
    lookahead bias, knowingly accepted: a stock that became compliant in 2025
    is treated as compliant in 2019, when you could not have held it. It exists
    because one SC release leaves ten months of usable history and eight years
    of empty universe, and a biased backtest you have labelled beats no
    backtest at all. The bias is optimistic and its size is unknown.

    The honest fix is more releases. The SC archives them; each parsed file
    appended to data/shariah.csv shrinks the backfilled window.
    """
    date = pd.Timestamp(date).normalize()
    eligible = lists.loc[lists["list_date"] <= date, "list_date"]
    if eligible.empty:
        if not backfill or lists.empty:
            return set()
        earliest = lists["list_date"].min()
        return set(lists.loc[lists["list_date"] == earliest, "ticker"])
    latest = eligible.max()
    return set(lists.loc[lists["list_date"] == latest, "ticker"])


def add_shariah_flag(panel: pd.DataFrame, lists: pd.DataFrame,
                     backfill: bool = False) -> pd.DataFrame:
    """Add a point-in-time `shariah` boolean to a long panel.

    Implemented as a merge_asof on release date rather than a per-row lookup,
    so it stays fast on a full panel.

    `backfill=True` extends the earliest release backwards -- see
    compliant_on() for why that is lookahead bias and when it is worth it.
    """
    out = panel.sort_values("date").copy()

    releases = sorted(lists["list_date"].unique())
    membership = {d: set(lists.loc[lists["list_date"] == d, "ticker"])
                  for d in releases}

    rel = pd.DataFrame({"date": releases, "release": releases})
    out = pd.merge_asof(out, rel, on="date", direction="backward")

    earliest = membership[releases[0]] if (backfill and releases) else None
    out["shariah"] = [
        (t in membership[r]) if pd.notna(r)
        else (earliest is not None and t in earliest)
        for t, r in zip(out["ticker"], out["release"])
    ]
    return out.drop(columns="release")


# --------------------------------------------------------------------------
# Liquidity
# --------------------------------------------------------------------------
def add_liquidity(panel: pd.DataFrame, cfg: ScreenConfig = ScreenConfig()
                  ) -> pd.DataFrame:
    """Add daily traded value and its trailing median.

    The trailing median is **shifted by one bar**. Eligibility on day t is
    decided using data through t-1 only, so a stock cannot qualify itself with
    the same day's volume that a signal then trades on.
    """
    out = panel.sort_values(["ticker", "date"]).copy()
    out["dtv"] = out["close"] * out["volume"]

    g = out.groupby("ticker", observed=True)["dtv"]
    out["median_dtv"] = (
        g.shift(1)
         .groupby(out["ticker"], observed=True)
         .rolling(cfg.lookback_days, min_periods=cfg.lookback_days // 2)
         .median()
         .reset_index(level=0, drop=True)
    )
    out["bars_available"] = (
        out.groupby("ticker", observed=True).cumcount()
    )
    return out


# --------------------------------------------------------------------------
# The screen
# --------------------------------------------------------------------------
def screen(panel: pd.DataFrame,
           cfg: ScreenConfig = ScreenConfig(),
           shariah_lists: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add an `eligible` boolean, plus the reason columns behind it.

    Keeping the individual conditions as columns rather than collapsing
    straight to a boolean means you can answer "why is this stock missing?"
    without re-running anything.
    """
    out = add_liquidity(panel, cfg)

    if cfg.require_shariah:
        if shariah_lists is None:
            raise ValueError(
                "require_shariah=True but no shariah_lists supplied. "
                "Pass the parsed SC list, or set require_shariah=False.")
        out = add_shariah_flag(out, shariah_lists, cfg.backfill_shariah)
    else:
        out["shariah"] = True

    out["liquid"] = out["median_dtv"] >= cfg.min_median_dtv
    out["priced"] = out["close"].between(cfg.min_price, cfg.max_price)
    out["seasoned"] = out["bars_available"] >= cfg.min_history

    out["eligible"] = (out["shariah"] & out["liquid"]
                       & out["priced"] & out["seasoned"])
    return out


def universe_on(screened: pd.DataFrame, date: pd.Timestamp) -> list[str]:
    """Tickers tradable on `date`, sorted. Reads the screen; computes nothing."""
    date = pd.Timestamp(date).normalize()
    rows = screened[(screened["date"] == date) & screened["eligible"]]
    return sorted(rows["ticker"].unique())


def rejection_reasons(screened: pd.DataFrame,
                      date: pd.Timestamp) -> pd.DataFrame:
    """Why each ticker failed on `date`. For debugging an empty universe."""
    date = pd.Timestamp(date).normalize()
    rows = screened[screened["date"] == date]
    cols = ["ticker", "close", "median_dtv", "bars_available",
            "shariah", "liquid", "priced", "seasoned", "eligible"]
    return rows[cols].sort_values("ticker").reset_index(drop=True)


def screen_summary(screened: pd.DataFrame) -> pd.DataFrame:
    """Universe size over time, and how many names each filter removes.

    Worth looking at before trusting any backtest: a universe that collapses
    to three names in 2020 explains a lot of apparent alpha.
    """
    g = screened.groupby("date", observed=True)
    return pd.DataFrame({
        "n_listed": g["ticker"].nunique(),
        "n_shariah": g["shariah"].sum(),
        "n_liquid": g["liquid"].sum(),
        "n_priced": g["priced"].sum(),
        "n_seasoned": g["seasoned"].sum(),
        "n_eligible": g["eligible"].sum(),
    }).astype(int)


def max_affordable_names(prices: pd.Series, capital: float) -> int:
    """How many distinct names `capital` can hold, given board lots.

    Board lots are the reason a small account cannot diversify: at the median
    price of the universe, one lot of 100 shares may be a tenth of the account.
    """
    if prices.empty or capital <= 0:
        return 0
    lot_cost = prices.median() * BOARD_LOT
    return int(capital // lot_cost) if lot_cost > 0 else 0
