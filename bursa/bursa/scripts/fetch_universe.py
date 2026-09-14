"""Bootstrap the real Bursa dataset. Run this once, on your own machine.

    python -m scripts.fetch_universe

What it does, in order:

  1. Attempts every candidate stock code from core.tickers.
  2. Reports which returned data and which did not -- a failure here is a bad
     guess or a genuine delisting, and you want to know which.
  3. Writes the verified codes to data/universe.csv (committed: it is the
     record of what your results were computed against).
  4. Runs the ten integrity checks over the assembled panel.

Expect failures on step 4. Real market data contains genuine 40% days, real
suspensions, and the occasional oddity. The point is to SEE them before a
strategy quietly trades on them -- not to get a clean bill of health.

Nothing here is committed to git except data/universe.csv: the raw pulls are
gitignored, bulky, and re-downloadable.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from core.loader import add_returns, fetch_many, load_panel
from core.repair import repair
from core.tickers import load_candidates, save_verified
from core.validate import validate

RULE = "=" * 74
START = "2018-01-01"          # ~8 years: enough for walk-forward, not so much
                              # that the market regime is unrecognisable


def main() -> int:
    candidates = load_candidates()
    print(f"{RULE}\nFETCHING {len(candidates)} candidate Bursa codes from {START}\n{RULE}")
    print("Codes that fail are bad guesses or delistings -- both worth knowing.\n")

    results = fetch_many(candidates, start=START, verbose=True)

    good = [r.ticker.replace(".KL", "") for r in results if r.ok]
    bad = [(r.ticker, r.error) for r in results if not r.ok]

    print(f"\n{RULE}\nRESULT\n{RULE}")
    print(f"  {len(good)}/{len(candidates)} codes returned data")
    if bad:
        print(f"\n  did not resolve ({len(bad)}) -- check whether these are "
              f"typos or real delistings:")
        for t, err in bad:
            print(f"    {t:<12} {err}")

    if not good:
        print("\nNothing fetched. Check your internet connection and that "
              "yfinance is installed.")
        return 1

    path = save_verified(good)
    print(f"\n  verified universe written to {path}  (commit this file)")

    # ---------------------------------------------------------------- panel
    panel = add_returns(load_panel())
    print(f"\n{RULE}\nPANEL\n{RULE}")
    print(f"  {len(panel):,} bars | {panel['ticker'].nunique()} tickers | "
          f"{panel['date'].min().date()} to {panel['date'].max().date()}")

    per_ticker = panel.groupby("ticker").size()
    print(f"  bars per ticker: min {per_ticker.min()}, "
          f"median {int(per_ticker.median())}, max {per_ticker.max()}")

    # A first look at what the liquidity screen will do.
    latest = panel[panel["date"] > panel["date"].max() - pd.Timedelta(days=90)]
    dtv = (latest.assign(dtv=latest["close"] * latest["volume"])
                 .groupby("ticker")["dtv"].median().sort_values(ascending=False))
    print(f"\n  median daily traded value, last 90 days:")
    print(f"    most liquid : {dtv.index[0]:<12} RM {dtv.iloc[0]:>14,.0f}")
    print(f"    median      : {dtv.index[len(dtv)//2]:<12} RM {dtv.median():>14,.0f}")
    print(f"    least liquid: {dtv.index[-1]:<12} RM {dtv.iloc[-1]:>14,.0f}")
    print(f"    above RM 500k/day: {(dtv >= 500_000).sum()}/{len(dtv)} tickers")

    # ----------------------------------------------------------- integrity
    print(f"\n{RULE}\nDATA HEALTH, BEFORE REPAIR\n{RULE}")
    validate(panel, verbose=True)

    # -------------------------------------------------------------- repair
    print(f"\n{RULE}\nREPAIR\n{RULE}")
    clean, _ = repair(panel, verbose=True)

    print(f"\n{RULE}\nDATA HEALTH, AFTER REPAIR\n{RULE}")
    for r in validate(add_returns(clean), verbose=False):
        print(r)
    print("\n  Remaining failures are expected: genuine large moves want "
          "reading,\n  and an untradeable ticker is removed by the liquidity "
          "screen, not here.")

    print(f"\n{RULE}")
    print("NEXT: parse the SC Shariah-compliant list into data/shariah.csv")
    print("      (schema in README), then the universe screen can run.")
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
