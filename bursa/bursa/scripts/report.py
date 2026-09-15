"""One stock, everything known about it, and what happened to names like it.

    python -m scripts.report 0270
    python -m scripts.report 0270 --held --entry 1.66 --shares 6000 --days-left 15
    python -m scripts.report 0270 --factor volatility

## What this is

The answer to "should I buy this?", given honestly. Not a direction with a
confidence percentage attached -- a distribution, drawn from every stock that
has ever occupied the same position in the same universe.

Four sections, in the order they matter:

1. **Is it tradable at all?** Shariah, liquidity, price band, history depth.
   A name that fails here needs no further analysis.
2. **Where does it sit today?** Rank out of the live universe, and which
   decile that puts it in.
3. **What happened to names in that decile?** The base rate: median, the
   middle half, the tails, how often it went up, how often it fell 30%.
   Over the horizon you actually care about.
4. **What to do**, if you hold it -- straight from `tournament.rules`.

## What the numbers are and are not

The base rate is what *happened*, not what *will* happen. Over a 21-day
horizon on Bursa small caps the spread between the 5th and 95th percentile is
enormous, and that spread IS the finding: the median tells you very little and
the range tells you almost everything. A decile with a +2% median and a
-25% to +45% range is not "expected to make 2%", it is "anything can happen
and the centre of mass is barely positive".

Two things the sample cannot fix, both stated on every run:

* **Overlap.** Daily observations with 21-day windows share 20 of 21 days,
  so the row count massively overstates the evidence. The report prints how
  many genuinely independent periods are behind each figure.
* **Survivorship.** Roughly 52 delisted companies are missing because Yahoo
  drops their history. The downside percentages are floors.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from core.costs import BOARD_LOT, MOOMOO, affordable_lots, round_trip_pct
from research.baserates import (
    DEFAULT_HORIZON, DEFAULT_LOOKBACK, base_rates, current_bucket,
)
from scripts.pick import latest_tradable_date
from tournament.rules import Action, Situation, decide, explain_no_stop_loss


def normalise(code: str) -> str:
    c = str(code).strip().upper()
    return c if c.endswith(".KL") else f"{c}.KL"


def eligibility_report(panel: pd.DataFrame, ticker: str,
                       as_of: pd.Timestamp) -> list[tuple[str, bool, str]]:
    """Each screen condition, passed or not, with the number behind it."""
    row = panel[(panel["date"] == as_of) & (panel["ticker"] == ticker)]
    if row.empty:
        return [("present in panel", False, f"no bar on {as_of.date()}")]
    r = row.iloc[0]
    checks = []
    for col, label, detail in (
        ("shariah", "Shariah-compliant", "per the SC list in force that day"),
        ("liquid", "liquid enough",
         f"median daily value RM {r.get('median_dtv', float('nan')):,.0f}"),
        ("priced", "inside the price band", f"RM {r['close']:.3f}"),
        ("seasoned", "enough history",
         f"{int(r.get('bars_available', 0))} bars"),
    ):
        if col in row.columns:
            checks.append((label, bool(r[col]), detail))
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ticker")
    ap.add_argument("--panel", type=Path, default=Path("data/screened.parquet"))
    ap.add_argument("--factor", default="momentum",
                    choices=("momentum", "volatility", "reversal"))
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--held", action="store_true",
                    help="you already own it; include the hold/switch verdict")
    ap.add_argument("--entry", type=float, default=0.0)
    ap.add_argument("--shares", type=int, default=0)
    ap.add_argument("--days-left", type=int, default=15)
    args = ap.parse_args()

    if not args.panel.exists():
        print(f"ERROR: {args.panel} not found. Run scripts.build_panel.",
              file=sys.stderr)
        return 2

    ticker = normalise(args.ticker)
    panel = pd.read_parquet(args.panel)
    as_of = latest_tradable_date(panel)

    print(f"\n{'=' * 68}")
    print(f"  {ticker}   as of {as_of.date()}   "
          f"factor: {args.factor} ({args.lookback}d)")
    print("=" * 68)

    # ---------------------------------------------------------- 1. tradable
    print("\n1. IS IT TRADABLE")
    checks = eligibility_report(panel, ticker, as_of)
    for label, ok, detail in checks:
        print(f"   [{'ok  ' if ok else 'FAIL'}] {label:<26} {detail}")
    if not all(ok for _, ok, _ in checks):
        print("\n   Fails the screen. Nothing below applies — you could not "
              "hold this\n   name under the rules this project trades by.")
        return 1

    row = panel[(panel["date"] == as_of) & (panel["ticker"] == ticker)].iloc[0]
    px = float(row["close"])
    lots = affordable_lots(px, args.capital)
    print(f"\n   RM {px:.3f}  |  RM {args.capital:,.0f} buys {lots} lot(s) "
          f"= {lots * BOARD_LOT:,} shares = RM {lots * BOARD_LOT * px:,.2f}")
    print(f"   round trip ~{round_trip_pct(lots * BOARD_LOT * px, MOOMOO):.2f}%")

    # ------------------------------------------------------------ 2. ranking
    print(f"\n2. WHERE IT SITS TODAY")
    pos = current_bucket(panel, ticker, args.factor, args.lookback)
    if not pos["found"]:
        print(f"   {pos['reason']}")
        return 1
    print(f"   rank {pos['rank']} of {pos['n_universe']} eligible names  "
          f"({pos['percentile']:.0%} percentile)")
    print(f"   decile {pos['bucket']} of 10   "
          f"{args.factor} = {pos['factor_value']:+.1%}")

    # ---------------------------------------------------------- 3. base rate
    print(f"\n3. WHAT HAPPENED TO NAMES IN DECILE {pos['bucket']}, "
          f"NEXT {args.horizon} TRADING DAYS")
    stats = base_rates(panel, args.factor, args.horizon, args.lookback)
    mine = next((s for s in stats if s.bucket == pos["bucket"]), None)
    if mine is None:
        print("   no history for this decile.")
        return 1

    print(f"\n   {'median':<26}{mine.median:>+8.2%}")
    print(f"   {'middle half (25-75th)':<26}"
          f"{mine.p25:>+8.2%} to {mine.p75:>+7.2%}")
    print(f"   {'5th to 95th':<26}{mine.p05:>+8.2%} to {mine.p95:>+7.2%}")
    print(f"\n   {'ended higher':<26}{mine.p_positive:>8.1%}")
    print(f"   {'gained more than 20%':<26}{mine.p_gain_20:>8.1%}")
    print(f"   {'lost more than 20%':<26}{mine.p_loss_20:>8.1%}")
    print(f"   {'lost more than 30%':<26}{mine.p_loss_30:>8.1%}")

    print(f"\n   Based on {mine.n:,} observations — but only about "
          f"{mine.independent_periods:.0f} INDEPENDENT")
    print(f"   {args.horizon}-day periods, because daily windows overlap. "
          f"Treat the tails as\n   rough. Delisted companies are missing, so "
          f"the loss figures are floors.")
    if mine.thin:
        print("\n   *** THIN SAMPLE — too few observations to trust. ***")

    print("\n   All deciles, for context:")
    for s in stats:
        mark = "  <-- this one" if s.bucket == pos["bucket"] else ""
        print(f"{s}{mark}")

    # ------------------------------------------------------------ 4. verdict
    print(f"\n4. WHAT TO DO")
    if args.held:
        if not args.entry:
            print("   --held needs --entry to judge the position.",
                  file=sys.stderr)
            return 2
        v = decide(Situation(
            days_left=args.days_left,
            my_return=px / args.entry - 1.0,
            position_return=px / args.entry - 1.0,
            halted=False, still_eligible=True,
            momentum_at_entry=pos["factor_value"],
            momentum_now=pos["factor_value"]))
        print(f"   {v.action.value}: {v.reason}")
        print(f"   {v.detail}")
        if v.action is Action.HOLD:
            print(f"\n   {explain_no_stop_loss()}")
    else:
        # Not a recommendation dressed as one: the decile and the spread are
        # the content, and the contest rule is what turns them into an action.
        top = max(s.bucket for s in stats)
        if pos["bucket"] == top:
            print(f"   This is the bucket the contest strategy buys — top "
                  f"{args.factor}\n   decile, widest forward distribution. "
                  f"Note the median above is\n   {mine.median:+.1%}: you are "
                  f"buying the RANGE, not the centre.")
        else:
            print(f"   Decile {pos['bucket']}, not the top. The contest "
                  f"strategy would hold\n   the decile-{top} name instead — "
                  f"run `python -m scripts.pick`.")
        print(f"\n   Reminder: this is the right trade only because the "
              f"contest capital\n   is the organisers'. On your own money "
              f"the same concentration is not.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
