"""Assemble the cached raw tickers into one clean, validated panel.

    python -m scripts.build_panel
    python -m scripts.build_panel --no-splits     # see what the record buys

This is the only sanctioned path from `data/raw/*.parquet` to
`data/clean_panel.parquet`. Everything downstream -- universe screen,
backtest, walk-forward -- reads the output of this script, so the repairs it
makes are the repairs the whole project is standing on.

## Why the split record is not optional

Running the detector's heuristic over 58 hand-picked large caps found three
real faults. Running the same heuristic over all 865 Shariah-compliant
tickers reported **1,104 splits**, including ~30 on a single ticker inside
five months, alternating +50% and -33.3%.

Those are two-sen stocks. Bursa's tick below RM 1.00 is half a sen, so
RM 0.02 -> RM 0.03 is exactly +50%, which is exactly a 1.5 ratio, which is
exactly a KNOWN_RATIO. No volume spike, because nothing happened. No drift
afterwards, because it ticks straight back. All three original gates pass on
pure quantisation noise.

So this script feeds `data/splits.csv` -- the corporate actions Yahoo has on
record -- into the repair layer, which then treats the price pattern as a way
to LOCATE a split the issuer already declared rather than as evidence that
one occurred. `--no-splits` reproduces the broken behaviour on purpose, so
the difference is measurable rather than asserted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from core.repair import repair
from core.universe import ScreenConfig, load_shariah_lists, screen
from core.validate import validate


def load_raw(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no cached tickers in {raw_dir} -- run "
                         f"scripts.fetch_shariah_universe first")
    frames = [pd.read_parquet(f) for f in files]
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["ticker", "date"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/clean_panel.parquet"))
    ap.add_argument("--shariah", type=Path, default=Path("data/shariah.csv"))
    ap.add_argument("--screened", type=Path,
                    default=Path("data/screened.parquet"))
    ap.add_argument("--backfill", action="store_true", default=True,
                    help="apply the earliest SC release to all prior dates "
                         "(lookahead; on by default so the early years are "
                         "usable, and reported as backfilled every run)")
    ap.add_argument("--no-backfill", dest="backfill", action="store_false",
                    help="exclude everything before the earliest SC release")
    ap.add_argument("--no-splits", action="store_true",
                    help="ignore the recorded corporate actions (shows what "
                         "the heuristic does unsupervised -- do not ship this)")
    args = ap.parse_args()

    panel = load_raw(args.raw)
    print(f"raw: {len(panel):,} bars | {panel['ticker'].nunique()} tickers | "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}\n")

    known = None
    if not args.no_splits and args.splits.exists():
        known = pd.read_csv(args.splits, parse_dates=["date"])
        print(f"corporate actions on record: {len(known)} across "
              f"{known['ticker'].nunique()} tickers\n")
    elif args.no_splits:
        print("*** running WITHOUT the split record -- results are not "
              "trustworthy ***\n")
    else:
        print(f"*** {args.splits} not found -- falling back to the heuristic "
              f"***\n")

    clean, report = repair(panel, known_splits=known, verbose=True)

    print("\n" + "=" * 66)
    validate(clean, verbose=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(clean):,} bars, "
          f"{clean['ticker'].nunique()} tickers)")

    # The screen belongs here, not in a separate step someone forgets to run.
    # The dashboard and the pick script both read data/screened.parquet, and a
    # stale one is worse than a missing one: it silently answers with an old
    # universe instead of failing.
    if args.shariah.exists():
        lists = load_shariah_lists(args.shariah)
        releases = sorted(lists["list_date"].unique())
        first = releases[0]
        print(f"\nscreening against {len(releases)} SC release(s), "
              f"earliest {first.date()}")

        cfg = ScreenConfig(backfill_shariah=args.backfill)
        screened = screen(clean, cfg, lists)
        screened.to_parquet(args.screened, index=False)

        elig = screened[screened["eligible"]]
        real = elig[elig["date"] >= first]
        print(f"  eligible bars: {len(elig):,}  "
              f"({elig['ticker'].nunique()} distinct names)")
        # Saying which part is real and which is assumed, every run. A single
        # eligible-count headline hides the fact that most of it may rest on
        # a compliance list applied backwards.
        share = len(real) / len(elig) if len(elig) else 0.0
        print(f"  point-in-time compliance: {len(real):,} bars ({share:.0%})")
        print(f"  BACKFILLED (lookahead):   {len(elig) - len(real):,} bars "
              f"({1 - share:.0%})  — everything before {first.date()}")
        if not args.backfill:
            print("  backfill is OFF: the pre-release window is excluded "
                  "entirely.")
        print(f"\nwrote {args.screened}")
    else:
        print(f"\n{args.shariah} not found — no screen written. "
              f"Run scripts.fetch_sc_lists first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
