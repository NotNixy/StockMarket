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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
