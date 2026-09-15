"""Fetch every Shariah-compliant Bursa security in the SC list.

    python -m scripts.fetch_shariah_universe            # resumable
    python -m scripts.fetch_shariah_universe --limit 200

## Why this replaces core.tickers for real work

`core/tickers.py` holds ~60 candidate codes assembled from memory of notable
Bursa companies. That is a biased sample twice over: every name survived to
today, and every name was memorable enough to recall. Momentum strategies
look far better than they are on a universe of winners.

The SC's list is a mechanical enumeration -- 855 codes, chosen by a regulator
applying published criteria, not by anyone's recollection. Using it removes
the selection bias entirely.

## What it still does NOT fix

Survivorship. The November 2025 list contains companies compliant in November
2025. A company that delisted in 2021 is absent, so a backtest over 2018-2026
still only sees survivors. That requires Bursa's historical listings, which is
a separate download.

So: this removes the bias I introduced. The bias inherent to the data source
remains, and any result should say so.

Resumable: already-cached tickers are skipped, so interrupting and re-running
costs nothing.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from core.yahoo import fetch

RAW = Path("data/raw")
SPLITS = Path("data/splits.csv")
PAUSE = 1.2


def cached(code: str, raw_dir: Path) -> bool:
    return (raw_dir / f"{code}_KL.parquet").exists()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", type=Path, default=Path("data/shariah.csv"))
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--range", default="10y")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many NEW fetches (0 = no limit)")
    ap.add_argument("--pause", type=float, default=PAUSE)
    args = ap.parse_args()

    if not args.list.exists():
        raise SystemExit(f"{args.list} not found. Run scripts.parse_shariah first.")

    codes = (pd.read_csv(args.list, dtype=str)["ticker"]
             .str.replace(".KL", "", regex=False).drop_duplicates().tolist())
    args.raw.mkdir(parents=True, exist_ok=True)

    todo = [c for c in codes if not cached(c, args.raw)]
    done_already = len(codes) - len(todo)
    if args.limit:
        todo = todo[:args.limit]

    print(f"SC list: {len(codes)} codes | already cached: {done_already} | "
          f"fetching now: {len(todo)}")
    if not todo:
        print("nothing to do -- everything is cached")
        return 0

    ok, failed, split_rows = 0, [], []
    t0 = time.time()
    for i, code in enumerate(todo, 1):
        f = fetch(code, rng=args.range)
        if f.ok:
            f.bars.to_parquet(args.raw / f"{code}_KL.parquet", index=False)
            ok += 1
            if len(f.splits):
                split_rows.append(f.splits)
        else:
            failed.append((code, f.error))
        if i % 25 == 0 or i == len(todo):
            rate = (time.time() - t0) / i
            print(f"  [{i:>4}/{len(todo)}] ok {ok:>4}  failed {len(failed):>3}  "
                  f"~{rate * (len(todo) - i) / 60:.0f} min left")
        time.sleep(args.pause)

    # Corporate actions are worth keeping: core.repair uses them to confirm a
    # split rather than infer one from a price pattern.
    if split_rows:
        fresh = pd.concat(split_rows, ignore_index=True)
        if SPLITS.exists():
            fresh = pd.concat([pd.read_csv(SPLITS, parse_dates=["date"]), fresh],
                              ignore_index=True).drop_duplicates()
        fresh.sort_values(["ticker", "date"]).to_csv(SPLITS, index=False)
        print(f"\n  {len(fresh)} split events recorded in {SPLITS}")

    print(f"\n  fetched {ok}/{len(todo)}  ({len(failed)} failed)")
    if failed:
        print("  failures (delisted, suspended, or not on Yahoo):")
        for c, e in failed[:20]:
            print(f"    {c:<8} {str(e)[:60]}")
        if len(failed) > 20:
            print(f"    ... and {len(failed) - 20} more")
    print(f"\n  cache now holds {len(list(args.raw.glob('*.parquet')))} tickers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
