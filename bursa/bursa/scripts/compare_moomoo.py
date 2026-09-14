"""Check the research data against your broker's. Run on YOUR machine.

    python -m scripts.compare_moomoo --tickers 1155 0097 5183 7113 8869

Requires OpenD running and logged in, and `pip install futu-api`.

## Why

`core.validate` checks whether a panel is internally consistent. It cannot
tell you whether the prices are RIGHT -- only an independent source can, and
your broker's prices are the ones you would actually trade at.

Yahoo's Bursa data has already produced an unapplied 2-for-1 split, a ticker
with no dividend adjustment at all, and a corrupt bar. Those were caught
because the checks looked for them. This closes the gap between "internally
consistent" and "correct".

## Keep it small

moomoo issues historical candlestick quota based on your account assets, and
limits how many stocks you can pull within a 7-day window. This script is for
SPOT-CHECKING ten or twenty names, not for bulk history -- that is what the
Yahoo fetch is for. Pulling your whole universe here would burn a week of
quota for data you already have.

## One thing you may need to change

moomoo prefixes codes by market -- 'US.AAPL', 'HK.00700'. Malaysia is assumed
to be 'MY.' below. If every fetch fails with an unknown-code error, print what
`ctx.get_stock_basicinfo(Market.MY, SecurityType.STOCK)` returns and set
--prefix accordingly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from core.compare import compare_panels, report, splits_missing_from
from core.loader import load_panel
from core.yahoo import fetch_many as yahoo_fetch, to_panel, to_splits

MOOMOO_HOST = "127.0.0.1"
MOOMOO_PORT = 11111


def fetch_moomoo(codes: list[str], start: str, end: str,
                 prefix: str = "MY.", host: str = MOOMOO_HOST,
                 port: int = MOOMOO_PORT) -> pd.DataFrame:
    """Pull daily bars from OpenD. Raises with a readable message on failure."""
    try:
        from futu import AuType, KLType, OpenQuoteContext, RET_OK
    except ImportError:
        raise SystemExit(
            "futu-api not installed.  pip install futu-api\n"
            "You also need OpenD running and logged in:\n"
            "  https://www.moomoo.com/download/OpenAPI")

    ctx = OpenQuoteContext(host=host, port=port)
    frames = []
    try:
        for code in codes:
            sym = f"{prefix}{code}"
            # autype=QFQ gives the adjusted series, which is what we want to
            # compare against Yahoo's adj_close.
            ret, data, page = ctx.request_history_kline(
                sym, start=start, end=end,
                ktype=KLType.K_DAY, autype=AuType.QFQ, max_count=5000)
            if ret != RET_OK:
                print(f"  {sym:<12} FAILED: {data}")
                continue
            if data is None or data.empty:
                print(f"  {sym:<12} no data returned")
                continue

            df = pd.DataFrame({
                "date": pd.to_datetime(data["time_key"]).dt.normalize(),
                "ticker": f"{code}.KL",
                "close": pd.to_numeric(data["close"]),
                # QFQ is already adjusted; keep both columns so the comparison
                # has the same schema as the Yahoo panel.
                "adj_close": pd.to_numeric(data["close"]),
            })
            frames.append(df)
            print(f"  {sym:<12} {len(df):>5} bars")
    finally:
        ctx.close()

    if not frames:
        raise SystemExit("moomoo returned nothing. Is OpenD running and logged "
                         "in? Is the market prefix right? (see --prefix)")
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", nargs="+", required=True,
                    help="Bursa codes to spot-check. Keep it to ~10-20; "
                         "moomoo's historical quota is small.")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--prefix", default="MY.", help="moomoo market prefix")
    ap.add_argument("--from-cache", action="store_true",
                    help="use the existing data/raw cache instead of "
                         "re-fetching from Yahoo")
    ap.add_argument("--out", type=Path, default=Path("results/moomoo_diff.csv"))
    args = ap.parse_args()

    if len(args.tickers) > 25:
        raise SystemExit(f"{len(args.tickers)} tickers is too many for a spot "
                         f"check -- moomoo's quota is per-stock and slow to "
                         f"release. Use 10-20.")

    print("=" * 74)
    print("YAHOO")
    print("=" * 74)
    if args.from_cache:
        yp = load_panel([f"{t}.KL" for t in args.tickers])
        ysplits = pd.DataFrame(columns=["date", "ticker", "value"])
        print(f"  {len(yp):,} bars from the local cache")
    else:
        res = yahoo_fetch(args.tickers, rng="10y")
        yp, ysplits = to_panel(res), to_splits(res)

    yp = yp[(yp["date"] >= args.start) & (yp["date"] <= args.end)]

    print("\n" + "=" * 74)
    print("MOOMOO")
    print("=" * 74)
    mp = fetch_moomoo(args.tickers, args.start, args.end, args.prefix)

    print("\n" + "=" * 74)
    print("COMPARISON")
    print("=" * 74)
    table = report(compare_panels(yp, mp))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"\n  written to {args.out}")

    if len(ysplits):
        print("\n" + "=" * 74)
        print("SPLITS ON RECORD, AND WHETHER YAHOO APPLIED THEM")
        print("=" * 74)
        s = splits_missing_from(yp, ysplits)
        if len(s):
            print(s.to_string(index=False))
            unapplied = s[~s["applied"]]
            if len(unapplied):
                print(f"\n  {len(unapplied)} split(s) NOT applied to the "
                      f"adjusted series -- core.repair handles these, but it "
                      f"is worth knowing\n  which tickers need it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
