"""Download the SC's archived Shariah lists and build point-in-time history.

    python -m scripts.fetch_sc_lists            # download + parse all known
    python -m scripts.fetch_sc_lists --list     # just show what is known
    python -m scripts.fetch_sc_lists --no-parse # download only

## Why this matters more than it sounds

`core.universe.compliant_on()` needs to know who was Shariah-compliant **on
the day being simulated**. With one release on file, the only options were an
empty universe before November 2025, or `backfill_shariah=True` -- applying
the 2025 list backwards across ten years of history. That is lookahead: a
company that became compliant in 2025 is treated as holdable in 2019, when
you could not have held it.

Each release added here converts part of that backfilled window into real
point-in-time data. Six releases take the honest boundary from November 2025
back to November 2020.

## And it partially attacks survivorship

The November 2020 list contains AirAsia Group, AirAsia X and APFT. Those are
names that later collapsed, were suspended, or went PN17 -- and they are
completely absent from the 2025 list, because the 2025 list only knows about
companies that were still around in 2025.

Every backtest run on the 2025 list alone is therefore run on survivors. It
cannot see the stock that halved and never traded again, which is exactly the
outcome a one-stock contest portfolio is most exposed to. Older lists put
some of those companies back.

This does NOT fully fix survivorship -- price history for a delisted name
still has to come from somewhere, and Yahoo often drops it. It narrows the
gap and, more usefully, it makes the size of the gap measurable.

## The download is deliberately explicit

Each release is one PDF from the SC's own document endpoint, listed below by
date and id so there is no guessing about what is being fetched. They are
public regulatory publications. If a URL 404s the SC has rotated the id --
find the current one on the ICM publications page and add it to RELEASES.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

BASE = "https://www.sc.com.my/api/documentms/download.ashx?id={doc_id}"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; bursa-research/1.0)"}

# date (effective) -> SC document id. Verified by opening each and reading the
# effective date off the cover, not inferred from the id or the search result.
RELEASES: dict[str, str] = {
    "2020-11-27": "7b5e5a18-0108-48fe-8d2e-1f6bfc78f217",
    "2021-05-28": "48a2810f-415f-48e7-bfac-ff77dc473c0e",
    "2022-05-27": "f594c8f5-17a1-4007-9334-4db1330e705f",
    "2024-11-29": "1920c06f-61e5-46d7-8016-28a918acd4c8",
    "2025-05-30": "2671e073-8b4c-4291-af90-7cb34ad7715f",
    "2025-11-28": "5f0bb08b-802c-49a0-b093-d6f0edf6c276",
}

OUT_DIR = Path("data/sc_lists")


def download(date: str, doc_id: str, out_dir: Path,
             timeout: int = 90) -> Path | None:
    """Fetch one release. Returns the path, or None if it failed."""
    out = out_dir / f"sc_shariah_{date}.pdf"
    if out.exists() and out.stat().st_size > 100_000:
        print(f"  {date}  already downloaded ({out.stat().st_size:,} bytes)")
        return out
    url = BASE.format(doc_id=doc_id)
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
    except Exception as e:                                # noqa: BLE001
        print(f"  {date}  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return None

    # A rotated id or an error page returns HTML, not a PDF, and would
    # otherwise be saved as a .pdf that the parser then fails on obscurely.
    if not data.startswith(b"%PDF"):
        print(f"  {date}  FAILED: not a PDF ({len(data):,} bytes). The "
              f"document id has probably rotated.", file=sys.stderr)
        return None

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f"  {date}  downloaded ({len(data):,} bytes)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--list", action="store_true",
                    help="print the known releases and exit")
    ap.add_argument("--engine", choices=("pdftotext", "pdfplumber"),
                    default="pdftotext",
                    help="pdfplumber needs no external binary; use it if "
                         "pdftotext is not installed")
    ap.add_argument("--no-parse", action="store_true",
                    help="download only; do not append to data/shariah.csv")
    args = ap.parse_args()

    if args.list:
        print(f"{len(RELEASES)} known SC releases:")
        for d, i in sorted(RELEASES.items()):
            print(f"  {d}  {BASE.format(doc_id=i)}")
        return 0

    print(f"downloading {len(RELEASES)} SC releases to {args.out}/\n")
    got: dict[str, Path] = {}
    for date, doc_id in sorted(RELEASES.items()):
        p = download(date, doc_id, args.out)
        if p:
            got[date] = p

    print(f"\n  {len(got)}/{len(RELEASES)} available")
    if not got:
        return 1
    if args.no_parse:
        return 0

    print("\nparsing into data/shariah.csv ...")
    # parse() does the whole job: text extraction, slicing to the FULL list
    # (not the newly-compliant or newly-NON-compliant delta tables), and the
    # MIN_PLAUSIBLE refusal. Calling extract_codes directly would skip the
    # section slicing and could hand back a universe of forbidden stocks.
    from scripts.parse_shariah import parse, write_csv

    ok = 0
    for date, path in sorted(got.items()):
        try:
            codes = parse(path, engine=args.engine)
            write_csv(codes, date)
            print(f"  {date}  {len(codes)} codes")
            ok += 1
        except (Exception, SystemExit) as e:               # noqa: BLE001
            # One malformed release must not abandon the others. A partial
            # point-in-time history still beats a fully backfilled one.
            print(f"  {date}  PARSE FAILED: {type(e).__name__}: {e}",
                  file=sys.stderr)

    print(f"\n  {ok}/{len(got)} parsed. Now rebuild the screen:")
    print("    python -m scripts.build_panel")
    print("  and set backfill_shariah=False for everything from the earliest")
    print("  release onward -- that window is now real, not assumed.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
