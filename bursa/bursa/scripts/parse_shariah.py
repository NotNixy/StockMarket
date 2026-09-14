"""Parse the SC's List of Shariah-Compliant Securities PDF into a CSV.

    python -m scripts.parse_shariah data/sc_shariah_2025-11-28.pdf --date 2025-11-28

Writes/appends `data/shariah.csv` in the schema core.universe expects:

    list_date,ticker
    2025-11-28,0365
    ...

Run it once per SC release. Each release is a COMPLETE snapshot, so appending
several builds the point-in-time history that makes the backtest honest.

## The trap this file exists to avoid

The PDF contains three separate tables, and picking the wrong one is silent
and catastrophic:

  Table 1  securities NEWLY classified as compliant this release  (49 names)
  Table 2  securities newly classified as NON-compliant          (36 names)
  "LIST OF SHARIAH-COMPLIANT SECURITIES"  the FULL list          (~900 names)

Parsing Table 1 gives you 49 stocks and a universe that looks plausible.
Parsing Table 2 gives you a portfolio of exactly the stocks you must not hold.
So the parser anchors on the full-list heading and refuses to emit anything
that looks like it came from the wrong table.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

# The heading that begins the FULL list. Everything before it is commentary,
# methodology, and the two delta tables.
FULL_LIST_START = re.compile(r"LIST OF SHARIAH-COMPLIANT SECURITIES\s*[-–—]\s*\w+",
                             re.IGNORECASE)
# The appendix after the equities, which lists funds and other instruments.
FULL_LIST_END = re.compile(r"ADDITIONAL LIST", re.IGNORECASE)

# "  12.   5183   Petronas Chemicals Group Bhd"  -- appears twice per line,
# because the PDF lays the list out in two columns. Codes are 4 OR 5 digits;
# ACE market codes like 03065 lose meaning if truncated to four.
ENTRY = re.compile(r"(\d{1,3})\.\s+(\d{4,5})\s+([A-Za-z0-9][^\d]{2,60}?)(?=\s{2,}|$)")

# A real release has hundreds of names. Anything near 49 or 36 means the
# wrong table was parsed.
MIN_PLAUSIBLE = 300


def pdf_to_text(pdf: Path) -> str:
    """Extract text with layout preserved, so the two columns stay separable."""
    try:
        out = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                             capture_output=True, text=True, check=True)
        return out.stdout
    except FileNotFoundError:
        raise SystemExit(
            "pdftotext not found. Install poppler-utils, or on Windows:\n"
            "  winget install oschwartz10612.Poppler\n"
            "Alternatively pip install pdfplumber and use --engine pdfplumber")


def pdf_to_text_pdfplumber(pdf: Path) -> str:
    """Fallback that needs no external binary."""
    try:
        import pdfplumber
    except ImportError:
        raise SystemExit("pip install pdfplumber")
    with pdfplumber.open(pdf) as doc:
        return "\n".join((p.extract_text(layout=True) or "") for p in doc.pages)


def slice_full_list(text: str) -> str:
    """Keep only the section between the full-list heading and the appendix."""
    start = FULL_LIST_START.search(text)
    if not start:
        raise SystemExit(
            "Could not find the 'LIST OF SHARIAH-COMPLIANT SECURITIES' heading.\n"
            "Refusing to parse: without it there is no way to tell the full "
            "list from\nthe newly-compliant and newly-NON-compliant tables, and "
            "guessing wrong\nwould hand you a universe of forbidden stocks.")
    body = text[start.end():]
    end = FULL_LIST_END.search(body)
    return body[:end.start()] if end else body


def extract_codes(section: str) -> list[str]:
    """Every stock code in the section, de-duplicated, order preserved."""
    seen, codes = set(), []
    for line in section.splitlines():
        for _, code, _name in ENTRY.findall(line):
            if code not in seen:
                seen.add(code)
                codes.append(code)
    return codes


def parse(pdf: Path, engine: str = "pdftotext") -> list[str]:
    text = (pdf_to_text(pdf) if engine == "pdftotext"
            else pdf_to_text_pdfplumber(pdf))
    codes = extract_codes(slice_full_list(text))

    if len(codes) < MIN_PLAUSIBLE:
        raise SystemExit(
            f"Only {len(codes)} codes found. A real SC release lists several "
            f"hundred.\nThis usually means one of the delta tables was parsed "
            f"instead of the\nfull list -- Table 1 (newly compliant) or Table 2 "
            f"(newly NON-compliant).\nRefusing to write a universe that might be "
            f"the securities you must avoid.")
    return codes


def write_csv(codes: list[str], list_date: str,
              out: Path = Path("data/shariah.csv")) -> Path:
    """Append this release. Re-running the same date replaces it, not duplicates."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fresh = pd.DataFrame({"list_date": list_date, "ticker": sorted(set(codes))})

    if out.exists():
        old = pd.read_csv(out, dtype=str)
        old = old[old["list_date"] != list_date]        # idempotent re-run
        fresh = pd.concat([old, fresh], ignore_index=True)

    fresh = fresh.sort_values(["list_date", "ticker"]).drop_duplicates()
    fresh.to_csv(out, index=False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", type=Path, help="the SC release PDF")
    ap.add_argument("--date", required=True,
                    help="effective date of THIS release, YYYY-MM-DD "
                         "(printed on the PDF cover, e.g. 28 NOV 2025)")
    ap.add_argument("--engine", choices=("pdftotext", "pdfplumber"),
                    default="pdftotext")
    ap.add_argument("--out", type=Path, default=Path("data/shariah.csv"))
    args = ap.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"{args.pdf} not found")

    codes = parse(args.pdf, args.engine)
    path = write_csv(codes, args.date, args.out)

    table = pd.read_csv(path, dtype=str)
    print(f"parsed {len(codes)} Shariah-compliant codes from {args.pdf.name}")
    print(f"  effective {args.date}")
    print(f"  sample: {', '.join(codes[:8])} ...")
    print(f"\nwritten to {path}")
    print(f"  releases now held: "
          f"{', '.join(sorted(table['list_date'].unique()))}")
    print(f"  total rows: {len(table):,}")
    print("\n  This file IS committed -- it cannot be regenerated from an API,")
    print("  and re-deriving it later would silently change the universe under")
    print("  results you have already recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
