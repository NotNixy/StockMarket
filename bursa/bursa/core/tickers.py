"""Candidate Bursa Malaysia stock codes.

This is a STARTING POINT, not a verified universe. Bursa has no tidy free API
for "all listings", so the list below is assembled from well-known large and
mid caps by stock code. Some entries may be wrong, renamed, merged or
delisted.

That is handled rather than hidden: `scripts/fetch_universe.py` attempts every
code and reports which returned data. Codes that fail are simply not in your
universe, and the script prints them so you can decide whether they were a bad
guess or a genuine delisting.

To replace this with the real thing, download Bursa's own listed-companies
list and save it as `data/bursa_listings.csv` with a `code` column;
`load_candidates()` prefers that file whenever it exists.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Large caps -- the index names. Highest confidence, most liquid, and where
# the Shariah screen bites hardest (conventional banks are excluded).
LARGE_CAP = [
    "1155",  # Malayan Banking
    "1023",  # CIMB Group
    "1066",  # RHB Bank
    "1015",  # AMMB Holdings
    "5819",  # Hong Leong Bank
    "1082",  # Hong Leong Financial Group
    "2488",  # Alliance Bank
    "5347",  # Tenaga Nasional
    "6012",  # Maxis
    "4863",  # Telekom Malaysia
    "6888",  # Axiata Group
    "1961",  # IOI Corporation
    "2445",  # Kuala Lumpur Kepong
    "5183",  # Petronas Chemicals
    "5681",  # Petronas Dagangan
    "6033",  # Petronas Gas
    "3816",  # MISC
    "5225",  # IHH Healthcare
    "4707",  # Nestle Malaysia
    "3182",  # Genting
    "4715",  # Genting Malaysia
    "4197",  # Sime Darby
    "8869",  # Press Metal Aluminium
    "6742",  # YTL Power International
    "4677",  # YTL Corporation
    "1818",  # Bursa Malaysia
    "5878",  # KPJ Healthcare
    "7084",  # QL Resources
    "5211",  # Sunway
    "3034",  # Hap Seng Consolidated
]

# Mid caps -- more volatile, which matters for a tournament, but the liquidity
# screen in core.universe will cut whatever cannot actually be traded.
MID_CAP = [
    "5168",  # Hartalega
    "7113",  # Top Glove
    "7106",  # Supermax
    "6399",  # Astro Malaysia
    "5139",  # AEON Credit Service
    "5099",  # Capital A
    "6947",  # CelcomDigi
    "5296",  # MR DIY
    "7033",  # Dufu Technology
    "0097",  # VITROX
    "5286",  # Inari Amertron
    "7160",  # Kossan Rubber
    "5008",  # Harrisons
    "0166",  # Inari (alt listing code, verified by fetch)
    "6963",  # Eco World
    "5235",  # KLCCP Stapled
    "1171",  # MBSB
    "5158",  # Ann Joo Resources
    "2828",  # Ancom Nylex
    "7277",  # Dialog Group
    "0208",  # Frontken
    "5075",  # Plenitude
    "5202",  # Integrated Logistics
    "7153",  # Kossan alt
    "4006",  # Oriental Holdings
    "3689",  # Fraser & Neave
    "6351",  # AMWAY Malaysia
    "5031",  # Time dotCom
    "0128",  # Frontken alt
    "5027",  # Kim Loong Resources
]

CANDIDATES = LARGE_CAP + MID_CAP

LISTINGS_CSV = Path("data/bursa_listings.csv")


def load_candidates(path: Path | str = LISTINGS_CSV) -> list[str]:
    """Candidate stock codes, preferring a real Bursa listings file if present.

    The CSV needs one column named `code`. Anything else is ignored, so a
    direct download from Bursa with extra columns works unmodified.
    """
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path, dtype=str)
        if "code" not in df.columns:
            raise ValueError(f"{path}: expected a 'code' column, got "
                             f"{list(df.columns)}")
        codes = (df["code"].astype(str).str.strip().str.zfill(4)
                 .dropna().unique().tolist())
        return sorted(codes)
    return sorted(set(CANDIDATES))


def save_verified(codes: list[str], path: Path | str = "data/universe.csv") -> Path:
    """Write the codes that actually returned data.

    This file IS committed -- it is the record of which tickers your results
    were computed against, and regenerating it later would silently change the
    universe under old results.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"code": sorted(set(codes))}).to_csv(path, index=False)
    return path
