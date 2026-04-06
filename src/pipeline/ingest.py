"""
ingest.py — Raw data ingestion for the Eligibility Gap AI pipeline.

Pulls all data from the US Census Bureau ACS 5-Year API and saves raw CSVs
to data/raw/. Four tables are fetched for all US counties:

  B19001 — Household income brackets (SNAP eligibility proxy)
  B11001 — Household type (household count denominator)
  B22001 — SNAP/food stamp receipt (actual enrollment at county level)
  B16004 — Language spoken at home + English ability (LEP barrier signal)

Why Census-only for v1:
  The USDA FNS website (fns.usda.gov) blocks all programmatic access via CDN
  bot protection. Census ACS table B22001 tracks self-reported SNAP receipt at
  the county level, giving us enrollment data from the same source and vintage
  as our eligibility inputs — no external dependency needed.

v2 extension points:
  - FNS county participation (administrative enrollment, more precise):
      Implement fetch_fns_county() following the same pattern and call it from
      run_all(). Drop-in replacement for B22001-derived enrollment.
  - FNS QC denial-reason data (state-level barrier signals):
      Implement fetch_fns_qc(). clean.py already has a join hook for this.
  - FOIA state admin data:
      Implement fetch_foia_state(). Nothing else in the pipeline changes.

Requires:
  CENSUS_API_KEY in environment (or .env file).
  Free key: https://api.census.gov/data/key_signup.html
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Census ACS configuration
# ---------------------------------------------------------------------------

CENSUS_BASE = "https://api.census.gov/data"
ACS_YEAR = 2023
ACS_DATASET = "acs/acs5"

# B19001: Household income in the past 12 months (2023 inflation-adjusted $)
#   _001E = total households (universe)
#   _002E–_017E = count of households in each income bracket
#   Used in clean.py to estimate households below 130% FPL (SNAP income limit)
INCOME_VARS = (
    "B19001_001E,"  # total households
    "B19001_002E,"  # < $10,000
    "B19001_003E,"  # $10,000–$14,999
    "B19001_004E,"  # $15,000–$19,999
    "B19001_005E,"  # $20,000–$24,999
    "B19001_006E,"  # $25,000–$29,999
    "B19001_007E,"  # $30,000–$34,999
    "B19001_008E,"  # $35,000–$39,999
    "B19001_009E,"  # $40,000–$44,999
    "B19001_010E,"  # $45,000–$49,999
    "B19001_011E,"  # $50,000–$59,999
    "B19001_012E,"  # $60,000–$74,999
    "B19001_013E,"  # $75,000–$99,999
    "B19001_014E,"  # $100,000–$124,999
    "B19001_015E,"  # $125,000–$149,999
    "B19001_016E,"  # $150,000–$199,999
    "B19001_017E"   # $200,000+
)

# B11001: Household type
#   _001E = total households (sanity-check denominator vs B19001_001E)
HOUSEHOLD_VARS = "B11001_001E"

# B22001: Receipt of food stamps/SNAP in the past 12 months
#   _001E = total households (universe)
#   _002E = households that received SNAP  ← this is our enrollment numerator
#   _003E = households that did not receive SNAP
SNAP_RECEIPT_VARS = (
    "B22001_001E,"  # total households
    "B22001_002E,"  # received SNAP
    "B22001_003E"   # did not receive SNAP
)

# C16002: Household Language by Household Limited English Speaking Status
#   Simpler and more accurate than B16004 for our purposes — directly gives
#   "limited English speaking households" at the household level, matching our
#   other tables. B16004 has a complex multi-age-group structure with subtotals
#   that makes variable mapping error-prone.
#   _001E = total households
#   _004E = Spanish-speaking, limited English speaking household
#   _007E = Other Indo-European, limited English speaking household
#   _010E = Asian and Pacific Islander, limited English speaking household
#   _013E = Other languages, limited English speaking household
LANGUAGE_VARS = (
    "C16002_001E,"  # total households
    "C16002_004E,"  # Spanish – limited English speaking
    "C16002_007E,"  # Other Indo-European – limited English speaking
    "C16002_010E,"  # Asian and Pacific Islander – limited English speaking
    "C16002_013E"   # Other languages – limited English speaking
)

# B05001: Citizenship Status
#   Used to derive the foreign-born non-citizen share per county — the primary
#   "documentation barrier" signal (fear of benefit use affecting immigration
#   status, limited access to application assistance).
#   _001E = total population
#   _006E = foreign born, not a U.S. citizen
CITIZENSHIP_VARS = (
    "B05001_001E,"  # total population
    "B05001_006E"   # foreign born, not a U.S. citizen
)

# B15003: Educational Attainment for Population 25 Years and Over
#   Used to derive the share without a high school diploma — one of two signals
#   for the "awareness barrier" (alongside rurality, proxied by county size).
#   _001E = total population 25+
#   _002E–_016E = no schooling through 12th grade without diploma (all < HS)
#   _017E = regular high school diploma (first level with credential)
EDUCATION_VARS = (
    "B15003_001E,"  # total pop 25+
    "B15003_002E,"  # no schooling completed
    "B15003_003E,"  # nursery school
    "B15003_004E,"  # kindergarten
    "B15003_005E,"  # 1st grade
    "B15003_006E,"  # 2nd grade
    "B15003_007E,"  # 3rd grade
    "B15003_008E,"  # 4th grade
    "B15003_009E,"  # 5th grade
    "B15003_010E,"  # 6th grade
    "B15003_011E,"  # 7th grade
    "B15003_012E,"  # 8th grade
    "B15003_013E,"  # 9th grade
    "B15003_014E,"  # 10th grade
    "B15003_015E,"  # 11th grade
    "B15003_016E"   # 12th grade, no diploma
)

# Map each ACS table to its variable string and output filename
ACS_TABLES = [
    ("B19001", INCOME_VARS,       f"acs_b19001_county_{ACS_YEAR}.csv"),
    ("B11001", HOUSEHOLD_VARS,    f"acs_b11001_county_{ACS_YEAR}.csv"),
    ("B22001", SNAP_RECEIPT_VARS, f"acs_b22001_county_{ACS_YEAR}.csv"),
    ("B16004", LANGUAGE_VARS,     f"acs_b16004_county_{ACS_YEAR}.csv"),
]

# ---------------------------------------------------------------------------
# Census API helper
# ---------------------------------------------------------------------------

def _census_get(dataset: str, year: int, get: str, for_geo: str, api_key: str) -> pd.DataFrame:
    """
    Make one Census API request and return a DataFrame.

    The API returns a JSON array where row 0 is the header and rows 1-N are data.
    Raises RuntimeError on non-200 responses.
    """
    url = f"{CENSUS_BASE}/{year}/{dataset}"
    params = {
        "get": get,
        "for": for_geo,
        "key": api_key,
    }
    resp = requests.get(url, params=params, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Census API error {resp.status_code} for {url}\n"
            f"Params: {params}\nBody: {resp.text[:500]}"
        )
    data = resp.json()
    # Row 0 is the header; rows 1–N are county records
    return pd.DataFrame(data[1:], columns=data[0])


# ---------------------------------------------------------------------------
# Per-table fetch functions
# ---------------------------------------------------------------------------

def fetch_acs_income(api_key: str) -> pd.DataFrame:
    """Pull B19001 (household income brackets) for all US counties."""
    print(f"[Census] Fetching B19001 (household income) — {ACS_YEAR} ACS 5-Year...")
    df = _census_get(ACS_DATASET, ACS_YEAR, INCOME_VARS, "county:*", api_key)
    print(f"  → {len(df):,} counties")
    return df


def fetch_acs_households(api_key: str) -> pd.DataFrame:
    """Pull B11001 (household type totals) for all US counties."""
    print(f"[Census] Fetching B11001 (household type) — {ACS_YEAR} ACS 5-Year...")
    df = _census_get(ACS_DATASET, ACS_YEAR, HOUSEHOLD_VARS, "county:*", api_key)
    print(f"  → {len(df):,} counties")
    return df


def fetch_acs_snap_receipt(api_key: str) -> pd.DataFrame:
    """
    Pull B22001 (SNAP/food stamp receipt) for all US counties.

    B22001_002E is the enrolled household count — our enrollment numerator.
    This replaces FNS administrative data for v1 (FNS site blocks all
    programmatic access). Self-reported, same ACS vintage as eligibility inputs.
    """
    print(f"[Census] Fetching B22001 (SNAP receipt) — {ACS_YEAR} ACS 5-Year...")
    df = _census_get(ACS_DATASET, ACS_YEAR, SNAP_RECEIPT_VARS, "county:*", api_key)
    print(f"  → {len(df):,} counties")
    return df


def fetch_acs_language(api_key: str) -> pd.DataFrame:
    """Pull C16002 (household LEP status) for all US counties."""
    print(f"[Census] Fetching C16002 (household LEP status) — {ACS_YEAR} ACS 5-Year...")
    df = _census_get(ACS_DATASET, ACS_YEAR, LANGUAGE_VARS, "county:*", api_key)
    print(f"  → {len(df):,} counties")
    return df


def fetch_acs_citizenship(api_key: str) -> pd.DataFrame:
    """Pull B05001 (citizenship status) for all US counties."""
    print(f"[Census] Fetching B05001 (citizenship status) — {ACS_YEAR} ACS 5-Year...")
    df = _census_get(ACS_DATASET, ACS_YEAR, CITIZENSHIP_VARS, "county:*", api_key)
    print(f"  → {len(df):,} counties")
    return df


def fetch_acs_education(api_key: str) -> pd.DataFrame:
    """Pull B15003 (educational attainment 25+) for all US counties."""
    print(f"[Census] Fetching B15003 (educational attainment) — {ACS_YEAR} ACS 5-Year...")
    df = _census_get(ACS_DATASET, ACS_YEAR, EDUCATION_VARS, "county:*", api_key)
    print(f"  → {len(df):,} counties")
    return df


# ---------------------------------------------------------------------------
# Save all tables
# ---------------------------------------------------------------------------

def save_acs_raw(api_key: str) -> None:
    """
    Fetch all six ACS tables and save raw CSVs to data/raw/.

    Output files:
      acs_b19001_county_2023.csv  — income brackets (eligibility proxy)
      acs_b11001_county_2023.csv  — household type (denominator check)
      acs_b22001_county_2023.csv  — SNAP receipt (enrollment)
      acs_c16002_county_2023.csv  — household LEP status (language barrier)
      acs_b05001_county_2023.csv  — citizenship status (documentation barrier)
      acs_b15003_county_2023.csv  — educational attainment (awareness barrier)
    """
    fetchers = [
        (fetch_acs_income,       f"acs_b19001_county_{ACS_YEAR}.csv"),
        (fetch_acs_households,   f"acs_b11001_county_{ACS_YEAR}.csv"),
        (fetch_acs_snap_receipt, f"acs_b22001_county_{ACS_YEAR}.csv"),
        (fetch_acs_language,     f"acs_c16002_county_{ACS_YEAR}.csv"),
        (fetch_acs_citizenship,  f"acs_b05001_county_{ACS_YEAR}.csv"),
        (fetch_acs_education,    f"acs_b15003_county_{ACS_YEAR}.csv"),
    ]
    for i, (fetcher, filename) in enumerate(fetchers):
        path = RAW_DIR / filename
        if path.exists():
            print(f"[Census] Skipping {filename} (already exists)")
            continue
        df = fetcher(api_key)
        df.to_csv(path, index=False)
        print(f"  Saved → {path}")
        # Pause between requests — Census API rate limit is generous but not unlimited
        if i < len(fetchers) - 1:
            time.sleep(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all() -> None:
    """
    Run all ingestion steps.

    Currently: Census ACS only (4 tables).

    v2 extension points (add calls here, nothing else changes):
      fetch_fns_county()  — FNS administrative enrollment (if site access restored)
      fetch_fns_qc()      — FNS state-level QC denial data
      fetch_foia_state()  — state SNAP admin data from FOIA requests
    """
    api_key = os.getenv("CENSUS_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "CENSUS_API_KEY is not set.\n"
            "Get a free key at https://api.census.gov/data/key_signup.html\n"
            "Then add CENSUS_API_KEY=<your_key> to your .env file."
        )

    print("=== Eligibility Gap AI — Raw Ingestion ===\n")
    save_acs_raw(api_key)
    print("\n=== Ingestion complete. Raw files in data/raw/ ===")


if __name__ == "__main__":
    run_all()
