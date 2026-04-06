"""
ingest.py — Raw data ingestion for the Eligibility Gap AI pipeline.

Pulls all data from the US Census Bureau ACS 5-Year API and saves raw CSVs
to data/raw/. Ten tables are fetched for all US counties:

  B19001 — Household income brackets (SNAP eligibility proxy)
  B11001 — Household type (household count denominator)
  B22001 — SNAP/food stamp receipt (actual enrollment at county level)
  C16002 — Household LEP status (language barrier signal)
  B05001 — Citizenship status (documentation barrier signal)
  B15003 — Educational attainment (awareness barrier signal)
  B01001 — Age by sex (senior_share, child_share demographics)
  B28002 — Internet access (no_broadband_share — digital barrier signal)
  C18130 — Age by disability by poverty status (disability signals)
  B17001 — Poverty by sex by age (poverty_rate, child/senior poverty rates)
  NAME   — County display name (baked into pipeline output)

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

# B01001: Sex by Age
#   Used to derive senior_share (65+) and child_share (under 18) per county.
#   These demographic signals sharpen outreach targeting for disability/poverty.
#   Male: _003E (under 5), _004E (5-9), _005E (10-14), _006E (15-17)
#         _020E (65-66), _021E (67-69), _022E (70-74), _023E (75-79), _024E (80-84), _025E (85+)
#   Female: _027E (under 5), _028E (5-9), _029E (10-14), _030E (15-17)
#           _044E (65-66), _045E (67-69), _046E (70-74), _047E (75-79), _048E (80-84), _049E (85+)
AGE_VARS = (
    "B01001_001E,"  # total population
    # Male under 18
    "B01001_003E,"  # M under 5
    "B01001_004E,"  # M 5-9
    "B01001_005E,"  # M 10-14
    "B01001_006E,"  # M 15-17
    # Male 65+
    "B01001_020E,"  # M 65-66
    "B01001_021E,"  # M 67-69
    "B01001_022E,"  # M 70-74
    "B01001_023E,"  # M 75-79
    "B01001_024E,"  # M 80-84
    "B01001_025E,"  # M 85+
    # Female under 18
    "B01001_027E,"  # F under 5
    "B01001_028E,"  # F 5-9
    "B01001_029E,"  # F 10-14
    "B01001_030E,"  # F 15-17
    # Female 65+
    "B01001_044E,"  # F 65-66
    "B01001_045E,"  # F 67-69
    "B01001_046E,"  # F 70-74
    "B01001_047E,"  # F 75-79
    "B01001_048E,"  # F 80-84
    "B01001_049E"   # F 85+
)

# B28002: Presence and Types of Internet Subscriptions in Household
#   _001E = total households
#   _013E = no internet access at all
#   no_broadband_share signals a digital access barrier for online applications.
INTERNET_VARS = (
    "B28002_001E,"  # total households
    "B28002_013E"   # no internet access
)

# C18130: Age by Disability Status by Poverty Status
#   Civilian noninstitutionalized population. Used to derive disability_share
#   and disability_poverty_share — populations with both barriers simultaneously.
#   Under 18: _002E (total), _003E (with disability), _004E (disability + below poverty)
#   18 to 64: _009E (total), _010E (with disability), _011E (disability + below poverty)
#   65+:      _016E (total), _017E (with disability), _018E (disability + below poverty)
DISABILITY_VARS = (
    "C18130_001E,"  # total
    "C18130_002E,"  # under 18 total
    "C18130_003E,"  # under 18, with disability
    "C18130_004E,"  # under 18, disability + below poverty
    "C18130_009E,"  # 18-64 total
    "C18130_010E,"  # 18-64, with disability
    "C18130_011E,"  # 18-64, disability + below poverty
    "C18130_016E,"  # 65+ total
    "C18130_017E,"  # 65+, with disability
    "C18130_018E"   # 65+, disability + below poverty
)

# B17001: Poverty Status in the Past 12 Months by Sex by Age
#   _001E = universe (total for whom poverty is determined)
#   _002E = total below poverty
#   Children below poverty: _004E (M<5) + _005E (M 5-17) + _014E (F<5) + _015E (F 5-17)
#   Seniors below poverty:  _011E (M 65-74) + _012E (M 75+) + _021E (F 65-74) + _022E (F 75+)
POVERTY_AGE_VARS = (
    "B17001_001E,"  # total universe
    "B17001_002E,"  # total below poverty
    "B17001_004E,"  # M under 5, below poverty
    "B17001_005E,"  # M 5-17, below poverty
    "B17001_011E,"  # M 65-74, below poverty
    "B17001_012E,"  # M 75+, below poverty
    "B17001_014E,"  # F under 5, below poverty
    "B17001_015E,"  # F 5-17, below poverty
    "B17001_021E,"  # F 65-74, below poverty
    "B17001_022E"   # F 75+, below poverty
)

# C17002: Ratio of Income to Poverty Level in the Past 12 Months
#   Person-level poverty ratio distribution used to estimate households at ≤130% FPL
#   (the SNAP gross income limit), replacing the cruder B19001 income-bracket approach.
#   130% FPL falls within the 1.25–1.49 band (_005E); we interpolate to 21% of that band.
#   _001E = total persons (universe)
#   _002E = under .50 (deep poverty)
#   _003E = .50 to .99
#   _004E = 1.00 to 1.24
#   _005E = 1.25 to 1.49  ← SNAP cutoff (130% FPL) is 21% into this band
POVERTY_RATIO_VARS = (
    "C17002_001E,"  # total
    "C17002_002E,"  # under .50
    "C17002_003E,"  # .50 to .99
    "C17002_004E,"  # 1.00 to 1.24
    "C17002_005E"   # 1.25 to 1.49
)

# B19083: Gini Index of Income Inequality
#   Direct measure of income inequality within a county (0 = perfect equality, 1 = maximum).
#   Replaces the cruder high_income_share proxy for the stigma barrier signal.
#   High Gini + low enrollment = structural inequality suppressing benefit access.
GINI_VARS = "B19083_001E"

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


def fetch_acs_poverty_ratio(api_key: str) -> pd.DataFrame:
    """Pull C17002 (ratio of income to poverty level) for all US counties."""
    print(f"[Census] Fetching C17002 (poverty ratio) — {ACS_YEAR} ACS 5-Year...")
    df = _census_get(ACS_DATASET, ACS_YEAR, POVERTY_RATIO_VARS, "county:*", api_key)
    print(f"  → {len(df):,} counties")
    return df


def fetch_acs_gini(api_key: str) -> pd.DataFrame:
    """Pull B19083 (Gini index of income inequality) for all US counties."""
    print(f"[Census] Fetching B19083 (Gini index) — {ACS_YEAR} ACS 5-Year...")
    df = _census_get(ACS_DATASET, ACS_YEAR, GINI_VARS, "county:*", api_key)
    print(f"  → {len(df):,} counties")
    return df


def fetch_county_area(api_key: str) -> pd.DataFrame:
    """
    Fetch county land area (ALAND, sq meters) from the 2020 Decennial Census PL file.

    Land area is a fixed geographic attribute — 2020 values are current for our purposes.
    Used to compute population density, the direct rurality signal for the awareness barrier.
    """
    print("[Census] Fetching county land area (ALAND) — 2020 Decennial Census...")
    url = f"{CENSUS_BASE}/2020/dec/pl"
    params = {"get": "ALAND", "for": "county:*", "key": api_key}
    resp = requests.get(url, params=params, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Census API error {resp.status_code} for {url}\nBody: {resp.text[:500]}"
        )
    data = resp.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    print(f"  → {len(df):,} counties")
    return df


def fetch_acs_snap_prior(api_key: str) -> pd.DataFrame:
    """
    Pull B22001 (SNAP receipt) for the prior ACS year (2022) to compute enrollment trend.

    Comparing 2022 → 2023 enrollment reveals whether a county's gap is growing or shrinking,
    which modifies priority score: counties moving in the wrong direction rank higher.
    """
    print(f"[Census] Fetching B22001 (SNAP receipt) — 2022 ACS 5-Year (prior year)...")
    df = _census_get(ACS_DATASET, 2022, SNAP_RECEIPT_VARS, "county:*", api_key)
    print(f"  → {len(df):,} counties")
    return df


def fetch_acs_age(api_key: str) -> pd.DataFrame:
    """Pull B01001 (sex by age) for all US counties."""
    print(f"[Census] Fetching B01001 (age by sex) — {ACS_YEAR} ACS 5-Year...")
    df = _census_get(ACS_DATASET, ACS_YEAR, AGE_VARS, "county:*", api_key)
    print(f"  → {len(df):,} counties")
    return df


def fetch_acs_internet(api_key: str) -> pd.DataFrame:
    """Pull B28002 (internet access) for all US counties."""
    print(f"[Census] Fetching B28002 (internet access) — {ACS_YEAR} ACS 5-Year...")
    df = _census_get(ACS_DATASET, ACS_YEAR, INTERNET_VARS, "county:*", api_key)
    print(f"  → {len(df):,} counties")
    return df


def fetch_acs_disability(api_key: str) -> pd.DataFrame:
    """Pull C18130 (age by disability by poverty) for all US counties."""
    print(f"[Census] Fetching C18130 (disability × poverty) — {ACS_YEAR} ACS 5-Year...")
    df = _census_get(ACS_DATASET, ACS_YEAR, DISABILITY_VARS, "county:*", api_key)
    print(f"  → {len(df):,} counties")
    return df


def fetch_acs_poverty_age(api_key: str) -> pd.DataFrame:
    """Pull B17001 (poverty by sex by age) for all US counties."""
    print(f"[Census] Fetching B17001 (poverty by age) — {ACS_YEAR} ACS 5-Year...")
    df = _census_get(ACS_DATASET, ACS_YEAR, POVERTY_AGE_VARS, "county:*", api_key)
    print(f"  → {len(df):,} counties")
    return df


def fetch_county_names(api_key: str) -> pd.DataFrame:
    """
    Fetch the NAME variable for all US counties.

    Returns a DataFrame with columns [NAME, state, county] where NAME is a
    human-readable string like "Autauga County, Alabama". Baking this into
    the pipeline avoids a runtime API call from the dashboard.
    """
    print(f"[Census] Fetching county names (NAME) — {ACS_YEAR} ACS 5-Year...")
    df = _census_get(ACS_DATASET, ACS_YEAR, "NAME", "county:*", api_key)
    print(f"  → {len(df):,} counties")
    return df


# ---------------------------------------------------------------------------
# Save all tables
# ---------------------------------------------------------------------------

def save_acs_raw(api_key: str) -> None:
    """
    Fetch all ACS tables and save raw CSVs to data/raw/.

    Output files:
      acs_b19001_county_2023.csv  — income brackets (eligibility proxy)
      acs_b11001_county_2023.csv  — household type (denominator check)
      acs_b22001_county_2023.csv  — SNAP receipt (enrollment)
      acs_c16002_county_2023.csv  — household LEP status (language barrier)
      acs_b05001_county_2023.csv  — citizenship status (documentation barrier)
      acs_b15003_county_2023.csv  — educational attainment (awareness barrier)
      acs_b01001_county_2023.csv  — age by sex (senior/child demographics)
      acs_b28002_county_2023.csv  — internet access (digital barrier)
      acs_c18130_county_2023.csv  — disability × poverty
      acs_b17001_county_2023.csv  — poverty by age (child/senior poverty rates)
      acs_names_county_2023.csv   — county display names (baked in)
    """
    fetchers = [
        (fetch_acs_income,       f"acs_b19001_county_{ACS_YEAR}.csv"),
        (fetch_acs_households,   f"acs_b11001_county_{ACS_YEAR}.csv"),
        (fetch_acs_snap_receipt, f"acs_b22001_county_{ACS_YEAR}.csv"),
        (fetch_acs_language,     f"acs_c16002_county_{ACS_YEAR}.csv"),
        (fetch_acs_citizenship,  f"acs_b05001_county_{ACS_YEAR}.csv"),
        (fetch_acs_education,    f"acs_b15003_county_{ACS_YEAR}.csv"),
        (fetch_acs_age,          f"acs_b01001_county_{ACS_YEAR}.csv"),
        (fetch_acs_internet,     f"acs_b28002_county_{ACS_YEAR}.csv"),
        (fetch_acs_disability,   f"acs_c18130_county_{ACS_YEAR}.csv"),
        (fetch_acs_poverty_age,  f"acs_b17001_county_{ACS_YEAR}.csv"),
        (fetch_county_names,     f"acs_names_county_{ACS_YEAR}.csv"),
        (fetch_acs_poverty_ratio, f"acs_c17002_county_{ACS_YEAR}.csv"),
        (fetch_acs_gini,          f"acs_b19083_county_{ACS_YEAR}.csv"),
        (fetch_acs_snap_prior,    "acs_b22001_county_2022.csv"),
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
