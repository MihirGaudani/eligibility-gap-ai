"""
clean.py — Merge, clean, and feature-engineer six raw ACS tables into
a single county-level analytical dataset saved to data/processed/counties.csv.

Steps:
  1. Load all six raw CSVs and build a 5-digit county FIPS key
  2. Merge into one row per county (inner join — drops territories)
  3. Cast all ACS estimate columns to numeric
  4. Estimate SNAP-eligible households from B19001 income brackets
  5. Extract SNAP enrollment from B22001
  6. Compute the eligibility gap (eligible minus enrolled)
  7. Derive barrier signals for the classifier:
       lep_share          — language barrier (C16002)
       noncitizen_share   — documentation barrier (B05001)
       no_hs_share        — educational attainment component of awareness (B15003)
       high_income_share  — income component of stigma signal (B19001)
       poverty_share      — poverty depth component of stigma signal (B19001)
  8. Save to data/processed/counties.csv

Eligibility estimation (simple proxy):
  SNAP gross income limit ≈ 130% FPL. For an average US household (~2.5 persons),
  130% FPL ≈ $33,000. We apply:
    B19001_002–006 (< $30k): 100% eligible
    B19001_007 ($30–35k):     50% eligible (straddles the cutoff)
    B19001_008+ ($35k+):       0% eligible
  Defensible for county ranking; refine with B17001 in v2 if needed.

v2 FNS hook:
  Replace enrolled_hh with FNS administrative count when available.
  The gap and rate calculations only depend on column names.
"""

import pandas as pd
from pathlib import Path

RAW_DIR       = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ACS_YEAR = 2023


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _load_raw(filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found: {path}\n"
            f"Run src/pipeline/ingest.py first."
        )
    return pd.read_csv(path, dtype=str)


def load_raw_tables() -> tuple:
    """Load all six raw ACS CSVs."""
    print("[Clean] Loading raw tables...")
    income      = _load_raw(f"acs_b19001_county_{ACS_YEAR}.csv")
    hh          = _load_raw(f"acs_b11001_county_{ACS_YEAR}.csv")
    snap        = _load_raw(f"acs_b22001_county_{ACS_YEAR}.csv")
    language    = _load_raw(f"acs_c16002_county_{ACS_YEAR}.csv")
    citizenship = _load_raw(f"acs_b05001_county_{ACS_YEAR}.csv")
    education   = _load_raw(f"acs_b15003_county_{ACS_YEAR}.csv")
    print(
        f"  B19001: {len(income):,} | B11001: {len(hh):,} | B22001: {len(snap):,} | "
        f"C16002: {len(language):,} | B05001: {len(citizenship):,} | B15003: {len(education):,}"
    )
    return income, hh, snap, language, citizenship, education


# ---------------------------------------------------------------------------
# FIPS key + merge
# ---------------------------------------------------------------------------

def _add_fips(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["county_fips"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)
    return df


# B15003 education columns for grades 1–12 no diploma (all < HS diploma)
NO_HS_COLS = [f"B15003_{str(i).zfill(3)}E" for i in range(2, 17)]  # _002E–_016E


def merge_tables(income, hh, snap, language, citizenship, education) -> pd.DataFrame:
    """
    Merge all six tables on county_fips. Inner join drops territories and
    counties missing from any table.
    """
    print("[Clean] Merging tables on county_fips...")
    for df in (income, hh, snap, language, citizenship, education):
        _add_fips(df)  # modifies copy — we re-assign below
    income      = _add_fips(income)
    hh          = _add_fips(hh)
    snap        = _add_fips(snap)
    language    = _add_fips(language)
    citizenship = _add_fips(citizenship)
    education   = _add_fips(education)

    df = (
        income
        .merge(hh[["county_fips", "B11001_001E"]], on="county_fips", how="inner")
        .merge(
            snap[["county_fips", "B22001_001E", "B22001_002E", "B22001_003E"]],
            on="county_fips", how="inner"
        )
        .merge(
            language[["county_fips", "C16002_001E", "C16002_004E",
                       "C16002_007E", "C16002_010E", "C16002_013E"]],
            on="county_fips", how="inner"
        )
        .merge(
            citizenship[["county_fips", "B05001_001E", "B05001_006E"]],
            on="county_fips", how="inner"
        )
        .merge(
            education[["county_fips", "B15003_001E"] + NO_HS_COLS],
            on="county_fips", how="inner"
        )
    )
    print(f"  → {len(df):,} counties after merge")
    return df


# ---------------------------------------------------------------------------
# Cast to numeric
# ---------------------------------------------------------------------------

def cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Cast all ACS estimate columns to numeric. Census encodes missing as -666666666."""
    print("[Clean] Casting ACS estimates to numeric...")
    est_cols = [c for c in df.columns if c[0] in ("B", "C") and c.endswith("E")]
    df = df.copy()
    df[est_cols] = (
        df[est_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace(-666666666, pd.NA)
    )
    return df


# ---------------------------------------------------------------------------
# Eligibility estimate (B19001)
# ---------------------------------------------------------------------------

FULLY_ELIGIBLE_BRACKETS = [
    "B19001_002E",  # < $10,000
    "B19001_003E",  # $10,000–$14,999
    "B19001_004E",  # $15,000–$19,999
    "B19001_005E",  # $20,000–$24,999
    "B19001_006E",  # $25,000–$29,999
]
PARTIAL_BRACKET = "B19001_007E"  # $30,000–$34,999
PARTIAL_WEIGHT  = 0.5


def estimate_eligible_households(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["eligible_hh"] = (
        df[FULLY_ELIGIBLE_BRACKETS].sum(axis=1)
        + df[PARTIAL_BRACKET] * PARTIAL_WEIGHT
    )
    return df


# ---------------------------------------------------------------------------
# Enrollment + gap (B22001)
# ---------------------------------------------------------------------------

def extract_enrollment(df: pd.DataFrame) -> pd.DataFrame:
    """B22001_002E = households that received SNAP. v2 hook: swap for FNS admin count."""
    df = df.copy()
    df["enrolled_hh"] = df["B22001_002E"]
    return df


def calculate_gap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gap_hh"]   = (df["eligible_hh"] - df["enrolled_hh"]).clip(lower=0)
    df["gap_rate"] = df["gap_hh"] / df["eligible_hh"].replace(0, pd.NA)
    return df


# ---------------------------------------------------------------------------
# Barrier signals
# ---------------------------------------------------------------------------

# -- Language (C16002) --

LEP_COLS = ["C16002_004E", "C16002_007E", "C16002_010E", "C16002_013E"]


def compute_lep_share(df: pd.DataFrame) -> pd.DataFrame:
    """
    lep_share = limited English speaking households / total households (C16002).
    Household-level — matches the grain of our enrollment data.
    """
    df = df.copy()
    df["lep_hh"]    = df[LEP_COLS].sum(axis=1)
    df["lep_share"] = df["lep_hh"] / df["C16002_001E"].replace(0, pd.NA)
    return df


# -- Documentation (B05001) --

def compute_noncitizen_share(df: pd.DataFrame) -> pd.DataFrame:
    """
    noncitizen_share = foreign-born non-citizens / total population (B05001).
    Proxy for documentation barrier: fear of benefit use affecting immigration
    status, and limited access to application assistance.
    """
    df = df.copy()
    df["noncitizen_share"] = df["B05001_006E"] / df["B05001_001E"].replace(0, pd.NA)
    return df


# -- Awareness (B15003 + county size proxy for rurality) --

def compute_awareness_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    no_hs_share = population 25+ without HS diploma / total population 25+ (B15003).
    Combined with county size (total_hh) as a rurality proxy in the classifier.
    Smaller counties → more rural → harder to reach with outreach.
    """
    df = df.copy()
    df["no_hs_pop"]   = df[NO_HS_COLS].sum(axis=1)
    df["no_hs_share"] = df["no_hs_pop"] / df["B15003_001E"].replace(0, pd.NA)
    return df


# -- Stigma (B19001 income distribution) --

# High-income households: $75,000+ (brackets 012–017)
HIGH_INCOME_COLS = [
    "B19001_012E",  # $60,000–$74,999  — included as near-affluent
    "B19001_013E",  # $75,000–$99,999
    "B19001_014E",  # $100,000–$124,999
    "B19001_015E",  # $125,000–$149,999
    "B19001_016E",  # $150,000–$199,999
    "B19001_017E",  # $200,000+
]

# Deep poverty households: < $15,000 (brackets 002–003)
# Used to compute the "high poverty but low enrollment" stigma signal
DEEP_POVERTY_COLS = [
    "B19001_002E",  # < $10,000
    "B19001_003E",  # $10,000–$14,999
]


def compute_stigma_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    high_income_share = households earning $60k+ / total households.
      Captures counties where cultural norms around self-sufficiency suppress
      benefit-seeking even among eligible lower-income households.

    poverty_share = households < $15k / total households.
      Combined with gap_rate in the classifier to flag counties where poverty
      is high but enrollment is low — a strong residual stigma signal once
      language and documentation barriers are controlled for.
    """
    df = df.copy()
    df["high_income_hh"]    = df[HIGH_INCOME_COLS].sum(axis=1)
    df["high_income_share"] = df["high_income_hh"] / df["total_hh"].replace(0, pd.NA)
    df["deep_poverty_hh"]   = df[DEEP_POVERTY_COLS].sum(axis=1)
    df["poverty_share"]     = df["deep_poverty_hh"] / df["total_hh"].replace(0, pd.NA)
    return df


# ---------------------------------------------------------------------------
# Select output columns
# ---------------------------------------------------------------------------

def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[[
        # Geography
        "county_fips", "state", "county",
        # Core counts
        "total_hh", "eligible_hh", "enrolled_hh",
        # Gap metrics
        "gap_hh", "gap_rate",
        # Barrier signals
        "lep_hh",    "lep_share",          # language
        "noncitizen_share",                # documentation
        "no_hs_share",                     # awareness (education component)
        "high_income_share", "poverty_share",  # stigma
    ]]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all() -> pd.DataFrame:
    income, hh, snap, language, citizenship, education = load_raw_tables()
    df = merge_tables(income, hh, snap, language, citizenship, education)
    df = cast_numeric(df)
    df = df.rename(columns={"B19001_001E": "total_hh"})

    df = estimate_eligible_households(df)
    df = extract_enrollment(df)
    df = calculate_gap(df)
    df = compute_lep_share(df)
    df = compute_noncitizen_share(df)
    df = compute_awareness_signals(df)
    df = compute_stigma_signals(df)
    df = select_output_columns(df)

    print(f"\n[Clean] Summary:")
    print(f"  Counties:             {len(df):,}")
    print(f"  Total households:     {df['total_hh'].sum():,.0f}")
    print(f"  Estimated eligible:   {df['eligible_hh'].sum():,.0f}")
    print(f"  Enrolled (self-rpt):  {df['enrolled_hh'].sum():,.0f}")
    print(f"  Gap (households):     {df['gap_hh'].sum():,.0f}")
    print(f"  Median gap rate:      {df['gap_rate'].median():.1%}")
    print(f"  Median LEP share:     {df['lep_share'].median():.1%}")
    print(f"  Median noncitizen:    {df['noncitizen_share'].median():.1%}")
    print(f"  Median no-HS share:   {df['no_hs_share'].median():.1%}")
    print(f"  Counties w/ NaN gap:  {df['gap_rate'].isna().sum()}")

    out_path = PROCESSED_DIR / "counties.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[Clean] Saved → {out_path}")
    return df


if __name__ == "__main__":
    run_all()
