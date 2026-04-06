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
    """Load all raw ACS CSVs."""
    print("[Clean] Loading raw tables...")
    income        = _load_raw(f"acs_b19001_county_{ACS_YEAR}.csv")
    hh            = _load_raw(f"acs_b11001_county_{ACS_YEAR}.csv")
    snap          = _load_raw(f"acs_b22001_county_{ACS_YEAR}.csv")
    language      = _load_raw(f"acs_c16002_county_{ACS_YEAR}.csv")
    citizenship   = _load_raw(f"acs_b05001_county_{ACS_YEAR}.csv")
    education     = _load_raw(f"acs_b15003_county_{ACS_YEAR}.csv")
    age           = _load_raw(f"acs_b01001_county_{ACS_YEAR}.csv")
    internet      = _load_raw(f"acs_b28002_county_{ACS_YEAR}.csv")
    disability    = _load_raw(f"acs_c18130_county_{ACS_YEAR}.csv")
    poverty_age   = _load_raw(f"acs_b17001_county_{ACS_YEAR}.csv")
    names         = _load_raw(f"acs_names_county_{ACS_YEAR}.csv")
    poverty_ratio = _load_raw(f"acs_c17002_county_{ACS_YEAR}.csv")
    gini          = _load_raw(f"acs_b19083_county_{ACS_YEAR}.csv")
    snap_prior    = _load_raw("acs_b22001_county_2022.csv")
    print(
        f"  B19001: {len(income):,} | B11001: {len(hh):,} | B22001: {len(snap):,} | "
        f"C16002: {len(language):,} | B05001: {len(citizenship):,} | B15003: {len(education):,} | "
        f"B01001: {len(age):,} | B28002: {len(internet):,} | C18130: {len(disability):,} | "
        f"B17001: {len(poverty_age):,} | C17002: {len(poverty_ratio):,} | "
        f"B19083: {len(gini):,} | B22001-2022: {len(snap_prior):,}"
    )
    return (income, hh, snap, language, citizenship, education, age, internet,
            disability, poverty_age, names, poverty_ratio, gini, snap_prior)


# ---------------------------------------------------------------------------
# FIPS key + merge
# ---------------------------------------------------------------------------

def _add_fips(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["county_fips"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)
    return df


# B15003 education columns for grades 1–12 no diploma (all < HS diploma)
NO_HS_COLS = [f"B15003_{str(i).zfill(3)}E" for i in range(2, 17)]  # _002E–_016E

# B01001 age columns
MALE_CHILD_COLS   = [f"B01001_{str(i).zfill(3)}E" for i in [3, 4, 5, 6]]      # M under 18
MALE_SENIOR_COLS  = [f"B01001_{str(i).zfill(3)}E" for i in [20, 21, 22, 23, 24, 25]]  # M 65+
FEMALE_CHILD_COLS = [f"B01001_{str(i).zfill(3)}E" for i in [27, 28, 29, 30]]  # F under 18
FEMALE_SENIOR_COLS = [f"B01001_{str(i).zfill(3)}E" for i in [44, 45, 46, 47, 48, 49]]  # F 65+
ALL_AGE_COLS = MALE_CHILD_COLS + MALE_SENIOR_COLS + FEMALE_CHILD_COLS + FEMALE_SENIOR_COLS


def merge_tables(income, hh, snap, language, citizenship, education,
                 age, internet, disability, poverty_age, names,
                 poverty_ratio, gini, snap_prior) -> pd.DataFrame:
    """
    Merge all tables on county_fips. Inner join on core tables; left join for
    snap_prior (2022) to retain all 3,222 counties even if prior-year data gaps exist.
    """
    print("[Clean] Merging tables on county_fips...")
    core_tables = (income, hh, snap, language, citizenship, education,
                   age, internet, disability, poverty_age, names,
                   poverty_ratio, gini)
    (income, hh, snap, language, citizenship, education, age, internet,
     disability, poverty_age, names, poverty_ratio, gini) = (_add_fips(t) for t in core_tables)

    # snap_prior uses left join — some counties may be absent in 2022 ACS
    snap_prior = _add_fips(snap_prior)

    # Rename prior-year SNAP columns to avoid collision with 2023 columns
    snap_prior = snap_prior.rename(columns={
        "B22001_001E": "B22001_001E_2022",
        "B22001_002E": "B22001_002E_2022",
        "B22001_003E": "B22001_003E_2022",
    })

    age_cols         = ["county_fips", "B01001_001E"] + ALL_AGE_COLS
    internet_cols    = ["county_fips", "B28002_001E", "B28002_013E"]
    disability_cols  = ["county_fips", "C18130_001E", "C18130_002E", "C18130_003E",
                        "C18130_004E", "C18130_009E", "C18130_010E", "C18130_011E",
                        "C18130_016E", "C18130_017E", "C18130_018E"]
    poverty_age_cols = ["county_fips", "B17001_001E", "B17001_002E", "B17001_004E",
                        "B17001_005E", "B17001_011E", "B17001_012E", "B17001_014E",
                        "B17001_015E", "B17001_021E", "B17001_022E"]
    poverty_ratio_cols = ["county_fips", "C17002_001E", "C17002_002E",
                          "C17002_003E", "C17002_004E", "C17002_005E"]

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
        .merge(age[age_cols],                    on="county_fips", how="inner")
        .merge(internet[internet_cols],          on="county_fips", how="inner")
        .merge(disability[disability_cols],      on="county_fips", how="inner")
        .merge(poverty_age[poverty_age_cols],    on="county_fips", how="inner")
        .merge(names[["county_fips", "NAME"]],   on="county_fips", how="inner")
        .merge(poverty_ratio[poverty_ratio_cols], on="county_fips", how="inner")
        .merge(gini[["county_fips", "B19083_001E"]], on="county_fips", how="inner")
        .merge(
            snap_prior[["county_fips", "B22001_002E_2022"]],
            on="county_fips", how="left"
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
# Eligibility estimate (C17002 — replaces B19001 income-bracket proxy)
# ---------------------------------------------------------------------------

# SNAP gross income limit = 130% FPL. C17002_005E spans 1.25–1.49 (width 0.24).
# 130% FPL is (1.30 - 1.25) / (1.49 - 1.25) = 20.8% into that band.
FRAC_130FPL = (1.30 - 1.25) / (1.49 - 1.25)  # 0.208


def estimate_eligible_households(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate SNAP-eligible households using C17002 poverty ratio distribution.

    Method: compute the fraction of persons at or below 130% FPL, then apply
    that fraction to total households. Substantially more accurate than the
    B19001 income-bracket approach because:
      1. Uses the actual 130% FPL threshold, not a $30-35k income bracket proxy
      2. Accounts for household-size variation (FPL is size-adjusted)
      3. Same Census source and vintage — no additional assumptions needed
    """
    df = df.copy()
    total_persons = df["C17002_001E"].replace(0, pd.NA)
    at_130fpl = (
        df["C17002_002E"]
        + df["C17002_003E"]
        + df["C17002_004E"]
        + df["C17002_005E"] * FRAC_130FPL
    )
    eligible_rate    = (at_130fpl / total_persons).clip(upper=1.0)
    df["eligible_hh"] = (df["total_hh"] * eligible_rate).round(1)
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
# New demographic / barrier features
# ---------------------------------------------------------------------------

def compute_age_shares(df: pd.DataFrame) -> pd.DataFrame:
    """
    senior_share = population 65+ / total population (B01001).
    child_share  = population under 18 / total population (B01001).
    Both are key demographic modifiers for outreach targeting.
    """
    df = df.copy()
    df["child_pop"]    = df[MALE_CHILD_COLS + FEMALE_CHILD_COLS].sum(axis=1)
    df["senior_pop"]   = df[MALE_SENIOR_COLS + FEMALE_SENIOR_COLS].sum(axis=1)
    total_pop          = df["B01001_001E"].replace(0, pd.NA)
    df["child_share"]  = df["child_pop"] / total_pop
    df["senior_share"] = df["senior_pop"] / total_pop
    return df


def compute_internet_share(df: pd.DataFrame) -> pd.DataFrame:
    """
    no_broadband_share = households with no internet access / total HH (B28002).
    High values signal a digital access barrier for online applications.
    """
    df = df.copy()
    df["no_broadband_share"] = df["B28002_013E"] / df["B28002_001E"].replace(0, pd.NA)
    return df


def compute_disability_shares(df: pd.DataFrame) -> pd.DataFrame:
    """
    disability_share = all ages with disability / total civilian pop (C18130).
    disability_poverty_share = with disability + below poverty / total civilian pop.
    """
    df = df.copy()
    total = df["C18130_001E"].replace(0, pd.NA)
    df["disability_pop"]           = df["C18130_003E"] + df["C18130_010E"] + df["C18130_017E"]
    df["disability_poverty_pop"]   = df["C18130_004E"] + df["C18130_011E"] + df["C18130_018E"]
    df["disability_share"]         = df["disability_pop"] / total
    df["disability_poverty_share"] = df["disability_poverty_pop"] / total
    return df


def compute_poverty_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    poverty_rate        = total below poverty / universe (B17001).
    child_poverty_rate  = children (under 18) below poverty / child_pop (B01001).
    senior_poverty_rate = seniors (65+) below poverty / senior_pop (B01001).
    """
    df = df.copy()
    df["poverty_rate"] = df["B17001_002E"] / df["B17001_001E"].replace(0, pd.NA)

    child_below_poverty = (
        df["B17001_004E"] + df["B17001_005E"]   # M under 5, M 5-17
        + df["B17001_014E"] + df["B17001_015E"]  # F under 5, F 5-17
    )
    senior_below_poverty = (
        df["B17001_011E"] + df["B17001_012E"]   # M 65-74, M 75+
        + df["B17001_021E"] + df["B17001_022E"]  # F 65-74, F 75+
    )
    df["child_poverty_rate"]  = child_below_poverty / df["child_pop"].replace(0, pd.NA)
    df["senior_poverty_rate"] = senior_below_poverty / df["senior_pop"].replace(0, pd.NA)
    return df


# ---------------------------------------------------------------------------
# New model features
# ---------------------------------------------------------------------------

def compute_total_pop(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expose total_pop (B01001_001E) as a named column.

    Used as the rurality proxy in the awareness barrier score — a genuine improvement
    over total_hh because population directly measures county size without confounding
    it with household composition (large families in rural areas, small households in cities).
    """
    df = df.copy()
    df["total_pop"] = df["B01001_001E"]
    return df


def compute_gini(df: pd.DataFrame) -> pd.DataFrame:
    """
    gini = Gini index of income inequality (B19083, 0–1).
    Replaces high_income_share as the primary stigma signal: directly measures
    structural inequality rather than inferring it from income bracket counts.
    """
    df = df.copy()
    df["gini"] = df["B19083_001E"]
    return df


def compute_enrollment_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare 2022 → 2023 SNAP enrollment to determine trend direction.

    enrollment_change = enrolled_hh_2023 - enrolled_hh_2022 (households)
    enrollment_rate_change = change / enrolled_hh_2022

    gap_trend label:
      worsening — enrollment fell ≥5% (gap is growing, higher urgency)
      stable    — change within ±5%
      improving — enrollment rose ≥5% (gap is shrinking)
      unknown   — 2022 data missing for this county
    """
    df = df.copy()
    prior = pd.to_numeric(df["B22001_002E_2022"], errors="coerce").replace(-666666666, pd.NA)
    df["enrolled_hh_2022"]       = prior
    df["enrollment_change"]      = df["enrolled_hh"] - prior
    df["enrollment_rate_change"] = df["enrollment_change"] / prior.replace(0, pd.NA)

    def _trend_label(rate):
        if pd.isna(rate):
            return "unknown"
        if rate <= -0.05:
            return "worsening"
        if rate >= 0.05:
            return "improving"
        return "stable"

    df["gap_trend"] = df["enrollment_rate_change"].apply(_trend_label)
    return df


# ---------------------------------------------------------------------------
# Select output columns
# ---------------------------------------------------------------------------

def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[[
        # Geography
        "county_fips", "state", "county", "county_name",
        # Core counts
        "total_hh", "eligible_hh", "enrolled_hh",
        # Gap metrics
        "gap_hh", "gap_rate",
        # Barrier signals
        "lep_hh",    "lep_share",                  # language
        "noncitizen_share",                         # documentation
        "no_hs_share",                              # awareness (education component)
        "total_pop",                                # awareness (rurality proxy)
        "gini",                                     # stigma (inequality signal)
        "high_income_share", "poverty_share",       # stigma (retained for reference)
        # Demographics
        "child_share", "senior_share",
        # Digital access
        "no_broadband_share",
        # Disability
        "disability_share", "disability_poverty_share",
        # Poverty rates
        "poverty_rate", "child_poverty_rate", "senior_poverty_rate",
        # Enrollment trend
        "enrollment_change", "gap_trend",
    ]]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all() -> pd.DataFrame:
    (income, hh, snap, language, citizenship, education, age, internet,
     disability, poverty_age, names, poverty_ratio, gini, snap_prior) = load_raw_tables()
    df = merge_tables(income, hh, snap, language, citizenship, education, age, internet,
                      disability, poverty_age, names, poverty_ratio, gini, snap_prior)
    df = cast_numeric(df)
    df = df.rename(columns={"B19001_001E": "total_hh", "NAME": "county_name"})

    df = estimate_eligible_households(df)
    df = extract_enrollment(df)
    df = calculate_gap(df)
    df = compute_lep_share(df)
    df = compute_noncitizen_share(df)
    df = compute_awareness_signals(df)
    df = compute_stigma_signals(df)
    df = compute_age_shares(df)
    df = compute_internet_share(df)
    df = compute_disability_shares(df)
    df = compute_poverty_rates(df)
    df = compute_total_pop(df)
    df = compute_gini(df)
    df = compute_enrollment_trend(df)
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
