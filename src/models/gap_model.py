"""
gap_model.py — Formalizes raw gap metrics into ranked, tiered scores for
downstream use by the barrier classifier and priority ranker.

clean.py already computes gap_hh and gap_rate per county. This module adds:

  gap_hh_rank    — rank by absolute household gap (1 = largest absolute gap)
  gap_rate_rank  — rank by gap rate (1 = highest fraction unenrolled)
  gap_severity   — quartile-based tier label: low / moderate / high / critical
  gap_score      — composite 0–100 score blending rank and rate (see below)

Composite score rationale:
  We want to surface counties that are both large in absolute terms (many
  households unreached) AND high-rate (strong signal of a systemic barrier).
  A county with 50,000 missing households but 20% gap rate is a volume problem.
  A county with 200 missing households but 95% gap rate is a penetration problem.
  Both matter; we weight them equally (50/50) to avoid purely chasing population
  centers or purely chasing extreme small-county rates.

Output:
  data/processed/gap_scores.csv  — one row per county, all gap columns + scores
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

INPUT_FILE  = PROCESSED_DIR / "counties.csv"
OUTPUT_FILE = PROCESSED_DIR / "gap_scores.csv"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_counties() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"counties.csv not found at {INPUT_FILE}\n"
            f"Run src/pipeline/clean.py first."
        )
    df = pd.read_csv(INPUT_FILE, dtype={"county_fips": str, "state": str, "county": str})
    print(f"[GapModel] Loaded {len(df):,} counties from {INPUT_FILE.name}")
    return df


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------

def add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add gap_hh_rank and gap_rate_rank (ascending rank, 1 = worst gap).
    Counties with NaN gap_rate are ranked last.
    """
    df = df.copy()
    df["gap_hh_rank"]   = df["gap_hh"].rank(method="min", ascending=False).astype("Int64")
    df["gap_rate_rank"] = df["gap_rate"].rank(method="min", ascending=False, na_option="bottom").astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Severity tiers
# ---------------------------------------------------------------------------

# Quartile breakpoints on gap_rate across all counties
TIER_LABELS = ["low", "moderate", "high", "critical"]

def add_severity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a gap_severity tier based on gap_rate quartiles.

      low      — bottom 25% of counties by gap rate
      moderate — 25th–50th percentile
      high     — 50th–75th percentile
      critical — top 25% (worst enrollment gap relative to eligible population)

    Quartiles are computed on the current dataset so thresholds are relative,
    not absolute. This means "low" still means unenrolled eligible households
    exist — it just means fewer than most other counties.
    """
    df = df.copy()
    df["gap_severity"] = pd.qcut(
        df["gap_rate"],
        q=4,
        labels=TIER_LABELS,
        duplicates="drop",
    )
    return df


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def add_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute gap_score: a 0–100 composite blending absolute and rate signals.

    Method:
      1. Normalize gap_hh_rank to [0, 1]: hh_pct = 1 - (rank - 1) / (N - 1)
         (rank 1 → 1.0, rank N → 0.0)
      2. Normalize gap_rate directly to [0, 1]: rate_pct = gap_rate / max(gap_rate)
      3. gap_score = 100 × (0.5 × hh_pct + 0.5 × rate_pct)

    Counties with missing gap_rate get gap_score = NaN.
    """
    df = df.copy()
    n = len(df)

    hh_pct   = 1 - (df["gap_hh_rank"].astype(float) - 1) / (n - 1)
    rate_max  = df["gap_rate"].max()
    rate_pct  = df["gap_rate"] / rate_max if rate_max > 0 else df["gap_rate"]

    df["gap_score"] = (100 * (0.5 * hh_pct + 0.5 * rate_pct)).round(2)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all() -> pd.DataFrame:
    """
    Load counties.csv, add gap scores, save gap_scores.csv. Returns the DataFrame.
    """
    df = load_counties()
    df = add_ranks(df)
    df = add_severity(df)
    df = add_composite_score(df)

    # Print tier distribution
    tier_counts = df["gap_severity"].value_counts().reindex(TIER_LABELS)
    print("\n[GapModel] Severity distribution:")
    for tier, count in tier_counts.items():
        print(f"  {tier:<10} {count:>5,} counties")

    # Print top 10 by composite score
    top10 = (
        df.nlargest(10, "gap_score")[
            ["county_fips", "state", "gap_hh", "gap_rate", "gap_score", "gap_severity"]
        ]
    )
    # All barrier signal columns from counties.csv pass through unchanged —
    # the classifier reads gap_scores.csv and needs them.
    print("\n[GapModel] Top 10 counties by gap_score:")
    print(top10.to_string(index=False))

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[GapModel] Saved → {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    run_all()
