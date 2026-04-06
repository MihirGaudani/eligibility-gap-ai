"""
barrier_classifier.py — Classify the primary enrollment barrier for each county
and produce a ranked list of all four barriers.

The four barriers and their feature proxies:

  language      lep_share (C16002: limited English speaking households)
  documentation noncitizen_share (B05001: foreign-born non-citizens)
                  — captures chilling effect of public charge concerns
  awareness     no_hs_share (B15003: no HS diploma) +
                rural_score (inverse of log-scaled total_hh)
                  — low attainment and geographic isolation both reduce
                    awareness of and access to SNAP outreach
  stigma        high_income_share (B19001: households earning $60k+) +
                residual_gap (gap_rate unexplained by language/documentation)
                  — counties where poverty exists but enrollment is low
                    even after controlling for other barriers

Scoring approach:
  1. Compute a raw score for each barrier from its feature proxies.
  2. Min-max normalize each barrier score to [0, 100] across all counties
     so scores are comparable.
  3. top_barrier = barrier with the highest normalized score.
  4. barrier_scores = JSON dict of all four normalized scores (for frontend).
  5. barrier_rank = ordered list of barriers from highest to lowest score.

No ML model is trained here. The scoring is rule-based and interpretable by
design — each score is directly traceable to a Census variable. This is
appropriate for a policy tool where explainability matters.

Output:
  data/processed/barrier_results.csv
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
INPUT_FILE    = PROCESSED_DIR / "gap_scores.csv"
OUTPUT_FILE   = PROCESSED_DIR / "barrier_results.csv"

BARRIERS = ["language", "documentation", "awareness", "stigma"]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_gap_scores() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found.\n"
            f"Run src/models/gap_model.py first."
        )
    df = pd.read_csv(
        INPUT_FILE,
        dtype={"county_fips": str, "state": str, "county": str},
    )
    print(f"[Barrier] Loaded {len(df):,} counties from {INPUT_FILE.name}")
    return df


# ---------------------------------------------------------------------------
# Raw barrier scores
# ---------------------------------------------------------------------------

def _pct_rank(series: pd.Series) -> pd.Series:
    """
    Convert a feature to its 0–100 percentile rank across all counties.
    NaN → 0 (treated as lowest rank).

    Why percentile ranks: raw feature scales are wildly different (lep_share
    median 0.8%, high_income_share median ~45%, gap_rate median 51%). Blending
    raw values with weights lets the largest-magnitude feature dominate every
    county. Percentile ranks normalize this — each component contributes based
    on relative extremity across counties, not absolute size.
    """
    return series.fillna(0).rank(pct=True) * 100


def compute_raw_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute one raw score per barrier using percentile-ranked component features.

    Barrier components (all inputs are percentile-ranked before combining):
      language:      lep_share
      documentation: noncitizen_share
      awareness:     50% no_hs_share + 50% rurality (inverse of county size)
      stigma:        50% high_income_share
                   + 50% low_participation signal (gap_rate × poverty_share)
                     — this is the user-specified 'historically low SNAP
                       despite high poverty' cultural signal
    """
    df = df.copy()

    # Rurality: smaller county = more rural = higher awareness barrier
    rural_pct = 100 - _pct_rank(df["total_hh"])

    # Stigma: 'historically low SNAP despite high poverty'
    low_participation = df["gap_rate"] * df["poverty_share"]

    df["_raw_language"]      = _pct_rank(df["lep_share"])
    df["_raw_documentation"] = _pct_rank(df["noncitizen_share"])
    df["_raw_awareness"]     = 0.5 * _pct_rank(df["no_hs_share"]) + 0.5 * rural_pct
    df["_raw_stigma"]        = (
        0.5 * _pct_rank(df["high_income_share"])
        + 0.5 * _pct_rank(low_participation)
    )
    return df


# ---------------------------------------------------------------------------
# Normalize to [0, 100]
# ---------------------------------------------------------------------------

def normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize each raw barrier score to [0, 100]."""
    df = df.copy()
    for barrier in BARRIERS:
        raw = df[f"_raw_{barrier}"].fillna(0)
        lo, hi = raw.min(), raw.max()
        if hi > lo:
            df[f"score_{barrier}"] = ((raw - lo) / (hi - lo) * 100).round(2)
        else:
            df[f"score_{barrier}"] = 0.0
    return df


# ---------------------------------------------------------------------------
# Classify
# ---------------------------------------------------------------------------

def classify_barriers(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each county:
      top_barrier    — barrier with the highest normalized score (str)
      barrier_rank   — all four barriers ordered high→low (JSON array string)
      barrier_scores — all four normalized scores as a JSON object string
    """
    df = df.copy()
    scores = df[[f"score_{b}" for b in BARRIERS]].copy()
    scores.columns = BARRIERS

    df["top_barrier"] = scores.idxmax(axis=1)

    df["barrier_rank"] = scores.apply(
        lambda row: json.dumps(list(row.sort_values(ascending=False).index)),
        axis=1,
    )
    df["barrier_scores"] = scores.apply(
        lambda row: json.dumps(row.round(2).to_dict()),
        axis=1,
    )
    return df


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if c.startswith("_raw_")])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all() -> pd.DataFrame:
    df = load_gap_scores()
    df = compute_raw_scores(df)
    df = normalize_scores(df)
    df = classify_barriers(df)
    df = select_output_columns(df)

    top_counts = df["top_barrier"].value_counts()
    print("\n[Barrier] Top barrier distribution:")
    for barrier in BARRIERS:
        count = top_counts.get(barrier, 0)
        print(f"  {barrier:<15} {count:>5,} counties  ({count/len(df)*100:.1f}%)")

    print("\n[Barrier] Highest-scoring county per barrier:")
    for barrier in BARRIERS:
        row = df.nlargest(1, f"score_{barrier}")[
            ["county_fips", "state", "top_barrier", f"score_{barrier}"]
        ]
        print(f"  {barrier:<15}", row.to_string(index=False, header=False))

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[Barrier] Saved → {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    run_all()
