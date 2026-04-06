"""
ranker.py — Combine gap severity with barrier actionability to produce a final
intervention priority ranking for all US counties.

Priority score formula:
  priority_score = gap_score × barrier_weight[top_barrier]

  gap_score (0–100) captures both the absolute size and the rate of the
  enrollment gap. barrier_weight adjusts for how actionable the primary
  barrier is — counties where the barrier is addressable with known
  interventions rank higher than counties of equal gap severity where
  the barrier is harder to move.

Actionability weights (by top_barrier):
  language      1.2  — translated materials, multilingual outreach workers;
                        well-documented interventions with strong evidence
  awareness     1.1  — targeted outreach campaigns, community events,
                        navigator programs
  documentation 1.0  — legal aid, trusted community orgs, application
                        assistance; effective but resource-intensive
  stigma        0.8  — cultural norm change; slowest and least predictable
                        intervention pathway

Priority tiers (based on priority_score percentile):
  P1 — top 10%   (score ≥ 90th percentile): immediate focus
  P2 — 10–25%    (75th–90th): high priority
  P3 — 25–50%    (50th–75th): medium priority
  P4 — bottom 50% (< 50th): monitor

Output:
  data/processed/priority_rankings.csv — final output consumed by the dashboard
"""

import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
INPUT_FILE    = PROCESSED_DIR / "barrier_results.csv"
OUTPUT_FILE   = PROCESSED_DIR / "priority_rankings.csv"

BARRIER_WEIGHTS = {
    "language":      1.2,
    "awareness":     1.1,
    "documentation": 1.0,
    "stigma":        0.8,
}

TIER_THRESHOLDS = {
    "P1": 90,
    "P2": 75,
    "P3": 50,
    # P4: everything below 50th percentile
}


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_barrier_results() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found.\n"
            f"Run src/models/barrier_classifier.py first."
        )
    df = pd.read_csv(
        INPUT_FILE,
        dtype={"county_fips": str, "state": str, "county": str},
    )
    print(f"[Ranker] Loaded {len(df):,} counties from {INPUT_FILE.name}")
    return df


# ---------------------------------------------------------------------------
# Priority score
# ---------------------------------------------------------------------------

# Trend boost: counties where enrollment is worsening (gap growing) receive
# a small priority boost — same gap severity but more urgent given trajectory.
TREND_BOOST = {"worsening": 3.0, "stable": 0.0, "improving": 0.0, "unknown": 0.0}


def compute_priority_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    priority_score = (gap_score × barrier_weight) + trend_boost, capped at 100.

    trend_boost adds 3 points for counties where enrollment declined ≥5% vs prior year
    — same gap severity but deteriorating trajectory warrants higher urgency.
    """
    df = df.copy()
    df["barrier_weight"] = df["top_barrier"].map(BARRIER_WEIGHTS)
    base_score = df["gap_score"] * df["barrier_weight"]

    if "gap_trend" in df.columns:
        boost = df["gap_trend"].map(TREND_BOOST).fillna(0)
    else:
        boost = 0

    df["priority_score"] = (base_score + boost).clip(upper=100).round(2)
    return df


# ---------------------------------------------------------------------------
# Priority rank + tier
# ---------------------------------------------------------------------------

def assign_rank_and_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    priority_rank: 1 = highest priority county nationally.
    priority_tier: P1 / P2 / P3 / P4 based on priority_score percentile.
    """
    df = df.copy()
    df["priority_rank"] = df["priority_score"].rank(method="min", ascending=False).astype("Int64")

    p90 = df["priority_score"].quantile(0.90)
    p75 = df["priority_score"].quantile(0.75)
    p50 = df["priority_score"].quantile(0.50)

    def _tier(score: float) -> str:
        if score >= p90:
            return "P1"
        elif score >= p75:
            return "P2"
        elif score >= p50:
            return "P3"
        else:
            return "P4"

    df["priority_tier"] = df["priority_score"].apply(_tier)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all() -> pd.DataFrame:
    df = load_barrier_results()
    df = compute_priority_score(df)
    df = assign_rank_and_tier(df)

    # Tier distribution
    tier_counts = df["priority_tier"].value_counts().reindex(["P1", "P2", "P3", "P4"])
    print("\n[Ranker] Priority tier distribution:")
    for tier, count in tier_counts.items():
        print(f"  {tier}  {count:>5,} counties  ({count/len(df)*100:.1f}%)")

    # Top 15 nationally
    top15 = df.nlargest(15, "priority_score")[[
        "priority_rank", "county_fips", "state",
        "gap_hh", "gap_rate", "gap_score",
        "top_barrier", "barrier_weight", "priority_score", "priority_tier",
    ]]
    print("\n[Ranker] Top 15 counties by priority:")
    print(top15.to_string(index=False))

    # Top 3 per barrier
    print("\n[Ranker] Top 3 counties per barrier:")
    for barrier in BARRIER_WEIGHTS:
        subset = df[df["top_barrier"] == barrier].nlargest(3, "priority_score")[[
            "priority_rank", "county_fips", "state", "gap_hh", "priority_score"
        ]]
        print(f"\n  {barrier}:")
        print(subset.to_string(index=False))

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[Ranker] Saved → {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    run_all()
