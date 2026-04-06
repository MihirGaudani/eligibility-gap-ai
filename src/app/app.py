"""
app.py — Streamlit dashboard for the Eligibility Gap AI tool.

Layout:
  Sidebar    — filters: state, priority tier, primary barrier
  Main top   — national summary stats (4 metric cards)
  Main mid   — choropleth map (county-level priority score)
  Main bot   — ranked county table + county detail panel
               with Claude-generated plain-English explanation

Data:
  Reads data/processed/priority_rankings.csv (output of ranker.py).
  County names fetched once from Census API on startup, cached in session.

Claude:
  Generates a plain-English explanation + outreach recommendation for each
  county on demand. Called only when the user clicks "Generate Explanation"
  to avoid unnecessary API calls.

Run:
  streamlit run src/app/app.py
"""

import json
import os
import requests
import anthropic
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROCESSED_DIR  = Path(__file__).resolve().parents[2] / "data" / "processed"
RANKINGS_FILE  = PROCESSED_DIR / "priority_rankings.csv"

CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "")
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")

BARRIER_LABELS = {
    "language":      "Language",
    "documentation": "Documentation",
    "awareness":     "Awareness",
    "stigma":        "Stigma",
}

BARRIER_DESCRIPTIONS = {
    "language":      "Limited English proficiency makes it difficult to navigate the application process.",
    "documentation": "Fear of immigration consequences or difficulty obtaining required documents.",
    "awareness":     "Low awareness of SNAP eligibility due to rurality or limited outreach infrastructure.",
    "stigma":        "Cultural norms around self-sufficiency suppress benefit-seeking behavior.",
}

TIER_COLORS = {"P1": "#c0392b", "P2": "#e67e22", "P3": "#f1c40f", "P4": "#2ecc71"}

STATE_FIPS_TO_NAME = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut", "10": "Delaware",
    "11": "District of Columbia", "12": "Florida", "13": "Georgia", "15": "Hawaii",
    "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa",
    "20": "Kansas", "21": "Kentucky", "22": "Louisiana", "23": "Maine",
    "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
    "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska",
    "32": "Nevada", "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico",
    "36": "New York", "37": "North Carolina", "38": "North Dakota", "39": "Ohio",
    "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island",
    "45": "South Carolina", "46": "South Dakota", "47": "Tennessee", "48": "Texas",
    "49": "Utah", "50": "Vermont", "51": "Virginia", "53": "Washington",
    "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming", "72": "Puerto Rico",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_rankings() -> pd.DataFrame:
    if not RANKINGS_FILE.exists():
        st.error(f"Rankings file not found: {RANKINGS_FILE}\nRun the full pipeline first.")
        st.stop()
    df = pd.read_csv(RANKINGS_FILE, dtype={"county_fips": str, "state": str, "county": str})
    df["state_name"] = df["state"].map(STATE_FIPS_TO_NAME).fillna("Unknown")
    return df


@st.cache_data
def fetch_county_names(api_key: str) -> dict[str, str]:
    """
    Fetch county names from Census API. Returns dict: county_fips → display name.
    E.g. "01001" → "Autauga County, Alabama"
    Cached by Streamlit so it only runs once per session.
    """
    if not api_key:
        return {}
    try:
        url = "https://api.census.gov/data/2023/acs/acs5"
        resp = requests.get(
            url,
            params={"get": "NAME", "for": "county:*", "key": api_key},
            timeout=30,
        )
        if resp.status_code != 200:
            return {}
        data = resp.json()
        # data[0] = ["NAME", "state", "county"], data[1:] = rows
        return {
            row[1].zfill(2) + row[2].zfill(3): row[0]
            for row in data[1:]
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Claude explanation
# ---------------------------------------------------------------------------

def generate_explanation(county_row: pd.Series, county_name: str) -> str:
    """
    Call Claude to generate a plain-English explanation of the enrollment gap
    and a specific outreach recommendation for this county.
    """
    if not ANTHROPIC_KEY:
        return "ANTHROPIC_API_KEY not set. Add it to your .env file to enable AI explanations."

    barrier_scores = json.loads(county_row["barrier_scores"])
    barrier_rank   = json.loads(county_row["barrier_rank"])

    prompt = f"""You are a policy analyst helping local government officials understand SNAP (food assistance) enrollment gaps.

Here is data for {county_name}:

- Estimated SNAP-eligible households: {int(county_row['eligible_hh']):,}
- Currently enrolled households: {int(county_row['enrolled_hh']):,}
- Gap (not enrolled): {int(county_row['gap_hh']):,} households ({county_row['gap_rate']:.1%} of eligible)
- National priority rank: #{int(county_row['priority_rank'])} out of 3,222 counties
- Priority tier: {county_row['priority_tier']} (P1 = highest national priority)
- Primary enrollment barrier: {county_row['top_barrier'].title()}
- Barrier scores (0–100, higher = stronger barrier signal):
  - Language (LEP households): {barrier_scores['language']:.1f}
  - Documentation (noncitizen share): {barrier_scores['documentation']:.1f}
  - Awareness (low education + rurality): {barrier_scores['awareness']:.1f}
  - Stigma (affluence + low participation despite poverty): {barrier_scores['stigma']:.1f}
- Barrier ranking for this county: {', '.join(b.title() for b in barrier_rank)}

Write a response with exactly two sections:

**Why the gap exists**
2–3 sentences explaining why so many eligible households in this county are not enrolled in SNAP, based on the data above. Be specific to this county's profile. Avoid jargon.

**Recommended outreach action**
2–3 sentences describing the single most effective outreach intervention for this county given its primary barrier. Be concrete — name the type of organization to partner with, the communication channel, and the specific tactic. Do not give generic advice."""

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

@st.cache_data
def build_map(df: pd.DataFrame) -> px.choropleth:
    fig = px.choropleth(
        df,
        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        locations="county_fips",
        color="priority_score",
        color_continuous_scale=["#2ecc71", "#f1c40f", "#e67e22", "#c0392b"],
        range_color=(0, 100),
        scope="usa",
        hover_name="display_name",
        labels={"priority_score": "Priority Score"},
        hover_data={
            "county_fips": False,
            "display_name": False,
            "priority_score": ":.1f",
            "gap_hh": ":,.0f",
            "gap_rate": ":.1%",
            "top_barrier": True,
            "priority_tier": True,
        },
    )
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar={"title": "Priority\nScore"},
        geo={"bgcolor": "rgba(0,0,0,0)"},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Eligibility Gap AI",
        page_icon="🍎",
        layout="wide",
    )

    st.title("🍎 Eligibility Gap AI")
    st.caption("Identifying where eligible US residents are not enrolled in SNAP — and why.")

    df_full    = load_rankings()
    county_names = fetch_county_names(CENSUS_API_KEY)

    # Add display name column
    df_full["county_name"] = df_full["county_fips"].map(county_names)
    df_full["display_name"] = df_full.apply(
        lambda r: r["county_name"] if pd.notna(r["county_name"]) else f"County {r['county_fips']}",
        axis=1,
    )

    # -----------------------------------------------------------------------
    # Sidebar filters
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("Filters")

        states = sorted(df_full["state_name"].unique())
        selected_states = st.multiselect("State", states, placeholder="All states")

        selected_tiers = st.multiselect(
            "Priority tier",
            ["P1", "P2", "P3", "P4"],
            default=["P1", "P2"],
            help="P1 = top 10% nationally, P2 = top 25%, P3 = top 50%, P4 = rest",
        )

        selected_barriers = st.multiselect(
            "Primary barrier",
            list(BARRIER_LABELS.values()),
            placeholder="All barriers",
        )

        st.divider()
        st.caption(
            "Data: US Census ACS 5-Year 2023.\n"
            "Enrollment is self-reported (B22001).\n"
            "Eligibility estimated from income brackets (B19001)."
        )

    # Apply filters
    df = df_full.copy()
    if selected_states:
        df = df[df["state_name"].isin(selected_states)]
    if selected_tiers:
        df = df[df["priority_tier"].isin(selected_tiers)]
    if selected_barriers:
        barrier_keys = [k for k, v in BARRIER_LABELS.items() if v in selected_barriers]
        df = df[df["top_barrier"].isin(barrier_keys)]

    # -----------------------------------------------------------------------
    # Summary stats
    # -----------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Counties shown", f"{len(df):,}", help="After filters")
    col2.metric("Eligible households", f"{df['eligible_hh'].sum():,.0f}")
    col3.metric("Enrolled households", f"{df['enrolled_hh'].sum():,.0f}")
    col4.metric(
        "Gap (not enrolled)",
        f"{df['gap_hh'].sum():,.0f}",
        delta=f"{df['gap_hh'].sum() / df['eligible_hh'].sum():.1%} of eligible",
        delta_color="inverse",
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Map
    # -----------------------------------------------------------------------
    st.subheader("Priority Score by County")
    st.caption("Color = priority score (0–100). Hover for details. Click a county to select it.")

    if "selected_fips" not in st.session_state:
        st.session_state.selected_fips = None

    fig = build_map(df_full)
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")

    # Extract FIPS from map click and store in session state
    if event and event.selection and event.selection.points:
        loc = event.selection.points[0].get("location")
        if loc:
            st.session_state.selected_fips = str(loc).zfill(5)

    st.divider()

    # -----------------------------------------------------------------------
    # Barrier breakdown (filtered)
    # -----------------------------------------------------------------------
    st.subheader("What's Preventing Enrollment?")
    st.markdown(
        "Each county is classified by its **primary enrollment barrier** — the structural "
        "characteristic most strongly associated with low SNAP uptake relative to eligibility. "
        "The left chart shows how many households are unreached per barrier type (the scale of "
        "the problem). The right chart shows how many counties share that barrier (the breadth). "
        "Together they guide where to concentrate which type of intervention."
    )

    with st.expander("What does each barrier mean?"):
        for barrier in ["language", "documentation", "awareness", "stigma"]:
            st.markdown(f"**{BARRIER_LABELS[barrier]}** — {BARRIER_DESCRIPTIONS[barrier]}")

    BARRIER_COLORS = {
        "Language":      "#3498db",
        "Documentation": "#9b59b6",
        "Awareness":     "#e67e22",
        "Stigma":        "#e74c3c",
    }

    barrier_stats = (
        df.groupby("top_barrier")
        .agg(counties=("county_fips", "count"), gap_hh=("gap_hh", "sum"))
        .reindex(["language", "documentation", "awareness", "stigma"])
        .fillna(0)
        .reset_index()
    )
    barrier_stats["label"]        = barrier_stats["top_barrier"].map(BARRIER_LABELS)
    barrier_stats["gap_hh_label"] = barrier_stats["gap_hh"].apply(lambda x: f"{x:,.0f}")
    barrier_stats["pct_gap"]      = (barrier_stats["gap_hh"] / barrier_stats["gap_hh"].sum() * 100).round(1)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.caption("Households in the gap — by primary barrier")
        hh_fig = px.bar(
            barrier_stats,
            x="label",
            y="gap_hh",
            color="label",
            color_discrete_map=BARRIER_COLORS,
            text=barrier_stats["pct_gap"].apply(lambda x: f"{x}%"),
            labels={"label": "", "gap_hh": "Gap households"},
        )
        hh_fig.update_traces(textposition="outside")
        hh_fig.update_layout(
            showlegend=False,
            margin={"t": 10, "b": 10},
            yaxis_tickformat=",",
        )
        st.plotly_chart(hh_fig, use_container_width=True)

    with chart_col2:
        st.caption("Counties per barrier — actionability weighted")
        # Weight county count by actionability so chart reflects intervention ease
        barrier_weights = {"language": 1.2, "documentation": 1.0, "awareness": 1.1, "stigma": 0.8}
        barrier_stats["weight"]           = barrier_stats["top_barrier"].map(barrier_weights)
        barrier_stats["weighted_counties"] = (barrier_stats["counties"] * barrier_stats["weight"]).round(0)

        county_fig = px.bar(
            barrier_stats,
            x="label",
            y="counties",
            color="label",
            color_discrete_map=BARRIER_COLORS,
            text="counties",
            labels={"label": "", "counties": "Counties"},
            custom_data=["weighted_counties", "weight"],
        )
        county_fig.update_traces(
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Counties: %{y}<br>"
                "Actionability weight: %{customdata[1]}<br>"
                "Weighted priority: %{customdata[0]}<extra></extra>"
            ),
        )
        county_fig.update_layout(showlegend=False, margin={"t": 10, "b": 10})
        st.plotly_chart(county_fig, use_container_width=True)

    st.divider()

    # -----------------------------------------------------------------------
    # County table
    # -----------------------------------------------------------------------
    st.subheader("County Rankings")

    display_df = df[[
        "priority_rank", "display_name", "state_name",
        "gap_hh", "gap_rate", "top_barrier", "priority_score", "priority_tier",
    ]].copy()
    display_df.columns = [
        "Rank", "County", "State",
        "Gap (HH)", "Gap Rate", "Top Barrier", "Priority Score", "Tier",
    ]
    display_df["Gap Rate"] = display_df["Gap Rate"].map("{:.1%}".format)
    display_df["Gap (HH)"] = display_df["Gap (HH)"].map("{:,.0f}".format)
    display_df["Top Barrier"] = display_df["Top Barrier"].map(BARRIER_LABELS)
    display_df = display_df.sort_values("Rank")

    st.dataframe(display_df.reset_index(drop=True), use_container_width=True, height=400)

    # -----------------------------------------------------------------------
    # County detail panel
    # -----------------------------------------------------------------------
    st.subheader("County Detail")

    # Search bar — filters the selectbox options
    search = st.text_input("🔍 Search county", placeholder="e.g. Palm Beach, Starr, Decatur")

    all_county_options = df_full.sort_values("priority_rank")["display_name"].tolist()
    if search:
        filtered_options = [c for c in all_county_options if search.lower() in c.lower()]
        if not filtered_options:
            st.warning("No counties match that search.")
            filtered_options = all_county_options
    else:
        filtered_options = all_county_options

    # Pre-select the county clicked on the map (if any and still in filtered list)
    default_idx = 0
    if st.session_state.get("selected_fips"):
        match = df_full[df_full["county_fips"] == st.session_state.selected_fips]
        if not match.empty:
            name = match.iloc[0]["display_name"]
            if name in filtered_options:
                default_idx = filtered_options.index(name)

    selected_name = st.selectbox("Select a county to explore", filtered_options, index=default_idx)
    selected_fips = df_full[df_full["display_name"] == selected_name]["county_fips"].iloc[0]
    county_row    = df_full[df_full["county_fips"] == selected_fips].iloc[0]
    county_name   = county_row["display_name"]

    if selected_name:

        st.divider()
        st.subheader(f"📍 {county_name}")

        # Stat columns
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Priority Rank", f"#{int(county_row['priority_rank'])}")
        c2.metric("Priority Tier", county_row["priority_tier"])
        c3.metric("Gap (households)", f"{int(county_row['gap_hh']):,}")
        c4.metric("Gap Rate", f"{county_row['gap_rate']:.1%}")

        # Barrier scores
        st.markdown("**Barrier Scores**")
        barrier_scores = json.loads(county_row["barrier_scores"])
        barrier_rank   = json.loads(county_row["barrier_rank"])

        bcols = st.columns(4)
        for i, barrier in enumerate(barrier_rank):
            score = barrier_scores[barrier]
            is_top = barrier == county_row["top_barrier"]
            bcols[i].metric(
                f"{'🔴 ' if is_top else ''}{BARRIER_LABELS[barrier]}",
                f"{score:.1f} / 100",
                help=BARRIER_DESCRIPTIONS[barrier],
            )

        # Claude explanation
        st.markdown("**AI Explanation & Recommendation**")
        if st.button("Generate Explanation", key=f"explain_{selected_fips}"):
            with st.spinner("Generating explanation..."):
                explanation = generate_explanation(county_row, county_name)
            st.markdown(explanation)
            st.caption("Generated by Claude · Based on 2023 ACS 5-Year estimates")


if __name__ == "__main__":
    main()
