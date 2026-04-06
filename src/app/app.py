"""
app.py — Streamlit dashboard for the Eligibility Gap AI tool.

Layout:
  Sidebar    — filters: state, priority tier, primary barrier
  Main top   — national summary stats (4 metric cards)
  Tab 1 (County View):
    - Choropleth map (county-level priority score)
    - Barrier breakdown charts
    - County rankings table with CSV export
    - County detail panel:
        · Demographics snapshot
        · Intervention playbook (barrier-specific tactics)
        · Impact calculator (reach slider → HH, people, $ benefit)
        · Similar counties (cosine similarity on barrier scores)
        · Claude AI explanation
  Tab 2 (State Summary):
    - State-level aggregation table
    - Top-10 states by gap bar chart

Data:
  Reads data/processed/priority_rankings.csv (output of ranker.py).
  County names are baked into the CSV by the pipeline — no runtime API call.

Run:
  streamlit run src/app/app.py
"""

import io
import json
import os
import numpy as np
import anthropic
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROCESSED_DIR  = Path(__file__).resolve().parents[2] / "data" / "processed"
RANKINGS_FILE  = PROCESSED_DIR / "priority_rankings.csv"

ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")

# Average SNAP monthly benefit per person (FY2023) and avg household size
AVG_MONTHLY_BENEFIT_PER_PERSON = 185   # USD
AVG_HH_SIZE = 2.2                      # persons per household

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

BARRIER_COLORS = {
    "Language":      "#3498db",
    "Documentation": "#9b59b6",
    "Awareness":     "#e67e22",
    "Stigma":        "#e74c3c",
}

# Intervention playbooks — concrete tactics per barrier
PLAYBOOKS = {
    "language": {
        "headline": "Bilingual outreach and in-language application support",
        "steps": [
            ("Partner", "Ethnic community organizations, promotoras (community health workers), and ESL programs that already have trust with LEP populations."),
            ("Channel", "In-language radio, translated printed materials distributed through ethnic grocery stores, religious institutions, and cultural centers."),
            ("Tactic 1", "Hire bilingual outreach workers from the community to conduct door-to-door eligibility screenings and assist with applications in-language."),
            ("Tactic 2", "Set up recurring in-person application clinics at community anchor sites (churches, mosques, cultural centers) with multilingual staff."),
            ("Tactic 3", "Distribute a one-page translated eligibility checklist at every point of contact — healthcare visits, school enrollment, WIC appointments."),
        ],
    },
    "documentation": {
        "headline": "Legal clarity and trusted navigator model for mixed-status families",
        "steps": [
            ("Partner", "Legal aid organizations, immigrant services nonprofits, and community health centers that serve undocumented and mixed-status families."),
            ("Channel", "One-on-one counseling sessions and small group workshops in trusted, private settings — not government buildings."),
            ("Tactic 1", "Train benefits navigators to clearly explain the public charge rule: SNAP does NOT count against immigration cases for most applicants, and US-citizen children in mixed-status families are fully eligible."),
            ("Tactic 2", "Embed SNAP application assistance inside existing immigration legal clinics to reduce the separate trip burden and stigma."),
            ("Tactic 3", "Use 'warm handoffs' — legal aid staff directly introduce clients to SNAP caseworkers rather than issuing a referral card."),
        ],
    },
    "awareness": {
        "headline": "Proactive screening through healthcare and trusted rural institutions",
        "steps": [
            ("Partner", "Rural health clinics, county extension offices, school districts, and community action agencies with existing rural reach."),
            ("Channel", "Healthcare referral pipelines, local TV/radio, community events and fairs — channels that work in low-broadband environments."),
            ("Tactic 1", "Train front-line healthcare workers (nurses, social workers, WIC staff) to ask a two-question SNAP screening at every patient visit and initiate applications on-site."),
            ("Tactic 2", "Partner with county cooperative extension offices to reach agricultural workers and rural households through farm bureau networks and 4-H events."),
            ("Tactic 3", "Place SNAP enrollment kiosks or tablets with navigator support at rural grocery stores, public libraries, and post offices."),
        ],
    },
    "stigma": {
        "headline": "Peer messenger strategy and reframing benefit use",
        "steps": [
            ("Partner", "Faith-based organizations, local employers (especially agriculture and service sectors), food banks, and trusted community leaders."),
            ("Channel", "Peer-to-peer outreach, faith community networks, employer communications — channels that carry social proof."),
            ("Tactic 1", "Recruit 'SNAP champions' — local community members willing to share their personal story — to speak at faith services, community meetings, and local media."),
            ("Tactic 2", "Work with faith leaders to frame SNAP enrollment as responsible stewardship and community care, neutralizing self-sufficiency stigma through trusted messengers."),
            ("Tactic 3", "Offer private, appointment-based application assistance (home visits or private office settings) to reduce the visibility of applying, especially in small tight-knit communities."),
        ],
    },
}

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
    df["display_name"] = df["county_name"].fillna("County " + df["county_fips"])
    return df


# ---------------------------------------------------------------------------
# Similar counties (cosine similarity on barrier scores)
# ---------------------------------------------------------------------------

SCORE_COLS = ["score_language", "score_documentation", "score_awareness", "score_stigma"]


@st.cache_data
def compute_similarity_matrix(df: pd.DataFrame) -> np.ndarray:
    matrix = df[SCORE_COLS].fillna(0).values
    return cosine_similarity(matrix)


def find_similar_counties(df_full: pd.DataFrame, sim_matrix: np.ndarray,
                          selected_fips: str, n: int = 5) -> pd.DataFrame:
    """Return the n most similar counties to selected_fips by barrier score profile."""
    df_r = df_full.reset_index(drop=True)
    matches = df_r[df_r["county_fips"] == selected_fips]
    if matches.empty:
        return pd.DataFrame()
    row_idx = matches.index[0]
    sims = sim_matrix[row_idx]
    df_r = df_r.copy()
    df_r["_sim"] = sims
    return (
        df_r[df_r["county_fips"] != selected_fips]
        .nlargest(n, "_sim")
        [["display_name", "state_name", "top_barrier", "gap_hh", "gap_rate",
          "priority_score", "priority_tier", "_sim"]]
    )


# ---------------------------------------------------------------------------
# Claude explanation
# ---------------------------------------------------------------------------

def generate_explanation(county_row: pd.Series, county_name: str) -> str:
    if not ANTHROPIC_KEY:
        return "ANTHROPIC_API_KEY not set. Add it to your .env file to enable AI explanations."

    barrier_scores = json.loads(county_row["barrier_scores"])
    barrier_rank   = json.loads(county_row["barrier_rank"])

    prompt = f"""You are a policy analyst helping local government agency workers understand SNAP (food assistance) enrollment gaps and design outreach interventions.

Here is data for {county_name}:

ENROLLMENT GAP
- Estimated SNAP-eligible households: {int(county_row['eligible_hh']):,}
- Currently enrolled: {int(county_row['enrolled_hh']):,}
- Gap (not enrolled): {int(county_row['gap_hh']):,} households ({county_row['gap_rate']:.1%} of eligible)
- National priority rank: #{int(county_row['priority_rank'])} out of 3,222 counties
- Priority tier: {county_row['priority_tier']} (P1 = highest national priority)

BARRIERS
- Primary barrier: {county_row['top_barrier'].title()}
- Barrier scores (0–100, higher = stronger signal):
  Language (LEP households): {barrier_scores['language']:.1f}
  Documentation (noncitizen share): {barrier_scores['documentation']:.1f}
  Awareness (low education + rurality): {barrier_scores['awareness']:.1f}
  Stigma (affluence + low participation despite poverty): {barrier_scores['stigma']:.1f}
- Barrier ranking: {', '.join(b.title() for b in barrier_rank)}

DEMOGRAPHICS
- Child share (under 18): {county_row.get('child_share', 0):.1%}
- Senior share (65+): {county_row.get('senior_share', 0):.1%}
- No broadband access: {county_row.get('no_broadband_share', 0):.1%}
- Disability share: {county_row.get('disability_share', 0):.1%}
- Poverty rate: {county_row.get('poverty_rate', 0):.1%}
- Child poverty rate: {county_row.get('child_poverty_rate', 0):.1%}
- Senior poverty rate: {county_row.get('senior_poverty_rate', 0):.1%}

Write a response with exactly two sections:

**Why the gap exists**
2–3 sentences explaining why so many eligible households in this specific county are not enrolled. Reference the demographic and barrier data above. Be concrete — mention the actual numbers. No jargon.

**Recommended outreach action**
2–3 sentences on the single highest-leverage intervention for this county. Name the type of partner organization, the channel, and the specific tactic. Tie it directly to the primary barrier and local demographic profile. Do not give generic advice."""

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=450,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

@st.cache_data
def build_map(df: pd.DataFrame) -> go.Figure:
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
# State summary
# ---------------------------------------------------------------------------

@st.cache_data
def build_state_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate county-level data to state level."""
    agg = df.groupby("state_name").agg(
        counties=("county_fips", "count"),
        gap_hh=("gap_hh", "sum"),
        eligible_hh=("eligible_hh", "sum"),
        enrolled_hh=("enrolled_hh", "sum"),
        p1_counties=("priority_tier", lambda x: (x == "P1").sum()),
        p2_counties=("priority_tier", lambda x: (x == "P2").sum()),
        median_gap_rate=("gap_rate", "median"),
        median_priority_score=("priority_score", "median"),
    ).reset_index()

    agg["gap_rate"] = agg["gap_hh"] / agg["eligible_hh"].replace(0, pd.NA)
    agg["p1_p2_counties"] = agg["p1_counties"] + agg["p2_counties"]

    # Dominant barrier per state
    dominant = (
        df.groupby(["state_name", "top_barrier"])
        .size()
        .reset_index(name="cnt")
        .sort_values("cnt", ascending=False)
        .drop_duplicates("state_name")
        .set_index("state_name")["top_barrier"]
        .map(BARRIER_LABELS)
    )
    agg["dominant_barrier"] = agg["state_name"].map(dominant)
    return agg.sort_values("gap_hh", ascending=False)


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

    df_full = load_rankings()
    sim_matrix = compute_similarity_matrix(df_full)

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
            "Eligibility estimated from income brackets (B19001).\n"
            "Average SNAP benefit: $185/person/month (FY2023)."
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
    total_gap = df["gap_hh"].sum()
    total_eligible = df["eligible_hh"].sum()
    col4.metric(
        "Gap (not enrolled)",
        f"{total_gap:,.0f}",
        delta=f"{total_gap / total_eligible:.1%} of eligible" if total_eligible else "—",
        delta_color="inverse",
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Tabs
    # -----------------------------------------------------------------------
    if "selected_fips" not in st.session_state:
        st.session_state.selected_fips = None

    tab1, tab2 = st.tabs(["County View", "State Summary"])

    # =======================================================================
    # TAB 1 — County View
    # =======================================================================
    with tab1:

        # -------------------------------------------------------------------
        # Map
        # -------------------------------------------------------------------
        st.subheader("Priority Score by County")
        st.caption("Color = priority score (0–100). Hover for details. Click a county to select it below.")

        fig = build_map(df_full)
        event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")

        if event and event.selection and event.selection.points:
            loc = event.selection.points[0].get("location")
            if loc:
                st.session_state.selected_fips = str(loc).zfill(5)

        st.divider()

        # -------------------------------------------------------------------
        # Barrier breakdown
        # -------------------------------------------------------------------
        st.subheader("What's Preventing Enrollment?")
        st.markdown(
            "Each county is classified by its **primary enrollment barrier**. "
            "The left chart shows unreached households per barrier (scale). "
            "The right chart shows county count weighted by actionability (breadth × leverage)."
        )

        with st.expander("What does each barrier mean?"):
            for b in ["language", "documentation", "awareness", "stigma"]:
                st.markdown(f"**{BARRIER_LABELS[b]}** — {BARRIER_DESCRIPTIONS[b]}")

        barrier_stats = (
            df.groupby("top_barrier")
            .agg(counties=("county_fips", "count"), gap_hh=("gap_hh", "sum"))
            .reindex(["language", "documentation", "awareness", "stigma"])
            .fillna(0)
            .reset_index()
        )
        barrier_stats["label"]    = barrier_stats["top_barrier"].map(BARRIER_LABELS)
        barrier_stats["pct_gap"]  = (barrier_stats["gap_hh"] / barrier_stats["gap_hh"].sum() * 100).round(1)

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.caption("Households in the gap — by primary barrier")
            hh_fig = px.bar(
                barrier_stats, x="label", y="gap_hh", color="label",
                color_discrete_map=BARRIER_COLORS,
                text=barrier_stats["pct_gap"].apply(lambda x: f"{x}%"),
                labels={"label": "", "gap_hh": "Gap households"},
            )
            hh_fig.update_traces(textposition="outside")
            hh_fig.update_layout(showlegend=False, margin={"t": 10, "b": 10}, yaxis_tickformat=",")
            st.plotly_chart(hh_fig, use_container_width=True)

        with chart_col2:
            st.caption("Counties per barrier — actionability weighted")
            bw = {"language": 1.2, "documentation": 1.0, "awareness": 1.1, "stigma": 0.8}
            barrier_stats["weight"]           = barrier_stats["top_barrier"].map(bw)
            barrier_stats["weighted_counties"] = (barrier_stats["counties"] * barrier_stats["weight"]).round(0)
            county_fig = px.bar(
                barrier_stats, x="label", y="counties", color="label",
                color_discrete_map=BARRIER_COLORS,
                text="counties",
                labels={"label": "", "counties": "Counties"},
                custom_data=["weighted_counties", "weight"],
            )
            county_fig.update_traces(
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>Counties: %{y}<br>"
                    "Actionability weight: %{customdata[1]}<br>"
                    "Weighted priority: %{customdata[0]}<extra></extra>"
                ),
            )
            county_fig.update_layout(showlegend=False, margin={"t": 10, "b": 10})
            st.plotly_chart(county_fig, use_container_width=True)

        st.divider()

        # -------------------------------------------------------------------
        # County rankings table + CSV export (Feature 5)
        # -------------------------------------------------------------------
        st.subheader("County Rankings")

        display_df = df[[
            "priority_rank", "display_name", "state_name",
            "gap_hh", "gap_rate", "top_barrier", "priority_score", "priority_tier",
        ]].copy()
        display_df.columns = ["Rank", "County", "State", "Gap (HH)", "Gap Rate",
                               "Top Barrier", "Priority Score", "Tier"]
        display_df["Gap Rate"]    = display_df["Gap Rate"].map("{:.1%}".format)
        display_df["Gap (HH)"]    = display_df["Gap (HH)"].map("{:,.0f}".format)
        display_df["Top Barrier"] = display_df["Top Barrier"].map(BARRIER_LABELS)
        display_df = display_df.sort_values("Rank")

        st.dataframe(display_df.reset_index(drop=True), use_container_width=True, height=380)

        # Export button
        csv_bytes = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export to CSV",
            data=csv_bytes,
            file_name="snap_gap_counties.csv",
            mime="text/csv",
            help="Download the filtered county list as a CSV file",
        )

        st.divider()

        # -------------------------------------------------------------------
        # County detail panel
        # -------------------------------------------------------------------
        st.subheader("County Detail")

        search = st.text_input("Search county", placeholder="e.g. Palm Beach, Starr, Decatur")

        all_county_options = df_full.sort_values("priority_rank")["display_name"].tolist()
        if search:
            filtered_options = [c for c in all_county_options if search.lower() in c.lower()]
            if not filtered_options:
                st.warning("No counties match that search.")
                filtered_options = all_county_options
        else:
            filtered_options = all_county_options

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

        st.divider()
        st.subheader(f"📍 {county_name}")

        # Core metrics
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
            score  = barrier_scores[barrier]
            is_top = barrier == county_row["top_barrier"]
            bcols[i].metric(
                f"{'🔴 ' if is_top else ''}{BARRIER_LABELS[barrier]}",
                f"{score:.1f} / 100",
                help=BARRIER_DESCRIPTIONS[barrier],
            )

        st.divider()

        # ---------------------------------------------------------------
        # Demographics snapshot (uses new pipeline data)
        # ---------------------------------------------------------------
        st.markdown("**Demographics Snapshot**")
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Poverty Rate",         f"{county_row.get('poverty_rate', 0):.1%}")
        d2.metric("Child Poverty",         f"{county_row.get('child_poverty_rate', 0):.1%}")
        d3.metric("Senior Poverty",        f"{county_row.get('senior_poverty_rate', 0):.1%}")
        d4.metric("Disability Share",      f"{county_row.get('disability_share', 0):.1%}")
        d5.metric("No Broadband",          f"{county_row.get('no_broadband_share', 0):.1%}")

        e1, e2 = st.columns(2)
        e1.metric("Child Share (under 18)", f"{county_row.get('child_share', 0):.1%}")
        e2.metric("Senior Share (65+)",     f"{county_row.get('senior_share', 0):.1%}")

        st.divider()

        # ---------------------------------------------------------------
        # Feature 1 — Intervention playbook
        # ---------------------------------------------------------------
        top_barrier = county_row["top_barrier"]
        playbook    = PLAYBOOKS[top_barrier]

        with st.expander(f"Intervention Playbook — {BARRIER_LABELS[top_barrier]} Barrier", expanded=True):
            st.markdown(f"**Strategy:** {playbook['headline']}")
            st.markdown("")
            for label, text in playbook["steps"]:
                st.markdown(f"**{label}:** {text}")

        st.divider()

        # ---------------------------------------------------------------
        # Feature 2 — Impact calculator
        # ---------------------------------------------------------------
        st.markdown("**Impact Calculator**")
        st.caption(
            "Estimate the impact of an outreach campaign. "
            "Based on FY2023 average SNAP benefit of $185/person/month and average household size of 2.2."
        )

        gap_hh = int(county_row["gap_hh"])
        reach_pct = st.slider(
            "% of the gap households reached by intervention",
            min_value=1, max_value=100, value=10, step=1,
            key=f"reach_{selected_fips}",
            help="Slide to estimate impact at different levels of outreach reach.",
        )

        hh_reached    = int(gap_hh * reach_pct / 100)
        people_fed    = int(hh_reached * AVG_HH_SIZE)
        monthly_ben   = int(hh_reached * AVG_HH_SIZE * AVG_MONTHLY_BENEFIT_PER_PERSON)
        annual_ben    = monthly_ben * 12

        ic1, ic2, ic3 = st.columns(3)
        ic1.metric("Households enrolled",  f"{hh_reached:,}")
        ic2.metric("People fed",           f"{people_fed:,}")
        ic3.metric("Annual benefit injected", f"${annual_ben:,.0f}")

        st.divider()

        # ---------------------------------------------------------------
        # Feature 4 — Similar counties
        # ---------------------------------------------------------------
        st.markdown("**Similar Counties**")
        st.caption(
            "Counties with the most similar barrier profile (cosine similarity on all four barrier scores). "
            "Use these for cross-county learning and coordinated outreach planning."
        )

        similar = find_similar_counties(df_full, sim_matrix, selected_fips, n=5)
        if not similar.empty:
            sim_display = similar.copy()
            sim_display["gap_rate"]       = sim_display["gap_rate"].map("{:.1%}".format)
            sim_display["gap_hh"]         = sim_display["gap_hh"].map("{:,.0f}".format)
            sim_display["priority_score"] = sim_display["priority_score"].map("{:.1f}".format)
            sim_display["similarity"]     = sim_display["_sim"].map("{:.3f}".format)
            sim_display["top_barrier"]    = sim_display["top_barrier"].map(BARRIER_LABELS)
            sim_display = sim_display.drop(columns=["_sim"]).rename(columns={
                "display_name":     "County",
                "state_name":       "State",
                "top_barrier":      "Top Barrier",
                "gap_hh":           "Gap (HH)",
                "gap_rate":         "Gap Rate",
                "priority_score":   "Priority Score",
                "priority_tier":    "Tier",
                "similarity":       "Similarity",
            })
            st.dataframe(sim_display.reset_index(drop=True), use_container_width=True, hide_index=True)
        else:
            st.info("No similar counties found.")

        st.divider()

        # ---------------------------------------------------------------
        # Claude explanation
        # ---------------------------------------------------------------
        st.markdown("**AI Explanation & Recommendation**")
        if st.button("Generate Explanation", key=f"explain_{selected_fips}"):
            with st.spinner("Generating explanation..."):
                explanation = generate_explanation(county_row, county_name)
            st.markdown(explanation)
            st.caption("Generated by Claude · Based on 2023 ACS 5-Year estimates")

    # =======================================================================
    # TAB 2 — State Summary (Feature 3)
    # =======================================================================
    with tab2:
        st.subheader("State-Level Summary")
        st.markdown(
            "Aggregated from county-level data. Use this view to prioritize state-level "
            "resource allocation and identify which states need the most intervention support."
        )

        state_df = build_state_summary(df_full)

        # Top-10 states chart
        top10_states = state_df.head(10)
        fig_state = px.bar(
            top10_states,
            x="state_name",
            y="gap_hh",
            color="dominant_barrier",
            color_discrete_map={v: BARRIER_COLORS[v] for v in BARRIER_COLORS},
            text=top10_states["gap_hh"].apply(lambda x: f"{x:,.0f}"),
            labels={"state_name": "", "gap_hh": "Gap households", "dominant_barrier": "Dominant Barrier"},
            title="Top 10 States by Total SNAP Gap (Households)",
        )
        fig_state.update_traces(textposition="outside")
        fig_state.update_layout(margin={"t": 40, "b": 10}, yaxis_tickformat=",", showlegend=True)
        st.plotly_chart(fig_state, use_container_width=True)

        st.divider()

        # State summary table
        state_display = state_df[[
            "state_name", "counties", "gap_hh", "gap_rate",
            "p1_p2_counties", "median_priority_score", "dominant_barrier",
        ]].copy()
        state_display.columns = [
            "State", "Counties", "Total Gap (HH)", "Median Gap Rate",
            "P1+P2 Counties", "Median Priority Score", "Dominant Barrier",
        ]
        state_display["Median Gap Rate"]       = state_display["Median Gap Rate"].map("{:.1%}".format)
        state_display["Total Gap (HH)"]        = state_display["Total Gap (HH)"].map("{:,.0f}".format)
        state_display["Median Priority Score"] = state_display["Median Priority Score"].map("{:.1f}".format)

        st.dataframe(state_display.reset_index(drop=True), use_container_width=True, height=600)

        # Export state summary
        state_csv = state_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export State Summary to CSV",
            data=state_csv,
            file_name="snap_gap_state_summary.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
