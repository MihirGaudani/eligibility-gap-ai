# Eligibility Gap AI

An AI-powered tool to identify where eligible US residents are not enrolled 
in SNAP (food assistance), understand why, and prioritize outreach interventions.

## What it does
- Calculates the gap between SNAP-eligible and SNAP-enrolled population by county
- Classifies the primary barrier to enrollment per county (language, documentation, awareness, stigma)
- Ranks counties by intervention priority
- Uses Claude AI to generate plain-English explanations and outreach recommendations

## Data sources
- US Census ACS 5-Year estimates (income, household size, demographics)
- USDA FNS SNAP participation data (county-level enrollment)
- USDA FNS Quality Control data (denial reasons, processing patterns)
- State SNAP administrative data (via FOIA requests)

## Stack
- Python 3.11+
- scikit-learn (ML models)
- geopandas + Folium (geographic analysis and maps)
- Streamlit (frontend dashboard)
- Anthropic Claude API (natural language explanations)

## Structure
- src/pipeline/   — data ingestion and cleaning
- src/models/     — gap model, barrier classifier, priority ranker
- src/app/        — Streamlit dashboard
- data/raw/       — original downloaded datasets (gitignored)
- data/processed/ — cleaned, analysis-ready files (gitignored)
- models/         — saved trained model files
- notebooks/      — exploration and analysis notebooks
