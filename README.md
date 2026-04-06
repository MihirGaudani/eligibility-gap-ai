# Eligibility Gap AI

An AI-powered tool to identify where eligible US residents are not enrolled 
in SNAP (food assistance), understand why, and prioritize outreach interventions.

**[Live Dashboard →](https://eligibility-gap-a-9mjdautvejuysulstmybdu.streamlit.app/)**

## What it does
- Calculates the gap between SNAP-eligible and SNAP-enrolled population by county across all 3,222 US counties
- Classifies the primary barrier to enrollment per county (language, documentation, awareness, stigma)
- Ranks counties by intervention priority, weighted by barrier actionability
- Uses Claude AI to generate plain-English explanations and outreach recommendations

## Data sources
- US Census ACS 5-Year 2023 estimates — income brackets (B19001), SNAP receipt (B22001), household LEP status (C16002), citizenship status (B05001), educational attainment (B15003)
- All data pulled programmatically via the Census Bureau API

## Stack
- Python 3.11+
- pandas (data pipeline)
- Plotly (interactive choropleth map)
- Streamlit (frontend dashboard)
- Anthropic Claude API (natural language explanations and outreach recommendations)

## Structure
```
src/pipeline/
  ingest.py       — pulls 6 ACS tables from Census API
  clean.py        — merges tables, estimates eligibility, derives barrier signals

src/models/
  gap_model.py           — scores and tiers the enrollment gap per county
  barrier_classifier.py  — classifies primary barrier (language / documentation / awareness / stigma)
  ranker.py              — final priority ranking weighted by barrier actionability

src/app/
  app.py          — Streamlit dashboard with map, filters, and Claude explanations

data/
  raw/            — Census API downloads (gitignored)
  processed/      — pipeline outputs; priority_rankings.csv drives the dashboard
```

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your CENSUS_API_KEY and ANTHROPIC_API_KEY

# Run the full pipeline (only needed once)
python src/pipeline/ingest.py
python src/pipeline/clean.py
python src/models/gap_model.py
python src/models/barrier_classifier.py
python src/models/ranker.py

# Launch the dashboard
streamlit run src/app/app.py
```

## API keys
- Census API key (free): https://api.census.gov/data/key_signup.html
- Anthropic API key: https://console.anthropic.com
