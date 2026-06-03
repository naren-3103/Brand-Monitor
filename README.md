# Lay's Brand Health Monitor

A multi-agent AI pipeline built with **CrewAI** and **Streamlit** that delivers weekly brand health analysis across social media, search trends, customer reviews, and competitor activity.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit App                       │
│  (Dashboards · KPI Cards · Ask AI · Agent Observability)│
└───────────────────────┬─────────────────────────────────┘
                        │ user question
                        ▼
              ┌─────────────────┐
              │ Relevance Check │  (rejects off-topic queries)
              └────────┬────────┘
                       │
          ┌────────────┼─────────────┬────────────┐
          ▼            ▼             ▼             ▼
  ┌──────────────┐ ┌──────────┐ ┌────────┐ ┌────────────┐
  │Social Listen.│ │  Search  │ │ Review │ │ Competitor │
  │    Agent     │ │  Trend   │ │ Theme  │ │ Monitoring │
  │              │ │  Agent   │ │ Agent  │ │   Agent    │
  └──────┬───────┘ └────┬─────┘ └───┬────┘ └─────┬──────┘
         │              │           │             │
         └──────────────┴─────┬─────┴─────────────┘
                               ▼
                  ┌────────────────────────┐
                  │  Insight Synthesizer   │
                  │  (Chief Brand Strat.)  │
                  └────────────┬───────────┘
                               ▼
                  ┌────────────────────────┐
                  │     Critic QA Agent    │
                  │  (Executive Reviewer)  │
                  └────────────────────────┘
```

---

## Project Structure

```
brand_monitor_pipeline/
├── app.py                  # Streamlit entry point
├── crew.py                 # CrewAI orchestration (run_brand_health_crew)
├── requirements.txt        # Pinned production dependencies
├── requirements-dev.txt    # Dev/test dependencies
├── Makefile                # Common commands (run, test, lint, format)
├── .env.example            # Environment variable template
│
├── agents/                 # One file per specialist agent
├── tasks/                  # One file per agent task (prompts + data prep)
│
├── config/
│   └── settings.py         # All thresholds, paths, and constants
│
├── utils/
│   ├── anomaly_detection.py
│   ├── azure_openai_client.py
│   ├── contradiction_checker.py
│   ├── evidence_card.py
│   ├── observability.py
│   └── timeframe_utils.py
│
├── data/                   # CSV data files (gitignored if sensitive)
├── logs/                   # Runtime logs (gitignored)
├── tests/                  # Pytest test suite
├── scripts/                # One-off / maintenance scripts
└── notebooks/              # Exploratory Jupyter notebooks
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
# Production
pip install -r requirements.txt

# Development (adds pytest, black, flake8)
pip install -r requirements-dev.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your Azure OpenAI credentials
```

### 4. Add data files

Place the following CSV files in the `data/` directory:

| File | Description |
|------|-------------|
| `social_posts.csv` | Social media posts with sentiment labels |
| `search_trends.csv` | Weekly keyword search volumes |
| `reviews.csv` | Customer reviews with star ratings |
| `competitor_news.csv` | Competitor news with sentiment impact scores |
| `brand_tracker_summary.csv` | Brand tracker survey data |
| `weekly_kpi_dashboard.csv` | Pre-aggregated weekly KPIs |

---

## Running

```bash
# Start the Streamlit app
make run
# or
streamlit run app.py

# Run the full crew pipeline directly (no UI)
make run-crew
# or
python crew.py
```

---

## Testing

```bash
make test
# or
pytest tests/ -v
```

---

## Configuration

All tuneable values are in [`config/settings.py`](config/settings.py):

| Setting | Default | Description |
|---------|---------|-------------|
| `BRAND_NAME` | `"Lay's"` | Brand to analyse |
| `NEGATIVE_SENTIMENT_SPIKE_THRESHOLD` | `30` | % negative posts to flag a spike |
| `ANOMALY_ZSCORE_THRESHOLD` | `2` | Z-score for search anomaly detection |
| `COMPETITOR_HIGH_IMPACT_THRESHOLD` | `-0.05` | Sentiment impact threshold for threats |
| `COMPETITOR_OPPORTUNITY_THRESHOLD` | `+0.05` | Sentiment impact threshold for opportunities |
| `CREW_VERBOSE` | `True` | Show per-agent reasoning in console |

---

## Code Quality

```bash
make lint      # flake8
make format    # black
```
