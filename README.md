# Lay's Brand Health Monitor

A multi-agent AI pipeline built with **CrewAI** and **Streamlit** that delivers weekly brand health analysis across social media, search trends, customer reviews, and competitor activity — with a **closed feedback loop** that iteratively improves report quality until a quality threshold is met.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Streamlit App                         │
│   Dashboards · KPI Cards · Ask AI · Agent Observability     │
└─────────────────────────┬───────────────────────────────────┘
                          │ user question
                          ▼
                ┌─────────────────┐
                │ Relevance Check │  rejects off-topic queries
                └────────┬────────┘
                         │
        ╔════════════════╪═════════════════════════╗
        ║        PHASE 1 — Specialist Agents        ║
        ║              (runs once)                  ║
        ╠═══════════╦═══════════╦════════╦══════════╣
        ║  Social   ║  Search   ║ Review ║Competitor║
        ║ Listening ║   Trend   ║ Theme  ║Monitoring║
        ║   Agent   ║   Agent   ║ Agent  ║  Agent   ║
        ╚═════╤═════╩═════╤═════╩════╤═══╩═════╤════╝
              │           │          │          │
              └───────────┴────┬─────┴──────────┘
                               │  specialist evidence cards
        ╔══════════════════════╪══════════════════════════════╗
        ║       PHASE 2 — Closed Feedback Loop                ║
        ║         (up to 3 iterations)                        ║
        ║                      ▼                              ║
        ║       ┌──────────────────────────┐                  ║
        ║       │   Insight Synthesizer    │◄─────────────┐   ║
        ║       │  (Chief Brand Strat.)    │              │   ║
        ║       └──────────────┬───────────┘              │   ║
        ║                      ▼                          │   ║
        ║       ┌──────────────────────────┐              │   ║
        ║       │      Critic QA Agent     │              │   ║
        ║       │  (Executive Reviewer)    │              │   ║
        ║       └──────────────┬───────────┘              │   ║
        ║                      │                          │   ║
        ║          ┌───────────┴───────────┐              │   ║
        ║          │  quality score ≥ 7.5? │              │   ║
        ║          └───────┬───────────────┘              │   ║
        ║               YES│              NO               │   ║
        ║                  ▼               └──inject issues┘   ║
        ║           FINAL REPORT                               ║
        ╚══════════════════════════════════════════════════════╝
```

---

## How the Feedback Loop Works

| Step | What happens |
|------|-------------|
| **Phase 1** | The 4 specialist agents run once and produce evidence cards. Their data analysis is deterministic — they never re-run. |
| **Iteration 1** | The Synthesizer compiles all 4 evidence cards into an executive report. The Critic QA agent validates it and assigns a quality score out of 10. |
| **Score check** | If the score is **≥ 7.5**, the loop exits immediately and the report is returned. |
| **Iteration 2+** | If the score is below threshold, the Critic's `Issues Found` and `Feedback for Next Iteration` sections are extracted and injected directly into the Synthesizer's next prompt as mandatory fixes. |
| **Exit condition** | Loop stops when score ≥ 7.5 **or** 3 iterations have run — whichever comes first. |

The result object includes the full `iteration_history` (score per iteration, synthesizer output, critic output) which is surfaced in the **Agent Observability** tab.

---

## Project Structure

```
brand_monitor_pipeline/
├── app.py                  # Streamlit entry point
├── crew.py                 # CrewAI orchestration — specialist phase + feedback loop
├── requirements.txt        # Pinned production dependencies
├── requirements-dev.txt    # Dev/test dependencies
├── Makefile                # Common commands (run, test, lint, format)
├── .env.example            # Environment variable template
│
├── agents/                 # One file per specialist agent
│   ├── social_listening_agent.py
│   ├── search_trend_agent.py
│   ├── review_theme_agent.py
│   ├── competitor_monitoring_agent.py
│   ├── insight_synthesizer_agent.py
│   └── critic_qa_agent.py
│
├── tasks/                  # One file per agent task (prompts + data prep)
│   ├── social_listening_task.py
│   ├── search_trend_task.py
│   ├── review_theme_task.py
│   ├── competitor_monitoring_task.py
│   ├── insight_synthesizer_task.py   # accepts critic_feedback for revisions
│   ├── critic_qa_task.py             # outputs structured Feedback for Next Iteration
│   └── relevance_checker.py
│
├── config/
│   └── settings.py         # All thresholds, paths, and constants
│
├── utils/
│   ├── anomaly_detection.py
│   ├── azure_openai_client.py
│   ├── contradiction_checker.py
│   ├── evidence_card.py
│   ├── feedback_loop.py              # parse_quality_score, FeedbackLoopResult
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
.\venv\Scripts\Activate.ps1
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

Feedback loop parameters are passed directly to `run_brand_health_crew()`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_feedback_iterations` | `3` | Maximum Synthesizer → Critic cycles |
| `quality_threshold` | `7.5` | Score out of 10 at which the loop exits early |

---

## Agent Observability

The **Agent Observability** tab in the Streamlit app shows live feedback loop data after each analysis run:

- **Quality score per iteration** — plotted as a line chart with the threshold marked
- **Per-iteration detail** — expandable panels showing the Synthesizer output and Critic QA report side by side
- **Loop status** — whether the run converged (hit threshold) or exhausted max iterations

---

## Code Quality

```bash
make lint      # flake8
make format    # black
```
