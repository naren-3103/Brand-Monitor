# Lay's Brand Health Monitor

A multi-agent AI pipeline built with **CrewAI** and **Streamlit** that delivers brand health analysis across social media, search trends, customer reviews, and competitor activity — with a **closed feedback loop** that iteratively improves report quality using a rubric-based scoring system.

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
              ┌──────────┴──────────┐
              │  Comparison query?  │
              └──────┬──────────────┘
               YES   │          NO
        ┌────────────┘          └──────────────────────┐
        │ Parse Period A & B                            │ Parse single timeframe
        │ Run two pipelines in parallel                 │
        └─────────────┬─────────────┘                  │
                      │                                 │
        ╔═════════════╪══════════════════╗   ╔══════════╪══════════════════╗
        ║     PHASE 1 — Specialist Agents (parallel, runs once per period) ║
        ╠═══════════╦═══════════╦════════╦══════════╣
        ║  Social   ║  Search   ║ Review ║Competitor║
        ║ Listening ║   Trend   ║ Theme  ║Monitoring║
        ║  Agent    ║   Agent   ║  Agent ║  Agent   ║
        ╚═════╤═════╩═════╤═════╩════╤═══╩═════╤════╝
              └───────────┴────┬─────┴──────────┘
                               │  evidence cards
        ╔══════════════════════╪═════════════════════════════════════════╗
        ║              PHASE 2 — Closed Feedback Loop                    ║
        ║                      ▼                                         ║
        ║       ┌──────────────────────────┐                             ║
        ║       │   Insight Synthesizer    │◄────────────────────────┐   ║
        ║       │  (Chief Brand Strat.)    │                         │   ║
        ║       └──────────────┬───────────┘                         │   ║
        ║                      ▼                                      │   ║
        ║       ┌──────────────────────────┐                         │   ║
        ║       │     Critic QA Agent      │  5-dimension rubric     │   ║
        ║       │  scores D1–D5 (0–2 each) │  Pydantic-validated     │   ║
        ║       └──────────────┬───────────┘                         │   ║
        ║                      │                                      │   ║
        ║          ┌───────────┴────────────────┐                    │   ║
        ║          │  total ≥ 7.5               │                    │   ║
        ║          │  AND D1 Factual = 2?        │                    │   ║
        ║          └──────┬─────────────────────┘                    │   ║
        ║              YES│                NO                         │   ║
        ║                 ▼                 └──inject issues + scores─┘   ║
        ║          FINAL REPORT                                           ║
        ╚═════════════════════════════════════════════════════════════════╝
                      │
        (comparison mode only)
                      ▼
        ┌──────────────────────────────┐
        │   Comparison Synthesizer     │
        │  Python delta table + LLM    │
        │  narrative (no re-scoring)   │
        └──────────────────────────────┘
```

---

## How the Feedback Loop Works

| Step | What happens |
|------|-------------|
| **Phase 1** | The 4 specialist agents run **in parallel** (ThreadPoolExecutor) and produce evidence cards. They never re-run across iterations. |
| **Iteration 1** | The Synthesizer compiles the evidence cards into an executive report. The Critic QA agent scores it across 5 dimensions (0–2 pts each, max 10). |
| **Exit check** | Loop exits if **total score ≥ 7.5 AND D1 Factual Accuracy = 2**. A report with any wrong numbers never exits, even at 9/10. |
| **Iteration 2+** | If the check fails, the Critic's structured issues and `Feedback for Next Iteration` are injected into the Synthesizer's next prompt as mandatory fixes. |
| **Hard limit** | Loop stops after 3 iterations regardless of score. |

---

## Critic QA Scoring — 5-Dimension Rubric

The Critic QA Agent scores every report using a **fixed rubric** (not free-form opinion). Scores are returned as a validated Pydantic object — no regex parsing, no LLM score inflation.

| Dimension | 0 pts | 1 pt | 2 pts |
|-----------|-------|------|-------|
| **D1 Factual Accuracy** | 1+ numbers wrong vs verified data | Rounding differences only | All cited numbers exact |
| **D2 Claim Support** | 3+ claims with no data backing | 1–2 unsupported claims | Every claim references a metric |
| **D3 Contradiction Handling** | Clear contradiction unresolved | Tension flagged but not fixed | All cross-channel signals reconciled |
| **D4 Recommendation Quality** | Vague or missing actions | Actions present but generic | Specific, prioritised, metric-tied |
| **D5 Executive Completeness** | 2+ sections missing | 1 section missing | All sections present, query answered |

**Total = D1 + D2 + D3 + D4 + D5 (max 10)**

### Calibration anchors (prevent LLM score inflation)

| Score | Meaning |
|-------|---------|
| 2/10 | Multiple factual errors, vague recommendations |
| 5/10 | Some errors or unsupported claims, generic recommendations |
| 7/10 | Factually accurate but recommendations not metric-tied |
| 9/10 | Accurate, specific, complete, contradictions resolved, query answered |
| 10/10 | Truly flawless — reserved for perfect analysis on all dimensions |

### Exit condition (implemented in `crew.py`)

```python
def _should_exit_loop(score, dimension_scores, threshold):
    if dimension_scores and dimension_scores.factual_accuracy < 2:
        return False   # never ship a report with wrong numbers
    return score >= threshold
```

---

## Comparison Queries ⚠️ Work in Progress

> **Status:** Comparison query support is currently under active development. Detection and period parsing are functional, but the end-to-end comparison report (dual pipeline + synthesis) may produce incomplete or inconsistent results in some cases. Use with caution and verify outputs.

Ask AI understands period-over-period comparisons. When a comparison is detected, both periods run through the full pipeline in parallel and the results are synthesised into a structured diff report.

**Supported formats:**

| Query | Detected as |
|-------|-------------|
| "How does May 2026 compare to April 2026?" | April 2026 vs May 2026 |
| "Q1 2026 vs Q2 2026" | Q1 2026 vs Q2 2026 |
| "Compare this month to last month" | Last Month vs This Month |
| "this week vs last week" | Last Week vs This Week |
| "What changed between Q4 2025 and Q1 2026?" | Q4 2025 vs Q1 2026 |
| "How does 2025 compare to 2026?" | 2025 vs 2026 |

**Comparison report structure:**
1. **At a Glance** — Python-computed delta table (exact numbers, no LLM rounding)
2. **What Improved** — metrics with positive deltas
3. **What Declined** — metrics with negative deltas
4. **Key Driver of Change** — single most important shift
5. **Strategic Implications** — brand strategy context
6. **Recommended Actions** — tied to specific metrics

Single-period queries ("how was brand health last week?", "brand health past 10 days") continue to use the full 6-agent feedback loop as normal.

---

## Project Structure

```
brand_monitor_pipeline/
├── app.py                  # Streamlit entry point
├── crew.py                 # Orchestration: specialist phase + feedback loop + exit logic
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
│   ├── insight_synthesizer_task.py   # accepts critic_feedback + iteration for revisions
│   ├── critic_qa_task.py             # 5-dimension rubric, output_pydantic=CriticQAOutput
│   └── relevance_checker.py
│
├── config/
│   └── settings.py         # All thresholds, paths, and constants
│
├── utils/
│   ├── anomaly_detection.py
│   ├── azure_openai_client.py
│   ├── comparison_synthesizer.py     # delta table + comparison LLM synthesis
│   ├── contradiction_checker.py
│   ├── critic_models.py              # DimensionScores, CriticQAOutput (Pydantic)
│   ├── evidence_card.py
│   ├── feedback_loop.py              # FeedbackLoopResult, extract_scores_from_task_output
│   ├── observability.py
│   └── timeframe_utils.py            # parse_timeframe + parse_comparison_timeframe
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
| `quality_threshold` | `7.5` | Total score at which the loop may exit (subject to factual accuracy gate) |

> **Note:** The loop will not exit even if `quality_threshold` is met if `D1 Factual Accuracy < 2`. Factual correctness is always a hard requirement.

---

## Agent Observability

The **Agent Observability** tab shows live feedback loop data after each analysis run:

- **Total score progression** — line chart across iterations with threshold line
- **Dimension score breakdown** — stacked bar chart (D1–D5 per iteration) showing exactly which dimension improved and which is still holding the score down
- **Per-iteration scorecard** — colour-coded chips (green/amber/red) for each of the 5 dimensions
- **Side-by-side panels** — Synthesizer output and Critic QA report for every iteration
- **Loop status** — whether the run converged (hit threshold + factual gate) or exhausted max iterations

---

## Code Quality

```bash
make lint      # flake8
make format    # black
```
