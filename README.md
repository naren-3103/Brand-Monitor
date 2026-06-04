# Lay's Brand Health Monitor

A multi-agent AI pipeline built with **CrewAI** and **Streamlit** that delivers brand health analysis across social media, search trends, customer reviews, and competitor activity — with a closed feedback loop that iteratively improves report quality using a rubric-based scoring system.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Streamlit App                           │
│     Dashboards · KPI Cards · Ask AI · Agent Observability       │
└─────────────────────────────┬───────────────────────────────────┘
                              │ user question
                              ▼
            ┌─────────────────────────────────┐
            │         LLM Query Parser        │  utils/query_parser.py
            │                                 │
            │  • Relevance check              │  → "I don't know" if off-topic
            │  • Comparison detection         │  → single vs. two-period branch
            │  • Date range extraction        │  → ISO dates from natural language
            │                                 │
            │  (one LLM call — replaces       │
            │   regex + separate relevance)   │
            └──────────────┬──────────────────┘
                           │
            ┌──────────────┴──────────────┐
            │                             │
     COMPARISON?                    SINGLE PERIOD
     (period A vs B)                (or no period)
            │                             │
     Run two full                  Run one full
     pipelines in parallel         pipeline
            │                             │
            └──────────────┬──────────────┘
                           │
        ╔══════════════════╪════════════════════════════════════════╗
        ║          PHASE 1 — Specialist Agents (parallel)           ║
        ╠══════════════╦══════════════╦════════════╦════════════════╣
        ║ Social       ║ Search       ║ Review     ║ Competitor     ║
        ║ Listening    ║ Trend        ║ Theme      ║ Monitoring     ║
        ║ Agent        ║ Agent        ║ Agent      ║ Agent          ║
        ╚══════╤═══════╩══════╤═══════╩═════╤══════╩═══════╤════════╝
               └──────────────┴──────┬───────┴──────────────┘
                                     │  evidence cards
        ╔════════════════════════════╪═══════════════════════════════╗
        ║            PHASE 2 — Closed Feedback Loop                  ║
        ║                           ▼                                ║
        ║        ┌─────────────────────────────┐                     ║
        ║        │      Insight Synthesizer    │◄──────────────┐     ║
        ║        │    (Chief Brand Strat.)     │               │     ║
        ║        └──────────────┬──────────────┘               │     ║
        ║                       ▼                               │     ║
        ║        ┌─────────────────────────────┐               │     ║
        ║        │      Critic QA Agent        │  5-dimension  │     ║
        ║        │   scores D1–D5 (0–2 each)   │  rubric       │     ║
        ║        │   output_pydantic validated  │               │     ║
        ║        └──────────────┬──────────────┘               │     ║
        ║                       │                               │     ║
        ║         ┌─────────────┴──────────────┐               │     ║
        ║         │  total ≥ 7.5               │               │     ║
        ║         │  AND D1 Factual Accuracy=2? │               │     ║
        ║         └──────┬─────────────────────┘               │     ║
        ║             YES│               NO                     │     ║
        ║                ▼                └── inject issues ────┘     ║
        ║         FINAL REPORT                  (up to 3 iterations)  ║
        ╚════════════════════════════════════════════════════════════╝
                       │
          (comparison mode only)
                       ▼
        ┌──────────────────────────────────┐
        │      Comparison Synthesizer      │
        │  Python delta table (exact) +    │
        │  LLM narrative                   │
        └──────────────────────────────────┘
```

---

## Query Parser — One LLM Call Does Everything

The entry point for every Ask AI query is a **single LLM call** in `utils/query_parser.py` that replaces three separate operations that previously existed:

| Old (removed) | New |
|---------------|-----|
| Separate `is_question_relevant()` — LLM call #1 | Combined into `parse_query()` |
| Regex-based comparison detection (`is_comparison_query`) | Combined into `parse_query()` |
| Regex-based date extraction (`parse_timeframe`, 500+ lines) | Combined into `parse_query()` |

The LLM receives: today's date, the available data range, and the user's question. It returns a validated `ParsedQuery` Pydantic object:

```python
class ParsedQuery(BaseModel):
    is_relevant:   bool          # False → "I don't know"
    is_comparison: bool          # True → run two pipelines in parallel
    period_label:  Optional[str] # "January 2026", "Last Week", etc.
    period_start:  Optional[str] # ISO date — null means use all data
    period_end:    Optional[str] # ISO date
    period_a_*:    ...           # baseline period (comparison only)
    period_b_*:    ...           # subject period (comparison only)
    reason:        str           # one-sentence explanation for logging
```

Python then clamps extracted dates to the available data range and builds the dict that `app.py` consumes. If the LLM fails to return valid JSON, a safe fallback (relevant, no date filter) is used.

**Why this is better than regex:**
- Handles any natural language — "the holiday season", "past couple months", "since Q4'25", abbreviated months, numeric formats (2025-12), dotted abbreviations (Dec.) — without code changes
- No edge case maintenance — adding a new date format requires zero code
- Relevance and timeframe are evaluated together with full semantic context

---

## How the Feedback Loop Works

| Step | What happens |
|------|-------------|
| **Phase 1** | The 4 specialist agents run **in parallel** (ThreadPoolExecutor) and produce evidence cards. They never re-run across iterations. |
| **Iteration 1** | The Synthesizer compiles all 4 evidence cards into an executive report. The Critic QA agent scores it across 5 dimensions (0–2 pts each, max 10). |
| **Exit check** | Loop exits if **total score ≥ 7.5 AND D1 Factual Accuracy = 2**. A report with any wrong numbers never exits, even at 9/10. |
| **Iteration 2+** | If the check fails, the Critic's structured issues and `Feedback for Next Iteration` are injected into the Synthesizer's next prompt as mandatory fixes. |
| **Hard limit** | Loop stops after 3 iterations regardless of score. |

---

## Critic QA Scoring — 5-Dimension Rubric

The Critic QA Agent scores every report using a **fixed rubric** (not free-form opinion). Scores are returned as a validated Pydantic object — no regex, no LLM score inflation.

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

### Exit condition

```python
def _should_exit_loop(score, dimension_scores, threshold):
    if dimension_scores and dimension_scores.factual_accuracy < 2:
        return False   # never ship a report with wrong numbers
    return score >= threshold
```

---

## Comparison Queries ⚠️ Work in Progress

> **Status:** Comparison query support is under active development. The LLM query parser detects and extracts both periods correctly, but the end-to-end dual pipeline and synthesis may produce inconsistent results in some cases. Use with caution and verify outputs.

When a comparison is detected, both periods run through the full pipeline in parallel and the results are synthesised into a structured diff report.

**Example queries the LLM parser handles:**

| Query | Detected as |
|-------|-------------|
| "How does May 2026 compare to April 2026?" | April 2026 vs May 2026 |
| "Dec 2025 vs Jan 2026" | December 2025 vs January 2026 |
| "Q1 2026 vs Q2 2026" | Q1 2026 vs Q2 2026 |
| "Compare this month to last month" | Last Month vs This Month |
| "this week vs last week" | Last Week vs This Week |
| "What changed between Q4 2025 and Q1 2026?" | Q4 2025 vs Q1 2026 |
| "2025-12 vs 2026-01" | December 2025 vs January 2026 |

**Comparison report structure:**
1. **At a Glance** — Python-computed delta table (exact numbers, no LLM rounding)
2. **What Improved** — metrics with positive deltas
3. **What Declined** — metrics with negative deltas
4. **Key Driver of Change** — single most important shift
5. **Strategic Implications** — brand strategy context
6. **Recommended Actions** — tied to specific metrics

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
│   └── critic_qa_task.py             # 5-dimension rubric, output_pydantic=CriticQAOutput
│
├── config/
│   └── settings.py         # All thresholds, paths, and constants
│
├── utils/
│   ├── anomaly_detection.py          # Z-score anomaly detection for search trends
│   ├── azure_openai_client.py        # Azure OpenAI LLM client builder
│   ├── comparison_synthesizer.py     # Python delta table + LLM comparison narrative
│   ├── contradiction_checker.py      # Cross-channel signal contradiction detection
│   ├── critic_models.py              # DimensionScores, CriticQAOutput (Pydantic)
│   ├── evidence_card.py              # EvidenceCard dataclass for agent outputs
│   ├── feedback_loop.py              # FeedbackLoopResult, extract_scores_from_task_output
│   ├── query_parser.py               # LLM query parser: relevance + comparison + dates
│   └── timeframe_utils.py            # Date utilities only (get_data_availability, etc.)
│
├── data/                   # CSV data files (gitignored if sensitive)
├── logs/                   # Runtime logs (gitignored)
├── tests/                  # Pytest test suite
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
| `max_feedback_iterations` | `3` | Maximum Synthesizer → Critic cycles (2 for comparison mode) |
| `quality_threshold` | `7.5` | Total score at which the loop may exit (subject to factual accuracy gate) |

> **Note:** The loop will not exit even if `quality_threshold` is met if `D1 Factual Accuracy < 2`. Factual correctness is always a hard requirement.

---

## Agent Observability

The **Agent Observability** tab shows live feedback loop data after each analysis run:

- **Total score progression** — line chart across iterations with threshold line
- **Dimension score breakdown** — stacked bar chart (D1–D5 per iteration) showing exactly which dimension improved and which is holding the score down
- **Per-iteration scorecard** — colour-coded chips (green/amber/red) for each of the 5 dimensions
- **Side-by-side panels** — Synthesizer output and Critic QA report for every iteration
- **Loop status** — whether the run converged (hit threshold + factual gate) or exhausted max iterations

---

## Code Quality

```bash
make lint      # flake8
make format    # black
```
