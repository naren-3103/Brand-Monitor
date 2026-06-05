# Lay's Brand Health Monitor

A multi-agent AI pipeline built with **CrewAI** and **Streamlit** that delivers brand health analysis across social media, search trends, customer reviews, and competitor activity — with a closed feedback loop that iteratively improves report quality using a rubric-based scoring system.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Streamlit App                           │
│  Dashboards · KPI Cards · Ask AI · Human Review · Observability  │
└─────────────────────────────┬───────────────────────────────────┘
                              │ user question
                              ▼
        ╔═════════════════════════════════════╗
        ║  PHASE 0 — Query Parser Agent       ║  agents/query_parser_agent.py
        ║                                     ║  tasks/query_parser_task.py
        ║  • Relevance check                  ║  → "I don't know" if off-topic
        ║  • Comparison detection             ║  → single vs. two-period branch
        ║  • Date range extraction            ║  → ISO dates from natural language
        ║                                     ║
        ║  CrewAI Agent · output_pydantic     ║
        ║  ParsedQuery · date clamping        ║
        ╚══════════════════╤══════════════════╝
                           │  orchestrated by crew.run_query_parser()
                           ▼
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

## Phase 0 — Query Parser Agent

Every Ask AI query passes through **Phase 0** before any specialist agent runs. This is a first-class **CrewAI agent** (not a utility function) that replaces three operations that previously existed as separate code paths:

| Old (removed) | New (Phase 0) |
|---------------|---------------|
| Separate `is_question_relevant()` — LLM call #1 | Combined into Query Parser Agent |
| Regex-based comparison detection (`is_comparison_query`) | Combined into Query Parser Agent |
| Regex-based date extraction (`parse_timeframe`, 500+ lines) | Combined into Query Parser Agent |

**Files:**
- `agents/query_parser_agent.py` — agent definition (role, goal, backstory)
- `tasks/query_parser_task.py` — full task prompt with today's date + available range injected; `output_pydantic=ParsedQuery`
- `utils/query_parser.py` — `ParsedQuery` Pydantic model + helper functions (`_to_date`, `_clamp`, `_build_result`)
- `crew.py → run_query_parser()` — Phase 0 orchestration: creates mini-crew, runs it, extracts result

The agent receives today's date, the available data range, and the user's question. It returns a validated `ParsedQuery` Pydantic object:

```python
class ParsedQuery(BaseModel):
    is_relevant:   bool          # False → "I don't know"
    is_comparison: bool          # True → run two pipelines in parallel
    period_label:  Optional[str] # "January 2026", "Last Week", etc.
    period_start:  Optional[str] # ISO date — null means use all data
    period_end:    Optional[str] # ISO date
    period_a_*:    ...           # baseline period (comparison only)
    period_b_*:    ...           # subject period (comparison only)
    reason:        str           # one-sentence explanation (visible in Phase 0 logs)
```

Python then clamps extracted dates to the available data range and builds the routing dict that `app.py` consumes. If Pydantic parsing fails, a safe fallback (relevant, no date filter) is applied.

**Phase 0 console output example:**
```
======================================================================
🧠 PHASE 0 — QUERY PARSER AGENT
   Question  : Compare brand health Dec 2025 vs Jan 2026
   Data range: 2025-01-01 → 2026-06-05
======================================================================
 [CrewAI agent logs appear here...]
✅ PHASE 0 COMPLETE — Query parsed successfully
   Relevant   : True
   Comparison : True
   Period A   : December 2025  (2025-12-01 → 2025-12-31, has_data=True)
   Period B   : January 2026   (2026-01-01 → 2026-01-31, has_data=True)
   Reason     : Explicit month-vs-month comparison detected.
======================================================================
```

**Why this is better than regex:**
- Handles any natural language — "the holiday season", "past couple months", "since Q4'25", abbreviated months, numeric formats (2025-12), dotted abbreviations (Dec.) — without code changes
- No edge case maintenance — adding a new date format requires zero code
- Relevance and timeframe are evaluated together with full semantic context
- Logs are now visible alongside all other agent logs in the console

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

## Ask AI — Human Review & Report Export

After the pipeline returns an answer, every response in the **Ask AI** tab includes a **Human Review** section that allows an analyst to read, edit, and export the AI-generated report before it is used anywhere.

### Human Review (HITL)

An editable text area pre-populated with the AI answer lets reviewers correct factual errors, rewrite phrasing, or add analyst commentary. Changes are reflected instantly in the export outputs.

### Action buttons

Three equal-width buttons sit below the editable area:

| Button | What it does |
|--------|-------------|
| **✅ Submit Review** | Marks the review as accepted (in-session confirmation) |
| **📄 Download as PDF** | Generates a formatted PDF from the current (edited) text and downloads it |
| **📝 Download as Text** | Downloads a plain-text `.txt` file with a header block and the full report body |

Both download buttons use the **edited** content — not the original AI output — so any reviewer changes are captured in the export.

### PDF generation

PDFs are built with **`fpdf2`** (pure Python, no system dependencies). The pipeline:

1. `_strip_markdown(text)` — removes `##`, `**bold**`, `*italic*`, backtick code, markdown links, and horizontal rules; normalises smart quotes and em-dashes to latin-1-safe equivalents
2. `_safe_latin1(text)` — encodes to latin-1 with `errors='replace'` so no character encoding exception can break a download
3. `_make_pdf(question, content)` — assembles: title header, generated date, blue-tinted question block, and the stripped report body
4. `_report_filename(question, ext)` — slugifies the question into a descriptive filename, e.g. `brand_health_how_is_lays_brand_health_20260605.pdf`

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
├── crew.py                 # Orchestration: Phase 0 (query parser) + Phase 1 (specialists) + Phase 2 (feedback loop)
├── requirements.txt        # Pinned production dependencies
├── requirements-dev.txt    # Dev/test dependencies
├── Makefile                # Common commands (run, lint, format)
├── .env.example            # Environment variable template
│
├── agents/                 # One file per agent (Phase 0 + 4 specialists + 2 synthesis)
│   ├── query_parser_agent.py         # Phase 0 — relevance + comparison + date extraction
│   ├── social_listening_agent.py
│   ├── search_trend_agent.py
│   ├── review_theme_agent.py
│   ├── competitor_monitoring_agent.py
│   ├── insight_synthesizer_agent.py
│   └── critic_qa_agent.py
│
├── tasks/                  # One file per agent task (prompts + data prep)
│   ├── query_parser_task.py          # Phase 0 — full prompt + output_pydantic=ParsedQuery
│   ├── social_listening_task.py
│   ├── search_trend_task.py
│   ├── review_theme_task.py
│   ├── competitor_monitoring_task.py
│   ├── insight_synthesizer_task.py   # accepts critic_feedback + iteration for revisions
│   └── critic_qa_task.py             # 5-dimension rubric, output_pydantic=CriticQAOutput
│
├── utils/
│   ├── anomaly_detection.py          # Z-score anomaly detection for search trends
│   ├── azure_openai_client.py        # Azure OpenAI LLM client builder
│   ├── comparison_synthesizer.py     # Python delta table + LLM comparison narrative
│   ├── contradiction_checker.py      # Cross-channel signal contradiction detection
│   ├── critic_models.py              # DimensionScores, CriticQAOutput (Pydantic)
│   ├── feedback_loop.py              # FeedbackLoopResult, extract_scores_from_task_output
│   ├── query_parser.py               # ParsedQuery Pydantic model + _to_date/_clamp/_build_result helpers
│   └── timeframe_utils.py            # get_data_availability + format_date_availability
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

Key production dependencies:

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `crewai` | Agent orchestration |
| `plotly` / `pandas` | Charts and data manipulation |
| `fpdf2` | Pure-Python PDF generation for report downloads |
| `openai` | Azure OpenAI LLM client |

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
