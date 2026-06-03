"""
Comparison Synthesizer
======================
Computes period-over-period deltas in Python (exact numbers, no LLM rounding)
and uses a focused LLM call to write the narrative sections of the comparison
report.
"""

from crewai import Agent, Task, Crew, Process


# Metric descriptors: (verified_metrics key, display label, unit, higher_is_better)
_METRICS = [
    ('social_positive_pct',       'Positive Sentiment',    '%',   True),
    ('social_negative_pct',       'Negative Sentiment',    '%',   False),
    ('social_avg_likes',          'Avg Likes / Post',      '',    True),
    ('social_avg_shares',         'Avg Shares / Post',     '',    True),
    ('social_avg_comments',       'Avg Comments / Post',   '',    True),
    ('social_total_posts',        'Total Social Posts',    '',    True),
    ('review_avg_rating',         'Avg Review Rating',     '/5',  True),
    ('review_pct_positive',       'Positive Reviews',      '%',   True),
    ('review_pct_negative',       'Negative Reviews',      '%',   False),
    ('review_total',              'Total Reviews',         '',    True),
    ('search_total_volume',       'Search Volume',         '',    True),
    ('search_growth_pct',         'Search Growth',         '%',   True),
    ('competitor_high_impact_count', 'High-Impact Threats','',    False),
    ('competitor_opportunity_count', 'Opportunities',      '',    True),
]


def compute_comparison_deltas(metrics_a: dict, metrics_b: dict) -> list:
    """
    Compute period-over-period deltas for all available metrics.
    Returns a list of dicts — one per metric that exists in both periods.
    """
    rows = []
    for key, label, unit, higher_is_better in _METRICS:
        if key not in metrics_a or key not in metrics_b:
            continue
        val_a = metrics_a[key]
        val_b = metrics_b[key]
        raw_delta = val_b - val_a
        delta = round(raw_delta, 1) if isinstance(raw_delta, float) else int(raw_delta)
        direction = '▲' if delta > 0 else ('▼' if delta < 0 else '→')
        is_improvement = (delta > 0 and higher_is_better) or (delta < 0 and not higher_is_better)
        rows.append({
            'key':              key,
            'label':            label,
            'unit':             unit,
            'val_a':            val_a,
            'val_b':            val_b,
            'delta':            delta,
            'direction':        direction,
            'is_improvement':   is_improvement,
            'higher_is_better': higher_is_better,
        })
    return rows


def format_delta_table(deltas: list, label_a: str, label_b: str) -> str:
    """Render the delta list as a Markdown table."""
    if not deltas:
        return "_No comparable metrics found for these two periods._"

    def _fmt(val, unit):
        if isinstance(val, float):
            return f"{val:,.1f}{unit}"
        return f"{val:,}{unit}"

    lines = [
        f"| Metric | {label_a} | {label_b} | Change |",
        "|--------|-----------|-----------|--------|",
    ]
    for d in deltas:
        unit = d['unit']
        a_str = _fmt(d['val_a'], unit)
        b_str = _fmt(d['val_b'], unit)
        abs_delta = abs(d['delta'])
        delta_str = f"{d['direction']} {_fmt(abs_delta, unit)}"
        lines.append(f"| {d['label']} | {a_str} | {b_str} | {delta_str} |")
    return "\n".join(lines)


def synthesize_comparison(
    result_a: dict,
    result_b: dict,
    comparison: dict,
    user_question: str,
    llm,
) -> str:
    """
    Produce a structured comparison report.

    - Delta table is Python-computed (exact) — the LLM is told not to recalculate.
    - The LLM only writes the narrative (What Improved / Declined / Implications / Actions).
    - Each individual crew result already went through its own Synthesizer→Critic
      feedback loop, so the executive reports are already QA-validated.
    """
    label_a = comparison['period_a']['label']
    label_b = comparison['period_b']['label']

    metrics_a = result_a.get('verified_metrics', {})
    metrics_b = result_b.get('verified_metrics', {})

    deltas     = compute_comparison_deltas(metrics_a, metrics_b)
    delta_table = format_delta_table(deltas, label_a, label_b)

    # Truncate individual reports to keep prompt size manageable
    exec_a = (result_a.get('executive_report') or '').strip()
    exec_b = (result_b.get('executive_report') or '').strip()
    _trunc = 1800
    exec_a_snippet = exec_a[:_trunc] + ("\n…[truncated]" if len(exec_a) > _trunc else "")
    exec_b_snippet = exec_b[:_trunc] + ("\n…[truncated]" if len(exec_b) > _trunc else "")

    no_data_a = not result_a.get('verified_metrics') or sum(result_a.get('data_summary', {}).values()) == 0
    no_data_b = not result_b.get('verified_metrics') or sum(result_b.get('data_summary', {}).values()) == 0

    if no_data_a:
        exec_a_snippet = f"_No data available for {label_a}._"
    if no_data_b:
        exec_b_snippet = f"_No data available for {label_b}._"

    agent = Agent(
        role="Brand Health Comparison Analyst",
        goal=(
            "Produce a clear, insight-driven comparison of brand health across "
            "two time periods using pre-validated reports and exact delta metrics."
        ),
        backstory=(
            "You are a senior brand analyst specialising in period-over-period "
            "performance analysis. You take QA-validated reports from two periods "
            "and distil the key changes, drivers, and strategic implications into "
            "an executive-ready comparison briefing. You never invent numbers — "
            "you use only the metrics given to you."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description=f"""
You are comparing brand health for Lay's across two periods.

USER QUESTION: "{user_question}"

COMPARISON: {label_a} (baseline) vs {label_b} (subject)

==================================================
DELTA TABLE — PYTHON-COMPUTED (do NOT recalculate any of these numbers)
==================================================
{delta_table}

==================================================
{label_a.upper()} — EXECUTIVE REPORT (QA-validated)
==================================================
{exec_a_snippet}

==================================================
{label_b.upper()} — EXECUTIVE REPORT (QA-validated)
==================================================
{exec_b_snippet}

==================================================
YOUR TASK
==================================================

Write an executive comparison report. Rules:
- Paste the delta table EXACTLY as given under "At a Glance" — do not alter it.
- Reference only numbers from the delta table — never re-derive or estimate.
- For missing data periods, state "No data available" — do not estimate.
- Be direct: name the metric, name the change, name the implication.
- End with 2–3 prioritised, specific recommended actions tied to the data.

OUTPUT FORMAT (use these exact headings):

## Brand Health Comparison: {label_a} vs {label_b}

### At a Glance
[Insert delta table exactly]

### What Improved
[Bullet each metric that improved with the exact delta. If nothing improved, write "No improvements detected."]

### What Declined
[Bullet each metric that declined with the exact delta. If nothing declined, write "No declines detected."]

### Key Driver of Change
[1–2 sentences: the single most important shift and why it matters for the brand]

### Strategic Implications
[2–3 sentences: what this means for brand strategy in the next period]

### Recommended Actions
- [Action 1 — tied to a specific metric]
- [Action 2 — tied to a specific metric]
- [Action 3 — tied to a specific metric]
""",
        expected_output=(
            "Structured brand health comparison report with delta table, "
            "narrative analysis, and recommended actions."
        ),
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )
    result = crew.kickoff()
    outputs = getattr(result, 'tasks_output', [])
    return outputs[0].raw if outputs else str(result)
