from crewai import Task
from datetime import date

from utils.query_parser import ParsedQuery


def create_query_parser_task(
    agent,
    question: str,
    avail_start: date,
    avail_end: date,
) -> Task:
    """
    Task that runs the query parser agent.

    The full prompt lives here — not in the agent — so the date context
    (today, available range) can be injected at call time.
    """
    today = date.today()

    return Task(
        description=f"""
Parse the following brand health question.

TODAY          : {today.isoformat()}
AVAILABLE DATA : {avail_start.isoformat()}  to  {avail_end.isoformat()}
QUESTION       : "{question}"

══════════════════════════════════════════
STEP 1 — RELEVANCE
══════════════════════════════════════════
Set is_relevant = true if the question is about ANY of:
  • Brand health, reputation, or consumer perception
  • Sentiment across social media (positive / negative / neutral)
  • Social posts, likes, shares, comments, engagement
  • Search volume, search trends, or keyword performance
  • Customer reviews or star ratings
  • Competitor activity, news, pricing, or market share
  • Brand KPIs, campaign performance, or product quality

Set is_relevant = false for:
  General knowledge, weather, sports scores, coding help,
  personal advice, or anything unrelated to brand monitoring data.

══════════════════════════════════════════
STEP 2 — COMPARISON DETECTION
══════════════════════════════════════════
Set is_comparison = true ONLY when the question explicitly compares
TWO distinct time periods. Look for trigger words:
  "vs", "versus", "compare", "compared to / with",
  "difference between", "changed between", "between X and Y",
  "contrast", "relative to", "against"

True examples:
  "Dec 2025 vs Jan 2026"
  "How does Q1 compare to Q2 2026?"
  "This month vs last month"
  "What changed between November and December?"

False examples (single period):
  "How was brand health in January?"
  "What happened last week?"
  "Overall brand health?"

══════════════════════════════════════════
STEP 3 — DATE EXTRACTION
══════════════════════════════════════════
Convert every temporal expression to ISO dates (YYYY-MM-DD).
Use TODAY = {today.isoformat()} as the anchor for all relative terms.

Relative terms:
  "today"                → {today.isoformat()} to {today.isoformat()}
  "this week"            → 7-day window ending today
  "last week"            → 7-day window ending yesterday
  "this month"           → first to last day of current calendar month
  "last month"           → first to last day of previous calendar month
  "this quarter"         → first day of current quarter to today
  "last quarter"         → full previous calendar quarter
  "this year"            → Jan 1 to Dec 31 of current year
  "last year"            → Jan 1 to Dec 31 of previous year
  "past N days"          → N days back from today
  "past N weeks"         → N×7 days back from today
  "past N months"        → N calendar months back from today

Named periods:
  "January 2026" / "Jan 2026" / "jan 2026"  → 2026-01-01 to 2026-01-31
  "Dec 2025" / "december 2025"              → 2025-12-01 to 2025-12-31
  "Q1 2026"   → 2026-01-01 to 2026-03-31
  "Q2 2026"   → 2026-04-01 to 2026-06-30
  "Q3 2026"   → 2026-07-01 to 2026-09-30
  "Q4 2025"   → 2025-10-01 to 2025-12-31
  "2025"      → 2025-01-01 to 2025-12-31
  "2025-12"   → 2025-12-01 to 2025-12-31

Abbreviated months are valid: "Dec" = December, "Jan" = January, etc.

If NO time period is stated:
  → set period_start = null and period_end = null (use all available data)

For comparisons:
  → period_a = the OLDER / baseline period
  → period_b = the NEWER / subject period

All dates MUST be ISO format YYYY-MM-DD.
        """,
        expected_output=(
            "JSON matching ParsedQuery: is_relevant bool, is_comparison bool, "
            "ISO date strings for the extracted period(s), and a one-sentence reason."
        ),
        output_pydantic=ParsedQuery,
        agent=agent,
    )
