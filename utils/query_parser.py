"""
Query parser data models and helpers.

The CrewAI agent and task definitions live in:
  agents/query_parser_agent.py
  tasks/query_parser_task.py

Orchestration (running the mini-crew and returning the result dict) lives in:
  crew.run_query_parser()

This module only contains:
  - ParsedQuery  — the Pydantic output model for the query parser task
  - _to_date     — ISO string → date conversion
  - _clamp       — clip a date range to the available data window
  - _build_result — convert a validated ParsedQuery into the dict app.py consumes
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


# ── Pydantic output model (used as output_pydantic on the task) ───────────────

class ParsedQuery(BaseModel):
    """Structured output from the query parser task."""

    is_relevant: bool = Field(
        description=(
            "True if the question is about brand health, consumer sentiment, "
            "social media, search trends, reviews, competitor activity, or "
            "product/brand KPIs. False for everything else."
        )
    )
    is_comparison: bool = Field(
        description=(
            "True if the question explicitly compares two distinct time periods. "
            "Examples: 'Dec 2025 vs Jan 2026', 'Q1 vs Q2', 'this month vs last month'."
        )
    )

    # Single-period fields (fill when is_comparison=False and is_relevant=True)
    period_label: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable label e.g. 'January 2026', 'Last Week', 'Q1 2026'. "
            "Null if no specific period is mentioned (meaning: use all available data)."
        )
    )
    period_start: Optional[str] = Field(
        default=None,
        description="ISO date YYYY-MM-DD for the period start. Null = no date filter."
    )
    period_end: Optional[str] = Field(
        default=None,
        description="ISO date YYYY-MM-DD for the period end. Null = no date filter."
    )

    # Comparison baseline — older / reference period
    period_a_label: Optional[str] = Field(
        default=None, description="Baseline period label e.g. 'December 2025'"
    )
    period_a_start: Optional[str] = Field(default=None, description="ISO YYYY-MM-DD")
    period_a_end:   Optional[str] = Field(default=None, description="ISO YYYY-MM-DD")

    # Comparison subject — newer / current period
    period_b_label: Optional[str] = Field(
        default=None, description="Subject period label e.g. 'January 2026'"
    )
    period_b_start: Optional[str] = Field(default=None, description="ISO YYYY-MM-DD")
    period_b_end:   Optional[str] = Field(default=None, description="ISO YYYY-MM-DD")

    reason: str = Field(
        default="",
        description="One sentence explaining the relevance and period decisions."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_date(s: Optional[str]) -> Optional[date]:
    """Parse an ISO date string. Returns None on any failure."""
    if not s:
        return None
    try:
        return date.fromisoformat(str(s).strip()[:10])
    except (ValueError, TypeError):
        return None


def _clamp(
    start: Optional[date],
    end:   Optional[date],
    avail_start: date,
    avail_end:   date,
) -> tuple:
    """Clip [start, end] to [avail_start, avail_end]. Returns (eff_s, eff_e, has_data)."""
    if start is None or end is None:
        return None, None, False
    eff_s = max(start, avail_start)
    eff_e = min(end,   avail_end)
    ok    = eff_s <= eff_e
    return (eff_s if ok else None, eff_e if ok else None, ok)


def _build_result(p: ParsedQuery, avail_start: date, avail_end: date) -> dict:
    """
    Convert a validated ParsedQuery into the dict that app.py and crew.py consume.
    Clamps all extracted dates to the available data range.
    For comparison queries, also ensures period_a is the older (baseline) period.
    """
    base: dict = {
        'is_relevant':   p.is_relevant,
        'is_comparison': p.is_comparison,
        'reason':        p.reason,
    }

    if p.is_comparison:
        a_s = _to_date(p.period_a_start)
        a_e = _to_date(p.period_a_end)
        b_s = _to_date(p.period_b_start)
        b_e = _to_date(p.period_b_end)
        a_label = p.period_a_label or str(a_s)
        b_label = p.period_b_label or str(b_s)

        # Guarantee period_a is the older (baseline) period
        if a_s and b_s and a_s > b_s:
            a_s, b_s = b_s, a_s
            a_e, b_e = b_e, a_e
            a_label, b_label = b_label, a_label

        eff_a_s, eff_a_e, a_has = _clamp(a_s, a_e, avail_start, avail_end)
        eff_b_s, eff_b_e, b_has = _clamp(b_s, b_e, avail_start, avail_end)

        base.update({
            'label': f"{a_label} vs {b_label}",
            'period_a': {
                'label':           a_label,
                'start':           a_s,
                'end':             a_e,
                'effective_start': eff_a_s,
                'effective_end':   eff_a_e,
                'has_data':        a_has,
            },
            'period_b': {
                'label':           b_label,
                'start':           b_s,
                'end':             b_e,
                'effective_start': eff_b_s,
                'effective_end':   eff_b_e,
                'has_data':        b_has,
            },
        })

    else:
        p_s = _to_date(p.period_start)
        p_e = _to_date(p.period_end)

        if p_s is not None and p_e is not None:
            eff_s, eff_e, has = _clamp(p_s, p_e, avail_start, avail_end)
        else:
            # No specific period — pass None dates to crew (load all available data)
            eff_s, eff_e, has = None, None, True

        base.update({
            'period': {
                'label':           p.period_label or 'All available data',
                'start':           p_s,
                'end':             p_e,
                'effective_start': eff_s,
                'effective_end':   eff_e,
                'has_data':        has,
            },
        })

    return base
