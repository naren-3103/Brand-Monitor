"""
Date utility helpers for the Brand Health Monitor.

All timeframe *parsing* (natural language → date range) has been moved to
utils/query_parser.py which uses an LLM instead of regex.

This file only contains:
  - get_data_availability  — find the min/max dates in the dataset
  - format_date_availability — format them as a human-readable string
  - format_date             — ISO date formatter
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Optional, Tuple, Union


def format_date(dt: date) -> str:
    return dt.strftime('%Y-%m-%d')


def _parse_date_value(value: Union[date, datetime, str]) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        value = value.strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None
    return None


def get_data_availability(
    week_start_dates: Iterable[Union[date, datetime, str]]
) -> Tuple[date, date]:
    """Return the (min_date, max_date) from a column of week-start dates."""
    if week_start_dates is None:
        raise ValueError("No week start dates provided")

    try:
        if len(week_start_dates) == 0:
            raise ValueError("No week start dates provided")
    except TypeError:
        pass

    if hasattr(week_start_dates, "tolist"):
        week_start_dates = week_start_dates.tolist()

    clean = [
        d for d in (_parse_date_value(v) for v in week_start_dates)
        if d is not None
    ]
    if not clean:
        raise ValueError("No valid dates provided")

    return min(clean), max(clean)


def format_date_availability(available_start: date, available_end: date) -> str:
    return f"{format_date(available_start)} to {format_date(available_end)}"
