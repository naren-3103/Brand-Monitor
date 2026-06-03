from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta
from typing import Dict, Optional, Tuple, Iterable, Union

MONTHS = {
    'january': 1,
    'february': 2,
    'march': 3,
    'april': 4,
    'may': 5,
    'june': 6,
    'july': 7,
    'august': 8,
    'september': 9,
    'october': 10,
    'november': 11,
    'december': 12,
}


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


def get_data_availability(week_start_dates: Iterable[Union[date, datetime, str]]) -> Tuple[date, date]:
    if week_start_dates is None:
        raise ValueError('No week start dates provided for data availability')

    try:
        if len(week_start_dates) == 0:
            raise ValueError('No week start dates provided for data availability')
    except TypeError:
        pass

    if hasattr(week_start_dates, 'tolist'):
        week_start_dates = week_start_dates.tolist()

    clean_dates = [
        parsed for parsed in (
            _parse_date_value(value) for value in week_start_dates
        ) if parsed is not None
    ]

    if len(clean_dates) == 0:
        raise ValueError('No valid dates provided for data availability')

    return min(clean_dates), max(clean_dates)


def parse_timeframe(question: str, available_start: date, available_end: date) -> Dict[str, Optional[object]]:
    text = (question or '').strip().lower()
    result = {
        'matched': False,
        'has_data': False,
        'requested_start': None,
        'requested_end': None,
        'effective_start': None,
        'effective_end': None,
        'label': None,
    }

    if not text:
        return result

    def clamp_range(start_dt: date, end_dt: date) -> Tuple[Optional[date], Optional[date]]:
        effective_start = max(start_dt, available_start)
        effective_end = min(end_dt, available_end)
        if effective_start > effective_end:
            return None, None
        return effective_start, effective_end

    def fill(label: str, start: date, end: date) -> Dict:
        result['matched'] = True
        result['label'] = label
        result['requested_start'] = start
        result['requested_end'] = end
        eff_s, eff_e = clamp_range(start, end)
        result['effective_start'] = eff_s
        result['effective_end'] = eff_e
        result['has_data'] = eff_s is not None
        return result

    def parse_date_string(value: str) -> Optional[date]:
        value = value.strip().lower()
        patterns = [
            r'(?P<month>\w+)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?,?\s*(?P<year>20\d{2})',
            r'(?P<day>\d{1,2})\s+(?P<month>\w+),?\s*(?P<year>20\d{2})',
            r'(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>20\d{2})',
        ]
        for pattern in patterns:
            match = re.search(pattern, value)
            if match:
                try:
                    month = int(match.group('month')) if match.group('month').isdigit() else MONTHS.get(match.group('month'))
                    day = int(match.group('day'))
                    year = int(match.group('year'))
                    if month and 1 <= month <= 12:
                        return date(year, month, day)
                except ValueError:
                    continue
        return None

    def parse_explicit_range(text_value: str) -> Optional[tuple]:
        direct_range = re.search(
            r'\b(?P<start_month>\w+)\s+(?P<start_day>\d{1,2})(?:st|nd|rd|th)?\s*(?:-|to|through)\s*'
            r'(?:(?P<end_month>\w+)\s+)?(?P<end_day>\d{1,2})(?:st|nd|rd|th)?,?\s*(?P<year>20\d{2})\b',
            text_value
        )
        if direct_range:
            start_month = MONTHS.get(direct_range.group('start_month'))
            end_month = MONTHS.get(direct_range.group('end_month') or direct_range.group('start_month'))
            start_day = int(direct_range.group('start_day'))
            end_day = int(direct_range.group('end_day'))
            year = int(direct_range.group('year'))
            if start_month and end_month:
                try:
                    return date(year, start_month, start_day), date(year, end_month, end_day)
                except ValueError:
                    return None

        range_keywords = re.search(
            r'\b(?:from|between)\s+(?P<start>.+?)\s+(?:to|and|through)\s+(?P<end>.+?)\b',
            text_value
        )
        if range_keywords:
            start_dt = parse_date_string(range_keywords.group('start'))
            end_dt = parse_date_string(range_keywords.group('end'))
            if start_dt and end_dt:
                return start_dt, end_dt

        return None

    def parse_relative_week(text_value: str) -> Optional[tuple]:
        ordinal_match = re.search(
            r'\b(first|second|third|fourth|fifth|last)\s+week\s+of\s+(' + '|'.join(MONTHS.keys()) + r')\s+(20\d{2})\b',
            text_value
        )
        if ordinal_match:
            ordinal = ordinal_match.group(1)
            month_name = ordinal_match.group(2)
            year = int(ordinal_match.group(3))
            month = MONTHS[month_name]
            month_start = date(year, month, 1)
            month_end = date(year, month, calendar.monthrange(year, month)[1])
            if ordinal == 'last':
                s = max(month_end - timedelta(days=6), month_start)
                e = month_end
            else:
                week_index = ['first', 'second', 'third', 'fourth', 'fifth'].index(ordinal)
                s = month_start + timedelta(days=week_index * 7)
                e = min(s + timedelta(days=6), month_end)
            return s, e, f'{ordinal.title()} week of {month_name.title()} {year}'

        week_of_match = re.search(
            r'\bweek\s+of\s+(?P<month>\w+)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?,?\s*(?P<year>20\d{2})\b',
            text_value
        )
        if week_of_match:
            s = parse_date_string(week_of_match.group(0))
            if s:
                return s, s + timedelta(days=6), f'Week of {s:%B %d, %Y}'

        return None

    def parse_quarter(text_value: str) -> Optional[tuple]:
        m = re.search(r'\bq([1-4])\s*(20\d{2})\b', text_value)
        if m:
            q = int(m.group(1))
            year = int(m.group(2))
            start_month = 3 * (q - 1) + 1
            end_month = start_month + 2
            s = date(year, start_month, 1)
            e = date(year, end_month, calendar.monthrange(year, end_month)[1])
            return s, e, f'Q{q} {year}'
        return None

    # ── 1. Ordinal / "week of" relative weeks ──────────────────────────────────
    week_phrase = parse_relative_week(text)
    if week_phrase:
        s, e, label = week_phrase
        return fill(label, s, e)

    today = date.today()

    # ── 2. Today / yesterday ───────────────────────────────────────────────────
    if 'today' in text or 'this day' in text:
        return fill('today', today, today)

    if 'yesterday' in text:
        d = today - timedelta(days=1)
        return fill('yesterday', d, d)

    # ── 3. This week / last week ───────────────────────────────────────────────
    # Anchored to the real current date so queries outside the dataset
    # correctly return has_data=False instead of silently using available_end.
    if 'this week' in text:
        return fill('this week', today - timedelta(days=6), today)

    if 'last week' in text:
        e = today - timedelta(days=7)
        return fill('last week', e - timedelta(days=6), e)

    # ── 4. This month / last month ─────────────────────────────────────────────
    if 'this month' in text:
        year = today.year
        month = today.month
        return fill(
            'this month',
            date(year, month, 1),
            date(year, month, calendar.monthrange(year, month)[1])
        )

    if 'last month' in text:
        year = today.year
        month = today.month - 1
        if month == 0:
            year -= 1
            month = 12
        return fill(
            'last month',
            date(year, month, 1),
            date(year, month, calendar.monthrange(year, month)[1])
        )

    # ── 5. This quarter / last quarter ─────────────────────────────────────────
    if 'this quarter' in text or 'current quarter' in text:
        year = today.year
        q = (today.month - 1) // 3 + 1
        start_month = 3 * (q - 1) + 1
        end_month = start_month + 2
        return fill(
            f'Q{q} {year}',
            date(year, start_month, 1),
            date(year, end_month, calendar.monthrange(year, end_month)[1])
        )

    if 'last quarter' in text or 'previous quarter' in text:
        year = today.year
        q = (today.month - 1) // 3 + 1
        q -= 1
        if q == 0:
            q = 4
            year -= 1
        start_month = 3 * (q - 1) + 1
        end_month = start_month + 2
        return fill(
            f'Q{q} {year}',
            date(year, start_month, 1),
            date(year, end_month, calendar.monthrange(year, end_month)[1])
        )

    # ── 6. This year / last year ───────────────────────────────────────────────
    if 'this year' in text or 'current year' in text:
        year = today.year
        return fill(str(year), date(year, 1, 1), date(year, 12, 31))

    if 'last year' in text or 'previous year' in text:
        year = today.year - 1
        return fill(str(year), date(year, 1, 1), date(year, 12, 31))

    # ── 7. Rolling window: "past/last N days|weeks|months|years" ──────────────
    rolling = re.search(
        r'\b(?:past|last|recent)\s+(\d+)\s+(day|week|month|year)s?\b',
        text
    )
    if rolling:
        n = int(rolling.group(1))
        unit = rolling.group(2)
        e = today
        if unit == 'day':
            s = e - timedelta(days=n - 1)
            label = f'past {n} day{"s" if n != 1 else ""}'
        elif unit == 'week':
            s = e - timedelta(weeks=n) + timedelta(days=1)
            label = f'past {n} week{"s" if n != 1 else ""}'
        elif unit == 'month':
            s = e - timedelta(days=n * 30)
            label = f'past {n} month{"s" if n != 1 else ""}'
        else:  # year
            s = e - timedelta(days=n * 365)
            label = f'past {n} year{"s" if n != 1 else ""}'
        return fill(label, s, e)

    # ── 8. Explicit date ranges (from X to Y, X-Y with year) ──────────────────
    specific_range = parse_explicit_range(text)
    if specific_range:
        s, e = specific_range
        return fill(f'{s:%Y-%m-%d} to {e:%Y-%m-%d}', s, e)

    # ── 9. Named quarter: Q1 2024 ─────────────────────────────────────────────
    quarter_phrase = parse_quarter(text)
    if quarter_phrase:
        s, e, label = quarter_phrase
        return fill(label, s, e)

    # ── 10. Explicit single date ───────────────────────────────────────────────
    explicit_date = parse_date_string(text)
    if explicit_date:
        return fill(explicit_date.strftime('%B %d, %Y'), explicit_date, explicit_date)

    # ── 11. Month + year: "january 2024" ──────────────────────────────────────
    month_match = re.search(r'\b(' + '|'.join(MONTHS.keys()) + r')\s+(20\d{2})\b', text)
    if month_match:
        month_name = month_match.group(1)
        year = int(month_match.group(2))
        month = MONTHS[month_name]
        return fill(
            f'{month_name.title()} {year}',
            date(year, month, 1),
            date(year, month, calendar.monthrange(year, month)[1])
        )

    # ── 12. Year only: "2024" ──────────────────────────────────────────────────
    year_match = re.search(r'\b(20\d{2})\b', text)
    if year_match:
        year = int(year_match.group(1))
        return fill(str(year), date(year, 1, 1), date(year, 12, 31))

    return result


def format_timeframe_label(parsed: Dict[str, Optional[object]]) -> str:
    if not parsed.get('matched'):
        return ''
    if parsed.get('requested_start') and parsed.get('requested_end'):
        start = format_date(parsed['requested_start'])
        end = format_date(parsed['requested_end'])
        return f"{parsed.get('label', 'Requested timeframe')} ({start} to {end})"
    return parsed.get('label', '')


def format_date_availability(available_start: date, available_end: date) -> str:
    return f"{format_date(available_start)} to {format_date(available_end)}"


# ─────────────────────────────────────────────────────────────────────────────
# Comparison query support
# ─────────────────────────────────────────────────────────────────────────────

_COMPARISON_PATTERNS = [
    r'\bcompare\b',
    r'\bvs\.?\b',
    r'\bversus\b',
    r'\bcompared\s+to\b',
    r'\bcompared\s+with\b',
    r'\bdifference\s+between\b',
    r'\bchanged?\s+between\b',
    r'\bbetween\b',
    r'\bhow\s+.{0,40}compare\b',
    r'\bcontrast\b',
    r'\brelative\s+to\b',
    r'\bagainst\b',
]


def is_comparison_query(text: str) -> bool:
    """Return True if the question is asking for a period-over-period comparison."""
    tl = text.lower()
    return any(re.search(p, tl) for p in _COMPARISON_PATTERNS)


def _add_relative_periods(text: str, today: date, raw: list) -> None:
    """Append relative time periods (this week, last month, …) to *raw*."""
    def _last_month():
        m = today.month - 1 or 12
        y = today.year - (1 if today.month == 1 else 0)
        return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1]), 'Last Month'

    def _last_quarter():
        q = (today.month - 1) // 3          # 0-based current quarter
        if q == 0:
            q, y = 4, today.year - 1
        else:
            q, y = q, today.year
        sm = 3 * (q - 1) + 1
        em = sm + 2
        return date(y, sm, 1), date(y, em, calendar.monthrange(y, em)[1]), f'Q{q} {y}'

    candidates = [
        ('this week',       today - timedelta(days=6),  today,                          'This Week'),
        ('last week',       today - timedelta(days=13), today - timedelta(days=7),      'Last Week'),
        ('previous week',   today - timedelta(days=13), today - timedelta(days=7),      'Last Week'),
        ('this month',      date(today.year, today.month, 1),
                            date(today.year, today.month, calendar.monthrange(today.year, today.month)[1]),
                            'This Month'),
        ('this year',       date(today.year, 1, 1),     date(today.year, 12, 31),       str(today.year)),
        ('last year',       date(today.year-1, 1, 1),   date(today.year-1, 12, 31),     str(today.year-1)),
        ('previous year',   date(today.year-1, 1, 1),   date(today.year-1, 12, 31),     str(today.year-1)),
    ]

    # last month / previous month (computed dynamically)
    lm_s, lm_e, lm_l = _last_month()
    candidates += [
        ('last month',      lm_s, lm_e, lm_l),
        ('previous month',  lm_s, lm_e, lm_l),
    ]

    # last quarter / previous quarter
    lq_s, lq_e, lq_l = _last_quarter()
    candidates += [
        ('last quarter',     lq_s, lq_e, lq_l),
        ('previous quarter', lq_s, lq_e, lq_l),
    ]

    # this quarter
    cq = (today.month - 1) // 3 + 1
    cqsm = 3 * (cq - 1) + 1
    cqem = cqsm + 2
    candidates.append((
        'this quarter',
        date(today.year, cqsm, 1),
        date(today.year, cqem, calendar.monthrange(today.year, cqem)[1]),
        f'Q{cq} {today.year}',
    ))

    added_labels: set = set()
    for phrase, s, e, label in candidates:
        if phrase in text and label not in added_labels:
            raw.append((s, e, label))
            added_labels.add(label)


def _extract_periods_from_text(
    text: str, available_start: date, available_end: date
) -> list:
    """
    Extract every distinct time-period reference in *text* and return them as
    a list of dicts sorted by start date (oldest first = period_a / baseline).
    """
    today = date.today()
    raw: list = []          # (start, end, label)
    claimed_years: set = set()

    # 1. Quarters: "q1 2026", "q2 2025"
    for m in re.finditer(r'\bq([1-4])\s*(20\d{2})\b', text):
        q, year = int(m.group(1)), int(m.group(2))
        sm = 3 * (q - 1) + 1
        em = sm + 2
        raw.append((
            date(year, sm, 1),
            date(year, em, calendar.monthrange(year, em)[1]),
            f'Q{q} {year}',
        ))
        claimed_years.add(year)

    # 2. Month + year: "may 2026", "april 2026"
    for m in re.finditer(r'\b(' + '|'.join(MONTHS.keys()) + r')\s+(20\d{2})\b', text):
        month_num = MONTHS[m.group(1)]
        year = int(m.group(2))
        raw.append((
            date(year, month_num, 1),
            date(year, month_num, calendar.monthrange(year, month_num)[1]),
            f'{m.group(1).title()} {year}',
        ))
        claimed_years.add(year)

    # 3. Relative terms (this week / last month / etc.)
    _add_relative_periods(text, today, raw)

    # 4. Year-only ("2025", "2026") — only for years not already covered above
    for m in re.finditer(r'\b(20\d{2})\b', text):
        year = int(m.group(1))
        if year not in claimed_years:
            raw.append((date(year, 1, 1), date(year, 12, 31), str(year)))
            claimed_years.add(year)

    # Deduplicate by (start, end) keeping first occurrence
    seen: set = set()
    unique: list = []
    for s, e, label in raw:
        key = (s, e)
        if key not in seen:
            seen.add(key)
            unique.append((s, e, label))

    # Build result dicts, sorted oldest → newest
    result = []
    for s, e, label in sorted(unique, key=lambda x: x[0]):
        eff_s = max(s, available_start)
        eff_e = min(e, available_end)
        has_data = eff_s <= eff_e
        result.append({
            'label':            label,
            'start':            s,
            'end':              e,
            'effective_start':  eff_s if has_data else None,
            'effective_end':    eff_e if has_data else None,
            'has_data':         has_data,
        })

    return result


def parse_comparison_timeframe(
    question: str, available_start: date, available_end: date
) -> dict:
    """
    Detect whether *question* is a period-over-period comparison and, if so,
    extract both periods.

    Returns:
        {'is_comparison': False}  — not a comparison query
        {
          'is_comparison': True,
          'period_a': {...},   # earlier period (baseline)
          'period_b': {...},   # later period (subject)
          'label': 'April 2026 vs May 2026',
        }
    """
    text = (question or '').strip().lower()

    if not is_comparison_query(text):
        return {'is_comparison': False}

    periods = _extract_periods_from_text(text, available_start, available_end)

    if len(periods) < 2:
        return {'is_comparison': False}

    period_a = periods[0]   # older = baseline
    period_b = periods[1]   # newer = subject

    return {
        'is_comparison': True,
        'period_a':      period_a,
        'period_b':      period_b,
        'label':         f"{period_a['label']} vs {period_b['label']}",
    }
