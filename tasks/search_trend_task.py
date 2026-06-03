from crewai import Task
from utils.anomaly_detection import detect_anomaly


def create_search_trend_task(agent, search_data, brand, user_prompt, verified_metrics=None):

    total_searches = int(search_data['search_volume'].sum())

    top_keywords = (
        search_data
        .groupby('keyword')['search_volume']
        .sum()
        .nlargest(5)
        .to_dict()
    )

    avg_volume = int(search_data['search_volume'].mean())

    weekly_volume = (
        search_data
        .groupby('week')['search_volume']
        .sum()
    )

    anomaly_result = detect_anomaly(weekly_volume)

    # Bottom 5 declining keywords
    bottom_keywords = (
        search_data
        .groupby('keyword')['search_volume']
        .sum()
        .nsmallest(5)
        .to_dict()
    )

    # All keywords ranked by volume
    all_keywords_ranked = (
        search_data
        .groupby('keyword')['search_volume']
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )

    # Week-over-week growth: first vs last week
    sorted_weeks = weekly_volume.sort_index()
    first_week_vol = int(sorted_weeks.iloc[0]) if len(sorted_weeks) > 0 else 0
    last_week_vol = int(sorted_weeks.iloc[-1]) if len(sorted_weeks) > 0 else 0
    wow_growth_pct = (
        round((last_week_vol - first_week_vol) / first_week_vol * 100, 1)
        if first_week_vol > 0 else 0
    )

    # Peak week
    peak_week = str(weekly_volume.idxmax()) if len(weekly_volume) > 0 else "N/A"
    peak_volume = int(weekly_volume.max()) if len(weekly_volume) > 0 else 0

    # Per-keyword weekly trend (top 5 keywords)
    top_keyword_names = list(top_keywords.keys())
    keyword_weekly_trend = (
        search_data[search_data['keyword'].isin(top_keyword_names)]
        .groupby(['week', 'keyword'])['search_volume']
        .sum()
        .unstack(fill_value=0)
        .to_dict()
    )

    if verified_metrics is None:
        verified_metrics = {}

    return Task(

        description=f"""
You are analyzing search trend data for **{brand}** brand.

USER QUERY:
"{user_prompt}"

==================================================
EXACT VERIFIED FIGURES — COPY VERBATIM, DO NOT RECALCULATE
==================================================

Total search volume:    {verified_metrics.get('search_total_volume', total_searches):,}   ← USE THIS EXACT NUMBER
Avg weekly volume:      {verified_metrics.get('search_avg_weekly_volume', avg_volume):,}
Number of weeks:        {verified_metrics.get('search_num_weeks', len(weekly_volume))}
First week ({verified_metrics.get('search_first_week', 'N/A')}): {verified_metrics.get('search_first_week_vol', first_week_vol):,}
Last week  ({verified_metrics.get('search_last_week',  'N/A')}): {verified_metrics.get('search_last_week_vol',  last_week_vol):,}
Period growth:          {verified_metrics.get('search_growth_pct', wow_growth_pct):+.1f}%   ← USE THIS EXACT NUMBER
Top keyword:            {verified_metrics.get('search_top_keyword', 'N/A')} ({verified_metrics.get('search_top_keyword_vol', 0):,} searches)   ← USE THIS EXACT NUMBER

==================================================
SEARCH TREND SUMMARY (raw data for context)
==================================================

Total Search Volume:
{total_searches:,}

Average Weekly Search Volume:
{avg_volume:,}

Period Growth (First Week → Last Week):
{wow_growth_pct:+.1f}% ({first_week_vol:,} → {last_week_vol:,})

Peak Week: {peak_week} ({peak_volume:,} searches)

Top 5 Keywords by Volume:
{top_keywords}

Bottom 5 Keywords by Volume:
{bottom_keywords}

All Keywords Ranked:
{all_keywords_ranked}

Weekly Search Volume:
{weekly_volume.to_dict()}

Top Keyword Weekly Trends:
{keyword_weekly_trend}

==================================================
ANOMALY DETECTION
==================================================

{anomaly_result}

==================================================
YOUR TASK
==================================================

Analyze search behavior and consumer intent.

Include:

1. Overall search health
2. Rising keywords
3. Consumer intent
4. Search anomalies
5. SEO/content opportunities
6. Strategic recommendations

7. KEY STRENGTHS (MANDATORY)
Identify 2-3 search-based positive signals backed by exact numbers.
Each strength MUST cite a real figure from the data above.
Examples of the required format:
- "Total search volume was X,XXX for the period, [X%] above/below average."
- "Top keyword '[keyword]' drove X,XXX searches, the highest volume keyword."
- "Period growth of +XX% from week X (X,XXX) to week X (X,XXX) signals rising consumer interest."

==================================================
FINAL OUTPUT FORMAT
==================================================

## Search Trend Agent

### Insight

### Evidence
- Point 1 (must include a specific number)
- Point 2 (must include a specific number)
- Point 3 (must include a specific number)

### Key Strengths
- Strength 1 — cite exact metric (e.g. "Top keyword '[keyword]': X,XXX searches")
- Strength 2 — cite exact metric (e.g. "+XX% search growth from week X to week X")

### Confidence
High / Medium / Low

### Recommendation

### Executive Summary

        """,

        expected_output=(
            "Structured evidence card with insights, evidence, confidence, "
            "recommendations, anomalies, and executive summary."
        ),

        agent=agent,
    )