from crewai import Task
import pandas as pd


def create_competitor_monitoring_task(
    agent,
    competitor_data,
    brand,
    user_prompt,
    verified_metrics=None
):

    total_news = len(competitor_data)

    competitor_breakdown = (
        competitor_data['competitor_brand']
        .value_counts()
        .to_dict()
    )

    competitor_data_copy = competitor_data.copy()

    competitor_data_copy['week_start_date'] = pd.to_datetime(
        competitor_data_copy['week_start_date']
    )

    recent_news = competitor_data_copy.nlargest(
        5,
        'week_start_date'
    )[
        [
            'competitor_brand',
            'headline',
            'category',
            'estimated_lays_sentiment_impact'
        ]
    ].to_dict('records')

    high_impact = competitor_data[
        competitor_data[
            'estimated_lays_sentiment_impact'
        ] < -0.05
    ]

    # All high-impact threat headlines
    high_impact_headlines = high_impact[
        ['competitor_brand', 'headline', 'category', 'estimated_lays_sentiment_impact']
    ].sort_values('estimated_lays_sentiment_impact').to_dict('records')

    # Average sentiment impact per competitor
    avg_impact_by_competitor = (
        competitor_data.groupby('competitor_brand')['estimated_lays_sentiment_impact']
        .mean()
        .round(3)
        .sort_values()
        .to_dict()
    )

    # Most threatening competitor (lowest avg impact)
    most_threatening = (
        min(avg_impact_by_competitor, key=avg_impact_by_competitor.get)
        if avg_impact_by_competitor else "N/A"
    )

    # News category breakdown
    category_breakdown = (
        competitor_data['category']
        .value_counts()
        .to_dict()
    )

    # Category breakdown per competitor
    category_by_competitor = (
        competitor_data.groupby(['competitor_brand', 'category'])
        .size()
        .unstack(fill_value=0)
        .to_dict('index')
    )

    # Positive opportunities: competitor news with positive impact on brand
    positive_impact_news = competitor_data[
        competitor_data['estimated_lays_sentiment_impact'] > 0.05
    ][
        ['competitor_brand', 'headline', 'category', 'estimated_lays_sentiment_impact']
    ].to_dict('records')

    if verified_metrics is None:
        verified_metrics = {}

    return Task(

        description=f"""
You are monitoring competitors for **{brand}**.

USER QUERY:
"{user_prompt}"

==================================================
EXACT VERIFIED FIGURES — COPY VERBATIM, DO NOT RECALCULATE
==================================================

Total competitor news items:  {verified_metrics.get('competitor_total_news', total_news):,}   ← USE THIS EXACT NUMBER
High-impact threats (< -0.05): {verified_metrics.get('competitor_high_impact_count', len(high_impact)):,}   ← USE THIS EXACT NUMBER
Competitive opportunities (> +0.05): {verified_metrics.get('competitor_opportunity_count', len(positive_impact_news)):,}   ← USE THIS EXACT NUMBER

==================================================
COMPETITOR SUMMARY (raw data for context)
==================================================

Total Competitor News:
{total_news}

Competitor Breakdown:
{competitor_breakdown}

Most Threatening Competitor: {most_threatening}

Average Sentiment Impact by Competitor:
{avg_impact_by_competitor}

High Impact Threats (impact < -0.05):
{len(high_impact)}

All High-Impact Threat Headlines:
{high_impact_headlines}

Competitive Opportunities (impact > +0.05):
{len(positive_impact_news)} items
{positive_impact_news}

News Category Breakdown:
{category_breakdown}

Category Activity by Competitor:
{category_by_competitor}

Recent Competitor Activity:
{recent_news}

==================================================
YOUR TASK
==================================================

Analyze:

1. Competitive landscape
2. Threat assessment
3. Pricing/promotional attacks
4. Product launches
5. Competitive opportunities
6. Strategic response actions

7. KEY STRENGTHS / COMPETITIVE ADVANTAGES (MANDATORY)
Identify 2-3 competitive advantages or opportunities for {brand} backed by exact data.
Pull from the Competitive Opportunities list and sentiment impact data above.
Each strength MUST cite a real figure.
Examples of the required format:
- "X competitor news items had a positive sentiment impact on {brand} (impact > +0.05)."
- "Only X high-impact threats this period vs X the prior period — competitive pressure is easing."
- "Competitor [X]'s [category] activity creates a direct opening for {brand} in [area]."

==================================================
FINAL OUTPUT FORMAT
==================================================

## Competitor Monitoring Agent

### Insight

### Evidence
- Point 1 (must include a specific number)
- Point 2 (must include a specific number)
- Point 3 (must include a specific number)

### Key Strengths
- Strength 1 — cite exact competitive advantage with data
- Strength 2 — cite exact opportunity with data

### Confidence
High / Medium / Low

### Recommendation

### Executive Summary

        """,

        expected_output=(
            "Structured competitor intelligence evidence card "
            "with threats, opportunities, recommendations, "
            "and executive summary."
        ),

        agent=agent,
    )