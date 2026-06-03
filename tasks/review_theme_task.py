from crewai import Task
import pandas as pd


def create_review_theme_task(agent, review_data, brand, user_prompt, verified_metrics=None):

    total_reviews = len(review_data)

    avg_rating = round(
        review_data['star_rating'].mean(),
        2
    )

    rating_dist = (
        review_data['star_rating']
        .value_counts()
        .sort_index(ascending=False)
        .to_dict()
    )

    platform_dist = (
        review_data['platform']
        .value_counts()
        .to_dict()
    )

    review_data_copy = review_data.copy()

    review_data_copy['review_date'] = pd.to_datetime(
        review_data_copy['review_date']
    )

    recent_reviews = review_data_copy.nlargest(
        5,
        'review_date'
    )[
        ['star_rating', 'review_text', 'platform']
    ].to_dict('records')

    # Positive review samples (4-5 stars)
    positive_pool = review_data[review_data['star_rating'] >= 4]
    positive_reviews = (
        positive_pool
        .sample(min(5, len(positive_pool)), random_state=42)
        [['star_rating', 'review_text', 'platform']]
        .to_dict('records')
    )

    # Negative review samples (1-2 stars)
    negative_pool = review_data[review_data['star_rating'] <= 2]
    negative_reviews = (
        negative_pool
        .sample(min(5, len(negative_pool)), random_state=42)
        [['star_rating', 'review_text', 'platform']]
        .to_dict('records')
    )

    # Average rating by platform
    avg_rating_by_platform = (
        review_data.groupby('platform')['star_rating']
        .mean()
        .round(2)
        .to_dict()
    )

    # Monthly rating trend
    review_data_copy['month'] = (
        review_data_copy['review_date'].dt.to_period('M').astype(str)
    )
    monthly_avg_rating = (
        review_data_copy.groupby('month')['star_rating']
        .mean()
        .round(2)
        .to_dict()
    )

    one_star_count = int((review_data['star_rating'] == 1).sum())
    five_star_count = int((review_data['star_rating'] == 5).sum())

    return Task(

        description=f"""
You are analyzing customer reviews for **{brand}** brand.

USER QUERY:
"{user_prompt}"

==================================================
EXACT VERIFIED FIGURES — COPY VERBATIM, DO NOT RECALCULATE
==================================================

Total reviews:       {verified_metrics.get('review_total', total_reviews):,}
Average rating:      {verified_metrics.get('review_avg_rating', avg_rating)}/5   ← USE THIS EXACT NUMBER
5-star reviews:      {verified_metrics.get('review_5star_count', five_star_count):,}
1-star reviews:      {verified_metrics.get('review_1star_count', one_star_count):,}
% positive (4-5★):   {verified_metrics.get('review_pct_positive', 0)}%   ← USE THIS EXACT NUMBER
% negative (1-2★):   {verified_metrics.get('review_pct_negative', 0)}%   ← USE THIS EXACT NUMBER

Average rating by platform (exact):
{verified_metrics.get('review_avg_by_platform', avg_rating_by_platform)}

==================================================
REVIEW DATA SUMMARY (raw distribution for context)
==================================================

Rating Distribution:
{rating_dist}

Platform Distribution:
{platform_dist}

Monthly Average Rating Trend:
{monthly_avg_rating}

Recent Reviews:
{recent_reviews}

Positive Review Samples (4-5 Stars):
{positive_reviews}

Negative Review Samples (1-2 Stars):
{negative_reviews}

==================================================
YOUR TASK
==================================================

Analyze customer satisfaction and review themes.

Include:

1. Review health
2. Satisfaction drivers
3. Complaints
4. Product issues
5. Platform-specific differences
6. Product improvement opportunities

7. KEY STRENGTHS (MANDATORY)
Identify 2-3 review-based positive signals backed by exact numbers.
Pull directly from the Positive Review Samples and rating data above.
Each strength MUST cite a real figure.
Examples of the required format:
- "Average rating of X.X/5 across X,XXX reviews, with X,XXX five-star reviews (XX%)."
- "Platform [X] leads with average rating of X.X/5."
- "Positive reviews highlight themes of [theme] — sample: '[quote from positive review sample]'."

==================================================
FINAL OUTPUT FORMAT
==================================================

## Review Theme Agent

### Insight

### Evidence
- Point 1 (must include a specific number)
- Point 2 (must include a specific number)
- Point 3 (must include a specific number)

### Key Strengths
- Strength 1 — cite exact metric (e.g. "X.X/5 avg rating; X,XXX five-star reviews")
- Strength 2 — cite a real positive review quote and its platform/rating

### Confidence
High / Medium / Low

### Recommendation

### Executive Summary

        """,

        expected_output=(
            "Structured evidence card with review insights, complaints, "
            "recommendations, and executive summary."
        ),

        agent=agent,
    )