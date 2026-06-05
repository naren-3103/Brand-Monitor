from crewai import Task


def create_social_listening_task(agent, social_data, brand, user_prompt, verified_metrics=None):
    """
    Task for Social Listening Agent
    Analyzes social media posts from the synthetic data
    """

    # Convert DataFrame to summary string for the prompt
    total_posts = len(social_data)

    platforms = social_data['platform'].value_counts().to_dict()

    sentiment_dist = social_data['sentiment'].value_counts().to_dict()

    avg_engagement = {
        'likes': int(social_data['likes'].mean()),
        'shares': int(social_data['shares'].mean()),
        'comments': int(social_data['comments'].mean())
    }

    # Detect negative spikes (weeks with >30% negative sentiment)
    weekly_sentiment = social_data.groupby('week')['sentiment'].apply(
        lambda x: (x == 'negative').sum() / len(x) * 100
    )

    negative_spikes = weekly_sentiment[
        weekly_sentiment > 30
    ].to_dict()

    # Weekly sentiment trend (positive %)
    weekly_positive_pct = social_data.groupby('week')['sentiment'].apply(
        lambda x: round((x == 'positive').sum() / len(x) * 100, 1)
    ).to_dict()

    # Sentiment breakdown by platform
    sentiment_by_platform = (
        social_data.groupby(['platform', 'sentiment'])
        .size()
        .unstack(fill_value=0)
        .to_dict('index')
    )

    # Top positive posts (highest likes)
    positive_posts = (
        social_data[social_data['sentiment'] == 'positive']
        .nlargest(3, 'likes')[['platform', 'text', 'likes', 'shares', 'comments']]
        .to_dict('records')
    )

    # Top negative posts (highest engagement — most viral complaints)
    social_data_copy = social_data.copy()
    social_data_copy['total_engagement'] = (
        social_data_copy['likes']
        + social_data_copy['shares']
        + social_data_copy['comments']
    )
    negative_posts = (
        social_data_copy[social_data_copy['sentiment'] == 'negative']
        .nlargest(3, 'total_engagement')[['platform', 'text', 'likes', 'shares', 'comments']]
        .to_dict('records')
    )

    # Most viral posts overall
    top_viral_posts = (
        social_data_copy
        .nlargest(3, 'total_engagement')[['platform', 'sentiment', 'text', 'likes', 'shares', 'comments']]
        .to_dict('records')
    )

    return Task(

        description=f"""
You are analyzing social media data for **{brand}** brand.

USER QUERY:
"{user_prompt}"

==================================================
EXACT VERIFIED FIGURES — COPY VERBATIM, DO NOT RECALCULATE
==================================================

Total posts:        {verified_metrics.get('social_total_posts', total_posts):,}
Positive posts:     {verified_metrics.get('social_positive_count', 0):,}
Negative posts:     {verified_metrics.get('social_negative_count', 0):,}
Neutral posts:      {verified_metrics.get('social_neutral_count',  0):,}

Positive sentiment: {verified_metrics.get('social_positive_pct', 0)}%   ← USE THIS EXACT NUMBER
Negative sentiment: {verified_metrics.get('social_negative_pct', 0)}%   ← USE THIS EXACT NUMBER
Neutral:            {verified_metrics.get('social_neutral_pct',  0)}%

Avg likes/post:     {verified_metrics.get('social_avg_likes', 0)}
Avg shares/post:    {verified_metrics.get('social_avg_shares', 0)}
Avg comments/post:  {verified_metrics.get('social_avg_comments', 0)}

Positive % by platform (exact):
{verified_metrics.get('social_positive_pct_by_platform', {})}

Weekly positive sentiment % (exact):
{verified_metrics.get('social_weekly_positive_pct', {})}

==================================================
SOCIAL MEDIA DATA SUMMARY (raw counts for context)
==================================================

Platform Breakdown:
{platforms}

Engagement per Post:
{avg_engagement}

Negative Sentiment Spikes:
{len(negative_spikes)} weeks exceeded 30% negative sentiment.

Weekly Negative Sentiment Details:
{negative_spikes if negative_spikes else "No major spikes detected"}

Sentiment Breakdown by Platform (counts):
{sentiment_by_platform}

Top Positive Posts (by Likes):
{positive_posts}

Top Negative Posts (by Total Engagement):
{negative_posts}

Most Viral Posts Overall:
{top_viral_posts}

==================================================
YOUR TASK
==================================================

Perform a detailed executive-level social listening analysis.

Your analysis MUST include:

1. OVERALL SOCIAL HEALTH
- Is the brand perception healthy or declining?
- What is the dominant customer emotion?
- Is the sentiment trend improving or worsening?

2. PLATFORM PERFORMANCE
- Which platform has highest engagement?
- Which platform has highest negative sentiment?
- Which platform deserves more investment?
- Strategic recommendation by platform.

3. NEGATIVE SPIKE ANALYSIS
- Identify negative spikes.
- Explain possible reasons.
- Estimate severity of impact.
- Mention if issue looks temporary or systemic.

4. ENGAGEMENT ANALYSIS
- Analyze likes, comments, and shares.
- What does engagement quality indicate?
- Are customers emotionally engaged or passive?

5. KEY RISKS
- Mention top reputation risks.
- Mention any potential PR concerns.

6. ACTIONABLE RECOMMENDATIONS
Provide 2-3 highly specific actions marketing team should take immediately.

7. KEY STRENGTHS (MANDATORY)
Identify 2-3 concrete positive signals backed by real numbers.
Each strength MUST cite an exact metric from the data above.
Examples of the required format:
- "Positive sentiment is XX% (X,XXX of X,XXX posts), highest on [platform] at XX%."
- "Top positive post on [platform] drove [X] likes and [X] shares, indicating strong organic advocacy."
- "No negative spikes in [X] consecutive weeks, suggesting stable brand perception."

==================================================
IMPORTANT INSTRUCTIONS
==================================================

- Use REAL numbers from the dataset.
- Be highly specific.
- Think like a senior FMCG brand strategist.
- Do NOT give generic marketing advice.
- Focus on business impact.
- Keep analysis executive-ready.

==================================================
FINAL OUTPUT FORMAT (MANDATORY)
==================================================

## Social Listening Agent

### Insight
Write the main insight here.

### Evidence
- Evidence point 1 (must include a specific number)
- Evidence point 2 (must include a specific number)
- Evidence point 3 (must include a specific number)

### Key Strengths
- Strength 1 — cite exact metric (e.g. "XX% positive sentiment across X,XXX posts")
- Strength 2 — cite exact metric (e.g. "Top positive post: X likes, X shares on [platform]")

### Confidence
High / Medium / Low

### Recommendation
Write recommendation here.

### Executive Summary
Write 1 concise executive summary paragraph.

        """,

        expected_output=(
            "Structured markdown evidence card containing:\n"
            "- Agent Name\n"
            "- Key Insight\n"
            "- Supporting Evidence\n"
            "- Confidence Level\n"
            "- Recommendation\n"
            "- Executive Summary\n"
        ),

        agent=agent,
    )