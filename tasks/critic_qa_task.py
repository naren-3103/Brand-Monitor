from crewai import Task
from utils.contradiction_checker import detect_contradictions


def create_critic_qa_task(
    agent,
    brand,
    user_prompt,
    social_data=None,
    review_data=None,
    search_data=None
):
    # ── Social metrics ────────────────────────────────────────────────────────
    if social_data is not None and len(social_data) > 0:
        sentiment_counts = social_data['sentiment'].value_counts()
        total_posts = len(social_data)
        positive_count  = int(sentiment_counts.get('positive',  0))
        negative_count  = int(sentiment_counts.get('negative',  0))
        neutral_count   = int(sentiment_counts.get('neutral',   0))
        social_sentiment_pct = round(positive_count / total_posts * 100, 1)
        social_negative_pct  = round(negative_count / total_posts * 100, 1)
        avg_likes    = round(float(social_data['likes'].mean()),    1)
        avg_shares   = round(float(social_data['shares'].mean()),   1)
        avg_comments = round(float(social_data['comments'].mean()), 1)
        platform_sentiment = (
            social_data.groupby('platform')['sentiment']
            .value_counts(normalize=True)
            .unstack(fill_value=0)
            .mul(100).round(1)
            .to_dict('index')
        )
    else:
        total_posts = positive_count = negative_count = neutral_count = 0
        social_sentiment_pct = social_negative_pct = 0
        avg_likes = avg_shares = avg_comments = 0
        platform_sentiment = {}

    # ── Review metrics ────────────────────────────────────────────────────────
    if review_data is not None and len(review_data) > 0:
        total_reviews    = len(review_data)
        avg_review_rating = round(float(review_data['star_rating'].mean()), 2)
        five_star_count  = int((review_data['star_rating'] == 5).sum())
        one_star_count   = int((review_data['star_rating'] == 1).sum())
        pct_positive_reviews = round(
            (review_data['star_rating'] >= 4).sum() / total_reviews * 100, 1
        )
        pct_negative_reviews = round(
            (review_data['star_rating'] <= 2).sum() / total_reviews * 100, 1
        )
        avg_rating_by_platform = (
            review_data.groupby('platform')['star_rating']
            .mean().round(2).to_dict()
        )
    else:
        total_reviews = five_star_count = one_star_count = 0
        avg_review_rating = pct_positive_reviews = pct_negative_reviews = 0
        avg_rating_by_platform = {}

    # ── Search metrics ────────────────────────────────────────────────────────
    if search_data is not None and len(search_data) > 0:
        total_search_volume = int(search_data['search_volume'].sum())
        weekly_volume = search_data.groupby('week')['search_volume'].sum().sort_index()
        avg_weekly_volume = int(weekly_volume.mean())
        num_weeks = len(weekly_volume)
        if num_weeks >= 2:
            first_week_label = str(weekly_volume.index[0])
            last_week_label  = str(weekly_volume.index[-1])
            first_vol = int(weekly_volume.iloc[0])
            last_vol  = int(weekly_volume.iloc[-1])
            search_growth_pct = round(
                (last_vol - first_vol) / first_vol * 100, 1
            ) if first_vol > 0 else 0
        else:
            first_week_label = last_week_label = "N/A"
            first_vol = last_vol = 0
            search_growth_pct = 0
        weekly_volume_dict = {str(k): int(v) for k, v in weekly_volume.to_dict().items()}
        top_keywords = (
            search_data.groupby('keyword')['search_volume']
            .sum().nlargest(5).to_dict()
        )
    else:
        total_search_volume = avg_weekly_volume = num_weeks = 0
        first_week_label = last_week_label = "N/A"
        first_vol = last_vol = 0
        search_growth_pct = 0
        weekly_volume_dict = {}
        top_keywords = {}

    contradictions = detect_contradictions(
        social_sentiment_pct=social_sentiment_pct,
        avg_review_rating=avg_review_rating,
        search_growth_pct=search_growth_pct
    )

    return Task(

        description=f"""
You are the QA reviewer for the brand health analysis of **{brand}**.

USER QUERY:
"{user_prompt}"

==================================================
METRIC DEFINITIONS  (read before validating)
==================================================

SOCIAL
- "Total posts" = count of all posts analysed in the period.
- "Positive sentiment %" = positive posts ÷ total posts × 100.
- "Negative sentiment %" = negative posts ÷ total posts × 100.
- Positive + Negative + Neutral = 100%.  They do NOT contradict each other.

SEARCH
- "Total search volume" = SUM of all weekly volumes across the entire period.
  A large total volume is COMPATIBLE with a negative growth rate.
  Example: 2.4 M total volume with -8% growth means the last week had fewer
  searches than the first week, but the cumulative total is still 2.4 M.
- "Period growth %" = (last week volume − first week volume) ÷ first week volume × 100.
  This is a TREND metric, not a total. DO NOT compare it to the total volume.
- These two figures are NEVER contradictory and MUST NOT be flagged as an issue.

REVIEWS
- "Avg rating" = mean of all star ratings (1–5).
- "% positive reviews" = share of 4–5 star reviews.
- "% negative reviews" = share of 1–2 star reviews.
- A low average rating alongside positive social sentiment is a REAL contradiction.
- A high total review count alongside a low avg rating is NOT a contradiction.

==================================================
VERIFIED LIVE DATA  (ground truth for fact-checking)
==================================================

── SOCIAL ──────────────────────────────────────────
Total posts analysed:   {total_posts:,}
Positive posts:         {positive_count:,}  ({social_sentiment_pct}%)
Negative posts:         {negative_count:,}  ({social_negative_pct}%)
Neutral posts:          {neutral_count:,}
Avg likes per post:     {avg_likes}
Avg shares per post:    {avg_shares}
Avg comments per post:  {avg_comments}
Positive % by platform: {platform_sentiment}

── REVIEWS ─────────────────────────────────────────
Total reviews:          {total_reviews:,}
Average star rating:    {avg_review_rating}/5
5-star reviews:         {five_star_count:,}
1-star reviews:         {one_star_count:,}
% positive (4-5 ★):     {pct_positive_reviews}%
% negative (1-2 ★):     {pct_negative_reviews}%
Avg rating by platform: {avg_rating_by_platform}

── SEARCH ──────────────────────────────────────────
Total search volume:    {total_search_volume:,}   ← SUM of all {num_weeks} weeks
Average weekly volume:  {avg_weekly_volume:,}
First week ({first_week_label}): {first_vol:,}
Last week  ({last_week_label}):  {last_vol:,}
Period growth:          {search_growth_pct:+.1f}%  ← TREND only, not related to total
Top 5 keywords:         {top_keywords}
Weekly volume table:    {weekly_volume_dict}

==================================================
YOUR TASK
==================================================

Perform executive-level QA validation of the brand health analysis above.

CRITICAL RULE — do NOT use the phrase "the report" anywhere in your output.
Instead, quote the exact claim or finding you are reviewing.
For every issue you flag, use this format:
  "Analysis states: '[exact quoted claim]' — Verified value: [correct number from live data]."

Check ONLY the following — do NOT flag things that are not real issues:

1. Fact consistency — does a number stated in the analysis match the verified
   live data above? Flag ONLY if the number differs by more than rounding.
   Quote the exact number stated AND the correct verified value.

2. Unsupported claims — is a specific claim made with NO data behind it?
   Quote the unsupported claim directly.

3. Real contradictions — signals that genuinely conflict across channels
   (e.g. "positive social sentiment" when social_sentiment_pct < 40%).
   Use the CONTRADICTIONS DETECTED section below as your primary guide.
   DO NOT invent contradictions that are not in that list.

4. Weak causal logic — a recommendation stated as fact without reasoning.
   Quote the specific recommendation.

5. Recommendation quality — are the top actions specific and prioritised?

IMPORTANT — do NOT flag as issues:
- Total search volume vs. period growth % (they measure different things — see definitions above).
- High positive sentiment alongside any level of total volume.
- Any metric that simply has a large absolute value.
- Differences that are explainable by the metric definitions above.

==================================================
CONTRADICTIONS DETECTED  (system-verified)
==================================================

{contradictions if contradictions else "No contradictions detected"}

==================================================
FINAL OUTPUT FORMAT
==================================================

## Critic QA Agent

### Quality Score
Give score out of 10.

### Strengths
Cite exact metrics from the analysis that are correct and well-supported.
- Strength 1 — quote the metric and state why it checks out
- Strength 2 — quote the metric and state why it checks out

### Issues Found
Each issue MUST quote the exact claim AND state the correct verified value.
Format: "Stated: '[quoted claim]' — Correct: [verified number]"
- Issue 1
- Issue 2

### Contradictions
Quote the conflicting claims and state which verified figure resolves them.

### Executive QA Summary

        """,

        expected_output=(
            "QA report with contradictions, quality score, and corrections."
        ),

        agent=agent,
    )