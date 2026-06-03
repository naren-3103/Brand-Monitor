from crewai import Task


def create_insight_synthesizer_task(
    agent,
    brand,
    user_prompt,
    verified_metrics=None,
    specialist_outputs: dict = None,
    critic_feedback: str = None,
    iteration: int = 1,
):

    if verified_metrics is None:
        verified_metrics = {}

    # Build verified metrics block — only include metrics that are present
    _vm = verified_metrics
    _social_block = ""
    if _vm.get('social_total_posts'):
        _social_block = f"""
── SOCIAL ──────────────────────────────────────────────────────
Total posts:            {_vm.get('social_total_posts', 0):,}
Positive sentiment:     {_vm.get('social_positive_pct', 0)}%   ← COPY VERBATIM
Negative sentiment:     {_vm.get('social_negative_pct', 0)}%   ← COPY VERBATIM
Neutral:                {_vm.get('social_neutral_pct', 0)}%
Avg likes/post:         {_vm.get('social_avg_likes', 0)}
Avg shares/post:        {_vm.get('social_avg_shares', 0)}
Avg comments/post:      {_vm.get('social_avg_comments', 0)}
Positive % by platform: {_vm.get('social_positive_pct_by_platform', {})}"""

    _review_block = ""
    if _vm.get('review_total'):
        _review_block = f"""
── REVIEWS ─────────────────────────────────────────────────────
Total reviews:          {_vm.get('review_total', 0):,}
Average star rating:    {_vm.get('review_avg_rating', 0)}/5   ← COPY VERBATIM
5-star reviews:         {_vm.get('review_5star_count', 0):,}
1-star reviews:         {_vm.get('review_1star_count', 0):,}
% positive (4-5★):      {_vm.get('review_pct_positive', 0)}%   ← COPY VERBATIM
% negative (1-2★):      {_vm.get('review_pct_negative', 0)}%   ← COPY VERBATIM
Avg rating by platform: {_vm.get('review_avg_by_platform', {})}"""

    _search_block = ""
    if _vm.get('search_total_volume'):
        _search_block = f"""
── SEARCH ──────────────────────────────────────────────────────
Total search volume:    {_vm.get('search_total_volume', 0):,}   ← COPY VERBATIM
Avg weekly volume:      {_vm.get('search_avg_weekly_volume', 0):,}
Period growth:          {_vm.get('search_growth_pct', 0):+.1f}%   ← COPY VERBATIM
Top keyword:            {_vm.get('search_top_keyword', 'N/A')} ({_vm.get('search_top_keyword_vol', 0):,} searches)"""

    _competitor_block = ""
    if _vm.get('competitor_total_news'):
        _competitor_block = f"""
── COMPETITOR ──────────────────────────────────────────────────
Total competitor news:  {_vm.get('competitor_total_news', 0):,}
High-impact threats:    {_vm.get('competitor_high_impact_count', 0):,}
Opportunities:          {_vm.get('competitor_opportunity_count', 0):,}"""

    # ── Specialist outputs block ──────────────────────────────────────────────
    _specialist_block = ""
    if specialist_outputs:
        parts = []
        label_map = {
            'social':     'Social Listening Agent',
            'search':     'Search Trend Agent',
            'review':     'Review Theme Agent',
            'competitor': 'Competitor Monitoring Agent',
        }
        for key, label in label_map.items():
            text = (specialist_outputs.get(key) or "").strip()
            if text:
                parts.append(f"### {label}\n{text}")
        if parts:
            _specialist_block = (
                "\n==================================================\n"
                "SPECIALIST AGENT EVIDENCE CARDS\n"
                "==================================================\n\n"
                + "\n\n".join(parts)
            )

    # ── Critic feedback block (only for iteration 2+) ─────────────────────────
    _critic_block = ""
    if critic_feedback and iteration > 1:
        _critic_block = f"""
==================================================
CRITIC QA FEEDBACK FROM ITERATION {iteration - 1} — MANDATORY TO ADDRESS
==================================================

{critic_feedback}

MANDATORY INSTRUCTIONS FOR THIS REVISION:
- Address EVERY issue listed in "### Issues Found" above.
- For each corrected item, show the exact verified number.
- Do NOT repeat errors from the previous iteration.
- This is revision {iteration} — your score must improve.
==================================================
"""

    _task_header = (
        f"REVISION {iteration} — Address all critic issues and produce an improved report."
        if iteration > 1
        else "Create a final executive brand-health report."
    )

    return Task(

        description=f"""
You are the Chief Brand Strategist for **{brand}**.

USER QUERY:
"{user_prompt}"

==================================================
VERIFIED METRICS — NEVER RECALCULATE THESE NUMBERS
Copy these exact figures into your report. Do NOT derive, round differently, or
recompute any of these values. If a number appears below, use it as-is.
==================================================
{_social_block}
{_review_block}
{_search_block}
{_competitor_block}
{_specialist_block}
{_critic_block}
==================================================
YOUR TASK
==================================================

{_task_header}

Include:

1. Executive summary
2. Brand health score
3. Key strengths  ← MUST be built from the ### Key Strengths sections of the 4 evidence cards
4. Key risks
5. Strategic priorities
6. Recommended actions
7. Future outlook

==================================================
IMPORTANT
==================================================

- Every claim must reference evidence.
- For the Key Strengths section: read the "### Key Strengths" section in EACH of the 4 evidence cards and consolidate them. Every strength bullet MUST include the exact metric or data point from the evidence card (e.g. "XX% positive sentiment across X,XXX posts — Social Listening Agent").
- Do NOT write a generic strength like "strong brand awareness" without a number. If no number exists, omit that strength.
- Use specialist findings.
- Think like a CMO.
- Keep recommendations actionable.
- Be concise and executive-ready.
- Address the USER QUERY directly.
- Do NOT make data quality issues the main story.
- Only mention data limitations if they materially affect confidence, and keep that mention to a single sentence in Outlook.
- If you include a data quality note, it must not appear in the Executive Summary or Recommendations.
- Do NOT use phrases like "this report", "the report shows", or "as noted in this report". State findings directly: "Positive sentiment is XX%" not "the report states positive sentiment is XX%".

==================================================
FINAL OUTPUT FORMAT
==================================================

# WEEKLY BRAND HEALTH REPORT

## Executive Summary
Write a concise, tailored executive summary paragraph that directly answers the user query and highlights the top business conclusion. Do not mention data inconsistencies or data quality issues in this section.

## Brand Health Score

## Key Strengths
Each bullet MUST contain a specific number and name the source agent.
Format: "- [Strength statement with exact metric] — [Source Agent]"
Example: "- Positive sentiment at XX.X% across X,XXX posts — Social Listening Agent" (use the exact % from the VERIFIED METRICS block above)

## Critical Risks
- Risk 1
- Risk 2

## Strategic Priorities
- Priority 1
- Priority 2

## Recommended Actions
- Action 1
- Action 2

## Outlook

        """,

        expected_output=(
            "Executive brand health report with strengths, "
            "risks, recommendations, and outlook."
        ),

        agent=agent,
    )