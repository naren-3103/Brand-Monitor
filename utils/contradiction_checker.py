def detect_contradictions(
    social_sentiment_pct,
    avg_review_rating,
    search_growth_pct=None
):
    """
    Detect contradictory signals across channels
    """

    contradictions = []

    # Social positive but reviews poor
    if social_sentiment_pct > 70 and avg_review_rating < 3:
        contradictions.append(
            "Social sentiment is positive but customer reviews are poor."
        )

    # Search growing but sentiment bad
    if (
        search_growth_pct is not None
        and search_growth_pct > 20
        and social_sentiment_pct < 40
    ):
        contradictions.append(
            "Search interest is growing while social sentiment is declining."
        )

    # Reviews strong but sentiment weak
    if avg_review_rating > 4 and social_sentiment_pct < 40:
        contradictions.append(
            "Customer reviews are strong but social media sentiment is weak."
        )

    return contradictions