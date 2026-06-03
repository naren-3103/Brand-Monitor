"""
Centralised configuration for the Brand Health Monitor pipeline.

All magic numbers, file paths, and tunable thresholds live here.
Import from this module instead of hard-coding values in tasks/utils.
"""

from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Brand ─────────────────────────────────────────────────────────────────────
BRAND_NAME = "Lay's"

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "data"

DATA_FILES = {
    "social":     DATA_DIR / "social_posts.csv",
    "search":     DATA_DIR / "search_trends.csv",
    "reviews":    DATA_DIR / "reviews.csv",
    "competitor": DATA_DIR / "competitor_news.csv",
    "tracker":    DATA_DIR / "brand_tracker_summary.csv",
    "kpi":        DATA_DIR / "weekly_kpi_dashboard.csv",
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = ROOT_DIR / "logs"
LOG_FILE = LOG_DIR / "pipeline.log"

# ── Social Listening thresholds ───────────────────────────────────────────────
# Weeks where negative sentiment exceeds this % are flagged as spikes
NEGATIVE_SENTIMENT_SPIKE_THRESHOLD = 30  # percent

# ── Anomaly detection ─────────────────────────────────────────────────────────
# Z-score threshold; values beyond this are treated as anomalies
ANOMALY_ZSCORE_THRESHOLD = 2

# ── Competitor monitoring thresholds ─────────────────────────────────────────
# News items with sentiment impact below this value are treated as high threats
COMPETITOR_HIGH_IMPACT_THRESHOLD = -0.05
# News items above this value are treated as competitive opportunities
COMPETITOR_OPPORTUNITY_THRESHOLD = 0.05

# ── Contradiction detection thresholds ───────────────────────────────────────
CONTRADICTION_POSITIVE_SENTIMENT_HIGH = 70   # % above which social is "positive"
CONTRADICTION_REVIEW_RATING_LOW       = 3.0  # avg rating below which reviews are "poor"
CONTRADICTION_REVIEW_RATING_HIGH      = 4.0  # avg rating above which reviews are "strong"
CONTRADICTION_SEARCH_GROWTH_HIGH      = 20   # % growth above which search is "growing"
CONTRADICTION_SENTIMENT_LOW           = 40   # % below which social is "declining"

# ── Crew settings ────────────────────────────────────────────────────────────
CREW_VERBOSE = True

# ── Streamlit / UI ────────────────────────────────────────────────────────────
APP_TITLE  = "Lay's Brand Health Monitor"
APP_ICON   = "🥔"
APP_LAYOUT = "wide"
