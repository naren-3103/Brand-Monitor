from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from crewai import Crew, Process
from utils.azure_openai_client import build_azure_openai_client


def _to_timestamp(value):
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value
    try:
        return pd.to_datetime(value, errors='coerce')
    except Exception:
        return pd.NaT


def _filter_by_week_start_date(df: pd.DataFrame, start_date, end_date):
    if start_date is None or end_date is None:
        return df
    start_ts = _to_timestamp(start_date)
    end_ts = _to_timestamp(end_date)
    if start_ts is pd.NaT or end_ts is pd.NaT:
        return df
    week_dates = pd.to_datetime(df['week_start_date'], errors='coerce')
    return df[(week_dates >= start_ts) & (week_dates <= end_ts)]

def compute_verified_metrics(social_data, reviews_data, search_data, competitor_data) -> dict:
    """
    Single source of truth for all key brand-health metrics.
    Computed once in Python so every agent uses identical numbers.
    """
    m = {}

    # ── Social ────────────────────────────────────────────────────────────────
    if social_data is not None and len(social_data) > 0:
        sc = social_data['sentiment'].value_counts()
        n = len(social_data)
        m['social_total_posts']    = n
        m['social_positive_count'] = int(sc.get('positive', 0))
        m['social_negative_count'] = int(sc.get('negative', 0))
        m['social_neutral_count']  = int(sc.get('neutral',  0))
        m['social_positive_pct']   = round(m['social_positive_count'] / n * 100, 1)
        m['social_negative_pct']   = round(m['social_negative_count'] / n * 100, 1)
        m['social_neutral_pct']    = round(m['social_neutral_count']  / n * 100, 1)
        m['social_avg_likes']      = round(float(social_data['likes'].mean()),    1)
        m['social_avg_shares']     = round(float(social_data['shares'].mean()),   1)
        m['social_avg_comments']   = round(float(social_data['comments'].mean()), 1)
        m['social_positive_pct_by_platform'] = (
            social_data.groupby('platform')['sentiment']
            .apply(lambda x: round((x == 'positive').sum() / len(x) * 100, 1))
            .to_dict()
        )
        # Weekly positive % for trend reporting
        m['social_weekly_positive_pct'] = (
            social_data.groupby('week')['sentiment']
            .apply(lambda x: round((x == 'positive').sum() / len(x) * 100, 1))
            .to_dict()
        )

    # ── Reviews ───────────────────────────────────────────────────────────────
    if reviews_data is not None and len(reviews_data) > 0:
        n = len(reviews_data)
        m['review_total']            = n
        m['review_avg_rating']       = round(float(reviews_data['star_rating'].mean()), 2)
        m['review_5star_count']      = int((reviews_data['star_rating'] == 5).sum())
        m['review_1star_count']      = int((reviews_data['star_rating'] == 1).sum())
        m['review_pct_positive']     = round((reviews_data['star_rating'] >= 4).sum() / n * 100, 1)
        m['review_pct_negative']     = round((reviews_data['star_rating'] <= 2).sum() / n * 100, 1)
        m['review_avg_by_platform']  = (
            reviews_data.groupby('platform')['star_rating'].mean().round(2).to_dict()
        )

    # ── Search ────────────────────────────────────────────────────────────────
    if search_data is not None and len(search_data) > 0:
        wv = search_data.groupby('week')['search_volume'].sum().sort_index()
        m['search_total_volume']       = int(search_data['search_volume'].sum())
        m['search_avg_weekly_volume']  = int(wv.mean())
        m['search_num_weeks']          = len(wv)
        if len(wv) >= 2:
            m['search_first_week']     = str(wv.index[0])
            m['search_last_week']      = str(wv.index[-1])
            m['search_first_week_vol'] = int(wv.iloc[0])
            m['search_last_week_vol']  = int(wv.iloc[-1])
            fv = m['search_first_week_vol']
            m['search_growth_pct']     = round((m['search_last_week_vol'] - fv) / fv * 100, 1) if fv else 0
        top_kw = search_data.groupby('keyword')['search_volume'].sum().nlargest(1)
        if len(top_kw):
            m['search_top_keyword']     = top_kw.index[0]
            m['search_top_keyword_vol'] = int(top_kw.iloc[0])

    # ── Competitor ────────────────────────────────────────────────────────────
    if competitor_data is not None and len(competitor_data) > 0:
        n = len(competitor_data)
        m['competitor_total_news']        = n
        m['competitor_high_impact_count'] = int(
            (competitor_data['estimated_lays_sentiment_impact'] < -0.05).sum()
        )
        m['competitor_opportunity_count'] = int(
            (competitor_data['estimated_lays_sentiment_impact'] > 0.05).sum()
        )

    return m


# Import all agents
from agents.social_listening_agent import create_social_listening_agent
from agents.search_trend_agent import create_search_trend_agent
from agents.review_theme_agent import create_review_theme_agent
from agents.competitor_monitoring_agent import create_competitor_monitoring_agent
from agents.insight_synthesizer_agent import create_insight_synthesizer_agent
from agents.critic_qa_agent import create_critic_qa_agent

# Import all tasks
from tasks.social_listening_task import create_social_listening_task
from tasks.search_trend_task import create_search_trend_task
from tasks.review_theme_task import create_review_theme_task
from tasks.competitor_monitoring_task import create_competitor_monitoring_task
from tasks.insight_synthesizer_task import create_insight_synthesizer_task
from tasks.critic_qa_task import create_critic_qa_task


def run_brand_health_crew(
    brand: str,
    user_prompt: str,
    llm=None,
    start_date=None,
    end_date=None,
):
    """
    Run the complete 6-agent brand health monitoring crew.
    
    Args:
        brand: Brand name (e.g., "Lay's")
        user_prompt: User's question/request
        llm: Optional LLM instance (uses Azure OpenAI from .env if None)
        start_date: Optional start date for timeframe filtering
        end_date: Optional end date for timeframe filtering
    
    Returns:
        dict with all analysis results
    """
    
    print("📊 Loading data...")
    
    # Load all data
    social_df = pd.read_csv('data/social_posts.csv')
    search_df = pd.read_csv('data/search_trends.csv')
    reviews_df = pd.read_csv('data/reviews.csv')
    competitor_df = pd.read_csv('data/competitor_news.csv')
    
    # Filter for the brand
    social_data = social_df[social_df['brand_mentioned'] == brand]
    search_data = search_df[search_df['brand'] == brand]
    reviews_data = reviews_df[reviews_df['brand'] == brand]
    competitor_data = competitor_df
    
    timeframe_note = ""
    if start_date is not None and end_date is not None:
        social_data = _filter_by_week_start_date(social_data, start_date, end_date)
        search_data = _filter_by_week_start_date(search_data, start_date, end_date)
        reviews_data = _filter_by_week_start_date(reviews_data, start_date, end_date)
        competitor_data = _filter_by_week_start_date(competitor_data, start_date, end_date)
        timeframe_note = (
            f"Timeframe: analyze only data from {start_date:%Y-%m-%d} "
            f"to {end_date:%Y-%m-%d}."
        )

    print(f"✅ Loaded data:")
    print(f"   - {len(social_data):,} social posts")
    print(f"   - {len(search_data):,} search trends")
    print(f"   - {len(reviews_data):,} reviews")
    print(f"   - {len(competitor_data):,} competitor news items")
    
    # Create all 6 agents
    print("\n🤖 Creating agents...")
    
    if llm is None:
        llm = build_azure_openai_client()
    
    social_agent = create_social_listening_agent(llm)
    search_agent = create_search_trend_agent(llm)
    review_agent = create_review_theme_agent(llm)
    competitor_agent = create_competitor_monitoring_agent(llm)
    synthesizer_agent = create_insight_synthesizer_agent(llm)
    critic_agent = create_critic_qa_agent(llm)
    
    print("✅ All 6 agents created")
    
    # Create all tasks
    print("\n📋 Creating tasks...")
    
    prompt_with_timeframe = user_prompt
    if timeframe_note:
        prompt_with_timeframe = f"{user_prompt}\n\n{timeframe_note}"

    # Compute all key metrics once in Python — agents use these exact numbers
    verified_metrics = compute_verified_metrics(social_data, reviews_data, search_data, competitor_data)

    social_task = create_social_listening_task(social_agent, social_data, brand, prompt_with_timeframe, verified_metrics)
    search_task = create_search_trend_task(search_agent, search_data, brand, prompt_with_timeframe, verified_metrics)
    review_task = create_review_theme_task(review_agent, reviews_data, brand, prompt_with_timeframe, verified_metrics)
    competitor_task = create_competitor_monitoring_task(competitor_agent, competitor_data, brand, prompt_with_timeframe, verified_metrics)

    # Synthesizer gets context from all previous tasks + the same verified metrics
    synthesizer_task = create_insight_synthesizer_task(synthesizer_agent, brand, user_prompt, verified_metrics)
    synthesizer_task.context = [social_task, search_task, review_task, competitor_task]
    
    # Critic reviews the synthesizer's output
    critic_task = create_critic_qa_task(
        critic_agent,
        brand,
        user_prompt,
        social_data=social_data,
        review_data=reviews_data,
        search_data=search_data
    )
    critic_task.context = [synthesizer_task]
    
    print("✅ All tasks created")
    
    # Build the crew
    print("\n🚀 Building crew pipeline...")
    
    crew = Crew(
        agents=[
            social_agent,
            search_agent,
            review_agent,
            competitor_agent,
            synthesizer_agent,
            critic_agent
        ],
        tasks=[
            social_task,
            search_task,
            review_task,
            competitor_task,
            synthesizer_task,
            critic_task
        ],
        process=Process.sequential,  # Run agents one after another
        verbose=True
    )
    
    print("✅ Crew ready\n")
    print("=" * 70)
    print("🔥 STARTING 6-AGENT BRAND HEALTH ANALYSIS")
    print("=" * 70)
    
    # Run the crew!
    result = crew.kickoff()

    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)

    # Extract individual task outputs so the full executive report is visible
    # alongside the QA review — not just the critic's output alone.
    tasks_output = getattr(result, 'tasks_output', None) or []
    synthesizer_raw = tasks_output[-2].raw if len(tasks_output) >= 2 else ""
    critic_raw = tasks_output[-1].raw if len(tasks_output) >= 1 else str(result)

    if synthesizer_raw and critic_raw:
        final_report = synthesizer_raw + "\n\n---\n\n" + critic_raw
    else:
        final_report = str(result)

    return {
        "brand": brand,
        "prompt": user_prompt,
        "final_report": final_report,
        "executive_report": synthesizer_raw,
        "qa_review": critic_raw,
        "data_summary": {
            "social_posts": len(social_data),
            "search_trends": len(search_data),
            "reviews": len(reviews_data),
            "competitor_news": len(competitor_data)
        }
    }


if __name__ == "__main__":
    # Test the crew
    result = run_brand_health_crew(
        brand="Lay's",
        user_prompt="How is Lay's brand health overall? What are the top 3 priorities for the next quarter?"
    )
    
    print("\n📊 FINAL BRAND HEALTH REPORT:\n")
    print(result["final_report"])