from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from crewai import Crew, Process
from utils.azure_openai_client import build_azure_openai_client
from utils.feedback_loop import (
    FeedbackLoopResult,
    IterationRecord,
    parse_quality_score,
    extract_critic_issues,
)


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


def _run_specialist_phase(
    brand, social_data, search_data, reviews_data, competitor_data,
    prompt_with_timeframe, verified_metrics, llm
) -> dict:
    """
    Run the 4 specialist agents in parallel using a thread pool.

    Each specialist runs as its own single-agent Crew so they are fully
    independent. ThreadPoolExecutor fires all 4 at once; the wall-clock
    time becomes the slowest agent's time instead of the sum of all four.

    Process.parallel does not exist in CrewAI 0.x, so we use
    concurrent.futures as the parallelism layer instead.
    """

    # Descriptor: (output_key, agent_factory, task_factory, data)
    specs = [
        ('social',     create_social_listening_agent,     create_social_listening_task,     social_data),
        ('search',     create_search_trend_agent,         create_search_trend_task,         search_data),
        ('review',     create_review_theme_agent,         create_review_theme_task,         reviews_data),
        ('competitor', create_competitor_monitoring_agent, create_competitor_monitoring_task, competitor_data),
    ]

    def _run_one(key, create_agent_fn, create_task_fn, data):
        agent = create_agent_fn(llm)
        task  = create_task_fn(agent, data, brand, prompt_with_timeframe, verified_metrics)
        crew  = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()
        outputs = getattr(result, 'tasks_output', [])
        raw = outputs[0].raw if outputs else ""
        print(f"  ✅ [{key}] specialist complete")
        return key, raw

    results = {key: "" for key, *_ in specs}

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_run_one, *spec): spec[0] for spec in specs}
        for future in as_completed(futures):
            key = futures[future]
            try:
                key, raw = future.result()
                results[key] = raw
            except Exception as exc:
                print(f"  ⚠️  [{key}] specialist failed: {exc}")

    return results


def _run_synthesis_iteration(
    brand, user_prompt, verified_metrics,
    specialist_outputs: dict,
    social_data, reviews_data, search_data,
    llm,
    iteration: int,
    max_iterations: int,
    critic_feedback: str = None,
) -> tuple:
    """
    Run one Synthesizer → Critic cycle.
    Returns (synthesizer_raw, critic_raw, quality_score).
    """
    synthesizer_agent = create_insight_synthesizer_agent(llm)
    critic_agent      = create_critic_qa_agent(llm)

    synthesizer_task = create_insight_synthesizer_task(
        synthesizer_agent,
        brand,
        user_prompt,
        verified_metrics,
        specialist_outputs=specialist_outputs,
        critic_feedback=critic_feedback,
        iteration=iteration,
    )

    critic_task = create_critic_qa_task(
        critic_agent,
        brand,
        user_prompt,
        social_data=social_data,
        review_data=reviews_data,
        search_data=search_data,
        iteration=iteration,
        max_iterations=max_iterations,
    )
    critic_task.context = [synthesizer_task]

    crew = Crew(
        agents=[synthesizer_agent, critic_agent],
        tasks=[synthesizer_task, critic_task],
        process=Process.sequential,
        verbose=True,
    )
    result = crew.kickoff()

    outputs = getattr(result, 'tasks_output', [])
    synthesizer_raw = outputs[0].raw if len(outputs) >= 1 else ""
    critic_raw      = outputs[1].raw if len(outputs) >= 2 else str(result)
    quality_score   = parse_quality_score(critic_raw)

    return synthesizer_raw, critic_raw, quality_score


def run_brand_health_crew(
    brand: str,
    user_prompt: str,
    llm=None,
    start_date=None,
    end_date=None,
    max_feedback_iterations: int = 3,
    quality_threshold: float = 7.5,
):
    """
    Run the complete brand health monitoring pipeline with a closed feedback loop.

    Phase 1 — Specialist agents (run once):
      Social Listening → Search Trend → Review Theme → Competitor Monitoring

    Phase 2 — Feedback loop (up to max_feedback_iterations):
      Synthesizer → Critic QA → parse quality score
        if score < quality_threshold: inject critic issues back → re-synthesize
        if score >= quality_threshold OR max iterations reached: exit

    Returns a dict that includes iteration_history for the Observability tab.
    """

    print("📊 Loading data...")

    social_df     = pd.read_csv('data/social_posts.csv')
    search_df     = pd.read_csv('data/search_trends.csv')
    reviews_df    = pd.read_csv('data/reviews.csv')
    competitor_df = pd.read_csv('data/competitor_news.csv')

    social_data     = social_df[social_df['brand_mentioned'] == brand]
    search_data     = search_df[search_df['brand'] == brand]
    reviews_data    = reviews_df[reviews_df['brand'] == brand]
    competitor_data = competitor_df

    timeframe_note = ""
    if start_date is not None and end_date is not None:
        social_data     = _filter_by_week_start_date(social_data,     start_date, end_date)
        search_data     = _filter_by_week_start_date(search_data,     start_date, end_date)
        reviews_data    = _filter_by_week_start_date(reviews_data,    start_date, end_date)
        competitor_data = _filter_by_week_start_date(competitor_data, start_date, end_date)
        timeframe_note  = (
            f"Timeframe: analyze only data from {start_date:%Y-%m-%d} "
            f"to {end_date:%Y-%m-%d}."
        )

    print(f"✅ Loaded data:")
    print(f"   - {len(social_data):,} social posts")
    print(f"   - {len(search_data):,} search trends")
    print(f"   - {len(reviews_data):,} reviews")
    print(f"   - {len(competitor_data):,} competitor news items")

    if llm is None:
        llm = build_azure_openai_client()

    prompt_with_timeframe = user_prompt
    if timeframe_note:
        prompt_with_timeframe = f"{user_prompt}\n\n{timeframe_note}"

    verified_metrics = compute_verified_metrics(
        social_data, reviews_data, search_data, competitor_data
    )

    # ── Phase 1: Specialist agents (run once) ─────────────────────────────────
    print("\n" + "=" * 70)
    print("🔍 PHASE 1 — SPECIALIST AGENTS (run once)")
    print("=" * 70)

    specialist_outputs = _run_specialist_phase(
        brand, social_data, search_data, reviews_data, competitor_data,
        prompt_with_timeframe, verified_metrics, llm,
    )

    # ── Phase 2: Synthesizer → Critic feedback loop ───────────────────────────
    print("\n" + "=" * 70)
    print("🔄 PHASE 2 — SYNTHESIZER ↔ CRITIC FEEDBACK LOOP")
    print(f"   Max iterations: {max_feedback_iterations}  |  Quality threshold: {quality_threshold}/10")
    print("=" * 70)

    loop_result = FeedbackLoopResult()
    critic_feedback = None  # None on first iteration

    for iteration in range(1, max_feedback_iterations + 1):
        print(f"\n🔁 Feedback Loop — Iteration {iteration}/{max_feedback_iterations}")

        synthesizer_raw, critic_raw, quality_score = _run_synthesis_iteration(
            brand=brand,
            user_prompt=user_prompt,
            verified_metrics=verified_metrics,
            specialist_outputs=specialist_outputs,
            social_data=social_data,
            reviews_data=reviews_data,
            search_data=search_data,
            llm=llm,
            iteration=iteration,
            max_iterations=max_feedback_iterations,
            critic_feedback=critic_feedback,
        )

        record = IterationRecord(
            iteration=iteration,
            synthesizer_output=synthesizer_raw,
            critic_output=critic_raw,
            quality_score=quality_score,
        )
        loop_result.append(record)

        print(f"   ✅ Quality score: {quality_score}/10")

        if quality_score >= quality_threshold:
            loop_result.converged = True
            print(f"   🎯 Quality threshold reached ({quality_score} >= {quality_threshold}) — stopping loop")
            break

        if iteration < max_feedback_iterations:
            critic_feedback = extract_critic_issues(critic_raw)
            print(f"   ↩️  Score below threshold — running revision {iteration + 1}")
        else:
            print(f"   ⚠️  Max iterations reached — using best result")

    print("\n" + "=" * 70)
    print(f"✅ FEEDBACK LOOP COMPLETE  |  Final score: {loop_result.final_score}/10  "
          f"|  Iterations: {len(loop_result.iterations)}")
    print("=" * 70)

    synthesizer_raw = loop_result.final_synthesizer
    critic_raw      = loop_result.final_critic

    if synthesizer_raw and critic_raw:
        final_report = synthesizer_raw + "\n\n---\n\n" + critic_raw
    else:
        final_report = synthesizer_raw or critic_raw

    return {
        "brand":            brand,
        "prompt":           user_prompt,
        "final_report":     final_report,
        "executive_report": synthesizer_raw,
        "qa_review":        critic_raw,
        "data_summary": {
            "social_posts":    len(social_data),
            "search_trends":   len(search_data),
            "reviews":         len(reviews_data),
            "competitor_news": len(competitor_data),
        },
        "feedback_loop": {
            "iterations":         len(loop_result.iterations),
            "converged":          loop_result.converged,
            "quality_threshold":  quality_threshold,
            "score_progression":  loop_result.score_progression(),
            "iteration_history":  [
                {
                    "iteration":          r.iteration,
                    "quality_score":      r.quality_score,
                    "improved":           r.improved,
                    "synthesizer_output": r.synthesizer_output,
                    "critic_output":      r.critic_output,
                }
                for r in loop_result.iterations
            ],
        },
    }


if __name__ == "__main__":
    result = run_brand_health_crew(
        brand="Lay's",
        user_prompt="How is Lay's brand health overall? What are the top 3 priorities for the next quarter?"
    )

    print("\n📊 FINAL BRAND HEALTH REPORT:\n")
    print(result["final_report"])
    print(f"\n📈 Score progression: {result['feedback_loop']['score_progression']}")
