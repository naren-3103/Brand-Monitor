from dotenv import load_dotenv

load_dotenv()



from pathlib import Path
# Edited by Copilot: enable apply-filters feature

import streamlit as st

import pandas as pd

import plotly.graph_objects as go

from utils.azure_openai_client import build_azure_openai_client

from crew import run_brand_health_crew

from tasks.relevance_checker import is_question_relevant

from utils.timeframe_utils import parse_timeframe, format_date_availability, get_data_availability



def get_data_signature():

    data_files = [

        Path('data/social_posts.csv'),

        Path('data/search_trends.csv'),

        Path('data/reviews.csv'),

        Path('data/competitor_news.csv'),

        Path('data/brand_tracker_summary.csv'),

        Path('data/weekly_kpi_dashboard.csv'),

    ]

    return tuple(p.stat().st_mtime if p.exists() else 0 for p in data_files)



# Azure OpenAI Client

client = build_azure_openai_client()



# Page Config

st.set_page_config(
    page_title="Lay's Brand Health Monitor",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS

st.markdown("""

<style>

    .main-header {

        font-size: 2.5rem;

        font-weight: bold;

        color: #FF6B35;

        text-align: center;

        margin-bottom: 1rem;

    }



    .metric-card {

        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

        padding: 1rem 0.75rem;

        border-radius: 12px;

        color: white;

        text-align: center;

        box-shadow: 0 4px 6px rgba(0,0,0,0.1);

        min-height: 110px;

        display: flex;

        flex-direction: column;

        justify-content: center;

        align-items: center;

        box-sizing: border-box;

    }



    .metric-value {

        font-size: clamp(1rem, 2.2vw, 2rem);

        font-weight: bold;

        white-space: nowrap;

        overflow: hidden;

        text-overflow: ellipsis;

        width: 100%;

        line-height: 1.2;

    }



    .metric-label {

        font-size: 0.8rem;

        opacity: 0.9;

        margin-top: 0.3rem;

        white-space: nowrap;

    }

</style>

""", unsafe_allow_html=True)


# Load Data



# Load Data

@st.cache_data

def load_data(data_signature):



    return {

        'social': pd.read_csv(r'data\social_posts.csv'),

        'search': pd.read_csv(r'data\search_trends.csv'),

        'reviews': pd.read_csv(r'data\reviews.csv'),

        'competitor': pd.read_csv(r'data\competitor_news.csv'),

        'tracker': pd.read_csv(r'data\brand_tracker_summary.csv'),

        'kpi': pd.read_csv(r'data\weekly_kpi_dashboard.csv')

    }



def format_kpi(value: float) -> str:
    """Abbreviate large numbers so they always fit inside a metric card."""
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 10_000:
        return f"{value / 1_000:.1f}K"
    return f"{int(value):,}"


# Load

data = load_data(get_data_signature())



availability_start, availability_end = get_data_availability(data['kpi']['week_start_date'])

availability_text = format_date_availability(availability_start, availability_end)



# Header

st.markdown(

    '<div class="main-header">🥔 Lay\'s Brand Health Monitor</div>',

    unsafe_allow_html=True

)



st.markdown("---")



# Sidebar

with st.sidebar:



    st.header("⚙️ Settings")



    markets = ['All'] + sorted(

        data['social']['market'].unique().tolist()

    )



    selected_market = st.selectbox(

        "📍 Select Market",

        markets

    )



    weeks = sorted(

        data['kpi']['week'].unique()

    )



    week_range = st.select_slider(

        "📅 Week Range",

        options=weeks,

        value=(weeks[0], weeks[-1])

    )



    # Initialize applied filters in session state (persistent until user reapplies)
    if 'applied_market' not in st.session_state:
        st.session_state['applied_market'] = 'All'
    if 'applied_week_range' not in st.session_state:
        st.session_state['applied_week_range'] = (weeks[0], weeks[-1])

    if st.button("Apply Filters"):
        st.session_state['applied_market'] = selected_market
        st.session_state['applied_week_range'] = week_range
        st.rerun()

    st.caption(
        f"**Active:** {st.session_state['applied_market']} · "
        f"Weeks {st.session_state['applied_week_range'][0]}–{st.session_state['applied_week_range'][1]}"
    )

    st.markdown("---")



    st.markdown("### 📊 Dataset Info")



    st.caption(

        f"Social Posts: {len(data['social']):,}"

    )



    st.caption(

        f"Search Trends: {len(data['search']):,}"

    )



    st.caption(

        f"Reviews: {len(data['reviews']):,}"

    )



    st.caption(

        f"Competitor News: {len(data['competitor']):,}"

    )



# Filter Data — only uses values committed via "Apply Filters" button

applied_market = st.session_state['applied_market']
applied_week_range = st.session_state['applied_week_range']


def _apply_market_filter(df, market):
    if market != 'All' and 'market' in df.columns:
        return df[df['market'] == market]
    return df


def _apply_week_filter(df, week_range):
    if 'week' in df.columns:
        return df[(df['week'] >= week_range[0]) & (df['week'] <= week_range[1])]
    return df


social_df = _apply_week_filter(
    _apply_market_filter(data['social'], applied_market),
    applied_week_range
)

search_df = _apply_week_filter(
    _apply_market_filter(data['search'], applied_market),
    applied_week_range
)

reviews_df = _apply_week_filter(
    _apply_market_filter(data['reviews'], applied_market),
    applied_week_range
)

competitor_df = _apply_week_filter(
    _apply_market_filter(data['competitor'], applied_market),
    applied_week_range
)

kpi_df = _apply_week_filter(
    _apply_market_filter(data['kpi'], applied_market),
    applied_week_range
)



# KPI Row

st.header("📈 Key Performance Indicators")



col1, col2, col3, col4, col5 = st.columns(5)



with col1:



    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{format_kpi(len(social_df))}</div><div class="metric-label">Total Posts</div></div>',
        unsafe_allow_html=True
    )



with col2:



    positive_pct = (

        social_df['sentiment']

        .value_counts(normalize=True)

        .get('positive', 0) * 100

    )

    if positive_pct < 40:
        sentiment_color = '#e74c3c'
        sentiment_text = 'white'
    elif positive_pct < 60:
        sentiment_color = '#f1c40f'
        sentiment_text = 'black'
    else:
        sentiment_color = '#2ecc71'
        sentiment_text = 'white'


    st.markdown(
        f'<div class="metric-card" style="background:{sentiment_color};color:{sentiment_text};"><div class="metric-value">{positive_pct:.1f}%</div><div class="metric-label">Positive Sentiment</div></div>',
        unsafe_allow_html=True
    )



with col3:



    avg_rating = reviews_df['star_rating'].mean()

    if avg_rating <= 3.0:
        rating_color = '#e74c3c'
        rating_text = 'white'
    elif avg_rating <= 3.5:
        rating_color = '#f1c40f'
        rating_text = 'black'
    else:
        rating_color = '#2ecc71'
        rating_text = 'white'


    st.markdown(
        f'<div class="metric-card" style="background:{rating_color};color:{rating_text};"><div class="metric-value">{avg_rating:.2f}/5</div><div class="metric-label">Avg Review Rating</div></div>',
        unsafe_allow_html=True
    )



with col4:



    total_search = (

        search_df['search_volume']

        .sum()

    )



    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{format_kpi(total_search)}</div><div class="metric-label">Search Volume</div></div>',
        unsafe_allow_html=True
    )



with col5:



    total_reviews = len(reviews_df)



    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{format_kpi(total_reviews)}</div><div class="metric-label">Reviews</div></div>',
        unsafe_allow_html=True
    )



st.markdown("---")



# Tabs

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([

    "🌐 Social Media",

    "🔍 Search Trends",

    "⭐ Reviews",

    "📰 Competitor News",

    "📊 Weekly Dashboard",

    "💬 Ask AI",

    "⚡ Agent Observability"

])



# TAB 1

with tab1:



    st.header("🌐 Social Media Analysis")



    col1, col2 = st.columns(2)



    with col1:



        sentiment_counts = (

            social_df['sentiment']

            .value_counts()

        )



        fig = go.Figure(data=[go.Pie(

            labels=sentiment_counts.index,

            values=sentiment_counts.values,

            hole=0.4

        )])

        fig.update_layout(
            title_text="Sentiment Distribution",
            title_x=0.5
        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )



    with col2:



        platform_counts = (

            social_df['platform']

            .value_counts()

        )



        fig = go.Figure(data=[go.Bar(

            x=platform_counts.index,

            y=platform_counts.values

        )])

        fig.update_layout(
            title_text="Posts by Platform",
            title_x=0.5,
            xaxis_title="Platform",
            yaxis_title="Number of Posts"
        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )



# TAB 2

with tab2:



    st.header("🔍 Search Trend Analysis")



    top_keywords = (

        search_df

        .groupby('keyword')['search_volume']

        .sum()

        .nlargest(10)

    )



    fig = go.Figure(data=[go.Bar(

        y=top_keywords.index,

        x=top_keywords.values,

        orientation='h'

    )])

    fig.update_layout(
        title_text="Top 10 Keywords by Search Volume",
        title_x=0.5,
        xaxis_title="Total Search Volume",
        yaxis_title="Keyword"
    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )



# TAB 3

with tab3:



    st.header("⭐ Customer Review Analysis")



    rating_dist = (

        reviews_df['star_rating']

        .value_counts()

        .sort_index(ascending=False)

    )



    fig = go.Figure(data=[go.Bar(

        x=[f"{r}★" for r in rating_dist.index],

        y=rating_dist.values

    )])

    fig.update_layout(
        title_text="Review Rating Distribution",
        title_x=0.5,
        xaxis_title="Star Rating",
        yaxis_title="Number of Reviews"
    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )



# TAB 4

with tab4:



    st.header("📰 Competitor Intelligence")



    comp_news = (

        competitor_df['competitor_brand']

        .value_counts()

    )



    fig = go.Figure(data=[go.Bar(

        x=comp_news.index,

        y=comp_news.values

    )])

    fig.update_layout(
        title_text="News Volume by Competitor",
        title_x=0.5,
        xaxis_title="Competitor Brand",
        yaxis_title="Number of News Items"
    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )



# TAB 5

with tab5:



    st.header("📊 Weekly KPI Dashboard")



    fig = go.Figure()



    fig.add_trace(go.Scatter(

        x=kpi_df['week'],

        y=kpi_df['positive_sentiment_pct'] * 100,

        name='Positive Sentiment'

    ))



    fig.add_trace(go.Scatter(

        x=kpi_df['week'],

        y=kpi_df['negative_sentiment_pct'] * 100,

        name='Negative Sentiment'

    ))

    fig.update_layout(
        title_text="Weekly Sentiment Trend",
        title_x=0.5,
        xaxis_title="Week",
        yaxis_title="Sentiment (%)"
    )



    st.plotly_chart(

        fig,

        use_container_width=True

    )



# TAB 6

with tab6:



    st.header("💬 Ask AI About Brand Health")



    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    def _timeframe_label(timeframe: dict) -> str:
        """Return a human-readable timeframe string for the report button label."""
        start = timeframe.get('effective_start')
        end   = timeframe.get('effective_end')
        if start is None or end is None:
            return ""
        days = (end - start).days + 1
        if days == 1:
            return f"for {start:%b %d, %Y}"
        elif days <= 7:
            return f"for the past {days} days"
        elif days <= 14:
            return "for the past 2 weeks"
        elif days <= 31:
            weeks = round(days / 7)
            return f"for the past {weeks} week{'s' if weeks > 1 else ''}"
        elif days <= 93:
            months = round(days / 30)
            return f"for the past {months} month{'s' if months > 1 else ''}"
        else:
            return f"for {start:%b %d} – {end:%b %d, %Y}"



    st.info(

        f"Data availability for Ask AI: {availability_text}. "

        "Any timeframe outside this range will return 'Data not available'."

    )



    user_question = st.text_input(

        "Ask a question",

        placeholder="What changed in brand health this week?"

    )



    if st.button("🚀 Ask"):



        if not user_question.strip():

            st.warning("Please enter a question first")

        else:

            executive_report = ""
            tf_label = ""

            timeframe = parse_timeframe(

                user_question,

                availability_start,

                availability_end

            )



            if timeframe['matched'] and not timeframe['has_data']:

                requested_start = timeframe['requested_start']

                requested_end = timeframe['requested_end']

                ai_answer = (

                    f"Data not available for the requested timeframe "

                    f"({requested_start:%Y-%m-%d} to {requested_end:%Y-%m-%d}). "

                    f"Available data is from {availability_text}."

                )
                executive_report = ""
                tf_label = ""

            else:

                with st.spinner("🔍 Checking question relevance..."):

                    is_relevant = is_question_relevant(user_question, client)



                if is_relevant:

                    with st.spinner("📊 Running 6-Agent Brand Health Analysis with Feedback Loop..."):

                        try:

                            crew_result = run_brand_health_crew(

                                brand="Lay's",

                                user_prompt=user_question,

                                llm=client,

                                start_date=timeframe.get('effective_start'),

                                end_date=timeframe.get('effective_end'),

                                max_feedback_iterations=3,

                                quality_threshold=7.5,

                            )



                            if sum(crew_result["data_summary"].values()) == 0:

                                if timeframe['matched']:

                                    requested_start = timeframe['requested_start']

                                    requested_end = timeframe['requested_end']

                                    ai_answer = (

                                        f"Data not available for the requested timeframe "

                                        f"({requested_start:%Y-%m-%d} to {requested_end:%Y-%m-%d}). "

                                        f"Available data is from {availability_text}."

                                    )
                                    executive_report = ""
                                    tf_label = ""

                                else:

                                    ai_answer = "No relevant data available for this query."
                                    executive_report = ""
                                    tf_label = ""

                            else:

                                ai_answer = crew_result.get("qa_review") or crew_result["final_report"]
                                executive_report = crew_result.get("executive_report", "")
                                tf_label = _timeframe_label(timeframe)
                                _fb = crew_result.get("feedback_loop", {})
                                # Store last feedback loop result in session state for Observability tab
                                st.session_state['last_feedback_loop'] = _fb



                        except Exception as e:

                            ai_answer = f"""

⚠️ Error Running Crew Analysis:



{str(e)}



Please ensure all data files are present in the 'data/' directory.

"""
                            executive_report = ""
                            tf_label = ""

                else:

                    ai_answer = "I don't know"
                    executive_report = ""
                    tf_label = ""

            # STEP 3: Store in chat history

            chat_id = len(st.session_state.chat_history)

            st.session_state.chat_history.append({

                "id": chat_id,

                "question": user_question,

                "answer": ai_answer,

                "executive_report": executive_report,

                "tf_label": tf_label

            })



    if st.session_state.chat_history:



        st.subheader("📜 Chat History")



        for chat in reversed(st.session_state.chat_history):

            st.markdown(f"### Q: {chat['question']}")

            # Show QA review as the main answer
            st.info(chat['answer'])

            # Collapsible brand health report button
            exec_report = chat.get('executive_report', '')
            if exec_report:
                label = chat.get('tf_label', '')
                button_text = f"📄 Brand Health Report {label}".strip()
                with st.expander(button_text):
                    st.markdown(exec_report)

            # HITL — only the QA review in the editable area
            st.subheader("🧑 Human Review")

            edited_report = st.text_area(
                "Edit QA Review",
                value=st.session_state.get(f"edit_{chat['id']}", chat['answer']),
                height=250,
                key=f"edit_{chat['id']}"
            )

            if st.button(
                "Submit Review",
                key=f"submit_{chat['id']}"
            ):
                st.success("Review Submitted")

            st.markdown("---")



# TAB 7

with tab7:

    st.header("⚡ Agent Observability — Feedback Loop")

    fb = st.session_state.get('last_feedback_loop', {})

    if not fb:
        st.info(
            "No analysis run yet. Ask a question in the **Ask AI** tab to see "
            "the feedback loop in action."
        )
    else:
        iterations    = fb.get('iteration_history', [])
        final_score   = fb.get('score_progression', [])[-1] if fb.get('score_progression') else 0
        converged     = fb.get('converged', False)
        threshold     = fb.get('quality_threshold', 7.5)
        num_iters     = fb.get('iterations', len(iterations))

        # ── Summary banner ────────────────────────────────────────────────────
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Final Quality Score", f"{final_score:.1f} / 10")
        with col_b:
            st.metric("Iterations Used", f"{num_iters}")
        with col_c:
            status_label = f"✅ Converged (≥ {threshold})" if converged else f"⚠️ Max iterations reached"
            st.metric("Loop Status", status_label)

        st.markdown("---")

        # ── Score progression chart ───────────────────────────────────────────
        if len(fb.get('score_progression', [])) > 1:
            fig_scores = go.Figure()
            fig_scores.add_trace(go.Scatter(
                x=list(range(1, num_iters + 1)),
                y=fb['score_progression'],
                mode='lines+markers',
                marker=dict(size=10),
                line=dict(color='#667eea', width=3),
                name='Quality Score',
            ))
            fig_scores.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Threshold ({threshold})",
            )
            fig_scores.update_layout(
                title="Quality Score Across Feedback Iterations",
                xaxis_title="Iteration",
                yaxis_title="Score / 10",
                yaxis=dict(range=[0, 10]),
                title_x=0.5,
            )
            st.plotly_chart(fig_scores, use_container_width=True)

        # ── Per-iteration detail ──────────────────────────────────────────────
        st.subheader("📋 Iteration-by-Iteration Detail")

        for rec in iterations:
            i         = rec['iteration']
            score     = rec['quality_score']
            improved  = rec.get('improved', False)
            badge     = "🆕 First pass" if i == 1 else ("⬆️ Improved" if improved else "➡️ Same")

            with st.expander(f"Iteration {i}  —  Score: {score}/10   {badge}", expanded=(i == num_iters)):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Synthesizer Output**")
                    st.markdown(rec['synthesizer_output'])
                with col2:
                    st.markdown("**Critic QA Output**")
                    st.markdown(rec['critic_output'])

        # ── Pipeline architecture note ────────────────────────────────────────
        st.markdown("---")
        st.markdown("""
**Pipeline Architecture**

```
Phase 1 (run once)
  Social Listening → Search Trend → Review Theme → Competitor Monitoring
                                ↓
Phase 2 — Closed Feedback Loop
  Synthesizer ──────────────────────────────────────────┐
      ↓                                                  │
  Critic QA  → parse quality score                       │
      │                                                  │
      ├── score ≥ 7.5  →  DONE                          │
      │                                                  │
      └── score < 7.5  →  inject issues as feedback ────┘
                           (up to 3 iterations)
```
""")




# Footer

st.markdown("---")



st.markdown(

    """

    <div style='text-align:center;color:#7f8c8d'>

    🥔 Lay's Brand Health Monitor |

    Built with Streamlit + CrewAI

    </div>

    """,

    unsafe_allow_html=True

)

