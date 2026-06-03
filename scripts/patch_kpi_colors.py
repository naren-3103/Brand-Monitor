"""
One-off patch: added colour coding to the Positive Sentiment and
Avg Review Rating KPI cards in app.py.

This script has already been applied. It is kept here for reference only.
"""
from pathlib import Path

path = Path(__file__).resolve().parent.parent / 'app.py'
text = path.read_text(encoding='utf-8')
old1 = '''    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{positive_pct:.1f}%</div><div class="metric-label">Positive Sentiment</div></div>',
        unsafe_allow_html=True
    )


with col3:
'''
new1 = '''    if positive_pct < 40:
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
'''
old2 = '''with col3:


    avg_rating = reviews_df['star_rating'].mean()


    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{avg_rating:.2f}/5</div><div class="metric-label">Avg Review Rating</div></div>',
        unsafe_allow_html=True
    )


with col4:
'''
new2 = '''with col3:


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
'''
if old1 not in text:
    raise SystemExit('Patch already applied or app.py has changed.')
text = text.replace(old1, new1, 1)
if old2 not in text:
    raise SystemExit('Patch already applied or app.py has changed.')
text = text.replace(old2, new2, 1)
path.write_text(text, encoding='utf-8')
print('Patch applied.')
