from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from crewai import Crew, Process
from agents.social_listening_agent import create_social_listening_agent
from tasks.social_listening_task import create_social_listening_task


def test_social_listening_agent_runs():
    """Smoke-test: social listening agent completes without error."""
    social_df = pd.read_csv('data/social_posts.csv')
    lays_posts = social_df[social_df['brand_mentioned'] == "Lay's"]

    agent = create_social_listening_agent()
    task = create_social_listening_task(
        agent=agent,
        social_data=lays_posts,
        brand="Lay's",
        user_prompt=(
            "How is Lay's performing on social media? "
            "Are there any negative sentiment spikes we should be concerned about?"
        ),
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    assert result is not None
    assert len(str(result)) > 0
