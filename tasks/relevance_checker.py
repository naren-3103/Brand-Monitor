"""
Relevance Checker for Brand Health Questions
Determines if a user question is relevant to brand monitoring before running the full crew.
"""

from crewai import Agent, Task, Crew


def is_question_relevant(question: str, llm) -> bool:
    """
    Checks if a question is relevant to brand health monitoring.
    
    Returns:
        True if the question is relevant to brand monitoring
        False if the question is out of context
    """
    
    if not question or len(question.strip()) == 0:
        return False
    
    relevance_agent = Agent(
        role="Question Relevance Checker",
        goal="Strictly determine if a user question is relevant to brand health/monitoring",
        backstory="You are a strict gatekeeper. You only approve questions directly related to brand health, customer sentiment, market analysis, and business metrics. Everything else is rejected.",
        llm=llm,
        verbose=False
    )
    
    relevance_task = Task(
        description=f"""
TASK: Determine if this question is relevant to brand health analysis.

QUESTION: "{question}"

RELEVANT = questions about:
- Lay's brand health, reputation, perception
- Customer reviews, satisfaction, feedback
- Market trends, search volume, keywords
- Competitor analysis and activity
- Social media sentiment and engagement
- Product quality issues
- Sales performance, market position
- Brand metrics and KPIs

NOT_RELEVANT = questions about:
- General knowledge (capitals, science, math, history)
- Weather, sports, entertainment, news
- Technical IT or coding questions
- Personal advice or philosophy
- Other brands or companies
- How-to guides unrelated to brand analysis
- Anything not about business/brand metrics

---
RESPOND WITH EXACTLY ONE LINE:
RELEVANT
or
NOT_RELEVANT
---

Think step by step then respond.
        """,
        expected_output="RELEVANT or NOT_RELEVANT (one word)",
        agent=relevance_agent
    )
    
    # Create crew and execute
    crew = Crew(
        agents=[relevance_agent],
        tasks=[relevance_task],
        verbose=False
    )
    
    try:
        result = crew.kickoff()
        response_text = str(result).upper().strip()
        
        print(f"[DEBUG] Relevance check response: {response_text}")
        
        # Check for NOT_RELEVANT first (since it contains "RELEVANT")
        if "NOT_RELEVANT" in response_text:
            print(f"[DEBUG] Question deemed NOT_RELEVANT")
            return False
        
        # Then check for RELEVANT
        if "RELEVANT" in response_text:
            print(f"[DEBUG] Question deemed RELEVANT")
            return True
        
        # If neither keyword found, default to NOT relevant
        print(f"[DEBUG] Could not parse response, defaulting to NOT_RELEVANT")
        return False
        
    except Exception as e:
        print(f"[DEBUG] Relevance check error: {e}")
        return False  # Default to NOT relevant on error (conservative approach)
