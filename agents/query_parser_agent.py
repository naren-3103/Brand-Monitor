from crewai import Agent


def create_query_parser_agent(llm=None):
    """
    Agent that parses brand health questions before the main pipeline runs.
    Determines relevance, detects comparison intent, and extracts ISO date ranges
    from natural language — all in one shot.
    """
    kwargs = {"llm": llm} if llm else {}
    return Agent(
        role="Brand Health Query Analyst",
        goal=(
            "Precisely determine whether a brand health question is relevant, "
            "detect period-over-period comparisons, and extract exact ISO date ranges "
            "from natural language."
        ),
        backstory=(
            "You are a meticulous analyst who sits at the front of a brand health "
            "pipeline. Before any specialist agent processes data, you read the user's "
            "question and decide: Is it on-topic? Does it compare two time periods? "
            "What exact calendar dates does it refer to? You output structured JSON "
            "so the downstream pipeline knows exactly what to analyse and over which "
            "dates. You never guess on dates — if a period is ambiguous you set it "
            "to null rather than hallucinate a range."
        ),
        verbose=False,
        allow_delegation=False,
        **kwargs,
    )
