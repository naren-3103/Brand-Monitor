import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()


def build_azure_openai_client():
    """Build a CrewAI Azure OpenAI LLM using environment variables from .env."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or os.getenv("AZURE_OPENAI_MODEL_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    missing = [
        name
        for name, value in [
            ("AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_OPENAI_ENDPOINT", endpoint),
            ("AZURE_OPENAI_API_KEY", api_key),
            ("AZURE_OPENAI_DEPLOYMENT_NAME or AZURE_OPENAI_MODEL_NAME", deployment),
            ("AZURE_OPENAI_API_VERSION", api_version),
        ]
        if not value
    ]

    if missing:
        raise ValueError(
            "Missing Azure OpenAI environment variables: " + ", ".join(missing)
        )

    # Azure OpenAI in litellm expects the model to be prefixed with "azure/"
    # so the provider is resolved correctly.
    if not deployment.startswith("azure/"):
        model = f"azure/{deployment}"
    else:
        model = deployment

    # Normalize endpoint values: allow either the resource root or the OpenAI v1 URL.
    if endpoint.endswith("/openai/v1/") or endpoint.endswith("/openai/v1"):
        endpoint = endpoint[: endpoint.rfind("/openai/v1")] + "/"

    if not endpoint.endswith("/"):
        endpoint = endpoint + "/"

    return LLM(
        model=model,
        api_base=endpoint,
        api_version=api_version,
        api_key=api_key,
    )
