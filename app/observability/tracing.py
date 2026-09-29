"""LangSmith tracing (plan Section 12): "enabled via env vars in non-local environments."

LangChain/LangGraph pick up tracing purely from `LANGCHAIN_TRACING_V2`/`LANGCHAIN_API_KEY`
env vars — there's no SDK call to make here, just setting them from our own settings at
startup so operators configure one thing (`.env`) instead of two.

TODO: unverified against a real LangSmith account — no API key available in this sandbox
(see CLAUDE.md). The env vars get set correctly; whether traces actually show up in LangSmith
is untested.
"""

import os

from app.config import get_settings


def configure_tracing() -> None:
    settings = get_settings()
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.app_name
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
