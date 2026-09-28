"""Shared chat-LLM access point for the router and agents.

Provider priority (see app/config.py): Google AI Studio > Azure OpenAI > `None`. Callers
(router, agents) must have a deterministic rule-based fallback for the `None` case; there is
no local chat-model substitute here (unlike embeddings, a small offline model can't
meaningfully replace structured LLM reasoning, so the fallback lives in each caller's domain
logic instead).
"""

from typing import Any

from app.config import get_settings


def get_chat_llm() -> Any | None:
    settings = get_settings()

    if settings.google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI  # optional dep, lazy

        return ChatGoogleGenerativeAI(
            model=settings.google_chat_model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )

    if settings.azure_openai_api_key and settings.azure_openai_endpoint:
        from langchain_openai import AzureChatOpenAI  # optional dep, imported lazily

        return AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            azure_deployment="gpt-4o",
            temperature=0,
        )

    return None
