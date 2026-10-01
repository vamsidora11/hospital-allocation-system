from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "hospital-allocation-system"
    environment: str = "local"

    # Database — defaults to SQLite for local/dev/test so the app runs without
    # Docker/Postgres; docker-compose.yml points DATABASE_URL at Postgres for
    # anything closer to production.
    database_url: str = "sqlite:///./dev.db"

    # Vector store
    chroma_persist_dir: str = "./.chroma"
    chroma_host: str | None = None
    chroma_port: int = 8000

    # LLM / embeddings — provider priority in app/llm.py and app/rag/embeddings.py is
    # Google AI Studio > Azure OpenAI > local offline fallback. Unset in this dev sandbox
    # except for Google (see PROGRESS.md Phase 3 entry).
    google_api_key: str | None = None
    google_embedding_model: str = "models/gemini-embedding-001"
    google_chat_model: str = "gemini-3.5-flash-lite"
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    openai_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"

    # Workflow thresholds
    human_approval_threshold: float = 0.70
    max_agent_retries: int = 3
    # Minimum similarity score for a retrieved policy chunk to count as a grounding citation.
    # 0.55 is calibrated for real semantic embeddings (Google/Azure). When running with the
    # local hash-based fallback embedder (no API key configured), scores top out around 0.40,
    # so lower this to 0.20 in .env to prevent every agent from being marked ungrounded.
    rag_grounding_floor: float = 0.20

    # Force rule-based agents (fast, deterministic, no API calls) for testing/demo.
    # When True, ignores real LLM and uses decision rules instead. Default False = real LLM.
    use_rule_based_agents: bool = False

    # Auth — stubbed for the first pass, see CLAUDE.md TODO.
    api_key: str = "dev-local-api-key"

    # Observability
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
