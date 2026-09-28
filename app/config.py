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
    google_embedding_model: str = "models/text-embedding-004"
    google_chat_model: str = "gemini-1.5-flash"
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    openai_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"

    # Workflow thresholds
    human_approval_threshold: float = 0.70
    max_agent_retries: int = 3

    # Auth — stubbed for the first pass, see CLAUDE.md TODO.
    api_key: str = "dev-local-api-key"

    # Observability
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
