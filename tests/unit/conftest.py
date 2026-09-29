import pytest

from app.rag.embeddings import LocalHashingEmbeddingClient


@pytest.fixture(autouse=True)
def _no_live_llm_calls(monkeypatch):
    """Unit tests stay deterministic/offline regardless of a real key in .env — per the
    plan's Phase 4 definition of done ("mocked LLM"), force the rule-based fallback path.
    Integration/e2e tests are the place to exercise the real Gemini call."""
    monkeypatch.setattr("app.graph.nodes.router.get_chat_llm", lambda: None)
    monkeypatch.setattr("app.rag.retriever.get_embedding_client", LocalHashingEmbeddingClient)
