"""Embedding client interface.

Provider priority (see app/config.py): Google AI Studio > Azure OpenAI / OpenAI
`text-embedding-3-small` (plan Section 6's original default) > a deterministic local
hashing-trick fallback used only when no provider is configured, so RAG ingestion,
retrieval, and tests still run fully offline in that case.
"""

import hashlib
import math
import re
from abc import ABC, abstractmethod

from app.config import get_settings

HASH_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EmbeddingClient(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LocalHashingEmbeddingClient(EmbeddingClient):
    """Deterministic, offline, no external calls or downloads.

    Feature-hashes lowercased tokens into a fixed-size vector (bag-of-words hashing
    trick), then L2-normalizes so cosine similarity is meaningful. Not semantically as
    strong as a trained embedding model, but stable and dependency-light for dev/tests.
    """

    def __init__(self, dim: int = HASH_DIM):
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class GoogleAIEmbeddingClient(EmbeddingClient):
    """Calls the Gemini REST API directly with httpx rather than through
    `langchain_google_genai`/`google-generativeai` — that SDK's client hung indefinitely in
    this sandbox (likely gRPC transport or ambient-credential probing) even though plain
    HTTPS calls to the same endpoint succeed in under a second. See PROGRESS.md Phase 3."""

    def __init__(self):
        settings = get_settings()
        self._api_key = settings.google_api_key
        self._model = settings.google_embedding_model  # e.g. "models/gemini-embedding-001"

    def embed(self, texts: list[str]) -> list[list[float]]:
        import time

        import httpx

        url = f"https://generativelanguage.googleapis.com/v1beta/{self._model}:embedContent"
        results = []
        with httpx.Client(timeout=30.0) as client:
            for text in texts:
                # Retries a transient connection drop seen live against this API
                # (httpx.ReadError / WinError 10054 mid-ingestion) — not speculative.
                last_error: Exception | None = None
                for attempt in range(3):
                    try:
                        response = client.post(
                            url,
                            params={"key": self._api_key},
                            json={"content": {"parts": [{"text": text}]}},
                        )
                        response.raise_for_status()
                        results.append(response.json()["embedding"]["values"])
                        last_error = None
                        break
                    except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                        last_error = exc
                        if attempt < 2:
                            time.sleep(1.5 * (attempt + 1))
                if last_error is not None:
                    raise last_error
        return results


class AzureOpenAIEmbeddingClient(EmbeddingClient):
    def __init__(self):
        from langchain_openai import AzureOpenAIEmbeddings  # optional dep, imported lazily

        settings = get_settings()
        self._client = AzureOpenAIEmbeddings(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            model=settings.embedding_model,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)


def get_embedding_client() -> EmbeddingClient:
    settings = get_settings()
    if settings.google_api_key:
        return GoogleAIEmbeddingClient()
    if settings.azure_openai_api_key and settings.azure_openai_endpoint:
        return AzureOpenAIEmbeddingClient()
    return LocalHashingEmbeddingClient()
