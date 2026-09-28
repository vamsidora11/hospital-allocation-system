"""Embedding client interface.

Default (per plan Section 6) is Azure OpenAI / OpenAI `text-embedding-3-small`, used when
`AZURE_OPENAI_API_KEY`/`OPENAI_API_KEY` is configured. With no key configured — the case in
this dev sandbox, which has no live credentials and no verified internet access to a model
hub — this falls back to a deterministic local hashing-trick embedding so RAG ingestion,
retrieval, and tests all run fully offline. Swap in a real `sentence-transformers` model
here instead if one is available; the interface (`EmbeddingClient.embed`) doesn't change.
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
    if settings.azure_openai_api_key and settings.azure_openai_endpoint:
        return AzureOpenAIEmbeddingClient()
    return LocalHashingEmbeddingClient()
