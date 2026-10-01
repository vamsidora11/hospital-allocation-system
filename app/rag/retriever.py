"""Top-k policy retrieval, domain-filtered, returning cited `PolicyCitation`s."""

import re
import uuid

from app.models.domain import PolicyCitation
from app.rag.embeddings import get_embedding_client
from app.rag.vector_store import get_collection

DEFAULT_K = 4

def _grounding_floor() -> float:
    from app.config import get_settings
    return get_settings().rag_grounding_floor

# Kept as module-level alias so existing imports in tests still resolve.
GROUNDING_FLOOR = 0.55

# An agent working a given domain should also see cross-cutting infection-control /
# compliance chunks (mirrors the plan's Section 6 domain list plus the two cross-cutting
# policies split out in Phase 2 — see CLAUDE.md / PROGRESS.md Phase 2 entry for rationale).
DOMAIN_EXPANSION: dict[str, list[str]] = {
    "bed": ["bed", "infection_control"],
    "staffing": ["staffing", "infection_control"],
    "emergency": ["emergency", "compliance"],
    "compliance": ["compliance"],
    "infection_control": ["infection_control"],
}

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _excerpt(text: str, max_sentences: int = 2) -> str:
    sentences = _SENTENCE_RE.split(text.strip())
    return " ".join(sentences[:max_sentences]).strip()


def retrieve(query: str, domain: str, k: int = DEFAULT_K, collection=None) -> list[PolicyCitation]:
    """`collection` is injectable so tests can query an ephemeral, pre-populated
    collection instead of the persistent global one (see tests/unit/test_retriever.py)."""
    domains = DOMAIN_EXPANSION.get(domain, [domain])
    collection = collection or get_collection()
    embedder = get_embedding_client()
    query_embedding = embedder.embed([query])[0]

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"domain": {"$in": domains}} if len(domains) > 1 else {"domain": domains[0]},
    )

    citations: list[PolicyCitation] = []
    documents = result.get("documents") or [[]]
    metadatas = result.get("metadatas") or [[]]
    distances = result.get("distances") or [[]]

    for doc_text, meta, distance in zip(documents[0], metadatas[0], distances[0]):
        similarity = max(0.0, 1.0 - distance)
        citations.append(
            PolicyCitation(
                citation_id=f"CIT-{uuid.uuid4().hex[:8].upper()}",
                policy_name=meta["policy_name"],
                section=meta["section_heading"],
                excerpt=_excerpt(doc_text),
                similarity_score=round(similarity, 4),
            )
        )
    return citations
