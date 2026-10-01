"""Shared grounding rule (plan Section 1 / Section 6): every agent output is required to
carry >=1 citation above the similarity floor, or be marked ungrounded with lowered confidence
rather than reasoning from unsupported general knowledge."""

from app.config import get_settings
from app.models.domain import PolicyCitation

UNGROUNDED_CONFIDENCE_MULTIPLIER = 0.6
NO_CITATIONS_CONFIDENCE_MULTIPLIER = 0.5


def apply_grounding(citations: list[PolicyCitation], base_confidence: float) -> tuple[float, bool]:
    if not citations:
        return round(base_confidence * NO_CITATIONS_CONFIDENCE_MULTIPLIER, 2), False

    floor = get_settings().rag_grounding_floor
    top_similarity = max(c.similarity_score for c in citations)
    if top_similarity < floor:
        return round(base_confidence * UNGROUNDED_CONFIDENCE_MULTIPLIER, 2), False

    return round(base_confidence, 2), True
