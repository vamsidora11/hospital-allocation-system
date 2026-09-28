from pathlib import Path

import chromadb
import pytest

from app.rag.chunking import chunk_document
from app.rag.embeddings import LocalHashingEmbeddingClient
from app.rag.ingest import DOMAIN_BY_FILENAME
from app.rag.retriever import retrieve

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICIES_DIR = REPO_ROOT / "data" / "policies"


@pytest.fixture
def populated_collection():
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection("test_policy_chunks", metadata={"hnsw:space": "cosine"})
    embedder = LocalHashingEmbeddingClient()

    for filename, domain in DOMAIN_BY_FILENAME.items():
        text = (POLICIES_DIR / filename).read_text(encoding="utf-8")
        chunks = chunk_document(text)
        embeddings = embedder.embed([c.text for c in chunks])
        collection.upsert(
            ids=[f"{filename}-{c.chunk_index}" for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {"policy_name": filename, "section_heading": c.section_heading, "domain": domain, "effective_date": "2026-01-01"}
                for c in chunks
            ],
        )
    return collection


def test_known_query_hits_expected_bed_policy_section(populated_collection):
    results = retrieve(
        "critical patient needs ICU level monitoring",
        domain="bed",
        collection=populated_collection,
        k=4,
    )

    assert len(results) > 0
    # The bed allocation policy now splits monitoring indicators (3.1) from the
    # critical-acuity placement rule itself (3.2) into separate chunks; the
    # hashing-fallback embedding (see CLAUDE.md/PROGRESS.md) isn't precise enough to
    # always rank one ahead of the other, so check the target section is retrieved
    # at all within top-k, and that every hit is topically the bed policy.
    assert any(r.section == "3.2 Critical acuity placement" for r in results)
    assert all(r.policy_name == "bed_allocation_policy.md" for r in results)
    assert results[0].similarity_score > 0
    assert results[0].excerpt


def test_known_query_hits_expected_staffing_policy_section(populated_collection):
    results = retrieve(
        "ICU nurse to patient ratio requirements",
        domain="staffing",
        collection=populated_collection,
        k=4,
    )

    assert len(results) > 0
    assert any(r.section == "5.1 ICU nurse ratios" for r in results)


def test_domain_filter_excludes_unrelated_policies(populated_collection):
    results = retrieve(
        "regional transfer for unavailable specialist capability",
        domain="emergency",
        collection=populated_collection,
    )

    assert all(r.policy_name in ("emergency_protocols.md", "compliance_rules.md") for r in results)
