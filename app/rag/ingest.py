"""Ingest policy documents from `data/policies/` into Chroma + `policy_chunks`.

Run with: py -3 -m app.rag.ingest
"""

import re
from pathlib import Path

from app.db import SessionLocal
from app.models.orm import PolicyChunk, PolicyDocument
from app.rag.chunking import chunk_document
from app.rag.embeddings import get_embedding_client
from app.rag.vector_store import reset_collection

DOMAIN_BY_FILENAME = {
    "bed_allocation_policy.md": "bed",
    "staffing_policy.md": "staffing",
    "emergency_protocols.md": "emergency",
    "infection_control_standards.md": "infection_control",
    "compliance_rules.md": "compliance",
}

_EFFECTIVE_DATE_RE = re.compile(r"Effective date:\s*(\S+)")


def _read_effective_date(text: str, fallback: str) -> str:
    m = _EFFECTIVE_DATE_RE.search(text)
    return m.group(1) if m else fallback


def ingest_all(repo_root: Path | None = None) -> int:
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    session = SessionLocal()
    collection = reset_collection()
    embedder = get_embedding_client()
    total_chunks = 0

    try:
        # clear existing policy_chunks rows so re-ingestion doesn't duplicate
        session.query(PolicyChunk).delete()
        session.commit()

        documents = session.query(PolicyDocument).all()
        for doc in documents:
            domain = DOMAIN_BY_FILENAME.get(Path(doc.source_path).name, "compliance")
            path = repo_root / doc.source_path
            text = path.read_text(encoding="utf-8")
            effective_date = _read_effective_date(text, doc.effective_date.date().isoformat())

            chunks = chunk_document(text)
            if not chunks:
                continue

            embeddings = embedder.embed([c.text for c in chunks])
            ids = [f"{doc.id}-{c.chunk_index}" for c in chunks]
            metadatas = [
                {
                    "policy_name": doc.name,
                    "section_heading": c.section_heading,
                    "effective_date": effective_date,
                    "domain": domain,
                }
                for c in chunks
            ]
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=[c.text for c in chunks],
                metadatas=metadatas,
            )

            for chunk_id, c in zip(ids, chunks):
                session.add(
                    PolicyChunk(
                        policy_document_id=doc.id,
                        chunk_text=c.text,
                        chroma_id=chunk_id,
                        chunk_index=c.chunk_index,
                    )
                )
            total_chunks += len(chunks)

        session.commit()
    finally:
        session.close()

    return total_chunks


if __name__ == "__main__":
    n = ingest_all()
    print(f"Ingested {n} policy chunks.")
