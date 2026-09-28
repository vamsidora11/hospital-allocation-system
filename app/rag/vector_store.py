"""Chroma persistent vector store — one collection, metadata-filtered per domain."""

import chromadb

from app.config import get_settings

COLLECTION_NAME = "policy_chunks"


def get_client() -> chromadb.ClientAPI:
    settings = get_settings()
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_collection(client: chromadb.ClientAPI | None = None):
    client = client or get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(client: chromadb.ClientAPI | None = None):
    client = client or get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return get_collection(client)
