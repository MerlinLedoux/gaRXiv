import json
import logging
import sqlite3
from dataclasses import dataclass

from garxiv import db
from garxiv.config import Config
from garxiv.embeddings import factory as embedding_factory
from garxiv.vectorstore import factory as vectorstore_factory

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingSummary:
    embedded: int = 0
    errors: int = 0


def _batched(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def run_embedding(config: Config, limit: int | None = None) -> EmbeddingSummary:
    """Embed every chunk with status='pending_embedding' and upsert it into
    the vector store, one batch at a time. Idempotent: a chunk already
    'embedded' is never reprocessed; a failed batch is marked with an error
    and doesn't block the remaining batches. Once all of a document's chunks
    are embedded, the parent document is marked 'embedded' too."""
    conn = db.get_connection(config.storage.db_path)
    provider = embedding_factory.get_provider(config.embeddings)
    name = vectorstore_factory.collection_name(provider.name, provider.dimension)
    store = vectorstore_factory.get_store(config.vector_store, collection_name=name)
    store.ensure_collection(provider.dimension)

    summary = EmbeddingSummary()
    pending = db.get_chunks_by_status(conn, "pending_embedding")
    if limit is not None:
        pending = pending[:limit]

    doc_cache: dict[str, sqlite3.Row | None] = {}

    for batch in _batched(pending, config.embeddings.batch_size):
        try:
            vectors = provider.embed([row["text"] for row in batch], is_query=False)
            metadatas = []
            for row in batch:
                arxiv_id = row["arxiv_id"]
                if arxiv_id not in doc_cache:
                    doc_cache[arxiv_id] = db.get_document(conn, arxiv_id)
                doc = doc_cache[arxiv_id]
                metadatas.append({
                    "arxiv_id": arxiv_id,
                    "section_title": row["section_title"],
                    "chunk_index": row["chunk_index"],
                    "page_start": row["page_start"],
                    "page_end": row["page_end"],
                    "categories": json.loads(doc["categories"]) if doc else [],
                    "authors": json.loads(doc["authors"]) if doc else [],
                    "published_date": doc["published_date"] if doc else None,
                })

            store.upsert(
                ids=[row["chunk_id"] for row in batch], vectors=vectors, metadatas=metadatas
            )
            db.mark_chunks_embedded(
                conn, [row["chunk_id"] for row in batch], provider.name, provider.dimension
            )
            summary.embedded += len(batch)
        except Exception as exc:
            logger.warning("embedding batch failed: %s", exc)
            for row in batch:
                db.mark_chunk_error(conn, row["chunk_id"], str(exc))
            summary.errors += len(batch)

    for arxiv_id in doc_cache:
        if db.count_chunks_not_embedded(conn, arxiv_id) == 0:
            db.mark_document_status(conn, arxiv_id, "embedded")

    conn.close()
    return summary
