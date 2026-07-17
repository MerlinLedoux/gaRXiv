import json
import logging
from dataclasses import dataclass
from pathlib import Path

from garxiv import db
from garxiv.chunking import chunker, cleaner
from garxiv.config import Config

logger = logging.getLogger(__name__)


@dataclass
class ChunkingSummary:
    documents_chunked: int = 0
    chunks_created: int = 0
    errors: int = 0


def run_chunking(config: Config, limit: int | None = None) -> ChunkingSummary:
    """Clean and chunk every document with status='parsed'. Idempotent:
    only processes documents at that status, and moves them to 'chunked'
    on success so a re-run only picks up newly parsed documents."""
    conn = db.get_connection(config.storage.db_path)
    summary = ChunkingSummary()

    rows = db.get_by_status(conn, "parsed")
    if limit is not None:
        rows = rows[:limit]

    for row in rows:
        try:
            parsed = json.loads(Path(row["parsed_path"]).read_text(encoding="utf-8"))
            cleaned = cleaner.clean_document(parsed)
            chunks = chunker.chunk_document(cleaned, config.chunking)

            db.delete_chunks_for_document(conn, row["arxiv_id"])
            records = [
                db.ChunkRecord(
                    arxiv_id=row["arxiv_id"],
                    chunk_index=c.chunk_index,
                    section_title=c.section_title,
                    text=c.text,
                    token_count=c.token_count,
                    page_start=c.page_start,
                    page_end=c.page_end,
                )
                for c in chunks
            ]
            db.insert_chunks(conn, records)
            db.mark_document_status(conn, row["arxiv_id"], "chunked")
            summary.documents_chunked += 1
            summary.chunks_created += len(records)
        except Exception as exc:
            logger.warning("chunking failed for %s: %s", row["arxiv_id"], exc)
            db.mark_error(conn, row["arxiv_id"], str(exc))
            summary.errors += 1

    conn.close()
    return summary
