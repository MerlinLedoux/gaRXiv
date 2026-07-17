import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    arxiv_id        TEXT PRIMARY KEY,
    version         INTEGER NOT NULL,
    title           TEXT NOT NULL,
    abstract        TEXT NOT NULL,
    authors         TEXT NOT NULL,
    categories      TEXT NOT NULL,
    published_date  TEXT NOT NULL,
    updated_date    TEXT NOT NULL,
    pdf_url         TEXT NOT NULL,
    pdf_path        TEXT,
    parsed_path     TEXT,
    status          TEXT NOT NULL DEFAULT 'discovered',
    error_message   TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    query_key       TEXT PRIMARY KEY,
    watermark_date  TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id            TEXT PRIMARY KEY,
    arxiv_id             TEXT NOT NULL REFERENCES documents(arxiv_id),
    chunk_index          INTEGER NOT NULL,
    section_title        TEXT NOT NULL,
    text                  TEXT NOT NULL,
    token_count           INTEGER NOT NULL,
    page_start            INTEGER,
    page_end              INTEGER,
    status                TEXT NOT NULL DEFAULT 'pending_embedding',
    embedding_provider    TEXT,
    embedding_dim         INTEGER,
    error_message         TEXT,
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_arxiv_id ON chunks(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_chunks_status ON chunks(status);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED, text, tokenize = 'porter unicode61'
);
"""


@dataclass
class ChunkRecord:
    arxiv_id: str
    chunk_index: int
    section_title: str
    text: str
    token_count: int
    page_start: int | None
    page_end: int | None


@dataclass
class ArxivEntry:
    arxiv_id: str
    version: int
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published_date: str
    updated_date: str
    pdf_url: str


def get_connection(db_path: Path | str) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_discovered(conn: sqlite3.Connection, entry: ArxivEntry) -> bool:
    """Insert a newly discovered document, or reset it to 'discovered' if a
    newer version showed up. Already-known documents at their current version
    are left untouched (no re-download/re-parse/re-embed).

    Returns True if a row was inserted or updated, False if the document was
    already known at its current version (nothing to do).
    """
    existing = conn.execute(
        "SELECT version FROM documents WHERE arxiv_id = ?", (entry.arxiv_id,)
    ).fetchone()

    now = _now()
    if existing is None:
        conn.execute(
            """
            INSERT INTO documents (
                arxiv_id, version, title, abstract, authors, categories,
                published_date, updated_date, pdf_url, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'discovered', ?, ?)
            """,
            (
                entry.arxiv_id,
                entry.version,
                entry.title,
                entry.abstract,
                json.dumps(entry.authors),
                json.dumps(entry.categories),
                entry.published_date,
                entry.updated_date,
                entry.pdf_url,
                now,
                now,
            ),
        )
    elif entry.version > existing["version"]:
        delete_chunks_for_document(conn, entry.arxiv_id)
        conn.execute(
            """
            UPDATE documents SET
                version = ?, title = ?, abstract = ?, authors = ?, categories = ?,
                published_date = ?, updated_date = ?, pdf_url = ?,
                pdf_path = NULL, parsed_path = NULL, status = 'discovered',
                error_message = NULL, updated_at = ?
            WHERE arxiv_id = ?
            """,
            (
                entry.version,
                entry.title,
                entry.abstract,
                json.dumps(entry.authors),
                json.dumps(entry.categories),
                entry.published_date,
                entry.updated_date,
                entry.pdf_url,
                now,
                entry.arxiv_id,
            ),
        )
    else:
        return False

    conn.commit()
    return True


def get_by_status(conn: sqlite3.Connection, status: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM documents WHERE status = ?", (status,)
    ).fetchall()


def mark_downloaded(conn: sqlite3.Connection, arxiv_id: str, pdf_path: str) -> None:
    conn.execute(
        "UPDATE documents SET status = 'downloaded', pdf_path = ?, "
        "error_message = NULL, updated_at = ? WHERE arxiv_id = ?",
        (pdf_path, _now(), arxiv_id),
    )
    conn.commit()


def mark_parsed(conn: sqlite3.Connection, arxiv_id: str, parsed_path: str) -> None:
    conn.execute(
        "UPDATE documents SET status = 'parsed', parsed_path = ?, "
        "error_message = NULL, updated_at = ? WHERE arxiv_id = ?",
        (parsed_path, _now(), arxiv_id),
    )
    conn.commit()


def mark_error(conn: sqlite3.Connection, arxiv_id: str, error_message: str) -> None:
    conn.execute(
        "UPDATE documents SET error_message = ?, updated_at = ? WHERE arxiv_id = ?",
        (error_message, _now(), arxiv_id),
    )
    conn.commit()


def get_watermark(conn: sqlite3.Connection, query_key: str) -> str | None:
    row = conn.execute(
        "SELECT watermark_date FROM ingestion_runs WHERE query_key = ?", (query_key,)
    ).fetchone()
    return row["watermark_date"] if row else None


def set_watermark(conn: sqlite3.Connection, query_key: str, watermark_date: str) -> None:
    conn.execute(
        """
        INSERT INTO ingestion_runs (query_key, watermark_date, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(query_key) DO UPDATE SET
            watermark_date = excluded.watermark_date,
            updated_at = excluded.updated_at
        """,
        (query_key, watermark_date, _now()),
    )
    conn.commit()


def get_document(conn: sqlite3.Connection, arxiv_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM documents WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchone()


def mark_document_status(conn: sqlite3.Connection, arxiv_id: str, status: str) -> None:
    conn.execute(
        "UPDATE documents SET status = ?, updated_at = ? WHERE arxiv_id = ?",
        (status, _now(), arxiv_id),
    )
    conn.commit()


def insert_chunks(conn: sqlite3.Connection, records: list[ChunkRecord]) -> list[str]:
    now = _now()
    chunk_ids = []
    for record in records:
        chunk_id = f"{record.arxiv_id}::{record.chunk_index:04d}"
        chunk_ids.append(chunk_id)
        conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, arxiv_id, chunk_index, section_title, text,
                token_count, page_start, page_end, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending_embedding', ?, ?)
            """,
            (
                chunk_id,
                record.arxiv_id,
                record.chunk_index,
                record.section_title,
                record.text,
                record.token_count,
                record.page_start,
                record.page_end,
                now,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)",
            (chunk_id, record.text),
        )
    conn.commit()
    return chunk_ids


def delete_chunks_for_document(conn: sqlite3.Connection, arxiv_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT chunk_id FROM chunks WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchall()
    chunk_ids = [row["chunk_id"] for row in rows]
    conn.execute("DELETE FROM chunks WHERE arxiv_id = ?", (arxiv_id,))
    if chunk_ids:
        placeholders = ",".join("?" for _ in chunk_ids)
        conn.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})", chunk_ids)
    conn.commit()
    return chunk_ids


def get_chunks_by_status(
    conn: sqlite3.Connection, status: str, limit: int | None = None
) -> list[sqlite3.Row]:
    query = "SELECT * FROM chunks WHERE status = ? ORDER BY arxiv_id, chunk_index"
    params: tuple = (status,)
    if limit is not None:
        query += " LIMIT ?"
        params = (status, limit)
    return conn.execute(query, params).fetchall()


def get_chunks_by_document(conn: sqlite3.Connection, arxiv_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM chunks WHERE arxiv_id = ? ORDER BY chunk_index", (arxiv_id,)
    ).fetchall()


def mark_chunks_embedded(
    conn: sqlite3.Connection, chunk_ids: list[str], provider: str, dimension: int
) -> None:
    now = _now()
    conn.executemany(
        """
        UPDATE chunks SET status = 'embedded', embedding_provider = ?,
            embedding_dim = ?, error_message = NULL, updated_at = ?
        WHERE chunk_id = ?
        """,
        [(provider, dimension, now, chunk_id) for chunk_id in chunk_ids],
    )
    conn.commit()


def mark_chunk_error(conn: sqlite3.Connection, chunk_id: str, error_message: str) -> None:
    conn.execute(
        "UPDATE chunks SET error_message = ?, updated_at = ? WHERE chunk_id = ?",
        (error_message, _now(), chunk_id),
    )
    conn.commit()


def count_chunks_not_embedded(conn: sqlite3.Connection, arxiv_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM chunks WHERE arxiv_id = ? AND status != 'embedded'",
        (arxiv_id,),
    ).fetchone()
    return row["n"]


def search_fts(
    conn: sqlite3.Connection,
    fts_query: str,
    limit: int,
    categories: list[str] | None = None,
    authors: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[sqlite3.Row]:
    """Lexical search via SQLite FTS5. `fts_query` must already be a safe
    MATCH expression (see retrieval.fts.sanitize_query) — no further
    escaping is done here. `score` is -bm25(...), so higher is better,
    consistent with VectorMatch.score (cosine similarity, higher is better).
    """
    sql = """
        SELECT c.chunk_id AS chunk_id, c.arxiv_id AS arxiv_id, -bm25(chunks_fts) AS score
        FROM chunks_fts
        JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
        JOIN documents d ON d.arxiv_id = c.arxiv_id
        WHERE chunks_fts MATCH ?
    """
    params: list = [fts_query]

    if categories:
        placeholders = ",".join("?" for _ in categories)
        sql += f" AND EXISTS (SELECT 1 FROM json_each(d.categories) WHERE value IN ({placeholders}))"
        params.extend(categories)
    if authors:
        placeholders = ",".join("?" for _ in authors)
        sql += f" AND EXISTS (SELECT 1 FROM json_each(d.authors) WHERE value IN ({placeholders}))"
        params.extend(authors)
    if date_from:
        sql += " AND d.published_date >= ?"
        params.append(date_from)
    if date_to:
        sql += " AND d.published_date <= ?"
        params.append(date_to)

    sql += " ORDER BY bm25(chunks_fts) ASC LIMIT ?"
    params.append(limit)
    return conn.execute(sql, params).fetchall()


def get_chunks_by_ids(conn: sqlite3.Connection, chunk_ids: list[str]) -> list[sqlite3.Row]:
    if not chunk_ids:
        return []
    placeholders = ",".join("?" for _ in chunk_ids)
    return conn.execute(
        f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids
    ).fetchall()


def backfill_fts(conn: sqlite3.Connection) -> int:
    """Reindex into chunks_fts any chunk present in `chunks` but missing from
    the FTS5 index (e.g. chunks inserted before FTS5 support was added).
    Safe to call repeatedly."""
    rows = conn.execute(
        """
        SELECT c.chunk_id, c.text FROM chunks c
        LEFT JOIN chunks_fts f ON f.chunk_id = c.chunk_id
        WHERE f.chunk_id IS NULL
        """
    ).fetchall()
    conn.executemany(
        "INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)",
        [(r["chunk_id"], r["text"]) for r in rows],
    )
    conn.commit()
    return len(rows)
