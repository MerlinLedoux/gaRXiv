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
"""


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
