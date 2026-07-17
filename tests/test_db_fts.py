from pathlib import Path

from garxiv import db
from garxiv.config import Config, StorageConfig


def _make_config(tmp_path: Path) -> Config:
    return Config(
        categories=["cs.CL"],
        authors=[],
        storage=StorageConfig(
            db_path=tmp_path / "garxiv.db",
            pdf_dir=tmp_path / "pdfs",
            parsed_dir=tmp_path / "parsed",
        ),
        max_results_per_run=10,
    )


def _seed_document(conn, arxiv_id: str, authors: list[str], categories: list[str], published_date: str) -> None:
    entry = db.ArxivEntry(
        arxiv_id=arxiv_id,
        version=1,
        title="T",
        abstract="A",
        authors=authors,
        categories=categories,
        published_date=published_date,
        updated_date=published_date,
        pdf_url=f"http://arxiv.org/pdf/{arxiv_id}v1",
    )
    db.upsert_discovered(conn, entry)


def test_insert_chunks_populates_fts(tmp_path):
    config = _make_config(tmp_path)
    conn = db.get_connection(config.storage.db_path)
    _seed_document(conn, "2401.00001", ["A. Author"], ["cs.CL"], "2024-01-01T00:00:00Z")

    db.insert_chunks(
        conn,
        [
            db.ChunkRecord(
                arxiv_id="2401.00001", chunk_index=0, section_title="Body",
                text="the transformer attention mechanism", token_count=4,
                page_start=0, page_end=0,
            )
        ],
    )

    rows = db.search_fts(conn, '"transformer"', limit=10)
    assert len(rows) == 1
    assert rows[0]["chunk_id"] == "2401.00001::0000"
    conn.close()


def test_delete_chunks_for_document_removes_fts_rows(tmp_path):
    config = _make_config(tmp_path)
    conn = db.get_connection(config.storage.db_path)
    _seed_document(conn, "2401.00001", [], ["cs.CL"], "2024-01-01T00:00:00Z")
    db.insert_chunks(
        conn,
        [
            db.ChunkRecord(
                arxiv_id="2401.00001", chunk_index=0, section_title="Body",
                text="quantum field theory", token_count=3, page_start=0, page_end=0,
            )
        ],
    )

    db.delete_chunks_for_document(conn, "2401.00001")

    rows = db.search_fts(conn, '"quantum"', limit=10)
    assert rows == []
    conn.close()


def test_search_fts_orders_by_relevance(tmp_path):
    config = _make_config(tmp_path)
    conn = db.get_connection(config.storage.db_path)
    _seed_document(conn, "2401.00001", [], ["cs.CL"], "2024-01-01T00:00:00Z")
    db.insert_chunks(
        conn,
        [
            db.ChunkRecord(
                arxiv_id="2401.00001", chunk_index=0, section_title="Body",
                text="attention attention attention mechanism", token_count=4,
                page_start=0, page_end=0,
            ),
            db.ChunkRecord(
                arxiv_id="2401.00001", chunk_index=1, section_title="Body",
                text="attention is mentioned once here", token_count=6,
                page_start=0, page_end=0,
            ),
        ],
    )

    rows = db.search_fts(conn, '"attention"', limit=10)

    assert [r["chunk_id"] for r in rows] == ["2401.00001::0000", "2401.00001::0001"]
    assert rows[0]["score"] > rows[1]["score"]
    conn.close()


def test_search_fts_filters_by_category_author_and_date(tmp_path):
    config = _make_config(tmp_path)
    conn = db.get_connection(config.storage.db_path)
    _seed_document(conn, "2401.00001", ["Yoshua Bengio"], ["cs.CL"], "2024-01-01T00:00:00Z")
    _seed_document(conn, "2401.00002", ["Someone Else"], ["cs.LG"], "2023-01-01T00:00:00Z")
    db.insert_chunks(
        conn,
        [
            db.ChunkRecord(
                arxiv_id="2401.00001", chunk_index=0, section_title="Body",
                text="neural network training", token_count=3, page_start=0, page_end=0,
            ),
            db.ChunkRecord(
                arxiv_id="2401.00002", chunk_index=0, section_title="Body",
                text="neural network training", token_count=3, page_start=0, page_end=0,
            ),
        ],
    )

    by_category = db.search_fts(conn, '"neural"', limit=10, categories=["cs.CL"])
    assert [r["chunk_id"] for r in by_category] == ["2401.00001::0000"]

    by_author = db.search_fts(conn, '"neural"', limit=10, authors=["Someone Else"])
    assert [r["chunk_id"] for r in by_author] == ["2401.00002::0000"]

    by_date = db.search_fts(conn, '"neural"', limit=10, date_from="2024-01-01")
    assert [r["chunk_id"] for r in by_date] == ["2401.00001::0000"]
    conn.close()


def test_backfill_fts_indexes_missing_chunks_and_is_idempotent(tmp_path):
    config = _make_config(tmp_path)
    conn = db.get_connection(config.storage.db_path)
    _seed_document(conn, "2401.00001", [], ["cs.CL"], "2024-01-01T00:00:00Z")
    now = "2024-01-01T00:00:00Z"
    conn.execute(
        """
        INSERT INTO chunks (
            chunk_id, arxiv_id, chunk_index, section_title, text,
            token_count, page_start, page_end, status, created_at, updated_at
        ) VALUES ('2401.00001::0000', '2401.00001', 0, 'Body', 'legacy chunk text',
                  3, 0, 0, 'pending_embedding', ?, ?)
        """,
        (now, now),
    )
    conn.commit()

    assert db.search_fts(conn, '"legacy"', limit=10) == []

    n = db.backfill_fts(conn)
    assert n == 1
    rows = db.search_fts(conn, '"legacy"', limit=10)
    assert [r["chunk_id"] for r in rows] == ["2401.00001::0000"]

    n_again = db.backfill_fts(conn)
    assert n_again == 0
    conn.close()


def test_get_chunks_by_ids(tmp_path):
    config = _make_config(tmp_path)
    conn = db.get_connection(config.storage.db_path)
    _seed_document(conn, "2401.00001", [], ["cs.CL"], "2024-01-01T00:00:00Z")
    db.insert_chunks(
        conn,
        [
            db.ChunkRecord(
                arxiv_id="2401.00001", chunk_index=0, section_title="Body",
                text="first", token_count=1, page_start=0, page_end=0,
            ),
            db.ChunkRecord(
                arxiv_id="2401.00001", chunk_index=1, section_title="Body",
                text="second", token_count=1, page_start=0, page_end=0,
            ),
        ],
    )

    rows = db.get_chunks_by_ids(conn, ["2401.00001::0001"])
    assert len(rows) == 1
    assert rows[0]["text"] == "second"

    assert db.get_chunks_by_ids(conn, []) == []
    conn.close()
