import json
from pathlib import Path

from garxiv import db
from garxiv.chunking.pipeline import run_chunking
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


def _seed_parsed_document(config: Config) -> None:
    conn = db.get_connection(config.storage.db_path)
    entry = db.ArxivEntry(
        arxiv_id="2401.00001",
        version=1,
        title="Sample Paper",
        abstract="An abstract.",
        authors=["A. Author"],
        categories=["cs.CL"],
        published_date="2024-01-01T00:00:00Z",
        updated_date="2024-01-01T00:00:00Z",
        pdf_url="http://arxiv.org/pdf/2401.00001v1",
    )
    db.upsert_discovered(conn, entry)

    parsed_dir = Path(config.storage.parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    parsed_path = parsed_dir / "2401.00001.json"
    payload = {
        "arxiv_id": "2401.00001",
        "page_count": 1,
        "full_text": "Introduction text.",
        "sections": [
            {
                "title": "Introduction",
                "text": "Introduction text.",
                "lines": [{"text": "Introduction text.", "page": 0}],
            }
        ],
    }
    parsed_path.write_text(json.dumps(payload), encoding="utf-8")

    db.mark_downloaded(conn, "2401.00001", str(parsed_dir / "dummy.pdf"))
    db.mark_parsed(conn, "2401.00001", str(parsed_path))
    conn.close()


def test_run_chunking_creates_chunks_and_marks_document(tmp_path):
    config = _make_config(tmp_path)
    _seed_parsed_document(config)

    summary = run_chunking(config)

    assert summary.documents_chunked == 1
    assert summary.chunks_created >= 1
    assert summary.errors == 0

    conn = db.get_connection(config.storage.db_path)
    doc = db.get_document(conn, "2401.00001")
    assert doc["status"] == "chunked"
    chunks = db.get_chunks_by_document(conn, "2401.00001")
    assert len(chunks) == summary.chunks_created
    conn.close()


def test_second_run_skips_already_chunked_documents(tmp_path):
    config = _make_config(tmp_path)
    _seed_parsed_document(config)
    run_chunking(config)

    summary = run_chunking(config)

    assert summary.documents_chunked == 0
    assert summary.chunks_created == 0
