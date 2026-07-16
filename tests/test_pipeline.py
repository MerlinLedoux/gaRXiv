from pathlib import Path

import pymupdf
import responses

from garxiv.config import Config, StorageConfig
from garxiv.ingestion.arxiv_client import API_URL
from garxiv.ingestion.pipeline import run_ingestion

FIXTURES = Path(__file__).parent / "fixtures"


def _sample_pdf_bytes() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Title", fontsize=18, fontname="helv")
    page.insert_text((72, 100), "Body text.", fontsize=10, fontname="helv")
    data = doc.tobytes()
    doc.close()
    return data


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


def _mock_arxiv_and_pdfs():
    feed_xml = (FIXTURES / "sample_feed.xml").read_bytes()
    responses.add(responses.GET, API_URL, body=feed_xml, status=200)
    pdf_bytes = _sample_pdf_bytes()
    responses.add(
        responses.GET, "http://arxiv.org/pdf/2401.00001v1", body=pdf_bytes, status=200
    )
    responses.add(
        responses.GET, "http://arxiv.org/pdf/2401.00002v2", body=pdf_bytes, status=200
    )


@responses.activate
def test_first_run_ingests_everything(tmp_path):
    _mock_arxiv_and_pdfs()
    config = _make_config(tmp_path)

    summary = run_ingestion(config)

    assert summary.discovered == 2
    assert summary.skipped == 0
    assert summary.downloaded == 2
    assert summary.parsed == 2
    assert summary.errors == 0

    parsed_files = list((tmp_path / "parsed").glob("*.json"))
    assert len(parsed_files) == 2


@responses.activate
def test_second_run_skips_already_processed_documents(tmp_path):
    _mock_arxiv_and_pdfs()
    config = _make_config(tmp_path)
    run_ingestion(config)

    responses.reset()
    _mock_arxiv_and_pdfs()

    summary = run_ingestion(config)

    assert summary.discovered == 2
    assert summary.skipped == 2
    assert summary.downloaded == 0
    assert summary.parsed == 0
    assert summary.errors == 0
