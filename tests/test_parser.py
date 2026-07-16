from pathlib import Path

import pymupdf

from garxiv.ingestion.parser import extract_structure, parse_pdf


def _make_sample_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Introduction", fontsize=18, fontname="helv")
    page.insert_text((72, 100), "This is the introduction body text.", fontsize=10, fontname="helv")
    page.insert_text((72, 130), "Methods", fontsize=18, fontname="helv")
    page.insert_text((72, 158), "This is the methods body text.", fontsize=10, fontname="helv")
    doc.save(path)
    doc.close()


def test_extract_structure_splits_sections(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    _make_sample_pdf(pdf_path)

    structure = extract_structure(pdf_path)

    titles = [s["title"] for s in structure["sections"]]
    assert "Introduction" in titles
    assert "Methods" in titles
    assert "introduction body text" in structure["full_text"]


def test_parse_pdf_writes_json(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    _make_sample_pdf(pdf_path)
    parsed_dir = tmp_path / "parsed"

    dest = parse_pdf(pdf_path, "2401.00001", parsed_dir)

    assert dest.exists()
    assert dest.name == "2401.00001.json"
