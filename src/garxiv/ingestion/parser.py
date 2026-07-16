import json
import statistics
from pathlib import Path

import pymupdf

HEADING_SIZE_RATIO = 1.15
HEADING_MAX_CHARS = 120


def extract_structure(pdf_path: Path | str) -> dict:
    """Extract plain text and a heuristic section breakdown from a PDF.

    Lines that are noticeably larger than the document's median font size,
    or bold, and short enough to plausibly be a heading, start a new section.
    """
    doc = pymupdf.open(pdf_path)
    lines: list[tuple[str, float, bool]] = []

    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                size = max(s["size"] for s in spans)
                bold = any(s["flags"] & pymupdf.TEXT_FONT_BOLD for s in spans)
                lines.append((text, size, bold))
    doc.close()

    if not lines:
        return {"sections": [], "full_text": ""}

    median_size = statistics.median(size for _, size, _ in lines)
    heading_threshold = median_size * HEADING_SIZE_RATIO

    sections: list[dict] = []
    current_title = "Preamble"
    current_text: list[str] = []

    for text, size, bold in lines:
        is_heading = (size >= heading_threshold or bold) and len(text) < HEADING_MAX_CHARS
        if is_heading:
            if current_text:
                sections.append({"title": current_title, "text": "\n".join(current_text)})
            current_title = text
            current_text = []
        else:
            current_text.append(text)

    if current_text:
        sections.append({"title": current_title, "text": "\n".join(current_text)})

    full_text = "\n".join(text for text, _, _ in lines)
    return {"sections": sections, "full_text": full_text}


def parse_pdf(pdf_path: Path | str, arxiv_id: str, parsed_dir: Path | str) -> Path:
    parsed_dir = Path(parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    dest = parsed_dir / f"{arxiv_id}.json"

    structure = extract_structure(pdf_path)
    payload = {"arxiv_id": arxiv_id, **structure}
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest
