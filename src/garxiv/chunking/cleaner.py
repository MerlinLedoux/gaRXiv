import re

WATERMARK_RE = re.compile(r"^arXiv:\d{4}\.\d{4,5}(v\d+)?\s*(\[[\w.\-]+\])?", re.IGNORECASE)
PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")
HEADER_FOOTER_MAX_CHARS = 200


def _is_noise_line(text: str) -> bool:
    return bool(WATERMARK_RE.match(text) or PAGE_NUMBER_RE.match(text))


def _find_recurring_lines(sections: list[dict], page_count: int, min_frequency: float) -> set[str]:
    pages_by_line: dict[str, set[int]] = {}
    for section in sections:
        for line in section.get("lines", []):
            normalized = line["text"].strip().lower()
            if not normalized or len(normalized) >= HEADER_FOOTER_MAX_CHARS:
                continue
            pages_by_line.setdefault(normalized, set()).add(line["page"])

    if page_count <= 0:
        return set()
    threshold = page_count * min_frequency
    return {
        text
        for text, pages in pages_by_line.items()
        if len(pages) > 1 and len(pages) > threshold
    }


def _merge_hyphenated(lines: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for line in lines:
        if merged and merged[-1]["text"].endswith("-") and line["text"][:1].islower():
            merged[-1] = {
                "text": merged[-1]["text"][:-1] + line["text"],
                "page": merged[-1]["page"],
            }
        else:
            merged.append(dict(line))
    return merged


def clean_document(parsed: dict, header_footer_min_frequency: float = 0.4) -> dict:
    """Remove arXiv watermark lines, isolated page numbers, recurring
    headers/footers, and merge words split by a line-end hyphen.

    Operates on the parsed JSON produced by `ingestion.parser.extract_structure`
    (each section has a `lines: [{"text", "page"}]` list) and returns a new
    dict with the same shape, `text` rebuilt from the cleaned lines.
    """
    sections = parsed.get("sections", [])
    page_count = parsed.get("page_count", 0)
    recurring = _find_recurring_lines(sections, page_count, header_footer_min_frequency)

    cleaned_sections = []
    for section in sections:
        lines = [
            line
            for line in section.get("lines", [])
            if not _is_noise_line(line["text"]) and line["text"].strip().lower() not in recurring
        ]
        lines = _merge_hyphenated(lines)
        cleaned_sections.append({
            "title": section["title"],
            "text": "\n".join(line["text"] for line in lines),
            "lines": lines,
        })

    return {**parsed, "sections": cleaned_sections}
