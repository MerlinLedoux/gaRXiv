from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from garxiv.config import ChunkingConfig


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


@dataclass
class Chunk:
    chunk_index: int
    section_title: str
    text: str
    token_count: int
    page_start: int | None
    page_end: int | None


def _tokenize_section(lines: list[dict]) -> tuple[list[int], list[int | None]]:
    enc = _encoding()
    tokens: list[int] = []
    pages: list[int | None] = []
    for line in lines:
        line_tokens = enc.encode(line["text"] + "\n")
        tokens.extend(line_tokens)
        pages.extend([line.get("page")] * len(line_tokens))
    return tokens, pages


def _windowed_spans(
    n: int, target_tokens: int, overlap_ratio: float, min_chunk_tokens: int
) -> list[tuple[int, int]]:
    step = max(target_tokens - int(target_tokens * overlap_ratio), 1)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + target_tokens, n)
        if n - start < min_chunk_tokens and spans:
            spans[-1] = (spans[-1][0], n)
            break
        spans.append((start, end))
        if end == n:
            break
        start += step
    return spans


def chunk_document(parsed: dict, config: ChunkingConfig) -> list[Chunk]:
    """Split a cleaned, parsed document into overlapping token windows,
    section by section. Sections in `config.skip_sections` (case-insensitive)
    are excluded. Each chunk's page range is derived exactly from the pages
    of the tokens it contains (no character-offset approximation)."""
    enc = _encoding()
    skip = {s.strip().lower() for s in config.skip_sections}
    chunks: list[Chunk] = []
    index = 0

    for section in parsed.get("sections", []):
        if section["title"].strip().lower() in skip:
            continue
        lines = section.get("lines") or []
        tokens, pages = _tokenize_section(lines)
        if not tokens:
            continue

        for start, end in _windowed_spans(
            len(tokens), config.target_tokens, config.overlap_ratio, config.min_chunk_tokens
        ):
            text = enc.decode(tokens[start:end])
            window_pages = [p for p in pages[start:end] if p is not None]
            chunks.append(Chunk(
                chunk_index=index,
                section_title=section["title"],
                text=text,
                token_count=end - start,
                page_start=min(window_pages) if window_pages else None,
                page_end=max(window_pages) if window_pages else None,
            ))
            index += 1

    return chunks
