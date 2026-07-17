from garxiv.chunking.chunker import chunk_document, count_tokens
from garxiv.config import ChunkingConfig


def _section(title: str, n_lines: int, words_per_line: int = 8, page_span: int = 5) -> dict:
    lines = []
    for i in range(n_lines):
        text = " ".join(f"{title.lower()}word{i}_{j}" for j in range(words_per_line))
        page = i * page_span // max(n_lines, 1)
        lines.append({"text": text, "page": page})
    return {"title": title, "lines": lines}


def test_large_section_splits_into_multiple_overlapping_chunks():
    config = ChunkingConfig(target_tokens=50, overlap_ratio=0.2, min_chunk_tokens=10)
    parsed = {"sections": [_section("Body", n_lines=40)]}

    chunks = chunk_document(parsed, config)

    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count == count_tokens(c.text)


def test_chunk_indices_and_page_ranges_are_ordered():
    config = ChunkingConfig(target_tokens=50, overlap_ratio=0.2, min_chunk_tokens=10)
    parsed = {"sections": [_section("Body", n_lines=40, page_span=5)]}

    chunks = chunk_document(parsed, config)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert c.page_start is not None and c.page_end is not None
        assert c.page_start <= c.page_end


def test_skip_sections_are_excluded():
    config = ChunkingConfig(
        target_tokens=50, overlap_ratio=0.2, min_chunk_tokens=10, skip_sections=["references"]
    )
    parsed = {
        "sections": [
            _section("Introduction", n_lines=5),
            _section("References", n_lines=20),
        ]
    }

    chunks = chunk_document(parsed, config)

    assert all(c.section_title != "References" for c in chunks)


def test_small_section_produces_single_chunk():
    config = ChunkingConfig(target_tokens=800, overlap_ratio=0.125, min_chunk_tokens=100)
    parsed = {"sections": [_section("Body", n_lines=2, words_per_line=5)]}

    chunks = chunk_document(parsed, config)

    assert len(chunks) == 1
    assert chunks[0].token_count == count_tokens(chunks[0].text)
