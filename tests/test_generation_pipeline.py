from pathlib import Path

import pytest

from garxiv.config import Config, GenerationConfig, StorageConfig
from garxiv.generation.pipeline import run_generation
from garxiv.llm.base import LLMProvider, LLMUnavailableError
from garxiv.retrieval.pipeline import SearchFilters, SearchResult


class FakeLLMProvider(LLMProvider):
    name = "fake-llm"

    def __init__(self, response: str = "the answer", fail: bool = False):
        self._response = response
        self._fail = fail
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        if self._fail:
            raise LLMUnavailableError("simulated failure")
        return self._response


def _make_config(tmp_path: Path, **generation_kwargs) -> Config:
    return Config(
        categories=["cs.CL"],
        authors=[],
        storage=StorageConfig(
            db_path=tmp_path / "garxiv.db",
            pdf_dir=tmp_path / "pdfs",
            parsed_dir=tmp_path / "parsed",
        ),
        max_results_per_run=10,
        generation=GenerationConfig(**generation_kwargs),
    )


def _make_result(n: int, arxiv_id: str = "2401.00001", section: str = "Body") -> SearchResult:
    return SearchResult(
        chunk_id=f"chunk-{n}",
        score=1.0 / n,
        text=f"text of chunk {n}",
        metadata={
            "arxiv_id": arxiv_id,
            "section_title": section,
            "chunk_index": n,
            "page_start": n,
            "page_end": n,
        },
    )


def _patch_search(monkeypatch, results: list[SearchResult]):
    calls: list[tuple] = []

    def fake_run_search(config, query, top_k=None, filters=None):
        calls.append((query, top_k, filters))
        return results

    monkeypatch.setattr("garxiv.generation.pipeline.run_search", fake_run_search)
    return calls


def _patch_llm(monkeypatch, llm: FakeLLMProvider):
    monkeypatch.setattr("garxiv.generation.pipeline.llm_factory.get_provider", lambda cfg: llm)


def test_builds_grounded_answer_from_top_k_chunks(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    results = [_make_result(1, arxiv_id="2401.00001"), _make_result(2, arxiv_id="2401.00002"), _make_result(3, arxiv_id="2401.00003")]
    _patch_search(monkeypatch, results)
    llm = FakeLLMProvider(response="synthesized answer")
    _patch_llm(monkeypatch, llm)

    result = run_generation(config, "my question", top_k=5)

    assert result.answer == "synthesized answer"
    prompt = llm.calls[0][0]
    assert "[1]" in prompt and "2401.00001" in prompt
    assert "[2]" in prompt and "2401.00002" in prompt
    assert "[3]" in prompt and "2401.00003" in prompt


def test_citation_numbering_matches_sources_order(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    results = [_make_result(1, arxiv_id="A"), _make_result(2, arxiv_id="B")]
    _patch_search(monkeypatch, results)
    llm = FakeLLMProvider()
    _patch_llm(monkeypatch, llm)

    result = run_generation(config, "q", top_k=5)

    assert [s.metadata["arxiv_id"] for s in result.sources] == ["A", "B"]
    prompt = llm.calls[0][0]
    assert prompt.index("[1]") < prompt.index("[2]")
    assert prompt.index("A") < prompt.index("B")


def test_context_top_k_truncates_chunks_sent_to_llm(tmp_path, monkeypatch):
    config = _make_config(tmp_path, context_top_k=3)
    results = [_make_result(i) for i in range(1, 11)]
    _patch_search(monkeypatch, results)
    llm = FakeLLMProvider()
    _patch_llm(monkeypatch, llm)

    result = run_generation(config, "q", top_k=10)

    assert len(result.sources) == 3
    prompt = llm.calls[0][0]
    assert "[3]" in prompt
    assert "[4]" not in prompt


def test_max_chars_per_chunk_truncates_chunk_text(tmp_path, monkeypatch):
    config = _make_config(tmp_path, max_chars_per_chunk=10)
    long_result = SearchResult(
        chunk_id="c1",
        score=1.0,
        text="x" * 100,
        metadata={"arxiv_id": "A", "section_title": "Body", "chunk_index": 0, "page_start": 0, "page_end": 0},
    )
    _patch_search(monkeypatch, [long_result])
    llm = FakeLLMProvider()
    _patch_llm(monkeypatch, llm)

    run_generation(config, "q", top_k=5)

    prompt = llm.calls[0][0]
    assert "x" * 100 not in prompt
    assert "x" * 10 in prompt


def test_zero_results_skips_llm_and_returns_no_results_answer(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    _patch_search(monkeypatch, [])
    llm = FakeLLMProvider()
    _patch_llm(monkeypatch, llm)

    result = run_generation(config, "q", top_k=5)

    assert llm.calls == []
    assert result.sources == []
    assert "Aucun document pertinent" in result.answer


def test_llm_unavailable_error_propagates(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    _patch_search(monkeypatch, [_make_result(1)])
    llm = FakeLLMProvider(fail=True)
    _patch_llm(monkeypatch, llm)

    with pytest.raises(LLMUnavailableError):
        run_generation(config, "q", top_k=5)


def test_run_generation_passes_through_top_k_and_filters_to_search(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    calls = _patch_search(monkeypatch, [_make_result(1)])
    llm = FakeLLMProvider()
    _patch_llm(monkeypatch, llm)
    filters = SearchFilters(categories=["cs.CL"])

    run_generation(config, "q", top_k=7, filters=filters)

    assert calls[0] == ("q", 7, filters)
