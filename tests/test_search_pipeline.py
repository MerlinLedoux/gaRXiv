from pathlib import Path

from garxiv import db
from garxiv.config import (
    Config,
    LLMConfig,
    QueryTransformConfig,
    RerankConfig,
    SearchConfig,
    StorageConfig,
)
from garxiv.embeddings.base import EmbeddingProvider
from garxiv.llm.base import LLMProvider, LLMUnavailableError
from garxiv.rerank.base import Reranker
from garxiv.retrieval.pipeline import SearchFilters, run_search
from garxiv.vectorstore.base import VectorMatch, VectorStore


class FakeEmbeddingProvider(EmbeddingProvider):
    name = "fake-4d"
    dimension = 4

    def __init__(self):
        self.calls: list[tuple[list[str], bool]] = []

    def embed(self, texts, is_query=False):
        self.calls.append((list(texts), is_query))
        return [[float(len(t))] * self.dimension for t in texts]


class FakeVectorStore(VectorStore):
    def __init__(self, matches: list[VectorMatch] | None = None):
        self._matches = matches or []
        self.query_calls: list[tuple[list[float], int, dict | None]] = []

    def ensure_collection(self, dimension):
        pass

    def upsert(self, ids, vectors, metadatas):
        raise NotImplementedError

    def query(self, vector, top_k=10, filters=None):
        self.query_calls.append((vector, top_k, filters))
        return self._matches[:top_k]

    def delete(self, ids):
        raise NotImplementedError


class FakeLLMProvider(LLMProvider):
    name = "fake-llm"

    def __init__(self, multi_query_response: str = "", hyde_response: str = "", fail: bool = False):
        self._multi_query_response = multi_query_response
        self._hyde_response = hyde_response
        self._fail = fail

    def generate(self, prompt: str, system: str | None = None) -> str:
        if self._fail:
            raise LLMUnavailableError("simulated failure")
        return self._hyde_response if "excerpt" in prompt else self._multi_query_response


class FakeReranker(Reranker):
    name = "fake-reranker"

    def __init__(self, scores_by_text: dict[str, float]):
        self._scores_by_text = scores_by_text
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        return [self._scores_by_text.get(t, 0.0) for t in texts]


def _make_config(tmp_path: Path, **search_kwargs) -> Config:
    search_kwargs.setdefault("hybrid_enabled", False)
    return Config(
        categories=["cs.CL"],
        authors=[],
        storage=StorageConfig(
            db_path=tmp_path / "garxiv.db",
            pdf_dir=tmp_path / "pdfs",
            parsed_dir=tmp_path / "parsed",
        ),
        max_results_per_run=10,
        search=SearchConfig(**search_kwargs),
    )


def _seed_chunk(
    conn,
    arxiv_id: str,
    chunk_index: int,
    text: str,
    categories: list[str],
    authors: list[str] | None = None,
    published_date: str = "2024-01-01T00:00:00Z",
) -> str:
    if conn.execute("SELECT 1 FROM documents WHERE arxiv_id = ?", (arxiv_id,)).fetchone() is None:
        entry = db.ArxivEntry(
            arxiv_id=arxiv_id,
            version=1,
            title="T",
            abstract="A",
            authors=authors or [],
            categories=categories,
            published_date=published_date,
            updated_date=published_date,
            pdf_url=f"http://arxiv.org/pdf/{arxiv_id}v1",
        )
        db.upsert_discovered(conn, entry)
    chunk_ids = db.insert_chunks(
        conn,
        [
            db.ChunkRecord(
                arxiv_id=arxiv_id, chunk_index=chunk_index, section_title="Body",
                text=text, token_count=len(text.split()), page_start=0, page_end=0,
            )
        ],
    )
    return chunk_ids[0]


def _patch_factories(monkeypatch, provider, store, llm=None, reranker=None):
    monkeypatch.setattr("garxiv.retrieval.pipeline.embedding_factory.get_provider", lambda cfg: provider)
    monkeypatch.setattr(
        "garxiv.retrieval.pipeline.vectorstore_factory.get_store", lambda cfg, collection_name: store
    )
    if llm is not None:
        monkeypatch.setattr("garxiv.retrieval.pipeline.llm_factory.get_provider", lambda cfg: llm)
    if reranker is not None:
        monkeypatch.setattr("garxiv.retrieval.pipeline.rerank_factory.get_reranker", lambda cfg: reranker)


def test_dense_only_returns_store_order(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    conn = db.get_connection(config.storage.db_path)
    cid_a = _seed_chunk(conn, "2401.00001", 0, "alpha content", ["cs.CL"])
    cid_b = _seed_chunk(conn, "2401.00001", 1, "beta content", ["cs.CL"])
    conn.close()

    provider = FakeEmbeddingProvider()
    store = FakeVectorStore(matches=[
        VectorMatch(id=cid_a, score=0.9, metadata={}),
        VectorMatch(id=cid_b, score=0.5, metadata={}),
    ])
    _patch_factories(monkeypatch, provider, store)

    results = run_search(config, "my query", top_k=5)

    assert [r.chunk_id for r in results] == [cid_a, cid_b]
    assert provider.calls == [(["my query"], True)]


def test_hybrid_search_surfaces_lexical_only_match(tmp_path, monkeypatch):
    config = _make_config(tmp_path, hybrid_enabled=True)
    conn = db.get_connection(config.storage.db_path)
    cid_lexical = _seed_chunk(conn, "2401.00001", 0, "mentions XQZ123 explicitly", ["cs.CL"])
    conn.close()

    provider = FakeEmbeddingProvider()
    store = FakeVectorStore(matches=[])  # dense finds nothing
    _patch_factories(monkeypatch, provider, store)

    results = run_search(config, "XQZ123", top_k=5)

    assert [r.chunk_id for r in results] == [cid_lexical]


def test_filters_restrict_lexical_results_to_matching_category(tmp_path, monkeypatch):
    config = _make_config(tmp_path, hybrid_enabled=True)
    conn = db.get_connection(config.storage.db_path)
    cid_cl = _seed_chunk(conn, "2401.00001", 0, "shared unique term acme", ["cs.CL"])
    _seed_chunk(conn, "2401.00002", 0, "shared unique term acme", ["cs.LG"])
    conn.close()

    provider = FakeEmbeddingProvider()
    store = FakeVectorStore(matches=[])
    _patch_factories(monkeypatch, provider, store)

    results = run_search(
        config, "acme", top_k=5, filters=SearchFilters(categories=["cs.CL"])
    )

    assert [r.chunk_id for r in results] == [cid_cl]


def test_filters_propagated_to_vectorstore_query(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    conn = db.get_connection(config.storage.db_path)
    conn.close()

    provider = FakeEmbeddingProvider()
    store = FakeVectorStore(matches=[])
    _patch_factories(monkeypatch, provider, store)

    run_search(
        config, "q", top_k=5,
        filters=SearchFilters(categories=["cs.CL"], authors=["Ada"], date_from="2024-01-01"),
    )

    _, _, filters = store.query_calls[0]
    assert filters == {
        "categories": {"any": ["cs.CL"]},
        "authors": {"any": ["Ada"]},
        "published_date": {"gte": "2024-01-01"},
    }


def test_multi_query_expands_dense_search(tmp_path, monkeypatch):
    config = _make_config(tmp_path, query_transform=QueryTransformConfig(mode="multi_query", multi_query_count=2))
    conn = db.get_connection(config.storage.db_path)
    cid = _seed_chunk(conn, "2401.00001", 0, "content", ["cs.CL"])
    conn.close()

    provider = FakeEmbeddingProvider()
    store = FakeVectorStore(matches=[VectorMatch(id=cid, score=1.0, metadata={})])
    llm = FakeLLMProvider(multi_query_response="variant one\nvariant two\n")
    _patch_factories(monkeypatch, provider, store, llm=llm)

    run_search(config, "original", top_k=5)

    assert provider.calls[0][0] == ["original", "variant one", "variant two"]
    assert provider.calls[0][1] is True
    assert len(store.query_calls) == 3


def test_hyde_embeds_hypothetical_document_as_non_query(tmp_path, monkeypatch):
    config = _make_config(tmp_path, query_transform=QueryTransformConfig(mode="hyde"))
    conn = db.get_connection(config.storage.db_path)
    cid = _seed_chunk(conn, "2401.00001", 0, "content", ["cs.CL"])
    conn.close()

    provider = FakeEmbeddingProvider()
    store = FakeVectorStore(matches=[VectorMatch(id=cid, score=1.0, metadata={})])
    llm = FakeLLMProvider(hyde_response="a hypothetical scientific excerpt")
    _patch_factories(monkeypatch, provider, store, llm=llm)

    run_search(config, "original", top_k=5)

    # first dense call: the original query, is_query=True
    assert provider.calls[0] == (["original"], True)
    # second dense call: the HyDE document, is_query=False
    assert provider.calls[1] == (["a hypothetical scientific excerpt"], False)


def test_llm_unavailable_falls_back_to_original_query(tmp_path, monkeypatch):
    config = _make_config(tmp_path, query_transform=QueryTransformConfig(mode="multi_query"))
    conn = db.get_connection(config.storage.db_path)
    cid = _seed_chunk(conn, "2401.00001", 0, "content", ["cs.CL"])
    conn.close()

    provider = FakeEmbeddingProvider()
    store = FakeVectorStore(matches=[VectorMatch(id=cid, score=1.0, metadata={})])
    llm = FakeLLMProvider(fail=True)
    _patch_factories(monkeypatch, provider, store, llm=llm)

    results = run_search(config, "original", top_k=5)

    assert provider.calls[0][0] == ["original"]
    assert [r.chunk_id for r in results] == [cid]


def test_rerank_overrides_fusion_order(tmp_path, monkeypatch):
    config = _make_config(tmp_path, rerank=RerankConfig(enabled=True, top_k_candidates=10))
    conn = db.get_connection(config.storage.db_path)
    cid_a = _seed_chunk(conn, "2401.00001", 0, "text a", ["cs.CL"])
    cid_b = _seed_chunk(conn, "2401.00001", 1, "text b", ["cs.CL"])
    conn.close()

    provider = FakeEmbeddingProvider()
    store = FakeVectorStore(matches=[
        VectorMatch(id=cid_a, score=0.9, metadata={}),
        VectorMatch(id=cid_b, score=0.5, metadata={}),
    ])
    reranker = FakeReranker(scores_by_text={"text a": 0.1, "text b": 0.9})
    _patch_factories(monkeypatch, provider, store, reranker=reranker)

    results = run_search(config, "q", top_k=5)

    assert [r.chunk_id for r in results] == [cid_b, cid_a]


def test_no_results_returns_empty_list(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    conn = db.get_connection(config.storage.db_path)
    conn.close()

    provider = FakeEmbeddingProvider()
    store = FakeVectorStore(matches=[])
    _patch_factories(monkeypatch, provider, store)

    assert run_search(config, "nothing", top_k=5) == []
