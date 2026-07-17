import pytest

from garxiv import db
from garxiv.config import Config, EmbeddingConfig, StorageConfig
from garxiv.embeddings import factory as embedding_factory
from garxiv.embeddings.base import EmbeddingProvider
from garxiv.embeddings.pipeline import run_embedding
from garxiv.vectorstore.base import VectorMatch, VectorStore


class FakeEmbeddingProvider(EmbeddingProvider):
    name = "fake-8d"
    dimension = 8

    def __init__(self, fail_on: set[str] | None = None):
        self.fail_on = fail_on or set()
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        self.calls.append(texts)
        if self.fail_on & set(texts):
            raise RuntimeError("simulated embedding failure")
        return [[float(len(t))] * self.dimension for t in texts]


class FakeVectorStore(VectorStore):
    def __init__(self):
        self.points: dict[str, tuple[list[float], dict]] = {}
        self.ensure_calls: list[int] = []

    def ensure_collection(self, dimension: int) -> None:
        self.ensure_calls.append(dimension)

    def upsert(self, ids, vectors, metadatas) -> None:
        for cid, vec, meta in zip(ids, vectors, metadatas):
            self.points[cid] = (vec, meta)

    def query(self, vector, top_k=10, filters=None) -> list[VectorMatch]:
        return [
            VectorMatch(id=cid, score=1.0, metadata=meta) for cid, (_, meta) in self.points.items()
        ][:top_k]

    def delete(self, ids) -> None:
        for cid in ids:
            self.points.pop(cid, None)


def _make_config(tmp_path) -> Config:
    return Config(
        categories=["cs.CL"],
        authors=[],
        storage=StorageConfig(
            db_path=tmp_path / "garxiv.db",
            pdf_dir=tmp_path / "pdfs",
            parsed_dir=tmp_path / "parsed",
        ),
        max_results_per_run=10,
        embeddings=EmbeddingConfig(provider="local_bge", batch_size=2),
    )


def _seed_chunks(config: Config, n: int) -> list[str]:
    conn = db.get_connection(config.storage.db_path)
    entry = db.ArxivEntry(
        arxiv_id="2401.00001",
        version=1,
        title="T",
        abstract="A",
        authors=[],
        categories=["cs.CL"],
        published_date="2024-01-01T00:00:00Z",
        updated_date="2024-01-01T00:00:00Z",
        pdf_url="http://arxiv.org/pdf/2401.00001v1",
    )
    db.upsert_discovered(conn, entry)
    records = [
        db.ChunkRecord(
            arxiv_id="2401.00001",
            chunk_index=i,
            section_title="Body",
            text=f"chunk text {i}",
            token_count=3,
            page_start=0,
            page_end=0,
        )
        for i in range(n)
    ]
    chunk_ids = db.insert_chunks(conn, records)
    conn.close()
    return chunk_ids


def test_azure_openai_provider_is_not_implemented():
    config = EmbeddingConfig(
        provider="azure_openai", azure_endpoint="https://x", azure_deployment="d", azure_api_key_env="X"
    )
    with pytest.raises(NotImplementedError):
        embedding_factory.get_provider(config)


def test_run_embedding_batches_and_marks_chunks_embedded(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    chunk_ids = _seed_chunks(config, n=5)

    provider = FakeEmbeddingProvider()
    store = FakeVectorStore()
    monkeypatch.setattr(
        "garxiv.embeddings.pipeline.embedding_factory.get_provider", lambda cfg: provider
    )
    monkeypatch.setattr(
        "garxiv.embeddings.pipeline.vectorstore_factory.get_store", lambda cfg, collection_name: store
    )

    summary = run_embedding(config)

    assert summary.embedded == 5
    assert summary.errors == 0
    assert store.ensure_calls == [8]
    assert len(store.points) == 5
    assert [len(c) for c in provider.calls] == [2, 2, 1]

    conn = db.get_connection(config.storage.db_path)
    for chunk_id in chunk_ids:
        row = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        assert row["status"] == "embedded"
        assert row["embedding_provider"] == "fake-8d"
        assert row["embedding_dim"] == 8
    doc = db.get_document(conn, "2401.00001")
    assert doc["status"] == "embedded"
    conn.close()


def test_run_embedding_does_not_reprocess_embedded_chunks(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    _seed_chunks(config, n=2)

    provider = FakeEmbeddingProvider()
    store = FakeVectorStore()
    monkeypatch.setattr(
        "garxiv.embeddings.pipeline.embedding_factory.get_provider", lambda cfg: provider
    )
    monkeypatch.setattr(
        "garxiv.embeddings.pipeline.vectorstore_factory.get_store", lambda cfg, collection_name: store
    )

    run_embedding(config)
    summary = run_embedding(config)

    assert summary.embedded == 0
    assert summary.errors == 0


def test_run_embedding_failed_batch_does_not_block_others(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    _seed_chunks(config, n=3)

    provider = FakeEmbeddingProvider(fail_on={"chunk text 0"})
    store = FakeVectorStore()
    monkeypatch.setattr(
        "garxiv.embeddings.pipeline.embedding_factory.get_provider", lambda cfg: provider
    )
    monkeypatch.setattr(
        "garxiv.embeddings.pipeline.vectorstore_factory.get_store", lambda cfg, collection_name: store
    )

    summary = run_embedding(config)

    assert summary.errors == 2
    assert summary.embedded == 1

    conn = db.get_connection(config.storage.db_path)
    rows = conn.execute("SELECT status, error_message FROM chunks ORDER BY chunk_index").fetchall()
    # batch_size=2: chunk 0 and 1 share a batch with the failing chunk 0, so both
    # are marked in error; chunk 2 is alone in the next batch and succeeds.
    assert rows[0]["status"] == "pending_embedding"
    assert rows[0]["error_message"] is not None
    assert rows[1]["status"] == "pending_embedding"
    assert rows[1]["error_message"] is not None
    assert rows[2]["status"] == "embedded"
    conn.close()
