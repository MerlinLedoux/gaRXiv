from garxiv import db
from garxiv.config import Config, EmbeddingConfig, StorageConfig
from garxiv.embeddings.base import EmbeddingProvider
from garxiv.embeddings.pipeline import run_embedding
from garxiv.vectorstore import factory as vectorstore_factory
from garxiv.vectorstore.qdrant_store import QdrantVectorStore


class FakeEmbeddingProvider(EmbeddingProvider):
    name = "fake-8d"
    dimension = 8

    def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        return [[float(len(t))] * self.dimension for t in texts]


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


def test_end_to_end_embedding_upserts_metadata_into_qdrant(tmp_path, monkeypatch):
    config = _make_config(tmp_path)

    conn = db.get_connection(config.storage.db_path)
    entry = db.ArxivEntry(
        arxiv_id="2401.00001",
        version=1,
        title="T",
        abstract="A",
        authors=[],
        categories=["cs.CL", "cs.LG"],
        published_date="2024-01-15T00:00:00Z",
        updated_date="2024-01-15T00:00:00Z",
        pdf_url="http://arxiv.org/pdf/2401.00001v1",
    )
    db.upsert_discovered(conn, entry)
    chunk_ids = db.insert_chunks(conn, [
        db.ChunkRecord(
            arxiv_id="2401.00001", chunk_index=0, section_title="Intro",
            text="hello world", token_count=2, page_start=0, page_end=0,
        ),
    ])
    conn.close()

    provider = FakeEmbeddingProvider()
    store = QdrantVectorStore(url=":memory:", collection_name="unused", distance="cosine")
    monkeypatch.setattr(
        "garxiv.embeddings.pipeline.embedding_factory.get_provider", lambda cfg: provider
    )
    monkeypatch.setattr(
        "garxiv.embeddings.pipeline.vectorstore_factory.get_store",
        lambda cfg, collection_name: store,
    )

    expected_name = vectorstore_factory.collection_name(provider.name, provider.dimension)
    summary = run_embedding(config)

    assert summary.embedded == 1
    matches = store.query([1.0] * provider.dimension, top_k=10)
    assert len(matches) == 1
    assert matches[0].id == chunk_ids[0]
    assert matches[0].metadata["arxiv_id"] == "2401.00001"
    assert matches[0].metadata["categories"] == ["cs.CL", "cs.LG"]
    assert matches[0].metadata["page_start"] == 0
    assert expected_name == "garxiv_fake_8d_8d"
