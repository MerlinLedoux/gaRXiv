from garxiv.vectorstore.qdrant_store import QdrantVectorStore


def _make_store() -> QdrantVectorStore:
    return QdrantVectorStore(url=":memory:", collection_name="test_collection", distance="cosine")


def test_ensure_collection_is_idempotent():
    store = _make_store()
    store.ensure_collection(dimension=4)
    store.ensure_collection(dimension=4)  # should not raise


def test_upsert_and_query_returns_chunk_id_and_metadata():
    store = _make_store()
    store.ensure_collection(dimension=4)

    store.upsert(
        ids=["2401.00001::0000", "2401.00001::0001"],
        vectors=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        metadatas=[
            {"arxiv_id": "2401.00001", "section_title": "Intro"},
            {"arxiv_id": "2401.00001", "section_title": "Methods"},
        ],
    )

    matches = store.query([1.0, 0.0, 0.0, 0.0], top_k=1)

    assert len(matches) == 1
    assert matches[0].id == "2401.00001::0000"
    assert matches[0].metadata["section_title"] == "Intro"


def test_delete_removes_points():
    store = _make_store()
    store.ensure_collection(dimension=4)
    store.upsert(
        ids=["a::0000"], vectors=[[1.0, 0.0, 0.0, 0.0]], metadatas=[{"arxiv_id": "a"}]
    )

    store.delete(["a::0000"])
    matches = store.query([1.0, 0.0, 0.0, 0.0], top_k=10)

    assert matches == []
