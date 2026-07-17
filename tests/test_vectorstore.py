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


def _seed_filterable(store: QdrantVectorStore) -> None:
    store.ensure_collection(dimension=4)
    store.upsert(
        ids=["a::0000", "a::0001", "a::0002"],
        vectors=[[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0]],
        metadatas=[
            {"categories": ["cs.CL"], "published_date": "2024-01-15T00:00:00Z"},
            {"categories": ["cs.LG"], "published_date": "2023-06-01T00:00:00Z"},
            {"categories": ["cs.CL", "cs.LG"], "published_date": "2024-06-01T00:00:00Z"},
        ],
    )


def test_query_filter_match_any():
    store = _make_store()
    _seed_filterable(store)

    matches = store.query(
        [1.0, 0.0, 0.0, 0.0], top_k=10, filters={"categories": {"any": ["cs.LG"]}}
    )

    assert {m.id for m in matches} == {"a::0001", "a::0002"}


def test_query_filter_datetime_range():
    store = _make_store()
    _seed_filterable(store)

    matches = store.query(
        [1.0, 0.0, 0.0, 0.0],
        top_k=10,
        filters={"published_date": {"gte": "2024-01-01T00:00:00Z"}},
    )

    assert {m.id for m in matches} == {"a::0000", "a::0002"}


def test_query_filter_combined_any_and_range():
    store = _make_store()
    _seed_filterable(store)

    matches = store.query(
        [1.0, 0.0, 0.0, 0.0],
        top_k=10,
        filters={
            "categories": {"any": ["cs.CL"]},
            "published_date": {"gte": "2024-01-01T00:00:00Z", "lte": "2024-01-31T23:59:59Z"},
        },
    )

    assert {m.id for m in matches} == {"a::0000"}


def test_query_filter_scalar_equality_still_works():
    store = _make_store()
    store.ensure_collection(dimension=4)
    store.upsert(
        ids=["a::0000", "a::0001"],
        vectors=[[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
        metadatas=[{"arxiv_id": "a"}, {"arxiv_id": "b"}],
    )

    matches = store.query([1.0, 0.0, 0.0, 0.0], top_k=10, filters={"arxiv_id": "b"})

    assert {m.id for m in matches} == {"a::0001"}
