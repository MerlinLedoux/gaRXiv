import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from garxiv.vectorstore.qdrant_store import QdrantVectorStore

QDRANT_URL = "http://localhost:6333"


@pytest.fixture
def qdrant_url():
    try:
        QdrantClient(url=QDRANT_URL).get_collections()
    except (ResponseHandlingException, ConnectionError, OSError) as exc:
        pytest.skip(f"Qdrant non joignable sur {QDRANT_URL} (docker-compose up -d requis): {exc}")
    return QDRANT_URL


@pytest.mark.integration
def test_upsert_and_query_against_real_qdrant(qdrant_url):
    store = QdrantVectorStore(url=qdrant_url, collection_name="garxiv_test_integration")
    store.ensure_collection(dimension=4)

    store.upsert(
        ids=["it::0000"], vectors=[[1.0, 0.0, 0.0, 0.0]], metadatas=[{"arxiv_id": "it"}]
    )

    matches = store.query([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert matches[0].id == "it::0000"

    store.delete(["it::0000"])
