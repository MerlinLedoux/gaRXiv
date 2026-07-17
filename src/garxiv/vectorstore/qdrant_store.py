import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from garxiv.vectorstore.base import VectorMatch, VectorStore

_DISTANCE_MAP = {
    "cosine": qm.Distance.COSINE,
    "dot": qm.Distance.DOT,
    "euclid": qm.Distance.EUCLID,
}


def _build_filter(filters: dict) -> qm.Filter:
    return qm.Filter(
        must=[qm.FieldCondition(key=key, match=qm.MatchValue(value=value)) for key, value in filters.items()]
    )


class QdrantVectorStore(VectorStore):
    """Chunk identity (`chunk_id`, a readable string like "2401.00001::0007")
    is used everywhere in the VectorStore interface. Qdrant only accepts
    unsigned ints or UUIDs as point ids, so the conversion to a deterministic
    UUID is kept internal here; the original chunk_id is restored from the
    payload on read so callers never see the UUID."""

    def __init__(
        self,
        url: str,
        collection_name: str,
        distance: str = "cosine",
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 100,
    ):
        self._client = (
            QdrantClient(location=url) if url == ":memory:" else QdrantClient(url=url)
        )
        self._collection = collection_name
        self._distance = _DISTANCE_MAP[distance]
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construct = hnsw_ef_construct

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def ensure_collection(self, dimension: int) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            self._collection,
            vectors_config=qm.VectorParams(size=dimension, distance=self._distance),
            hnsw_config=qm.HnswConfigDiff(m=self._hnsw_m, ef_construct=self._hnsw_ef_construct),
        )

    def upsert(self, ids: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None:
        points = [
            qm.PointStruct(id=self._point_id(cid), vector=vec, payload={**meta, "chunk_id": cid})
            for cid, vec, meta in zip(ids, vectors, metadatas)
        ]
        self._client.upsert(self._collection, points=points)

    def query(
        self, vector: list[float], top_k: int = 10, filters: dict | None = None
    ) -> list[VectorMatch]:
        qfilter = _build_filter(filters) if filters else None
        result = self._client.query_points(
            self._collection, query=vector, limit=top_k, query_filter=qfilter
        )
        return [
            VectorMatch(id=hit.payload["chunk_id"], score=hit.score, metadata=hit.payload)
            for hit in result.points
        ]

    def delete(self, ids: list[str]) -> None:
        self._client.delete(
            self._collection, points_selector=[self._point_id(cid) for cid in ids]
        )
