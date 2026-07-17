from garxiv.vectorstore.base import VectorMatch, VectorStore


class AzurePgVectorStore(VectorStore):
    """Not implemented yet — Azure Database for PostgreSQL isn't available
    in this environment. Target design: pgvector column + HNSW index on the
    prod Postgres instance, SQL WHERE clauses for metadata filtering instead
    of Qdrant's filter DSL. Interface is ready: swapping the dev Qdrant store
    for this one is a one-line config change (`vector_store.backend: azure_pgvector`)."""

    def __init__(self, connection_string_env: str, table_name: str = "chunk_embeddings"):
        raise NotImplementedError("Azure pgvector store pas encore implémenté")

    def ensure_collection(self, dimension: int) -> None:
        raise NotImplementedError

    def upsert(self, ids: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None:
        raise NotImplementedError

    def query(
        self, vector: list[float], top_k: int = 10, filters: dict | None = None
    ) -> list[VectorMatch]:
        raise NotImplementedError

    def delete(self, ids: list[str]) -> None:
        raise NotImplementedError
