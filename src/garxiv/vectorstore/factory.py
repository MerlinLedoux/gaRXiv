import re

from garxiv.config import VectorStoreConfig
from garxiv.vectorstore.base import VectorStore


def collection_name(provider_name: str, dimension: int) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", provider_name.lower()).strip("_")
    return f"garxiv_{safe}_{dimension}d"


def get_store(config: VectorStoreConfig, collection_name: str) -> VectorStore:
    if config.backend == "qdrant":
        from garxiv.vectorstore.qdrant_store import QdrantVectorStore

        return QdrantVectorStore(
            url=config.url,
            collection_name=collection_name,
            distance=config.distance,
            hnsw_m=config.hnsw_m,
            hnsw_ef_construct=config.hnsw_ef_construct,
        )

    if config.backend == "azure_pgvector":
        from garxiv.vectorstore.azure_pgvector_store import AzurePgVectorStore

        return AzurePgVectorStore(connection_string_env=config.pg_connection_string_env)

    raise ValueError(f"unknown vector store backend: {config.backend}")
