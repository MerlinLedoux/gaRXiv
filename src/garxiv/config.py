from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class StorageConfig(BaseModel):
    db_path: Path
    pdf_dir: Path
    parsed_dir: Path


class ChunkingConfig(BaseModel):
    target_tokens: int = 800
    overlap_ratio: float = 0.125
    min_chunk_tokens: int = 100
    skip_sections: list[str] = Field(
        default_factory=lambda: ["references", "bibliography", "acknowledgments", "acknowledgements"]
    )

    @field_validator("overlap_ratio")
    @classmethod
    def _validate_overlap(cls, v: float) -> float:
        if not 0 <= v < 0.9:
            raise ValueError("overlap_ratio doit être dans [0, 0.9)")
        return v


class EmbeddingConfig(BaseModel):
    provider: Literal["local_bge", "azure_openai"] = "local_bge"
    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 32
    device: str | None = None
    azure_endpoint: str | None = None
    azure_deployment: str | None = None
    azure_api_key_env: str | None = None


class VectorStoreConfig(BaseModel):
    backend: Literal["qdrant", "azure_pgvector"] = "qdrant"
    url: str = "http://localhost:6333"
    distance: Literal["cosine", "dot", "euclid"] = "cosine"
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    pg_connection_string_env: str | None = None


class Config(BaseModel):
    categories: list[str]
    authors: list[str]
    storage: StorageConfig
    max_results_per_run: int
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)


def load_config(path: Path | str = "config.yaml") -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
