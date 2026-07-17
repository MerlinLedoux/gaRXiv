from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VectorMatch:
    id: str  # == chunk_id, never the vector store's internal point id
    score: float
    metadata: dict


class VectorStore(ABC):
    @abstractmethod
    def ensure_collection(self, dimension: int) -> None: ...

    @abstractmethod
    def upsert(self, ids: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None: ...

    @abstractmethod
    def query(
        self, vector: list[float], top_k: int = 10, filters: dict | None = None
    ) -> list[VectorMatch]: ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None: ...
