from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Interchangeable embedding backend. `name` and `dimension` are used to
    name the vector store collection so different models never mix vectors."""

    name: str
    dimension: int

    @abstractmethod
    def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """Embed a batch of texts, same order in/out.

        `is_query=True` lets asymmetric models (e.g. bge) add their query
        instruction prefix; ingestion always calls with is_query=False.
        """
