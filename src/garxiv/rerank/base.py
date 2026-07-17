from abc import ABC, abstractmethod


class Reranker(ABC):
    name: str

    @abstractmethod
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Returns a relevance score per text, same order in/out. Higher is
        more relevant (no guaranteed bounded range)."""
