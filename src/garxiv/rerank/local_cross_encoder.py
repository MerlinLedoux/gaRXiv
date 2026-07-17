from garxiv.rerank.base import Reranker


class LocalCrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str | None = None):
        from sentence_transformers import CrossEncoder  # lazy import: heavy dependency

        self.name = model_name
        self._model = CrossEncoder(model_name, device=device)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        return self._model.predict(pairs, show_progress_bar=False).tolist()
