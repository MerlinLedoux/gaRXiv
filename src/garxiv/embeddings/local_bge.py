from garxiv.embeddings.base import EmbeddingProvider

_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_DIMENSIONS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}


class LocalBgeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None):
        from sentence_transformers import SentenceTransformer  # lazy import: heavy dependency

        self.name = model_name
        self.dimension = _DIMENSIONS.get(model_name, 384)
        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        if is_query:
            texts = [_QUERY_PREFIX + t for t in texts]
        vectors = self._model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        return vectors.tolist()
