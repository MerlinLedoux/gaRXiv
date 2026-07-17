from garxiv.embeddings.base import EmbeddingProvider

_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


class AzureOpenAIEmbeddingProvider(EmbeddingProvider):
    """Not implemented yet — Azure isn't available in this environment.
    Interface is ready: swapping the dev local_bge provider for this one
    is a one-line config change (`embeddings.provider: azure_openai`) once
    an Azure OpenAI endpoint/deployment exists."""

    def __init__(
        self,
        endpoint: str,
        deployment: str,
        api_key_env: str,
        dimension: int | None = None,
    ):
        self.name = deployment
        self.dimension = dimension or _DIMENSIONS.get(deployment, 1536)
        raise NotImplementedError("Azure OpenAI embedding provider pas encore implémenté")

    def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        raise NotImplementedError("Azure OpenAI embedding provider pas encore implémenté")
