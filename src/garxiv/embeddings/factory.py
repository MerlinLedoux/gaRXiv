from garxiv.config import EmbeddingConfig
from garxiv.embeddings.base import EmbeddingProvider


def get_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    if config.provider == "local_bge":
        from garxiv.embeddings.local_bge import LocalBgeEmbeddingProvider

        return LocalBgeEmbeddingProvider(model_name=config.model_name, device=config.device)

    if config.provider == "azure_openai":
        from garxiv.embeddings.azure_openai import AzureOpenAIEmbeddingProvider

        return AzureOpenAIEmbeddingProvider(
            endpoint=config.azure_endpoint,
            deployment=config.azure_deployment,
            api_key_env=config.azure_api_key_env,
        )

    raise ValueError(f"unknown embedding provider: {config.provider}")
