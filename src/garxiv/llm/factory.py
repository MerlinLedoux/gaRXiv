from garxiv.config import LLMConfig
from garxiv.llm.base import LLMProvider


def get_provider(config: LLMConfig) -> LLMProvider:
    if config.provider == "ollama":
        from garxiv.llm.ollama_provider import OllamaLLMProvider

        return OllamaLLMProvider(
            model_name=config.model_name,
            base_url=config.base_url,
            timeout_seconds=config.timeout_seconds,
        )

    if config.provider == "azure_openai":
        from garxiv.llm.azure_openai_provider import AzureOpenAILLMProvider

        return AzureOpenAILLMProvider(
            endpoint=config.azure_endpoint,
            deployment=config.azure_deployment,
            api_key_env=config.azure_api_key_env,
        )

    raise ValueError(f"unknown LLM provider: {config.provider}")
