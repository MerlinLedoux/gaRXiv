from garxiv.llm.base import LLMProvider


class AzureOpenAILLMProvider(LLMProvider):
    """Not implemented yet — Azure OpenAI isn't available in this
    environment. Interface is ready: switching from ollama (dev) to this
    provider (prod) is a one-line config change (`llm.provider: azure_openai`)
    once an Azure OpenAI chat/completions deployment exists."""

    def __init__(self, endpoint: str, deployment: str, api_key_env: str):
        self.name = deployment
        raise NotImplementedError("Azure OpenAI LLM provider pas encore implémenté")

    def generate(self, prompt: str, system: str | None = None) -> str:
        raise NotImplementedError("Azure OpenAI LLM provider pas encore implémenté")
