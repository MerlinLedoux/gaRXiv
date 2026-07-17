from abc import ABC, abstractmethod


class LLMUnavailableError(Exception):
    """Raised when the configured LLM backend is unreachable (Ollama not
    running, network timeout, etc.). Callers should catch this to degrade
    gracefully (skip query transformation) rather than crash the search."""


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Non-streaming completion, returns the generated text (stripped)."""
