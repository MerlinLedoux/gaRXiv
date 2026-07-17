import requests

from garxiv.llm.base import LLMProvider, LLMUnavailableError


class OllamaLLMProvider(LLMProvider):
    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 30.0,
    ):
        self.name = model_name
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    def generate(self, prompt: str, system: str | None = None) -> str:
        payload: dict = {"model": self.name, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        try:
            resp = requests.post(
                f"{self._base_url}/api/generate", json=payload, timeout=self._timeout
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise LLMUnavailableError(f"Ollama injoignable sur {self._base_url}: {exc}") from exc
        return resp.json()["response"].strip()
