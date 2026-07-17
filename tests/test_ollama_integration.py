import pytest
import requests

from garxiv.llm.ollama_provider import OllamaLLMProvider

OLLAMA_URL = "http://localhost:11434"


@pytest.fixture
def ollama_url():
    try:
        requests.get(OLLAMA_URL, timeout=2).raise_for_status()
    except requests.exceptions.RequestException as exc:
        pytest.skip(f"Ollama non joignable sur {OLLAMA_URL} (ollama serve requis): {exc}")
    return OLLAMA_URL


@pytest.mark.integration
def test_ollama_provider_generates_text(ollama_url):
    provider = OllamaLLMProvider(model_name="llama3.2:3b", base_url=ollama_url)
    result = provider.generate("Reply with exactly the word: pong")
    assert isinstance(result, str)
    assert len(result) > 0
