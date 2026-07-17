import json

import pytest
import requests
import responses

from garxiv.llm.base import LLMProvider, LLMUnavailableError
from garxiv.llm.ollama_provider import OllamaLLMProvider
from garxiv.retrieval.query_transform import generate_hyde_document, generate_multi_queries


class FakeLLMProvider(LLMProvider):
    name = "fake-llm"

    def __init__(self, response: str | None = None, fail: bool = False):
        self._response = response or ""
        self._fail = fail
        self.calls: list[str] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls.append(prompt)
        if self._fail:
            raise LLMUnavailableError("simulated failure")
        return self._response


def test_azure_openai_llm_provider_is_not_implemented():
    from garxiv.config import LLMConfig
    from garxiv.llm import factory

    config = LLMConfig(
        provider="azure_openai", azure_endpoint="https://x", azure_deployment="d", azure_api_key_env="X"
    )
    with pytest.raises(NotImplementedError):
        factory.get_provider(config)


@responses.activate
def test_ollama_provider_generate_success():
    responses.add(
        responses.POST,
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2:3b", "response": "  hello world  ", "done": True},
        status=200,
    )

    provider = OllamaLLMProvider(model_name="llama3.2:3b")
    result = provider.generate("say hello", system="be terse")

    assert result == "hello world"
    sent = json.loads(responses.calls[0].request.body)
    assert sent == {"model": "llama3.2:3b", "prompt": "say hello", "stream": False, "system": "be terse"}


@responses.activate
def test_ollama_provider_raises_llm_unavailable_on_connection_error():
    responses.add(
        responses.POST,
        "http://localhost:11434/api/generate",
        body=requests.exceptions.ConnectionError("refused"),
    )

    provider = OllamaLLMProvider(model_name="llama3.2:3b")
    with pytest.raises(LLMUnavailableError):
        provider.generate("say hello")


def test_generate_multi_queries_parses_lines_and_strips_numbering():
    llm = FakeLLMProvider(response="1. first variant\n2) second variant\n- third variant\n")

    variants = generate_multi_queries(llm, "original query", n=3)

    assert variants == ["first variant", "second variant", "third variant"]


def test_generate_multi_queries_truncates_to_n():
    llm = FakeLLMProvider(response="a\nb\nc\nd\n")

    assert generate_multi_queries(llm, "q", n=2) == ["a", "b"]


def test_generate_multi_queries_returns_empty_list_when_llm_unavailable():
    llm = FakeLLMProvider(fail=True)

    assert generate_multi_queries(llm, "q", n=3) == []


def test_generate_hyde_document_returns_text():
    llm = FakeLLMProvider(response="a hypothetical passage")

    assert generate_hyde_document(llm, "q") == "a hypothetical passage"


def test_generate_hyde_document_returns_none_when_llm_unavailable():
    llm = FakeLLMProvider(fail=True)

    assert generate_hyde_document(llm, "q") is None
