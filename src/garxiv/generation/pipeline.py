from dataclasses import dataclass

from garxiv.config import Config
from garxiv.generation.prompt import build_prompt
from garxiv.llm import factory as llm_factory
from garxiv.retrieval.pipeline import SearchFilters, SearchResult, run_search

_NO_RESULTS_ANSWER = "Aucun document pertinent n'a été trouvé pour répondre à cette question."


@dataclass
class AnswerResult:
    answer: str
    sources: list[SearchResult]


def run_generation(
    config: Config,
    question: str,
    top_k: int | None = None,
    filters: SearchFilters | None = None,
) -> AnswerResult:
    """Generation pipeline: run_search -> (skip the LLM entirely if there are
    no hits) -> build a grounded, numbered-citation prompt from up to
    config.generation.context_top_k of the retrieved chunks -> LLM.generate.
    Read-only. Unlike query transformation, there is no fallback here if the
    LLM is unreachable: LLMUnavailableError propagates to the caller."""
    results = run_search(config, question, top_k=top_k, filters=filters)
    if not results:
        return AnswerResult(answer=_NO_RESULTS_ANSWER, sources=[])

    context_sources = results[: config.generation.context_top_k]
    system_prompt, user_prompt = build_prompt(
        question, context_sources, config.generation.max_chars_per_chunk
    )

    llm = llm_factory.get_provider(config.llm)
    answer = llm.generate(user_prompt, system=system_prompt)

    return AnswerResult(answer=answer, sources=context_sources)
