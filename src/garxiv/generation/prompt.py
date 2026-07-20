from garxiv.retrieval.pipeline import SearchResult

_SYSTEM_PROMPT = """You are a scientific research assistant helping a user understand recent \
arXiv papers. Answer ONLY using the numbered excerpts provided below the question — do not \
rely on prior knowledge about the topic. Every factual claim in your answer must be followed \
by a citation to the excerpt(s) it is drawn from, using square brackets like [1] or [2][3], \
matching the excerpt numbers given. If the excerpts do not contain enough information to \
answer the question, say so plainly instead of guessing. Reply in the same language as the \
user's question, regardless of the language of these instructions or of the excerpts."""

_USER_TEMPLATE = """Question: {question}

Excerpts:
{context_block}"""

_EXCERPT_TEMPLATE = "[{n}] (arXiv:{arxiv_id}, {section_title}, p.{page_start}-{page_end})\n{text}"


def build_prompt(
    question: str, sources: list[SearchResult], max_chars_per_chunk: int
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt). `sources` must already be the
    final, ordered, size-capped list the caller intends to send to the LLM
    (context_top_k slicing happens in the pipeline) — excerpts are numbered
    [1..len(sources)] in list order, and that same order/index is what
    AnswerResult.sources must preserve so citation [i] maps to sources[i-1]."""
    blocks = [
        _EXCERPT_TEMPLATE.format(
            n=i,
            arxiv_id=r.metadata.get("arxiv_id"),
            section_title=r.metadata.get("section_title"),
            page_start=r.metadata.get("page_start"),
            page_end=r.metadata.get("page_end"),
            text=r.text[:max_chars_per_chunk],
        )
        for i, r in enumerate(sources, start=1)
    ]
    user_prompt = _USER_TEMPLATE.format(question=question, context_block="\n\n".join(blocks))
    return _SYSTEM_PROMPT, user_prompt
