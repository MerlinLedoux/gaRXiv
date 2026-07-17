import logging
import re

from garxiv.llm.base import LLMProvider, LLMUnavailableError

logger = logging.getLogger(__name__)

_NUMBERING_RE = re.compile(r"^\s*(?:[\d]+[.)]|[-*])\s*")

_MULTI_QUERY_PROMPT = """You are a search query rewriting assistant for an academic paper \
retrieval system. Given the user's search query, write {n} alternative phrasings that \
would help retrieve relevant passages from scientific papers on arXiv. Output exactly \
{n} lines, one phrasing per line, with no numbering, no bullets, and no extra commentary.

Query: {query}"""

_HYDE_PROMPT = """Write a short, plausible excerpt (2-4 sentences) from a scientific \
paper that would directly answer the following query. Use technical language as if it \
were extracted from a paper's abstract or body. Do not mention that this is \
hypothetical and do not add any preamble.

Query: {query}"""


def generate_multi_queries(llm: LLMProvider, query: str, n: int) -> list[str]:
    """Returns up to n rephrasings of `query`. Empty list (silent fallback
    handled by the caller) if the LLM is unreachable or returns nothing
    usable."""
    try:
        raw = llm.generate(_MULTI_QUERY_PROMPT.format(n=n, query=query))
    except LLMUnavailableError as exc:
        logger.warning("multi-query transform skipped: %s", exc)
        return []
    variants = [_NUMBERING_RE.sub("", line).strip() for line in raw.splitlines()]
    return [v for v in variants if v][:n]


def generate_hyde_document(llm: LLMProvider, query: str) -> str | None:
    """Returns a hypothetical document, or None if the LLM is unreachable."""
    try:
        text = llm.generate(_HYDE_PROMPT.format(query=query)).strip()
    except LLMUnavailableError as exc:
        logger.warning("HyDE transform skipped: %s", exc)
        return None
    return text or None
