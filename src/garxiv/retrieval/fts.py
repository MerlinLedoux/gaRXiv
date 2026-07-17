import re

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def sanitize_query(raw: str) -> str:
    """Turn an arbitrary user string into a safe FTS5 MATCH expression.

    Each alphanumeric token is individually double-quoted (an FTS5 string
    literal) and combined with OR, so any FTS5 keyword/operator the user
    types (AND, NOT, *, parentheses...) is never interpreted as query
    syntax. Tokens are OR-combined to maximize recall — fusion and
    reranking downstream handle precision.
    """
    tokens = _TOKEN_RE.findall(raw)
    if not tokens:
        return '""'
    return " OR ".join(f'"{t}"' for t in tokens)
