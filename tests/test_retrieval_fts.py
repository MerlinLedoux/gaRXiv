import sqlite3

import pytest

from garxiv.retrieval.fts import sanitize_query


def test_simple_query_is_or_combined_and_quoted():
    assert sanitize_query("neural network") == '"neural" OR "network"'


def test_empty_query_returns_matchless_expression():
    assert sanitize_query("   ") == '""'
    assert sanitize_query("") == '""'


@pytest.mark.parametrize(
    "raw",
    [
        "AND OR NOT",
        "foo AND bar",
        "foo* bar",
        '"quoted" (parens)',
        "foo NEAR/2 bar",
        "col:value",
        "-exclude me",
    ],
)
def test_sanitized_query_never_raises_fts5_syntax_error(raw):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(text)")
    conn.execute("INSERT INTO t (text) VALUES ('some sample document text')")

    query = sanitize_query(raw)
    # must not raise sqlite3.OperationalError regardless of `raw`'s content
    conn.execute("SELECT text FROM t WHERE t MATCH ?", (query,)).fetchall()
    conn.close()
