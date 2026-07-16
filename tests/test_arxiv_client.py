from pathlib import Path

import responses

from garxiv.ingestion.arxiv_client import (
    API_URL,
    build_search_query,
    fetch_entries,
    to_arxiv_datetime,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_build_search_query_categories_only():
    assert build_search_query(["cs.CL", "cs.LG"], []) == "(cat:cs.CL OR cat:cs.LG)"


def test_build_search_query_authors_only():
    assert build_search_query([], ["Ada Lovelace"]) == '(au:"Ada Lovelace")'


def test_build_search_query_combines_with_or():
    query = build_search_query(["cs.CL"], ["Ada Lovelace"])
    assert query == '(cat:cs.CL) OR (au:"Ada Lovelace")'


def test_build_search_query_requires_at_least_one_filter():
    try:
        build_search_query([], [])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_to_arxiv_datetime():
    assert to_arxiv_datetime("2024-01-15T10:30:00Z") == "202401151030"


@responses.activate
def test_fetch_entries_parses_feed():
    feed_xml = (FIXTURES / "sample_feed.xml").read_bytes()
    responses.add(responses.GET, API_URL, body=feed_xml, status=200)

    entries = list(
        fetch_entries(categories=["cs.CL"], authors=[], max_results=10)
    )

    assert len(entries) == 2

    first, second = entries
    assert first.arxiv_id == "2401.00001"
    assert first.version == 1
    assert first.title == "A Sample Paper About Testing"
    assert first.authors == ["Ada Lovelace", "Alan Turing"]
    assert first.categories == ["cs.CL"]
    assert first.pdf_url == "http://arxiv.org/pdf/2401.00001v1"

    assert second.arxiv_id == "2401.00002"
    assert second.version == 2


@responses.activate
def test_fetch_entries_stops_after_short_page():
    feed_xml = (FIXTURES / "sample_feed.xml").read_bytes()
    responses.add(responses.GET, API_URL, body=feed_xml, status=200)

    entries = list(fetch_entries(categories=["cs.CL"], authors=[], max_results=100))

    assert len(entries) == 2
    assert len(responses.calls) == 1
