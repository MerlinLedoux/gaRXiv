import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import datetime, timezone

import requests

from garxiv.db import ArxivEntry

ATOM_NS = "{http://www.w3.org/2005/Atom}"
API_URL = "http://export.arxiv.org/api/query"
RATE_LIMIT_SECONDS = 3.0
PAGE_SIZE = 100


def build_search_query(categories: list[str], authors: list[str]) -> str:
    """Papers matching a watched category OR by a followed author."""
    parts = []
    if categories:
        parts.append("(" + " OR ".join(f"cat:{c}" for c in categories) + ")")
    if authors:
        parts.append("(" + " OR ".join(f'au:"{a}"' for a in authors) + ")")
    if not parts:
        raise ValueError("At least one category or author must be configured")
    return parts[0] if len(parts) == 1 else " OR ".join(parts)


def to_arxiv_datetime(iso_str: str) -> str:
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return dt.strftime("%Y%m%d%H%M")


def _parse_entry(entry: ET.Element) -> ArxivEntry:
    raw_id = entry.find(f"{ATOM_NS}id").text.strip()
    ident = raw_id.rsplit("/", 1)[-1]
    if "v" in ident:
        base_id, version_str = ident.rsplit("v", 1)
        version = int(version_str)
    else:
        base_id, version = ident, 1

    title = entry.find(f"{ATOM_NS}title").text.strip().replace("\n", " ")
    abstract = entry.find(f"{ATOM_NS}summary").text.strip()
    published = entry.find(f"{ATOM_NS}published").text.strip()
    updated = entry.find(f"{ATOM_NS}updated").text.strip()

    authors = [
        a.find(f"{ATOM_NS}name").text.strip()
        for a in entry.findall(f"{ATOM_NS}author")
    ]
    categories = [c.attrib["term"] for c in entry.findall(f"{ATOM_NS}category")]

    pdf_url = None
    for link in entry.findall(f"{ATOM_NS}link"):
        if link.attrib.get("title") == "pdf":
            pdf_url = link.attrib["href"]
            break
    if pdf_url is None:
        pdf_url = f"https://arxiv.org/pdf/{base_id}v{version}"

    return ArxivEntry(
        arxiv_id=base_id,
        version=version,
        title=title,
        abstract=abstract,
        authors=authors,
        categories=categories,
        published_date=published,
        updated_date=updated,
        pdf_url=pdf_url,
    )


def fetch_entries(
    categories: list[str],
    authors: list[str],
    since: str | None = None,
    max_results: int | None = None,
    session: requests.Session | None = None,
) -> Iterator[ArxivEntry]:
    """Yield ArxivEntry objects for the configured categories/authors.

    If `since` (an ISO datetime, e.g. a stored watermark) is given, only
    entries submitted after that date are requested, so repeated runs don't
    re-scan the full history.
    """
    session = session or requests.Session()
    search_query = build_search_query(categories, authors)
    if since:
        start = to_arxiv_datetime(since)
        end = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
        search_query = f"({search_query}) AND submittedDate:[{start} TO {end}]"

    start_idx = 0
    fetched = 0
    while True:
        page_size = PAGE_SIZE
        if max_results is not None:
            page_size = min(page_size, max_results - fetched)
            if page_size <= 0:
                break

        params = {
            "search_query": search_query,
            "start": start_idx,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }
        response = session.get(API_URL, params=params, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        entries = root.findall(f"{ATOM_NS}entry")
        if not entries:
            break

        for entry in entries:
            yield _parse_entry(entry)
            fetched += 1

        start_idx += len(entries)
        if max_results is not None and fetched >= max_results:
            break
        if len(entries) < page_size:
            break
        time.sleep(RATE_LIMIT_SECONDS)
