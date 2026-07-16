import logging
from dataclasses import dataclass

import requests

from garxiv import db
from garxiv.config import Config
from garxiv.ingestion import arxiv_client, downloader, parser

logger = logging.getLogger(__name__)


@dataclass
class IngestionSummary:
    discovered: int = 0
    skipped: int = 0
    downloaded: int = 0
    parsed: int = 0
    errors: int = 0


def _query_key(config: Config) -> str:
    return "|".join(sorted(config.categories)) + "::" + "|".join(sorted(config.authors))


def run_ingestion(
    config: Config, limit: int | None = None, dry_run: bool = False
) -> IngestionSummary:
    """Run one ingestion cycle: fetch -> upsert -> download -> parse.

    Idempotent: documents already known at their current version are
    skipped, and download/parse only ever act on documents that haven't
    reached that stage yet (tracked via `documents.status`).
    """
    conn = db.get_connection(config.storage.db_path)
    summary = IngestionSummary()
    session = requests.Session()

    query_key = _query_key(config)
    since = db.get_watermark(conn, query_key)
    latest_seen = since

    for entry in arxiv_client.fetch_entries(
        categories=config.categories,
        authors=config.authors,
        since=since,
        max_results=limit or config.max_results_per_run,
        session=session,
    ):
        summary.discovered += 1
        if latest_seen is None or entry.updated_date > latest_seen:
            latest_seen = entry.updated_date

        if dry_run:
            continue

        changed = db.upsert_discovered(conn, entry)
        if not changed:
            summary.skipped += 1

    if dry_run:
        conn.close()
        return summary

    if latest_seen:
        db.set_watermark(conn, query_key, latest_seen)

    for row in db.get_by_status(conn, "discovered"):
        try:
            pdf_path = downloader.download_pdf(
                row["pdf_url"],
                row["arxiv_id"],
                row["version"],
                config.storage.pdf_dir,
                session=session,
            )
            db.mark_downloaded(conn, row["arxiv_id"], str(pdf_path))
            summary.downloaded += 1
        except Exception as exc:
            logger.warning("download failed for %s: %s", row["arxiv_id"], exc)
            db.mark_error(conn, row["arxiv_id"], str(exc))
            summary.errors += 1

    for row in db.get_by_status(conn, "downloaded"):
        try:
            parsed_path = parser.parse_pdf(
                row["pdf_path"], row["arxiv_id"], config.storage.parsed_dir
            )
            db.mark_parsed(conn, row["arxiv_id"], str(parsed_path))
            summary.parsed += 1
        except Exception as exc:
            logger.warning("parse failed for %s: %s", row["arxiv_id"], exc)
            db.mark_error(conn, row["arxiv_id"], str(exc))
            summary.errors += 1

    conn.close()
    return summary
