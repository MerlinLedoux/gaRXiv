# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

gaRXiv is a personalized RAG (retrieval-augmented generation) system for scientific watch on arXiv: targeted retrieval by domain, author, and source, automated daily monitoring, and summaries of new publications.

Only the first stage, **ingestion**, is implemented so far: fetching arXiv metadata, downloading PDFs, and parsing them into structured text. Embedding, vector storage/retrieval, and the summarization/RAG layer are not built yet.

## Commands

- Install deps: `uv sync`
- Run ingestion: `uv run garxiv ingest [--limit N] [--dry-run] [--config-path config.yaml]`
- Run tests: `uv run pytest`
- Run a single test: `uv run pytest tests/test_pipeline.py::test_second_run_skips_already_processed_documents`

## Architecture

Entry point is `src/garxiv/cli.py` (Typer), which loads `config.yaml` (categories, followed authors, storage paths — validated via `src/garxiv/config.py`, a pydantic model) and calls `run_ingestion` in `src/garxiv/ingestion/pipeline.py`.

The pipeline runs in three idempotent stages, each driven by a `status` column on the `documents` table in SQLite (`src/garxiv/db.py`):

1. **Fetch** (`ingestion/arxiv_client.py`) — queries the arXiv Atom API for the configured categories/authors, filtered by a per-query watermark date (`ingestion_runs` table) so repeated runs only pull newly submitted papers instead of rescanning the archive. Results are upserted via `db.upsert_discovered`, which is a no-op for documents already known at their current version, and only resets an existing document back to `discovered` status when a newer version appears (triggering re-download/re-parse of that document specifically).
2. **Download** (`ingestion/downloader.py`) — processes only rows with `status = 'discovered'`, skips if the PDF file already exists on disk, then marks `status = 'downloaded'`.
3. **Parse** (`ingestion/parser.py`) — processes only rows with `status = 'downloaded'`, uses PyMuPDF to extract text and a heuristic section breakdown (headings detected via font size/boldness relative to the document's median), writes a JSON file, then marks `status = 'parsed'`.

This status-per-stage design is what makes the pipeline resumable after a crash and avoids redundant work (e.g. re-embedding an already-processed document) in future stages — a document's `status` will eventually progress to `embedded` once that stage is built.

Each stage function takes an injectable `requests.Session`, which is how tests (`tests/test_arxiv_client.py`, `tests/test_pipeline.py`) mock the arXiv API and PDF downloads via the `responses` library instead of hitting the network. `tests/test_parser.py` generates synthetic PDFs on the fly with PyMuPDF rather than committing binary fixtures.

Local data (SQLite DB, downloaded PDFs, parsed JSON) lives under `data/`, which is gitignored.
