# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

gaRXiv is a personalized RAG (retrieval-augmented generation) system for scientific watch on arXiv: targeted retrieval by domain, author, and source, automated daily monitoring, and summaries of new publications.

Ingestion, chunking, and embedding/vector-storage are implemented: fetching arXiv metadata, downloading PDFs, parsing them into structured text, cleaning and splitting them into overlapping token chunks, and embedding those chunks into a vector store (Qdrant in dev, Azure Database for PostgreSQL + pgvector planned for prod). The summarization/RAG (retrieval + generation) layer is not built yet.

## Commands

- Install deps: `uv sync`
- Run ingestion: `uv run garxiv ingest [--limit N] [--dry-run] [--config-path config.yaml]`
- Run chunking: `uv run garxiv chunk [--limit N] [--config-path config.yaml]`
- Run embedding (needs a reachable vector store — see below): `uv run garxiv embed [--limit N] [--config-path config.yaml]`
- Create/verify the vector store collection: `uv run garxiv vectorstore-init [--config-path config.yaml]`
- Run tests (excludes `slow`/`integration`-marked tests by default): `uv run pytest`
- Run a single test: `uv run pytest tests/test_pipeline.py::test_second_run_skips_already_processed_documents`
- Run the slow tests (loads the real BGE model) or integration tests (needs Qdrant): `uv run pytest -m slow`, `uv run pytest -m integration`
- Start the dev vector store: `docker-compose up -d` (requires Docker Desktop running)

## Architecture

Entry point is `src/garxiv/cli.py` (Typer), which loads `config.yaml` (categories, followed authors, storage paths, chunking/embedding/vector-store settings — validated via `src/garxiv/config.py`, a pydantic model) and dispatches to the pipeline functions below.

Every stage is idempotent and driven by a `status` column, either on `documents` or on `chunks`, both in SQLite (`src/garxiv/db.py`):

1. **Fetch** (`ingestion/arxiv_client.py`) — queries the arXiv Atom API for the configured categories/authors, filtered by a per-query watermark date (`ingestion_runs` table) so repeated runs only pull newly submitted papers instead of rescanning the archive. Results are upserted via `db.upsert_discovered`, which is a no-op for documents already known at their current version, and only resets an existing document back to `discovered` status (and deletes any of its existing chunks) when a newer version appears.
2. **Download** (`ingestion/downloader.py`) — processes only rows with `status = 'discovered'`, skips if the PDF file already exists on disk, then marks `status = 'downloaded'`.
3. **Parse** (`ingestion/parser.py`) — processes only rows with `status = 'downloaded'`, uses PyMuPDF to extract text and a heuristic section breakdown (headings detected via font size/boldness relative to the document's median; each line also keeps its page number), writes a JSON file, then marks `status = 'parsed'`.
4. **Chunk** (`chunking/pipeline.py`, via `run_chunking`) — processes only rows with `status = 'parsed'`: cleans each document (`chunking/cleaner.py` — strips arXiv watermark lines, isolated page numbers, recurring headers/footers, merges hyphen-split words), then splits it into overlapping token windows per section (`chunking/chunker.py`, tiktoken `cl100k_base`, configurable target tokens/overlap/min chunk size, skips sections like References by title). Chunks are stored in the `chunks` table (FK to `documents.arxiv_id`) with `section_title`, `page_start`/`page_end`, and `token_count`; the parent document moves to `status = 'chunked'`.
5. **Embed** (`embeddings/pipeline.py`, via `run_embedding`) — processes only chunks with `status = 'pending_embedding'`, in batches: embeds via the configured `EmbeddingProvider` (`embeddings/base.py`; `local_bge` using `sentence-transformers`/BAAI bge-small in dev, `azure_openai` stubbed for prod), upserts into the configured `VectorStore` (`vectorstore/base.py`; `qdrant` in dev via `vectorstore/qdrant_store.py`, `azure_pgvector` stubbed for prod), then marks the chunk `status = 'embedded'`. The vector store collection name encodes the embedding provider and dimension (`vectorstore/factory.py:collection_name`), so switching models can never silently mix incompatible vectors — it just requires a full re-embed. Qdrant point ids are internal (deterministic UUIDs derived from `chunk_id`); every public interface (`VectorStore`, `VectorMatch`) always speaks in terms of `chunk_id`.

This status-per-stage design is what makes the pipeline resumable after a crash and avoids redundant work (e.g. re-embedding an already-processed chunk).

Each ingestion stage function takes an injectable `requests.Session`, which is how tests (`tests/test_arxiv_client.py`, `tests/test_pipeline.py`) mock the arXiv API and PDF downloads via the `responses` library instead of hitting the network. `tests/test_parser.py` generates synthetic PDFs on the fly with PyMuPDF rather than committing binary fixtures. Embedding/vector-store tests use fakes (`FakeEmbeddingProvider`, `FakeVectorStore`) or Qdrant's embedded `:memory:` mode to avoid touching the real model or requiring Docker; only `-m slow` and `-m integration` tests do.

Local data (SQLite DB, downloaded PDFs, parsed JSON) lives under `data/`, which is gitignored. The dev vector store (Qdrant) runs via `docker-compose.yml` and persists to a named Docker volume.
