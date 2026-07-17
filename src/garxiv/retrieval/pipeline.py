import logging
import sqlite3
from dataclasses import dataclass

from garxiv import db
from garxiv.config import Config
from garxiv.embeddings import factory as embedding_factory
from garxiv.llm import factory as llm_factory
from garxiv.rerank import factory as rerank_factory
from garxiv.retrieval import fusion, query_transform
from garxiv.retrieval.fts import sanitize_query
from garxiv.vectorstore import factory as vectorstore_factory

logger = logging.getLogger(__name__)


@dataclass
class SearchFilters:
    categories: list[str] | None = None
    authors: list[str] | None = None
    date_from: str | None = None
    date_to: str | None = None


@dataclass
class SearchResult:
    chunk_id: str
    score: float
    text: str
    metadata: dict


def _qdrant_filters(filters: SearchFilters) -> dict:
    out: dict = {}
    if filters.categories:
        out["categories"] = {"any": filters.categories}
    if filters.authors:
        out["authors"] = {"any": filters.authors}
    if filters.date_from or filters.date_to:
        rng: dict = {}
        if filters.date_from:
            rng["gte"] = filters.date_from
        if filters.date_to:
            rng["lte"] = filters.date_to
        out["published_date"] = rng
    return out


def run_search(
    config: Config,
    query: str,
    top_k: int | None = None,
    filters: SearchFilters | None = None,
) -> list[SearchResult]:
    """Search pipeline: (optional query transformation) -> (dense + lexical
    per variant) -> (RRF fusion) -> (optional reranking). Each stage is
    independently toggled via config.search. Read-only, never modifies the
    database."""
    cfg = config.search
    top_k = top_k or cfg.top_k
    filters = filters or SearchFilters()
    conn = db.get_connection(config.storage.db_path)

    query_variants = [query]
    hyde_texts: list[str] = []
    if cfg.query_transform.mode != "none":
        llm = llm_factory.get_provider(config.llm)
        if cfg.query_transform.mode == "multi_query":
            query_variants += query_transform.generate_multi_queries(
                llm, query, cfg.query_transform.multi_query_count
            )
        elif cfg.query_transform.mode == "hyde":
            hyde_doc = query_transform.generate_hyde_document(llm, query)
            if hyde_doc:
                hyde_texts.append(hyde_doc)

    provider = embedding_factory.get_provider(config.embeddings)
    collection = vectorstore_factory.collection_name(provider.name, provider.dimension)
    store = vectorstore_factory.get_store(config.vector_store, collection_name=collection)
    qfilters = _qdrant_filters(filters)

    dense_rankings: list[list[str]] = []
    chunk_meta: dict[str, dict] = {}

    def _run_dense(texts: list[str], is_query: bool) -> None:
        if not texts:
            return
        vectors = provider.embed(texts, is_query=is_query)
        for vec in vectors:
            matches = store.query(vec, top_k=cfg.dense_top_k, filters=qfilters or None)
            dense_rankings.append([m.id for m in matches])
            for m in matches:
                chunk_meta[m.id] = m.metadata

    _run_dense(query_variants, is_query=True)
    _run_dense(hyde_texts, is_query=False)

    lexical_rankings: list[list[str]] = []
    if cfg.hybrid_enabled:
        for variant in query_variants:
            rows = db.search_fts(
                conn,
                sanitize_query(variant),
                limit=cfg.lexical_top_k,
                categories=filters.categories,
                authors=filters.authors,
                date_from=filters.date_from,
                date_to=filters.date_to,
            )
            lexical_rankings.append([row["chunk_id"] for row in rows])

    fused = fusion.reciprocal_rank_fusion(dense_rankings + lexical_rankings, k=cfg.rrf_k)
    if not fused:
        conn.close()
        return []

    pool_size = max(top_k, cfg.rerank.top_k_candidates) if cfg.rerank.enabled else top_k
    candidate_ids = [cid for cid, _ in fused[:pool_size]]
    rows_by_id = {row["chunk_id"]: row for row in db.get_chunks_by_ids(conn, candidate_ids)}
    fused_by_id = dict(fused)

    if cfg.rerank.enabled:
        reranker = rerank_factory.get_reranker(cfg.rerank)
        ids = [cid for cid in candidate_ids if cid in rows_by_id]
        scores = reranker.score(query, [rows_by_id[cid]["text"] for cid in ids])
        ranked = sorted(zip(ids, scores), key=lambda t: t[1], reverse=True)[:top_k]
    else:
        ranked = [(cid, fused_by_id[cid]) for cid in candidate_ids if cid in rows_by_id][:top_k]

    results = [
        SearchResult(
            chunk_id=cid,
            score=float(score),
            text=rows_by_id[cid]["text"],
            metadata={
                "arxiv_id": rows_by_id[cid]["arxiv_id"],
                "section_title": rows_by_id[cid]["section_title"],
                "chunk_index": rows_by_id[cid]["chunk_index"],
                "page_start": rows_by_id[cid]["page_start"],
                "page_end": rows_by_id[cid]["page_end"],
                **chunk_meta.get(cid, {}),
            },
        )
        for cid, score in ranked
    ]
    conn.close()
    return results
