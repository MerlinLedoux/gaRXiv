import logging
import sys

import typer

from garxiv import db as db_module
from garxiv.chunking.pipeline import run_chunking
from garxiv.config import load_config
from garxiv.embeddings.pipeline import run_embedding
from garxiv.ingestion.pipeline import run_ingestion
from garxiv.retrieval.pipeline import SearchFilters, run_search
from garxiv.vectorstore import factory as vectorstore_factory
from garxiv.embeddings import factory as embedding_factory

app = typer.Typer()


@app.callback()
def callback() -> None:
    """gaRXiv - veille scientifique arXiv."""


@app.command()
def ingest(
    limit: int = typer.Option(None, help="Nombre max de nouveaux documents à récupérer"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Interroge l'API sans rien écrire (DB/disque)"
    ),
    config_path: str = typer.Option("config.yaml", help="Chemin du fichier de config"),
):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(config_path)
    summary = run_ingestion(config, limit=limit, dry_run=dry_run)
    typer.echo(
        f"{summary.discovered} découverts, {summary.skipped} ignorés (déjà traités), "
        f"{summary.downloaded} téléchargés, {summary.parsed} parsés, "
        f"{summary.errors} erreurs"
    )


@app.command()
def chunk(
    limit: int = typer.Option(None, help="Nombre max de documents à découper"),
    config_path: str = typer.Option("config.yaml", help="Chemin du fichier de config"),
):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(config_path)
    summary = run_chunking(config, limit=limit)
    typer.echo(
        f"{summary.documents_chunked} documents découpés, "
        f"{summary.chunks_created} chunks créés, {summary.errors} erreurs"
    )


@app.command()
def embed(
    limit: int = typer.Option(None, help="Nombre max de chunks à embedder"),
    config_path: str = typer.Option("config.yaml", help="Chemin du fichier de config"),
):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(config_path)
    summary = run_embedding(config, limit=limit)
    typer.echo(f"{summary.embedded} chunks embeddés, {summary.errors} erreurs")


@app.command("vectorstore-init")
def vectorstore_init(
    config_path: str = typer.Option("config.yaml", help="Chemin du fichier de config"),
):
    """Crée la collection du vector store si elle n'existe pas déjà (idempotent)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(config_path)
    provider = embedding_factory.get_provider(config.embeddings)
    name = vectorstore_factory.collection_name(provider.name, provider.dimension)
    store = vectorstore_factory.get_store(config.vector_store, collection_name=name)
    store.ensure_collection(provider.dimension)
    typer.echo(f"collection '{name}' prête (dimension={provider.dimension})")


@app.command()
def search(
    query: str = typer.Argument(..., help="Requête de recherche en langage naturel"),
    top_k: int = typer.Option(10, "--top-k", help="Nombre de résultats à retourner"),
    category: list[str] = typer.Option(None, "--category", help="Filtre catégorie arXiv (répétable)"),
    author: list[str] = typer.Option(None, "--author", help="Filtre auteur (répétable)"),
    date_from: str = typer.Option(None, "--date-from", help="Date de publication min. (YYYY-MM-DD)"),
    date_to: str = typer.Option(None, "--date-to", help="Date de publication max. (YYYY-MM-DD)"),
    hybrid: bool = typer.Option(
        None, "--hybrid/--no-hybrid", help="Active/désactive la recherche lexicale FTS5 (sinon config.yaml)"
    ),
    rerank: bool = typer.Option(
        None, "--rerank/--no-rerank", help="Active/désactive le reranking (sinon config.yaml)"
    ),
    query_transform: str = typer.Option(
        None, "--query-transform", help="none|multi_query|hyde (sinon config.yaml)"
    ),
    config_path: str = typer.Option("config.yaml", help="Chemin du fichier de config"),
):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(config_path)

    if hybrid is not None:
        config.search.hybrid_enabled = hybrid
    if rerank is not None:
        config.search.rerank.enabled = rerank
    if query_transform is not None:
        config.search.query_transform.mode = query_transform

    filters = SearchFilters(
        categories=category or None,
        authors=author or None,
        date_from=date_from,
        date_to=f"{date_to}T23:59:59Z" if date_to else None,
    )

    results = run_search(config, query, top_k=top_k, filters=filters)
    if not results:
        typer.echo("Aucun résultat.")
        raise typer.Exit()

    for i, r in enumerate(results, start=1):
        m = r.metadata
        snippet = r.text[:220].replace("\n", " ")
        typer.echo(
            f"{i}. [{r.score:.3f}] {m['arxiv_id']} — {m['section_title']} "
            f"(p.{m.get('page_start')}-{m.get('page_end')})"
        )
        typer.echo(f"   {snippet}...")


@app.command("fts-reindex")
def fts_reindex(
    config_path: str = typer.Option("config.yaml", help="Chemin du fichier de config"),
):
    """Réindexe dans chunks_fts les chunks absents de l'index (idempotent)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(config_path)
    conn = db_module.get_connection(config.storage.db_path)
    n = db_module.backfill_fts(conn)
    conn.close()
    typer.echo(f"{n} chunks réindexés dans chunks_fts")


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    app()


if __name__ == "__main__":
    main()
