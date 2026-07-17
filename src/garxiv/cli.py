import logging
import sys

import typer

from garxiv.chunking.pipeline import run_chunking
from garxiv.config import load_config
from garxiv.embeddings.pipeline import run_embedding
from garxiv.ingestion.pipeline import run_ingestion
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


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    app()


if __name__ == "__main__":
    main()
