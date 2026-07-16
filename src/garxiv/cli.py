import logging
import sys

import typer

from garxiv.config import load_config
from garxiv.ingestion.pipeline import run_ingestion

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


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    app()


if __name__ == "__main__":
    main()
