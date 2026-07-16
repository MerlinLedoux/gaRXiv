from pathlib import Path

import requests


def download_pdf(
    pdf_url: str,
    arxiv_id: str,
    version: int,
    pdf_dir: Path | str,
    session: requests.Session | None = None,
) -> Path:
    """Download a paper's PDF, skipping if it was already downloaded."""
    pdf_dir = Path(pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    dest = pdf_dir / f"{arxiv_id}v{version}.pdf"
    if dest.exists():
        return dest

    session = session or requests.Session()
    response = session.get(pdf_url, stream=True, timeout=60)
    response.raise_for_status()
    with dest.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest
