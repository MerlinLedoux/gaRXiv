from pathlib import Path

import yaml
from pydantic import BaseModel


class StorageConfig(BaseModel):
    db_path: Path
    pdf_dir: Path
    parsed_dir: Path


class Config(BaseModel):
    categories: list[str]
    authors: list[str]
    storage: StorageConfig
    max_results_per_run: int


def load_config(path: Path | str = "config.yaml") -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
