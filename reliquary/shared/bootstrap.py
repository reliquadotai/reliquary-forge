from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from ..storage import (
    LocalArtifactRegistry,
    PostgresMetadataIndex,
    R2ArtifactRegistry,
    SqliteMetadataIndex,
)
from ..storage.registry import ArtifactRegistry
from .config import load_runtime_config


def _build_metadata_index(cfg: dict[str, object], state_root: str):
    backend = str(cfg["metadata_index_backend"])
    dsn = str(cfg.get("metadata_index_dsn", "") or "")
    if backend == "postgres" and dsn:
        return PostgresMetadataIndex(dsn)
    return SqliteMetadataIndex(f"{state_root}/metadata/artifacts.sqlite3")


def _load_runtime_env() -> None:
    load_dotenv()
    explicit_env_file = os.getenv("RELIQUARY_ENV_FILE", "").strip()
    candidate_paths: list[Path] = []
    if explicit_env_file:
        candidate_paths.append(Path(explicit_env_file))
    candidate_paths.append(Path("/srv/reliquary/state/reliquary.env"))
    for env_path in candidate_paths:
        if env_path.exists():
            load_dotenv(env_path, override=False)


def bootstrap_runtime() -> tuple[dict, ArtifactRegistry]:
    _load_runtime_env()
    cfg = load_runtime_config()
    logging.basicConfig(
        level=getattr(logging, cfg["log_level"].upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    metadata_index = _build_metadata_index(cfg, cfg["state_root"])
    backend = cfg["artifact_registry_backend"]
    if backend == "r2":
        registry = R2ArtifactRegistry(
            cfg["artifact_dir"],
            bucket=cfg["r2_bucket"],
            prefix=cfg["r2_prefix"],
            endpoint_url=cfg["r2_endpoint_url"],
            access_key_id=cfg["r2_access_key_id"],
            secret_access_key=cfg["r2_secret_access_key"],
            region_name=cfg["r2_region"],
            metadata_index=metadata_index,
        )
    else:
        registry = LocalArtifactRegistry(cfg["artifact_dir"], metadata_index=metadata_index)
    return cfg, registry
