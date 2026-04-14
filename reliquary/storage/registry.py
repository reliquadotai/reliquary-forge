from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..protocol.artifacts import artifact_directory_name


class ArtifactRegistry(ABC):
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def put_artifact(self, artifact: dict[str, Any]) -> Path:
        raise NotImplementedError

    @abstractmethod
    def get_artifact(self, artifact_type: str, artifact_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def list_artifacts(
        self,
        artifact_type: str,
        window_id: int | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def latest_policy(self) -> dict[str, Any] | None:
        raise NotImplementedError

    def path_for(self, artifact_type: str, artifact_id: str) -> Path:
        folder = artifact_directory_name(artifact_type)
        return self.root / folder / f"{artifact_id}.json"

    def artifact_path(self, artifact: dict[str, Any]) -> Path:
        return self.path_for(artifact["artifact_type"], artifact["artifact_id"])

    def checkpoint_dir(self, checkpoint_id: str) -> Path:
        path = self.root / "checkpoints" / checkpoint_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def training_dir(self, window_id: int) -> Path:
        path = self.root / "training" / f"window_{window_id:04d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run_dir(self, run_id: str) -> Path:
        path = self.root / "runs" / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def pack_dir(self, pack_id: str) -> Path:
        path = self.root / "packs" / pack_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cache_dir(self) -> Path:
        path = self.root / "cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def export_artifacts(self, artifacts: list[dict[str, Any]], destination: Path) -> list[str]:
        copied: list[str] = []
        for artifact in artifacts:
            source = self.artifact_path(artifact)
            if not source.exists():
                continue
            target = destination / artifact_directory_name(artifact["artifact_type"]) / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(str(target))
        return copied
