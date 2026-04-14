from __future__ import annotations

from pathlib import Path
from typing import Any

from ..protocol.artifacts import artifact_directory_name
from ..utils.json_io import read_json, write_json
from .metadata_index import MetadataIndex, SqliteMetadataIndex
from .registry import ArtifactRegistry


class LocalArtifactRegistry(ArtifactRegistry):
    def __init__(self, root: str, *, metadata_index: MetadataIndex | None = None):
        super().__init__(root)
        self.metadata_index = metadata_index or SqliteMetadataIndex(self.root / "metadata" / "artifacts.sqlite3")
        for subdir in [
            "policies",
            "challenges",
            "task_batches",
            "rollouts",
            "distill_traces",
            "verdicts",
            "checkpoints",
            "training",
            "packs",
            "distill_packs",
            "training_batches",
            "scorecards",
            "policy_authority_decisions",
            "benchmark_results",
            "window_manifests",
            "run_manifests",
            "runs",
            "cache",
            "metadata",
        ]:
            (self.root / subdir).mkdir(parents=True, exist_ok=True)

    def put_artifact(self, artifact: dict[str, Any]) -> Path:
        path = self.artifact_path(artifact)
        write_json(path, artifact)
        self.metadata_index.record_artifact(
            artifact,
            local_path=str(path),
            storage_kind="local",
            storage_uri=str(path),
        )
        return path

    def get_artifact(self, artifact_type: str, artifact_id: str) -> dict[str, Any] | None:
        record = self.metadata_index.get_record(artifact_type, artifact_id)
        if record is None:
            path = self.path_for(artifact_type, artifact_id)
            return read_json(path) if path.exists() else None
        local_path = record.get("local_path")
        if local_path and Path(local_path).exists():
            return read_json(Path(local_path))
        path = self.path_for(artifact_type, artifact_id)
        return read_json(path) if path.exists() else None

    def list_artifacts(self, artifact_type: str, window_id: int | None = None) -> list[dict[str, Any]]:
        records = self.metadata_index.list_records(artifact_type, window_id=window_id)
        if records:
            results: list[dict[str, Any]] = []
            for record in records:
                local_path = record.get("local_path")
                if local_path and Path(local_path).exists():
                    results.append(read_json(Path(local_path)))
            if results:
                return results

        folder = self.root / artifact_directory_name(artifact_type)
        fallback: list[dict[str, Any]] = []
        for path in sorted(folder.glob("*.json")):
            artifact = read_json(path)
            if window_id is None or artifact.get("window_id") == window_id:
                fallback.append(artifact)
        return fallback

    def latest_policy(self) -> dict[str, Any] | None:
        record = self.metadata_index.latest_policy_record()
        if record is not None:
            local_path = record.get("local_path")
            if local_path and Path(local_path).exists():
                return read_json(Path(local_path))
        policies = self.list_artifacts("policy")
        if not policies:
            return None
        return sorted(policies, key=lambda item: (item["window_id"], item["created_at"]))[-1]
