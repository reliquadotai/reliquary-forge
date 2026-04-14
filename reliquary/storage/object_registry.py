from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.json_io import read_json, write_json
from .metadata_index import MetadataIndex
from .registry import ArtifactRegistry


class R2ArtifactRegistry(ArtifactRegistry):
    def __init__(
        self,
        root: str,
        *,
        bucket: str,
        prefix: str,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        region_name: str,
        metadata_index: MetadataIndex,
    ):
        super().__init__(root)
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.metadata_index = metadata_index
        import boto3

        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url or None,
            aws_access_key_id=access_key_id or None,
            aws_secret_access_key=secret_access_key or None,
            region_name=region_name or None,
        )

    def _object_key(self, artifact: dict[str, Any]) -> str:
        return (
            f"{self.prefix}/artifacts/{artifact['artifact_type']}/"
            f"{artifact['payload_hash']}-{artifact['created_at'].replace(':', '-')}.json"
        )

    def put_artifact(self, artifact: dict[str, Any]) -> Path:
        cache_path = self.artifact_path(artifact)
        write_json(cache_path, artifact)
        object_key = self._object_key(artifact)
        self.client.put_object(
            Bucket=self.bucket,
            Key=object_key,
            Body=json.dumps(artifact, indent=2, sort_keys=True).encode("utf-8"),
            ContentType="application/json",
        )
        storage_uri = f"r2://{self.bucket}/{object_key}"
        self.metadata_index.record_artifact(
            artifact,
            local_path=str(cache_path),
            storage_kind="r2",
            storage_uri=storage_uri,
        )
        return cache_path

    def get_artifact(self, artifact_type: str, artifact_id: str) -> dict[str, Any] | None:
        record = self.metadata_index.get_record(artifact_type, artifact_id)
        if record is None:
            return None
        local_path = record.get("local_path")
        if local_path and Path(local_path).exists():
            return read_json(Path(local_path))
        storage_uri = record.get("storage_uri")
        if not storage_uri:
            return None
        bucket, key = storage_uri.replace("r2://", "", 1).split("/", 1)
        response = self.client.get_object(Bucket=bucket, Key=key)
        artifact = json.loads(response["Body"].read().decode("utf-8"))
        cache_path = self.path_for(artifact_type, artifact_id)
        write_json(cache_path, artifact)
        return artifact

    def list_artifacts(
        self,
        artifact_type: str,
        window_id: int | None = None,
    ) -> list[dict[str, Any]]:
        artifacts: list[dict[str, Any]] = []
        for record in self.metadata_index.list_records(artifact_type, window_id=window_id):
            artifact = self.get_artifact(artifact_type, record["artifact_id"])
            if artifact is not None:
                artifacts.append(artifact)
        return artifacts

    def latest_policy(self) -> dict[str, Any] | None:
        record = self.metadata_index.latest_policy_record()
        if record is None:
            return None
        return self.get_artifact("policy", record["artifact_id"])
