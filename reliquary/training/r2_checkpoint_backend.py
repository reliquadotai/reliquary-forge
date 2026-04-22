"""R2/S3-compatible backend for delta-checkpoint storage.

Companion to ``reliquary_inference.validator.r2_backend`` but targets
the ``CheckpointStorageBackend`` Protocol from
``reliquary.training.checkpoint_storage``. Production Forge validators
point this at their R2 bucket so miners + sibling validators can pull
the reconstructed policy state across training windows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class R2CheckpointUnavailable(RuntimeError):
    """Raised when boto3 is required but not installed."""


class _S3Client(Protocol):
    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> Any: ...
    def get_object(self, *, Bucket: str, Key: str) -> Any: ...
    def list_objects_v2(self, *, Bucket: str, Prefix: str = "", ContinuationToken: str = "") -> Any: ...
    def delete_object(self, *, Bucket: str, Key: str) -> Any: ...


def _boto3_client(
    *,
    endpoint_url: str,
    access_key_id: str,
    secret_access_key: str,
    region: str,
) -> _S3Client:
    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError as exc:
        raise R2CheckpointUnavailable(
            "boto3 is not installed; install with `pip install reliquary[r2]` "
            "or pass a client explicitly to R2CheckpointBackend"
        ) from exc
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url or None,
        aws_access_key_id=access_key_id or None,
        aws_secret_access_key=secret_access_key or None,
        region_name=region or None,
    )


@dataclass
class R2CheckpointBackend:
    """S3-compatible CheckpointStorageBackend for delta-checkpoint publish/fetch.

    Matches the CheckpointStorageBackend Protocol exposed by
    ``reliquary.training.checkpoint_storage`` (put / get / list / delete).
    """

    bucket: str
    endpoint_url: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    region: str = "auto"
    key_prefix: str = ""
    client: _S3Client | None = None
    _initialized: bool = field(default=False, init=False, repr=False)

    def _ensure_client(self) -> _S3Client:
        if self.client is None:
            self.client = _boto3_client(
                endpoint_url=self.endpoint_url,
                access_key_id=self.access_key_id,
                secret_access_key=self.secret_access_key,
                region=self.region,
            )
        return self.client

    def _full_key(self, key: str) -> str:
        if not self.key_prefix:
            return key
        return f"{self.key_prefix.rstrip('/')}/{key.lstrip('/')}"

    def put(self, key: str, data: bytes) -> None:
        client = self._ensure_client()
        client.put_object(Bucket=self.bucket, Key=self._full_key(key), Body=data)

    def get(self, key: str) -> bytes | None:
        client = self._ensure_client()
        full_key = self._full_key(key)
        try:
            response = client.get_object(Bucket=self.bucket, Key=full_key)
        except Exception as exc:
            if _is_not_found(exc):
                return None
            raise
        body = response.get("Body")
        if body is None:
            return b""
        return body.read()

    def list(self, prefix: str) -> list[str]:
        client = self._ensure_client()
        full_prefix = self._full_key(prefix) if prefix else self.key_prefix
        keys: list[str] = []
        continuation: str | None = None
        strip_prefix = f"{self.key_prefix.rstrip('/')}/" if self.key_prefix else ""
        while True:
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": full_prefix}
            if continuation is not None:
                kwargs["ContinuationToken"] = continuation
            response = client.list_objects_v2(**kwargs)
            for entry in response.get("Contents") or []:
                stored_key = str(entry["Key"])
                if strip_prefix and stored_key.startswith(strip_prefix):
                    stored_key = stored_key[len(strip_prefix):]
                keys.append(stored_key)
            if response.get("IsTruncated") and response.get("NextContinuationToken"):
                continuation = response["NextContinuationToken"]
                continue
            break
        return sorted(keys)

    def delete(self, key: str) -> None:
        client = self._ensure_client()
        client.delete_object(Bucket=self.bucket, Key=self._full_key(key))


def _is_not_found(exc: Exception) -> bool:
    """Detect S3 404 across boto3 versions."""
    response_error = getattr(exc, "response", None)
    if isinstance(response_error, dict):
        code = str(response_error.get("Error", {}).get("Code", ""))
        if code in {"NoSuchKey", "404", "NotFound"}:
            return True
    if hasattr(exc, "response"):
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")  # type: ignore[attr-defined]
        if status == 404:
            return True
    text = str(exc).lower()
    return "nosuchkey" in text or "not found" in text


__all__ = [
    "R2CheckpointBackend",
    "R2CheckpointUnavailable",
]
