"""Tests for the Forge-side R2 checkpoint backend.

End-to-end: write + retrieve a DeltaBundle through R2CheckpointBackend
via a FakeS3 client; assert the bundle round-trips through the same
publish_bundle / fetch_bundle path the local filesystem backend uses.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from reliquary.training.checkpoint_storage import (
    CheckpointStorageBackend,
    fetch_bundle,
    publish_bundle,
)
from reliquary.training.delta_checkpoints import (
    DeltaBundle,
    compute_delta,
    reconstruct,
)
from reliquary.training.r2_checkpoint_backend import (
    R2CheckpointBackend,
    R2CheckpointUnavailable,
    _boto3_client,
    _is_not_found,
)


class _NotFound(Exception):
    def __init__(self) -> None:
        super().__init__("NoSuchKey")
        self.response = {
            "Error": {"Code": "NoSuchKey"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        }


@dataclass
class FakeS3:
    store: dict[tuple[str, str], bytes] = field(default_factory=dict)
    page_size: int = 4

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> dict:
        self.store[(Bucket, Key)] = bytes(Body)
        return {"ETag": "fake"}

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        if (Bucket, Key) not in self.store:
            raise _NotFound()
        return {"Body": io.BytesIO(self.store[(Bucket, Key)])}

    def list_objects_v2(
        self, *, Bucket: str, Prefix: str = "", ContinuationToken: str = ""
    ) -> dict:
        matching = sorted(
            key for (b, key) in self.store if b == Bucket and key.startswith(Prefix)
        )
        offset = int(ContinuationToken) if ContinuationToken else 0
        page = matching[offset : offset + self.page_size]
        is_truncated = offset + self.page_size < len(matching)
        response: dict[str, Any] = {
            "Contents": [{"Key": key} for key in page],
            "IsTruncated": is_truncated,
        }
        if is_truncated:
            response["NextContinuationToken"] = str(offset + self.page_size)
        return response

    def delete_object(self, *, Bucket: str, Key: str) -> dict:
        self.store.pop((Bucket, Key), None)
        return {}


def _build_base_target() -> tuple[dict, dict]:
    torch.manual_seed(0)
    base = {
        "encoder.weight": torch.randn(8, 4, dtype=torch.float32),
        "decoder.bias": torch.randn(4, dtype=torch.float32),
    }
    target = {
        "encoder.weight": base["encoder.weight"] + 0.01 * torch.randn(8, 4),
        "decoder.bias": base["decoder.bias"] + 0.01 * torch.randn(4),
    }
    return base, target


def _build_bundle() -> tuple[DeltaBundle, dict, dict]:
    base, target = _build_base_target()
    bundle = compute_delta(base, target)
    return bundle, base, target


# ---------------------------------------------------------------------------
# Backend unit behavior
# ---------------------------------------------------------------------------


def test_put_then_get_returns_bytes() -> None:
    client = FakeS3()
    backend = R2CheckpointBackend(bucket="ckpt", client=client)
    backend.put("k", b"hello")
    assert backend.get("k") == b"hello"


def test_get_missing_returns_none() -> None:
    client = FakeS3()
    backend = R2CheckpointBackend(bucket="ckpt", client=client)
    assert backend.get("absent") is None


def test_delete_removes_key() -> None:
    client = FakeS3()
    backend = R2CheckpointBackend(bucket="ckpt", client=client)
    backend.put("k", b"v")
    backend.delete("k")
    assert backend.get("k") is None


def test_key_prefix_applied_and_stripped_on_list() -> None:
    client = FakeS3()
    backend = R2CheckpointBackend(
        bucket="ckpt", key_prefix="reliquary/forge", client=client
    )
    backend.put("checkpoints/abc/bundle.json", b"data")
    # Stored with the full prefixed key.
    assert ("ckpt", "reliquary/forge/checkpoints/abc/bundle.json") in client.store
    # list() strips the prefix so callers can round-trip via get().
    listed = backend.list("checkpoints/abc/")
    assert listed == ["checkpoints/abc/bundle.json"]


def test_list_paginates_through_truncated_results() -> None:
    client = FakeS3()
    client.page_size = 2
    backend = R2CheckpointBackend(bucket="ckpt", client=client)
    for i in range(5):
        backend.put(f"checkpoints/ckpt-{i:02d}.bin", b"x")
    listed = backend.list("checkpoints/")
    assert len(listed) == 5


def test_is_not_found_detects_shapes() -> None:
    assert _is_not_found(_NotFound()) is True

    class _AccessDenied(Exception):
        response = {"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {"HTTPStatusCode": 403}}

    assert _is_not_found(_AccessDenied("AccessDenied")) is False


def test_boto3_missing_raises_r2_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    saved = sys.modules.pop("boto3", None)
    monkeypatch.setitem(sys.modules, "boto3", None)  # type: ignore[assignment]
    try:
        with pytest.raises(R2CheckpointUnavailable):
            _boto3_client(
                endpoint_url="", access_key_id="", secret_access_key="", region="auto"
            )
    finally:
        if saved is not None:
            sys.modules["boto3"] = saved


# ---------------------------------------------------------------------------
# End-to-end publish_bundle → fetch_bundle through R2CheckpointBackend
# ---------------------------------------------------------------------------


def test_publish_bundle_then_fetch_over_r2() -> None:
    client = FakeS3()
    backend = R2CheckpointBackend(bucket="ckpt", client=client)
    bundle, base, _ = _build_bundle()

    publish_bundle(bundle, run_id="run-X", window_id=3, backend=backend)
    fetched = fetch_bundle("run-X", 3, backend=backend)

    assert fetched.merkle_root_hex == bundle.merkle_root_hex
    # Reconstructed tensors match the original reconstruction.
    original_recon = reconstruct(base, bundle)
    fetched_recon = reconstruct(base, fetched)
    for name in original_recon:
        assert torch.equal(original_recon[name], fetched_recon[name])


def test_publish_bundle_paginates_shards_through_r2() -> None:
    client = FakeS3()
    client.page_size = 1
    backend = R2CheckpointBackend(bucket="ckpt", client=client)
    bundle, base, _ = _build_bundle()
    publish_bundle(bundle, run_id="run-Y", window_id=5, backend=backend)
    # Cross the pagination boundary on a fetch that has to list shards.
    fetched = fetch_bundle("run-Y", 5, backend=backend)
    assert fetched.merkle_root_hex == bundle.merkle_root_hex


def test_prefix_scoping_isolates_two_runs() -> None:
    client = FakeS3()
    backend_a = R2CheckpointBackend(
        bucket="ckpt", key_prefix="tenant-a", client=client
    )
    backend_b = R2CheckpointBackend(
        bucket="ckpt", key_prefix="tenant-b", client=client
    )
    bundle_a, _, _ = _build_bundle()
    publish_bundle(bundle_a, run_id="run-1", window_id=1, backend=backend_a)

    # Tenant B's backend cannot see tenant A's data.
    from reliquary.training.checkpoint_storage import DeltaStorageNotFound

    with pytest.raises(DeltaStorageNotFound):
        fetch_bundle("run-1", 1, backend=backend_b)


def test_r2_backend_matches_checkpoint_storage_protocol() -> None:
    """R2CheckpointBackend exposes put/get/list/delete so it can be passed
    anywhere CheckpointStorageBackend is expected."""
    client = FakeS3()
    backend = R2CheckpointBackend(bucket="ckpt", client=client)
    for method in ("put", "get", "list", "delete"):
        assert callable(getattr(backend, method))
