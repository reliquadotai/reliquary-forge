"""Tests for the shard-parallel DeltaBundle fetcher.

Verifies the parallel fetcher produces bundles byte-equivalent to the
sequential fetcher and rejects the same corruption patterns.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from reliquary.training.checkpoint_storage import (
    DeltaStorageShardCorrupted,
    DeltaStorageShardMissing,
    LocalFilesystemBackend,
    fetch_bundle,
    publish_bundle,
)
from reliquary.training.delta_checkpoints import (
    DeltaMerkleMismatch,
    compute_delta,
    reconstruct,
)
from reliquary.training.parallel_fetch import (
    DEFAULT_PARALLEL_WORKERS,
    fetch_bundle_parallel,
)


def _build_base_target() -> tuple[dict, dict]:
    torch.manual_seed(0)
    base = {
        "layer.0.weight": torch.randn(4, 3, dtype=torch.float32),
        "layer.0.bias": torch.randn(3, dtype=torch.float32),
        "layer.1.weight": torch.randn(3, 2, dtype=torch.float32),
        "layer.1.bias": torch.randn(2, dtype=torch.float32),
    }
    target = {
        name: value + 0.01 * torch.randn_like(value) for name, value in base.items()
    }
    return base, target


def _build_and_publish(tmp_path: Path) -> tuple[Path, dict]:
    backend = LocalFilesystemBackend(str(tmp_path))
    base, target = _build_base_target()
    bundle = compute_delta(base, target)
    publish_bundle(bundle, run_id="run-P", window_id=1, backend=backend)
    return tmp_path, base


# ---------------------------------------------------------------------------
# Parity between sequential and parallel fetchers
# ---------------------------------------------------------------------------


def test_parallel_fetch_equivalent_to_sequential(tmp_path: Path) -> None:
    root, base = _build_and_publish(tmp_path)
    backend = LocalFilesystemBackend(str(root))
    sequential = fetch_bundle("run-P", 1, backend=backend)
    parallel = fetch_bundle_parallel("run-P", 1, backend=backend)

    assert parallel.merkle_root_hex == sequential.merkle_root_hex
    assert parallel.from_checkpoint_hash == sequential.from_checkpoint_hash
    assert parallel.to_checkpoint_hash == sequential.to_checkpoint_hash

    seq_hashes = [s.payload_sha256 for s in sequential.shards]
    par_hashes = [s.payload_sha256 for s in parallel.shards]
    assert seq_hashes == par_hashes

    # Reconstruction from the parallel bundle matches the sequential one.
    seq_recon = reconstruct(base, sequential)
    par_recon = reconstruct(base, parallel)
    for name in seq_recon:
        assert torch.equal(seq_recon[name], par_recon[name])


def test_parallel_fetch_raises_on_missing_shard(tmp_path: Path) -> None:
    root, _ = _build_and_publish(tmp_path)
    backend = LocalFilesystemBackend(str(root))
    # Delete one shard file from disk to simulate the backend losing it.
    bundle = fetch_bundle("run-P", 1, backend=backend)
    target_shard = bundle.shards[0].tensor_name
    shard_path = (
        root
        / "checkpoints"
        / "run-P"
        / "1"
        / "shards"
        / f"{target_shard}.bin"
    )
    shard_path.unlink()

    with pytest.raises(DeltaStorageShardMissing):
        fetch_bundle_parallel("run-P", 1, backend=backend)


def test_parallel_fetch_raises_on_corrupted_shard(tmp_path: Path) -> None:
    root, _ = _build_and_publish(tmp_path)
    backend = LocalFilesystemBackend(str(root))
    # Byte-flip one shard.
    bundle = fetch_bundle("run-P", 1, backend=backend)
    target_shard = bundle.shards[0].tensor_name
    shard_path = (
        root
        / "checkpoints"
        / "run-P"
        / "1"
        / "shards"
        / f"{target_shard}.bin"
    )
    data = bytearray(shard_path.read_bytes())
    data[0] ^= 0xFF
    shard_path.write_bytes(bytes(data))

    with pytest.raises(DeltaStorageShardCorrupted):
        fetch_bundle_parallel("run-P", 1, backend=backend)


def test_parallel_fetch_merkle_mismatch_detected(tmp_path: Path) -> None:
    root, _ = _build_and_publish(tmp_path)
    backend = LocalFilesystemBackend(str(root))
    import json

    manifest_path = root / "checkpoints" / "run-P" / "1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["merkle_root_hex"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(DeltaMerkleMismatch):
        fetch_bundle_parallel("run-P", 1, backend=backend)


def test_parallel_fetch_runs_concurrently(tmp_path: Path) -> None:
    """Assert that at least two get() calls execute in overlapping time
    windows. We wrap the backend with a thread-observing proxy."""

    root, _ = _build_and_publish(tmp_path)
    inner = LocalFilesystemBackend(str(root))

    overlap_observed = threading.Event()
    active: dict[int, threading.Event] = {}
    active_lock = threading.Lock()

    class _OverlapProbe:
        def put(self, key, data):
            inner.put(key, data)

        def get(self, key):
            tid = threading.get_ident()
            enter_event = threading.Event()
            with active_lock:
                active[tid] = enter_event
                if len(active) >= 2:
                    overlap_observed.set()
            try:
                return inner.get(key)
            finally:
                with active_lock:
                    active.pop(tid, None)

        def list(self, prefix):
            return inner.list(prefix)

        def delete(self, key):
            inner.delete(key)

    probe = _OverlapProbe()
    fetch_bundle_parallel("run-P", 1, backend=probe, max_workers=4)
    # With >=2 shards in the bundle, the parallel pool should overlap
    # at least one pair of gets.
    assert overlap_observed.is_set()


def test_parallel_fetch_max_workers_validation(tmp_path: Path) -> None:
    root, _ = _build_and_publish(tmp_path)
    backend = LocalFilesystemBackend(str(root))
    with pytest.raises(ValueError):
        fetch_bundle_parallel("run-P", 1, backend=backend, max_workers=0)


def test_parallel_fetch_single_worker_still_works(tmp_path: Path) -> None:
    root, base = _build_and_publish(tmp_path)
    backend = LocalFilesystemBackend(str(root))
    bundle = fetch_bundle_parallel("run-P", 1, backend=backend, max_workers=1)
    sequential = fetch_bundle("run-P", 1, backend=backend)
    assert bundle.merkle_root_hex == sequential.merkle_root_hex


def test_default_parallel_workers_is_8() -> None:
    # Pins the PRD-target default (Tier 2 Epic 2 acceptance: "up to 8
    # parallel R2 fetches").
    assert DEFAULT_PARALLEL_WORKERS == 8


def test_parallel_fetch_not_found_when_manifest_absent(tmp_path: Path) -> None:
    backend = LocalFilesystemBackend(str(tmp_path))
    from reliquary.training.checkpoint_storage import DeltaStorageNotFound

    with pytest.raises(DeltaStorageNotFound):
        fetch_bundle_parallel("missing-run", 1, backend=backend)
