"""Acceptance tests for the delta-checkpoint storage pipeline.

Covers the 18 acceptance cases from
``private/reliquary-plan/notes/spec-delta-storage.md``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from reliquary.training.checkpoint_storage import (
    DELTA_STORAGE_KIND_DELTA,
    DELTA_STORAGE_KIND_FULL,
    DELTA_STORAGE_VERSION,
    CheckpointManifest,
    DeltaStorageInvalidName,
    DeltaStorageInvalidRunId,
    DeltaStorageManifestInvalid,
    DeltaStorageNotFound,
    DeltaStorageShardCorrupted,
    DeltaStorageShardMissing,
    LocalFilesystemBackend,
    RecordingBackend,
    SnapshotScheduler,
    checkpoint_manifest_key,
    checkpoint_shard_key,
    fetch_bundle,
    fetch_full_snapshot,
    publish_bundle,
    publish_full_snapshot,
)
from reliquary.training.delta_checkpoints import (
    DeltaBundle,
    DeltaMerkleMismatch,
    apply_chain,
    compute_delta,
    reconstruct,
)


def _build_base_target() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
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


def _build_bundle() -> tuple[DeltaBundle, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    base, target = _build_base_target()
    bundle = compute_delta(base, target)
    return bundle, base, target


def test_publish_fetch_delta_roundtrip(tmp_path: Path) -> None:
    backend = LocalFilesystemBackend(tmp_path)
    bundle, base, _ = _build_bundle()

    manifest = publish_bundle(bundle, run_id="run-A", window_id=7, backend=backend)
    assert isinstance(manifest, CheckpointManifest)
    assert manifest.kind == DELTA_STORAGE_KIND_DELTA
    assert manifest.manifest_key.endswith("manifest.json")

    fetched = fetch_bundle("run-A", 7, backend=backend)
    assert fetched.merkle_root_hex == bundle.merkle_root_hex
    assert fetched.from_checkpoint_hash == bundle.from_checkpoint_hash
    assert fetched.to_checkpoint_hash == bundle.to_checkpoint_hash
    assert len(fetched.shards) == len(bundle.shards)

    original_recon = reconstruct(base, bundle)
    fetched_recon = reconstruct(base, fetched)
    for name in original_recon:
        assert torch.equal(original_recon[name], fetched_recon[name])


def test_publish_fetch_full_snapshot_roundtrip(tmp_path: Path) -> None:
    backend = LocalFilesystemBackend(tmp_path)
    _, target = _build_base_target()

    manifest = publish_full_snapshot(
        target, run_id="run-B", window_id=0, backend=backend, dtype="fp16"
    )
    assert manifest.kind == DELTA_STORAGE_KIND_FULL
    fetched = fetch_full_snapshot("run-B", 0, backend=backend)

    for name, tensor in target.items():
        assert name in fetched
        expected_fp16 = tensor.to(torch.float16)
        assert fetched[name].shape == expected_fp16.shape
        assert fetched[name].dtype == torch.float16
        assert torch.equal(fetched[name], expected_fp16)


def test_manifest_canonical_bytes(tmp_path: Path) -> None:
    backend_one = LocalFilesystemBackend(tmp_path / "one")
    backend_two = LocalFilesystemBackend(tmp_path / "two")
    bundle, _, _ = _build_bundle()

    publish_bundle(bundle, run_id="run-C", window_id=1, backend=backend_one)
    publish_bundle(bundle, run_id="run-C", window_id=1, backend=backend_two)

    manifest_bytes_one = backend_one.get(checkpoint_manifest_key("run-C", 1))
    manifest_bytes_two = backend_two.get(checkpoint_manifest_key("run-C", 1))
    assert manifest_bytes_one == manifest_bytes_two
    parsed = json.loads(manifest_bytes_one.decode("utf-8"))
    assert parsed["version"] == DELTA_STORAGE_VERSION


def test_atomic_manifest_publish(tmp_path: Path) -> None:
    backend = LocalFilesystemBackend(tmp_path)
    bundle, _, _ = _build_bundle()
    publish_bundle(bundle, run_id="run-D", window_id=2, backend=backend)

    manifest_path = tmp_path / "checkpoints" / "run-D" / "2" / "manifest.json"
    assert manifest_path.is_file()
    leftover_tempfiles = [
        p for p in manifest_path.parent.iterdir() if p.name.startswith(".tmp-")
    ]
    assert leftover_tempfiles == []


def test_shard_before_manifest_ordering(tmp_path: Path) -> None:
    inner = LocalFilesystemBackend(tmp_path)
    recorder = RecordingBackend(inner)
    bundle, _, _ = _build_bundle()

    publish_bundle(bundle, run_id="run-E", window_id=3, backend=recorder)

    manifest_key = checkpoint_manifest_key("run-E", 3)
    assert manifest_key in recorder.put_calls
    manifest_index = recorder.put_calls.index(manifest_key)
    shard_keys = [
        checkpoint_shard_key("run-E", 3, s.tensor_name) for s in bundle.shards
    ]
    for sk in shard_keys:
        assert sk in recorder.put_calls
        assert recorder.put_calls.index(sk) < manifest_index


def test_corrupted_shard_detected(tmp_path: Path) -> None:
    backend = LocalFilesystemBackend(tmp_path)
    bundle, _, _ = _build_bundle()
    publish_bundle(bundle, run_id="run-F", window_id=4, backend=backend)

    shard_name = bundle.shards[0].tensor_name
    shard_path = tmp_path / "checkpoints" / "run-F" / "4" / "shards" / f"{shard_name}.bin"
    data = bytearray(shard_path.read_bytes())
    data[0] ^= 0xFF
    shard_path.write_bytes(bytes(data))

    with pytest.raises(DeltaStorageShardCorrupted):
        fetch_bundle("run-F", 4, backend=backend)


def test_missing_shard_detected(tmp_path: Path) -> None:
    backend = LocalFilesystemBackend(tmp_path)
    bundle, _, _ = _build_bundle()
    publish_bundle(bundle, run_id="run-G", window_id=5, backend=backend)

    shard_name = bundle.shards[0].tensor_name
    shard_path = tmp_path / "checkpoints" / "run-G" / "5" / "shards" / f"{shard_name}.bin"
    shard_path.unlink()

    with pytest.raises(DeltaStorageShardMissing):
        fetch_bundle("run-G", 5, backend=backend)


def test_manifest_schema_enforcement(tmp_path: Path) -> None:
    backend = LocalFilesystemBackend(tmp_path)
    bundle, _, _ = _build_bundle()
    publish_bundle(bundle, run_id="run-H", window_id=6, backend=backend)

    manifest_key = checkpoint_manifest_key("run-H", 6)
    manifest = json.loads(backend.get(manifest_key).decode("utf-8"))
    del manifest["merkle_root_hex"]
    backend.put(
        manifest_key,
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )

    with pytest.raises(DeltaStorageManifestInvalid):
        fetch_bundle("run-H", 6, backend=backend)


def test_tensor_name_path_traversal_rejected(tmp_path: Path) -> None:
    backend = LocalFilesystemBackend(tmp_path)
    bundle, _, _ = _build_bundle()

    malicious = DeltaBundle(
        from_checkpoint_hash=bundle.from_checkpoint_hash,
        to_checkpoint_hash=bundle.to_checkpoint_hash,
        shards=(
            *bundle.shards,
            type(bundle.shards[0])(
                tensor_name="../evil",
                dtype="int8",
                shape=(1,),
                scale=0.0,
                zero_point=0,
                payload_sha256=hashlib.sha256(b"x").hexdigest(),
                data_bytes=b"x",
            ),
        ),
        merkle_root_hex=bundle.merkle_root_hex,
    )

    with pytest.raises(DeltaStorageInvalidName):
        publish_bundle(malicious, run_id="run-I", window_id=7, backend=backend)


def test_run_id_validation(tmp_path: Path) -> None:
    backend = LocalFilesystemBackend(tmp_path)
    bundle, _, _ = _build_bundle()

    with pytest.raises(DeltaStorageInvalidRunId):
        publish_bundle(bundle, run_id="bad/run", window_id=0, backend=backend)
    with pytest.raises(DeltaStorageInvalidRunId):
        publish_bundle(bundle, run_id="", window_id=0, backend=backend)


def test_merkle_root_mismatch_on_fetch_detected(tmp_path: Path) -> None:
    backend = LocalFilesystemBackend(tmp_path)
    bundle, _, _ = _build_bundle()
    publish_bundle(bundle, run_id="run-K", window_id=8, backend=backend)

    manifest_key = checkpoint_manifest_key("run-K", 8)
    manifest = json.loads(backend.get(manifest_key).decode("utf-8"))
    forged = "0" * 64
    manifest["merkle_root_hex"] = forged
    backend.put(
        manifest_key,
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )

    with pytest.raises(DeltaMerkleMismatch):
        fetch_bundle("run-K", 8, backend=backend)


def test_cross_prefix_shard_key_rejected(tmp_path: Path) -> None:
    backend = LocalFilesystemBackend(tmp_path)
    bundle, _, _ = _build_bundle()
    publish_bundle(bundle, run_id="run-L", window_id=9, backend=backend)

    manifest_key = checkpoint_manifest_key("run-L", 9)
    manifest = json.loads(backend.get(manifest_key).decode("utf-8"))
    manifest["shards"][0]["shard_key"] = "checkpoints/other-run/9/shards/stolen.bin"
    backend.put(
        manifest_key,
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )

    with pytest.raises(DeltaStorageManifestInvalid):
        fetch_bundle("run-L", 9, backend=backend)


def test_snapshot_scheduler_bootstrap_full() -> None:
    scheduler = SnapshotScheduler(cadence_windows=64)
    decision = scheduler.decide(0)
    assert decision.kind == DELTA_STORAGE_KIND_FULL
    assert decision.reason == "bootstrap"


def test_snapshot_scheduler_cadence() -> None:
    scheduler = SnapshotScheduler(cadence_windows=64)
    scheduler.record_publish(0, kind=DELTA_STORAGE_KIND_FULL)

    assert scheduler.decide(1).kind == DELTA_STORAGE_KIND_DELTA
    assert scheduler.decide(63).kind == DELTA_STORAGE_KIND_DELTA
    cadence_decision = scheduler.decide(64)
    assert cadence_decision.kind == DELTA_STORAGE_KIND_FULL
    assert cadence_decision.reason == "cadence"

    scheduler.record_publish(64, kind=DELTA_STORAGE_KIND_FULL)
    assert scheduler.decide(65).kind == DELTA_STORAGE_KIND_DELTA


def test_snapshot_scheduler_force_full_override() -> None:
    scheduler = SnapshotScheduler(cadence_windows=64)
    scheduler.record_publish(0, kind=DELTA_STORAGE_KIND_FULL)
    decision = scheduler.decide(10, force_full=True)
    assert decision.kind == DELTA_STORAGE_KIND_FULL
    assert decision.reason == "forced"


def test_fetch_returns_reconstructable_bundle(tmp_path: Path) -> None:
    backend = LocalFilesystemBackend(tmp_path)
    bundle, base, _ = _build_bundle()
    publish_bundle(bundle, run_id="run-M", window_id=10, backend=backend)

    fetched = fetch_bundle("run-M", 10, backend=backend)
    chained = apply_chain(base, [fetched])
    original_recon = reconstruct(base, bundle)
    for name in original_recon:
        assert torch.equal(chained[name], original_recon[name])


def test_missing_manifest_raises_not_found(tmp_path: Path) -> None:
    backend = LocalFilesystemBackend(tmp_path)
    with pytest.raises(DeltaStorageNotFound):
        fetch_bundle("run-N", 11, backend=backend)


def test_idempotent_republish(tmp_path: Path) -> None:
    backend = LocalFilesystemBackend(tmp_path)
    bundle, _, _ = _build_bundle()
    publish_bundle(bundle, run_id="run-O", window_id=12, backend=backend)
    manifest_key = checkpoint_manifest_key("run-O", 12)
    first_bytes = backend.get(manifest_key)
    first_shards = {
        s.tensor_name: backend.get(checkpoint_shard_key("run-O", 12, s.tensor_name))
        for s in bundle.shards
    }

    publish_bundle(bundle, run_id="run-O", window_id=12, backend=backend)
    second_bytes = backend.get(manifest_key)
    second_shards = {
        s.tensor_name: backend.get(checkpoint_shard_key("run-O", 12, s.tensor_name))
        for s in bundle.shards
    }

    assert first_bytes == second_bytes
    assert first_shards == second_shards
