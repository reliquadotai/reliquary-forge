"""Delta checkpoint storage pipeline.

Publish and fetch ``DeltaBundle`` artifacts (and full snapshots) through a
pluggable object-store backend. Pure-python Protocol surface so callers can
drop in R2/S3/GCS backends without touching this module.

The pure-compute core — delta quantization, Merkle binding, reconstruction —
lives in ``reliquary.training.delta_checkpoints``. This module is the I/O
layer: serialization, key layout, shard-first write ordering, fail-before-
mutate fetch verification, and a full-vs-delta cadence scheduler.

Spec: private/reliquary-plan/notes/spec-delta-storage.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol

from reliquary.training.delta_checkpoints import (
    DeltaBundle,
    DeltaMerkleMismatch,
    DeltaShard,
    _merkle_root_hex,
)


DELTA_STORAGE_VERSION = "v1"
DELTA_STORAGE_MANIFEST_FILENAME = "manifest.json"
DELTA_STORAGE_SHARD_PREFIX = "shards"
DELTA_STORAGE_KIND_DELTA = "delta"
DELTA_STORAGE_KIND_FULL = "full"
DEFAULT_FULL_SNAPSHOT_CADENCE_WINDOWS = 64

RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_.\-]{1,64}$")
TENSOR_NAME_SAFE_PATTERN = re.compile(r"^[a-zA-Z0-9_.\-]+$")

_MANIFEST_REQUIRED_FIELDS = frozenset(
    {
        "version",
        "kind",
        "run_id",
        "window_id",
        "from_checkpoint_hash",
        "to_checkpoint_hash",
        "merkle_root_hex",
        "shards",
    }
)
_SHARD_DESCRIPTOR_REQUIRED_FIELDS = frozenset(
    {
        "tensor_name",
        "dtype",
        "shape",
        "scale",
        "zero_point",
        "payload_sha256",
        "byte_length",
        "shard_key",
    }
)


class DeltaStorageError(Exception):
    """Base class for storage-layer errors."""


class DeltaStorageNotFound(DeltaStorageError):
    """Manifest for the requested (run_id, window_id) is absent."""


class DeltaStorageManifestInvalid(DeltaStorageError):
    """Manifest JSON is structurally broken or references keys outside the window prefix."""


class DeltaStorageShardMissing(DeltaStorageError):
    """A shard referenced by the manifest is not present in the backend."""


class DeltaStorageShardCorrupted(DeltaStorageError):
    """A shard's byte length or sha256 disagrees with the manifest descriptor."""


class DeltaStorageInvalidName(DeltaStorageError):
    """Tensor name contains characters outside the path-safe whitelist."""


class DeltaStorageInvalidRunId(DeltaStorageError):
    """run_id fails the ``RUN_ID_PATTERN`` whitelist."""


class CheckpointStorageBackend(Protocol):
    """Object-store surface: put/get/list/delete by string key."""

    def put(self, key: str, data: bytes) -> None: ...

    def get(self, key: str) -> bytes | None: ...

    def list(self, prefix: str) -> list[str]: ...

    def delete(self, key: str) -> None: ...


@dataclass(frozen=True)
class CheckpointManifest:
    run_id: str
    window_id: int
    kind: str
    manifest_key: str
    shard_keys: tuple[str, ...]
    merkle_root_hex: str
    published_at: float


@dataclass(frozen=True)
class SnapshotDecision:
    kind: str
    reason: str


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise DeltaStorageInvalidRunId(
            f"run_id {run_id!r} must match {RUN_ID_PATTERN.pattern}"
        )


def _validate_window_id(window_id: int) -> None:
    if not isinstance(window_id, int) or isinstance(window_id, bool) or window_id < 0:
        raise DeltaStorageError(
            f"window_id must be a non-negative int, got {window_id!r}"
        )


def _validate_tensor_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\x00" in name
        or not TENSOR_NAME_SAFE_PATTERN.fullmatch(name)
    ):
        raise DeltaStorageInvalidName(
            f"tensor_name {name!r} must match {TENSOR_NAME_SAFE_PATTERN.pattern}"
        )


def checkpoint_window_prefix(run_id: str, window_id: int) -> str:
    _validate_run_id(run_id)
    _validate_window_id(window_id)
    return f"checkpoints/{run_id}/{window_id}/"


def checkpoint_manifest_key(run_id: str, window_id: int) -> str:
    return checkpoint_window_prefix(run_id, window_id) + DELTA_STORAGE_MANIFEST_FILENAME


def checkpoint_shard_key(run_id: str, window_id: int, tensor_name: str) -> str:
    _validate_tensor_name(tensor_name)
    return (
        checkpoint_window_prefix(run_id, window_id)
        + f"{DELTA_STORAGE_SHARD_PREFIX}/{tensor_name}.bin"
    )


def checkpoint_kind_marker_key(run_id: str, window_id: int, kind: str) -> str:
    if kind not in {DELTA_STORAGE_KIND_DELTA, DELTA_STORAGE_KIND_FULL}:
        raise DeltaStorageError(f"unknown kind {kind!r}")
    return checkpoint_window_prefix(run_id, window_id) + f"kind={kind}"


def _canonical_manifest_bytes(manifest: dict) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _shard_descriptor(shard: DeltaShard, shard_key: str) -> dict:
    return {
        "tensor_name": shard.tensor_name,
        "dtype": shard.dtype,
        "shape": list(shard.shape),
        "scale": shard.scale,
        "zero_point": shard.zero_point,
        "payload_sha256": shard.payload_sha256,
        "byte_length": len(shard.data_bytes),
        "shard_key": shard_key,
    }


def _validate_manifest_structure(
    manifest: dict, *, expected_kind: str, run_id: str, window_id: int
) -> None:
    if not isinstance(manifest, dict):
        raise DeltaStorageManifestInvalid("manifest is not an object")
    missing = _MANIFEST_REQUIRED_FIELDS - manifest.keys()
    if missing:
        raise DeltaStorageManifestInvalid(f"manifest missing fields: {sorted(missing)}")
    if manifest["version"] != DELTA_STORAGE_VERSION:
        raise DeltaStorageManifestInvalid(
            f"manifest version {manifest['version']!r} != {DELTA_STORAGE_VERSION!r}"
        )
    if manifest["kind"] != expected_kind:
        raise DeltaStorageManifestInvalid(
            f"manifest kind {manifest['kind']!r} != expected {expected_kind!r}"
        )
    if manifest["run_id"] != run_id:
        raise DeltaStorageManifestInvalid(
            f"manifest run_id {manifest['run_id']!r} != expected {run_id!r}"
        )
    if manifest["window_id"] != window_id:
        raise DeltaStorageManifestInvalid(
            f"manifest window_id {manifest['window_id']!r} != expected {window_id!r}"
        )
    shards = manifest["shards"]
    if not isinstance(shards, list):
        raise DeltaStorageManifestInvalid("manifest.shards is not a list")
    seen: set[str] = set()
    expected_prefix = checkpoint_window_prefix(run_id, window_id) + DELTA_STORAGE_SHARD_PREFIX + "/"
    for descriptor in shards:
        if not isinstance(descriptor, dict):
            raise DeltaStorageManifestInvalid("shard descriptor is not an object")
        missing_shard = _SHARD_DESCRIPTOR_REQUIRED_FIELDS - descriptor.keys()
        if missing_shard:
            raise DeltaStorageManifestInvalid(
                f"shard descriptor missing fields: {sorted(missing_shard)}"
            )
        tensor_name = descriptor["tensor_name"]
        _validate_tensor_name(tensor_name)
        if tensor_name in seen:
            raise DeltaStorageManifestInvalid(
                f"duplicate tensor_name {tensor_name!r} in manifest"
            )
        seen.add(tensor_name)
        shard_key = descriptor["shard_key"]
        if not isinstance(shard_key, str) or not shard_key.startswith(expected_prefix):
            raise DeltaStorageManifestInvalid(
                f"shard_key {shard_key!r} outside window prefix {expected_prefix!r}"
            )
        expected_key = checkpoint_shard_key(run_id, window_id, tensor_name)
        if shard_key != expected_key:
            raise DeltaStorageManifestInvalid(
                f"shard_key {shard_key!r} does not match canonical {expected_key!r}"
            )
        shape = descriptor["shape"]
        if not isinstance(shape, list) or not all(
            isinstance(d, int) and d >= 0 for d in shape
        ):
            raise DeltaStorageManifestInvalid(
                f"invalid shape for {tensor_name!r}: {shape!r}"
            )
        byte_length = descriptor["byte_length"]
        if not isinstance(byte_length, int) or byte_length < 0:
            raise DeltaStorageManifestInvalid(
                f"invalid byte_length for {tensor_name!r}: {byte_length!r}"
            )
        payload_sha = descriptor["payload_sha256"]
        if not isinstance(payload_sha, str) or len(payload_sha) != 64:
            raise DeltaStorageManifestInvalid(
                f"invalid payload_sha256 for {tensor_name!r}: {payload_sha!r}"
            )
    names_in_order = [d["tensor_name"] for d in shards]
    if names_in_order != sorted(names_in_order):
        raise DeltaStorageManifestInvalid(
            "manifest.shards must be sorted by tensor_name"
        )


class LocalFilesystemBackend:
    """Filesystem-backed ``CheckpointStorageBackend`` with atomic manifest writes.

    Backing directory = ``root``. Keys are stored as relative paths. All writes
    go through ``tempfile`` + ``os.replace`` on the same filesystem so readers
    never observe torn files.
    """

    def __init__(self, root: str | os.PathLike) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        if key.startswith("/") or ".." in key.split("/") or "\x00" in key:
            raise DeltaStorageManifestInvalid(f"unsafe key {key!r}")
        target = (self._root / key).resolve()
        try:
            target.relative_to(self._root)
        except ValueError as exc:
            raise DeltaStorageManifestInvalid(
                f"key {key!r} escapes storage root"
            ) from exc
        return target

    def put(self, key: str, data: bytes) -> None:
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".tmp-", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, target)
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

    def get(self, key: str) -> bytes | None:
        target = self._resolve(key)
        if not target.is_file():
            return None
        return target.read_bytes()

    def list(self, prefix: str) -> list[str]:
        base = self._resolve(prefix) if prefix else self._root
        if not base.exists():
            return []
        if base.is_file():
            return [str(base.relative_to(self._root))]
        results: list[str] = []
        for path in base.rglob("*"):
            if path.is_file():
                results.append(str(path.relative_to(self._root)))
        return sorted(results)

    def delete(self, key: str) -> None:
        target = self._resolve(key)
        if target.is_file():
            target.unlink()


class RecordingBackend:
    """Test-only wrapper that records put/get/delete order on an inner backend."""

    def __init__(self, inner: CheckpointStorageBackend) -> None:
        self._inner = inner
        self.put_calls: list[str] = []
        self.get_calls: list[str] = []
        self.delete_calls: list[str] = []

    def put(self, key: str, data: bytes) -> None:
        self.put_calls.append(key)
        self._inner.put(key, data)

    def get(self, key: str) -> bytes | None:
        self.get_calls.append(key)
        return self._inner.get(key)

    def list(self, prefix: str) -> list[str]:
        return self._inner.list(prefix)

    def delete(self, key: str) -> None:
        self.delete_calls.append(key)
        self._inner.delete(key)


def publish_bundle(
    bundle: DeltaBundle,
    *,
    run_id: str,
    window_id: int,
    backend: CheckpointStorageBackend,
) -> CheckpointManifest:
    """Publish a delta bundle: shards first, manifest last (read-after-write safe)."""
    _validate_run_id(run_id)
    _validate_window_id(window_id)
    for shard in bundle.shards:
        _validate_tensor_name(shard.tensor_name)

    sorted_shards = sorted(bundle.shards, key=lambda s: s.tensor_name)
    shard_keys: list[str] = []
    for shard in sorted_shards:
        key = checkpoint_shard_key(run_id, window_id, shard.tensor_name)
        backend.put(key, shard.data_bytes)
        shard_keys.append(key)

    manifest_dict = {
        "version": DELTA_STORAGE_VERSION,
        "kind": DELTA_STORAGE_KIND_DELTA,
        "run_id": run_id,
        "window_id": window_id,
        "from_checkpoint_hash": bundle.from_checkpoint_hash,
        "to_checkpoint_hash": bundle.to_checkpoint_hash,
        "merkle_root_hex": bundle.merkle_root_hex,
        "shards": [
            _shard_descriptor(s, k) for s, k in zip(sorted_shards, shard_keys)
        ],
    }
    manifest_bytes = _canonical_manifest_bytes(manifest_dict)
    manifest_key = checkpoint_manifest_key(run_id, window_id)
    backend.put(manifest_key, manifest_bytes)
    backend.put(
        checkpoint_kind_marker_key(run_id, window_id, DELTA_STORAGE_KIND_DELTA),
        b"",
    )

    return CheckpointManifest(
        run_id=run_id,
        window_id=window_id,
        kind=DELTA_STORAGE_KIND_DELTA,
        manifest_key=manifest_key,
        shard_keys=tuple(shard_keys),
        merkle_root_hex=bundle.merkle_root_hex,
        published_at=time.time(),
    )


def fetch_bundle(
    run_id: str,
    window_id: int,
    *,
    backend: CheckpointStorageBackend,
) -> DeltaBundle:
    """Fetch a delta bundle, verifying every shard sha256 + recomputed Merkle root."""
    _validate_run_id(run_id)
    _validate_window_id(window_id)
    manifest_key = checkpoint_manifest_key(run_id, window_id)
    manifest_bytes = backend.get(manifest_key)
    if manifest_bytes is None:
        raise DeltaStorageNotFound(
            f"no manifest at {manifest_key} (run_id={run_id} window_id={window_id})"
        )
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeltaStorageManifestInvalid(f"manifest decode failed: {exc}") from exc

    _validate_manifest_structure(
        manifest, expected_kind=DELTA_STORAGE_KIND_DELTA, run_id=run_id, window_id=window_id
    )

    shards: list[DeltaShard] = []
    for descriptor in manifest["shards"]:
        shard_bytes = backend.get(descriptor["shard_key"])
        if shard_bytes is None:
            raise DeltaStorageShardMissing(
                f"shard {descriptor['tensor_name']!r} missing at {descriptor['shard_key']}"
            )
        if len(shard_bytes) != descriptor["byte_length"]:
            raise DeltaStorageShardCorrupted(
                f"shard {descriptor['tensor_name']!r} byte_length mismatch: "
                f"got {len(shard_bytes)} expected {descriptor['byte_length']}"
            )
        actual_sha = hashlib.sha256(shard_bytes).hexdigest()
        if actual_sha != descriptor["payload_sha256"]:
            raise DeltaStorageShardCorrupted(
                f"shard {descriptor['tensor_name']!r} sha256 mismatch: "
                f"computed={actual_sha} claimed={descriptor['payload_sha256']}"
            )
        shards.append(
            DeltaShard(
                tensor_name=descriptor["tensor_name"],
                dtype=descriptor["dtype"],
                shape=tuple(descriptor["shape"]),
                scale=float(descriptor["scale"]),
                zero_point=int(descriptor["zero_point"]),
                payload_sha256=descriptor["payload_sha256"],
                data_bytes=shard_bytes,
            )
        )

    recomputed_root = _merkle_root_hex([s.payload_sha256 for s in shards])
    if recomputed_root != manifest["merkle_root_hex"]:
        raise DeltaMerkleMismatch(
            f"merkle root mismatch on fetch: computed={recomputed_root} "
            f"claimed={manifest['merkle_root_hex']}"
        )

    return DeltaBundle(
        from_checkpoint_hash=manifest["from_checkpoint_hash"],
        to_checkpoint_hash=manifest["to_checkpoint_hash"],
        shards=tuple(shards),
        merkle_root_hex=manifest["merkle_root_hex"],
    )


def publish_full_snapshot(
    state_dict: Mapping[str, "torch.Tensor"],  # noqa: F821
    *,
    run_id: str,
    window_id: int,
    backend: CheckpointStorageBackend,
    dtype: str = "fp16",
) -> CheckpointManifest:
    """Publish a full state_dict as a 1+K object set under the window prefix."""
    import torch

    _validate_run_id(run_id)
    _validate_window_id(window_id)
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if dtype not in dtype_map:
        raise DeltaStorageError(f"unsupported snapshot dtype {dtype!r}")
    torch_dtype = dtype_map[dtype]

    sorted_names = sorted(state_dict.keys())
    for name in sorted_names:
        _validate_tensor_name(name)

    shard_descriptors: list[dict] = []
    for name in sorted_names:
        tensor = state_dict[name].detach().to("cpu").to(torch_dtype).contiguous()
        data_bytes = tensor.numpy().tobytes()
        shard_key = checkpoint_shard_key(run_id, window_id, name)
        backend.put(shard_key, data_bytes)
        shard_descriptors.append(
            {
                "tensor_name": name,
                "dtype": dtype,
                "shape": list(tensor.shape),
                "scale": 0.0,
                "zero_point": 0,
                "payload_sha256": hashlib.sha256(data_bytes).hexdigest(),
                "byte_length": len(data_bytes),
                "shard_key": shard_key,
            }
        )

    merkle_root = _merkle_root_hex([d["payload_sha256"] for d in shard_descriptors])
    snapshot_hash = hashlib.sha256(b"|".join(
        d["payload_sha256"].encode() for d in shard_descriptors
    )).hexdigest()

    manifest_dict = {
        "version": DELTA_STORAGE_VERSION,
        "kind": DELTA_STORAGE_KIND_FULL,
        "run_id": run_id,
        "window_id": window_id,
        "from_checkpoint_hash": "",
        "to_checkpoint_hash": snapshot_hash,
        "merkle_root_hex": merkle_root,
        "shards": shard_descriptors,
    }
    manifest_bytes = _canonical_manifest_bytes(manifest_dict)
    manifest_key = checkpoint_manifest_key(run_id, window_id)
    backend.put(manifest_key, manifest_bytes)
    backend.put(
        checkpoint_kind_marker_key(run_id, window_id, DELTA_STORAGE_KIND_FULL),
        b"",
    )

    return CheckpointManifest(
        run_id=run_id,
        window_id=window_id,
        kind=DELTA_STORAGE_KIND_FULL,
        manifest_key=manifest_key,
        shard_keys=tuple(d["shard_key"] for d in shard_descriptors),
        merkle_root_hex=merkle_root,
        published_at=time.time(),
    )


def fetch_full_snapshot(
    run_id: str,
    window_id: int,
    *,
    backend: CheckpointStorageBackend,
) -> dict[str, "torch.Tensor"]:  # noqa: F821
    """Fetch a full state_dict, verifying every shard sha256 before returning."""
    import numpy as np
    import torch

    _validate_run_id(run_id)
    _validate_window_id(window_id)
    manifest_key = checkpoint_manifest_key(run_id, window_id)
    manifest_bytes = backend.get(manifest_key)
    if manifest_bytes is None:
        raise DeltaStorageNotFound(
            f"no manifest at {manifest_key} (run_id={run_id} window_id={window_id})"
        )
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeltaStorageManifestInvalid(f"manifest decode failed: {exc}") from exc

    _validate_manifest_structure(
        manifest, expected_kind=DELTA_STORAGE_KIND_FULL, run_id=run_id, window_id=window_id
    )

    dtype_map = {
        "fp16": (np.float16, torch.float16),
        "bf16": (None, torch.bfloat16),
        "fp32": (np.float32, torch.float32),
    }

    fetched_bytes: dict[str, bytes] = {}
    for descriptor in manifest["shards"]:
        shard_bytes = backend.get(descriptor["shard_key"])
        if shard_bytes is None:
            raise DeltaStorageShardMissing(
                f"snapshot shard {descriptor['tensor_name']!r} missing at {descriptor['shard_key']}"
            )
        if len(shard_bytes) != descriptor["byte_length"]:
            raise DeltaStorageShardCorrupted(
                f"snapshot shard {descriptor['tensor_name']!r} byte_length mismatch"
            )
        actual_sha = hashlib.sha256(shard_bytes).hexdigest()
        if actual_sha != descriptor["payload_sha256"]:
            raise DeltaStorageShardCorrupted(
                f"snapshot shard {descriptor['tensor_name']!r} sha256 mismatch"
            )
        fetched_bytes[descriptor["tensor_name"]] = shard_bytes

    payload_shas = [
        d["payload_sha256"]
        for d in sorted(manifest["shards"], key=lambda x: x["tensor_name"])
    ]
    recomputed_root = _merkle_root_hex(payload_shas)
    if recomputed_root != manifest["merkle_root_hex"]:
        raise DeltaMerkleMismatch(
            f"snapshot merkle root mismatch: computed={recomputed_root} "
            f"claimed={manifest['merkle_root_hex']}"
        )

    result: dict[str, torch.Tensor] = {}
    for descriptor in manifest["shards"]:
        dtype_name = descriptor["dtype"]
        if dtype_name not in dtype_map:
            raise DeltaStorageManifestInvalid(
                f"snapshot shard {descriptor['tensor_name']!r} unsupported dtype {dtype_name!r}"
            )
        np_dtype, torch_dtype = dtype_map[dtype_name]
        shape = tuple(descriptor["shape"])
        raw = fetched_bytes[descriptor["tensor_name"]]
        if np_dtype is not None:
            array = np.frombuffer(raw, dtype=np_dtype).reshape(shape).copy()
            tensor = torch.from_numpy(array).to(torch_dtype)
        else:
            array = np.frombuffer(raw, dtype=np.uint8).reshape(-1).copy()
            tensor = torch.from_numpy(array).view(torch.bfloat16).reshape(shape).clone()
        result[descriptor["tensor_name"]] = tensor
    return result


@dataclass
class SnapshotScheduler:
    """Decide full-vs-delta based on window cadence."""

    cadence_windows: int = DEFAULT_FULL_SNAPSHOT_CADENCE_WINDOWS
    last_full_window: int | None = None
    _last_decision_window: int | None = field(default=None, init=False, repr=False)

    def decide(self, current_window: int, *, force_full: bool = False) -> SnapshotDecision:
        if not isinstance(current_window, int) or current_window < 0:
            raise DeltaStorageError(
                f"current_window must be a non-negative int, got {current_window!r}"
            )
        if self.cadence_windows <= 0:
            raise DeltaStorageError(
                f"cadence_windows must be positive, got {self.cadence_windows!r}"
            )
        if force_full:
            return SnapshotDecision(kind=DELTA_STORAGE_KIND_FULL, reason="forced")
        if self.last_full_window is None:
            return SnapshotDecision(kind=DELTA_STORAGE_KIND_FULL, reason="bootstrap")
        if current_window - self.last_full_window >= self.cadence_windows:
            return SnapshotDecision(kind=DELTA_STORAGE_KIND_FULL, reason="cadence")
        return SnapshotDecision(kind=DELTA_STORAGE_KIND_DELTA, reason="in-cadence")

    def record_publish(self, window_id: int, *, kind: str) -> None:
        _validate_window_id(window_id)
        if kind == DELTA_STORAGE_KIND_FULL:
            self.last_full_window = window_id
        elif kind != DELTA_STORAGE_KIND_DELTA:
            raise DeltaStorageError(f"unknown kind {kind!r}")
        self._last_decision_window = window_id


__all__ = [
    "DELTA_STORAGE_VERSION",
    "DELTA_STORAGE_MANIFEST_FILENAME",
    "DELTA_STORAGE_KIND_DELTA",
    "DELTA_STORAGE_KIND_FULL",
    "DEFAULT_FULL_SNAPSHOT_CADENCE_WINDOWS",
    "CheckpointStorageBackend",
    "LocalFilesystemBackend",
    "RecordingBackend",
    "CheckpointManifest",
    "SnapshotDecision",
    "SnapshotScheduler",
    "DeltaStorageError",
    "DeltaStorageNotFound",
    "DeltaStorageManifestInvalid",
    "DeltaStorageShardMissing",
    "DeltaStorageShardCorrupted",
    "DeltaStorageInvalidName",
    "DeltaStorageInvalidRunId",
    "checkpoint_window_prefix",
    "checkpoint_manifest_key",
    "checkpoint_shard_key",
    "checkpoint_kind_marker_key",
    "publish_bundle",
    "fetch_bundle",
    "publish_full_snapshot",
    "fetch_full_snapshot",
]
