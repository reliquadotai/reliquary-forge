"""Shard-parallel checkpoint fetcher.

Drop-in replacement for ``fetch_bundle`` that downloads every shard
concurrently via a thread pool. Safe for I/O-bound backends (R2, S3,
HTTP) — the GIL doesn't hurt because the worker threads spend their
time inside the backend client's socket read path.

Invariants are identical to the sequential ``fetch_bundle``:
- Manifest parsed first; schema validated.
- Every shard's byte_length + sha256 checked before inclusion.
- Merkle root recomputed over shards in sorted-by-tensor-name order
  and compared against the manifest's claimed root.
- Fail-before-mutate: any shard / Merkle failure raises before any
  DeltaBundle is materialized.

Spec: private/reliquary-plan/notes/spec-delta-storage.md invariant 7.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .checkpoint_storage import (
    CheckpointStorageBackend,
    DELTA_STORAGE_KIND_DELTA,
    DeltaStorageManifestInvalid,
    DeltaStorageNotFound,
    DeltaStorageShardCorrupted,
    DeltaStorageShardMissing,
    _validate_manifest_structure,
    _validate_run_id,
    _validate_window_id,
    checkpoint_manifest_key,
)
from .delta_checkpoints import (
    DeltaBundle,
    DeltaMerkleMismatch,
    DeltaShard,
    _merkle_root_hex,
)


DEFAULT_PARALLEL_WORKERS = 8


@dataclass(frozen=True)
class _FetchedShard:
    descriptor: dict
    data: bytes


def _fetch_one_shard(
    backend: CheckpointStorageBackend, descriptor: dict
) -> _FetchedShard:
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
    return _FetchedShard(descriptor=descriptor, data=shard_bytes)


def fetch_bundle_parallel(
    run_id: str,
    window_id: int,
    *,
    backend: CheckpointStorageBackend,
    max_workers: int = DEFAULT_PARALLEL_WORKERS,
) -> DeltaBundle:
    """Identical contract to ``fetch_bundle`` but downloads shards in parallel.

    Correctness notes:
    - ThreadPoolExecutor.map preserves the input iteration order when the
      manifest is already sorted-by-tensor-name; we sort deterministically
      before merkle recompute so the root match is unaffected by thread
      completion ordering.
    - Any worker raising propagates immediately (the default
      ``map`` behavior), aborting the fetch before any DeltaBundle is
      assembled — matches the sequential fetcher's fail-before-mutate.
    - ``max_workers`` defaults to 8 (matches the PRD target); callers
      tuning for bandwidth-limited links pass a smaller number.
    """
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")

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
        manifest,
        expected_kind=DELTA_STORAGE_KIND_DELTA,
        run_id=run_id,
        window_id=window_id,
    )

    descriptors = manifest["shards"]
    if not descriptors:
        # Manifest with no shards — treat as an empty bundle. Matches the
        # sequential path's behavior: empty shard list → empty tuple,
        # Merkle root over the empty list is sha256("").
        recomputed_root = _merkle_root_hex([])
        if recomputed_root != manifest["merkle_root_hex"]:
            raise DeltaMerkleMismatch(
                f"merkle root mismatch on fetch: computed={recomputed_root} "
                f"claimed={manifest['merkle_root_hex']}"
            )
        return DeltaBundle(
            from_checkpoint_hash=manifest["from_checkpoint_hash"],
            to_checkpoint_hash=manifest["to_checkpoint_hash"],
            shards=tuple(),
            merkle_root_hex=manifest["merkle_root_hex"],
        )

    workers = min(max_workers, len(descriptors))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fetched = list(pool.map(lambda d: _fetch_one_shard(backend, d), descriptors))

    # Sort by tensor_name for deterministic Merkle recompute (matches
    # the publish-side invariant).
    fetched_sorted = sorted(fetched, key=lambda f: f.descriptor["tensor_name"])

    shards = tuple(
        DeltaShard(
            tensor_name=f.descriptor["tensor_name"],
            dtype=f.descriptor["dtype"],
            shape=tuple(f.descriptor["shape"]),
            scale=float(f.descriptor["scale"]),
            zero_point=int(f.descriptor["zero_point"]),
            payload_sha256=f.descriptor["payload_sha256"],
            data_bytes=f.data,
        )
        for f in fetched_sorted
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
        shards=shards,
        merkle_root_hex=manifest["merkle_root_hex"],
    )


__all__ = [
    "DEFAULT_PARALLEL_WORKERS",
    "fetch_bundle_parallel",
]
