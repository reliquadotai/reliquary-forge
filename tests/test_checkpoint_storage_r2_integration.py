"""Integration test: publish_bundle + fetch_bundle over a real R2 bucket.

Skipped unless RELIQUARY_R2_CF_API_TOKEN is set. Run locally via:

  set -a && source .env.local && set +a
  PYTHONPATH=../reliquary-protocol pytest tests/test_checkpoint_storage_r2_integration.py -v

Proves gap #11 closed on the Forge side: delta-checkpoint publishing
writes + reads real bytes through the shared R2 backend in
reliquary-protocol.
"""

from __future__ import annotations

import hashlib
import os
import time

import pytest

_R2_TOKEN_SET = bool(os.environ.get("RELIQUARY_R2_CF_API_TOKEN"))
pytestmark = pytest.mark.skipif(
    not _R2_TOKEN_SET,
    reason="RELIQUARY_R2_CF_API_TOKEN not set; source .env.local to run live R2 tests",
)

from reliquary.training.checkpoint_storage import (
    checkpoint_shard_key,
    fetch_bundle,
    publish_bundle,
)
from reliquary.training.delta_checkpoints import DeltaBundle, DeltaShard, _merkle_root_hex

try:  # pragma: no cover
    from reliquary_protocol.storage import R2ObjectBackend
except ImportError:  # pragma: no cover
    R2ObjectBackend = None  # type: ignore[assignment]


TEST_RUN_ID = "r2-int-test"


def _bundle(n_shards: int = 3) -> DeltaBundle:
    """Build a deterministic small DeltaBundle with N int8 shards."""
    shards: list[DeltaShard] = []
    for i in range(n_shards):
        name = f"layer.{i}.weight"
        data = bytes([i, (i + 7) % 256, (i * 31) % 256]) * 4  # 12 bytes
        shards.append(
            DeltaShard(
                tensor_name=name,
                dtype="int8",
                shape=(3, 4),
                scale=0.001 * (i + 1),
                zero_point=0,
                payload_sha256=hashlib.sha256(data).hexdigest(),
                data_bytes=data,
            )
        )
    merkle = _merkle_root_hex([s.payload_sha256 for s in shards])
    return DeltaBundle(
        from_checkpoint_hash="a" * 64,
        to_checkpoint_hash="b" * 64,
        shards=tuple(shards),
        merkle_root_hex=merkle,
    )


def _backend():
    """Backend configured exactly as the Forge publisher uses it (r2.dev on)."""
    assert R2ObjectBackend is not None, "reliquary-protocol not installed / importable"
    return R2ObjectBackend.from_env()


def _consistent_backend():
    """Backend with r2.dev disabled — for post-write verifications where
    CDN eventual-consistency would otherwise mask fresh writes. Tests that
    write and immediately read the SAME key must use this."""
    assert R2ObjectBackend is not None
    return R2ObjectBackend(
        account_id=os.environ["RELIQUARY_R2_ACCOUNT_ID"],
        bucket=os.environ["RELIQUARY_R2_BUCKET"],
        cf_api_token=os.environ["RELIQUARY_R2_CF_API_TOKEN"],
        public_url=None,
    )


def _cleanup(backend, prefix: str) -> None:
    for key in backend.list(prefix):
        backend.delete(key)


def test_delta_bundle_round_trip_over_r2() -> None:
    """publish → fetch → every shard byte matches + merkle root intact."""
    backend = _backend()
    window_id = 900_000 + int(time.time() % 1000)
    prefix = f"checkpoints/{TEST_RUN_ID}/{window_id}/"
    _cleanup(backend, prefix)

    original = _bundle(n_shards=4)
    manifest = publish_bundle(
        original, run_id=TEST_RUN_ID, window_id=window_id, backend=backend,
    )
    try:
        assert manifest.merkle_root_hex == original.merkle_root_hex
        assert manifest.kind == "delta"
        assert len(manifest.shard_keys) == len(original.shards)

        fetched = fetch_bundle(TEST_RUN_ID, window_id, backend=backend)

        assert fetched.merkle_root_hex == original.merkle_root_hex
        assert fetched.from_checkpoint_hash == original.from_checkpoint_hash
        assert fetched.to_checkpoint_hash == original.to_checkpoint_hash
        assert len(fetched.shards) == len(original.shards)

        by_name_original = {s.tensor_name: s for s in original.shards}
        by_name_fetched = {s.tensor_name: s for s in fetched.shards}
        for name, fetched_shard in by_name_fetched.items():
            original_shard = by_name_original[name]
            assert fetched_shard.data_bytes == original_shard.data_bytes
            assert fetched_shard.payload_sha256 == original_shard.payload_sha256
            assert fetched_shard.shape == original_shard.shape
    finally:
        _cleanup(backend, prefix)


# NOTE: corruption-detection on a fresh-key fetch is exhaustively
# covered by tests/test_checkpoint_storage.py against the local +
# recording backends. We deliberately do NOT re-test it here because
# tampering requires overwriting an existing R2 key — and the
# Cloudflare REST API edge-caches R2 GETs for up to 4 hours, so a
# read-immediately-after-overwrite returns stale bytes during the test
# window. This is a non-issue in production: the pipeline NEVER
# overwrites — every verdict, manifest, and shard lives at a unique
# content-addressable path (window_id × validator × completion_id or
# window_id × tensor_name), so the first write is always authoritative.
