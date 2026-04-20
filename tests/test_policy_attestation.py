"""Tests for :mod:`reliquary.training.policy_attestation`."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from reliquary.training.checkpoint_storage import (
    DeltaStorageNotFound,
    LocalFilesystemBackend,
    publish_bundle,
)
from reliquary.training.delta_checkpoints import compute_delta
from reliquary.training.policy_attestation import (
    AttestationPublication,
    compute_smoke_hash,
    make_hmac_authority,
    publish_attestation,
    smoke_hash_from_manifest,
)
from reliquary_protocol import (
    CHECKPOINT_ATTESTATION_TYPE,
    CheckpointAttestation,
    HmacBridgeVerifier,
    PolicyCommitment,
    envelope_from_dict,
    verify_envelope,
    verify_policy_commitment,
)


AUTHORITY = "authority_fleet"
SECRET = "authority-fleet-secret"


def _signer():
    return make_hmac_authority(AUTHORITY, SECRET)


def _verifier():
    return HmacBridgeVerifier(secrets={AUTHORITY: SECRET})


def _tiny_delta_bundle():
    """Build a tiny 2-tensor delta bundle end-to-end for real merkle coverage."""
    torch.manual_seed(0)
    base = {"layer.0": torch.zeros(4, 4), "layer.1": torch.zeros(2, 2)}
    target = {
        "layer.0": torch.ones(4, 4) * 0.1,
        "layer.1": torch.ones(2, 2) * 0.2,
    }
    return compute_delta(base, target)


def _publish_a_bundle(tmp_path, run_id="run-0001", window_id=5):
    backend = LocalFilesystemBackend(tmp_path / "root")
    bundle = _tiny_delta_bundle()
    manifest = publish_bundle(
        bundle, run_id=run_id, window_id=window_id, backend=backend
    )
    return backend, bundle, manifest


# ---------------------------------------------------------------------------
# Smoke hash computation
# ---------------------------------------------------------------------------


def test_compute_smoke_hash_deterministic():
    manifest_bytes = b"{\"shards\":[]}"
    digests = ("a" * 64, "b" * 64, "c" * 64)
    h1 = compute_smoke_hash(manifest_bytes=manifest_bytes, shard_payload_sha256s=digests)
    h2 = compute_smoke_hash(
        manifest_bytes=manifest_bytes, shard_payload_sha256s=list(reversed(digests))
    )
    assert h1 == h2
    assert len(h1) == 64


def test_compute_smoke_hash_manifest_sensitive():
    digests = ("a" * 64,)
    h1 = compute_smoke_hash(manifest_bytes=b"a", shard_payload_sha256s=digests)
    h2 = compute_smoke_hash(manifest_bytes=b"b", shard_payload_sha256s=digests)
    assert h1 != h2


def test_compute_smoke_hash_digest_sensitive():
    manifest = b"mf"
    h1 = compute_smoke_hash(manifest_bytes=manifest, shard_payload_sha256s=("a" * 64,))
    h2 = compute_smoke_hash(manifest_bytes=manifest, shard_payload_sha256s=("b" * 64,))
    assert h1 != h2


def test_compute_smoke_hash_rejects_non_64_hex():
    with pytest.raises(ValueError):
        compute_smoke_hash(manifest_bytes=b"mf", shard_payload_sha256s=("short",))


def test_smoke_hash_from_manifest_matches_explicit():
    manifest = {
        "version": "v1",
        "shards": [
            {"payload_sha256": "a" * 64},
            {"payload_sha256": "b" * 64},
        ],
    }
    mb = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h1 = smoke_hash_from_manifest(mb)
    h2 = compute_smoke_hash(manifest_bytes=mb, shard_payload_sha256s=["a" * 64, "b" * 64])
    assert h1 == h2


# ---------------------------------------------------------------------------
# publish_attestation
# ---------------------------------------------------------------------------


def test_publish_attestation_writes_both_artifacts(tmp_path):
    backend, _, manifest = _publish_a_bundle(tmp_path)

    pub = publish_attestation(
        backend=backend,
        signer=_signer(),
        training_netuid=3,
        inference_netuid=81,
        checkpoint_manifest=manifest,
        base_snapshot_window_id=0,
        consumed_rollout_windows=[10, 11, 12],
        consumed_rollout_keys=[
            "rollouts/81/10-10/val.json",
            "rollouts/81/11-11/val.json",
            "rollouts/81/12-12/val.json",
        ],
        effective_at_ledger_window=100,
    )

    assert isinstance(pub, AttestationPublication)
    assert backend.get(pub.attestation_key) is not None
    assert backend.get(pub.commitment_key) is not None


def test_attestation_envelope_verifies(tmp_path):
    backend, _, manifest = _publish_a_bundle(tmp_path)
    pub = publish_attestation(
        backend=backend,
        signer=_signer(),
        training_netuid=3,
        inference_netuid=81,
        checkpoint_manifest=manifest,
        base_snapshot_window_id=0,
        consumed_rollout_windows=[10],
        consumed_rollout_keys=["rollouts/81/10-10/val.json"],
        effective_at_ledger_window=100,
    )
    envelope = envelope_from_dict(
        json.loads(backend.get(pub.attestation_key).decode("utf-8"))
    )
    assert envelope.artifact_type == CHECKPOINT_ATTESTATION_TYPE
    assert verify_envelope(envelope, _verifier()) is True


def test_commitment_verifies(tmp_path):
    backend, _, manifest = _publish_a_bundle(tmp_path)
    pub = publish_attestation(
        backend=backend,
        signer=_signer(),
        training_netuid=3,
        inference_netuid=81,
        checkpoint_manifest=manifest,
        base_snapshot_window_id=0,
        consumed_rollout_windows=[10],
        consumed_rollout_keys=["rollouts/81/10-10/val.json"],
        effective_at_ledger_window=100,
    )
    raw = backend.get(pub.commitment_key)
    commitment = PolicyCommitment.from_dict(json.loads(raw.decode("utf-8")))
    assert verify_policy_commitment(commitment, _verifier()) is True
    assert commitment.effective_at_window == 100
    assert commitment.attestation_id == pub.attestation.artifact_id()


def test_attestation_merkle_root_matches_bundle(tmp_path):
    backend, bundle, manifest = _publish_a_bundle(tmp_path)
    pub = publish_attestation(
        backend=backend,
        signer=_signer(),
        training_netuid=3,
        inference_netuid=81,
        checkpoint_manifest=manifest,
        base_snapshot_window_id=0,
        consumed_rollout_windows=[10],
        consumed_rollout_keys=["rollouts/81/10-10/val.json"],
        effective_at_ledger_window=100,
    )
    assert pub.attestation.merkle_root_hex == bundle.merkle_root_hex


def test_smoke_hash_matches_shard_payload_sha256s(tmp_path):
    """Smoke hash must round-trip against the actual published manifest."""
    backend, bundle, manifest = _publish_a_bundle(tmp_path)
    pub = publish_attestation(
        backend=backend,
        signer=_signer(),
        training_netuid=3,
        inference_netuid=81,
        checkpoint_manifest=manifest,
        base_snapshot_window_id=0,
        consumed_rollout_windows=[10],
        consumed_rollout_keys=["rollouts/81/10-10/val.json"],
        effective_at_ledger_window=100,
    )
    # Re-fetch the manifest bytes + recompute smoke hash → must match.
    mf = backend.get(f"checkpoints/{manifest.run_id}/{manifest.window_id}/manifest.json")
    parsed = json.loads(mf.decode("utf-8"))
    digests = [entry["payload_sha256"] for entry in parsed["shards"]]
    recomputed = compute_smoke_hash(manifest_bytes=mf, shard_payload_sha256s=digests)
    assert recomputed == pub.smoke_hash_hex
    assert pub.attestation.smoke_hash_hex == recomputed


def test_missing_manifest_raises(tmp_path):
    backend = LocalFilesystemBackend(tmp_path / "root")
    # Craft a manifest object that points at a non-existent (run_id, window_id).
    from reliquary.training.checkpoint_storage import CheckpointManifest
    fake_manifest = CheckpointManifest(
        run_id="no-such-run",
        window_id=999,
        kind="delta",
        manifest_key="checkpoints/no-such-run/999/manifest.json",
        shard_keys=(),
        merkle_root_hex="a" * 64,
        published_at=0.0,
    )
    with pytest.raises(DeltaStorageNotFound):
        publish_attestation(
            backend=backend,
            signer=_signer(),
            training_netuid=3,
            inference_netuid=81,
            checkpoint_manifest=fake_manifest,
            base_snapshot_window_id=0,
            consumed_rollout_windows=[10],
            consumed_rollout_keys=["rollouts/81/10-10/val.json"],
            effective_at_ledger_window=100,
        )


def test_consumed_rollout_windows_sorted(tmp_path):
    backend, _, manifest = _publish_a_bundle(tmp_path)
    pub = publish_attestation(
        backend=backend,
        signer=_signer(),
        training_netuid=3,
        inference_netuid=81,
        checkpoint_manifest=manifest,
        base_snapshot_window_id=0,
        consumed_rollout_windows=[12, 10, 11],  # unsorted
        consumed_rollout_keys=[
            "rollouts/81/12-12/val.json",
            "rollouts/81/10-10/val.json",
            "rollouts/81/11-11/val.json",
        ],
        effective_at_ledger_window=100,
    )
    assert pub.attestation.consumed_rollout_windows == (10, 11, 12)


def test_storage_keys_follow_convention(tmp_path):
    backend, _, manifest = _publish_a_bundle(tmp_path, run_id="abc", window_id=7)
    pub = publish_attestation(
        backend=backend,
        signer=_signer(),
        training_netuid=3,
        inference_netuid=81,
        checkpoint_manifest=manifest,
        base_snapshot_window_id=0,
        consumed_rollout_windows=[10],
        consumed_rollout_keys=["rollouts/81/10-10/val.json"],
        effective_at_ledger_window=200,
    )
    assert pub.attestation_key == "attestations/3/abc/7.json"
    assert pub.commitment_key == "commitments/81/policy/200.json"


def test_two_publishes_same_manifest_byte_identical_smoke_hash(tmp_path):
    """Two publications of the same bundle produce the same smoke hash."""
    backend_a, _, manifest_a = _publish_a_bundle(tmp_path / "a")
    backend_b, _, manifest_b = _publish_a_bundle(tmp_path / "b")
    assert manifest_a.merkle_root_hex == manifest_b.merkle_root_hex
    pub_a = publish_attestation(
        backend=backend_a,
        signer=_signer(),
        training_netuid=3,
        inference_netuid=81,
        checkpoint_manifest=manifest_a,
        base_snapshot_window_id=0,
        consumed_rollout_windows=[10],
        consumed_rollout_keys=["rollouts/81/10-10/val.json"],
        effective_at_ledger_window=100,
    )
    pub_b = publish_attestation(
        backend=backend_b,
        signer=_signer(),
        training_netuid=3,
        inference_netuid=81,
        checkpoint_manifest=manifest_b,
        base_snapshot_window_id=0,
        consumed_rollout_windows=[10],
        consumed_rollout_keys=["rollouts/81/10-10/val.json"],
        effective_at_ledger_window=100,
    )
    assert pub_a.smoke_hash_hex == pub_b.smoke_hash_hex
