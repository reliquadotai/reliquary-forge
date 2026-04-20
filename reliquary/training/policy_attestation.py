"""Cross-subnet attestation — Forge → Ledger direction of the bridge.

Once a delta checkpoint has been produced by the training lane and
published via :func:`reliquary.training.checkpoint_storage.publish_bundle`,
the policy authority calls :func:`publish_attestation` here to emit two
linked artifacts:

1. :class:`~reliquary_protocol.CheckpointAttestation` — the provenance
   envelope binding the checkpoint's merkle root to the exact Ledger
   rollout windows that fed this training cycle. Gives any third party
   a complete onchain-anchored chain of custody from any checkpoint
   back to its sources.
2. :class:`~reliquary_protocol.PolicyCommitment` — the pointer-level
   artifact at ``commitments/<inference_netuid>/policy/<effective>.json``
   that Ledger miners poll to know "swap to this checkpoint at this
   window." In prod this is onchain via ``commit_policy_metadata``;
   during staging we simulate with R2 at the well-known prefix.

Smoke hash protocol (must stay byte-identical with
:func:`reliquary_inference.shared.policy_consumer.default_smoke_runner`):

  ``sha256(manifest_bytes || concat(sorted shard payload_sha256 hex))``

The Ledger side recomputes this on its side of the wire and refuses to
flip to the new policy if the hash disagrees. This is a corruption-in-
transit signal, not an anti-forgery signal: the real anti-forgery comes
from the Ledger's ``BridgeVerifier`` refusing any commitment not signed
by an allowlisted ``policy_authority`` hotkey.

Spec: ``private/reliquary-plan/notes/spec-closed-loop-bridge.md``
§6 (Forge cross-subnet attestation).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from reliquary_protocol import (
    BridgeSigner,
    CHECKPOINT_ATTESTATION_TYPE,
    CheckpointAttestation,
    HmacBridgeSigner,
    POLICY_COMMITMENT_TYPE,
    PolicyCommitment,
    build_checkpoint_attestation,
    build_policy_commitment,
    sign_envelope,
)

from .checkpoint_storage import (
    CheckpointManifest,
    CheckpointStorageBackend,
    DeltaStorageNotFound,
    checkpoint_manifest_key,
)


logger = logging.getLogger(__name__)


ATTESTATION_METRIC_PUBLISHED = "reliquary_bridge_attestation_published_total"
ATTESTATION_METRIC_FAILED = "reliquary_bridge_attestation_failed_total"


@dataclass(frozen=True)
class AttestationPublication:
    """Return value of :func:`publish_attestation`."""

    attestation: CheckpointAttestation
    attestation_key: str
    commitment: PolicyCommitment
    commitment_key: str
    smoke_hash_hex: str


def compute_smoke_hash(
    *, manifest_bytes: bytes, shard_payload_sha256s: Sequence[str]
) -> str:
    """Compute the deterministic smoke hash for a published checkpoint.

    MUST match the Ledger ``default_smoke_runner`` byte-for-byte.

    Parameters:
      - ``manifest_bytes`` — the exact bytes that ``publish_bundle``
        wrote at the manifest key (canonical sorted-keys JSON).
      - ``shard_payload_sha256s`` — the 64-char hex payload digests
        from the ``shards`` section of the manifest. Order is
        irrelevant (the function sorts).
    """
    h = hashlib.sha256()
    h.update(manifest_bytes)
    for digest in sorted(shard_payload_sha256s):
        if len(digest) != 64:
            raise ValueError(
                f"shard sha256 digest must be 64 hex chars, got {digest!r}"
            )
        h.update(digest.encode("utf-8"))
    return h.hexdigest()


def smoke_hash_from_manifest(
    manifest_bytes: bytes,
) -> str:
    """Convenience: parse a manifest dict out of bytes and compute smoke hash.

    Manifest must be the canonical JSON with a ``shards: [{payload_sha256, ...}]``
    array — i.e. exactly what ``publish_bundle`` wrote.
    """
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    shards = manifest.get("shards") or []
    shas = [str(entry["payload_sha256"]) for entry in shards]
    return compute_smoke_hash(
        manifest_bytes=manifest_bytes, shard_payload_sha256s=shas
    )


def publish_attestation(
    *,
    backend: CheckpointStorageBackend,
    signer: BridgeSigner,
    training_netuid: int,
    inference_netuid: int,
    checkpoint_manifest: CheckpointManifest,
    base_snapshot_window_id: int,
    consumed_rollout_windows: Sequence[int],
    consumed_rollout_keys: Sequence[str],
    effective_at_ledger_window: int,
) -> AttestationPublication:
    """Sign + publish the attestation + commitment pair.

    Reads the manifest blob back from the backend to compute the smoke
    hash deterministically. Both artifacts are written with atomic
    backend ``put`` semantics.
    """
    manifest_key = checkpoint_manifest_key(
        checkpoint_manifest.run_id, checkpoint_manifest.window_id
    )
    manifest_bytes = backend.get(manifest_key)
    if manifest_bytes is None:
        raise DeltaStorageNotFound(
            f"attestation: missing manifest at {manifest_key}"
        )

    smoke_hex = smoke_hash_from_manifest(manifest_bytes)

    attestation = build_checkpoint_attestation(
        training_netuid=training_netuid,
        inference_netuid=inference_netuid,
        checkpoint_run_id=checkpoint_manifest.run_id,
        checkpoint_window_id=checkpoint_manifest.window_id,
        merkle_root_hex=checkpoint_manifest.merkle_root_hex,
        base_snapshot_window_id=base_snapshot_window_id,
        consumed_rollout_windows=list(consumed_rollout_windows),
        consumed_rollout_keys=list(consumed_rollout_keys),
        policy_authority_hotkey=signer.signer_id,
        smoke_hash_hex=smoke_hex,
    )

    envelope = sign_envelope(CHECKPOINT_ATTESTATION_TYPE, attestation, signer)
    attestation_key = attestation.storage_key()
    backend.put(attestation_key, envelope.canonical_bytes())

    commitment = build_policy_commitment(
        attestation=attestation,
        attestation_key=attestation_key,
        effective_at_window=int(effective_at_ledger_window),
        signer=signer,
    )
    commitment_key = commitment.storage_key()
    commitment_bytes = json.dumps(
        commitment.to_dict(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    backend.put(commitment_key, commitment_bytes)

    logger.info(
        "policy_attestation.publish ok run_id=%s window=%d effective_at=%d "
        "attestation_key=%s commitment_key=%s",
        checkpoint_manifest.run_id,
        checkpoint_manifest.window_id,
        effective_at_ledger_window,
        attestation_key,
        commitment_key,
    )

    return AttestationPublication(
        attestation=attestation,
        attestation_key=attestation_key,
        commitment=commitment,
        commitment_key=commitment_key,
        smoke_hash_hex=smoke_hex,
    )


def make_hmac_authority(hotkey: str, secret: str) -> HmacBridgeSigner:
    """Construct an HMAC-backed :class:`BridgeSigner` for tests / staging."""
    return HmacBridgeSigner(signer_id=hotkey, secret=secret)


__all__ = [
    "ATTESTATION_METRIC_FAILED",
    "ATTESTATION_METRIC_PUBLISHED",
    "AttestationPublication",
    "compute_smoke_hash",
    "make_hmac_authority",
    "publish_attestation",
    "smoke_hash_from_manifest",
]
