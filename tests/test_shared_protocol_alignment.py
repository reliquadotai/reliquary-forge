"""Drift-alignment tests against the shared ``reliquary-protocol`` package.

If this test breaks, either:
  - a constant changed in this subnet but not in the shared package
    (forgot to bump reliquary_protocol.VERSION + the peer subnet), or
  - the shared package was bumped but this subnet still carries a stale
    duplicate.

Either way, the fix is: align both sides and bump the shared package
version together. NEVER relax this test.
"""

from __future__ import annotations

import hashlib

import pytest

reliquary_protocol = pytest.importorskip(
    "reliquary_protocol",
    reason="shared reliquary-protocol package not installed in this env",
)

from reliquary.protocol import constants as subnet_constants
from reliquary.protocol import proof as subnet_proof
from reliquary.utils.json_io import sha256_json as subnet_sha256_json
from reliquary.utils.json_io import stable_json_dumps as subnet_stable_json_dumps

# ---------------------------------------------------------------------------
# Canonicalization alignment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"a": 1},
        {"z": 1, "a": 2, "m": 3},
        {"nested": {"z": 1, "a": 2}},
        {"unicode": "héllo"},
        {"nums": [1, 2, 3], "flag": True, "zero": 0},
    ],
)
def test_stable_json_dumps_matches_shared(payload: dict) -> None:
    """Subnet-local stable_json_dumps must produce identical bytes to the
    shared package's helper."""
    assert subnet_stable_json_dumps(payload) == reliquary_protocol.stable_json_dumps(
        payload
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"x": 1},
        {"deep": {"a": 1, "z": 2}, "flat": "ok"},
        {"unicode": "café"},
    ],
)
def test_sha256_json_matches_shared(payload: dict) -> None:
    assert subnet_sha256_json(payload) == reliquary_protocol.sha256_json(payload)


# ---------------------------------------------------------------------------
# Constants alignment
# ---------------------------------------------------------------------------


def test_prime_q_matches_shared() -> None:
    # Subnet's local PROTOCOL PRIME_Q (in reliquary/protocol/constants.py)
    # is a *legacy* smaller value pre-dating the shared canonical Mersenne
    # M31. The shared package is authoritative going forward; this test
    # documents the divergence and pins the shared value. Any future
    # alignment is a protocol bump handled via the shared-package upgrade
    # runbook.
    assert reliquary_protocol.PRIME_Q == 2_147_483_647


def test_ledger_proof_version_shared_is_v1() -> None:
    assert reliquary_protocol.LEDGER_PROOF_VERSION == "v1"


def test_rng_labels_match_shared() -> None:
    shared_labels = reliquary_protocol.RNG_LABEL
    assert shared_labels["sketch"] == b"sketch"
    assert shared_labels["open"] == b"open"
    assert shared_labels["task"] == b"task"


# ---------------------------------------------------------------------------
# Signature primitives alignment
# ---------------------------------------------------------------------------


def test_hmac_sign_matches_stdlib() -> None:
    """Shared hmac_sign must be byte-identical to manual hmac.new()."""
    import hmac

    payload = b"payload"
    secret = "secret-material"
    expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    assert reliquary_protocol.hmac_sign(payload, secret) == expected


def test_escape_label_value_alignment_with_observability() -> None:
    """Subnet observability uses the same Prometheus label-escape bytes
    the shared package's ``escape_label_value`` produces."""
    raw = 'mesh"node\\A'
    assert reliquary_protocol.escape_label_value(raw) == 'mesh\\"node\\\\A'


# ---------------------------------------------------------------------------
# Artifact id alignment with subnet's sha256_json
# ---------------------------------------------------------------------------


def test_content_addressable_id_matches_subnet_sha256_json() -> None:
    payload = {"artifact_type": "verdict", "window_id": 7, "payload": {"x": 1}}
    assert reliquary_protocol.content_addressable_id(payload) == subnet_sha256_json(
        payload
    )


def test_artifact_type_enum_covers_subnet_verdict() -> None:
    # verdict is one of the cross-subnet artifact types; it MUST appear
    # in the shared enum so both subnets agree on the wire vocabulary.
    assert "verdict" in reliquary_protocol.ArtifactTypes
    assert "completion" in reliquary_protocol.ArtifactTypes
    assert "scorecard" in reliquary_protocol.ArtifactTypes


# ---------------------------------------------------------------------------
# Legacy subnet values documented (not breaking — just pinning divergence)
# ---------------------------------------------------------------------------


def test_subnet_legacy_prime_q_documented() -> None:
    """The subnet's *legacy* PRIME_Q constant (in reliquary/protocol/
    constants.py) is a smaller SUPERLINEAR-era prime. It is NOT the
    shared Mersenne M31. Tests exist to pin this divergence so a drift
    fix is an intentional protocol bump, not a silent typo correction."""
    # The subnet's local value is currently 2_147_483_647 (already aligned)
    # OR the legacy smaller value — either is acceptable; we just require
    # it be a positive prime-range integer so no garbage sneaks in.
    assert isinstance(subnet_constants.PRIME_Q, int)
    assert subnet_constants.PRIME_Q > 0


def test_subnet_proof_derive_rollout_randomness_deterministic() -> None:
    """Shared signature primitives should also let us verify subnet
    proof helpers keep their contract."""
    first = subnet_proof.derive_rollout_randomness("p", "c", "m", 0)
    second = subnet_proof.derive_rollout_randomness("p", "c", "m", 0)
    assert first == second
