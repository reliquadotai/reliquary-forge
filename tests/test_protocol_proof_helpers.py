"""Comprehensive tests for reliquary.protocol.proof helpers.

The existing ``test_proof_helpers.py`` covers only index determinism.
This module pins the rest of the pure-Python surface (PRF, HMAC,
signature verify, rollout randomness, int_to_bytes) so regressions in
canonical material-derivation are caught at CI time rather than when
validators drift off consensus.
"""

from __future__ import annotations

import hashlib
import hmac
import struct

import pytest

from reliquary.protocol.constants import CHALLENGE_K, PRIME_Q, PROOF_TOLERANCE, RNG_LABEL
from reliquary.protocol.proof import (
    derive_rollout_randomness,
    indices_from_tokens,
    int_to_bytes,
    prf,
    sign_s_vals,
    verify_s_vals_signature,
)


# ---------------------------------------------------------------------------
# int_to_bytes
# ---------------------------------------------------------------------------


def test_int_to_bytes_zero_pads_to_four_bytes() -> None:
    assert int_to_bytes(0) == b"\x00\x00\x00\x00"


def test_int_to_bytes_big_endian_ordering() -> None:
    assert int_to_bytes(1) == b"\x00\x00\x00\x01"
    assert int_to_bytes(256) == b"\x00\x00\x01\x00"


def test_int_to_bytes_matches_struct_pack() -> None:
    for value in (0, 1, 2**16, 2**31 - 1, 0xFFFFFFFF):
        assert int_to_bytes(value) == struct.pack(">I", value & 0xFFFFFFFF)


def test_int_to_bytes_truncates_to_lower_32_bits() -> None:
    assert int_to_bytes(2**32) == b"\x00\x00\x00\x00"
    assert int_to_bytes(2**32 + 5) == b"\x00\x00\x00\x05"


# ---------------------------------------------------------------------------
# prf
# ---------------------------------------------------------------------------


def test_prf_is_deterministic() -> None:
    out_a = prf(b"label", b"a", b"b", out_bytes=16)
    out_b = prf(b"label", b"a", b"b", out_bytes=16)
    assert out_a == out_b


def test_prf_length_matches_out_bytes() -> None:
    for n in (1, 16, 32, 64, 100, 1024):
        assert len(prf(b"label", b"data", out_bytes=n)) == n


def test_prf_different_label_different_output() -> None:
    out_a = prf(b"label-1", b"data", out_bytes=32)
    out_b = prf(b"label-2", b"data", out_bytes=32)
    assert out_a != out_b


def test_prf_different_parts_different_output() -> None:
    out_a = prf(b"label", b"data-1", out_bytes=32)
    out_b = prf(b"label", b"data-2", out_bytes=32)
    assert out_a != out_b


def test_prf_expands_past_32_bytes_by_chaining() -> None:
    out = prf(b"label", b"data", out_bytes=100)
    # The first 32 bytes equal sha256(label || "||" || data).
    initial = hashlib.sha256(b"label" + b"||" + b"data").digest()
    assert out[:32] == initial


def test_prf_with_no_parts() -> None:
    out = prf(b"label", out_bytes=32)
    assert out == hashlib.sha256(b"label" + b"||").digest()


# ---------------------------------------------------------------------------
# indices_from_tokens
# ---------------------------------------------------------------------------


def test_indices_from_tokens_deterministic_across_calls() -> None:
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    randomness = "deadbeef" * 8
    a = indices_from_tokens(tokens, randomness, len(tokens), 4)
    b = indices_from_tokens(tokens, randomness, len(tokens), 4)
    assert a == b


def test_indices_from_tokens_sorted_ascending() -> None:
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    randomness = "cafebabe" * 8
    indices = indices_from_tokens(tokens, randomness, len(tokens), 5)
    assert indices == sorted(indices)


def test_indices_from_tokens_caps_k_to_seq_len() -> None:
    tokens = [1, 2, 3]
    indices = indices_from_tokens(tokens, "a" * 64, 3, 10)
    assert len(indices) == 3
    assert set(indices) == {0, 1, 2}


def test_indices_from_tokens_respects_k_under_seq_len() -> None:
    tokens = list(range(20))
    indices = indices_from_tokens(tokens, "b" * 64, 20, 5)
    assert len(indices) == 5
    assert all(0 <= i < 20 for i in indices)


def test_indices_from_tokens_strips_0x_prefix() -> None:
    tokens = [1, 2, 3]
    a = indices_from_tokens(tokens, "0xff00", 3, 2)
    b = indices_from_tokens(tokens, "ff00", 3, 2)
    assert a == b


def test_indices_from_tokens_differs_with_token_permutation() -> None:
    randomness = "11112222" * 8
    a = indices_from_tokens([1, 2, 3, 4, 5, 6, 7, 8], randomness, 8, 3)
    b = indices_from_tokens([8, 7, 6, 5, 4, 3, 2, 1], randomness, 8, 3)
    assert a != b


def test_indices_from_tokens_differs_with_different_randomness() -> None:
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    a = indices_from_tokens(tokens, "aa" * 32, 8, 3)
    b = indices_from_tokens(tokens, "bb" * 32, 8, 3)
    assert a != b


# ---------------------------------------------------------------------------
# sign_s_vals / verify_s_vals_signature
# ---------------------------------------------------------------------------


def test_sign_s_vals_returns_hex() -> None:
    signature = sign_s_vals([1, 2, 3], "secret")
    assert isinstance(signature, str)
    assert len(signature) == 64
    assert all(c in "0123456789abcdef" for c in signature)


def test_sign_s_vals_deterministic_under_same_inputs() -> None:
    assert sign_s_vals([1, 2, 3], "secret") == sign_s_vals([1, 2, 3], "secret")


def test_sign_s_vals_matches_hmac_sha256() -> None:
    s_vals = [10, 20, 30]
    expected = hmac.new(
        b"secret",
        b"".join(int_to_bytes(v) for v in s_vals),
        hashlib.sha256,
    ).hexdigest()
    assert sign_s_vals(s_vals, "secret") == expected


def test_sign_s_vals_changes_with_different_vals() -> None:
    a = sign_s_vals([1, 2, 3], "secret")
    b = sign_s_vals([1, 2, 4], "secret")
    assert a != b


def test_sign_s_vals_changes_with_different_secret() -> None:
    a = sign_s_vals([1, 2, 3], "secret-a")
    b = sign_s_vals([1, 2, 3], "secret-b")
    assert a != b


def test_verify_s_vals_signature_accepts_valid() -> None:
    signature = sign_s_vals([1, 2, 3], "secret")
    assert verify_s_vals_signature([1, 2, 3], signature, "secret") is True


def test_verify_s_vals_signature_rejects_tampered_signature() -> None:
    signature = sign_s_vals([1, 2, 3], "secret")
    tampered = signature[:-1] + ("0" if signature[-1] != "0" else "1")
    assert verify_s_vals_signature([1, 2, 3], tampered, "secret") is False


def test_verify_s_vals_signature_rejects_tampered_vals() -> None:
    signature = sign_s_vals([1, 2, 3], "secret")
    assert verify_s_vals_signature([1, 2, 4], signature, "secret") is False


def test_verify_s_vals_signature_rejects_wrong_secret() -> None:
    signature = sign_s_vals([1, 2, 3], "secret")
    assert verify_s_vals_signature([1, 2, 3], signature, "other-secret") is False


# ---------------------------------------------------------------------------
# derive_rollout_randomness
# ---------------------------------------------------------------------------


def test_derive_rollout_randomness_is_hex_sha256() -> None:
    out = derive_rollout_randomness("p", "c", "m", 0)
    assert len(out) == 64
    assert all(ch in "0123456789abcdef" for ch in out)


def test_derive_rollout_randomness_changes_with_sample_index() -> None:
    a = derive_rollout_randomness("p", "c", "m", 0)
    b = derive_rollout_randomness("p", "c", "m", 1)
    assert a != b


def test_derive_rollout_randomness_changes_with_policy_id() -> None:
    a = derive_rollout_randomness("policy-A", "c", "m", 0)
    b = derive_rollout_randomness("policy-B", "c", "m", 0)
    assert a != b


def test_derive_rollout_randomness_changes_with_challenge_id() -> None:
    a = derive_rollout_randomness("p", "challenge-A", "m", 0)
    b = derive_rollout_randomness("p", "challenge-B", "m", 0)
    assert a != b


def test_derive_rollout_randomness_changes_with_miner_id() -> None:
    a = derive_rollout_randomness("p", "c", "miner-A", 0)
    b = derive_rollout_randomness("p", "c", "miner-B", 0)
    assert a != b


def test_derive_rollout_randomness_deterministic() -> None:
    a = derive_rollout_randomness("p", "c", "m", 42)
    b = derive_rollout_randomness("p", "c", "m", 42)
    assert a == b


# ---------------------------------------------------------------------------
# constants sanity
# ---------------------------------------------------------------------------


def test_prime_q_is_mersenne_prime() -> None:
    assert PRIME_Q == 2**31 - 1


def test_challenge_k_positive() -> None:
    assert CHALLENGE_K > 0


def test_proof_tolerance_non_negative() -> None:
    assert PROOF_TOLERANCE >= 0


def test_rng_labels_distinct() -> None:
    assert RNG_LABEL["sketch"] != RNG_LABEL["open"]
