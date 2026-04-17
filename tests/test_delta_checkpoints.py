"""Tests for Reliquary Forge delta checkpoints.

Spec reference: private/reliquary-plan/notes/spec-delta-checkpoints.md.
"""

from __future__ import annotations

import hashlib

import pytest

torch = pytest.importorskip("torch")

from reliquary.training.delta_checkpoints import (
    DeltaBundleStructural,
    DeltaMerkleMismatch,
    DeltaShard,
    DeltaShardCorrupted,
    apply_chain,
    compression_ratio,
    compute_delta,
    reconstruct,
)


def _rand_state(seed: int, tensors: dict[str, tuple[int, ...]]) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(shape, generator=gen, dtype=torch.float32) * 0.1
        for name, shape in tensors.items()
    }


TENSORS = {
    "embedding.weight": (16, 8),
    "layer0.proj": (8, 8),
    "layer0.bias": (8,),
    "head.weight": (32, 8),
}


def test_zero_delta_roundtrip_produces_empty_shards() -> None:
    state = _rand_state(seed=42, tensors=TENSORS)
    bundle = compute_delta(state, state)
    assert bundle.from_checkpoint_hash == bundle.to_checkpoint_hash
    for shard in bundle.shards:
        assert shard.scale == 0.0
        assert shard.data_bytes == bytes(int(torch.tensor(shard.shape).prod().item()))


def test_reconstruct_returns_base_on_zero_delta() -> None:
    state = _rand_state(seed=1, tensors=TENSORS)
    bundle = compute_delta(state, state)
    restored = reconstruct(state, bundle)
    for name in TENSORS:
        assert torch.equal(restored[name], state[name])


def test_small_delta_reconstruction_within_bounded_error() -> None:
    base = _rand_state(seed=1, tensors=TENSORS)
    noise = _rand_state(seed=2, tensors=TENSORS)
    target = {k: base[k] + 0.01 * noise[k] for k in TENSORS}

    bundle = compute_delta(base, target)
    restored = reconstruct(base, bundle)

    for name in TENSORS:
        err = (restored[name].to(torch.float32) - target[name].to(torch.float32)).abs().max().item()
        scale = next(s.scale for s in bundle.shards if s.tensor_name == name)
        # Per-element error bound: half the quantization bucket.
        assert err <= scale / 2 + 1e-6, (name, err, scale)


def test_compute_delta_is_deterministic() -> None:
    base = _rand_state(seed=1, tensors=TENSORS)
    target = _rand_state(seed=2, tensors=TENSORS)
    b1 = compute_delta(base, target)
    b2 = compute_delta(base, target)
    assert b1.merkle_root_hex == b2.merkle_root_hex
    assert [s.payload_sha256 for s in b1.shards] == [s.payload_sha256 for s in b2.shards]
    assert [s.data_bytes for s in b1.shards] == [s.data_bytes for s in b2.shards]


def test_tampered_shard_merkle_root_differs() -> None:
    base = _rand_state(seed=1, tensors=TENSORS)
    target = _rand_state(seed=2, tensors=TENSORS)
    bundle = compute_delta(base, target)

    # Flip a byte in one shard's data_bytes but recompute the hash and merkle root.
    shards_list = list(bundle.shards)
    victim = shards_list[1]
    tampered_bytes = bytes([victim.data_bytes[0] ^ 0xFF]) + victim.data_bytes[1:]
    tampered_hash = hashlib.sha256(tampered_bytes).hexdigest()
    tampered_shard = DeltaShard(
        tensor_name=victim.tensor_name,
        dtype=victim.dtype,
        shape=victim.shape,
        scale=victim.scale,
        zero_point=victim.zero_point,
        payload_sha256=tampered_hash,
        data_bytes=tampered_bytes,
    )
    shards_list[1] = tampered_shard
    from reliquary.training.delta_checkpoints import DeltaBundle, _merkle_root_hex

    recomputed_root = _merkle_root_hex([s.payload_sha256 for s in shards_list])
    assert recomputed_root != bundle.merkle_root_hex

    tampered_bundle = DeltaBundle(
        from_checkpoint_hash=bundle.from_checkpoint_hash,
        to_checkpoint_hash=bundle.to_checkpoint_hash,
        shards=tuple(shards_list),
        merkle_root_hex=bundle.merkle_root_hex,  # original root, now inconsistent
    )
    with pytest.raises(DeltaMerkleMismatch):
        reconstruct(base, tampered_bundle)


def test_shard_hash_mismatch_detected() -> None:
    base = _rand_state(seed=1, tensors=TENSORS)
    target = _rand_state(seed=2, tensors=TENSORS)
    bundle = compute_delta(base, target)

    shards_list = list(bundle.shards)
    victim = shards_list[0]
    tampered_shard = DeltaShard(
        tensor_name=victim.tensor_name,
        dtype=victim.dtype,
        shape=victim.shape,
        scale=victim.scale,
        zero_point=victim.zero_point,
        payload_sha256=victim.payload_sha256,
        data_bytes=bytes([victim.data_bytes[0] ^ 0xFF]) + victim.data_bytes[1:],
    )
    shards_list[0] = tampered_shard
    from reliquary.training.delta_checkpoints import DeltaBundle, _merkle_root_hex

    bundle_with_same_root = DeltaBundle(
        from_checkpoint_hash=bundle.from_checkpoint_hash,
        to_checkpoint_hash=bundle.to_checkpoint_hash,
        shards=tuple(shards_list),
        merkle_root_hex=_merkle_root_hex([s.payload_sha256 for s in shards_list]),
    )
    with pytest.raises(DeltaShardCorrupted):
        reconstruct(base, bundle_with_same_root)


def test_missing_tensor_detected() -> None:
    base = _rand_state(seed=1, tensors=TENSORS)
    partial_target = {k: v for k, v in _rand_state(seed=2, tensors=TENSORS).items() if k != "head.weight"}
    with pytest.raises(DeltaBundleStructural):
        compute_delta(base, partial_target)


def test_shape_mismatch_detected_at_reconstruct() -> None:
    base = _rand_state(seed=1, tensors=TENSORS)
    target = _rand_state(seed=2, tensors=TENSORS)
    bundle = compute_delta(base, target)

    wrong_base = dict(base)
    wrong_base["layer0.proj"] = torch.randn(4, 8, dtype=torch.float32)
    with pytest.raises(DeltaBundleStructural):
        reconstruct(wrong_base, bundle)


def test_reconstruct_preserves_shape_and_dtype() -> None:
    base = {k: v.to(torch.float16) for k, v in _rand_state(seed=1, tensors=TENSORS).items()}
    target = {k: v.to(torch.float16) for k, v in _rand_state(seed=2, tensors=TENSORS).items()}
    bundle = compute_delta(base, target)
    restored = reconstruct(base, bundle)
    for name in TENSORS:
        assert tuple(restored[name].shape) == tuple(target[name].shape)
        assert restored[name].dtype == base[name].dtype


def test_apply_chain_composes_two_deltas() -> None:
    base = _rand_state(seed=1, tensors=TENSORS)
    middle = _rand_state(seed=2, tensors=TENSORS)
    target = _rand_state(seed=3, tensors=TENSORS)

    b1 = compute_delta(base, middle)
    b2 = compute_delta(middle, target)
    chained = apply_chain(base, [b1, b2])

    # Composition allows up to 2x single-step quantization error.
    for name in TENSORS:
        err = (chained[name].to(torch.float32) - target[name].to(torch.float32)).abs().max().item()
        assert err < 0.01, (name, err)


def test_compression_ratio_below_one_for_realistic_delta() -> None:
    base = _rand_state(seed=1, tensors=TENSORS)
    noise = _rand_state(seed=2, tensors=TENSORS)
    target = {k: base[k] + 0.005 * noise[k] for k in TENSORS}
    bundle = compute_delta(base, target)
    ratio = compression_ratio(bundle, target)
    # int8 per element = 1 byte; fp16 baseline = 2 bytes per element.
    # Compression ratio = 0.5 (ignoring per-tensor metadata).
    assert 0.4 < ratio < 0.6, ratio


def test_empty_state_dict_produces_zero_shard_bundle() -> None:
    bundle = compute_delta({}, {})
    assert bundle.shards == ()
    assert bundle.merkle_root_hex == hashlib.sha256(b"").hexdigest()


def test_reconstruct_validates_merkle_root_before_applying() -> None:
    base = _rand_state(seed=1, tensors=TENSORS)
    target = _rand_state(seed=2, tensors=TENSORS)
    bundle = compute_delta(base, target)

    from reliquary.training.delta_checkpoints import DeltaBundle

    bogus = DeltaBundle(
        from_checkpoint_hash=bundle.from_checkpoint_hash,
        to_checkpoint_hash=bundle.to_checkpoint_hash,
        shards=bundle.shards,
        merkle_root_hex="00" * 32,
    )
    with pytest.raises(DeltaMerkleMismatch):
        reconstruct(base, bogus)
