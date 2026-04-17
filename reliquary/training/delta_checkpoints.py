"""Delta checkpoint compression + reconstruction.

Compute a Merkle-committed per-tensor int8 quantized delta between two
state_dicts and reconstruct the target state_dict from ``base + delta``.

The module is pure-compute (torch + hashlib only); upload to R2 and
onchain commitment of the merkle root are caller responsibilities.

Spec: private/reliquary-plan/notes/spec-delta-checkpoints.md.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping


class DeltaCheckpointError(Exception):
    """Base class for reconstruction errors."""


class DeltaMerkleMismatch(DeltaCheckpointError):
    """Computed merkle root disagrees with bundle-claimed root."""


class DeltaShardCorrupted(DeltaCheckpointError):
    """A shard's data_bytes hash disagrees with its payload_sha256."""


class DeltaBundleStructural(DeltaCheckpointError):
    """Bundle is missing tensors, has extras, or has wrong shapes."""


@dataclass(frozen=True)
class DeltaShard:
    tensor_name: str
    dtype: str
    shape: tuple[int, ...]
    scale: float
    zero_point: int
    payload_sha256: str
    data_bytes: bytes


@dataclass(frozen=True)
class DeltaBundle:
    from_checkpoint_hash: str
    to_checkpoint_hash: str
    shards: tuple[DeltaShard, ...]
    merkle_root_hex: str


def _tensor_hash(state_dict: Mapping[str, "torch.Tensor"]) -> str:  # noqa: F821
    import torch

    hasher = hashlib.sha256()
    for name in sorted(state_dict.keys()):
        tensor = state_dict[name]
        hasher.update(name.encode("utf-8"))
        hasher.update(b"|")
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(b"|")
        cpu_tensor = tensor.detach().to("cpu").to(torch.float32)
        hasher.update(cpu_tensor.numpy().tobytes())
    return hasher.hexdigest()


def _merkle_root_hex(leaves_hex: list[str]) -> str:
    if not leaves_hex:
        return hashlib.sha256(b"").hexdigest()
    layer = [bytes.fromhex(h) for h in leaves_hex]
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        layer = [
            hashlib.sha256(layer[i] + layer[i + 1]).digest()
            for i in range(0, len(layer), 2)
        ]
    return layer[0].hex()


def _quantize_delta(delta) -> tuple[float, bytes, tuple[int, ...]]:
    import torch

    shape = tuple(delta.shape)
    flat = delta.detach().to(torch.float32).reshape(-1)
    abs_max = float(flat.abs().max().item()) if flat.numel() > 0 else 0.0
    if abs_max == 0.0:
        zero_bytes = bytes(flat.numel())
        return 0.0, zero_bytes, shape
    scale = abs_max / 127.0
    q = torch.round(flat / scale).clamp(-127, 127).to(torch.int8)
    data_bytes = q.cpu().numpy().tobytes()
    return scale, data_bytes, shape


def _dequantize(data_bytes: bytes, scale: float, shape: tuple[int, ...]):
    import torch

    import numpy as np

    if scale == 0.0:
        return torch.zeros(shape, dtype=torch.float32)
    array = np.frombuffer(data_bytes, dtype=np.int8).reshape(shape).copy()
    tensor = torch.from_numpy(array).to(torch.float32) * scale
    return tensor


def compute_delta(
    base: Mapping[str, "torch.Tensor"],  # noqa: F821
    target: Mapping[str, "torch.Tensor"],  # noqa: F821
) -> DeltaBundle:
    """Compute a Merkle-committed delta from ``base`` to ``target``.

    Both state_dicts must share tensor keys and shapes. Tensors may live on
    any device; quantization runs on CPU after a detach+float32 copy.
    """
    import torch

    missing = sorted(set(target.keys()) - set(base.keys()))
    extra = sorted(set(base.keys()) - set(target.keys()))
    if missing or extra:
        raise DeltaBundleStructural(
            f"key mismatch: missing_from_base={missing}, extra_in_base={extra}"
        )

    shards: list[DeltaShard] = []
    for name in sorted(target.keys()):
        base_t = base[name]
        target_t = target[name]
        if tuple(base_t.shape) != tuple(target_t.shape):
            raise DeltaBundleStructural(
                f"shape mismatch for {name!r}: base={tuple(base_t.shape)} "
                f"target={tuple(target_t.shape)}"
            )
        delta = (
            target_t.detach().to(torch.float32).cpu()
            - base_t.detach().to(torch.float32).cpu()
        )
        scale, data_bytes, shape = _quantize_delta(delta)
        payload_sha = hashlib.sha256(data_bytes).hexdigest()
        shards.append(
            DeltaShard(
                tensor_name=name,
                dtype="int8",
                shape=shape,
                scale=scale,
                zero_point=0,
                payload_sha256=payload_sha,
                data_bytes=data_bytes,
            )
        )

    merkle_root = _merkle_root_hex([s.payload_sha256 for s in shards])
    return DeltaBundle(
        from_checkpoint_hash=_tensor_hash(base),
        to_checkpoint_hash=_tensor_hash(target),
        shards=tuple(shards),
        merkle_root_hex=merkle_root,
    )


def reconstruct(
    base: Mapping[str, "torch.Tensor"],  # noqa: F821
    bundle: DeltaBundle,
) -> dict[str, "torch.Tensor"]:  # noqa: F821
    """Reconstruct the target state_dict from ``base`` and ``bundle``.

    Raises ``DeltaMerkleMismatch`` or ``DeltaShardCorrupted`` before any
    tensor is mutated so partially-reconstructed state is never returned.
    """
    import torch

    recomputed_root = _merkle_root_hex([s.payload_sha256 for s in bundle.shards])
    if recomputed_root != bundle.merkle_root_hex:
        raise DeltaMerkleMismatch(
            f"merkle root mismatch: computed={recomputed_root} "
            f"claimed={bundle.merkle_root_hex}"
        )

    for shard in bundle.shards:
        actual = hashlib.sha256(shard.data_bytes).hexdigest()
        if actual != shard.payload_sha256:
            raise DeltaShardCorrupted(
                f"shard {shard.tensor_name!r}: computed_sha={actual} "
                f"claimed_sha={shard.payload_sha256}"
            )

    base_names = set(base.keys())
    bundle_names = {s.tensor_name for s in bundle.shards}
    missing = sorted(bundle_names - base_names)
    extra = sorted(base_names - bundle_names)
    if missing or extra:
        raise DeltaBundleStructural(
            f"bundle/base name mismatch: in_bundle_only={missing}, "
            f"in_base_only={extra}"
        )

    reconstructed: dict[str, torch.Tensor] = {}
    for shard in bundle.shards:
        base_t = base[shard.tensor_name]
        if tuple(base_t.shape) != shard.shape:
            raise DeltaBundleStructural(
                f"shape mismatch for {shard.tensor_name!r}: "
                f"base={tuple(base_t.shape)} shard={shard.shape}"
            )
        delta_fp32 = _dequantize(shard.data_bytes, shard.scale, shard.shape)
        restored = base_t.detach().to(torch.float32).cpu() + delta_fp32
        reconstructed[shard.tensor_name] = restored.to(base_t.dtype)
    return reconstructed


def apply_chain(
    base: Mapping[str, "torch.Tensor"],  # noqa: F821
    bundles: list[DeltaBundle],
) -> dict[str, "torch.Tensor"]:  # noqa: F821
    """Apply a sequence of deltas onto ``base``, folding each result into the next."""
    state = dict(base)
    for bundle in bundles:
        state = reconstruct(state, bundle)
    return state


def compression_ratio(bundle: DeltaBundle, target: Mapping[str, "torch.Tensor"]) -> float:  # noqa: F821
    """Return ``sum(shard.data_bytes) / sum(target.numel() * 2)``, treating fp16 as baseline."""
    total_bytes = sum(len(s.data_bytes) for s in bundle.shards)
    target_bytes = sum(t.numel() * 2 for t in target.values())  # 2 bytes per fp16 element
    return total_bytes / target_bytes if target_bytes else 0.0
