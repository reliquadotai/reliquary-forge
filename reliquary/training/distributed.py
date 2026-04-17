"""Distributed-training scaffolding for Reliquary Forge.

Pure-Python wrapper over PyTorch's native distributed primitives:

  - ``DistributedConfig`` describes a launch (world_size, rank, backend,
    master_addr, master_port). Constructed from env vars set by
    ``torchrun`` in the common case.
  - ``setup_distributed`` / ``teardown_distributed`` initialize and
    destroy the process group idempotently.
  - ``FSDP2Backend`` wraps a ``torch.nn.Module`` with
    ``torch.distributed.fsdp.fully_shard`` (aka FSDP2). Configurable
    sharding strategy and mixed-precision policy.
  - ``DiLoCoOuterLoop`` applies the DiLoCo outer-loop sync pattern:
    run ``inner_steps`` local optimizer updates, then all-reduce the
    accumulated delta across workers and apply a momentum-weighted
    step. One of the two training backends cited in
    ``02_TIER2_PRD.md`` Epic 3.

Single-process tests verify the no-op case (``world_size=1``).
Multi-process integration is exercised via ``torchrun`` on the staging
fleet; instructions in ``docs/training/distributed.md``.

Spec: ``02_TIER2_PRD.md`` Epic 3 + DiLoCo paper (v5 outer-loop).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DistributedConfig:
    world_size: int
    rank: int
    local_rank: int
    backend: str = "nccl"
    master_addr: str = "127.0.0.1"
    master_port: str = "29500"

    def is_single_process(self) -> bool:
        return self.world_size <= 1

    def is_root(self) -> bool:
        return self.rank == 0

    @classmethod
    def from_env(cls) -> "DistributedConfig":
        """Build from ``torchrun`` / ``torch.distributed.launch`` env vars."""
        return cls(
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
            rank=int(os.environ.get("RANK", "0")),
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            backend=os.environ.get("RELIQUARY_DIST_BACKEND", "nccl"),
            master_addr=os.environ.get("MASTER_ADDR", "127.0.0.1"),
            master_port=os.environ.get("MASTER_PORT", "29500"),
        )


def setup_distributed(config: DistributedConfig) -> bool:
    """Initialize the default process group if ``world_size > 1``.

    Returns ``True`` if a process group was initialized, ``False`` otherwise
    (single-process path — callers treat this as a no-op).
    """
    import torch.distributed as dist

    if config.is_single_process():
        return False
    if dist.is_initialized():
        return True
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = config.master_port
    dist.init_process_group(
        backend=config.backend,
        world_size=config.world_size,
        rank=config.rank,
    )
    return True


def teardown_distributed() -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


@dataclass
class MixedPrecisionPolicy:
    """FSDP2-compatible mixed-precision knobs."""

    param_dtype: str = "bfloat16"
    reduce_dtype: str = "float32"
    output_dtype: str = "bfloat16"

    def to_torch_policy(self) -> Any:
        import torch
        from torch.distributed.fsdp import MixedPrecisionPolicy as _Policy

        return _Policy(
            param_dtype=_resolve_dtype(self.param_dtype),
            reduce_dtype=_resolve_dtype(self.reduce_dtype),
            output_dtype=_resolve_dtype(self.output_dtype),
        )


def _resolve_dtype(name: str) -> Any:
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype {name!r}; choose from {sorted(mapping)}")
    return mapping[name]


class FSDP2Backend:
    """Wrap a module with ``torch.distributed.fsdp.fully_shard`` (FSDP2).

    Keeps the module trainable in single-process mode (no-op wrap) and in
    multi-process mode (real sharded training). Callers use ``wrap()`` to
    return the (possibly sharded) module; shape and API are identical.
    """

    def __init__(
        self,
        config: DistributedConfig,
        *,
        mp_policy: MixedPrecisionPolicy | None = None,
    ) -> None:
        self.config = config
        self.mp_policy = mp_policy or MixedPrecisionPolicy()

    def wrap(self, model: Any) -> Any:
        if self.config.is_single_process():
            return model
        from torch.distributed.fsdp import fully_shard

        fully_shard(model, mp_policy=self.mp_policy.to_torch_policy())
        return model


@dataclass
class DiLoCoConfig:
    inner_steps: int = 30
    outer_lr: float = 0.4
    outer_momentum: float = 0.95
    sync_dtype: str = "float32"


class DiLoCoOuterLoop:
    """DiLoCo outer-loop step.

    Workers run ``inner_steps`` local AdamW updates; at outer-loop time each
    worker computes the parameter delta since the last outer sync, the
    deltas are averaged (all-reduce sum / world_size), and the globally
    averaged delta is applied with outer-loop momentum.

    Usage:

        outer = DiLoCoOuterLoop(config, dist_config)
        outer.snapshot(model)
        for step in range(config.inner_steps):
            inner_optimizer.step()
        outer.apply(model)
    """

    def __init__(
        self,
        config: DiLoCoConfig,
        distributed: DistributedConfig,
    ) -> None:
        self.config = config
        self.distributed = distributed
        self._prev_params: dict[str, Any] = {}
        self._momentum: dict[str, Any] = {}

    def snapshot(self, model: Any) -> None:
        """Record the pre-inner-loop parameters."""
        self._prev_params = {
            name: p.detach().clone() for name, p in _iter_params(model)
        }

    def apply(self, model: Any) -> None:
        """All-reduce the delta and apply momentum-weighted outer step."""
        if not self._prev_params:
            raise RuntimeError("DiLoCoOuterLoop.apply called before snapshot")
        import torch

        for name, param in _iter_params(model):
            prev = self._prev_params[name]
            delta = (param.detach() - prev).to(_resolve_dtype(self.config.sync_dtype))
            if not self.distributed.is_single_process():
                self._all_reduce_mean(delta)
            momentum = self._momentum.get(name)
            if momentum is None:
                momentum = torch.zeros_like(delta)
            momentum.mul_(self.config.outer_momentum).add_(delta)
            self._momentum[name] = momentum
            param.data = prev + self.config.outer_lr * momentum.to(param.dtype)

    @staticmethod
    def _all_reduce_mean(tensor: Any) -> None:
        import torch.distributed as dist

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(dist.get_world_size())


def _iter_params(model: Any):
    return list(model.named_parameters())
