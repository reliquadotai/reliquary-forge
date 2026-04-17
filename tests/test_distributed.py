"""Single-process tests for the distributed-training scaffolding.

Multi-process integration tests run via ``torchrun`` on the staging fleet;
see docs/training/distributed.md. These tests exercise:

- DistributedConfig env parsing + predicates
- MixedPrecisionPolicy dtype resolution
- FSDP2Backend no-op wrap in single-process mode
- DiLoCoOuterLoop snapshot + apply with world_size=1 (no all-reduce)
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

torch = pytest.importorskip("torch")

from reliquary.training.distributed import (
    DiLoCoConfig,
    DiLoCoOuterLoop,
    DistributedConfig,
    FSDP2Backend,
    MixedPrecisionPolicy,
    _resolve_dtype,
    setup_distributed,
    teardown_distributed,
)


@contextmanager
def _env(overrides: dict[str, str]):
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_distributed_config_from_env_single_process_default() -> None:
    with _env({"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}):
        cfg = DistributedConfig.from_env()
    assert cfg.world_size == 1
    assert cfg.rank == 0
    assert cfg.is_single_process() is True
    assert cfg.is_root() is True
    assert cfg.backend == "nccl"


def test_distributed_config_from_env_multi_process() -> None:
    with _env(
        {
            "WORLD_SIZE": "4",
            "RANK": "2",
            "LOCAL_RANK": "1",
            "MASTER_ADDR": "10.0.0.1",
            "MASTER_PORT": "4321",
        }
    ):
        cfg = DistributedConfig.from_env()
    assert cfg.world_size == 4
    assert cfg.rank == 2
    assert cfg.local_rank == 1
    assert cfg.is_single_process() is False
    assert cfg.is_root() is False
    assert cfg.master_addr == "10.0.0.1"
    assert cfg.master_port == "4321"


def test_setup_distributed_single_process_noop() -> None:
    cfg = DistributedConfig(world_size=1, rank=0, local_rank=0)
    assert setup_distributed(cfg) is False
    teardown_distributed()


def test_mixed_precision_policy_defaults_are_bf16_fp32_bf16() -> None:
    policy = MixedPrecisionPolicy()
    assert policy.param_dtype == "bfloat16"
    assert policy.reduce_dtype == "float32"
    assert policy.output_dtype == "bfloat16"


def test_resolve_dtype_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        _resolve_dtype("not-a-dtype")


def test_resolve_dtype_aliases_accepted() -> None:
    assert _resolve_dtype("bf16") is torch.bfloat16
    assert _resolve_dtype("fp32") is torch.float32
    assert _resolve_dtype("float16") is torch.float16


def test_fsdp2_backend_single_process_is_identity_wrap() -> None:
    cfg = DistributedConfig(world_size=1, rank=0, local_rank=0)
    backend = FSDP2Backend(cfg)
    model = torch.nn.Linear(8, 4)
    wrapped = backend.wrap(model)
    assert wrapped is model


def test_diloco_snapshot_then_apply_with_no_inner_update_is_noop() -> None:
    """With world_size=1 and no inner optimizer step between snapshot and
    apply, the delta is zero and parameters are unchanged."""
    cfg_dist = DistributedConfig(world_size=1, rank=0, local_rank=0)
    outer = DiLoCoOuterLoop(DiLoCoConfig(), cfg_dist)

    model = torch.nn.Linear(8, 4)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    outer.snapshot(model)
    outer.apply(model)

    for name, p in model.named_parameters():
        assert torch.equal(p, before[name]), name


def test_diloco_apply_without_snapshot_raises() -> None:
    cfg_dist = DistributedConfig(world_size=1, rank=0, local_rank=0)
    outer = DiLoCoOuterLoop(DiLoCoConfig(), cfg_dist)
    with pytest.raises(RuntimeError):
        outer.apply(torch.nn.Linear(2, 2))


def test_diloco_applies_outer_lr_scaled_delta_single_process() -> None:
    cfg_dist = DistributedConfig(world_size=1, rank=0, local_rank=0)
    outer = DiLoCoOuterLoop(
        DiLoCoConfig(inner_steps=1, outer_lr=0.5, outer_momentum=0.0),
        cfg_dist,
    )

    model = torch.nn.Linear(4, 2)
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    outer.snapshot(model)

    # Pretend an inner optimizer took a step: add +1 to every parameter.
    with torch.no_grad():
        for _, p in model.named_parameters():
            p.add_(torch.ones_like(p))

    outer.apply(model)

    # With outer_lr=0.5, momentum=0: final = prev + 0.5 * delta = original + 0.5.
    for name, p in model.named_parameters():
        expected = original[name] + 0.5
        assert torch.allclose(p, expected, atol=1e-5), (name, p, expected)


def test_diloco_momentum_accumulates_across_outer_steps() -> None:
    cfg_dist = DistributedConfig(world_size=1, rank=0, local_rank=0)
    cfg_outer = DiLoCoConfig(inner_steps=1, outer_lr=1.0, outer_momentum=0.5)
    outer = DiLoCoOuterLoop(cfg_outer, cfg_dist)

    model = torch.nn.Linear(4, 2)

    # Outer step 1: inner adds 1; momentum starts at 0; new momentum = 1.0.
    outer.snapshot(model)
    with torch.no_grad():
        for _, p in model.named_parameters():
            p.add_(torch.ones_like(p))
    outer.apply(model)

    # Outer step 2: inner adds another 1; momentum becomes 0.5*1 + 1 = 1.5.
    outer.snapshot(model)
    with torch.no_grad():
        for _, p in model.named_parameters():
            p.add_(torch.ones_like(p))
    outer.apply(model)

    # Access stored momentum via internal state to verify accumulation.
    for name in outer._momentum:
        momentum = outer._momentum[name]
        assert torch.isclose(momentum, torch.full_like(momentum, 1.5), atol=1e-5).all()
