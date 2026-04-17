"""Multi-process DiLoCo integration test.

Spawns ``world_size=2`` worker processes on a single host via
``torch.multiprocessing``, initializes a ``gloo`` process group on
localhost, and exercises ``DiLoCoOuterLoop`` with divergent per-rank
inner updates. After the outer step, both ranks must agree on the
averaged-delta apply — the whole point of the all-reduce inside
``DiLoCoOuterLoop.apply``.

CPU-only (gloo backend, no CUDA dependency), so this runs in CI.
Production uses nccl on GPU and launches via ``torchrun``; the module
logic is identical.

Spec: 02_TIER2_PRD.md Epic 3 — DiLoCo outer-loop semantics across
physically separate workers.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

try:
    import torch.distributed as dist
    import torch.multiprocessing as mp
except Exception:  # pragma: no cover - platform without distributed
    dist = None
    mp = None


def _worker(
    rank: int,
    world_size: int,
    init_method: str,
    output_dir: str,
) -> None:
    """One participating worker in the multi-process DiLoCo run."""
    import torch as _torch
    from reliquary.training.distributed import (
        DiLoCoConfig,
        DiLoCoOuterLoop,
        DistributedConfig,
    )

    _torch.manual_seed(42)

    dist_cfg = DistributedConfig(
        world_size=world_size,
        rank=rank,
        local_rank=0,
        backend="gloo",
    )

    dist.init_process_group(
        backend=dist_cfg.backend,
        world_size=dist_cfg.world_size,
        rank=dist_cfg.rank,
        init_method=init_method,
    )

    try:
        model = _torch.nn.Linear(4, 2, bias=True)
        # Every rank starts from the same parameters (deterministic seed).
        initial = {name: p.detach().clone() for name, p in model.named_parameters()}

        outer = DiLoCoOuterLoop(
            DiLoCoConfig(inner_steps=1, outer_lr=1.0, outer_momentum=0.0),
            dist_cfg,
        )
        outer.snapshot(model)

        # Each rank applies a DIFFERENT fake inner update so the pre-apply
        # delta differs per worker. Rank 0 adds +1, rank 1 adds +3.
        # Averaged delta should be +2 on both ranks after all-reduce inside
        # DiLoCoOuterLoop.apply.
        rank_scalar = 1.0 if rank == 0 else 3.0
        with _torch.no_grad():
            for _, p in model.named_parameters():
                p.add_(_torch.full_like(p, rank_scalar))

        outer.apply(model)

        # Capture final parameters and write to disk; parent reads and asserts.
        final = {name: p.detach().clone() for name, p in model.named_parameters()}
        out_path = Path(output_dir) / f"rank_{rank}.pt"
        _torch.save(
            {
                "initial": initial,
                "final": final,
            },
            out_path,
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    dist is None or mp is None,
    reason="torch.distributed or torch.multiprocessing unavailable",
)
def test_diloco_outer_step_averages_deltas_across_two_processes() -> None:
    """Two workers emit divergent deltas (+1 and +3); after DiLoCo outer
    apply, both land on initial + 2 (the average)."""
    world_size = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        file_init = Path(tmpdir) / "init"
        init_method = f"file://{file_init}"

        mp.spawn(
            _worker,
            args=(world_size, init_method, tmpdir),
            nprocs=world_size,
            join=True,
        )

        rank_0 = torch.load(Path(tmpdir) / "rank_0.pt", weights_only=False)
        rank_1 = torch.load(Path(tmpdir) / "rank_1.pt", weights_only=False)

        # Both ranks started from the same initial weights (seed=42).
        for name in rank_0["initial"]:
            assert torch.equal(rank_0["initial"][name], rank_1["initial"][name]), (
                f"initial weights diverged at {name!r}"
            )

        # Expected final = initial + outer_lr * mean(delta_0, delta_1)
        # delta_0 = +1, delta_1 = +3, mean = +2, outer_lr = 1.0 so final = initial + 2.
        for name in rank_0["initial"]:
            expected = rank_0["initial"][name] + 2.0
            assert torch.allclose(rank_0["final"][name], expected, atol=1e-5), (
                f"rank 0 {name}: got {rank_0['final'][name]}, expected {expected}"
            )
            assert torch.allclose(rank_1["final"][name], expected, atol=1e-5), (
                f"rank 1 {name}: got {rank_1['final'][name]}, expected {expected}"
            )
            # All-reduce means ranks must CONVERGE (agree bit-exactly).
            assert torch.equal(rank_0["final"][name], rank_1["final"][name]), (
                f"{name!r}: ranks diverged after all-reduce"
            )
