from __future__ import annotations

from typing import Any

from ..envs.reasoning import generate_reasoning_tasks
from ..protocol.artifacts import make_artifact


def generate_reasoning_task_batch_artifact(
    cfg: dict[str, Any],
    policy_artifact: dict[str, Any],
    *,
    window_id: int,
    count: int,
    split: str = "train",
    seed_offset: int = 0,
) -> dict[str, Any]:
    tasks = generate_reasoning_tasks(
        count=count,
        seed=seed_offset + window_id * 10_000,
        split=split,
    )
    return make_artifact(
        artifact_type="task_batch",
        producer_id=cfg["trainer_id"],
        producer_role="policy_authority",
        policy_id=policy_artifact["artifact_id"],
        window_id=window_id,
        lane="distill",
        commodity_use="distill_only",
        parent_ids=[policy_artifact["artifact_id"]],
        payload={
            "task_batch_id": f"reasoning-train-{window_id:04d}",
            "task_family": "reasoning_traces",
            "split": split,
            "tasks": tasks,
        },
    )
