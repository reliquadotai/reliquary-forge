from __future__ import annotations

from typing import Any

from ..distill.teacher import build_teacher_backend
from ..protocol.artifacts import make_artifact


def generate_rl_rollouts(
    cfg: dict[str, Any],
    policy_artifact: dict[str, Any],
    challenge_artifacts: list[dict[str, Any]],
    *,
    window_id: int,
    miner_profile: dict[str, Any] | None = None,
    rollouts_per_challenge: int | None = None,
) -> list[dict[str, Any]]:
    from .miner import mine_window

    return mine_window(
        cfg,
        policy_artifact,
        challenge_artifacts,
        window_id=window_id,
        miner_profile=miner_profile,
        rollouts_per_challenge=rollouts_per_challenge,
    )


def generate_distill_traces(
    cfg: dict[str, Any],
    policy_artifact: dict[str, Any],
    task_batch_artifact: dict[str, Any],
    *,
    window_id: int,
    miner_profile: dict[str, Any],
    samples_per_task: int | None = None,
) -> list[dict[str, Any]]:
    backend = build_teacher_backend(cfg)
    miner_id = miner_profile.get("miner_id", cfg["miner_id"])
    miner_secret = miner_profile.get("miner_secret", cfg["miner_secret"])
    sample_total = samples_per_task or int(cfg.get("distill_samples_per_task", 1))
    traces: list[dict[str, Any]] = []
    for task_index, task in enumerate(task_batch_artifact["payload"]["tasks"]):
        for sample_index in range(sample_total):
            trace = backend.generate_trace(
                task=task,
                miner_profile=miner_profile,
                sample_index=(task_index * sample_total) + sample_index,
                miner_secret=miner_secret,
            )
            artifact = make_artifact(
                artifact_type="distill_trace",
                producer_id=miner_id,
                producer_role="generation_miner",
                policy_id=policy_artifact["artifact_id"],
                window_id=window_id,
                lane="distill",
                commodity_use="dual_use",
                parent_ids=[policy_artifact["artifact_id"], task_batch_artifact["artifact_id"]],
                payload={
                    "task_id": task["task_id"],
                    "task": task,
                    "trace": trace,
                    "sample_index": sample_index,
                    "trace_archetype": miner_profile.get("archetype", "reasoned_exact"),
                    "miner_profile": {
                        "miner_id": miner_id,
                        "archetype": miner_profile.get("archetype", "reasoned_exact"),
                        "strategy": miner_profile.get("strategy", "reasoned_exact"),
                    },
                },
                signature_bundle={"scheme": "local-hmac", "miner_id": miner_id},
            )
            traces.append(artifact)
    return traces
