from __future__ import annotations

from typing import Any

from ..protocol.artifacts import make_artifact
from ..roles.policy_authority import (
    make_policy_authority_decision,
    promote_checkpoint_to_policy,
    publish_distill_policy,
)
from ..storage.registry import ArtifactRegistry
from ..training.backends import build_distill_learner_backend


def train_distill_window(
    registry: ArtifactRegistry,
    cfg: dict[str, Any],
    policy_artifact: dict[str, Any],
    pack_artifact: dict[str, Any],
    *,
    window_id: int,
) -> dict[str, Any]:
    backend = build_distill_learner_backend(cfg["distill_learner_backend"])
    training_result = backend.train(
        registry=registry,
        cfg=cfg,
        policy_artifact=policy_artifact,
        pack_artifact=pack_artifact,
        window_id=window_id,
    )
    checkpoint_artifact = training_result["checkpoint_artifact"]
    benchmark_result_artifact = training_result["benchmark_result_artifact"]
    holdout_delta = float(benchmark_result_artifact["payload"]["holdout_metric"])
    approved = holdout_delta >= 0.0
    rationale = "holdout_improved_or_stable" if approved else "holdout_regressed"
    decision_artifact = make_policy_authority_decision(
        cfg,
        window_id=window_id,
        parent_policy_id=policy_artifact["artifact_id"],
        checkpoint_artifact=checkpoint_artifact,
        benchmark_result_artifact=benchmark_result_artifact,
        approved=approved,
        rationale=rationale,
    )
    registry.put_artifact(decision_artifact)
    if approved:
        next_policy = promote_checkpoint_to_policy(
            cfg,
            checkpoint_artifact=checkpoint_artifact,
            window_id=window_id + 1,
        )
    else:
        next_policy = publish_distill_policy(
            cfg,
            window_id=window_id + 1,
            model_ref=policy_artifact["payload"]["model_ref"],
        )
        next_policy["parent_ids"] = [policy_artifact["artifact_id"], decision_artifact["artifact_id"]]
    registry.put_artifact(next_policy)
    training_batch_artifact = make_artifact(
        artifact_type="training_batch",
        producer_id=cfg["trainer_id"],
        producer_role="learner",
        policy_id=policy_artifact["artifact_id"],
        window_id=window_id,
        lane="distill",
        commodity_use="distill_only",
        parent_ids=[policy_artifact["artifact_id"], pack_artifact["artifact_id"]],
        payload={
            "source_pack_id": pack_artifact["artifact_id"],
            "checkpoint_artifact_id": checkpoint_artifact["artifact_id"],
            "benchmark_result_artifact_id": benchmark_result_artifact["artifact_id"],
            "accepted_record_count": pack_artifact["payload"]["record_count"],
            "backend": backend.name,
            "git_sha": cfg.get("git_sha", "unknown"),
        },
    )
    registry.put_artifact(training_batch_artifact)
    return {
        "window_id": window_id,
        "training_batch_artifact": training_batch_artifact,
        "checkpoint_artifact": checkpoint_artifact,
        "benchmark_result_artifact": benchmark_result_artifact,
        "decision_artifact": decision_artifact,
        "next_policy_artifact": next_policy,
        "eval_before": benchmark_result_artifact["payload"]["eval_before"],
        "eval_after": benchmark_result_artifact["payload"]["eval_after"],
    }
