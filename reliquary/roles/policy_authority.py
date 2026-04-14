from __future__ import annotations

from typing import Any

from ..protocol.artifacts import finalize_policy_artifact, make_artifact, make_policy_artifact


def _quorum_members(cfg: dict[str, Any]) -> list[str]:
    return [member.strip() for member in cfg.get("policy_authority_quorum", "").split(",") if member.strip()]


def publish_distill_policy(cfg: dict[str, Any], window_id: int, model_ref: str | None = None) -> dict[str, Any]:
    student_model = model_ref or cfg["distill_student_model_ref"]
    policy = make_policy_artifact(
        producer_id=cfg["trainer_id"],
        window_id=window_id,
        model_ref=student_model,
        env_id="reasoning",
        env_config={
            "task_family": "reasoning_traces",
            "train_task_count": cfg["distill_train_task_count"],
            "holdout_count": cfg["distill_holdout_count"],
            "samples_per_task": cfg.get("distill_samples_per_task", 1),
        },
        generation_config={
            "teacher_backend": cfg["teacher_backend"],
            "teacher_model": cfg.get("model_teacher", student_model),
            "student_model": student_model,
            "vllm_model": cfg.get("vllm_model", ""),
            "vllm_base_url": cfg.get("vllm_base_url", ""),
            "temperature": 0.2,
        },
        trainer_config={
            "backend": cfg["distill_learner_backend"],
            "policy_authority_mode": cfg["policy_authority_mode"],
            "git_sha": cfg.get("git_sha", "unknown"),
            "service_version": cfg.get("service_version", "0.1.0"),
        },
        lane_permissions=["distill"],
        evaluation_contract={
            "lane": "distill",
            "metric": "reasoning_contract_score",
            "holdout_count": cfg["distill_holdout_count"],
            "eval_set_version": "reasoning-holdout-v1",
        },
    )
    return finalize_policy_artifact(policy)


def make_policy_authority_decision(
    cfg: dict[str, Any],
    *,
    window_id: int,
    parent_policy_id: str,
    checkpoint_artifact: dict[str, Any],
    benchmark_result_artifact: dict[str, Any],
    approved: bool,
    rationale: str,
) -> dict[str, Any]:
    signers = _quorum_members(cfg)
    return make_artifact(
        artifact_type="policy_authority_decision",
        producer_id=cfg["trainer_id"],
        producer_role="policy_authority",
        policy_id=parent_policy_id,
        window_id=window_id,
        lane="shared",
        commodity_use="dual_use",
        parent_ids=[
            parent_policy_id,
            checkpoint_artifact["artifact_id"],
            benchmark_result_artifact["artifact_id"],
        ],
        payload={
            "authority_mode": cfg["policy_authority_mode"],
            "approved": approved,
            "rationale": rationale,
            "signers": signers,
            "min_signers": cfg["policy_authority_min_signers"],
            "checkpoint_artifact_id": checkpoint_artifact["artifact_id"],
            "benchmark_result_artifact_id": benchmark_result_artifact["artifact_id"],
        },
        verification_state={
            "status": "final",
            "validator_ids": signers,
            "accepted": approved,
        },
    )


def promote_checkpoint_to_policy(
    cfg: dict[str, Any],
    *,
    checkpoint_artifact: dict[str, Any],
    window_id: int,
) -> dict[str, Any]:
    policy = publish_distill_policy(
        cfg,
        window_id=window_id,
        model_ref=checkpoint_artifact["payload"]["checkpoint_path"],
    )
    policy["parent_ids"] = [checkpoint_artifact["artifact_id"]]
    return policy
