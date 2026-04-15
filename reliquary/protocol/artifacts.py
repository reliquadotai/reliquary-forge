from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..utils.json_io import sha256_json


Artifact = dict[str, Any]

ARTIFACT_DIRECTORIES = {
    "policy": "policies",
    "challenge": "challenges",
    "task_batch": "task_batches",
    "rollout": "rollouts",
    "distill_trace": "distill_traces",
    "verdict": "verdicts",
    "checkpoint": "checkpoints",
    "distill_pack": "distill_packs",
    "training_batch": "training_batches",
    "scorecard": "scorecards",
    "policy_authority_decision": "policy_authority_decisions",
    "benchmark_result": "benchmark_results",
    "window_manifest": "window_manifests",
    "run_manifest": "run_manifests",
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact_directory_name(artifact_type: str) -> str:
    return ARTIFACT_DIRECTORIES.get(artifact_type, f"{artifact_type}s")


def default_lane_for_artifact(artifact_type: str) -> str:
    if artifact_type in {"rollout", "training_batch"}:
        return "rl"
    if artifact_type in {"distill_trace", "distill_pack"}:
        return "distill"
    if artifact_type in {"challenge"}:
        return "rl"
    if artifact_type in {"task_batch"}:
        return "distill"
    return "shared"


def default_commodity_use_for_artifact(artifact_type: str) -> str:
    if artifact_type in {"rollout", "training_batch", "challenge"}:
        return "rl_only"
    if artifact_type in {"distill_trace", "distill_pack", "task_batch"}:
        return "distill_only"
    return "dual_use"


def make_artifact(
    artifact_type: str,
    producer_id: str,
    policy_id: str,
    window_id: int,
    payload: dict[str, Any],
    signature_bundle: dict[str, Any] | None = None,
    *,
    lane: str | None = None,
    producer_role: str | None = None,
    parent_ids: list[str] | None = None,
    storage_refs: list[dict[str, Any]] | None = None,
    verification_state: dict[str, Any] | None = None,
    commodity_use: str | None = None,
    created_at: str | None = None,
) -> Artifact:
    artifact = {
        "artifact_type": artifact_type,
        "lane": lane or default_lane_for_artifact(artifact_type),
        "producer_id": producer_id,
        "producer_role": producer_role or "unknown",
        "policy_id": policy_id,
        "window_id": window_id,
        "created_at": created_at or _timestamp(),
        "parent_ids": list(parent_ids or []),
        "payload_hash": sha256_json(payload),
        "storage_refs": list(storage_refs or []),
        "verification_state": verification_state
        or {
            "status": "pending",
            "validator_ids": [],
            "accepted": None,
        },
        "commodity_use": commodity_use or default_commodity_use_for_artifact(artifact_type),
        "payload": payload,
        "signature_bundle": signature_bundle or {"scheme": "none"},
    }
    artifact["artifact_id"] = sha256_json(artifact)
    return artifact


def make_policy_artifact(
    producer_id: str,
    window_id: int,
    model_ref: str,
    env_id: str,
    env_config: dict[str, Any],
    generation_config: dict[str, Any],
    trainer_config: dict[str, Any],
    lane_permissions: list[str] | None = None,
    evaluation_contract: dict[str, Any] | None = None,
    *,
    task_source: str | None = None,
    reward_contract: dict[str, Any] | None = None,
    evaluator_suite: dict[str, Any] | None = None,
    learner_backend: str | None = None,
    actor_model_ref: str | None = None,
    reference_model_ref: str | None = None,
    promotion_thresholds: dict[str, Any] | None = None,
    source_registry: dict[str, Any] | None = None,
    cluster_profile: dict[str, Any] | None = None,
) -> Artifact:
    payload = {
        "policy_version": "v1",
        "model_ref": model_ref,
        "actor_model_ref": actor_model_ref or model_ref,
        "reference_model_ref": reference_model_ref or model_ref,
        "env_id": env_id,
        "task_source": task_source,
        "env_config": env_config,
        "generation_config": generation_config,
        "trainer_config": trainer_config,
        "reward_contract": reward_contract or {"mode": "validator_defined"},
        "evaluator_suite": evaluator_suite or {"name": "default"},
        "learner_backend": learner_backend or trainer_config.get("backend"),
        "promotion_thresholds": promotion_thresholds or {"holdout_metric_min": 0.0},
        "source_registry": source_registry or {},
        "cluster_profile": cluster_profile or {},
        "lane_permissions": list(lane_permissions or ["rl", "distill"]),
        "evaluation_contract": evaluation_contract or {"mode": "validator_defined"},
        "allowed_artifact_types": [
            "challenge",
            "task_batch",
            "rollout",
            "distill_trace",
            "verdict",
            "distill_pack",
            "training_batch",
            "scorecard",
            "checkpoint",
            "policy_authority_decision",
            "benchmark_result",
            "window_manifest",
            "run_manifest",
        ],
    }
    return make_artifact(
        artifact_type="policy",
        producer_id=producer_id,
        policy_id="pending",
        window_id=window_id,
        payload=payload,
        lane="shared",
        producer_role="policy_authority",
        commodity_use="dual_use",
    )


def finalize_policy_artifact(policy_artifact: Artifact) -> Artifact:
    policy = dict(policy_artifact)
    canonical = {k: v for k, v in policy.items() if k not in {"artifact_id", "policy_id"}}
    policy_id = sha256_json(canonical)
    policy["policy_id"] = policy_id
    policy["artifact_id"] = policy_id
    return policy
