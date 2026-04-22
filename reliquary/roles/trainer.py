from __future__ import annotations

from typing import Any

from ..envs.sat import generate_sat_challenge
from ..inference_adapter import InferenceRegistryAdapter
from ..protocol.artifacts import finalize_policy_artifact, make_artifact, make_policy_artifact
from ..storage.registry import ArtifactRegistry
from ..training.backends import build_rl_learner_backend, evaluate_reasoning_checkpoint
from ..training.eval import evaluate_policy
from ..training.grpo import build_grpo_groups
from ..training.ray_cluster import evaluate_reasoning_checkpoint_remote
from ..utils.json_io import write_json
from .policy_authority import make_policy_authority_decision


def _active_env_id(cfg: dict[str, Any], env_id: str | None = None) -> str:
    return env_id or str(cfg.get("rl_env_id", "sat"))


def _cluster_profile(cfg: dict[str, Any]) -> dict[str, Any]:
    actor_endpoints = [
        endpoint.strip()
        for endpoint in str(cfg.get("rl_vllm_actor_endpoints", "")).split(",")
        if endpoint.strip()
    ]
    return {
        "control_plane": "ray",
        "profile": str(cfg.get("ray_cluster_profile", "single-node")),
        "head_address": str(cfg.get("ray_head_address", "local")),
        "actor_replicas": int(cfg.get("ray_actor_replicas", 1)),
        "learner_replicas": int(cfg.get("ray_learner_replicas", 1)),
        "evaluator_replicas": int(cfg.get("ray_evaluator_replicas", 1)),
        "actor_endpoints": actor_endpoints,
    }


def _reasoning_reward_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "weighted_reasoning",
        "weights": {
            "correctness": float(cfg.get("rl_reward_weight_correctness", 0.55)),
            "policy_compliance": float(cfg.get("rl_reward_weight_policy_compliance", 0.20)),
            "format_ok": float(cfg.get("rl_reward_weight_format_ok", 0.15)),
            "accepted": float(cfg.get("rl_reward_weight_accepted", 0.10)),
        },
        "difficulty_scaling": True,
    }


def _publish_rl_policy(
    cfg: dict[str, Any],
    *,
    window_id: int,
    model_ref: str,
    env_id: str,
    task_source: str | None = None,
    reference_model_ref: str | None = None,
) -> dict[str, Any]:
    if env_id == "reasoning":
        policy = make_policy_artifact(
            producer_id=cfg["trainer_id"],
            window_id=window_id,
            model_ref=model_ref,
            actor_model_ref=model_ref,
            reference_model_ref=reference_model_ref or model_ref,
            env_id="reasoning",
            task_source=task_source or str(cfg.get("rl_task_source", "reasoning_tasks")),
            env_config={
                "task_family": "reasoning_trajectories",
                "task_source": task_source or str(cfg.get("rl_task_source", "reasoning_tasks")),
                "eval_count": int(cfg.get("rl_reasoning_eval_count", cfg.get("distill_holdout_count", 8))),
            },
            generation_config={
                "max_new_tokens": cfg["max_new_tokens"],
                "temperature": cfg["temperature"],
                "top_p": cfg["top_p"],
                "top_k": cfg["top_k"],
                "repetition_penalty": cfg["repetition_penalty"],
                "vllm_actor_endpoints": [
                    endpoint.strip()
                    for endpoint in str(cfg.get("rl_vllm_actor_endpoints", "")).split(",")
                    if endpoint.strip()
                ],
            },
            trainer_config={
                "backend": cfg["rl_learner_backend"],
                "ray_cluster_profile": cfg.get("ray_cluster_profile", "single-node"),
                "git_sha": cfg.get("git_sha", "unknown"),
                "service_version": cfg.get("service_version", "0.1.0"),
            },
            lane_permissions=["rl"],
            evaluation_contract={
                "lane": "rl",
                "metric": "reasoning_contract_score",
                "eval_count": int(cfg.get("rl_reasoning_eval_count", cfg.get("distill_holdout_count", 8))),
            },
            reward_contract=_reasoning_reward_contract(cfg),
            evaluator_suite={
                "name": "reasoning_rl_default",
                "metrics": [
                    "answer_accuracy",
                    "accepted_rate",
                    "format_rate",
                    "policy_compliance",
                    "contract_score",
                ],
            },
            learner_backend=str(cfg.get("rl_learner_backend", "trl_async_grpo_fsdp2")),
            promotion_thresholds={
                "holdout_metric_min": float(cfg.get("rl_holdout_metric_min", 0.0)),
                "contract_score_min": float(cfg.get("rl_contract_score_min", 0.0)),
            },
            source_registry={
                "backend": str(cfg.get("inference_registry_backend", "local")),
                "local_root": str(cfg.get("inference_registry_local_root", "")),
                "bucket": str(cfg.get("inference_registry_bucket", "")),
            },
            cluster_profile=_cluster_profile(cfg),
        )
        return finalize_policy_artifact(policy)
    policy = make_policy_artifact(
        producer_id=cfg["trainer_id"],
        window_id=window_id,
        model_ref=model_ref,
        env_id="sat",
        env_config={"n": cfg["sat_n"], "k": cfg["sat_k"], "m": cfg["sat_m"]},
        generation_config={
            "max_new_tokens": cfg["max_new_tokens"],
            "temperature": cfg["temperature"],
            "top_p": cfg["top_p"],
            "top_k": cfg["top_k"],
            "repetition_penalty": cfg["repetition_penalty"],
        },
        trainer_config={
            "lr": cfg["train_lr"],
            "epochs": cfg["train_epochs"],
            "clip_eps": cfg["clip_eps"],
            "kl_coef": cfg["kl_coef"],
        },
        lane_permissions=["rl"],
        evaluation_contract={
            "lane": "rl",
            "metric": "sat_success_with_dense_reward",
            "eval_count": 8,
        },
    )
    return finalize_policy_artifact(policy)


def publish_policy(
    cfg: dict[str, Any],
    window_id: int,
    model_ref: str | None = None,
    *,
    env_id: str | None = None,
    task_source: str | None = None,
    reference_model_ref: str | None = None,
) -> dict[str, Any]:
    active_env = _active_env_id(cfg, env_id)
    default_model_ref = (
        cfg.get("rl_actor_model_ref")
        or cfg.get("model_student")
        or cfg["model_name"]
    )
    return _publish_rl_policy(
        cfg,
        window_id=window_id,
        model_ref=model_ref or str(default_model_ref),
        env_id=active_env,
        task_source=task_source,
        reference_model_ref=reference_model_ref or cfg.get("rl_reference_model_ref"),
    )


def generate_challenge_artifacts(
    cfg: dict[str, Any],
    policy_artifact: dict[str, Any],
    window_id: int,
    count: int,
    seed_offset: int = 0,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for offset in range(count):
        challenge = generate_sat_challenge(
            n=cfg["sat_n"],
            k=cfg["sat_k"],
            m=cfg["sat_m"],
            seed=seed_offset + window_id * 10_000 + offset,
        )
        artifacts.append(
            make_artifact(
                artifact_type="challenge",
                producer_id=cfg["trainer_id"],
                policy_id=policy_artifact["artifact_id"],
                window_id=window_id,
                payload={"challenge": challenge},
            )
        )
    return artifacts


def _evaluate_rl_model(*, cfg: dict[str, Any], policy_artifact: dict[str, Any], model_ref: str) -> dict[str, Any]:
    env_id = str(policy_artifact["payload"].get("env_id", "sat"))
    if env_id == "reasoning":
        execution_mode = str(cfg.get("rl_backend_execution_mode", "local_compat")).strip().lower()
        if execution_mode == "ray_cluster":
            return evaluate_reasoning_checkpoint_remote(
                model_ref=model_ref,
                holdout_count=int(cfg.get("rl_reasoning_eval_count", cfg.get("distill_holdout_count", 8))),
                cfg=cfg,
            )
        return evaluate_reasoning_checkpoint(
            model_ref=model_ref,
            holdout_count=int(cfg.get("rl_reasoning_eval_count", cfg.get("distill_holdout_count", 8))),
            cfg=cfg,
        )
    return evaluate_policy(
        model_ref=model_ref,
        device=cfg["device"],
        sat_n=cfg["sat_n"],
        sat_k=cfg["sat_k"],
        sat_m=cfg["sat_m"],
        count=8,
        max_new_tokens=cfg["max_new_tokens"],
    )


def _holdout_metric(policy_artifact: dict[str, Any], before_eval: dict[str, Any], after_eval: dict[str, Any]) -> float:
    metric_name = str(policy_artifact["payload"].get("evaluation_contract", {}).get("metric", ""))
    if metric_name == "reasoning_contract_score":
        return float(after_eval.get("contract_score", 0.0)) - float(before_eval.get("contract_score", 0.0))
    return float(after_eval.get("mean_dense_reward", 0.0)) - float(before_eval.get("mean_dense_reward", 0.0))


def import_inference_window(
    registry: ArtifactRegistry,
    cfg: dict[str, Any],
    policy_artifact: dict[str, Any],
    *,
    source_window_id: int | None = None,
    target_window_id: int,
    cursor_name: str | None = None,
) -> dict[str, Any]:
    adapter = InferenceRegistryAdapter(cfg, registry.cache_dir())
    cursor_name = cursor_name or str(cfg.get("inference_import_cursor_name", "reasoning_rl"))
    next_window = source_window_id if source_window_id is not None else adapter.next_finalized_window(cursor_name=cursor_name)
    if next_window is None:
        raise FileNotFoundError("No finalized inference window available for import")
    return adapter.import_reasoning_window(
        registry=registry,
        policy_artifact=policy_artifact,
        target_window_id=target_window_id,
        source_window_id=int(next_window),
        cursor_name=cursor_name,
    )


def _rl_training_batch_payload(
    *,
    verdict_artifacts: list[dict[str, Any]],
    policy_artifact: dict[str, Any],
    source_training_batch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    groups = build_grpo_groups(verdict_artifacts)
    env_id = str(policy_artifact["payload"].get("env_id", "sat"))
    training_verdicts = [
        verdict for verdict in verdict_artifacts if verdict["payload"].get("training_eligible", False)
    ]
    payload = {
        "env_id": env_id,
        "task_source": policy_artifact["payload"].get("task_source"),
        "source_verdict_ids": [verdict["artifact_id"] for verdict in training_verdicts],
        "challenge_ids": sorted(
            {
                verdict["payload"].get("challenge_id", verdict["payload"].get("task_id"))
                for verdict in training_verdicts
            }
        ),
        "group_count": len(groups),
        "accepted_verdict_count": len([item for item in verdict_artifacts if item["payload"].get("accepted")]),
        "training_eligible_count": len(training_verdicts),
        "mean_dense_reward": (
            sum(float(verdict["payload"]["dense_reward"]) for verdict in training_verdicts)
            / len(training_verdicts)
            if training_verdicts
            else 0.0
        ),
        "mean_binary_score": (
            sum(float(verdict["payload"]["score"]) for verdict in training_verdicts)
            / len(training_verdicts)
            if training_verdicts
            else 0.0
        ),
        "reward_contract": policy_artifact["payload"].get("reward_contract", {}),
    }
    if source_training_batch is not None:
        payload["source_registry"] = source_training_batch["payload"].get("source_registry", {})
        payload["source_window_ids"] = source_training_batch["payload"].get("source_window_ids", [])
        payload["imported_source_artifact_ids"] = source_training_batch["payload"].get("source_artifact_ids", {})
    return payload


def train_window(
    registry: ArtifactRegistry,
    cfg: dict[str, Any],
    policy_artifact: dict[str, Any],
    verdict_artifacts: list[dict[str, Any]],
    window_id: int,
    *,
    source_training_batch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    groups = build_grpo_groups(verdict_artifacts)
    training_dir = registry.training_dir(window_id)
    write_json(training_dir / "grpo_groups.json", groups)
    training_batch_artifact = make_artifact(
        artifact_type="training_batch",
        producer_id=cfg["trainer_id"],
        producer_role="learner",
        policy_id=policy_artifact["artifact_id"],
        window_id=window_id,
        lane="rl",
        commodity_use="dual_use",
        parent_ids=[policy_artifact["artifact_id"], *[verdict["artifact_id"] for verdict in verdict_artifacts]],
        payload=_rl_training_batch_payload(
            verdict_artifacts=verdict_artifacts,
            policy_artifact=policy_artifact,
            source_training_batch=source_training_batch,
        ),
    )
    registry.put_artifact(training_batch_artifact)

    before_eval = _evaluate_rl_model(
        cfg=cfg,
        policy_artifact=policy_artifact,
        model_ref=policy_artifact["payload"]["model_ref"],
    )

    checkpoint_id = f"checkpoint-window-{window_id + 1:04d}"
    checkpoint_dir = registry.checkpoint_dir(checkpoint_id)
    learner_backend = build_rl_learner_backend(cfg["rl_learner_backend"])
    training_summary = learner_backend.train(
        model_ref=policy_artifact["payload"]["model_ref"],
        groups=groups,
        output_dir=checkpoint_dir,
        device=cfg["device"],
        cfg=cfg,
        policy_artifact=policy_artifact,
        training_batch_artifact=training_batch_artifact,
        lr=cfg.get("train_lr"),
        clip_eps=cfg.get("clip_eps"),
        kl_coef=cfg.get("kl_coef"),
        epochs=cfg.get("train_epochs"),
    )
    checkpoint_path = str(training_summary.get("checkpoint_path", checkpoint_dir))
    write_json(checkpoint_dir / "training_summary.json", training_summary)

    after_eval = _evaluate_rl_model(
        cfg=cfg,
        policy_artifact=policy_artifact,
        model_ref=checkpoint_path,
    )
    holdout_metric = _holdout_metric(policy_artifact, before_eval, after_eval)
    benchmark_artifact = make_artifact(
        artifact_type="benchmark_result",
        producer_id=cfg["validator_id"],
        producer_role="benchmark_validator",
        policy_id=policy_artifact["artifact_id"],
        window_id=window_id,
        lane="rl",
        commodity_use="dual_use",
        parent_ids=[policy_artifact["artifact_id"], training_batch_artifact["artifact_id"]],
        payload={
            "backend": learner_backend.name,
            "model_id": checkpoint_id,
            "eval_set_version": f"{policy_artifact['payload'].get('env_id', 'sat')}-holdout-v1",
            "node_label": cfg.get("node_label", "local-node"),
            "git_sha": cfg.get("git_sha", "unknown"),
            "env_id": policy_artifact["payload"].get("env_id"),
            "task_source": policy_artifact["payload"].get("task_source"),
            "cluster_profile": _cluster_profile(cfg),
            "eval_before": before_eval,
            "eval_after": after_eval,
            "holdout_metric": holdout_metric,
            "metric_summary": {
                "before_success_rate": before_eval.get("success_rate", before_eval.get("answer_accuracy", 0.0)),
                "after_success_rate": after_eval.get("success_rate", after_eval.get("answer_accuracy", 0.0)),
                "before_contract_score": before_eval.get("contract_score", before_eval.get("mean_dense_reward", 0.0)),
                "after_contract_score": after_eval.get("contract_score", after_eval.get("mean_dense_reward", 0.0)),
            },
        },
    )
    registry.put_artifact(benchmark_artifact)

    checkpoint_artifact = make_artifact(
        artifact_type="checkpoint",
        producer_id=cfg["trainer_id"],
        producer_role="learner",
        policy_id=policy_artifact["artifact_id"],
        window_id=window_id + 1,
        lane="rl",
        commodity_use="dual_use",
        parent_ids=[policy_artifact["artifact_id"], training_batch_artifact["artifact_id"]],
        payload={
            "checkpoint_id": checkpoint_id,
            "checkpoint_path": checkpoint_path,
            "parent_policy_id": policy_artifact["artifact_id"],
            "base_model": policy_artifact["payload"]["model_ref"],
            "adapter_type": "full_checkpoint",
            "train_recipe": {
                "backend": learner_backend.name,
                "execution_mode": training_summary.get("execution_mode"),
                "group_count": len(groups),
                "cluster_profile": _cluster_profile(cfg),
            },
            "git_sha": cfg.get("git_sha", "unknown"),
            "window_id": window_id + 1,
            "training_backend": learner_backend.name,
            "env_id": policy_artifact["payload"].get("env_id"),
            "task_source": policy_artifact["payload"].get("task_source"),
            "reward_contract": policy_artifact["payload"].get("reward_contract", {}),
            "source_registry": training_batch_artifact["payload"].get("source_registry", {}),
            "source_window_ids": training_batch_artifact["payload"].get("source_window_ids", []),
            "cluster_profile": training_summary.get("cluster_profile", _cluster_profile(cfg)),
            "eval_before": before_eval,
            "eval_after": after_eval,
        },
    )
    registry.put_artifact(checkpoint_artifact)

    thresholds = policy_artifact["payload"].get("promotion_thresholds", {})
    approved = holdout_metric >= float(thresholds.get("holdout_metric_min", 0.0))
    rationale = "benchmark_and_quorum_passed" if approved else "benchmark_or_quorum_failed"
    decision_artifact = make_policy_authority_decision(
        cfg,
        window_id=window_id,
        parent_policy_id=policy_artifact["artifact_id"],
        checkpoint_artifact=checkpoint_artifact,
        benchmark_result_artifact=benchmark_artifact,
        approved=approved,
        rationale=rationale,
    )
    registry.put_artifact(decision_artifact)
    if approved:
        next_policy = _publish_rl_policy(
            cfg,
            window_id=window_id + 1,
            model_ref=checkpoint_path,
            env_id=str(policy_artifact["payload"].get("env_id", "sat")),
            task_source=policy_artifact["payload"].get("task_source"),
            reference_model_ref=policy_artifact["payload"].get("reference_model_ref"),
        )
        next_policy["parent_ids"] = [checkpoint_artifact["artifact_id"], decision_artifact["artifact_id"]]
    else:
        next_policy = _publish_rl_policy(
            cfg,
            window_id=window_id + 1,
            model_ref=policy_artifact["payload"]["model_ref"],
            env_id=str(policy_artifact["payload"].get("env_id", "sat")),
            task_source=policy_artifact["payload"].get("task_source"),
            reference_model_ref=policy_artifact["payload"].get("reference_model_ref"),
        )
        next_policy["parent_ids"] = [policy_artifact["artifact_id"], decision_artifact["artifact_id"]]
    registry.put_artifact(next_policy)

    summary = {
        "window_id": window_id,
        "training_batch_artifact": training_batch_artifact,
        "checkpoint_artifact": checkpoint_artifact,
        "benchmark_result_artifact": benchmark_artifact,
        "decision_artifact": decision_artifact,
        "next_policy_artifact": next_policy,
        "grpo_groups": len(groups),
        "eval_before": before_eval,
        "eval_after": after_eval,
        "training_summary": training_summary,
    }
    write_json(training_dir / "window_summary.json", summary)
    return summary


def run_reasoning_rl_cycle(
    registry: ArtifactRegistry,
    cfg: dict[str, Any],
    *,
    source_window_id: int | None = None,
    window_id: int = 0,
) -> dict[str, Any]:
    policy = publish_policy(
        cfg,
        window_id=window_id,
        env_id="reasoning",
        task_source=str(cfg.get("rl_task_source", "reasoning_tasks")),
    )
    registry.put_artifact(policy)
    imported = import_inference_window(
        registry,
        cfg,
        policy,
        source_window_id=source_window_id,
        target_window_id=window_id,
    )
    training_summary = train_window(
        registry,
        cfg,
        policy,
        imported["verdict_artifacts"],
        window_id=window_id,
        source_training_batch=imported["training_batch_artifact"],
    )
    return {
        "policy_artifact": policy,
        "import_summary": imported,
        "training_summary": training_summary,
    }
