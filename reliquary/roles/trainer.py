from __future__ import annotations

from pathlib import Path
from typing import Any

from ..envs.sat import generate_sat_challenge
from ..protocol.artifacts import finalize_policy_artifact, make_artifact, make_policy_artifact
from ..storage.registry import ArtifactRegistry
from ..training.backends import build_rl_learner_backend
from ..training.eval import evaluate_policy
from ..training.grpo import build_grpo_groups
from ..utils.json_io import write_json


def publish_policy(cfg: dict[str, Any], window_id: int, model_ref: str | None = None) -> dict[str, Any]:
    policy = make_policy_artifact(
        producer_id=cfg["trainer_id"],
        window_id=window_id,
        model_ref=model_ref or cfg["model_name"],
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


def train_window(
    registry: ArtifactRegistry,
    cfg: dict[str, Any],
    policy_artifact: dict[str, Any],
    verdict_artifacts: list[dict[str, Any]],
    window_id: int,
) -> dict[str, Any]:
    groups = build_grpo_groups(verdict_artifacts)
    training_dir = registry.training_dir(window_id)
    write_json(training_dir / "grpo_groups.json", groups)
    training_verdicts = [
        verdict for verdict in verdict_artifacts if verdict["payload"].get("training_eligible", False)
    ]
    training_batch_artifact = make_artifact(
        artifact_type="training_batch",
        producer_id=cfg["trainer_id"],
        policy_id=policy_artifact["artifact_id"],
        window_id=window_id,
        payload={
            "source_verdict_ids": [verdict["artifact_id"] for verdict in training_verdicts],
            "challenge_ids": sorted(
                {verdict["payload"]["challenge_id"] for verdict in training_verdicts}
            ),
            "group_count": len(groups),
            "accepted_verdict_count": len(training_verdicts),
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
        },
    )
    registry.put_artifact(training_batch_artifact)

    before_eval = evaluate_policy(
        model_ref=policy_artifact["payload"]["model_ref"],
        device=cfg["device"],
        sat_n=cfg["sat_n"],
        sat_k=cfg["sat_k"],
        sat_m=cfg["sat_m"],
        count=8,
        max_new_tokens=cfg["max_new_tokens"],
    )

    checkpoint_id = f"checkpoint-window-{window_id + 1:04d}"
    checkpoint_dir = registry.checkpoint_dir(checkpoint_id)
    learner_backend = build_rl_learner_backend(cfg["rl_learner_backend"])
    if learner_backend.name == "local_grpo":
        training_summary = learner_backend.train(
            model_ref=policy_artifact["payload"]["model_ref"],
            groups=groups,
            output_dir=checkpoint_dir,
            device=cfg["device"],
            lr=cfg["train_lr"],
            clip_eps=cfg["clip_eps"],
            kl_coef=cfg["kl_coef"],
            epochs=cfg["train_epochs"],
        )
    else:
        training_summary = learner_backend.train(
            model_ref=policy_artifact["payload"]["model_ref"],
            groups=groups,
            output_dir=checkpoint_dir,
            device=cfg["device"],
        )
        write_json(checkpoint_dir / "training_summary.json", training_summary)

    after_eval = evaluate_policy(
        model_ref=str(checkpoint_dir),
        device=cfg["device"],
        sat_n=cfg["sat_n"],
        sat_k=cfg["sat_k"],
        sat_m=cfg["sat_m"],
        count=8,
        max_new_tokens=cfg["max_new_tokens"],
    )

    checkpoint_artifact = make_artifact(
        artifact_type="checkpoint",
        producer_id=cfg["trainer_id"],
        policy_id=policy_artifact["artifact_id"],
        window_id=window_id + 1,
        payload={
            "checkpoint_id": checkpoint_id,
            "checkpoint_path": str(checkpoint_dir),
            "parent_policy_id": policy_artifact["artifact_id"],
            "training_summary": training_summary,
            "eval_before": before_eval,
            "eval_after": after_eval,
        },
    )
    registry.put_artifact(checkpoint_artifact)

    next_policy = publish_policy(cfg, window_id=window_id + 1, model_ref=str(checkpoint_dir))
    registry.put_artifact(next_policy)

    summary = {
        "window_id": window_id,
        "training_batch_artifact": training_batch_artifact,
        "checkpoint_artifact": checkpoint_artifact,
        "next_policy_artifact": next_policy,
        "grpo_groups": len(groups),
        "eval_before": before_eval,
        "eval_after": after_eval,
        "training_summary": training_summary,
    }
    write_json(training_dir / "window_summary.json", summary)
    return summary
