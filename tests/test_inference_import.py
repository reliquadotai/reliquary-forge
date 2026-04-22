from __future__ import annotations

import gzip
import json
from pathlib import Path

from reliquary.protocol.artifacts import make_artifact
from reliquary.roles.trainer import (
    import_inference_window,
    publish_policy,
    run_reasoning_rl_cycle,
    train_window,
)
from reliquary.storage.local_registry import LocalArtifactRegistry
from reliquary.training.backends import build_rl_learner_backend


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_gzip_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(payload).encode("utf-8")))


def _cfg(tmp_path: Path) -> dict:
    return {
        "data_dir": str(tmp_path / "reliquary"),
        "artifact_dir": str(tmp_path / "reliquary"),
        "trainer_id": "trainer",
        "validator_id": "validator",
        "model_name": "reasoning-student://bootstrap",
        "device": "cpu",
        "max_new_tokens": 48,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "train_lr": 1e-5,
        "train_epochs": 1,
        "clip_eps": 0.2,
        "kl_coef": 0.0,
        "rl_env_id": "reasoning",
        "rl_task_source": "reasoning_tasks",
        "rl_actor_model_ref": "reasoning-student://bootstrap",
        "rl_reference_model_ref": "reasoning-student://bootstrap",
        "rl_learner_backend": "trl_async_grpo_fsdp2",
        "rl_backend_execution_mode": "local_compat",
        "rl_reasoning_eval_count": 4,
        "policy_authority_mode": "trusted_quorum",
        "policy_authority_quorum": "trainer,validator",
        "policy_authority_min_signers": 2,
        "node_label": "test-node",
        "git_sha": "test-sha",
        "service_version": "0.3.0",
        "inference_registry_backend": "local",
        "inference_registry_local_root": str(tmp_path / "inference-artifacts"),
        "inference_import_cursor_name": "reasoning-rl",
        "ray_cluster_profile": "single-node",
        "ray_head_address": "local",
        "ray_actor_replicas": 1,
        "ray_learner_replicas": 1,
        "ray_evaluator_replicas": 1,
        "rl_vllm_actor_endpoints": "",
        "rl_holdout_metric_min": 0.0,
        "rl_contract_score_min": 0.0,
        "sat_n": 4,
        "sat_k": 3,
        "sat_m": 17,
    }


def _seed_inference_window(root: Path, *, window_id: int = 5) -> None:
    task = {
        "task_id": "reasoning-train-1",
        "prompt": "Solve 2 + 3",
        "source_prompt": "Solve 2 + 3",
        "operation": "add",
        "inputs": {"left": 2, "right": 3},
        "reference_answer": "5",
        "difficulty": 0.55,
        "tags": ["operation:add"],
    }
    task_batch = {
        "artifact_id": "task-batch-1",
        "artifact_type": "task_batch",
        "producer_id": "validator",
        "producer_role": "task_source",
        "window_id": window_id,
        "created_at": "2026-04-15T10:00:00+00:00",
        "parent_ids": [],
        "payload": {
            "task_source": "reasoning_tasks",
            "window_id": window_id,
            "public_randomness": "abcd",
            "model_ref": "Qwen/Qwen2.5-1.5B-Instruct",
            "tasks": [task],
        },
    }
    completion = {
        "artifact_id": "completion-1",
        "artifact_type": "completion",
        "producer_id": "miner-a",
        "producer_role": "miner",
        "window_id": window_id,
        "created_at": "2026-04-15T10:00:01+00:00",
        "parent_ids": [],
        "payload": {
            "task_id": task["task_id"],
            "task_source": "reasoning_tasks",
            "task_index": 0,
            "tokens": [1, 2, 3, 4],
            "prompt_token_ids": [1, 2],
            "prompt_length": 2,
            "completion_token_ids": [3, 4],
            "completion_logprobs": [-0.1, -0.2],
            "old_sum_logprob": -0.3,
            "completion_text": "Reasoning: add\nFinal Answer: 5",
            "completion_digest": "digest-1",
            "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "nonce": 123,
            "proof_version": "v5",
            "layer_index": -1,
            "commitments": [[0.1], [0.2], [0.3], [0.4]],
            "signature": "sig",
            "signature_scheme": "local_hmac",
            "signer_id": "miner-a",
            "randomness": "abcd",
            "upload_ref": {},
            "sample_index": 0,
            "contamination_tags": [],
        },
    }
    verdict = {
        "artifact_id": "verdict-1",
        "artifact_type": "verdict",
        "producer_id": "validator",
        "producer_role": "validator",
        "window_id": window_id,
        "created_at": "2026-04-15T10:00:02+00:00",
        "parent_ids": ["completion-1", "task-batch-1"],
        "payload": {
            "completion_id": "completion-1",
            "accepted": True,
            "hard_fail_reason": None,
            "soft_fail_reason": None,
            "proof_summary": {"checked_positions": 2, "passed_positions": 2, "signature_status": "ok"},
            "task_binding_summary": {"reason": "ok"},
            "copycat_summary": {"status": "pending"},
            "task_source": "reasoning_tasks",
            "task_id": task["task_id"],
            "task_index": 0,
            "difficulty": 0.55,
            "task_tags": task["tags"],
            "contamination_tags": [],
            "contamination_overlap_tags": [],
            "final_answer": "5",
            "correctness": 1.0,
            "policy_compliance": 1.0,
            "format_ok": True,
            "format_reason": "ok",
            "semantic_evaluation": {
                "final_answer": "5",
                "correctness_or_judge": 1.0,
                "policy_compliance": 1.0,
                "format_ok": True,
                "format_reason": "ok",
                "accepted": True,
            },
        },
    }
    scorecard = {
        "artifact_id": "scorecard-1",
        "artifact_type": "scorecard",
        "producer_id": "validator",
        "producer_role": "validator",
        "window_id": window_id,
        "created_at": "2026-04-15T10:00:03+00:00",
        "parent_ids": [],
        "payload": {
            "window_id": window_id,
            "task_source": "reasoning_tasks",
            "weights": {"miner-a": 1.0},
            "miner_totals": {"miner-a": {"accepted": 1}},
            "verification_totals": {"submitted": 1, "accepted": 1, "hard_failed": 0, "soft_failed": 0},
            "window_metrics": {
                "submitted_count": 1,
                "accepted_count": 1,
                "training_eligible_count": 1,
                "reasoning_eval_count": 1,
                "reasoning_correct_total": 1.0,
                "reasoning_format_ok_total": 1.0,
                "reasoning_policy_compliance_total": 1.0,
            },
        },
    }
    completion_bundle_ref = {
        "backend": "local",
        "path": str(root / "completion_bundles" / f"window-{window_id:08d}" / "miner-a.json.gz"),
        "miner_id": "miner-a",
        "uploaded_at": 1.0,
    }
    verdict_bundle_ref = {
        "backend": "local",
        "path": str(root / "verdict_bundles" / f"window-{window_id:08d}" / "validator.json.gz"),
        "validator_id": "validator",
        "uploaded_at": 2.0,
    }
    manifest = {
        "artifact_id": "manifest-1",
        "artifact_type": "window_manifest",
        "producer_id": "validator",
        "producer_role": "validator",
        "window_id": window_id,
        "created_at": "2026-04-15T10:00:04+00:00",
        "parent_ids": ["task-batch-1", "scorecard-1"],
        "payload": {
            "task_batch_id": "task-batch-1",
            "completion_bundle_refs": [completion_bundle_ref],
            "verdict_bundle_ref": verdict_bundle_ref,
            "scorecard_id": "scorecard-1",
            "task_source": "reasoning_tasks",
            "window_randomness_ref": {"block_hash": "0x01", "public_randomness": "abcd"},
            "chain_publish_result": {"success": True, "uids": [103]},
        },
    }
    _write_json(root / "task_batches" / "task-batch-1.json", task_batch)
    _write_json(root / "scorecards" / "scorecard-1.json", scorecard)
    _write_json(root / "window_manifests" / "manifest-1.json", manifest)
    _write_gzip_json(Path(completion_bundle_ref["path"]), [completion])
    _write_gzip_json(Path(verdict_bundle_ref["path"]), [verdict])


def test_import_inference_window_materializes_reasoning_trajectories(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _seed_inference_window(Path(cfg["inference_registry_local_root"]))
    registry = LocalArtifactRegistry(cfg["artifact_dir"])
    policy = publish_policy(cfg, window_id=0, env_id="reasoning")
    registry.put_artifact(policy)

    summary = import_inference_window(
        registry,
        cfg,
        policy,
        source_window_id=5,
        target_window_id=0,
    )

    assert len(summary["rollout_artifacts"]) == 1
    assert len(summary["verdict_artifacts"]) == 1
    assert summary["training_batch_artifact"]["payload"]["source_window_ids"] == [5]
    assert summary["verdict_artifacts"][0]["payload"]["dense_reward"] > 0.0
    assert summary["verdict_artifacts"][0]["payload"]["generation"]["completion_token_ids"] == [3, 4]
    assert summary["verdict_artifacts"][0]["payload"]["source_registry_ref"]["window_id"] == 5

    repeated = import_inference_window(
        registry,
        cfg,
        policy,
        source_window_id=5,
        target_window_id=0,
    )
    assert repeated["training_batch_artifact"]["artifact_id"] == summary["training_batch_artifact"]["artifact_id"]
    assert len(registry.list_artifacts("training_batch", window_id=0)) == 1


def test_run_reasoning_rl_cycle_promotes_imported_checkpoint(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _seed_inference_window(Path(cfg["inference_registry_local_root"]))
    registry = LocalArtifactRegistry(cfg["artifact_dir"])

    result = run_reasoning_rl_cycle(registry, cfg, source_window_id=5, window_id=0)

    decision = result["training_summary"]["decision_artifact"]
    next_policy = result["training_summary"]["next_policy_artifact"]
    assert decision["payload"]["approved"] is True
    assert next_policy["payload"]["env_id"] == "reasoning"
    assert next_policy["payload"]["model_ref"].startswith(str(Path(cfg["artifact_dir"]) / "checkpoints"))


def test_rl_training_falls_back_to_previous_policy_when_benchmark_regresses(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    registry = LocalArtifactRegistry(cfg["artifact_dir"])
    policy = publish_policy(cfg, window_id=0, env_id="reasoning")
    registry.put_artifact(policy)
    verdict = make_artifact(
        artifact_type="verdict",
        producer_id="validator",
        producer_role="validator",
        policy_id=policy["artifact_id"],
        window_id=0,
        lane="rl",
        commodity_use="dual_use",
        payload={
            "env_id": "reasoning",
            "task_source": "reasoning_tasks",
            "challenge_id": "task-1",
            "task_id": "task-1",
            "accepted": True,
            "training_eligible": True,
            "score": 1.0,
            "dense_reward": 1.0,
            "challenge": {"prompt": "Add 1 and 2", "task": {"operation": "add"}},
            "generation": {
                "sequence_token_ids": [1, 2, 3],
                "prompt_token_ids": [1],
                "completion_token_ids": [2, 3],
                "completion_text": "Reasoning: add\nFinal Answer: 3",
                "completion_logprobs": [-0.1, -0.2],
                "sum_logprob": -0.3,
            },
        },
    )

    class FakeBackend:
        name = "trl_async_grpo_fsdp2"

        def train(self, **kwargs):
            output_dir = kwargs["output_dir"]
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_json(output_dir / "reasoning_capabilities.json", {"supported_operations": []})
            return {"checkpoint_path": str(output_dir), "execution_mode": "local_compat", "cluster_profile": {"control_plane": "ray"}}

    eval_states = iter(
        [
            {"contract_score": 1.0, "success_rate": 1.0},
            {"contract_score": 0.0, "success_rate": 0.0},
        ]
    )
    monkeypatch.setattr("reliquary.roles.trainer.build_rl_learner_backend", lambda name: FakeBackend())
    monkeypatch.setattr("reliquary.roles.trainer._evaluate_rl_model", lambda **kwargs: next(eval_states))

    summary = train_window(registry, cfg, policy, [verdict], window_id=0)

    assert summary["decision_artifact"]["payload"]["approved"] is False
    assert summary["next_policy_artifact"]["payload"]["model_ref"] == policy["payload"]["model_ref"]


def test_dense_backend_ray_cluster_mode_uses_remote_runner(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    cfg["rl_backend_execution_mode"] = "ray_cluster"
    cfg["rl_vllm_actor_endpoints"] = "http://127.0.0.1:8100"
    groups = [
        {
            "challenge_id": "task-1",
            "prompt": "Add 2 and 3",
            "env_id": "reasoning",
            "task_source": "reasoning_tasks",
            "samples": [
                {
                    "full_token_ids": [1, 2, 3],
                    "prompt_token_ids": [1],
                    "completion_token_ids": [2, 3],
                    "completion_text": "Reasoning: add\nFinal Answer: 5",
                    "old_logprobs": [-0.1, -0.2],
                    "old_sum_logprob": -0.3,
                    "reward": 1.0,
                    "binary_score": 1.0,
                    "advantage": 0.0,
                    "task_metadata": {"operation": "add"},
                }
            ],
        }
    ]
    calls: list[dict] = []

    def _fake_remote(**kwargs):
        calls.append(kwargs)
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "reasoning_capabilities.json", {"supported_operations": ["add"], "record_count": 1})
        _write_json(output_dir / "training_summary.json", {"status": "ok", "steps": 2})
        return {
            "accepted_samples": 1,
            "groups": 1,
            "mean_loss": 0.1,
            "steps": 2,
            "status": "ok",
            "checkpoint_path": str(output_dir),
        }

    monkeypatch.setattr("reliquary.training.backends.run_grpo_training_remote", _fake_remote)

    backend = build_rl_learner_backend("trl_async_grpo_fsdp2")
    summary = backend.train(
        model_ref="reasoning-student://bootstrap",
        groups=groups,
        output_dir=tmp_path / "ray-cluster-backend",
        device="cpu",
        cfg=cfg,
    )

    assert calls
    assert calls[0]["ray_address"] == "local"
    assert summary["execution_mode"] == "ray_cluster"
    assert summary["cluster_profile"]["actor_endpoints"] == ["http://127.0.0.1:8100"]
    assert Path(summary["run_spec_path"]).exists()


def test_train_window_uses_remote_reasoning_evaluator_in_ray_cluster_mode(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    cfg["rl_backend_execution_mode"] = "ray_cluster"
    cfg["rl_vllm_actor_endpoints"] = "http://127.0.0.1:8100"
    registry = LocalArtifactRegistry(cfg["artifact_dir"])
    policy = publish_policy(cfg, window_id=0, env_id="reasoning")
    registry.put_artifact(policy)
    verdict = make_artifact(
        artifact_type="verdict",
        producer_id="validator",
        producer_role="validator",
        policy_id=policy["artifact_id"],
        window_id=0,
        lane="rl",
        commodity_use="dual_use",
        payload={
            "env_id": "reasoning",
            "task_source": "reasoning_tasks",
            "challenge_id": "task-1",
            "task_id": "task-1",
            "accepted": True,
            "training_eligible": True,
            "score": 1.0,
            "dense_reward": 1.0,
            "challenge": {"prompt": "Add 1 and 2", "task": {"operation": "add"}},
            "generation": {
                "sequence_token_ids": [1, 2, 3],
                "prompt_token_ids": [1],
                "completion_token_ids": [2, 3],
                "completion_text": "Reasoning: add\nFinal Answer: 3",
                "completion_logprobs": [-0.1, -0.2],
                "sum_logprob": -0.3,
            },
        },
    )

    class FakeBackend:
        name = "trl_async_grpo_fsdp2"

        def train(self, **kwargs):
            output_dir = kwargs["output_dir"]
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_json(output_dir / "reasoning_capabilities.json", {"supported_operations": ["add"]})
            return {
                "checkpoint_path": str(output_dir),
                "execution_mode": "ray_cluster",
                "cluster_profile": {"control_plane": "ray"},
            }

    eval_calls: list[str] = []
    eval_states = iter(
        [
            {
                "answer_accuracy": 0.8,
                "accepted_rate": 0.8,
                "format_rate": 0.8,
                "policy_compliance": 0.8,
                "contract_score": 0.8,
            },
            {
                "answer_accuracy": 0.9,
                "accepted_rate": 0.9,
                "format_rate": 0.9,
                "policy_compliance": 0.9,
                "contract_score": 0.9,
            },
        ]
    )

    def _fake_remote_eval(**kwargs):
        eval_calls.append(kwargs["model_ref"])
        return next(eval_states)

    monkeypatch.setattr("reliquary.roles.trainer.build_rl_learner_backend", lambda name: FakeBackend())
    monkeypatch.setattr("reliquary.roles.trainer.evaluate_reasoning_checkpoint_remote", _fake_remote_eval)

    summary = train_window(registry, cfg, policy, [verdict], window_id=0)

    assert len(eval_calls) == 2
    assert summary["benchmark_result_artifact"]["payload"]["holdout_metric"] > 0.0
    assert summary["decision_artifact"]["payload"]["approved"] is True


def test_dense_and_nemo_backends_share_training_contract(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    groups = [
        {
            "challenge_id": "task-1",
            "prompt": "Add 2 and 3",
            "env_id": "reasoning",
            "task_source": "reasoning_tasks",
            "samples": [
                {
                    "full_token_ids": [1, 2, 3],
                    "prompt_token_ids": [1],
                    "completion_token_ids": [2, 3],
                    "completion_text": "Reasoning: add\nFinal Answer: 5",
                    "old_logprobs": [-0.1, -0.2],
                    "old_sum_logprob": -0.3,
                    "reward": 1.0,
                    "binary_score": 1.0,
                    "advantage": 0.0,
                    "task_metadata": {"operation": "add"},
                }
            ],
        }
    ]
    for backend_name in ("trl_async_grpo_fsdp2", "nemo_rl"):
        backend = build_rl_learner_backend(backend_name)
        output_dir = tmp_path / backend_name
        summary = backend.train(
            model_ref="reasoning-student://bootstrap",
            groups=groups,
            output_dir=output_dir,
            device="cpu",
            cfg=cfg,
        )
        assert summary["checkpoint_path"] == str(output_dir)
        assert Path(summary["run_spec_path"]).exists()
