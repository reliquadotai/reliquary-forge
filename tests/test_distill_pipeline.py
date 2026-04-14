from pathlib import Path

from reliquary.protocol.artifacts import make_artifact
from reliquary.roles.generation import generate_distill_traces
from reliquary.roles.learner import train_distill_window
from reliquary.roles.packer import build_distill_pack
from reliquary.roles.policy_authority import publish_distill_policy
from reliquary.roles.tasking import generate_reasoning_task_batch_artifact
from reliquary.roles.validator import validate_distill_window
from reliquary.showcase.distill_orchestrator import run_distill_showcase
from reliquary.showcase.profiles import get_demo_profile
from reliquary.storage.local_registry import LocalArtifactRegistry


def _cfg(tmp_path: Path) -> dict:
    return {
        "data_dir": str(tmp_path / "data"),
        "trainer_id": "trainer",
        "validator_id": "validator",
        "miner_id": "miner",
        "miner_secret": "default-secret",
        "miner_secrets": {
            "sage": "sage-secret",
            "echo": "echo-secret",
            "leak": "leak-secret",
        },
        "teacher_backend": "local_reasoning_teacher",
        "vllm_base_url": "",
        "vllm_model": "",
        "distill_student_model_ref": "reasoning-student://bootstrap",
        "distill_holdout_count": 4,
        "distill_train_task_count": 4,
        "distill_learner_backend": "local_reasoning_student",
        "policy_authority_mode": "trusted_quorum",
        "policy_authority_quorum": "trainer,validator",
        "policy_authority_min_signers": 2,
    }


def test_distill_pack_and_local_student_checkpoint(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    registry = LocalArtifactRegistry(cfg["data_dir"])
    policy = publish_distill_policy(cfg, window_id=0)
    registry.put_artifact(policy)
    task_batch = generate_reasoning_task_batch_artifact(cfg, policy, window_id=0, count=4)
    registry.put_artifact(task_batch)

    miner_profiles = [
        {"miner_id": "sage", "archetype": "reasoned_exact", "strategy": "reasoned_exact", "miner_secret": "sage-secret"},
        {"miner_id": "echo", "archetype": "reasoned_duplicate", "strategy": "reasoned_duplicate", "miner_secret": "echo-secret"},
        {"miner_id": "leak", "archetype": "reasoned_contaminated", "strategy": "reasoned_contaminated", "miner_secret": "leak-secret"},
    ]
    traces = []
    for miner in miner_profiles:
        artifacts = generate_distill_traces(cfg, policy, task_batch, window_id=0, miner_profile=miner)
        for artifact in artifacts:
            registry.put_artifact(artifact)
        traces.extend(artifacts)

    verdicts, scorecard = validate_distill_window(cfg, policy, traces, window_id=0)
    for artifact in verdicts:
        registry.put_artifact(artifact)
    registry.put_artifact(scorecard)
    assert scorecard["payload"]["window_metrics"]["accepted_count"] >= 1
    assert scorecard["payload"]["window_metrics"]["contamination_failures"] >= 1

    pack = build_distill_pack(registry, cfg, policy, verdicts, window_id=0)
    assert Path(pack["payload"]["export_refs"]["jsonl"]).exists()
    assert Path(pack["payload"]["export_refs"]["parquet"]).exists()

    summary = train_distill_window(registry, cfg, policy, pack, window_id=0)
    assert summary["checkpoint_artifact"]["artifact_type"] == "checkpoint"
    assert summary["benchmark_result_artifact"]["payload"]["eval_after"]["success_rate"] >= summary["benchmark_result_artifact"]["payload"]["eval_before"]["success_rate"]
    assert "contract_score" in summary["benchmark_result_artifact"]["payload"]["eval_after"]
    assert summary["decision_artifact"]["payload"]["approved"] is True


def test_distill_showcase_ignores_stale_registry_policy(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    registry = LocalArtifactRegistry(cfg["data_dir"])
    stale_policy = publish_distill_policy(
        cfg,
        window_id=4,
        model_ref=str(tmp_path / "missing-checkpoint"),
    )
    registry.put_artifact(stale_policy)

    profile = get_demo_profile("distill", windows=1, challenge_count=2)
    profile["miners"] = profile["miners"][:1]
    result = run_distill_showcase(
        registry,
        cfg,
        profile,
        run_id="distill-regression",
    )

    window_policy = result["window_records"][0]["policy_artifact"]
    assert window_policy["artifact_id"] != stale_policy["artifact_id"]
    assert window_policy["payload"]["model_ref"] == cfg["distill_student_model_ref"]


def test_distill_rejection_keeps_current_policy(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    registry = LocalArtifactRegistry(cfg["data_dir"])
    policy = publish_distill_policy(cfg, window_id=0)
    registry.put_artifact(policy)
    pack = make_artifact(
        artifact_type="distill_pack",
        producer_id="packer",
        producer_role="packer",
        policy_id=policy["artifact_id"],
        window_id=0,
        lane="distill",
        commodity_use="distill_only",
        parent_ids=[policy["artifact_id"]],
        payload={
            "pack_id": "pack-0",
            "record_count": 1,
            "export_refs": {"jsonl": str(tmp_path / "pack.jsonl"), "parquet": str(tmp_path / "pack.parquet")},
        },
    )

    class FakeBackend:
        name = "fake"

        def train(self, **kwargs):
            checkpoint_artifact = make_artifact(
                artifact_type="checkpoint",
                producer_id=cfg["trainer_id"],
                producer_role="learner",
                policy_id=policy["artifact_id"],
                window_id=1,
                lane="distill",
                commodity_use="dual_use",
                parent_ids=[policy["artifact_id"]],
                payload={
                    "checkpoint_id": "checkpoint-1",
                    "checkpoint_path": str(tmp_path / "checkpoint-1"),
                    "base_model": policy["payload"]["model_ref"],
                    "adapter_type": "lora",
                    "train_recipe": {"backend": "fake"},
                    "git_sha": "test",
                    "window_id": 1,
                },
            )
            benchmark_result_artifact = make_artifact(
                artifact_type="benchmark_result",
                producer_id=cfg["validator_id"],
                producer_role="benchmark_validator",
                policy_id=policy["artifact_id"],
                window_id=0,
                lane="distill",
                commodity_use="dual_use",
                parent_ids=[policy["artifact_id"]],
                payload={
                    "backend": "fake",
                    "model_id": "checkpoint-1",
                    "eval_set_version": "reasoning-holdout-v1",
                    "node_label": "test-node",
                    "git_sha": "test",
                    "eval_before": {"success_rate": 1.0},
                    "eval_after": {"success_rate": 0.0},
                    "holdout_metric": -1.0,
                    "metric_summary": {},
                },
            )
            return {
                "checkpoint_artifact": checkpoint_artifact,
                "benchmark_result_artifact": benchmark_result_artifact,
                "eval_before": {"success_rate": 1.0},
                "eval_after": {"success_rate": 0.0},
            }

    monkeypatch.setattr("reliquary.roles.learner.build_distill_learner_backend", lambda name: FakeBackend())
    summary = train_distill_window(registry, cfg, policy, pack, window_id=0)
    assert summary["decision_artifact"]["payload"]["approved"] is False
    assert summary["next_policy_artifact"]["payload"]["model_ref"] == policy["payload"]["model_ref"]
