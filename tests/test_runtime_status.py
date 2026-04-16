import json
from pathlib import Path

from reliquary.protocol.artifacts import make_artifact
from reliquary.runtime_status import build_platform_status, read_rl_worker_status, write_rl_worker_status


def _seed_inference_manifest(root: Path, *, window_id: int) -> None:
    manifest = {
        "artifact_id": f"manifest-{window_id}",
        "artifact_type": "window_manifest",
        "producer_id": "validator",
        "producer_role": "validator",
        "window_id": window_id,
        "created_at": "2026-04-16T00:00:00Z",
        "parent_ids": [],
        "payload": {
            "task_batch_id": "task-batch-1",
            "completion_bundle_refs": [],
            "verdict_bundle_ref": None,
            "scorecard_id": "scorecard-1",
            "task_source": "reasoning_tasks",
            "window_randomness_ref": {"public_randomness": "abcd"},
            "chain_publish_result": {"success": True},
        },
    }
    path = root / "window_manifests" / f"manifest-{window_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")


class _FakeRegistry:
    def __init__(self) -> None:
        self._policy = make_artifact(
            artifact_type="policy",
            producer_id="authority",
            producer_role="policy_authority",
            policy_id="policy-1",
            window_id=7,
            lane="shared",
            payload={"model_ref": "reasoning-student://bootstrap"},
        )
        self._artifacts = {
            "benchmark_result": [
                make_artifact(
                    artifact_type="benchmark_result",
                    producer_id="validator",
                    producer_role="benchmark_validator",
                    policy_id="policy-1",
                    window_id=7,
                    lane="rl",
                    payload={"holdout_metric": 0.25},
                )
            ],
            "policy_authority_decision": [
                make_artifact(
                    artifact_type="policy_authority_decision",
                    producer_id="authority",
                    producer_role="policy_authority",
                    policy_id="policy-1",
                    window_id=7,
                    lane="shared",
                    payload={"approved": True},
                )
            ],
            "training_batch": [
                make_artifact(
                    artifact_type="training_batch",
                    producer_id="trainer",
                    producer_role="learner",
                    policy_id="policy-1",
                    window_id=7,
                    lane="rl",
                    payload={"source_window_ids": [6911280], "training_eligible_count": 5},
                )
            ],
        }

    def latest_policy(self):
        return self._policy

    def list_artifacts(self, artifact_type: str, window_id=None):
        artifacts = list(self._artifacts.get(artifact_type, []))
        if window_id is None:
            return artifacts
        return [artifact for artifact in artifacts if int(artifact["window_id"]) == int(window_id)]


def test_rl_worker_status_roundtrip_and_platform_summary(tmp_path: Path) -> None:
    cfg = {
        "state_root": str(tmp_path),
        "host_mode": "rtx6000b",
        "node_label": "trainer-node",
        "service_version": "0.2.0",
        "artifact_registry_backend": "local",
        "rl_learner_backend": "trl_async_grpo_fsdp2",
        "ray_cluster_profile": "single-node",
        "inference_registry_backend": "local",
        "inference_registry_local_root": str(tmp_path / "inference"),
    }
    _seed_inference_manifest(Path(cfg["inference_registry_local_root"]), window_id=6911284)
    write_rl_worker_status(
        cfg,
        {
            "state": "running",
            "next_window": 8,
            "last_success_window": 7,
            "last_imported_source_window": 6911280,
            "last_decision_approved": True,
            "last_holdout_metric": 0.25,
            "consecutive_failures": 0,
        },
    )

    saved = read_rl_worker_status(cfg)
    summary = build_platform_status(cfg, _FakeRegistry())

    assert saved is not None
    assert saved["next_window"] == 8
    assert summary["latest_policy_window"] == 7
    assert summary["latest_available_source_window"] == 6911284
    assert summary["import_lag_windows"] == 4.0
    assert summary["latest_imported_source_window"] == 6911280
    assert summary["rl_worker"]["next_window"] == 8
    assert summary["rl_worker"]["last_decision_approved"] is True
