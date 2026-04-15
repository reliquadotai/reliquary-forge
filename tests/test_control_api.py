from fastapi.testclient import TestClient

from reliquary.control.api import app
from reliquary.protocol.artifacts import make_artifact


class _FakeRegistry:
    def __init__(self) -> None:
        self._policy = make_artifact(
            artifact_type="policy",
            producer_id="authority",
            producer_role="policy_authority",
            policy_id="policy-1",
            window_id=4,
            lane="shared",
            payload={"model_ref": "reasoning-student://bootstrap"},
        )
        self._scorecard = make_artifact(
            artifact_type="scorecard",
            producer_id="validator",
            producer_role="validator",
            policy_id="policy-1",
            window_id=4,
            lane="distill",
            payload={
                "window_metrics": {
                    "accepted_count": 6,
                    "training_eligible_count": 4,
                    "contamination_failures": 1,
                    "format_failures": 2,
                    "proof_failures": 3,
                }
            },
        )
        self._decision = make_artifact(
            artifact_type="policy_authority_decision",
            producer_id="authority",
            producer_role="policy_authority",
            policy_id="policy-1",
            window_id=4,
            lane="shared",
            payload={"approved": True},
        )
        self._benchmark = make_artifact(
            artifact_type="benchmark_result",
            producer_id="validator",
            producer_role="benchmark_validator",
            policy_id="policy-1",
            window_id=4,
            lane="rl",
            payload={"holdout_metric": 0.125},
        )
        self._rl_batch = make_artifact(
            artifact_type="training_batch",
            producer_id="trainer",
            producer_role="learner",
            policy_id="policy-1",
            window_id=4,
            lane="rl",
            payload={"source_window_ids": [6911280], "training_eligible_count": 3},
        )
        self._artifacts = {
            "policy": [self._policy],
            "scorecard": [self._scorecard],
            "policy_authority_decision": [self._decision],
            "benchmark_result": [self._benchmark],
            "training_batch": [self._rl_batch],
        }

    def list_artifacts(self, artifact_type: str, window_id=None):
        artifacts = list(self._artifacts.get(artifact_type, []))
        if window_id is not None:
            return [artifact for artifact in artifacts if int(artifact["window_id"]) == int(window_id)]
        return artifacts

    def latest_policy(self):
        return self._policy

    def get_artifact(self, artifact_type: str, artifact_id: str):
        for artifact in self._artifacts.get(artifact_type, []):
            if artifact["artifact_id"] == artifact_id:
                return artifact
        return None


def test_metrics_endpoint_exports_platform_summary(monkeypatch) -> None:
    monkeypatch.setattr(
        "reliquary.control.api.bootstrap_runtime",
        lambda: (
            {
                "host_mode": "rtx6000b",
                "node_label": "staging-node",
                "service_version": "0.2.0",
                "artifact_registry_backend": "local",
            },
            _FakeRegistry(),
        ),
    )
    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "reliquary_control_plane_up 1.0" in response.text
    assert "reliquary_latest_benchmark_delta 0.125" in response.text
    assert "reliquary_recent_accepted_traces 6.0" in response.text
    assert "reliquary_latest_imported_source_window 6911280.0" in response.text
    assert "reliquary_latest_rl_queue_depth 3.0" in response.text


def test_rl_cycle_endpoint_returns_summary(monkeypatch) -> None:
    monkeypatch.setattr(
        "reliquary.control.api.bootstrap_runtime",
        lambda: (
            {},
            _FakeRegistry(),
        ),
    )
    monkeypatch.setattr(
        "reliquary.control.api.run_reasoning_rl_cycle",
        lambda registry, cfg, source_window_id=None, window_id=0: {
            "policy_artifact": {"artifact_id": "policy-1"},
            "import_summary": {"training_batch_artifact": {"artifact_id": "batch-1"}},
            "training_summary": {
                "decision_artifact": {"artifact_id": "decision-1"},
                "next_policy_artifact": {"artifact_id": "policy-2"},
            },
        },
    )
    client = TestClient(app)

    response = client.post("/rl/cycle", json={"window_id": 0, "source_window_id": 6911280})

    assert response.status_code == 200
    assert response.json()["training_batch_artifact"]["artifact_id"] == "batch-1"
