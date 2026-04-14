from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from ..roles.policy_authority import make_policy_authority_decision, promote_checkpoint_to_policy, publish_distill_policy
from ..shared.bootstrap import bootstrap_runtime

app = FastAPI(title="Reliquary Control Plane", version="0.2.0")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/artifacts/{artifact_type}")
def list_artifacts(artifact_type: str, window_id: int | None = None) -> dict[str, Any]:
    _, registry = bootstrap_runtime()
    artifacts = registry.list_artifacts(artifact_type, window_id=window_id)
    return {"artifacts": artifacts}


@app.get("/artifacts/{artifact_type}/{artifact_id}")
def get_artifact(artifact_type: str, artifact_id: str) -> dict[str, Any]:
    _, registry = bootstrap_runtime()
    artifact = registry.get_artifact(artifact_type, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return artifact


@app.get("/policies/latest")
def latest_policy() -> dict[str, Any]:
    _, registry = bootstrap_runtime()
    policy = registry.latest_policy()
    if policy is None:
        raise HTTPException(status_code=404, detail="No policy found")
    return policy


@app.post("/policy-authority/publish")
def publish_policy(window_id: int, lane: str = "distill") -> dict[str, Any]:
    cfg, registry = bootstrap_runtime()
    if lane != "distill":
        raise HTTPException(status_code=400, detail="Only distill policy publication is supported by the control plane")
    policy = publish_distill_policy(cfg, window_id=window_id)
    registry.put_artifact(policy)
    return policy


@app.post("/policy-authority/promote")
def promote_policy(payload: dict[str, Any]) -> dict[str, Any]:
    cfg, registry = bootstrap_runtime()
    checkpoint_artifact = registry.get_artifact("checkpoint", payload["checkpoint_artifact_id"])
    benchmark_result_artifact = registry.get_artifact("benchmark_result", payload["benchmark_result_artifact_id"])
    parent_policy_id = payload["parent_policy_id"]
    if checkpoint_artifact is None or benchmark_result_artifact is None:
        raise HTTPException(status_code=404, detail="Checkpoint or benchmark result not found")
    approved = bool(payload.get("approved", True))
    decision = make_policy_authority_decision(
        cfg,
        window_id=int(payload["window_id"]),
        parent_policy_id=parent_policy_id,
        checkpoint_artifact=checkpoint_artifact,
        benchmark_result_artifact=benchmark_result_artifact,
        approved=approved,
        rationale=str(payload.get("rationale", "approved_by_control_plane")),
    )
    registry.put_artifact(decision)
    next_policy = None
    if approved:
        next_policy = promote_checkpoint_to_policy(
            cfg,
            checkpoint_artifact=checkpoint_artifact,
            window_id=int(payload["next_window_id"]),
        )
        registry.put_artifact(next_policy)
    return {"decision": decision, "next_policy": next_policy}
