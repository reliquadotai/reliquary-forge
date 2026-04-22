from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from ..observability import build_metrics_text
from ..roles.policy_authority import (
    make_policy_authority_decision,
    promote_checkpoint_to_policy,
    publish_distill_policy,
)
from ..roles.trainer import import_inference_window, run_reasoning_rl_cycle
from ..roles.trainer import publish_policy as publish_rl_policy
from ..runtime_status import build_platform_status
from ..shared.bootstrap import bootstrap_runtime

app = FastAPI(title="Reliquary Control Plane", version="0.2.0")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> PlainTextResponse:
    cfg, registry = bootstrap_runtime()
    return PlainTextResponse(
        content=build_metrics_text(cfg, registry),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/status")
def status() -> dict[str, Any]:
    cfg, registry = bootstrap_runtime()
    return build_platform_status(cfg, registry)


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


@app.post("/rl/import")
def import_reasoning_window(payload: dict[str, Any]) -> dict[str, Any]:
    cfg, registry = bootstrap_runtime()
    window_id = int(payload.get("window_id", 0))
    source_window_id = payload.get("source_window_id")
    policy = registry.latest_policy()
    if policy is None or policy["payload"].get("env_id") != "reasoning":
        policy = publish_rl_policy(cfg, window_id=window_id, env_id="reasoning")
        registry.put_artifact(policy)
    summary = import_inference_window(
        registry,
        cfg,
        policy,
        source_window_id=int(source_window_id) if source_window_id is not None else None,
        target_window_id=window_id,
    )
    return {
        "training_batch_artifact": summary["training_batch_artifact"],
        "verdict_count": len(summary["verdict_artifacts"]),
        "rollout_count": len(summary["rollout_artifacts"]),
    }


@app.post("/rl/cycle")
def run_rl_cycle(payload: dict[str, Any]) -> dict[str, Any]:
    cfg, registry = bootstrap_runtime()
    window_id = int(payload.get("window_id", 0))
    source_window_id = payload.get("source_window_id")
    result = run_reasoning_rl_cycle(
        registry,
        cfg,
        source_window_id=int(source_window_id) if source_window_id is not None else None,
        window_id=window_id,
    )
    return {
        "policy_artifact": result["policy_artifact"],
        "training_batch_artifact": result["import_summary"]["training_batch_artifact"],
        "decision_artifact": result["training_summary"]["decision_artifact"],
        "next_policy_artifact": result["training_summary"]["next_policy_artifact"],
    }
