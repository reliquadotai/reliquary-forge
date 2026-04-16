from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils.json_io import read_json, write_json


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _latest_artifact(registry, artifact_type: str) -> dict[str, Any] | None:
    artifacts = registry.list_artifacts(artifact_type)
    if not artifacts:
        return None
    return max(
        artifacts,
        key=lambda item: (int(item.get("window_id", -1)), str(item.get("created_at", ""))),
    )


def _latest_rl_training_batch(registry) -> dict[str, Any] | None:
    batches = [
        artifact
        for artifact in registry.list_artifacts("training_batch")
        if artifact.get("lane") == "rl"
    ]
    if not batches:
        return None
    return max(
        batches,
        key=lambda item: (int(item.get("window_id", -1)), str(item.get("created_at", ""))),
    )


def runtime_status_dir(cfg: dict[str, Any]) -> Path:
    return Path(str(cfg.get("state_root", "./data"))) / "runtime"


def rl_worker_status_path(cfg: dict[str, Any]) -> Path:
    return runtime_status_dir(cfg) / "rl_worker_status.json"


def read_rl_worker_status(cfg: dict[str, Any]) -> dict[str, Any] | None:
    path = rl_worker_status_path(cfg)
    if not path.exists():
        return None
    return read_json(path)


def write_rl_worker_status(cfg: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    current = read_rl_worker_status(cfg) or {}
    merged = {**current, **payload, "updated_at": _iso_now()}
    write_json(rl_worker_status_path(cfg), merged)
    return merged


def build_platform_status(cfg: dict[str, Any], registry) -> dict[str, Any]:
    latest_policy = registry.latest_policy()
    latest_benchmark = _latest_artifact(registry, "benchmark_result")
    latest_decision = _latest_artifact(registry, "policy_authority_decision")
    latest_rl_batch = _latest_rl_training_batch(registry)
    worker_status = read_rl_worker_status(cfg) or {}
    latest_imported_window = None
    latest_queue_depth = 0
    if latest_rl_batch is not None:
        source_window_ids = latest_rl_batch.get("payload", {}).get("source_window_ids", [])
        latest_imported_window = int(max(source_window_ids)) if source_window_ids else None
        latest_queue_depth = int(latest_rl_batch.get("payload", {}).get("training_eligible_count", 0))

    last_success_at = _parse_timestamp(worker_status.get("last_success_at"))
    last_failure_at = _parse_timestamp(worker_status.get("last_failure_at"))
    now = datetime.now(timezone.utc).timestamp()

    return {
        "host_mode": str(cfg.get("host_mode", "unknown")),
        "node_label": str(cfg.get("node_label", "unknown")),
        "service_version": str(cfg.get("service_version", "unknown")),
        "artifact_backend": str(cfg.get("artifact_registry_backend", "unknown")),
        "rl_backend": str(cfg.get("rl_learner_backend", "unknown")),
        "ray_cluster_profile": str(cfg.get("ray_cluster_profile", "unknown")),
        "latest_policy_window": int(latest_policy.get("window_id", -1)) if latest_policy else None,
        "latest_benchmark_delta": (
            float(latest_benchmark.get("payload", {}).get("holdout_metric", -1.0))
            if latest_benchmark is not None
            else None
        ),
        "latest_checkpoint_approved": (
            bool(latest_decision.get("payload", {}).get("approved"))
            if latest_decision is not None
            else None
        ),
        "latest_imported_source_window": latest_imported_window,
        "latest_rl_queue_depth": latest_queue_depth,
        "rl_worker": {
            "state": str(worker_status.get("state", "unknown")),
            "next_window": worker_status.get("next_window"),
            "configured_source_window": worker_status.get("configured_source_window"),
            "last_imported_source_window": worker_status.get("last_imported_source_window"),
            "last_success_window": worker_status.get("last_success_window"),
            "last_success_at": worker_status.get("last_success_at"),
            "last_success_age_seconds": (now - last_success_at) if last_success_at is not None else None,
            "last_failure_at": worker_status.get("last_failure_at"),
            "last_failure_age_seconds": (now - last_failure_at) if last_failure_at is not None else None,
            "last_decision_approved": worker_status.get("last_decision_approved"),
            "last_holdout_metric": worker_status.get("last_holdout_metric"),
            "waiting_for_data": bool(worker_status.get("waiting_for_data", False)),
            "consecutive_failures": int(worker_status.get("consecutive_failures", 0)),
            "cycle_count": int(worker_status.get("cycle_count", 0)),
            "last_error": worker_status.get("last_error"),
            "updated_at": worker_status.get("updated_at"),
        },
    }
