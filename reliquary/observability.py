from __future__ import annotations

from typing import Any


def _metric_lines(name: str, value: float, *, help_text: str, labels: dict[str, str] | None = None) -> list[str]:
    label_text = ""
    if labels:
        def _escape(value: str) -> str:
            return value.replace("\\", "\\\\").replace('"', '\\"')

        encoded = ",".join(
            f'{key}="{_escape(str(val))}"'
            for key, val in sorted(labels.items())
        )
        label_text = f"{{{encoded}}}"
    return [
        f"# HELP {name} {help_text}",
        f"# TYPE {name} gauge",
        f"{name}{label_text} {value}",
    ]


def _latest_artifact(registry, artifact_type: str) -> dict[str, Any] | None:
    artifacts = registry.list_artifacts(artifact_type)
    if not artifacts:
        return None
    return max(
        artifacts,
        key=lambda item: (int(item.get("window_id", -1)), str(item.get("created_at", ""))),
    )


def _recent_scorecard_totals(registry, limit: int = 5) -> dict[str, float]:
    scorecards = sorted(
        registry.list_artifacts("scorecard"),
        key=lambda item: (int(item.get("window_id", -1)), str(item.get("created_at", ""))),
        reverse=True,
    )[:limit]
    totals = {
        "accepted": 0.0,
        "training_eligible": 0.0,
        "contamination_failures": 0.0,
        "format_failures": 0.0,
        "proof_failures": 0.0,
    }
    for scorecard in scorecards:
        metrics = scorecard.get("payload", {}).get("window_metrics", {})
        totals["accepted"] += float(metrics.get("accepted_count", 0))
        totals["training_eligible"] += float(metrics.get("training_eligible_count", 0))
        totals["contamination_failures"] += float(metrics.get("contamination_failures", 0))
        totals["format_failures"] += float(metrics.get("format_failures", 0))
        totals["proof_failures"] += float(metrics.get("proof_failures", 0))
    return totals


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


def _decision_counts(registry) -> tuple[float, float]:
    approvals = 0.0
    rejections = 0.0
    for artifact in registry.list_artifacts("policy_authority_decision"):
        if bool(artifact.get("payload", {}).get("approved")):
            approvals += 1.0
        else:
            rejections += 1.0
    return approvals, rejections


def build_metrics_text(cfg: dict[str, Any], registry) -> str:
    latest_policy = registry.latest_policy()
    latest_benchmark = _latest_artifact(registry, "benchmark_result")
    latest_decision = _latest_artifact(registry, "policy_authority_decision")
    latest_rl_batch = _latest_rl_training_batch(registry)
    recent_totals = _recent_scorecard_totals(registry)
    approvals, rejections = _decision_counts(registry)

    benchmark_delta = -1.0
    if latest_benchmark is not None:
        benchmark_delta = float(latest_benchmark.get("payload", {}).get("holdout_metric", -1.0))
    latest_policy_window = float(latest_policy.get("window_id", -1)) if latest_policy else -1.0
    latest_decision_approved = (
        1.0 if latest_decision and latest_decision.get("payload", {}).get("approved") else 0.0
    )

    lines: list[str] = []
    lines.extend(
        _metric_lines(
            "reliquary_control_plane_up",
            1.0,
            help_text="Control plane liveness indicator.",
        )
    )
    lines.extend(
        _metric_lines(
            "reliquary_platform_info",
            1.0,
            help_text="Runtime labels for the broader Reliquary platform.",
            labels={
                "host_mode": str(cfg.get("host_mode", "unknown")),
                "node_label": str(cfg.get("node_label", "unknown")),
                "service_version": str(cfg.get("service_version", "unknown")),
                "artifact_backend": str(cfg.get("artifact_registry_backend", "unknown")),
                "rl_backend": str(cfg.get("rl_learner_backend", "unknown")),
                "ray_cluster_profile": str(cfg.get("ray_cluster_profile", "unknown")),
            },
        )
    )
    latest_imported_window = -1.0
    latest_queue_depth = 0.0
    if latest_rl_batch is not None:
        source_window_ids = latest_rl_batch.get("payload", {}).get("source_window_ids", [])
        latest_imported_window = float(max(source_window_ids)) if source_window_ids else -1.0
        latest_queue_depth = float(latest_rl_batch.get("payload", {}).get("training_eligible_count", 0))
    metrics = {
        "reliquary_latest_policy_window": (
            latest_policy_window,
            "Latest published policy window in the registry.",
        ),
        "reliquary_latest_benchmark_delta": (
            benchmark_delta,
            "Latest recorded holdout benchmark delta.",
        ),
        "reliquary_latest_checkpoint_approved": (
            latest_decision_approved,
            "Whether the latest checkpoint promotion decision was approved.",
        ),
        "reliquary_recent_accepted_traces": (
            recent_totals["accepted"],
            "Accepted traces across recent scorecards.",
        ),
        "reliquary_recent_training_eligible_traces": (
            recent_totals["training_eligible"],
            "Training-eligible traces across recent scorecards.",
        ),
        "reliquary_recent_contamination_failures": (
            recent_totals["contamination_failures"],
            "Recent contamination failures across scorecards.",
        ),
        "reliquary_recent_format_failures": (
            recent_totals["format_failures"],
            "Recent format failures across scorecards.",
        ),
        "reliquary_recent_proof_failures": (
            recent_totals["proof_failures"],
            "Recent proof failures across scorecards.",
        ),
        "reliquary_latest_imported_source_window": (
            latest_imported_window,
            "Latest finalized inference window imported into the RL lane.",
        ),
        "reliquary_latest_rl_queue_depth": (
            latest_queue_depth,
            "Training-eligible trajectory count in the latest RL training batch.",
        ),
        "reliquary_checkpoint_approval_count": (
            approvals,
            "Approved checkpoint promotions recorded by policy authority.",
        ),
        "reliquary_checkpoint_rejection_count": (
            rejections,
            "Rejected checkpoint promotions recorded by policy authority.",
        ),
    }
    for name, (value, help_text) in metrics.items():
        lines.extend(_metric_lines(name, value, help_text=help_text))
    return "\n".join(lines) + "\n"
