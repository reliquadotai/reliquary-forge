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


def build_metrics_text(cfg: dict[str, Any], registry) -> str:
    latest_policy = registry.latest_policy()
    latest_benchmark = _latest_artifact(registry, "benchmark_result")
    latest_decision = _latest_artifact(registry, "policy_authority_decision")
    recent_totals = _recent_scorecard_totals(registry)

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
            },
        )
    )
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
    }
    for name, (value, help_text) in metrics.items():
        lines.extend(_metric_lines(name, value, help_text=help_text))
    return "\n".join(lines) + "\n"
