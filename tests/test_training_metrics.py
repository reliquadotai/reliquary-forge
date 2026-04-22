"""Acceptance tests for the Forge training metrics + Grafana dashboard.

Covers the 14 acceptance cases from
``private/reliquary-plan/notes/spec-training-metrics.md``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from reliquary.training.metrics import (
    TRAINING_LOSS_WINDOW,
    TrainingMetrics,
    render_training_prometheus,
)

_DASHBOARD_PATH = (
    Path(__file__).resolve().parent.parent
    / "deploy"
    / "monitoring"
    / "grafana"
    / "dashboards"
    / "reliquary-training.json"
)

_ALLOWED_METRICS = {
    "reliquary_training_steps_total",
    "reliquary_training_samples_accepted_total",
    "reliquary_training_loss_last",
    "reliquary_training_loss_mean_window",
    "reliquary_training_grad_norm_last",
    "reliquary_training_advantage_variance_last",
    "reliquary_training_kl_last",
    "reliquary_training_step_seconds_last",
}
_METRIC_PATTERN = re.compile(r"reliquary_training_[a-z_]+")


def test_record_single_step_counters_advance() -> None:
    metrics = TrainingMetrics()
    metrics.record_step("run-A", loss=1.25, samples_accepted=4)
    output = render_training_prometheus(metrics)
    assert 'reliquary_training_steps_total{run_id="run-A"} 1' in output
    assert 'reliquary_training_samples_accepted_total{run_id="run-A"} 4' in output


def test_record_single_step_gauges_reflect_input() -> None:
    metrics = TrainingMetrics()
    metrics.record_step(
        "run-B",
        loss=0.875,
        grad_norm=2.5,
        advantage_variance=0.1,
        kl=0.05,
        step_seconds=1.75,
    )
    output = render_training_prometheus(metrics)
    assert 'reliquary_training_loss_last{run_id="run-B"} 0.875' in output
    assert 'reliquary_training_grad_norm_last{run_id="run-B"} 2.5' in output
    assert 'reliquary_training_advantage_variance_last{run_id="run-B"} 0.1' in output
    assert 'reliquary_training_kl_last{run_id="run-B"} 0.05' in output
    assert 'reliquary_training_step_seconds_last{run_id="run-B"} 1.75' in output


def test_optional_fields_skipped_when_absent() -> None:
    metrics = TrainingMetrics()
    metrics.record_step("run-C", loss=0.5)
    output = render_training_prometheus(metrics)
    assert 'reliquary_training_loss_last{run_id="run-C"} 0.5' in output
    assert 'reliquary_training_grad_norm_last{run_id="run-C"}' not in output
    assert 'reliquary_training_kl_last{run_id="run-C"}' not in output


def test_multiple_steps_loss_mean_window() -> None:
    metrics = TrainingMetrics()
    losses = [1.0, 2.0, 3.0, 4.0, 5.0]
    for loss in losses:
        metrics.record_step("run-D", loss=loss)
    snap = metrics.snapshot()
    assert snap["loss_mean_window"]["run-D"] == pytest.approx(sum(losses) / len(losses))


def test_loss_mean_window_caps_at_window_size() -> None:
    metrics = TrainingMetrics()
    for i in range(TRAINING_LOSS_WINDOW + 8):
        metrics.record_step("run-E", loss=float(i))
    snap = metrics.snapshot()
    expected_values = list(range(8, TRAINING_LOSS_WINDOW + 8))
    assert snap["loss_mean_window"]["run-E"] == pytest.approx(
        sum(expected_values) / len(expected_values)
    )


def test_run_id_isolation() -> None:
    metrics = TrainingMetrics()
    metrics.record_step("run-X", loss=1.0)
    metrics.record_step("run-X", loss=2.0)
    metrics.record_step("run-Y", loss=10.0)

    output = render_training_prometheus(metrics)
    assert 'reliquary_training_steps_total{run_id="run-X"} 2' in output
    assert 'reliquary_training_steps_total{run_id="run-Y"} 1' in output
    assert 'reliquary_training_loss_last{run_id="run-X"} 2.0' in output
    assert 'reliquary_training_loss_last{run_id="run-Y"} 10.0' in output


def test_nan_loss_rejected() -> None:
    metrics = TrainingMetrics()
    with pytest.raises(ValueError):
        metrics.record_step("run-F", loss=float("nan"))


def test_inf_loss_rejected() -> None:
    metrics = TrainingMetrics()
    with pytest.raises(ValueError):
        metrics.record_step("run-G", loss=float("inf"))


def test_nan_grad_norm_rejected() -> None:
    metrics = TrainingMetrics()
    with pytest.raises(ValueError):
        metrics.record_step("run-H", loss=0.5, grad_norm=float("nan"))


def test_negative_samples_rejected() -> None:
    metrics = TrainingMetrics()
    with pytest.raises(ValueError):
        metrics.record_step("run-I", loss=0.5, samples_accepted=-1)


def test_render_format_stable() -> None:
    m1 = TrainingMetrics()
    m2 = TrainingMetrics()
    for loss in [0.5, 0.4, 0.3]:
        m1.record_step("run-S", loss=loss, grad_norm=1.0, kl=0.01)
        m2.record_step("run-S", loss=loss, grad_norm=1.0, kl=0.01)
    assert render_training_prometheus(m1) == render_training_prometheus(m2)


def test_reset_clears_all() -> None:
    metrics = TrainingMetrics()
    metrics.record_step("run-R", loss=1.0, grad_norm=2.0)
    metrics.reset()
    snap = metrics.snapshot()
    assert snap["steps"] == {}
    assert snap["loss_last"] == {}
    assert snap["grad_norm_last"] == {}


def test_dashboard_json_parses() -> None:
    dashboard = json.loads(_DASHBOARD_PATH.read_text())
    assert dashboard["uid"] == "reliquary-training"
    assert "panels" in dashboard
    assert "schemaVersion" in dashboard


def test_dashboard_panels_present() -> None:
    dashboard = json.loads(_DASHBOARD_PATH.read_text())
    titles = {p.get("title") for p in dashboard["panels"]}
    assert "Loss" in titles
    assert "Advantage variance" in titles
    assert "KL divergence vs reference" in titles
    assert "Gradient norm" in titles
    assert "Step time" in titles


def test_dashboard_queries_reference_training_metrics() -> None:
    dashboard = json.loads(_DASHBOARD_PATH.read_text())
    for panel in dashboard["panels"]:
        for target in panel.get("targets", []):
            expr = target.get("expr", "")
            metrics = _METRIC_PATTERN.findall(expr)
            assert metrics, f"panel {panel['title']!r} has no metric reference"
            for metric in metrics:
                assert metric in _ALLOWED_METRICS, (
                    f"panel {panel['title']!r} references unknown metric {metric!r}"
                )
