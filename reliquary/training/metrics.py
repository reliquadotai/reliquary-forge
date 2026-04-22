"""Prometheus training metrics for the GRPO inner loop.

Thread-safe collector + text-exposition renderer. The trainer calls
``TrainingMetrics.record_step(run_id, loss=..., grad_norm=..., ...)`` once
per optimizer step; a scrape endpoint calls ``render_training_prometheus``
to emit the 0.0.4 exposition document.

Spec: private/reliquary-plan/notes/spec-training-metrics.md.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

TRAINING_LOSS_WINDOW = 32


_STEPS_TOTAL = "reliquary_training_steps_total"
_SAMPLES_ACCEPTED_TOTAL = "reliquary_training_samples_accepted_total"
_LOSS_LAST = "reliquary_training_loss_last"
_LOSS_MEAN_WINDOW = "reliquary_training_loss_mean_window"
_GRAD_NORM_LAST = "reliquary_training_grad_norm_last"
_ADVANTAGE_VARIANCE_LAST = "reliquary_training_advantage_variance_last"
_KL_LAST = "reliquary_training_kl_last"
_STEP_SECONDS_LAST = "reliquary_training_step_seconds_last"


def _check_finite(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


@dataclass
class TrainingMetrics:
    """Thread-safe per-run training-loop collector."""

    steps: dict[str, int] = field(default_factory=dict)
    samples_accepted: dict[str, int] = field(default_factory=dict)
    loss_last: dict[str, float] = field(default_factory=dict)
    loss_window: dict[str, Deque[float]] = field(default_factory=dict)
    loss_mean_window: dict[str, float] = field(default_factory=dict)
    grad_norm_last: dict[str, float] = field(default_factory=dict)
    advantage_variance_last: dict[str, float] = field(default_factory=dict)
    kl_last: dict[str, float] = field(default_factory=dict)
    step_seconds_last: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_step(
        self,
        run_id: str,
        *,
        loss: float,
        grad_norm: float | None = None,
        advantage_variance: float | None = None,
        kl: float | None = None,
        step_seconds: float | None = None,
        samples_accepted: int = 0,
    ) -> None:
        """Fold one step's telemetry into the collector."""
        loss_val = _check_finite("loss", loss)
        grad_norm_val = _check_finite("grad_norm", grad_norm)
        adv_var_val = _check_finite("advantage_variance", advantage_variance)
        kl_val = _check_finite("kl", kl)
        step_seconds_val = _check_finite("step_seconds", step_seconds)
        if not isinstance(samples_accepted, int) or isinstance(samples_accepted, bool):
            raise ValueError(f"samples_accepted must be int, got {samples_accepted!r}")
        if samples_accepted < 0:
            raise ValueError(
                f"samples_accepted must be >= 0, got {samples_accepted!r}"
            )

        with self._lock:
            self.steps[run_id] = self.steps.get(run_id, 0) + 1
            self.samples_accepted[run_id] = (
                self.samples_accepted.get(run_id, 0) + samples_accepted
            )
            self.loss_last[run_id] = loss_val  # type: ignore[assignment]
            window = self.loss_window.setdefault(
                run_id, deque(maxlen=TRAINING_LOSS_WINDOW)
            )
            window.append(loss_val)
            self.loss_mean_window[run_id] = sum(window) / len(window)
            if grad_norm_val is not None:
                self.grad_norm_last[run_id] = grad_norm_val
            if adv_var_val is not None:
                self.advantage_variance_last[run_id] = adv_var_val
            if kl_val is not None:
                self.kl_last[run_id] = kl_val
            if step_seconds_val is not None:
                self.step_seconds_last[run_id] = step_seconds_val

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {
                "steps": dict(self.steps),
                "samples_accepted": dict(self.samples_accepted),
                "loss_last": dict(self.loss_last),
                "loss_mean_window": dict(self.loss_mean_window),
                "grad_norm_last": dict(self.grad_norm_last),
                "advantage_variance_last": dict(self.advantage_variance_last),
                "kl_last": dict(self.kl_last),
                "step_seconds_last": dict(self.step_seconds_last),
            }

    def reset(self) -> None:
        with self._lock:
            self.steps.clear()
            self.samples_accepted.clear()
            self.loss_last.clear()
            self.loss_window.clear()
            self.loss_mean_window.clear()
            self.grad_norm_last.clear()
            self.advantage_variance_last.clear()
            self.kl_last.clear()
            self.step_seconds_last.clear()


def _escape(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace('"', '\\"')
    )


def _emit_gauge(
    lines: list[str], name: str, help_text: str, values: dict[str, float]
) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} gauge")
    for run_id, value in sorted(values.items()):
        lines.append(f'{name}{{run_id="{_escape(run_id)}"}} {value!r}')


def _emit_counter(
    lines: list[str], name: str, help_text: str, values: dict[str, int]
) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} counter")
    for run_id, value in sorted(values.items()):
        lines.append(f'{name}{{run_id="{_escape(run_id)}"}} {value}')


def render_training_prometheus(metrics: TrainingMetrics) -> str:
    """Render the collector as Prometheus 0.0.4 text exposition."""
    snap = metrics.snapshot()
    lines: list[str] = []

    _emit_counter(
        lines, _STEPS_TOTAL, "Total optimizer steps per training run.", snap["steps"]
    )
    _emit_counter(
        lines,
        _SAMPLES_ACCEPTED_TOTAL,
        "Cumulative training-eligible samples consumed by the GRPO loop.",
        snap["samples_accepted"],
    )
    _emit_gauge(
        lines, _LOSS_LAST, "Loss at the most recent optimizer step.", snap["loss_last"]
    )
    _emit_gauge(
        lines,
        _LOSS_MEAN_WINDOW,
        "Rolling mean loss over the last TRAINING_LOSS_WINDOW steps.",
        snap["loss_mean_window"],
    )
    _emit_gauge(
        lines,
        _GRAD_NORM_LAST,
        "Gradient L2 norm at the most recent step (if reported).",
        snap["grad_norm_last"],
    )
    _emit_gauge(
        lines,
        _ADVANTAGE_VARIANCE_LAST,
        "Variance of per-sample advantages in the step micro-batch (if reported).",
        snap["advantage_variance_last"],
    )
    _emit_gauge(
        lines,
        _KL_LAST,
        "KL divergence vs reference policy at the most recent step (if reported).",
        snap["kl_last"],
    )
    _emit_gauge(
        lines,
        _STEP_SECONDS_LAST,
        "Wall-clock seconds for the most recent optimizer step (if reported).",
        snap["step_seconds_last"],
    )

    return "\n".join(lines) + "\n"


__all__ = [
    "TRAINING_LOSS_WINDOW",
    "TrainingMetrics",
    "render_training_prometheus",
]
