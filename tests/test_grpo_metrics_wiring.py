"""Unit tests for the metric-helper additions to grpo.py.

The full ``train_grpo_checkpoint`` entry point needs a real transformers
model bundle which is GPU-gated; these tests cover the pure-Python helpers
(``_advantage_variance``, ``_gradient_norm``) and the signature-level
contract (metrics + run_id are keyword-only, default None, no regression
when absent).
"""

from __future__ import annotations

import inspect

import pytest

from reliquary.training.grpo import (
    _advantage_variance,
    _gradient_norm,
    train_grpo_checkpoint,
)
from reliquary.training.metrics import TrainingMetrics


def test_advantage_variance_empty_group() -> None:
    assert _advantage_variance({"samples": []}) == 0.0


def test_advantage_variance_single_sample() -> None:
    assert _advantage_variance({"samples": [{"advantage": 2.5}]}) == 0.0


def test_advantage_variance_matches_hand_computation() -> None:
    group = {"samples": [{"advantage": 1.0}, {"advantage": 3.0}, {"advantage": 5.0}]}
    # mean = 3.0, variance = ((1-3)^2 + (3-3)^2 + (5-3)^2) / 3 = 8/3
    assert _advantage_variance(group) == pytest.approx(8.0 / 3.0)


def test_advantage_variance_all_equal_is_zero() -> None:
    group = {"samples": [{"advantage": 2.0} for _ in range(4)]}
    assert _advantage_variance(group) == 0.0


def test_gradient_norm_zero_when_no_grads() -> None:
    torch = pytest.importorskip("torch")

    linear = torch.nn.Linear(4, 2)
    assert _gradient_norm(linear) == 0.0


def test_gradient_norm_computes_l2_norm() -> None:
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    linear = torch.nn.Linear(4, 2)
    x = torch.randn(8, 4)
    target = torch.randn(8, 2)
    loss = ((linear(x) - target) ** 2).mean()
    loss.backward()

    expected = sum(
        float(p.grad.detach().pow(2).sum().item())
        for p in linear.parameters()
        if p.grad is not None
    ) ** 0.5
    assert _gradient_norm(linear) == pytest.approx(expected, rel=1e-6)


def test_grpo_signature_accepts_metrics_keyword_only() -> None:
    sig = inspect.signature(train_grpo_checkpoint)
    assert "metrics" in sig.parameters
    assert "run_id" in sig.parameters
    assert sig.parameters["metrics"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["run_id"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["metrics"].default is None
    assert sig.parameters["run_id"].default is None


def test_training_metrics_fed_from_helpers_do_not_raise() -> None:
    metrics = TrainingMetrics()
    group = {"samples": [{"advantage": 1.0}, {"advantage": 3.0}]}
    variance = _advantage_variance(group)
    metrics.record_step(
        "run-grpo-smoke",
        loss=0.5,
        grad_norm=1.5,
        advantage_variance=variance,
        kl=0.01,
        step_seconds=0.25,
        samples_accepted=1,
    )
    snap = metrics.snapshot()
    assert snap["steps"]["run-grpo-smoke"] == 1
    assert snap["advantage_variance_last"]["run-grpo-smoke"] == pytest.approx(variance)
