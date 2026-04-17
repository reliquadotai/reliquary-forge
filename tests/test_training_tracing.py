"""Acceptance tests for the Forge training OpenTelemetry tracing facade."""

from __future__ import annotations

from typing import Any

import pytest

from reliquary.training import tracing


pytestmark = pytest.mark.skipif(
    not tracing.is_otel_available(),
    reason="opentelemetry not installed in this environment",
)


@pytest.fixture
def recorder() -> tracing.InMemorySpanRecorder:
    tracing._reset_for_tests()
    rec = tracing.InMemorySpanRecorder()
    installed = tracing.configure_tracing("reliquary-forge-test", exporters=[rec])
    assert installed, "expected fresh configuration"
    yield rec
    tracing._reset_for_tests()


def _flush() -> None:
    import opentelemetry.trace as trace

    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()


def test_get_tracer_returns_sdk_tracer() -> None:
    tracer = tracing.get_tracer("test")
    assert hasattr(tracer, "start_as_current_span")


def test_configure_tracing_is_idempotent(recorder: tracing.InMemorySpanRecorder) -> None:
    assert tracing.configure_tracing("x", exporters=[recorder]) is False


def test_start_span_recorded(recorder: tracing.InMemorySpanRecorder) -> None:
    tracer = tracing.get_tracer("reliquary-test")
    with tracer.start_as_current_span("grpo.outer_step"):
        pass
    _flush()
    assert any(s["name"] == "grpo.outer_step" for s in recorder.spans)


def test_traced_step_annotates_loss_grad_kl(
    recorder: tracing.InMemorySpanRecorder,
) -> None:
    @tracing.traced_step("grpo")
    def step(i: int) -> dict:
        return {"loss": 0.5, "grad_norm": 2.0, "kl": 0.01, "step_seconds": 1.5}

    step(0)
    _flush()
    grpo_spans = [s for s in recorder.spans if s["name"] == "training.step.grpo"]
    assert grpo_spans
    attrs = grpo_spans[-1]["attributes"]
    assert attrs["reliquary.training.step"] == "grpo"
    assert attrs["reliquary.training.loss"] == pytest.approx(0.5)
    assert attrs["reliquary.training.grad_norm"] == pytest.approx(2.0)
    assert attrs["reliquary.training.kl"] == pytest.approx(0.01)
    assert attrs["reliquary.training.step_seconds"] == pytest.approx(1.5)


def test_traced_step_records_error_type(recorder: tracing.InMemorySpanRecorder) -> None:
    @tracing.traced_step("diloco.apply")
    def step() -> dict:
        raise RuntimeError("nan detected")

    with pytest.raises(RuntimeError):
        step()
    _flush()
    spans = [s for s in recorder.spans if s["name"] == "training.step.diloco.apply"]
    assert spans
    assert spans[-1]["attributes"].get("reliquary.error_type") == "RuntimeError"


def test_traced_step_tolerates_non_dict_return(
    recorder: tracing.InMemorySpanRecorder,
) -> None:
    @tracing.traced_step("distill.student")
    def step() -> str:
        return "done"

    result = step()
    _flush()
    assert result == "done"
    spans = [s for s in recorder.spans if s["name"] == "training.step.distill.student"]
    assert spans


def test_traced_step_skips_non_numeric_values(
    recorder: tracing.InMemorySpanRecorder,
) -> None:
    @tracing.traced_step("grpo")
    def step() -> dict:
        return {"loss": "not-a-number", "grad_norm": 1.0}

    step()
    _flush()
    spans = [s for s in recorder.spans if s["name"] == "training.step.grpo"]
    assert spans
    attrs = spans[-1]["attributes"]
    assert "reliquary.training.loss" not in attrs
    assert attrs["reliquary.training.grad_norm"] == pytest.approx(1.0)


def test_nested_step_shares_trace_id(recorder: tracing.InMemorySpanRecorder) -> None:
    tracer = tracing.get_tracer("reliquary-test")

    @tracing.traced_step("inner")
    def inner() -> dict:
        return {"loss": 0.1}

    with tracer.start_as_current_span("outer.run"):
        inner()

    _flush()
    inner_spans = [s for s in recorder.spans if s["name"] == "training.step.inner"]
    outer_spans = [s for s in recorder.spans if s["name"] == "outer.run"]
    assert inner_spans and outer_spans
    assert inner_spans[-1]["trace_id"] == outer_spans[-1]["trace_id"]


def test_noop_tracer_does_not_raise_without_otel() -> None:
    noop = tracing._NoOpTracer("test")
    with noop.start_as_current_span("x") as span:
        span.set_attribute("key", "value")
    # Assert no exception — noop tracer is silent.


def test_advantage_variance_attribute_emitted(
    recorder: tracing.InMemorySpanRecorder,
) -> None:
    @tracing.traced_step("grpo")
    def step() -> dict:
        return {"loss": 0.5, "advantage_variance": 0.25}

    step()
    _flush()
    spans = [s for s in recorder.spans if s["name"] == "training.step.grpo"]
    assert spans
    assert spans[-1]["attributes"]["reliquary.training.advantage_variance"] == pytest.approx(0.25)
