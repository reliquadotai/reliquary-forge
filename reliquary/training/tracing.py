"""OpenTelemetry tracing facade for Reliquary Forge.

Parallel to ``reliquary_inference.observability.tracing``. Callers wrap
load-bearing training paths (GRPO outer loop, DiLoCo outer step,
distillation teacher/student cycles, checkpoint publish) in
``traced_step`` / ``start_as_current_span`` so operators can stitch a
full trace from an upstream policy commit down to the backward pass.

Dep-free at import time: when ``opentelemetry`` is absent, a silent
``_NoOpTracer`` is returned and callers observe no behavior change.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import threading
from typing import Any, Callable, Iterable, Iterator, Protocol

logger = logging.getLogger(__name__)


try:
    import opentelemetry.trace as _otel_trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _otel_trace = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment]
    TracerProvider = None  # type: ignore[assignment]
    BatchSpanProcessor = None  # type: ignore[assignment]
    SpanExporter = object  # type: ignore[assignment,misc]
    _OTEL_AVAILABLE = False


_configuration_lock = threading.Lock()
_configured = False


class SpanLike(Protocol):
    def set_attribute(self, key: str, value: Any) -> None: ...


class TracerLike(Protocol):
    def start_as_current_span(
        self, name: str, *args: Any, **kwargs: Any
    ) -> contextlib.AbstractContextManager[SpanLike]: ...


class _NoOpSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def set_status(self, *_: Any, **__: Any) -> None:
        return None

    def record_exception(self, *_: Any, **__: Any) -> None:
        return None

    def end(self) -> None:
        return None

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *_: Any) -> None:
        return None


class _NoOpTracer:
    def __init__(self, name: str) -> None:
        self.name = name

    @contextlib.contextmanager
    def start_as_current_span(
        self, name: str, *args: Any, **kwargs: Any
    ) -> Iterator[_NoOpSpan]:
        yield _NoOpSpan()


def get_tracer(name: str) -> TracerLike:
    if _OTEL_AVAILABLE:
        return _otel_trace.get_tracer(name)
    return _NoOpTracer(name)


def is_otel_available() -> bool:
    return _OTEL_AVAILABLE


def is_configured() -> bool:
    with _configuration_lock:
        return _configured


def configure_tracing(
    service_name: str,
    *,
    exporters: Iterable[Any] | None = None,
) -> bool:
    """One-shot SDK initialization; idempotent.

    Returns True if a fresh configuration was installed, False otherwise
    (already configured OR OTEL absent).
    """
    global _configured
    with _configuration_lock:
        if _configured:
            return False
        if not _OTEL_AVAILABLE:
            logger.debug("configure_tracing: opentelemetry SDK missing; skipping")
            return False
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        for exporter in exporters or ():
            provider.add_span_processor(BatchSpanProcessor(exporter))
        _otel_trace.set_tracer_provider(provider)
        _configured = True
        return True


def _reset_for_tests() -> None:
    """Clear the configured state (tests only)."""
    global _configured
    with _configuration_lock:
        _configured = False
    if _OTEL_AVAILABLE:
        _otel_trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
        _otel_trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]


class InMemorySpanRecorder(SpanExporter):  # type: ignore[misc]
    """Test-only span exporter capturing finished spans as plain dicts."""

    def __init__(self) -> None:
        self._spans: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def export(self, spans: Iterable[Any]) -> Any:  # type: ignore[override]
        with self._lock:
            for span in spans:
                record: dict[str, Any] = {
                    "name": span.name,
                    "attributes": dict(span.attributes or {}),
                    "status_code": span.status.status_code.name
                    if span.status and getattr(span.status, "status_code", None)
                    else None,
                    "parent_span_id": getattr(
                        getattr(span, "parent", None), "span_id", None
                    ),
                    "trace_id": span.context.trace_id if span.context else None,
                    "span_id": span.context.span_id if span.context else None,
                }
                self._spans.append(record)
        if _OTEL_AVAILABLE:
            from opentelemetry.sdk.trace.export import SpanExportResult

            return SpanExportResult.SUCCESS
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

    @property
    def spans(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()


def traced_step(step_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: emit a ``training.step.<step_name>`` span around a callable.

    Attributes set automatically when the return value exposes them:
    ``loss``, ``grad_norm``, ``kl``, ``step_seconds``. Exceptions record
    ``reliquary.error_type`` and re-raise.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer("reliquary.training")
            with tracer.start_as_current_span(f"training.step.{step_name}") as span:
                try:
                    span.set_attribute("reliquary.training.step", step_name)
                except Exception:  # pragma: no cover
                    pass
                try:
                    result = fn(*args, **kwargs)
                except Exception as exc:
                    try:
                        span.set_attribute("reliquary.error_type", type(exc).__name__)
                    except Exception:  # pragma: no cover
                        pass
                    raise
                _annotate_step_result(span, result)
                return result

        return wrapper

    return decorator


_NUMERIC_SPAN_ATTRIBUTES = ("loss", "grad_norm", "kl", "step_seconds", "advantage_variance")


def _annotate_step_result(span: SpanLike, result: Any) -> None:
    if not isinstance(result, dict):
        return
    for key in _NUMERIC_SPAN_ATTRIBUTES:
        if key in result and isinstance(result[key], (int, float)):
            try:
                span.set_attribute(f"reliquary.training.{key}", float(result[key]))
            except Exception:  # pragma: no cover
                pass


__all__ = [
    "InMemorySpanRecorder",
    "SpanLike",
    "TracerLike",
    "configure_tracing",
    "get_tracer",
    "is_configured",
    "is_otel_available",
    "traced_step",
]
