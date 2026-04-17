"""Unit tests for the W&B training sink.

Uses a ``FakeWandb`` that captures ``init`` / ``log`` / ``finish`` calls
so the suite runs without the real wandb SDK. Production callers pass
the actual wandb module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from reliquary.training.wandb_sink import (
    WandbSink,
    WandbUnavailableError,
    _default_client,
)


@dataclass
class FakeWandb:
    init_calls: list[dict[str, Any]] = field(default_factory=list)
    log_calls: list[tuple[dict[str, Any], int | None]] = field(default_factory=list)
    finish_calls: int = 0

    def init(self, **kwargs: Any) -> object:  # noqa: D401
        self.init_calls.append(kwargs)
        return object()

    def log(self, data: dict[str, Any], *, step: int | None = None, **_: Any) -> None:
        self.log_calls.append((dict(data), step))

    def finish(self, *_: Any, **__: Any) -> None:
        self.finish_calls += 1


def test_start_initializes_with_project_and_name() -> None:
    fake = FakeWandb()
    sink = WandbSink(project="reliquary", run_name="sn81-window-100-199", client=fake)
    sink.start()
    assert len(fake.init_calls) == 1
    init_kwargs = fake.init_calls[0]
    assert init_kwargs["project"] == "reliquary"
    assert init_kwargs["name"] == "sn81-window-100-199"


def test_start_is_idempotent() -> None:
    fake = FakeWandb()
    sink = WandbSink(project="reliquary", run_name="r", client=fake)
    sink.start()
    sink.start()
    sink.start()
    assert len(fake.init_calls) == 1


def test_start_propagates_entity_tags_config() -> None:
    fake = FakeWandb()
    sink = WandbSink(
        project="reliquary",
        run_name="r",
        client=fake,
        entity="grail",
        tags=["sn81", "audit"],
        config={"lr": 1e-5, "batch_size": 4},
    )
    sink.start()
    kwargs = fake.init_calls[0]
    assert kwargs["entity"] == "grail"
    assert kwargs["tags"] == ["sn81", "audit"]
    assert kwargs["config"] == {"lr": 1e-5, "batch_size": 4}


def test_record_step_lazily_initializes() -> None:
    fake = FakeWandb()
    sink = WandbSink(project="p", run_name="r", client=fake)
    sink.record_step("run-1", loss=0.5)
    assert len(fake.init_calls) == 1
    assert fake.log_calls


def test_record_step_logs_required_fields() -> None:
    fake = FakeWandb()
    sink = WandbSink(project="p", run_name="r", client=fake)
    sink.record_step("run-1", loss=0.4, samples_accepted=3)
    data, step = fake.log_calls[-1]
    assert data["run_id"] == "run-1"
    assert data["loss"] == pytest.approx(0.4)
    assert data["samples_accepted"] == 3
    assert data["step_count"] == 1
    assert step == 1


def test_record_step_increments_step_counter_per_run() -> None:
    fake = FakeWandb()
    sink = WandbSink(project="p", run_name="r", client=fake)
    for _ in range(4):
        sink.record_step("run-A", loss=1.0)
    sink.record_step("run-B", loss=1.0)
    sink.record_step("run-A", loss=1.0)

    run_a_steps = [data["step_count"] for data, _ in fake.log_calls if data["run_id"] == "run-A"]
    run_b_steps = [data["step_count"] for data, _ in fake.log_calls if data["run_id"] == "run-B"]
    assert run_a_steps == [1, 2, 3, 4, 5]
    assert run_b_steps == [1]


def test_record_step_omits_none_optional_fields() -> None:
    fake = FakeWandb()
    sink = WandbSink(project="p", run_name="r", client=fake)
    sink.record_step("run-1", loss=0.5)
    data, _ = fake.log_calls[-1]
    for optional in ("grad_norm", "kl", "step_seconds", "advantage_variance"):
        assert optional not in data


def test_record_step_emits_all_optional_fields() -> None:
    fake = FakeWandb()
    sink = WandbSink(project="p", run_name="r", client=fake)
    sink.record_step(
        "run-1",
        loss=0.5,
        grad_norm=2.5,
        kl=0.01,
        step_seconds=1.25,
        advantage_variance=0.1,
    )
    data, _ = fake.log_calls[-1]
    assert data["grad_norm"] == pytest.approx(2.5)
    assert data["kl"] == pytest.approx(0.01)
    assert data["step_seconds"] == pytest.approx(1.25)
    assert data["advantage_variance"] == pytest.approx(0.1)


def test_finish_is_idempotent() -> None:
    fake = FakeWandb()
    sink = WandbSink(project="p", run_name="r", client=fake)
    sink.start()
    sink.finish()
    sink.finish()
    sink.finish()
    assert fake.finish_calls == 1


def test_finish_without_start_is_noop() -> None:
    fake = FakeWandb()
    sink = WandbSink(project="p", run_name="r", client=fake)
    sink.finish()
    assert fake.finish_calls == 0


def test_default_client_raises_when_wandb_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "wandb", None)
    # Block any later import resolution by removing from sys.modules entirely.
    original = sys.modules.pop("wandb", None)
    try:
        with pytest.raises(WandbUnavailableError):
            _default_client()
    finally:
        if original is not None:
            sys.modules["wandb"] = original


def test_record_step_validates_samples_accepted_type() -> None:
    fake = FakeWandb()
    sink = WandbSink(project="p", run_name="r", client=fake)
    sink.record_step("run-1", loss=0.5, samples_accepted=0)
    data, _ = fake.log_calls[-1]
    assert data["samples_accepted"] == 0
    assert isinstance(data["samples_accepted"], int)


def test_record_step_forwards_step_to_wandb_log_positional() -> None:
    fake = FakeWandb()
    sink = WandbSink(project="p", run_name="r", client=fake)
    sink.record_step("run-1", loss=0.1)
    sink.record_step("run-1", loss=0.2)
    assert [step for _, step in fake.log_calls] == [1, 2]
