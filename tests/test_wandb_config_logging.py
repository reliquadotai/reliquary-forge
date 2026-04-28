"""Tests for richer W&B training-config logging.

Phase 1.7 of the parallel-harvest plan: every training run logs
``learning_rate``, ``kl_beta``, ``ppo_clip_epsilon``, ``grad_clip_norm``,
``lr_warmup_windows``, ``lr_cosine_max_windows``, ``b_batch``,
``m_rollouts_per_prompt``, ``window_length``, ``wandb_training_version``,
and ``reliquary_version`` as run config. The tests below lock in:

- ``build_training_config`` projects a trainer cfg dict + version strings
  to the canonical W&B run config shape, filtering out unrelated keys.
- ``WandbSink.record_config`` accepts hyperparameters before *and* after
  ``start()``: pre-start, the dict merges into ``config`` and is applied
  at run init; post-start, it calls ``client.config.update`` (or falls
  back to ``client.log`` if unavailable).

Parallel-work credit: romain13190/reliquary@b65085b.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from reliquary.training.wandb_sink import (
    WandbSink,
    build_training_config,
)


@dataclass
class FakeRunConfig:
    updates: list[tuple[dict[str, Any], dict[str, Any]]] = field(default_factory=list)

    def update(self, data: dict[str, Any], **kwargs: Any) -> None:
        self.updates.append((dict(data), dict(kwargs)))


@dataclass
class FakeRun:
    config: FakeRunConfig = field(default_factory=FakeRunConfig)


@dataclass
class FakeWandb:
    init_calls: list[dict[str, Any]] = field(default_factory=list)
    log_calls: list[tuple[dict[str, Any], int | None]] = field(default_factory=list)
    finish_calls: int = 0
    return_run: FakeRun | None = None

    def init(self, **kwargs: Any) -> object:
        self.init_calls.append(kwargs)
        if self.return_run is None:
            self.return_run = FakeRun()
        return self.return_run

    def log(self, data: dict[str, Any], *, step: int | None = None, **_: Any) -> None:
        self.log_calls.append((dict(data), step))

    def finish(self, *_: Any, **__: Any) -> None:
        self.finish_calls += 1


def test_build_training_config_filters_known_keys() -> None:
    cfg = {
        "learning_rate": 5e-6,
        "kl_beta": 0.04,
        "ppo_clip_epsilon": 0.2,
        "grad_clip_norm": 1.0,
        "lr_warmup_windows": 10,
        "lr_cosine_max_windows": 10_000,
        "b_batch": 8,
        "m_rollouts_per_prompt": 8,
        "window_length": 30,
        "model_ref": "Qwen/Qwen3-4B-Instruct",
        "netuid": 462,
        "task_source": "math",
        "irrelevant_runtime_knob": "some-internal-detail",
        "another_unknown_key": 42,
    }
    out = build_training_config(cfg, reliquary_version="0.4.2")
    assert out["learning_rate"] == 5e-6
    assert out["kl_beta"] == 0.04
    assert out["ppo_clip_epsilon"] == 0.2
    assert out["grad_clip_norm"] == 1.0
    assert out["lr_warmup_windows"] == 10
    assert out["lr_cosine_max_windows"] == 10_000
    assert out["b_batch"] == 8
    assert out["m_rollouts_per_prompt"] == 8
    assert out["window_length"] == 30
    assert out["model_ref"] == "Qwen/Qwen3-4B-Instruct"
    assert out["netuid"] == 462
    assert out["task_source"] == "math"
    assert out["reliquary_version"] == "0.4.2"
    assert out["wandb_training_version"] == "v1"
    assert "irrelevant_runtime_knob" not in out
    assert "another_unknown_key" not in out


def test_build_training_config_accepts_explicit_extra_overrides() -> None:
    out = build_training_config(
        {"learning_rate": 1e-5},
        reliquary_version="0.5.0",
        wandb_training_version="v2",
        extra={"experiment_tag": "calibration-2026-W18"},
    )
    assert out["learning_rate"] == 1e-5
    assert out["reliquary_version"] == "0.5.0"
    assert out["wandb_training_version"] == "v2"
    assert out["experiment_tag"] == "calibration-2026-W18"


def test_build_training_config_with_empty_input_still_sets_versions() -> None:
    out = build_training_config()
    assert out == {"wandb_training_version": "v1"}
    out2 = build_training_config(reliquary_version="0.0.1")
    assert out2 == {"wandb_training_version": "v1", "reliquary_version": "0.0.1"}


def test_record_config_before_start_merges_into_init_config() -> None:
    fake = FakeWandb()
    sink = WandbSink(project="reliquary", run_name="r", client=fake)
    sink.record_config({"learning_rate": 5e-6, "kl_beta": 0.04})
    sink.record_config({"ppo_clip_epsilon": 0.2})
    sink.start()
    init_kwargs = fake.init_calls[0]
    config = init_kwargs["config"]
    assert config["learning_rate"] == 5e-6
    assert config["kl_beta"] == 0.04
    assert config["ppo_clip_epsilon"] == 0.2


def test_record_config_after_start_updates_run_config() -> None:
    fake = FakeWandb()
    sink = WandbSink(project="reliquary", run_name="r", client=fake)
    sink.start()
    sink.record_config({"learning_rate": 5e-6, "ppo_clip_epsilon": 0.2})
    run = fake.return_run
    assert run is not None
    assert len(run.config.updates) == 1
    payload, kwargs = run.config.updates[0]
    assert payload == {"learning_rate": 5e-6, "ppo_clip_epsilon": 0.2}
    assert kwargs.get("allow_val_change") is True


def test_record_config_after_start_falls_back_to_log_when_no_run_config() -> None:
    @dataclass
    class FakeRunNoConfig:
        pass

    fake = FakeWandb(return_run=FakeRunNoConfig())  # type: ignore[arg-type]
    sink = WandbSink(project="reliquary", run_name="r", client=fake)
    sink.start()
    sink.record_config({"learning_rate": 5e-6})
    assert any(
        data.get("config") == {"learning_rate": 5e-6}
        for data, _ in fake.log_calls
    )


def test_record_config_preserves_existing_init_config_dict() -> None:
    fake = FakeWandb()
    sink = WandbSink(
        project="reliquary",
        run_name="r",
        client=fake,
        config={"existing_key": "preserved"},
    )
    sink.record_config({"learning_rate": 5e-6})
    sink.start()
    config = fake.init_calls[0]["config"]
    assert config["existing_key"] == "preserved"
    assert config["learning_rate"] == 5e-6
