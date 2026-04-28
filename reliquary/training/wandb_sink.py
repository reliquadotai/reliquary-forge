"""W&B sink for TrainingMetrics.

Closes the "W&B training run visible with loss curve, step time,
gradient norm" acceptance item of Tier 2 Epic 5. The sink mirrors the
signature of ``TrainingMetrics.record_step`` — callers route each
step's telemetry to both the Prometheus collector and the W&B run, or
either alone.

Design notes:
- ``wandb`` is an **optional** dependency. If absent, ``WandbSink``
  import raises ``WandbUnavailableError``; the training loop can catch
  and fall back to Prometheus-only.
- The ``WandbClient`` Protocol lets tests inject a fake without the real
  W&B SDK. Production code passes a ``wandb`` module as the client.
- Init arguments mirror the recommended W&B naming (``{netuid}-{window_range}``
  per the Epic 5 PRD acceptance item).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class WandbUnavailableError(RuntimeError):
    """Raised when wandb is required but the module can't be imported."""


class WandbClient(Protocol):
    """Subset of the wandb module we depend on."""

    def init(self, *args: Any, **kwargs: Any) -> Any: ...

    def log(self, data: dict[str, Any], *, step: int | None = None, **kwargs: Any) -> None: ...

    def finish(self, *args: Any, **kwargs: Any) -> None: ...


_TRAINING_CONFIG_KEYS: tuple[str, ...] = (
    # Core GRPO hyperparameters
    "learning_rate",
    "kl_beta",
    "ppo_clip_epsilon",
    "grad_clip_norm",
    "lr_warmup_windows",
    "lr_cosine_max_windows",
    # Batch shape
    "b_batch",
    "m_rollouts_per_prompt",
    "window_length",
    # Versioning
    "wandb_training_version",
    "reliquary_version",
    # Optional context
    "model_ref",
    "netuid",
    "task_source",
)


def build_training_config(
    cfg: dict[str, Any] | None = None,
    *,
    reliquary_version: str | None = None,
    wandb_training_version: str = "v1",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project a trainer cfg dict + version strings to a stable W&B config shape.

    Filters to ``_TRAINING_CONFIG_KEYS`` so unrelated runtime knobs don't
    pollute the run config. Inspired by the per-step config logging
    pattern from romain13190/reliquary commit b65085b (parallel-work
    credit).
    """
    out: dict[str, Any] = {}
    src = dict(cfg) if cfg else {}
    for key in _TRAINING_CONFIG_KEYS:
        if key in src:
            out[key] = src[key]
    if reliquary_version is not None:
        out["reliquary_version"] = reliquary_version
    out.setdefault("wandb_training_version", wandb_training_version)
    if extra:
        for key, value in extra.items():
            out[key] = value
    return out


def _default_client() -> WandbClient:
    try:
        import wandb  # type: ignore[import-not-found]
    except ImportError as exc:
        raise WandbUnavailableError(
            "wandb is not installed; install with `pip install reliquary[wandb]` "
            "or pass a client explicitly to WandbSink"
        ) from exc
    return wandb


@dataclass
class WandbSink:
    """Adapter that forwards per-step training telemetry to a W&B run."""

    project: str
    run_name: str
    client: WandbClient | None = None
    config: dict[str, Any] | None = None
    entity: str | None = None
    tags: list[str] = field(default_factory=list)
    _step_counter: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _run: Any = field(default=None, init=False, repr=False)
    _active: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = _default_client()

    def start(self) -> None:
        """Initialize the W&B run. Safe to call more than once (no-op after first)."""
        if self._active:
            return
        kwargs: dict[str, Any] = {
            "project": self.project,
            "name": self.run_name,
        }
        if self.entity is not None:
            kwargs["entity"] = self.entity
        if self.config is not None:
            kwargs["config"] = dict(self.config)
        if self.tags:
            kwargs["tags"] = list(self.tags)
        self._run = self.client.init(**kwargs)
        self._active = True

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
        """Log one step's telemetry to the active W&B run.

        Shape matches ``TrainingMetrics.record_step`` so the two sinks
        can be fed from the same trainer call site.
        """
        if not self._active:
            self.start()
        next_step = self._step_counter.get(run_id, 0) + 1
        self._step_counter[run_id] = next_step
        payload: dict[str, Any] = {
            "run_id": run_id,
            "loss": float(loss),
            "samples_accepted": int(samples_accepted),
            "step_count": next_step,
        }
        if grad_norm is not None:
            payload["grad_norm"] = float(grad_norm)
        if advantage_variance is not None:
            payload["advantage_variance"] = float(advantage_variance)
        if kl is not None:
            payload["kl"] = float(kl)
        if step_seconds is not None:
            payload["step_seconds"] = float(step_seconds)
        self.client.log(payload, step=next_step)

    def record_config(self, config: dict[str, Any]) -> None:
        """Update the active run's config with hyperparameters.

        If called before ``start()``, the config dict is merged into the
        sink's own ``config`` attribute and applied at run init. If called
        after ``start()``, the values are set on the active wandb run via
        ``client.config.update`` if the client exposes it; otherwise they
        are merged for visibility.

        Mirrors the per-step + per-config logging pattern that lets
        operators see lr, kl_beta, ppo_clip, etc. on the run page. Inspired
        by romain13190/reliquary@b65085b (parallel-work credit).
        """
        if not self._active:
            base = dict(self.config) if self.config else {}
            base.update(config)
            self.config = base
            return
        run_config = getattr(self._run, "config", None)
        update = getattr(run_config, "update", None) if run_config is not None else None
        if callable(update):
            try:
                update(dict(config), allow_val_change=True)
                return
            except TypeError:
                update(dict(config))
                return
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("wandb config.update failed: %s", exc)
        # Fallback: log config as a one-shot payload so it appears in the
        # run history even if config.update is unavailable.
        self.client.log({"config": dict(config)})

    def finish(self) -> None:
        """Mark the run finished; idempotent."""
        if not self._active:
            return
        try:
            self.client.finish()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("wandb finish failed: %s", exc)
        self._active = False


__all__ = [
    "WandbClient",
    "WandbSink",
    "WandbUnavailableError",
    "build_training_config",
]
