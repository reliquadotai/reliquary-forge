"""Entrypoint for multi-node DiLoCo runs launched under ``torchrun``.

Shell wrapper: ``deploy/scripts/launch_diloco.sh``.

Responsibilities:

1. Parse ``torchrun`` env vars into :class:`DistributedConfig`.
2. Initialize / tear down the default process group.
3. Build a model + inner optimizer from a caller-supplied factory.
4. Run a single DiLoCo outer-loop iteration:
   a. Snapshot current params.
   b. Run ``inner_steps`` fake or real inner updates.
   c. All-reduce the delta, apply outer-momentum step.
5. On rank 0 only: compute a delta bundle vs the pre-outer snapshot,
   publish via :func:`reliquary.training.checkpoint_storage.publish_bundle`,
   emit an attestation via :func:`reliquary.training.policy_attestation.publish_attestation`,
   record training metrics.

The top-level :func:`run_diloco_harness` is pure Python with every
external dependency injected — tests exercise it with a synthetic
``Linear(4, 2)`` model + a no-op trainer + a :class:`LocalFilesystemBackend`.
Production drives the same API with the real GRPO inner loop.

Metric naming:

- ``reliquary_launcher_inner_steps_total{run_id, rank}``
- ``reliquary_launcher_outer_applies_total{run_id}``
- ``reliquary_launcher_delta_bytes_last{run_id}``
- ``reliquary_launcher_publish_result_total{run_id, result}``

All emitted only on rank 0; lower ranks don't scrape.

Spec: ``02_TIER2_PRD.md`` Epic 3 + ``spec-closed-loop-bridge.md`` §7.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .checkpoint_storage import (
    CheckpointStorageBackend,
    LocalFilesystemBackend,
    publish_bundle,
)
from .delta_checkpoints import DeltaBundle, compute_delta
from .distributed import (
    DiLoCoConfig,
    DiLoCoOuterLoop,
    DistributedConfig,
    FSDP2Backend,
    MixedPrecisionPolicy,
    setup_distributed,
    teardown_distributed,
)
from .metrics import TrainingMetrics
from .policy_attestation import (
    AttestationPublication,
    make_hmac_authority,
    publish_attestation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Injection protocols
# ---------------------------------------------------------------------------


class ModelFactory(Protocol):
    """Builds a fresh ``torch.nn.Module`` on the calling rank."""

    def __call__(self, *, config: DistributedConfig) -> Any:  # pragma: no cover
        ...


class InnerTrainer(Protocol):
    """Runs a single inner step against ``model`` and reports a loss.

    Must be deterministic given the rank + step index if reproducibility
    across nodes matters.
    """

    def __call__(
        self, *, model: Any, rank: int, step: int
    ) -> float:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# Launch config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaunchConfig:
    run_id: str
    window_id: int
    backend_root: str
    publish: bool
    attestation_authority_hotkey: str
    attestation_authority_secret: str
    inference_netuid: int
    training_netuid: int
    effective_at_ledger_window: int
    consumed_rollout_windows: tuple[int, ...]
    consumed_rollout_keys: tuple[str, ...]
    base_snapshot_window_id: int
    dist_backend: str
    inner_steps: int
    outer_lr: float
    outer_momentum: float

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "LaunchConfig":
        """Parse env vars. Missing required keys raise with a precise name."""
        env = env if env is not None else os.environ
        run_id = _required(env, "RELIQUARY_DILOCO_RUN_ID")
        backend_root = _required(env, "RELIQUARY_DILOCO_BACKEND_ROOT")
        window_id = int(_required(env, "RELIQUARY_DILOCO_WINDOW_ID"))
        publish = env.get("RELIQUARY_DILOCO_PUBLISH", "0") == "1"
        return cls(
            run_id=run_id,
            window_id=window_id,
            backend_root=backend_root,
            publish=publish,
            attestation_authority_hotkey=env.get(
                "RELIQUARY_DILOCO_AUTHORITY_HOTKEY", "forge-authority"
            ),
            attestation_authority_secret=env.get(
                "RELIQUARY_DILOCO_AUTHORITY_SECRET", ""
            ),
            inference_netuid=int(env.get("RELIQUARY_DILOCO_INFERENCE_NETUID", "81")),
            training_netuid=int(env.get("RELIQUARY_DILOCO_TRAINING_NETUID", "3")),
            effective_at_ledger_window=int(
                env.get("RELIQUARY_DILOCO_EFFECTIVE_AT_WINDOW", "0")
            ),
            consumed_rollout_windows=tuple(
                int(w)
                for w in env.get("RELIQUARY_DILOCO_CONSUMED_WINDOWS", "").split(",")
                if w.strip()
            ),
            consumed_rollout_keys=tuple(
                k.strip()
                for k in env.get("RELIQUARY_DILOCO_CONSUMED_KEYS", "").split(",")
                if k.strip()
            ),
            base_snapshot_window_id=int(
                env.get("RELIQUARY_DILOCO_BASE_SNAPSHOT_WINDOW", "0")
            ),
            dist_backend=env.get("RELIQUARY_DIST_BACKEND", "nccl"),
            inner_steps=int(env.get("RELIQUARY_DILOCO_INNER_STEPS", "30")),
            outer_lr=float(env.get("RELIQUARY_DILOCO_OUTER_LR", "0.4")),
            outer_momentum=float(env.get("RELIQUARY_DILOCO_OUTER_MOMENTUM", "0.95")),
        )


@dataclass(frozen=True)
class LaunchResult:
    """Return value of :func:`run_diloco_harness`."""

    run_id: str
    window_id: int
    rank: int
    world_size: int
    inner_steps_executed: int
    outer_applied: bool
    delta_bundle: DeltaBundle | None
    publication: AttestationPublication | None
    metrics_counters: dict[str, int]
    last_loss: float | None


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def run_diloco_harness(
    *,
    launch: LaunchConfig,
    model_factory: ModelFactory,
    inner_trainer: InnerTrainer,
    distributed: DistributedConfig | None = None,
    backend: CheckpointStorageBackend | None = None,
    metrics: TrainingMetrics | None = None,
    publish_attestation_args: Mapping[str, Any] | None = None,
) -> LaunchResult:
    """Execute exactly one DiLoCo outer-loop iteration.

    Every dep is injected so tests can drive the harness with a
    synthetic model + trainer on CPU using the gloo backend.
    """
    distributed = distributed or DistributedConfig.from_env()
    counters: dict[str, int] = {
        "reliquary_launcher_inner_steps_total": 0,
        "reliquary_launcher_outer_applies_total": 0,
        "reliquary_launcher_delta_bytes_last": 0,
        "reliquary_launcher_publish_ok_total": 0,
        "reliquary_launcher_publish_fail_total": 0,
    }

    # Fresh process group for multi-proc; no-op for single-proc.
    initialized = setup_distributed(distributed)
    try:
        if metrics is None:
            metrics = TrainingMetrics()

        model = model_factory(config=distributed)
        fsdp_backend = FSDP2Backend(
            distributed, mp_policy=MixedPrecisionPolicy()
        )
        model = fsdp_backend.wrap(model)

        outer_cfg = DiLoCoConfig(
            inner_steps=launch.inner_steps,
            outer_lr=launch.outer_lr,
            outer_momentum=launch.outer_momentum,
        )
        outer = DiLoCoOuterLoop(outer_cfg, distributed)

        # Snapshot pre-inner-loop state so we can both DiLoCo-apply AND
        # compute a delta bundle for publishing.
        pre_params = _clone_state_dict(model)
        outer.snapshot(model)

        last_loss: float | None = None
        for step in range(launch.inner_steps):
            loss = inner_trainer(model=model, rank=distributed.rank, step=step)
            counters["reliquary_launcher_inner_steps_total"] += 1
            last_loss = float(loss)
            try:
                metrics.record_step(
                    run_id=launch.run_id,
                    loss=float(loss),
                )
            except Exception:  # pragma: no cover - metrics backend dep
                pass

        outer.apply(model)
        counters["reliquary_launcher_outer_applies_total"] += 1

        # Rank-0 only: publish delta + attestation.
        delta_bundle: DeltaBundle | None = None
        publication: AttestationPublication | None = None
        if distributed.is_root() and launch.publish:
            if backend is None:
                backend = LocalFilesystemBackend(launch.backend_root)
            post_params = _clone_state_dict(model)
            delta_bundle = compute_delta(pre_params, post_params)
            counters["reliquary_launcher_delta_bytes_last"] = sum(
                len(shard.data_bytes) for shard in delta_bundle.shards
            )

            try:
                manifest = publish_bundle(
                    delta_bundle,
                    run_id=launch.run_id,
                    window_id=launch.window_id,
                    backend=backend,
                )
                counters["reliquary_launcher_publish_ok_total"] += 1
            except Exception as exc:
                counters["reliquary_launcher_publish_fail_total"] += 1
                logger.exception(
                    "launcher.publish_bundle_failed run_id=%s window=%d err=%s",
                    launch.run_id,
                    launch.window_id,
                    exc,
                )
                manifest = None

            if manifest is not None and launch.attestation_authority_secret:
                signer = make_hmac_authority(
                    launch.attestation_authority_hotkey,
                    launch.attestation_authority_secret,
                )
                try:
                    publication = publish_attestation(
                        backend=backend,
                        signer=signer,
                        training_netuid=launch.training_netuid,
                        inference_netuid=launch.inference_netuid,
                        checkpoint_manifest=manifest,
                        base_snapshot_window_id=launch.base_snapshot_window_id,
                        consumed_rollout_windows=launch.consumed_rollout_windows,
                        consumed_rollout_keys=launch.consumed_rollout_keys,
                        effective_at_ledger_window=(
                            launch.effective_at_ledger_window
                        ),
                    )
                except Exception as exc:  # pragma: no cover - external
                    counters["reliquary_launcher_publish_fail_total"] += 1
                    logger.exception(
                        "launcher.publish_attestation_failed run_id=%s err=%s",
                        launch.run_id,
                        exc,
                    )
                    publication = None

        return LaunchResult(
            run_id=launch.run_id,
            window_id=launch.window_id,
            rank=distributed.rank,
            world_size=distributed.world_size,
            inner_steps_executed=counters["reliquary_launcher_inner_steps_total"],
            outer_applied=counters["reliquary_launcher_outer_applies_total"] > 0,
            delta_bundle=delta_bundle,
            publication=publication,
            metrics_counters=counters,
            last_loss=last_loss,
        )
    finally:
        if initialized:
            teardown_distributed()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clone_state_dict(model: Any) -> dict[str, Any]:
    """Clone every parameter tensor into a plain dict — DiLoCo-safe."""
    return {name: p.detach().clone().cpu() for name, p in _iter_named_parameters(model)}


def _iter_named_parameters(model: Any):
    # Tiny shim so the harness can accept dicts + real nn.Modules
    # interchangeably. Most models expose ``named_parameters``;
    # synthetic dict-based "models" provide this themselves.
    if hasattr(model, "named_parameters"):
        return list(model.named_parameters())
    if isinstance(model, Mapping):
        return list(model.items())
    raise TypeError(f"unsupported model type {type(model)!r}")


def _required(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required env var {name!r} is missing or empty")
    return value


# ---------------------------------------------------------------------------
# CLI entrypoint — invoked by deploy/scripts/launch_diloco.sh under torchrun
# ---------------------------------------------------------------------------


def _default_model_factory(*, config: DistributedConfig) -> Any:
    """Tiny synthetic model — safe fall-back for smoke runs.

    Production callers override this with a real transformers /
    GRPO-backed model factory. The synthetic path exists so the
    launcher is runnable end-to-end in CI without GPU.
    """
    import torch

    torch.manual_seed(0)
    return torch.nn.Linear(4, 2, bias=False)


def _default_inner_trainer(*, model: Any, rank: int, step: int) -> float:
    """Synthetic inner step — deterministic gradient nudge per rank.

    Adds rank-dependent noise so different workers end up with
    DIFFERENT param deltas, which is what makes the DiLoCo all-reduce
    non-trivial in the smoke test.
    """
    import torch

    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p) * (rank + 1) * 0.01)
    return float((step + 1) * 0.1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reliquary DiLoCo launcher")
    parser.add_argument("--verbose", action="store_true", help="Debug logging.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse env, print config, exit 0 without initializing distributed.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    try:
        launch = LaunchConfig.from_env()
    except RuntimeError as exc:
        print(f"launcher: {exc}", file=sys.stderr)
        return 2

    dist_cfg = DistributedConfig.from_env()
    logger.info(
        "launcher.start run_id=%s window=%d rank=%d/%d backend=%s",
        launch.run_id,
        launch.window_id,
        dist_cfg.rank,
        dist_cfg.world_size,
        dist_cfg.backend,
    )

    if args.dry_run:
        logger.info("launcher.dry_run launch=%s dist=%s", launch, dist_cfg)
        return 0

    result = run_diloco_harness(
        launch=launch,
        model_factory=_default_model_factory,
        inner_trainer=_default_inner_trainer,
        distributed=dist_cfg,
    )
    logger.info(
        "launcher.done run_id=%s rank=%d inner_steps=%d outer=%s "
        "publication=%s last_loss=%s",
        result.run_id,
        result.rank,
        result.inner_steps_executed,
        result.outer_applied,
        "yes" if result.publication is not None else "no",
        result.last_loss,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())


__all__ = [
    "InnerTrainer",
    "LaunchConfig",
    "LaunchResult",
    "ModelFactory",
    "main",
    "run_diloco_harness",
]
