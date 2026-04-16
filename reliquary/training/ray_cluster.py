from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_ray():
    try:
        import ray
    except ImportError as exc:  # pragma: no cover - exercised through callers
        raise RuntimeError(
            "ray_cluster execution requires the optional `ray` dependency to be installed."
        ) from exc
    return ray


def _ray_init(address: str):
    ray = _load_ray()
    if ray.is_initialized():
        return ray
    clean = str(address).strip()
    if clean in {"", "local", "auto"}:
        ray.init(ignore_reinit_error=True)
    else:
        ray.init(address=clean, ignore_reinit_error=True)
    return ray


def _gpu_requirement(device: str) -> float:
    return 1.0 if str(device).startswith("cuda") else 0.0


def _remote_train_grpo(
    *,
    model_ref: str,
    groups: list[dict[str, Any]],
    output_dir: str,
    device: str,
    lr: float,
    clip_eps: float,
    kl_coef: float,
    epochs: int,
) -> dict[str, Any]:
    from .grpo import train_grpo_checkpoint

    return train_grpo_checkpoint(
        model_ref=model_ref,
        groups=groups,
        output_dir=Path(output_dir),
        device=device,
        lr=lr,
        clip_eps=clip_eps,
        kl_coef=kl_coef,
        epochs=epochs,
    )


def _remote_evaluate_reasoning(
    *,
    model_ref: str,
    holdout_count: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    from .backends import evaluate_reasoning_checkpoint

    return evaluate_reasoning_checkpoint(
        model_ref=model_ref,
        holdout_count=holdout_count,
        cfg=cfg,
    )


def run_grpo_training_remote(
    *,
    model_ref: str,
    groups: list[dict[str, Any]],
    output_dir: Path,
    device: str,
    cfg: dict[str, Any],
    lr: float,
    clip_eps: float,
    kl_coef: float,
    epochs: int,
) -> dict[str, Any]:
    ray = _ray_init(str(cfg.get("ray_head_address", "local")))
    remote_fn = ray.remote(num_gpus=_gpu_requirement(device))(_remote_train_grpo)
    result = ray.get(
        remote_fn.remote(
            model_ref=model_ref,
            groups=groups,
            output_dir=str(output_dir),
            device=device,
            lr=lr,
            clip_eps=clip_eps,
            kl_coef=kl_coef,
            epochs=epochs,
        )
    )
    return dict(result)


def evaluate_reasoning_checkpoint_remote(
    *,
    model_ref: str,
    holdout_count: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    ray = _ray_init(str(cfg.get("ray_head_address", "local")))
    remote_fn = ray.remote(num_gpus=_gpu_requirement(str(cfg.get("device", "cpu"))))(_remote_evaluate_reasoning)
    result = ray.get(
        remote_fn.remote(
            model_ref=model_ref,
            holdout_count=holdout_count,
            cfg=cfg,
        )
    )
    return dict(result)
