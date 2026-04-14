from __future__ import annotations

from typing import Any

from .validator_mesh import validate_distill_window, validate_rl_window


def validate_window(
    cfg: dict[str, Any],
    policy_artifact: dict[str, Any],
    rollout_artifacts: list[dict[str, Any]],
    window_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return validate_rl_window(cfg, policy_artifact, rollout_artifacts, window_id=window_id)


__all__ = ["validate_window", "validate_distill_window", "validate_rl_window"]
