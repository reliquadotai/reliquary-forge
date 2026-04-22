"""Forge-side eval harness — runs the held-out MATH slice against the
current policy and publishes signed :class:`~reliquary_protocol.EvalBundle`
artifacts so third parties can track whether GRPO deltas actually move
the model over time."""

from .math_harness import EvalCycleResult, run_eval_cycle

__all__ = ["EvalCycleResult", "run_eval_cycle"]
