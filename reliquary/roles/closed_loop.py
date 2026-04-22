"""Closed-loop coordinator: orchestrate reliquary ↔ reliquary-inference cycles.

Pure-Python state machine deciding WHEN the Forge runs an RL cycle:
poll the inference adapter for a new finalized window, if one is
available above the last-imported cursor, dispatch an RL cycle via a
caller-supplied callable. No imports of transformers / torch / ray —
those land in the ``run_reasoning_rl_cycle`` path that gets invoked.

Ships alongside the existing ``run-rl-worker`` CLI loop in ``cli.py``
which was a straight ``while True`` with sleep. The coordinator
extracts the scheduling logic so it's unit-testable without a real
fleet: tests drive a fake ``InferenceAdapterFacade`` + fake
``RLCycleCallable`` and assert the state machine transitions cleanly
through idle → ready → running → cooldown + waiting-for-data.

Spec: 03_TIER3_PRD.md Epic 3 (closed-loop).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol

logger = logging.getLogger(__name__)


DEFAULT_POLL_INTERVAL_SECONDS = 60
DEFAULT_COOLDOWN_SECONDS = 5
DEFAULT_FAILURE_BACKOFF_SECONDS = 30


class CoordinatorState(str, Enum):
    IDLE = "idle"
    READY = "ready"
    RUNNING = "running"
    COOLDOWN = "cooldown"
    WAITING_FOR_DATA = "waiting_for_data"
    FAILED = "failed"


@dataclass
class CycleResult:
    """Outcome of a single RL cycle invocation."""

    success: bool
    window_id: int
    source_window_id: int | None
    approved: bool | None = None
    holdout_metric: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class CoordinatorPolicy:
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS
    failure_backoff_seconds: float = DEFAULT_FAILURE_BACKOFF_SECONDS
    max_consecutive_failures: int = 5


class InferenceAdapterFacade(Protocol):
    """Subset of InferenceRegistryAdapter the coordinator touches.

    Tests inject a fake without requiring filesystem / R2 state.
    """

    def next_finalized_window(self, *, cursor_name: str) -> int | None: ...

    def read_cursor(self, cursor_name: str) -> int | None: ...


RLCycleCallable = Callable[[int, int | None], CycleResult]
"""Signature: ``(target_window_id, source_window_id) → CycleResult``.

``source_window_id`` is None when the coordinator couldn't resolve one.
"""


@dataclass
class CoordinatorStatus:
    state: CoordinatorState = CoordinatorState.IDLE
    next_window: int = 0
    last_source_window: int | None = None
    last_success_window: int | None = None
    last_success_at: float | None = None
    last_failure_at: float | None = None
    consecutive_failures: int = 0
    cycle_count: int = 0
    last_error: str | None = None
    last_approved: bool | None = None
    last_holdout_metric: float | None = None


@dataclass
class ClosedLoopCoordinator:
    """State-machine wrapper around the RL training cycle + inference import.

    Usage:
        coord = ClosedLoopCoordinator(
            cursor_name="reasoning_rl",
            adapter=adapter,
            cycle_callable=_do_rl_cycle,
            policy=CoordinatorPolicy(),
        )
        while True:
            decision = coord.tick()
            if decision.action == "sleep":
                time.sleep(decision.sleep_seconds)
            # Otherwise tick() has already dispatched the cycle and
            # updated status; loop continues.
    """

    cursor_name: str
    adapter: InferenceAdapterFacade
    cycle_callable: RLCycleCallable
    policy: CoordinatorPolicy = field(default_factory=CoordinatorPolicy)
    status: CoordinatorStatus = field(default_factory=CoordinatorStatus)
    _clock: Callable[[], float] = field(default=time.time, repr=False)

    def tick(self) -> "CoordinatorDecision":
        """One step of the coordinator state machine.

        Returns a CoordinatorDecision describing what the caller should
        do next: ``sleep`` for the given duration or ``stop`` if the
        coordinator has entered a terminal failure state.
        """
        if self.status.state == CoordinatorState.FAILED and (
            self.status.consecutive_failures >= self.policy.max_consecutive_failures
        ):
            return CoordinatorDecision(
                action="stop",
                sleep_seconds=0.0,
                reason="max_consecutive_failures_reached",
            )

        next_source = self.adapter.next_finalized_window(cursor_name=self.cursor_name)
        if next_source is None:
            self._enter_waiting()
            return CoordinatorDecision(
                action="sleep",
                sleep_seconds=self.policy.poll_interval_seconds,
                reason="no_new_inference_window",
            )

        self.status.state = CoordinatorState.READY
        target_window = self.status.next_window

        try:
            self.status.state = CoordinatorState.RUNNING
            result = self.cycle_callable(target_window, next_source)
        except Exception as exc:
            self._on_failure(f"{type(exc).__name__}: {exc}")
            return CoordinatorDecision(
                action="sleep",
                sleep_seconds=self.policy.failure_backoff_seconds,
                reason=f"cycle_exception: {exc.__class__.__name__}",
            )

        if not result.success:
            self._on_failure(result.error or "cycle_returned_failure")
            return CoordinatorDecision(
                action="sleep",
                sleep_seconds=self.policy.failure_backoff_seconds,
                reason="cycle_failed",
            )

        self._on_success(result)
        return CoordinatorDecision(
            action="sleep",
            sleep_seconds=self.policy.cooldown_seconds,
            reason="cycle_complete",
        )

    def _enter_waiting(self) -> None:
        if self.status.state != CoordinatorState.WAITING_FOR_DATA:
            self.status.state = CoordinatorState.WAITING_FOR_DATA

    def _on_success(self, result: CycleResult) -> None:
        now = self._clock()
        self.status.state = CoordinatorState.COOLDOWN
        self.status.next_window = result.window_id + 1
        self.status.last_source_window = result.source_window_id
        self.status.last_success_window = result.window_id
        self.status.last_success_at = now
        self.status.consecutive_failures = 0
        self.status.cycle_count += 1
        self.status.last_approved = result.approved
        self.status.last_holdout_metric = result.holdout_metric
        self.status.last_error = None

    def _on_failure(self, error: str) -> None:
        now = self._clock()
        self.status.state = CoordinatorState.FAILED
        self.status.last_failure_at = now
        self.status.consecutive_failures += 1
        self.status.last_error = error


@dataclass(frozen=True)
class CoordinatorDecision:
    """What ``tick()`` wants the caller to do next."""

    action: str  # "sleep" | "stop"
    sleep_seconds: float
    reason: str


__all__ = [
    "DEFAULT_COOLDOWN_SECONDS",
    "DEFAULT_FAILURE_BACKOFF_SECONDS",
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "ClosedLoopCoordinator",
    "CoordinatorDecision",
    "CoordinatorPolicy",
    "CoordinatorState",
    "CoordinatorStatus",
    "CycleResult",
    "InferenceAdapterFacade",
    "RLCycleCallable",
]
