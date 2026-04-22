"""Tests for the closed-loop coordinator state machine."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from reliquary.roles.closed_loop import (
    ClosedLoopCoordinator,
    CoordinatorDecision,
    CoordinatorPolicy,
    CoordinatorState,
    CycleResult,
)


@dataclass
class FakeAdapter:
    next_window_queue: list[int | None] = field(default_factory=list)
    read_cursor_return: int | None = None
    calls_next_window: int = 0

    def next_finalized_window(self, *, cursor_name: str) -> int | None:
        self.calls_next_window += 1
        if not self.next_window_queue:
            return None
        return self.next_window_queue.pop(0)

    def read_cursor(self, cursor_name: str) -> int | None:
        return self.read_cursor_return


@dataclass
class RecordedCycle:
    window_id: int
    source_window_id: int | None


def _success_cycle(
    *, approved: bool = True, holdout: float = 0.1
) -> tuple[callable, list[RecordedCycle]]:
    recorded: list[RecordedCycle] = []

    def cycle(target_window: int, source_window: int | None) -> CycleResult:
        recorded.append(RecordedCycle(window_id=target_window, source_window_id=source_window))
        return CycleResult(
            success=True,
            window_id=target_window,
            source_window_id=source_window,
            approved=approved,
            holdout_metric=holdout,
        )

    return cycle, recorded


def _failing_cycle(error: str = "synthetic-failure") -> tuple[callable, list[RecordedCycle]]:
    recorded: list[RecordedCycle] = []

    def cycle(target_window: int, source_window: int | None) -> CycleResult:
        recorded.append(RecordedCycle(window_id=target_window, source_window_id=source_window))
        return CycleResult(
            success=False,
            window_id=target_window,
            source_window_id=source_window,
            error=error,
        )

    return cycle, recorded


def _raising_cycle(exc: Exception) -> tuple[callable, list[RecordedCycle]]:
    recorded: list[RecordedCycle] = []

    def cycle(target_window: int, source_window: int | None) -> CycleResult:
        recorded.append(RecordedCycle(window_id=target_window, source_window_id=source_window))
        raise exc

    return cycle, recorded


def _fast_policy() -> CoordinatorPolicy:
    return CoordinatorPolicy(
        poll_interval_seconds=0.001,
        cooldown_seconds=0.001,
        failure_backoff_seconds=0.001,
        max_consecutive_failures=3,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_tick_no_new_window_returns_sleep_for_waiting() -> None:
    adapter = FakeAdapter()  # empty queue → next_finalized_window returns None
    cycle, recorded = _success_cycle()
    coord = ClosedLoopCoordinator(
        cursor_name="test",
        adapter=adapter,
        cycle_callable=cycle,
        policy=_fast_policy(),
    )

    decision = coord.tick()
    assert decision.action == "sleep"
    assert decision.reason == "no_new_inference_window"
    assert coord.status.state == CoordinatorState.WAITING_FOR_DATA
    assert recorded == []  # no cycle dispatched


def test_tick_dispatches_cycle_when_new_window_available() -> None:
    adapter = FakeAdapter(next_window_queue=[42])
    cycle, recorded = _success_cycle(approved=True, holdout=0.25)
    coord = ClosedLoopCoordinator(
        cursor_name="test",
        adapter=adapter,
        cycle_callable=cycle,
        policy=_fast_policy(),
    )

    decision = coord.tick()

    assert decision.action == "sleep"
    assert decision.reason == "cycle_complete"
    assert len(recorded) == 1
    assert recorded[0].source_window_id == 42
    assert coord.status.state == CoordinatorState.COOLDOWN
    assert coord.status.cycle_count == 1
    assert coord.status.last_success_window == 0
    assert coord.status.last_approved is True
    assert coord.status.last_holdout_metric == pytest.approx(0.25)


def test_tick_advances_next_window_after_success() -> None:
    adapter = FakeAdapter(next_window_queue=[100, 101])
    cycle, _ = _success_cycle()
    coord = ClosedLoopCoordinator(
        cursor_name="test",
        adapter=adapter,
        cycle_callable=cycle,
        policy=_fast_policy(),
    )

    coord.tick()
    assert coord.status.next_window == 1
    coord.tick()
    assert coord.status.next_window == 2


def test_multiple_ticks_run_multiple_cycles() -> None:
    adapter = FakeAdapter(next_window_queue=[10, 11, 12, 13])
    cycle, recorded = _success_cycle()
    coord = ClosedLoopCoordinator(
        cursor_name="test",
        adapter=adapter,
        cycle_callable=cycle,
        policy=_fast_policy(),
    )

    for _ in range(4):
        coord.tick()

    assert len(recorded) == 4
    assert coord.status.cycle_count == 4
    assert coord.status.state == CoordinatorState.COOLDOWN


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_tick_cycle_returning_failure_transitions_to_failed() -> None:
    adapter = FakeAdapter(next_window_queue=[5])
    cycle, _ = _failing_cycle("cycle-rejected-by-policy")
    coord = ClosedLoopCoordinator(
        cursor_name="test",
        adapter=adapter,
        cycle_callable=cycle,
        policy=_fast_policy(),
    )

    decision = coord.tick()
    assert decision.reason == "cycle_failed"
    assert coord.status.state == CoordinatorState.FAILED
    assert coord.status.consecutive_failures == 1
    assert "cycle-rejected-by-policy" in (coord.status.last_error or "")


def test_tick_cycle_raising_exception_transitions_to_failed() -> None:
    adapter = FakeAdapter(next_window_queue=[5])
    cycle, _ = _raising_cycle(RuntimeError("subtensor gone"))
    coord = ClosedLoopCoordinator(
        cursor_name="test",
        adapter=adapter,
        cycle_callable=cycle,
        policy=_fast_policy(),
    )

    decision = coord.tick()
    assert decision.action == "sleep"
    assert "RuntimeError" in decision.reason or "cycle_exception" in decision.reason
    assert coord.status.state == CoordinatorState.FAILED
    assert coord.status.consecutive_failures == 1


def test_consecutive_failures_trigger_stop_decision() -> None:
    adapter = FakeAdapter(next_window_queue=[5, 6, 7, 8, 9])
    cycle, _ = _failing_cycle()
    coord = ClosedLoopCoordinator(
        cursor_name="test",
        adapter=adapter,
        cycle_callable=cycle,
        policy=CoordinatorPolicy(
            poll_interval_seconds=0.001,
            cooldown_seconds=0.001,
            failure_backoff_seconds=0.001,
            max_consecutive_failures=3,
        ),
    )

    # Burn three failures (reach max_consecutive_failures=3).
    for _ in range(3):
        coord.tick()
    # Fourth tick should short-circuit to STOP.
    decision = coord.tick()
    assert decision.action == "stop"
    assert decision.reason == "max_consecutive_failures_reached"


def test_success_after_failure_resets_failure_counter() -> None:
    adapter = FakeAdapter(next_window_queue=[5, 6, 7])
    outcomes = [False, False, True]

    def cycle(target_window: int, source_window: int | None) -> CycleResult:
        success = outcomes.pop(0)
        return CycleResult(
            success=success,
            window_id=target_window,
            source_window_id=source_window,
            error=None if success else "transient",
            approved=success,
            holdout_metric=0.1 if success else None,
        )

    coord = ClosedLoopCoordinator(
        cursor_name="test",
        adapter=adapter,
        cycle_callable=cycle,
        policy=_fast_policy(),
    )
    for _ in range(3):
        coord.tick()
    assert coord.status.consecutive_failures == 0
    assert coord.status.state == CoordinatorState.COOLDOWN


# ---------------------------------------------------------------------------
# Policy + clock behavior
# ---------------------------------------------------------------------------


def test_decision_sleep_duration_respects_policy_intervals() -> None:
    adapter = FakeAdapter(next_window_queue=[])  # always returns None
    cycle, _ = _success_cycle()
    coord = ClosedLoopCoordinator(
        cursor_name="test",
        adapter=adapter,
        cycle_callable=cycle,
        policy=CoordinatorPolicy(
            poll_interval_seconds=7.5,
            cooldown_seconds=2.0,
            failure_backoff_seconds=15.0,
        ),
    )

    decision = coord.tick()
    assert decision.sleep_seconds == pytest.approx(7.5)


def test_success_decision_uses_cooldown() -> None:
    adapter = FakeAdapter(next_window_queue=[1])
    cycle, _ = _success_cycle()
    coord = ClosedLoopCoordinator(
        cursor_name="test",
        adapter=adapter,
        cycle_callable=cycle,
        policy=CoordinatorPolicy(
            poll_interval_seconds=60.0,
            cooldown_seconds=3.0,
            failure_backoff_seconds=30.0,
        ),
    )
    decision = coord.tick()
    assert decision.sleep_seconds == pytest.approx(3.0)


def test_failure_decision_uses_backoff() -> None:
    adapter = FakeAdapter(next_window_queue=[1])
    cycle, _ = _failing_cycle()
    coord = ClosedLoopCoordinator(
        cursor_name="test",
        adapter=adapter,
        cycle_callable=cycle,
        policy=CoordinatorPolicy(
            poll_interval_seconds=60.0,
            cooldown_seconds=3.0,
            failure_backoff_seconds=30.0,
        ),
    )
    decision = coord.tick()
    assert decision.sleep_seconds == pytest.approx(30.0)


def test_clock_injection_for_deterministic_timestamps() -> None:
    adapter = FakeAdapter(next_window_queue=[1])
    cycle, _ = _success_cycle()
    coord = ClosedLoopCoordinator(
        cursor_name="test",
        adapter=adapter,
        cycle_callable=cycle,
        policy=_fast_policy(),
        _clock=lambda: 1_700_000_000.0,
    )
    coord.tick()
    assert coord.status.last_success_at == 1_700_000_000.0


# ---------------------------------------------------------------------------
# State machine invariants
# ---------------------------------------------------------------------------


def test_coordinator_initial_state_is_idle() -> None:
    adapter = FakeAdapter()
    cycle, _ = _success_cycle()
    coord = ClosedLoopCoordinator(
        cursor_name="test",
        adapter=adapter,
        cycle_callable=cycle,
    )
    assert coord.status.state == CoordinatorState.IDLE
    assert coord.status.cycle_count == 0


def test_cycle_count_monotone_across_ticks() -> None:
    adapter = FakeAdapter(next_window_queue=[1, 2, 3])
    cycle, _ = _success_cycle()
    coord = ClosedLoopCoordinator(
        cursor_name="test",
        adapter=adapter,
        cycle_callable=cycle,
        policy=_fast_policy(),
    )
    last_count = 0
    for _ in range(3):
        coord.tick()
        assert coord.status.cycle_count == last_count + 1
        last_count += 1


def test_waiting_state_does_not_consume_next_window_counter() -> None:
    adapter = FakeAdapter()  # always None → waiting
    cycle, _ = _success_cycle()
    coord = ClosedLoopCoordinator(
        cursor_name="test",
        adapter=adapter,
        cycle_callable=cycle,
        policy=_fast_policy(),
    )
    for _ in range(5):
        coord.tick()
    # next_window did NOT advance — no cycle happened.
    assert coord.status.next_window == 0
    assert coord.status.cycle_count == 0


def test_decision_is_frozen_dataclass() -> None:
    import dataclasses

    decision = CoordinatorDecision(action="sleep", sleep_seconds=1.0, reason="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.action = "stop"  # type: ignore[misc]
