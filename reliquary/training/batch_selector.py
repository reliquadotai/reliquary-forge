"""Event-driven GRPO batch selection.

Phase 1.3 of the parallel-harvest plan. Pure-python helper that picks
``B_BATCH`` distinct-prompt rollout groups from a candidate set, FIFO
by ``signed_round`` (or ``arrived_at`` fallback), with optional
cooldown gating. Drops in front of ``build_grpo_groups`` so the
trainer can opt into the event-driven-seal pattern: as soon as
``B_BATCH`` valid distinct-prompt groups are available, train; the
rest are deferred to the next window.

This is a *selector*, not a state machine. Callers supply a list of
candidate groups; the selector returns the chosen subset. The
trainer's main loop decides when to invoke the selector (on every
verdict-arrival event, on a fixed timer, etc.).

The function is pure and free of side effects so it's drop-in
testable without GPUs, transformers, or chain access. Determinism:
given the same input list (in the same order) and the same cooldown
set, the output is bit-identical across calls.

Default opt-in behavior (when wired into ``trainer.py`` via env
``RELIQUARY_GRPO_B_BATCH``):
  - unset: select all eligible groups (current behavior, no-op)
  - set to N: select up to N distinct-prompt groups, FIFO

Parallel-work credit: romain13190/reliquary@3c3553e — the pure
``select_batch`` shape (FIFO + distinct prompts + cooldown). Our
adaptation operates on Forge ``build_grpo_groups`` output rather
than on his ``ValidSubmission`` dataclass; the math is the same.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

__all__ = [
    "select_grpo_batch",
    "tiebreak_hash",
    "GroupKey",
]


def tiebreak_hash(group: dict[str, Any]) -> str:
    """Deterministic tiebreak when two groups share the same ``signed_round``.

    Hash is derived from a stable subset of group identity (challenge_id +
    first sample's prompt_token_ids prefix). Returns a hex string; sorted
    lexicographically alongside other hashes when needed.

    Pure function — same input, same hash, no clock, no entropy.
    """
    import hashlib

    h = hashlib.sha256()
    h.update(str(group.get("challenge_id", "")).encode("utf-8"))
    samples = group.get("samples", []) or []
    if samples:
        prompt_ids = samples[0].get("prompt_token_ids") or []
        # Truncate to keep the hash input small + bounded.
        for tok in prompt_ids[:64]:
            h.update(int(tok).to_bytes(4, "big", signed=False))
    return h.hexdigest()


def _signed_round(group: dict[str, Any]) -> int:
    """Pull a ``signed_round`` from the group payload; fall back to 0.

    Forge's ``build_grpo_groups`` doesn't set ``signed_round`` itself; the
    caller is expected to inject it (e.g. from the verdict's source-window
    id). When absent, all groups share signed_round=0 and tiebreak_hash
    decides ordering.
    """
    val = group.get("signed_round") or group.get("source_window_id") or 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _arrival_index(group: dict[str, Any], idx: int) -> int:
    """Stable arrival index — the position the group held in the input list.

    Used as a third-tier tiebreak when both ``signed_round`` and
    ``tiebreak_hash`` are equal (rare; only on identical groups). Keeps
    the selector deterministic.
    """
    return idx


class GroupKey:
    """Sort key for FIFO + tiebreak ordering. Lexicographic across (round,
    tiebreak_hash, arrival_index)."""

    __slots__ = ("signed_round", "hash_str", "arrival_index")

    def __init__(self, group: dict[str, Any], idx: int) -> None:
        self.signed_round = _signed_round(group)
        self.hash_str = tiebreak_hash(group)
        self.arrival_index = idx

    def __lt__(self, other: GroupKey) -> bool:
        return (self.signed_round, self.hash_str, self.arrival_index) < (
            other.signed_round,
            other.hash_str,
            other.arrival_index,
        )


def select_grpo_batch(
    groups: list[dict[str, Any]],
    *,
    b_batch: int | None,
    cooldown_prompt_ids: Iterable[int] | None = None,
    distinct_prompt: bool = True,
) -> list[dict[str, Any]]:
    """Select up to ``b_batch`` GRPO groups from ``groups`` by FIFO order.

    Args:
        groups: candidate groups, e.g. output of
            ``build_grpo_groups``. The caller is expected to have
            attached a ``signed_round`` (or ``source_window_id``)
            on each group; otherwise ordering falls back to
            tiebreak_hash + arrival_index.
        b_batch: maximum number of groups to return. ``None`` means
            "no cap" — selector returns every group that passes the
            distinct-prompt + cooldown filters. Negative values raise.
        cooldown_prompt_ids: prompt indices currently in cooldown.
            Groups whose ``prompt_idx`` is in the set are dropped.
            Pass ``None`` (default) to disable.
        distinct_prompt: when True (default), at most one group per
            distinct prompt is kept — first-seen-wins by FIFO. Set
            False to allow multiple groups per prompt (rare, only
            when caller has specific need).

    Returns:
        New list (input list never mutated). Length is
        ``min(b_batch, len(eligible))``.

    Raises:
        ValueError: if ``b_batch`` is negative.
    """
    if b_batch is not None and b_batch < 0:
        raise ValueError(f"b_batch must be non-negative or None, got {b_batch}")
    if not groups or b_batch == 0:
        return []

    cooldown = set(cooldown_prompt_ids) if cooldown_prompt_ids is not None else None

    keyed = [(GroupKey(g, i), g) for i, g in enumerate(groups)]
    keyed.sort(key=lambda pair: pair[0])

    selected: list[dict[str, Any]] = []
    seen_prompts: set = set()
    for _, group in keyed:
        prompt_idx = group.get("prompt_idx")
        if cooldown is not None and prompt_idx is not None and prompt_idx in cooldown:
            continue
        if distinct_prompt:
            challenge = group.get("challenge_id") or group.get("prompt") or id(group)
            if challenge in seen_prompts:
                continue
            seen_prompts.add(challenge)
        selected.append(group)
        if b_batch is not None and len(selected) >= b_batch:
            break

    return selected


def is_event_seal_ready(
    groups: list[dict[str, Any]],
    *,
    b_batch: int,
    cooldown_prompt_ids: Iterable[int] | None = None,
) -> bool:
    """Predicate version: True iff at least ``b_batch`` distinct-prompt groups
    are available right now.

    Useful for the trainer's main-loop event check::

        if is_event_seal_ready(groups, b_batch=8):
            batch = select_grpo_batch(groups, b_batch=8)
            train(batch)
    """
    if b_batch <= 0:
        return True
    return len(select_grpo_batch(groups, b_batch=b_batch, cooldown_prompt_ids=cooldown_prompt_ids)) >= b_batch
