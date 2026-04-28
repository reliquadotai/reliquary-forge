"""Tests for the event-driven GRPO batch selector.

Phase 1.3 of the parallel-harvest plan. Pure-python tests — no GPU,
no transformers, no chain — covering:

- FIFO ordering by ``signed_round``, with ``tiebreak_hash`` +
  ``arrival_index`` as deterministic secondary keys
- Distinct-prompt filter (one group per challenge_id)
- Cooldown filter (prompt_idx in cooldown set is dropped)
- ``b_batch`` cap with ``None`` = no cap
- Empty input + smaller-than-B input
- ``is_event_seal_ready`` predicate
- Determinism: same input yields the same output across calls
"""

from __future__ import annotations

from typing import Any

import pytest

from reliquary.training.batch_selector import (
    GroupKey,
    is_event_seal_ready,
    select_grpo_batch,
    tiebreak_hash,
)


def _group(
    *,
    challenge_id: str,
    signed_round: int = 0,
    prompt_idx: int | None = None,
    prompt_tokens: list[int] | None = None,
    prompt_text: str = "",
    sample_count: int = 8,
) -> dict[str, Any]:
    """Construct a Forge-shaped GRPO group payload for tests."""
    samples = [
        {
            "prompt_token_ids": prompt_tokens or [1, 2, 3],
            "completion_token_ids": [10, 20, 30],
            "reward": 0.5,
        }
        for _ in range(sample_count)
    ]
    return {
        "challenge_id": challenge_id,
        "prompt": prompt_text or f"prompt-{challenge_id}",
        "samples": samples,
        "signed_round": signed_round,
        **({"prompt_idx": prompt_idx} if prompt_idx is not None else {}),
    }


# ────────  tiebreak_hash determinism  ────────


def test_tiebreak_hash_is_deterministic() -> None:
    g = _group(challenge_id="abc")
    assert tiebreak_hash(g) == tiebreak_hash(g)
    # Same shape, fresh dict
    assert tiebreak_hash(_group(challenge_id="abc")) == tiebreak_hash(_group(challenge_id="abc"))


def test_tiebreak_hash_changes_with_challenge_id() -> None:
    a = tiebreak_hash(_group(challenge_id="abc"))
    b = tiebreak_hash(_group(challenge_id="def"))
    assert a != b


def test_tiebreak_hash_changes_with_prompt_tokens() -> None:
    a = tiebreak_hash(_group(challenge_id="x", prompt_tokens=[1, 2, 3]))
    b = tiebreak_hash(_group(challenge_id="x", prompt_tokens=[1, 2, 4]))
    assert a != b


# ────────  GroupKey ordering  ────────


def test_group_key_sorts_by_signed_round_first() -> None:
    a = GroupKey(_group(challenge_id="a", signed_round=10), 0)
    b = GroupKey(_group(challenge_id="b", signed_round=5), 0)
    assert b < a


def test_group_key_uses_tiebreak_hash_when_round_equal() -> None:
    a_grp = _group(challenge_id="aaa", signed_round=5)
    b_grp = _group(challenge_id="bbb", signed_round=5)
    a = GroupKey(a_grp, 0)
    b = GroupKey(b_grp, 0)
    if tiebreak_hash(a_grp) < tiebreak_hash(b_grp):
        assert a < b
    else:
        assert b < a


# ────────  select_grpo_batch  ────────


def test_select_empty_input_returns_empty() -> None:
    assert select_grpo_batch([], b_batch=8) == []


def test_select_returns_all_when_b_batch_is_none() -> None:
    groups = [_group(challenge_id=f"c{i}") for i in range(20)]
    out = select_grpo_batch(groups, b_batch=None)
    assert len(out) == 20


def test_select_caps_at_b_batch() -> None:
    groups = [_group(challenge_id=f"c{i}") for i in range(20)]
    out = select_grpo_batch(groups, b_batch=8)
    assert len(out) == 8


def test_select_picks_lowest_signed_round_first() -> None:
    groups = [
        _group(challenge_id="late", signed_round=99),
        _group(challenge_id="early", signed_round=1),
        _group(challenge_id="middle", signed_round=50),
    ]
    out = select_grpo_batch(groups, b_batch=2)
    assert [g["challenge_id"] for g in out] == ["early", "middle"]


def test_select_drops_duplicate_prompts_keeping_first() -> None:
    # Two groups with the same challenge_id; only the FIFO-first survives.
    groups = [
        _group(challenge_id="c1", signed_round=5),
        _group(challenge_id="c1", signed_round=10),  # duplicate
        _group(challenge_id="c2", signed_round=15),
    ]
    out = select_grpo_batch(groups, b_batch=8)
    assert [g["challenge_id"] for g in out] == ["c1", "c2"]


def test_select_distinct_prompt_off_keeps_duplicates() -> None:
    groups = [
        _group(challenge_id="c1", signed_round=5),
        _group(challenge_id="c1", signed_round=10),
        _group(challenge_id="c2", signed_round=15),
    ]
    out = select_grpo_batch(groups, b_batch=8, distinct_prompt=False)
    assert len(out) == 3


def test_select_drops_groups_in_cooldown() -> None:
    groups = [
        _group(challenge_id="c0", signed_round=5, prompt_idx=42),
        _group(challenge_id="c1", signed_round=10, prompt_idx=99),
        _group(challenge_id="c2", signed_round=15, prompt_idx=42),
    ]
    out = select_grpo_batch(
        groups,
        b_batch=8,
        cooldown_prompt_ids={42},
    )
    # Only the prompt_idx=99 group survives; the other two had prompt_idx=42.
    assert [g["challenge_id"] for g in out] == ["c1"]


def test_select_returns_fewer_when_pool_smaller_than_b_batch() -> None:
    groups = [_group(challenge_id=f"c{i}") for i in range(3)]
    out = select_grpo_batch(groups, b_batch=8)
    assert len(out) == 3


def test_select_does_not_mutate_input() -> None:
    groups = [
        _group(challenge_id="c1", signed_round=10),
        _group(challenge_id="c2", signed_round=5),
    ]
    snapshot = [g["challenge_id"] for g in groups]
    select_grpo_batch(groups, b_batch=2)
    assert [g["challenge_id"] for g in groups] == snapshot


def test_select_is_deterministic_across_calls() -> None:
    groups = [_group(challenge_id=f"c{i}", signed_round=i % 3) for i in range(20)]
    a = select_grpo_batch(groups, b_batch=8)
    b = select_grpo_batch(groups, b_batch=8)
    assert [g["challenge_id"] for g in a] == [g["challenge_id"] for g in b]


def test_select_rejects_negative_b_batch() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        select_grpo_batch([_group(challenge_id="c1")], b_batch=-1)


def test_select_b_batch_zero_returns_empty() -> None:
    groups = [_group(challenge_id=f"c{i}") for i in range(5)]
    assert select_grpo_batch(groups, b_batch=0) == []


def test_select_falls_back_to_source_window_id_when_signed_round_missing() -> None:
    # Drop signed_round, set source_window_id instead — the selector should
    # still order by it.
    groups = []
    for cid, swid in [("c1", 100), ("c2", 50), ("c3", 75)]:
        g = _group(challenge_id=cid)
        del g["signed_round"]
        g["source_window_id"] = swid
        groups.append(g)
    out = select_grpo_batch(groups, b_batch=2)
    assert [g["challenge_id"] for g in out] == ["c2", "c3"]


# ────────  is_event_seal_ready  ────────


def test_seal_ready_with_exactly_b_batch_distinct_prompts() -> None:
    groups = [_group(challenge_id=f"c{i}") for i in range(8)]
    assert is_event_seal_ready(groups, b_batch=8) is True


def test_seal_not_ready_with_fewer_groups() -> None:
    groups = [_group(challenge_id=f"c{i}") for i in range(7)]
    assert is_event_seal_ready(groups, b_batch=8) is False


def test_seal_not_ready_when_duplicates_reduce_distinct_count() -> None:
    # 8 raw groups but only 4 distinct prompts → not ready.
    groups = [
        _group(challenge_id="a"),
        _group(challenge_id="a"),
        _group(challenge_id="b"),
        _group(challenge_id="b"),
        _group(challenge_id="c"),
        _group(challenge_id="c"),
        _group(challenge_id="d"),
        _group(challenge_id="d"),
    ]
    assert is_event_seal_ready(groups, b_batch=8) is False


def test_seal_ready_handles_b_batch_zero() -> None:
    assert is_event_seal_ready([], b_batch=0) is True
    assert is_event_seal_ready([_group(challenge_id="x")], b_batch=0) is True


def test_seal_ready_drops_cooldown_groups_before_counting() -> None:
    groups = [
        _group(challenge_id=f"c{i}", prompt_idx=42 if i < 5 else 99) for i in range(8)
    ]
    # 8 distinct challenges, but 5 are gated; only 3 survive → not ready.
    assert is_event_seal_ready(groups, b_batch=8, cooldown_prompt_ids={42}) is False
    assert is_event_seal_ready(groups, b_batch=3, cooldown_prompt_ids={42}) is True
