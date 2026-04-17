"""Expanded unit tests for scoring/scorer.py.

The existing test_scoring.py covers one happy-path mark_uniqueness and
one scorecard path. This module adds coverage for the helper utilities
(uniqueness_key_for_rollout, uniqueness_key_for_trace) plus additional
scoring edge cases (weight power exponent, format / contamination /
judge failure counters, archetype passthrough, empty verdict list,
weight_score fallback).
"""

from __future__ import annotations

import hashlib

import pytest

from reliquary.protocol.artifacts import make_artifact
from reliquary.scoring.scorer import (
    compute_miner_scores,
    mark_uniqueness,
    uniqueness_key_for_rollout,
    uniqueness_key_for_trace,
)


def _rollout(completion_ids: list[int]) -> dict:
    return make_artifact(
        artifact_type="rollout",
        producer_id="miner-1",
        policy_id="policy-1",
        window_id=0,
        payload={"generation": {"completion_token_ids": completion_ids}},
    )


def _trace(trace_text: str) -> dict:
    return make_artifact(
        artifact_type="distill_trace",
        producer_id="miner-1",
        policy_id="policy-1",
        window_id=0,
        payload={"trace": {"trace_text": trace_text}},
    )


def _verdict(
    *,
    rollout_id: str = "rollout-x",
    producer: str = "miner-1",
    accepted: bool = True,
    uniqueness_key: str = "default-key",
    challenge_id: str = "challenge-1",
    dense_reward: float = 1.0,
    weight_score: float | None = None,
    format_ok: bool = True,
    proof_valid: bool = True,
    is_unique: bool | None = None,
    contamination_ok: bool = True,
    correctness_or_judge: float = 1.0,
    archetype: str = "arch",
) -> dict:
    payload = {
        "challenge_id": challenge_id,
        "rollout_artifact_id": rollout_id,
        "rollout_producer_id": producer,
        "rollout_archetype": archetype,
        "proof_valid": proof_valid,
        "score": 1.0 if accepted else 0.0,
        "dense_reward": dense_reward,
        "format_ok": format_ok,
        "accepted": accepted,
        "uniqueness_key": uniqueness_key,
        "contamination_ok": contamination_ok,
        "correctness_or_judge": correctness_or_judge,
    }
    if weight_score is not None:
        payload["weight_score"] = weight_score
    if is_unique is not None:
        payload["is_unique"] = is_unique
    return make_artifact(
        artifact_type="verdict",
        producer_id="validator",
        policy_id="policy-1",
        window_id=0,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# uniqueness_key_for_rollout + _for_trace
# ---------------------------------------------------------------------------


def test_uniqueness_key_for_rollout_is_sha256_of_tokens_csv() -> None:
    rollout = _rollout([1, 2, 3, 4, 5])
    expected = hashlib.sha256(b"1,2,3,4,5").hexdigest()
    assert uniqueness_key_for_rollout(rollout) == expected


def test_uniqueness_key_for_rollout_different_token_orders_differ() -> None:
    a = uniqueness_key_for_rollout(_rollout([1, 2, 3]))
    b = uniqueness_key_for_rollout(_rollout([3, 2, 1]))
    assert a != b


def test_uniqueness_key_for_rollout_empty_tokens() -> None:
    # Empty list hashes the empty string.
    assert uniqueness_key_for_rollout(_rollout([])) == hashlib.sha256(b"").hexdigest()


def test_uniqueness_key_for_trace_is_sha256_of_text() -> None:
    trace = _trace("Reasoning: X\nFinal Answer: 7")
    expected = hashlib.sha256("Reasoning: X\nFinal Answer: 7".encode("utf-8")).hexdigest()
    assert uniqueness_key_for_trace(trace) == expected


def test_uniqueness_key_for_trace_unicode() -> None:
    trace = _trace("héllo")
    assert uniqueness_key_for_trace(trace) == hashlib.sha256("héllo".encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# mark_uniqueness
# ---------------------------------------------------------------------------


def test_mark_uniqueness_unique_single_entry() -> None:
    verdicts = mark_uniqueness([_verdict(rollout_id="r-1", uniqueness_key="K")])
    assert verdicts[0]["payload"]["is_unique"] is True
    assert verdicts[0]["payload"]["duplicate_of_rollout_id"] is None
    assert verdicts[0]["payload"]["training_eligible"] is True


def test_mark_uniqueness_second_duplicate_not_eligible() -> None:
    first = _verdict(rollout_id="r-1", uniqueness_key="K")
    second = _verdict(rollout_id="r-2", uniqueness_key="K")
    processed = mark_uniqueness([first, second])
    assert processed[0]["payload"]["is_unique"] is True
    assert processed[1]["payload"]["is_unique"] is False
    assert processed[1]["payload"]["duplicate_of_rollout_id"] == "r-1"
    assert processed[1]["payload"]["training_eligible"] is False


def test_mark_uniqueness_different_subjects_independent() -> None:
    first = _verdict(rollout_id="r-1", uniqueness_key="K", challenge_id="c-A")
    second = _verdict(rollout_id="r-2", uniqueness_key="K", challenge_id="c-B")
    processed = mark_uniqueness([first, second])
    assert processed[0]["payload"]["is_unique"] is True
    assert processed[1]["payload"]["is_unique"] is True


def test_mark_uniqueness_rejected_verdict_not_training_eligible_even_if_unique() -> None:
    verdict = _verdict(
        rollout_id="r-1", uniqueness_key="K", accepted=False
    )
    processed = mark_uniqueness([verdict])
    assert processed[0]["payload"]["is_unique"] is True
    assert processed[0]["payload"]["training_eligible"] is False


def test_mark_uniqueness_updates_explanation_status() -> None:
    first = _verdict(rollout_id="r-1", uniqueness_key="K")
    second = _verdict(rollout_id="r-2", uniqueness_key="K")
    processed = mark_uniqueness([first, second])
    assert processed[0]["payload"]["verdict_explanation"]["final_decision"]["status"] == "accepted"
    assert processed[1]["payload"]["verdict_explanation"]["final_decision"]["status"] == "accepted_not_eligible"


def test_mark_uniqueness_rehashes_artifact_id() -> None:
    verdict = _verdict(rollout_id="r-1", uniqueness_key="K")
    processed = mark_uniqueness([verdict])
    # After marking, artifact_id must reflect the mutated payload.
    assert processed[0]["artifact_id"] != verdict["artifact_id"]


# ---------------------------------------------------------------------------
# compute_miner_scores
# ---------------------------------------------------------------------------


def test_compute_miner_scores_empty_list_produces_minimal_structure() -> None:
    summary = compute_miner_scores([])
    assert summary["totals"] == {}
    assert summary["weights"] == {}
    assert summary["miners"] == {}
    assert summary["window_metrics"]["rollouts_seen"] == 0


def test_compute_miner_scores_weights_sum_to_one() -> None:
    verdicts = mark_uniqueness(
        [
            _verdict(rollout_id="r-1", producer="miner-A", uniqueness_key="A"),
            _verdict(rollout_id="r-2", producer="miner-B", uniqueness_key="B"),
        ]
    )
    summary = compute_miner_scores(verdicts)
    assert pytest.approx(sum(summary["weights"].values()), rel=1e-9) == 1.0


def test_compute_miner_scores_exponent_shapes_weight() -> None:
    verdicts = mark_uniqueness(
        [
            _verdict(rollout_id="r-1", producer="miner-A", uniqueness_key="A", dense_reward=1.0, weight_score=1.0),
            _verdict(rollout_id="r-2", producer="miner-B", uniqueness_key="B", dense_reward=0.5, weight_score=0.5),
        ]
    )
    linear = compute_miner_scores(verdicts, exponent=1.0)
    quad = compute_miner_scores(verdicts, exponent=2.0)
    # Exponent>1 concentrates mass on the higher-reward producer.
    assert quad["weights"]["miner-A"] > linear["weights"]["miner-A"]


def test_compute_miner_scores_non_unique_count() -> None:
    verdicts = mark_uniqueness(
        [
            _verdict(rollout_id="r-1", producer="miner-A", uniqueness_key="K"),
            _verdict(rollout_id="r-2", producer="miner-B", uniqueness_key="K"),
        ]
    )
    summary = compute_miner_scores(verdicts)
    assert summary["window_metrics"]["non_unique_count"] == 1
    assert summary["miners"]["miner-B"]["non_unique_count"] == 1


def test_compute_miner_scores_contamination_failure_counter() -> None:
    verdict = _verdict(
        rollout_id="r-1",
        producer="miner-A",
        uniqueness_key="K",
        contamination_ok=False,
    )
    verdicts = mark_uniqueness([verdict])
    summary = compute_miner_scores(verdicts)
    assert summary["window_metrics"]["contamination_failures"] == 1


def test_compute_miner_scores_format_failure_counter() -> None:
    verdict = _verdict(
        rollout_id="r-1",
        producer="miner-A",
        uniqueness_key="K",
        format_ok=False,
    )
    verdicts = mark_uniqueness([verdict])
    summary = compute_miner_scores(verdicts)
    assert summary["window_metrics"]["format_failures"] == 1


def test_compute_miner_scores_judge_failure_counter() -> None:
    verdict = _verdict(
        rollout_id="r-1",
        producer="miner-A",
        uniqueness_key="K",
        correctness_or_judge=0.0,
    )
    verdicts = mark_uniqueness([verdict])
    summary = compute_miner_scores(verdicts)
    assert summary["window_metrics"]["judge_failures"] == 1


def test_compute_miner_scores_mean_reward_averages_across_rollouts() -> None:
    verdicts = mark_uniqueness(
        [
            _verdict(rollout_id="r-1", producer="miner-A", uniqueness_key="A", dense_reward=1.0),
            _verdict(rollout_id="r-2", producer="miner-A", uniqueness_key="B", dense_reward=0.5),
        ]
    )
    summary = compute_miner_scores(verdicts)
    assert summary["miners"]["miner-A"]["mean_dense_reward"] == pytest.approx(0.75)


def test_compute_miner_scores_weight_score_fallback_to_dense_reward() -> None:
    verdicts = mark_uniqueness(
        [
            _verdict(rollout_id="r-1", producer="miner-A", uniqueness_key="A", dense_reward=0.8),
        ]
    )
    summary = compute_miner_scores(verdicts)
    # weight_score wasn't set → falls back to dense_reward 0.8.
    assert summary["miners"]["miner-A"]["total_weight_score"] == pytest.approx(0.8)


def test_compute_miner_scores_archetype_passthrough() -> None:
    verdicts = mark_uniqueness(
        [
            _verdict(rollout_id="r-1", producer="miner-A", uniqueness_key="A", archetype="careful_reasoner"),
        ]
    )
    summary = compute_miner_scores(verdicts)
    assert summary["miners"]["miner-A"]["archetype"] == "careful_reasoner"


def test_compute_miner_scores_rollouts_seen_matches_input() -> None:
    verdicts = mark_uniqueness(
        [
            _verdict(rollout_id=f"r-{i}", producer="miner-A", uniqueness_key=f"U-{i}")
            for i in range(5)
        ]
    )
    summary = compute_miner_scores(verdicts)
    assert summary["window_metrics"]["rollouts_seen"] == 5
    assert summary["miners"]["miner-A"]["rollouts_seen"] == 5
