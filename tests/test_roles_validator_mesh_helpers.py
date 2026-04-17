"""Unit tests for the pure-Python helpers in reliquary.roles.validator_mesh.

The end-to-end validate_rl_window + validate_distill_window paths pull
in a transformers model bundle and are covered by GPU-gated integration
tests. These helpers — _sat_difficulty, verdict-explanation builders,
and the score-apply passes — are pure Python and testable without a
model.
"""

from __future__ import annotations

from typing import Any

import pytest

from reliquary.protocol.artifacts import make_artifact
from reliquary.roles.validator_mesh import (
    _apply_distill_scores,
    _apply_rl_scores,
    _distill_verdict_explanation,
    _rl_verdict_explanation,
    _sat_difficulty,
)


# ---------------------------------------------------------------------------
# _sat_difficulty
# ---------------------------------------------------------------------------


def test_sat_difficulty_bounded_to_one() -> None:
    assert _sat_difficulty({"extra": {"n": 1, "m": 10**6}}) == 1.0


def test_sat_difficulty_monotone_in_m() -> None:
    low = _sat_difficulty({"extra": {"n": 10, "m": 10}})
    high = _sat_difficulty({"extra": {"n": 10, "m": 100}})
    assert high > low


def test_sat_difficulty_monotone_inverse_in_n() -> None:
    narrow = _sat_difficulty({"extra": {"n": 10, "m": 50}})
    wide = _sat_difficulty({"extra": {"n": 100, "m": 50}})
    assert narrow > wide


def test_sat_difficulty_defaults_when_missing_keys() -> None:
    diff = _sat_difficulty({"extra": {}})
    assert 0.0 < diff <= 1.0


def test_sat_difficulty_defaults_when_extra_missing() -> None:
    # challenge.get("extra", {}) returns {} so defaults apply.
    diff = _sat_difficulty({})
    assert 0.0 < diff <= 1.0


def test_sat_difficulty_floor() -> None:
    # With n=m=1, ratio=1.0 → 0.35 + 0.1 = 0.45 (well above floor).
    diff = _sat_difficulty({"extra": {"n": 1, "m": 1}})
    assert diff == pytest.approx(0.45, rel=1e-9)


# ---------------------------------------------------------------------------
# _rl_verdict_explanation
# ---------------------------------------------------------------------------


def _sat_evaluation(accepted: bool = True) -> dict:
    return {
        "score": 1.0 if accepted else 0.0,
        "dense_reward": 1.0 if accepted else 0.2,
        "satisfied_fraction": 1.0 if accepted else 0.2,
        "format_ok": accepted,
        "format_reason": "ok" if accepted else "missing_assignments",
        "assignment_count": 4 if accepted else 0,
        "expected_assignment_count": 4,
        "missing_variables": [] if accepted else [1, 2, 3, 4],
        "out_of_range_variables": [],
    }


def test_rl_verdict_explanation_accepted() -> None:
    explanation = _rl_verdict_explanation(
        {"valid": True, "reason": "ok"}, _sat_evaluation(True), accepted=True
    )
    assert explanation["summary"] == "accepted_for_scoring"
    assert explanation["final_decision"]["accepted"] is True
    assert explanation["final_decision"]["status"] == "accepted"
    assert explanation["proof"]["ok"] is True


def test_rl_verdict_explanation_proof_rejection() -> None:
    explanation = _rl_verdict_explanation(
        {"valid": False, "reason": "bad_signature"}, _sat_evaluation(True), accepted=False
    )
    assert explanation["summary"] == "rejected:bad_signature"
    assert explanation["proof"]["ok"] is False
    assert explanation["final_decision"]["accepted"] is False


def test_rl_verdict_explanation_format_rejection() -> None:
    explanation = _rl_verdict_explanation(
        {"valid": True, "reason": "ok"}, _sat_evaluation(False), accepted=False
    )
    assert explanation["summary"] == "rejected:missing_assignments"
    assert explanation["format"]["ok"] is False


def test_rl_verdict_explanation_uniqueness_pending_tag() -> None:
    explanation = _rl_verdict_explanation(
        {"valid": True, "reason": "ok"}, _sat_evaluation(True), accepted=True
    )
    assert explanation["uniqueness"]["reason"] == "pending"
    assert explanation["uniqueness"]["duplicate_of_rollout_id"] is None


def test_rl_verdict_explanation_reward_fields() -> None:
    evaluation = _sat_evaluation(True)
    explanation = _rl_verdict_explanation(
        {"valid": True, "reason": "ok"}, evaluation, accepted=True
    )
    reward = explanation["reward"]
    assert reward["score"] == pytest.approx(1.0)
    assert reward["dense_reward"] == pytest.approx(1.0)
    assert reward["satisfied_fraction"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _distill_verdict_explanation
# ---------------------------------------------------------------------------


def _reasoning_evaluation(accepted: bool = True, format_ok: bool = True) -> dict:
    return {
        "final_answer": "42",
        "correctness_or_judge": 1.0 if accepted else 0.0,
        "policy_compliance": 1.0 if accepted else 0.25,
        "format_ok": format_ok,
        "format_reason": "ok" if format_ok else "missing_final_answer",
    }


def test_distill_verdict_explanation_accepted() -> None:
    explanation = _distill_verdict_explanation(
        provenance_result={"valid": True, "reason": "ok"},
        evaluation=_reasoning_evaluation(True),
        contamination_ok=True,
        accepted=True,
    )
    assert explanation["summary"] == "accepted_for_packing"
    assert explanation["provenance"]["ok"] is True
    assert explanation["contamination"]["ok"] is True


def test_distill_verdict_explanation_bad_provenance() -> None:
    explanation = _distill_verdict_explanation(
        provenance_result={"valid": False, "reason": "bad_trace_signature"},
        evaluation=_reasoning_evaluation(False),
        contamination_ok=True,
        accepted=False,
    )
    assert explanation["summary"] == "rejected:bad_trace_signature"


def test_distill_verdict_explanation_contaminated_rejection() -> None:
    explanation = _distill_verdict_explanation(
        provenance_result={"valid": True, "reason": "ok"},
        evaluation=_reasoning_evaluation(True),
        contamination_ok=False,
        accepted=False,
    )
    assert explanation["summary"] == "rejected:contaminated_trace"
    assert explanation["contamination"]["ok"] is False


def test_distill_verdict_explanation_format_rejection() -> None:
    explanation = _distill_verdict_explanation(
        provenance_result={"valid": True, "reason": "ok"},
        evaluation=_reasoning_evaluation(False, format_ok=False),
        contamination_ok=True,
        accepted=False,
    )
    assert explanation["summary"] == "rejected:missing_final_answer"


# ---------------------------------------------------------------------------
# _apply_rl_scores
# ---------------------------------------------------------------------------


def _rl_verdict(
    *,
    rollout_id: str,
    producer: str,
    dense_reward: float = 1.0,
    difficulty: float = 0.5,
    accepted: bool = True,
    uniqueness_key: str = "key-1",
) -> dict:
    return make_artifact(
        artifact_type="verdict",
        producer_id="validator-A",
        policy_id="policy-1",
        window_id=0,
        payload={
            "challenge_id": "c-1",
            "rollout_artifact_id": rollout_id,
            "rollout_producer_id": producer,
            "rollout_archetype": "default",
            "proof_valid": accepted,
            "score": 1.0 if accepted else 0.0,
            "dense_reward": dense_reward,
            "satisfied_fraction": dense_reward,
            "difficulty": difficulty,
            "format_ok": accepted,
            "accepted": accepted,
            "uniqueness_key": uniqueness_key,
            "contamination_ok": True,
            "correctness_or_judge": 1.0 if accepted else 0.0,
            "verdict_explanation": {
                "summary": "accepted_for_scoring",
                "proof": {"ok": True, "reason": "ok"},
                "format": {"ok": True, "reason": "ok"},
                "reward": {"score": 1.0, "dense_reward": dense_reward, "satisfied_fraction": dense_reward},
                "uniqueness": {"ok": None, "key": None, "duplicate_of_rollout_id": None, "reason": "pending"},
                "final_decision": {"accepted": accepted, "training_eligible": None, "status": "accepted"},
            },
        },
    )


def test_apply_rl_scores_accepted_unique_weight_matches_formula() -> None:
    verdict = _rl_verdict(rollout_id="r-1", producer="miner-A", dense_reward=1.0, difficulty=0.5)
    scored = _apply_rl_scores([verdict])
    expected_weight = 0.70 * 1.0 + 0.15 * 1.0 + 0.15 * 0.5
    assert scored[0]["payload"]["weight_score"] == pytest.approx(expected_weight)


def test_apply_rl_scores_duplicate_gets_zero_weight() -> None:
    first = _rl_verdict(rollout_id="r-1", producer="miner-A", uniqueness_key="K")
    second = _rl_verdict(rollout_id="r-2", producer="miner-B", uniqueness_key="K")
    scored = _apply_rl_scores([first, second])
    assert scored[0]["payload"]["weight_score"] > 0
    assert scored[1]["payload"]["weight_score"] == 0.0


def test_apply_rl_scores_rejected_gets_zero_weight() -> None:
    verdict = _rl_verdict(rollout_id="r-1", producer="miner-A", accepted=False)
    scored = _apply_rl_scores([verdict])
    assert scored[0]["payload"]["weight_score"] == 0.0


def test_apply_rl_scores_sets_novelty_flag() -> None:
    first = _rl_verdict(rollout_id="r-1", producer="miner-A", uniqueness_key="K1")
    second = _rl_verdict(rollout_id="r-2", producer="miner-A", uniqueness_key="K1")
    scored = _apply_rl_scores([first, second])
    assert scored[0]["payload"]["novelty"] == 1.0
    assert scored[1]["payload"]["novelty"] == 0.0


def test_apply_rl_scores_writes_explanation_fields() -> None:
    verdict = _rl_verdict(rollout_id="r-1", producer="miner-A")
    scored = _apply_rl_scores([verdict])
    reward = scored[0]["payload"]["verdict_explanation"]["reward"]
    assert "novelty" in reward
    assert "difficulty" in reward
    assert "weight_score" in reward


# ---------------------------------------------------------------------------
# _apply_distill_scores
# ---------------------------------------------------------------------------


def _distill_verdict(
    *,
    trace_id: str,
    producer: str,
    correctness: float = 1.0,
    difficulty: float = 0.5,
    policy_compliance: float = 1.0,
    accepted: bool = True,
    uniqueness_key: str = "trace-key-1",
) -> dict:
    return make_artifact(
        artifact_type="verdict",
        producer_id="validator-A",
        policy_id="policy-1",
        window_id=0,
        payload={
            "task_id": "task-1",
            "trace_artifact_id": trace_id,
            "trace_producer_id": producer,
            "trace_archetype": "reasoned_exact",
            "proof_valid": True,
            "score": correctness,
            "dense_reward": correctness,
            "correctness_or_judge": correctness,
            "policy_compliance": policy_compliance,
            "difficulty": difficulty,
            "format_ok": accepted,
            "accepted": accepted,
            "contamination_ok": True,
            "uniqueness_key": uniqueness_key,
            "verdict_explanation": {
                "summary": "accepted_for_packing",
                "provenance": {"ok": True, "reason": "ok"},
                "format": {"ok": True, "reason": "ok"},
                "judge": {
                    "correctness_or_judge": correctness,
                    "policy_compliance": policy_compliance,
                },
                "contamination": {"ok": True, "reason": "ok"},
                "uniqueness": {"ok": None, "key": None, "duplicate_of_rollout_id": None, "reason": "pending"},
                "final_decision": {"accepted": accepted, "training_eligible": None, "status": "accepted"},
            },
        },
    )


def test_apply_distill_scores_weight_formula() -> None:
    verdict = _distill_verdict(
        trace_id="t-1",
        producer="miner-A",
        correctness=1.0,
        difficulty=0.5,
        policy_compliance=1.0,
    )
    scored = _apply_distill_scores([verdict])
    expected = 0.40 * 1.0 + 0.20 * 1.0 + 0.20 * 0.5 + 0.20 * 1.0
    assert scored[0]["payload"]["weight_score"] == pytest.approx(expected)


def test_apply_distill_scores_duplicate_gets_zero() -> None:
    first = _distill_verdict(trace_id="t-1", producer="miner-A", uniqueness_key="K")
    second = _distill_verdict(trace_id="t-2", producer="miner-B", uniqueness_key="K")
    scored = _apply_distill_scores([first, second])
    assert scored[1]["payload"]["weight_score"] == 0.0


def test_apply_distill_scores_rejected_gets_zero() -> None:
    verdict = _distill_verdict(trace_id="t-1", producer="miner-A", accepted=False)
    scored = _apply_distill_scores([verdict])
    assert scored[0]["payload"]["weight_score"] == 0.0


def test_apply_distill_scores_novelty_flag_mirrors_uniqueness() -> None:
    first = _distill_verdict(trace_id="t-1", producer="miner-A", uniqueness_key="K")
    second = _distill_verdict(trace_id="t-2", producer="miner-A", uniqueness_key="K")
    scored = _apply_distill_scores([first, second])
    assert scored[0]["payload"]["novelty"] == 1.0
    assert scored[1]["payload"]["novelty"] == 0.0


def test_apply_distill_scores_writes_judge_fields() -> None:
    verdict = _distill_verdict(trace_id="t-1", producer="miner-A")
    scored = _apply_distill_scores([verdict])
    judge = scored[0]["payload"]["verdict_explanation"]["judge"]
    assert "novelty" in judge
    assert "difficulty" in judge
    assert "weight_score" in judge
