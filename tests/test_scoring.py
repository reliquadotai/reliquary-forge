from reliquary.protocol.artifacts import make_artifact
from reliquary.scoring.scorer import compute_miner_scores, mark_uniqueness


def _verdict(rollout_id: str, producer: str, accepted: bool = True) -> dict:
    return make_artifact(
        artifact_type="verdict",
        producer_id="validator",
        policy_id="policy-1",
        window_id=0,
        payload={
            "challenge_id": "challenge-1",
            "rollout_artifact_id": rollout_id,
            "rollout_producer_id": producer,
            "rollout_archetype": "test",
            "proof_valid": accepted,
            "proof_reason": "ok" if accepted else "bad_signature",
            "score": 1.0 if accepted else 0.0,
            "dense_reward": 1.0 if accepted else 0.0,
            "satisfied_fraction": 1.0 if accepted else 0.0,
            "format_ok": accepted,
            "format_reason": "ok" if accepted else "missing_assignments",
            "assignment_count": 4 if accepted else 0,
            "expected_assignment_count": 4,
            "missing_variables": [] if accepted else [1, 2, 3, 4],
            "out_of_range_variables": [],
            "accepted": accepted,
            "uniqueness_key": "duplicate-key",
            "challenge": {"challenge_id": "challenge-1"},
            "generation": {"completion_text": "x1=True"},
            "miner_profile": {"miner_id": producer, "archetype": "test"},
            "verdict_explanation": {
                "summary": "accepted_for_scoring" if accepted else "rejected:bad_signature",
                "proof": {"ok": accepted, "reason": "ok" if accepted else "bad_signature"},
                "format": {"ok": accepted, "reason": "ok" if accepted else "missing_assignments"},
                "reward": {"dense_reward": 1.0 if accepted else 0.0},
                "uniqueness": {"ok": None, "key": None, "duplicate_of_rollout_id": None, "reason": "pending"},
                "final_decision": {"accepted": accepted, "training_eligible": None, "status": "accepted"},
            },
        },
    )


def test_duplicates_are_marked_not_training_eligible() -> None:
    verdicts = mark_uniqueness(
        [
            _verdict("rollout-1", "miner-a", accepted=True),
            _verdict("rollout-2", "miner-b", accepted=True),
        ]
    )
    assert verdicts[0]["payload"]["is_unique"] is True
    assert verdicts[0]["payload"]["training_eligible"] is True
    assert verdicts[1]["payload"]["is_unique"] is False
    assert verdicts[1]["payload"]["training_eligible"] is False
    assert verdicts[1]["payload"]["duplicate_of_rollout_id"] == "rollout-1"


def test_scorecard_counts_training_eligible_only() -> None:
    verdicts = mark_uniqueness(
        [
            _verdict("rollout-1", "miner-a", accepted=True),
            _verdict("rollout-2", "miner-b", accepted=True),
            _verdict("rollout-3", "miner-c", accepted=False),
        ]
    )
    summary = compute_miner_scores(verdicts)
    assert summary["window_metrics"]["accepted_count"] == 2
    assert summary["window_metrics"]["training_eligible_count"] == 1
    assert summary["window_metrics"]["rejected_count"] == 1
    assert summary["weights"]["miner-a"] == 1.0
    assert summary["miners"]["miner-b"]["non_unique_count"] == 1
    assert summary["miners"]["miner-c"]["proof_failures"] == 1
