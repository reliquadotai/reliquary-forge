from reliquary.protocol.artifacts import make_artifact
from reliquary.roles.policy_authority import make_policy_authority_decision


def test_policy_authority_decision_tracks_lineage() -> None:
    checkpoint = make_artifact(
        artifact_type="checkpoint",
        producer_id="trainer",
        producer_role="learner",
        policy_id="policy-parent",
        window_id=1,
        payload={"checkpoint_path": "/tmp/checkpoint"},
        created_at="2026-04-15T00:00:00+00:00",
    )
    benchmark = make_artifact(
        artifact_type="benchmark_result",
        producer_id="validator",
        producer_role="benchmark_validator",
        policy_id="policy-parent",
        window_id=0,
        payload={"holdout_metric": 0.25},
        created_at="2026-04-15T00:00:01+00:00",
    )
    cfg = {
        "trainer_id": "trainer",
        "policy_authority_mode": "trusted_quorum",
        "policy_authority_quorum": "trainer,validator",
        "policy_authority_min_signers": 2,
    }
    decision = make_policy_authority_decision(
        cfg,
        window_id=0,
        parent_policy_id="policy-parent",
        checkpoint_artifact=checkpoint,
        benchmark_result_artifact=benchmark,
        approved=True,
        rationale="holdout_improved_or_stable",
    )
    assert decision["artifact_type"] == "policy_authority_decision"
    assert checkpoint["artifact_id"] in decision["parent_ids"]
    assert benchmark["artifact_id"] in decision["parent_ids"]
    assert decision["payload"]["approved"] is True
    assert decision["verification_state"]["accepted"] is True
