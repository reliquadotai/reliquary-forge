from reliquary.protocol.artifacts import make_artifact


def test_dual_lane_artifact_ids_are_stable_with_fixed_timestamp() -> None:
    payload = {"value": 1}
    artifact_a = make_artifact(
        artifact_type="distill_trace",
        producer_id="miner-a",
        producer_role="generation_miner",
        policy_id="policy-1",
        window_id=0,
        lane="distill",
        commodity_use="dual_use",
        parent_ids=["policy-1", "task-batch-1"],
        created_at="2026-04-15T00:00:00+00:00",
        payload=payload,
    )
    artifact_b = make_artifact(
        artifact_type="distill_trace",
        producer_id="miner-a",
        producer_role="generation_miner",
        policy_id="policy-1",
        window_id=0,
        lane="distill",
        commodity_use="dual_use",
        parent_ids=["policy-1", "task-batch-1"],
        created_at="2026-04-15T00:00:00+00:00",
        payload=payload,
    )
    assert artifact_a["artifact_id"] == artifact_b["artifact_id"]
    assert artifact_a["payload_hash"] == artifact_b["payload_hash"]
    assert artifact_a["lane"] == "distill"
    assert artifact_a["commodity_use"] == "dual_use"
    assert artifact_a["verification_state"]["accepted"] is None
