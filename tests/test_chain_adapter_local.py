"""Unit tests for the LocalChainAdapter.

LocalChainAdapter is an in-process record-replay chain simulator used by
integration tests and local dev. It exposes the same publish / submit
surface as BittensorChainAdapter so downstream orchestration code can
target either without branching. This suite covers every method and
verifies the event record ordering + payload shape.
"""

from __future__ import annotations

from typing import Any

from reliquary.chain.adapters import LocalChainAdapter
from reliquary.protocol.artifacts import make_artifact, make_policy_artifact


def _policy() -> dict:
    policy = make_policy_artifact(
        producer_id="policy-auth",
        window_id=7,
        model_ref="qwen3-4b",
        env_id="sat",
        env_config={},
        generation_config={},
        trainer_config={},
    )
    return policy


def _rollout() -> dict:
    return make_artifact(
        artifact_type="rollout",
        producer_id="miner-1",
        policy_id="policy-1",
        window_id=7,
        payload={"challenge_id": "c-7"},
    )


def _scorecard() -> dict:
    return make_artifact(
        artifact_type="scorecard",
        producer_id="validator-1",
        policy_id="policy-1",
        window_id=7,
        payload={
            "weights": {"miner-1": 0.6, "miner-2": 0.4},
            "totals": {"miner-1": 6.0, "miner-2": 4.0},
        },
    )


def _checkpoint() -> dict:
    return make_artifact(
        artifact_type="checkpoint",
        producer_id="trainer-1",
        policy_id="policy-1",
        window_id=7,
        payload={"checkpoint_path": "/tmp/ckpt-7"},
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_new_adapter_has_empty_events_and_weights() -> None:
    adapter = LocalChainAdapter()
    assert adapter.events() == []
    assert adapter._weights == {}


# ---------------------------------------------------------------------------
# publish_policy_reference
# ---------------------------------------------------------------------------


def test_publish_policy_reference_records_event() -> None:
    adapter = LocalChainAdapter()
    policy = _policy()
    event = adapter.publish_policy_reference(policy)

    assert event["event_type"] == "policy_reference"
    assert event["mode"] == "local"
    assert event["event_index"] == 0
    assert event["payload"]["policy_id"] == policy["artifact_id"]
    assert event["payload"]["window_id"] == policy["window_id"]
    assert event["payload"]["model_ref"] == "qwen3-4b"


def test_publish_policy_reference_appends_to_events() -> None:
    adapter = LocalChainAdapter()
    adapter.publish_policy_reference(_policy())
    adapter.publish_policy_reference(_policy())
    assert len(adapter.events()) == 2
    assert [e["event_index"] for e in adapter.events()] == [0, 1]


# ---------------------------------------------------------------------------
# publish_window_metadata
# ---------------------------------------------------------------------------


def test_publish_window_metadata_captures_ids() -> None:
    adapter = LocalChainAdapter()
    event = adapter.publish_window_metadata(
        run_id="run-1",
        window_id=7,
        policy_id="policy-1",
        challenge_ids=["c-1", "c-2"],
        miner_ids=["miner-1", "miner-2"],
    )
    assert event["event_type"] == "window_metadata"
    assert event["payload"]["run_id"] == "run-1"
    assert event["payload"]["challenge_ids"] == ["c-1", "c-2"]
    assert event["payload"]["miner_ids"] == ["miner-1", "miner-2"]


# ---------------------------------------------------------------------------
# submit_miner_result_reference
# ---------------------------------------------------------------------------


def test_submit_miner_result_reference_captures_rollout_fields() -> None:
    adapter = LocalChainAdapter()
    rollout = _rollout()
    event = adapter.submit_miner_result_reference(rollout)
    assert event["event_type"] == "miner_result_reference"
    assert event["payload"]["rollout_id"] == rollout["artifact_id"]
    assert event["payload"]["miner_id"] == "miner-1"
    assert event["payload"]["challenge_id"] == "c-7"
    assert event["payload"]["artifact_type"] == "rollout"


def test_submit_miner_result_reference_falls_back_to_task_id() -> None:
    adapter = LocalChainAdapter()
    rollout = make_artifact(
        artifact_type="rollout",
        producer_id="miner-1",
        policy_id="policy-1",
        window_id=7,
        payload={"task_id": "task-99"},
    )
    event = adapter.submit_miner_result_reference(rollout)
    assert event["payload"]["challenge_id"] == "task-99"


# ---------------------------------------------------------------------------
# publish_scorecard
# ---------------------------------------------------------------------------


def test_publish_scorecard_records_weights_and_updates_state() -> None:
    adapter = LocalChainAdapter()
    scorecard = _scorecard()
    event = adapter.publish_scorecard(scorecard)

    assert event["event_type"] == "scorecard"
    assert event["payload"]["scorecard_id"] == scorecard["artifact_id"]
    assert event["payload"]["weights"] == {"miner-1": 0.6, "miner-2": 0.4}
    assert event["payload"]["totals"] == {"miner-1": 6.0, "miner-2": 4.0}
    # Adapter caches the weights internally.
    assert adapter._weights == {"miner-1": 0.6, "miner-2": 0.4}


def test_publish_scorecard_isolates_internal_weights_from_input() -> None:
    adapter = LocalChainAdapter()
    scorecard = _scorecard()
    adapter.publish_scorecard(scorecard)
    scorecard["payload"]["weights"]["miner-1"] = 0.01
    assert adapter._weights["miner-1"] == 0.6


# ---------------------------------------------------------------------------
# publish_checkpoint_reference
# ---------------------------------------------------------------------------


def test_publish_checkpoint_reference_records_event() -> None:
    adapter = LocalChainAdapter()
    checkpoint = _checkpoint()
    next_policy = _policy()
    event = adapter.publish_checkpoint_reference(checkpoint, next_policy)

    assert event["event_type"] == "checkpoint_reference"
    assert event["payload"]["checkpoint_id"] == checkpoint["artifact_id"]
    assert event["payload"]["checkpoint_path"] == "/tmp/ckpt-7"
    assert event["payload"]["next_policy_id"] == next_policy["artifact_id"]
    assert event["payload"]["window_id"] == 7


# ---------------------------------------------------------------------------
# publish_weights
# ---------------------------------------------------------------------------


def test_publish_weights_records_and_caches() -> None:
    adapter = LocalChainAdapter()
    weights = {"miner-1": 0.3, "miner-2": 0.7}
    event = adapter.publish_weights(weights)
    assert event["event_type"] == "weights"
    assert event["payload"]["weights"] == weights
    assert adapter._weights == weights


def test_publish_weights_isolates_internal_state() -> None:
    adapter = LocalChainAdapter()
    weights = {"miner-1": 0.3}
    adapter.publish_weights(weights)
    weights["miner-1"] = 0.99
    assert adapter._weights["miner-1"] == 0.3


def test_publish_weights_event_payload_snapshot() -> None:
    adapter = LocalChainAdapter()
    weights = {"miner-1": 0.3}
    event = adapter.publish_weights(weights)
    adapter.publish_weights({"miner-1": 0.7})
    # Event-1's payload snapshot is the first weights dict; not overwritten by event-2.
    assert event["payload"]["weights"]["miner-1"] == 0.3


# ---------------------------------------------------------------------------
# events()
# ---------------------------------------------------------------------------


def test_events_returns_defensive_copy() -> None:
    adapter = LocalChainAdapter()
    adapter.publish_weights({"miner-1": 1.0})
    snapshot = adapter.events()
    snapshot.clear()
    assert len(adapter.events()) == 1


def test_event_indices_monotone_across_different_methods() -> None:
    adapter = LocalChainAdapter()
    adapter.publish_policy_reference(_policy())
    adapter.publish_weights({"miner-1": 1.0})
    adapter.publish_scorecard(_scorecard())
    adapter.publish_checkpoint_reference(_checkpoint(), _policy())
    indices = [e["event_index"] for e in adapter.events()]
    assert indices == list(range(len(indices)))
    assert [e["event_type"] for e in adapter.events()] == [
        "policy_reference",
        "weights",
        "scorecard",
        "checkpoint_reference",
    ]
