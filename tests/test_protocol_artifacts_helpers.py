"""Unit tests for the protocol-artifact helpers.

These helpers are the entry points every lane uses to produce shared-
format artifacts. A silent change in default lane/commodity routing
or in the content-addressable artifact_id derivation would corrupt the
artifact registry, so pin the invariants here.
"""

from __future__ import annotations

import re

from reliquary.protocol.artifacts import (
    ARTIFACT_DIRECTORIES,
    artifact_directory_name,
    default_commodity_use_for_artifact,
    default_lane_for_artifact,
    finalize_policy_artifact,
    make_artifact,
    make_policy_artifact,
)


def test_artifact_directory_name_known_types() -> None:
    assert artifact_directory_name("policy") == "policies"
    assert artifact_directory_name("rollout") == "rollouts"
    assert artifact_directory_name("checkpoint") == "checkpoints"


def test_artifact_directory_name_unknown_falls_back_to_pluralized() -> None:
    assert artifact_directory_name("widget") == "widgets"


def test_artifact_directory_name_covers_all_directories() -> None:
    for artifact_type, directory in ARTIFACT_DIRECTORIES.items():
        assert artifact_directory_name(artifact_type) == directory


def test_default_lane_for_rollout_and_training_batch_is_rl() -> None:
    assert default_lane_for_artifact("rollout") == "rl"
    assert default_lane_for_artifact("training_batch") == "rl"
    assert default_lane_for_artifact("challenge") == "rl"


def test_default_lane_for_distill_artifacts_is_distill() -> None:
    assert default_lane_for_artifact("distill_trace") == "distill"
    assert default_lane_for_artifact("distill_pack") == "distill"
    assert default_lane_for_artifact("task_batch") == "distill"


def test_default_lane_unknown_is_shared() -> None:
    assert default_lane_for_artifact("policy") == "shared"
    assert default_lane_for_artifact("verdict") == "shared"
    assert default_lane_for_artifact("widget") == "shared"


def test_default_commodity_use_rl_only() -> None:
    for artifact_type in ("rollout", "training_batch", "challenge"):
        assert default_commodity_use_for_artifact(artifact_type) == "rl_only"


def test_default_commodity_use_distill_only() -> None:
    for artifact_type in ("distill_trace", "distill_pack", "task_batch"):
        assert default_commodity_use_for_artifact(artifact_type) == "distill_only"


def test_default_commodity_use_dual_use_fallback() -> None:
    assert default_commodity_use_for_artifact("policy") == "dual_use"
    assert default_commodity_use_for_artifact("verdict") == "dual_use"


def test_make_artifact_populates_defaults() -> None:
    artifact = make_artifact(
        artifact_type="rollout",
        producer_id="miner-1",
        policy_id="policy-1",
        window_id=7,
        payload={"value": 42},
    )
    assert artifact["artifact_type"] == "rollout"
    assert artifact["lane"] == "rl"
    assert artifact["commodity_use"] == "rl_only"
    assert artifact["signature_bundle"] == {"scheme": "none"}
    assert artifact["verification_state"]["status"] == "pending"
    assert artifact["producer_role"] == "unknown"
    assert artifact["parent_ids"] == []
    assert artifact["storage_refs"] == []


def test_make_artifact_id_is_sha256_hex() -> None:
    artifact = make_artifact(
        artifact_type="rollout",
        producer_id="miner-1",
        policy_id="policy-1",
        window_id=1,
        payload={"k": "v"},
    )
    assert re.fullmatch(r"[0-9a-f]{64}", artifact["artifact_id"])


def test_make_artifact_is_deterministic_with_fixed_timestamp() -> None:
    kwargs = dict(
        artifact_type="rollout",
        producer_id="miner-1",
        policy_id="policy-1",
        window_id=5,
        payload={"value": 42},
        created_at="2026-04-18T00:00:00+00:00",
    )
    first = make_artifact(**kwargs)
    second = make_artifact(**kwargs)
    assert first["artifact_id"] == second["artifact_id"]


def test_make_artifact_id_depends_on_payload_hash() -> None:
    first = make_artifact(
        artifact_type="rollout",
        producer_id="miner-1",
        policy_id="policy-1",
        window_id=1,
        payload={"k": "A"},
        created_at="2026-04-18T00:00:00+00:00",
    )
    second = make_artifact(
        artifact_type="rollout",
        producer_id="miner-1",
        policy_id="policy-1",
        window_id=1,
        payload={"k": "B"},
        created_at="2026-04-18T00:00:00+00:00",
    )
    assert first["payload_hash"] != second["payload_hash"]
    assert first["artifact_id"] != second["artifact_id"]


def test_make_artifact_lane_override() -> None:
    artifact = make_artifact(
        artifact_type="rollout",
        producer_id="miner-1",
        policy_id="policy-1",
        window_id=1,
        payload={},
        lane="shared",
    )
    assert artifact["lane"] == "shared"


def test_make_artifact_commodity_override() -> None:
    artifact = make_artifact(
        artifact_type="rollout",
        producer_id="miner-1",
        policy_id="policy-1",
        window_id=1,
        payload={},
        commodity_use="dual_use",
    )
    assert artifact["commodity_use"] == "dual_use"


def test_make_artifact_parent_ids_are_copied() -> None:
    parents = ["parent-1", "parent-2"]
    artifact = make_artifact(
        artifact_type="rollout",
        producer_id="miner-1",
        policy_id="policy-1",
        window_id=1,
        payload={},
        parent_ids=parents,
    )
    parents.append("parent-3")
    assert artifact["parent_ids"] == ["parent-1", "parent-2"]


def test_make_artifact_storage_refs_copied() -> None:
    refs = [{"scheme": "r2", "key": "k"}]
    artifact = make_artifact(
        artifact_type="rollout",
        producer_id="miner-1",
        policy_id="policy-1",
        window_id=1,
        payload={},
        storage_refs=refs,
    )
    refs.append({"scheme": "local", "key": "k2"})
    assert artifact["storage_refs"] == [{"scheme": "r2", "key": "k"}]


def test_make_artifact_verification_state_override() -> None:
    custom = {"status": "verified", "validator_ids": ["v1"], "accepted": True}
    artifact = make_artifact(
        artifact_type="rollout",
        producer_id="miner-1",
        policy_id="policy-1",
        window_id=1,
        payload={},
        verification_state=custom,
    )
    assert artifact["verification_state"] == custom


def test_make_policy_artifact_fields() -> None:
    policy = make_policy_artifact(
        producer_id="policy-auth",
        window_id=9,
        model_ref="qwen3-4b",
        env_id="sat",
        env_config={"n": 20},
        generation_config={"max_tokens": 512},
        trainer_config={"backend": "grpo"},
    )
    assert policy["artifact_type"] == "policy"
    assert policy["lane"] == "shared"
    assert policy["commodity_use"] == "dual_use"
    assert policy["payload"]["model_ref"] == "qwen3-4b"
    assert policy["payload"]["env_id"] == "sat"
    assert policy["payload"]["learner_backend"] == "grpo"
    assert policy["payload"]["lane_permissions"] == ["rl", "distill"]
    assert "checkpoint" in policy["payload"]["allowed_artifact_types"]


def test_make_policy_artifact_lane_permissions_override() -> None:
    policy = make_policy_artifact(
        producer_id="policy-auth",
        window_id=9,
        model_ref="qwen3-4b",
        env_id="sat",
        env_config={},
        generation_config={},
        trainer_config={},
        lane_permissions=["rl"],
    )
    assert policy["payload"]["lane_permissions"] == ["rl"]


def test_finalize_policy_artifact_sets_policy_id_equal_to_artifact_id() -> None:
    policy = make_policy_artifact(
        producer_id="policy-auth",
        window_id=1,
        model_ref="qwen3-4b",
        env_id="sat",
        env_config={},
        generation_config={},
        trainer_config={},
    )
    finalized = finalize_policy_artifact(policy)
    assert finalized["policy_id"] == finalized["artifact_id"]


def test_finalize_policy_artifact_policy_id_is_content_addressable() -> None:
    policy = make_policy_artifact(
        producer_id="policy-auth",
        window_id=1,
        model_ref="qwen3-4b",
        env_id="sat",
        env_config={},
        generation_config={},
        trainer_config={},
    )
    first = finalize_policy_artifact(policy)
    second = finalize_policy_artifact(policy)
    assert first["policy_id"] == second["policy_id"]


def test_finalize_policy_artifact_changes_with_payload() -> None:
    policy_a = make_policy_artifact(
        producer_id="policy-auth",
        window_id=1,
        model_ref="qwen3-4b",
        env_id="sat",
        env_config={"variant": "A"},
        generation_config={},
        trainer_config={},
    )
    policy_b = make_policy_artifact(
        producer_id="policy-auth",
        window_id=1,
        model_ref="qwen3-4b",
        env_id="sat",
        env_config={"variant": "B"},
        generation_config={},
        trainer_config={},
    )
    assert finalize_policy_artifact(policy_a)["policy_id"] != finalize_policy_artifact(policy_b)["policy_id"]


def test_make_policy_artifact_reward_contract_default() -> None:
    policy = make_policy_artifact(
        producer_id="policy-auth",
        window_id=1,
        model_ref="qwen3-4b",
        env_id="sat",
        env_config={},
        generation_config={},
        trainer_config={},
    )
    assert policy["payload"]["reward_contract"] == {"mode": "validator_defined"}


def test_make_policy_artifact_reward_contract_override() -> None:
    custom = {"mode": "fixed", "target": 1.0}
    policy = make_policy_artifact(
        producer_id="policy-auth",
        window_id=1,
        model_ref="qwen3-4b",
        env_id="sat",
        env_config={},
        generation_config={},
        trainer_config={},
        reward_contract=custom,
    )
    assert policy["payload"]["reward_contract"] == custom
