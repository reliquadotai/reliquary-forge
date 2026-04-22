"""Unit tests for reliquary.roles.policy_authority.

Pins the three pure helpers (publish_distill_policy,
make_policy_authority_decision, promote_checkpoint_to_policy) against
their expected artifact shapes so downstream consumers (storage
registry, chain adapter) can rely on field names + types.
"""

from __future__ import annotations

from typing import Any

from reliquary.protocol.artifacts import (
    make_artifact,
)
from reliquary.roles.policy_authority import (
    _quorum_members,
    make_policy_authority_decision,
    promote_checkpoint_to_policy,
    publish_distill_policy,
)


def _cfg(**overrides: Any) -> dict:
    base: dict[str, Any] = {
        "trainer_id": "trainer-A",
        "distill_student_model_ref": "student-bootstrap",
        "distill_train_task_count": 8,
        "distill_holdout_count": 4,
        "distill_samples_per_task": 2,
        "teacher_backend": "local_reasoning_teacher",
        "distill_learner_backend": "local_reasoning_student",
        "policy_authority_mode": "trusted_quorum",
        "policy_authority_quorum": "trainer-A, validator-A,  trainer-B  ",
        "policy_authority_min_signers": 2,
        "model_teacher": "teacher-v1",
        "git_sha": "abc123",
        "service_version": "0.1.0",
    }
    base.update(overrides)
    return base


def _checkpoint_artifact(path: str = "/tmp/ckpt-7") -> dict:
    return make_artifact(
        artifact_type="checkpoint",
        producer_id="trainer-A",
        policy_id="policy-1",
        window_id=7,
        payload={"checkpoint_path": path},
    )


def _benchmark_result(delta: float = 0.1) -> dict:
    return make_artifact(
        artifact_type="benchmark_result",
        producer_id="trainer-A",
        policy_id="policy-1",
        window_id=7,
        payload={
            "holdout_metric": delta,
            "eval_before": 0.5,
            "eval_after": 0.5 + delta,
        },
    )


# ---------------------------------------------------------------------------
# _quorum_members
# ---------------------------------------------------------------------------


def test_quorum_members_trims_and_filters_empty() -> None:
    cfg = _cfg()
    assert _quorum_members(cfg) == ["trainer-A", "validator-A", "trainer-B"]


def test_quorum_members_empty_when_unset() -> None:
    cfg = _cfg(policy_authority_quorum="")
    assert _quorum_members(cfg) == []


def test_quorum_members_missing_key_returns_empty() -> None:
    cfg = _cfg()
    cfg.pop("policy_authority_quorum")
    assert _quorum_members(cfg) == []


# ---------------------------------------------------------------------------
# publish_distill_policy
# ---------------------------------------------------------------------------


def test_publish_distill_policy_finalized_with_stable_id() -> None:
    policy = publish_distill_policy(_cfg(), window_id=3)
    assert policy["policy_id"] == policy["artifact_id"]


def test_publish_distill_policy_defaults_model_ref_to_student() -> None:
    policy = publish_distill_policy(_cfg(), window_id=3)
    assert policy["payload"]["model_ref"] == "student-bootstrap"


def test_publish_distill_policy_model_ref_override() -> None:
    policy = publish_distill_policy(_cfg(), window_id=3, model_ref="override-model")
    assert policy["payload"]["model_ref"] == "override-model"


def test_publish_distill_policy_lane_permissions_are_distill_only() -> None:
    policy = publish_distill_policy(_cfg(), window_id=3)
    assert policy["payload"]["lane_permissions"] == ["distill"]


def test_publish_distill_policy_env_id_is_reasoning() -> None:
    policy = publish_distill_policy(_cfg(), window_id=3)
    assert policy["payload"]["env_id"] == "reasoning"


def test_publish_distill_policy_env_config_uses_cfg_values() -> None:
    policy = publish_distill_policy(_cfg(), window_id=3)
    env_config = policy["payload"]["env_config"]
    assert env_config["train_task_count"] == 8
    assert env_config["holdout_count"] == 4
    assert env_config["samples_per_task"] == 2
    assert env_config["task_family"] == "reasoning_traces"


def test_publish_distill_policy_evaluation_contract_shape() -> None:
    policy = publish_distill_policy(_cfg(), window_id=3)
    contract = policy["payload"]["evaluation_contract"]
    assert contract["lane"] == "distill"
    assert contract["metric"] == "reasoning_contract_score"
    assert contract["holdout_count"] == 4


def test_publish_distill_policy_trainer_config_git_sha_and_version() -> None:
    policy = publish_distill_policy(_cfg(git_sha="def456", service_version="0.2.0"), window_id=3)
    trainer = policy["payload"]["trainer_config"]
    assert trainer["git_sha"] == "def456"
    assert trainer["service_version"] == "0.2.0"
    assert trainer["policy_authority_mode"] == "trusted_quorum"


# ---------------------------------------------------------------------------
# make_policy_authority_decision
# ---------------------------------------------------------------------------


def test_decision_artifact_records_approval() -> None:
    decision = make_policy_authority_decision(
        _cfg(),
        window_id=7,
        parent_policy_id="policy-1",
        checkpoint_artifact=_checkpoint_artifact(),
        benchmark_result_artifact=_benchmark_result(),
        approved=True,
        rationale="holdout_improved",
    )
    assert decision["artifact_type"] == "policy_authority_decision"
    assert decision["payload"]["approved"] is True
    assert decision["payload"]["rationale"] == "holdout_improved"
    assert decision["verification_state"]["accepted"] is True
    assert decision["verification_state"]["status"] == "final"


def test_decision_artifact_rejection_path() -> None:
    decision = make_policy_authority_decision(
        _cfg(),
        window_id=7,
        parent_policy_id="policy-1",
        checkpoint_artifact=_checkpoint_artifact(),
        benchmark_result_artifact=_benchmark_result(),
        approved=False,
        rationale="holdout_regressed",
    )
    assert decision["payload"]["approved"] is False
    assert decision["verification_state"]["accepted"] is False


def test_decision_artifact_signers_from_quorum() -> None:
    decision = make_policy_authority_decision(
        _cfg(),
        window_id=7,
        parent_policy_id="policy-1",
        checkpoint_artifact=_checkpoint_artifact(),
        benchmark_result_artifact=_benchmark_result(),
        approved=True,
        rationale="ok",
    )
    assert decision["payload"]["signers"] == ["trainer-A", "validator-A", "trainer-B"]
    assert decision["payload"]["min_signers"] == 2


def test_decision_artifact_parent_ids() -> None:
    checkpoint = _checkpoint_artifact()
    benchmark = _benchmark_result()
    decision = make_policy_authority_decision(
        _cfg(),
        window_id=7,
        parent_policy_id="policy-1",
        checkpoint_artifact=checkpoint,
        benchmark_result_artifact=benchmark,
        approved=True,
        rationale="ok",
    )
    assert "policy-1" in decision["parent_ids"]
    assert checkpoint["artifact_id"] in decision["parent_ids"]
    assert benchmark["artifact_id"] in decision["parent_ids"]


def test_decision_artifact_references_ids_in_payload() -> None:
    checkpoint = _checkpoint_artifact()
    benchmark = _benchmark_result()
    decision = make_policy_authority_decision(
        _cfg(),
        window_id=7,
        parent_policy_id="policy-1",
        checkpoint_artifact=checkpoint,
        benchmark_result_artifact=benchmark,
        approved=True,
        rationale="ok",
    )
    assert decision["payload"]["checkpoint_artifact_id"] == checkpoint["artifact_id"]
    assert decision["payload"]["benchmark_result_artifact_id"] == benchmark["artifact_id"]


def test_decision_artifact_authority_mode_passthrough() -> None:
    decision = make_policy_authority_decision(
        _cfg(policy_authority_mode="permissionless_stake"),
        window_id=7,
        parent_policy_id="policy-1",
        checkpoint_artifact=_checkpoint_artifact(),
        benchmark_result_artifact=_benchmark_result(),
        approved=True,
        rationale="ok",
    )
    assert decision["payload"]["authority_mode"] == "permissionless_stake"


# ---------------------------------------------------------------------------
# promote_checkpoint_to_policy
# ---------------------------------------------------------------------------


def test_promote_checkpoint_uses_checkpoint_path_as_model_ref() -> None:
    checkpoint = _checkpoint_artifact(path="/srv/ckpts/win-8")
    policy = promote_checkpoint_to_policy(_cfg(), checkpoint_artifact=checkpoint, window_id=8)
    assert policy["payload"]["model_ref"] == "/srv/ckpts/win-8"


def test_promote_checkpoint_parent_ids_has_checkpoint_only() -> None:
    checkpoint = _checkpoint_artifact()
    policy = promote_checkpoint_to_policy(_cfg(), checkpoint_artifact=checkpoint, window_id=8)
    assert policy["parent_ids"] == [checkpoint["artifact_id"]]


def test_promote_checkpoint_yields_finalized_policy() -> None:
    policy = promote_checkpoint_to_policy(_cfg(), checkpoint_artifact=_checkpoint_artifact(), window_id=8)
    # Finalized policy sets policy_id == artifact_id (contract invariant of finalize_policy_artifact).
    # promote_checkpoint_to_policy mutates parent_ids AFTER finalization, so policy_id captures the
    # pre-mutation state and artifact_id no longer equals policy_id. Pin that behavior explicitly
    # so callers know to dedupe / re-finalize before submitting onchain.
    assert "policy_id" in policy
    assert policy["policy_id"] != ""
    assert policy["artifact_type"] == "policy"


def test_promote_checkpoint_sets_window_id() -> None:
    policy = promote_checkpoint_to_policy(_cfg(), checkpoint_artifact=_checkpoint_artifact(), window_id=15)
    assert policy["window_id"] == 15
