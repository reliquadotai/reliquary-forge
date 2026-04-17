"""Unit tests for role-level tasking + distill-trace generation helpers."""

from __future__ import annotations

from typing import Any

import pytest

from reliquary.distill.teacher import verify_trace_signature
from reliquary.protocol.artifacts import finalize_policy_artifact, make_policy_artifact
from reliquary.roles.generation import generate_distill_traces
from reliquary.roles.tasking import generate_reasoning_task_batch_artifact


def _cfg(**overrides: Any) -> dict:
    base = {
        "trainer_id": "trainer-1",
        "miner_id": "miner-1",
        "miner_secret": "super-secret",
        "model_teacher": "local-reasoning-teacher-v1",
        "distill_samples_per_task": 2,
        "max_new_tokens": 48,
        "vllm_timeout_seconds": 120,
        "vllm_temperature": 0.2,
    }
    base.update(overrides)
    return base


def _policy() -> dict:
    draft = make_policy_artifact(
        producer_id="policy-auth",
        window_id=3,
        model_ref="qwen3-4b",
        env_id="reasoning",
        env_config={},
        generation_config={},
        trainer_config={},
    )
    return finalize_policy_artifact(draft)


# ---------------------------------------------------------------------------
# generate_reasoning_task_batch_artifact
# ---------------------------------------------------------------------------


def test_task_batch_has_expected_task_count() -> None:
    batch = generate_reasoning_task_batch_artifact(
        _cfg(), _policy(), window_id=3, count=5
    )
    assert len(batch["payload"]["tasks"]) == 5


def test_task_batch_artifact_type_and_lane() -> None:
    batch = generate_reasoning_task_batch_artifact(
        _cfg(), _policy(), window_id=3, count=2
    )
    assert batch["artifact_type"] == "task_batch"
    assert batch["lane"] == "distill"
    assert batch["commodity_use"] == "distill_only"


def test_task_batch_id_formatted_with_four_digit_window() -> None:
    batch = generate_reasoning_task_batch_artifact(
        _cfg(), _policy(), window_id=7, count=1
    )
    assert batch["payload"]["task_batch_id"] == "reasoning-train-0007"


def test_task_batch_task_family_and_split() -> None:
    batch = generate_reasoning_task_batch_artifact(
        _cfg(), _policy(), window_id=1, count=1
    )
    assert batch["payload"]["task_family"] == "reasoning_traces"
    assert batch["payload"]["split"] == "train"


def test_task_batch_split_override() -> None:
    batch = generate_reasoning_task_batch_artifact(
        _cfg(), _policy(), window_id=1, count=1, split="holdout"
    )
    assert batch["payload"]["split"] == "holdout"


def test_task_batch_seed_offset_cascades_through_task_ids() -> None:
    batch_a = generate_reasoning_task_batch_artifact(
        _cfg(), _policy(), window_id=1, count=2, seed_offset=0
    )
    batch_b = generate_reasoning_task_batch_artifact(
        _cfg(), _policy(), window_id=1, count=2, seed_offset=1
    )
    ids_a = [t["task_id"] for t in batch_a["payload"]["tasks"]]
    ids_b = [t["task_id"] for t in batch_b["payload"]["tasks"]]
    assert ids_a != ids_b


def test_task_batch_parent_ids_points_to_policy() -> None:
    policy = _policy()
    batch = generate_reasoning_task_batch_artifact(_cfg(), policy, window_id=1, count=1)
    assert batch["parent_ids"] == [policy["artifact_id"]]


def test_task_batch_producer_role_is_policy_authority() -> None:
    batch = generate_reasoning_task_batch_artifact(
        _cfg(), _policy(), window_id=1, count=1
    )
    assert batch["producer_role"] == "policy_authority"


# ---------------------------------------------------------------------------
# generate_distill_traces
# ---------------------------------------------------------------------------


def test_distill_traces_produces_one_per_sample_per_task() -> None:
    cfg = _cfg(distill_samples_per_task=3)
    batch = generate_reasoning_task_batch_artifact(cfg, _policy(), window_id=1, count=4)
    traces = generate_distill_traces(
        cfg,
        _policy(),
        batch,
        window_id=1,
        miner_profile={"strategy": "reasoned_exact"},
    )
    assert len(traces) == 4 * 3


def test_distill_traces_artifact_type_is_distill_trace() -> None:
    cfg = _cfg(distill_samples_per_task=1)
    batch = generate_reasoning_task_batch_artifact(cfg, _policy(), window_id=1, count=1)
    traces = generate_distill_traces(
        cfg,
        _policy(),
        batch,
        window_id=1,
        miner_profile={"strategy": "reasoned_exact"},
    )
    assert traces[0]["artifact_type"] == "distill_trace"


def test_distill_traces_lane_and_commodity() -> None:
    cfg = _cfg(distill_samples_per_task=1)
    batch = generate_reasoning_task_batch_artifact(cfg, _policy(), window_id=1, count=1)
    traces = generate_distill_traces(
        cfg,
        _policy(),
        batch,
        window_id=1,
        miner_profile={"strategy": "reasoned_exact"},
    )
    assert traces[0]["lane"] == "distill"
    assert traces[0]["commodity_use"] == "dual_use"


def test_distill_traces_parent_ids_include_policy_and_task_batch() -> None:
    cfg = _cfg(distill_samples_per_task=1)
    policy = _policy()
    batch = generate_reasoning_task_batch_artifact(cfg, policy, window_id=1, count=1)
    traces = generate_distill_traces(
        cfg,
        policy,
        batch,
        window_id=1,
        miner_profile={"strategy": "reasoned_exact"},
    )
    parents = traces[0]["parent_ids"]
    assert policy["artifact_id"] in parents
    assert batch["artifact_id"] in parents


def test_distill_traces_signature_verifies_with_config_secret() -> None:
    cfg = _cfg(miner_secret="trace-secret", distill_samples_per_task=1)
    batch = generate_reasoning_task_batch_artifact(cfg, _policy(), window_id=1, count=1)
    traces = generate_distill_traces(
        cfg,
        _policy(),
        batch,
        window_id=1,
        miner_profile={"strategy": "reasoned_exact"},
    )
    payload = traces[0]["payload"]
    result = verify_trace_signature(
        task_id=payload["task_id"],
        trace_text=payload["trace"]["trace_text"],
        final_answer=payload["trace"]["final_answer"],
        secret="trace-secret",
        signature=payload["trace"]["teacher_signature"],
    )
    assert result["valid"] is True


def test_distill_traces_miner_profile_override_secret() -> None:
    cfg = _cfg(miner_secret="base-secret", distill_samples_per_task=1)
    batch = generate_reasoning_task_batch_artifact(cfg, _policy(), window_id=1, count=1)
    traces = generate_distill_traces(
        cfg,
        _policy(),
        batch,
        window_id=1,
        miner_profile={
            "strategy": "reasoned_exact",
            "miner_secret": "override-secret",
        },
    )
    payload = traces[0]["payload"]
    # base-secret no longer verifies
    assert (
        verify_trace_signature(
            task_id=payload["task_id"],
            trace_text=payload["trace"]["trace_text"],
            final_answer=payload["trace"]["final_answer"],
            secret="base-secret",
            signature=payload["trace"]["teacher_signature"],
        )["valid"]
        is False
    )
    # override-secret does
    assert (
        verify_trace_signature(
            task_id=payload["task_id"],
            trace_text=payload["trace"]["trace_text"],
            final_answer=payload["trace"]["final_answer"],
            secret="override-secret",
            signature=payload["trace"]["teacher_signature"],
        )["valid"]
        is True
    )


def test_distill_traces_records_miner_profile_fields() -> None:
    cfg = _cfg(distill_samples_per_task=1)
    batch = generate_reasoning_task_batch_artifact(cfg, _policy(), window_id=1, count=1)
    traces = generate_distill_traces(
        cfg,
        _policy(),
        batch,
        window_id=1,
        miner_profile={
            "miner_id": "custom-miner",
            "archetype": "careful_reasoner",
            "strategy": "reasoned_exact",
        },
    )
    profile = traces[0]["payload"]["miner_profile"]
    assert profile["miner_id"] == "custom-miner"
    assert profile["archetype"] == "careful_reasoner"
    assert profile["strategy"] == "reasoned_exact"


def test_distill_traces_signature_bundle_scheme() -> None:
    cfg = _cfg(distill_samples_per_task=1)
    batch = generate_reasoning_task_batch_artifact(cfg, _policy(), window_id=1, count=1)
    traces = generate_distill_traces(
        cfg,
        _policy(),
        batch,
        window_id=1,
        miner_profile={"strategy": "reasoned_exact"},
    )
    bundle = traces[0]["signature_bundle"]
    assert bundle["scheme"] == "local-hmac"
    assert "miner_id" in bundle


def test_distill_traces_sample_per_task_defaults_from_config() -> None:
    cfg = _cfg(distill_samples_per_task=2)
    batch = generate_reasoning_task_batch_artifact(cfg, _policy(), window_id=1, count=3)
    traces = generate_distill_traces(
        cfg,
        _policy(),
        batch,
        window_id=1,
        miner_profile={"strategy": "reasoned_exact"},
    )
    assert len(traces) == 3 * 2


def test_distill_traces_samples_per_task_override() -> None:
    cfg = _cfg(distill_samples_per_task=5)
    batch = generate_reasoning_task_batch_artifact(cfg, _policy(), window_id=1, count=2)
    traces = generate_distill_traces(
        cfg,
        _policy(),
        batch,
        window_id=1,
        miner_profile={"strategy": "reasoned_exact"},
        samples_per_task=1,
    )
    # Override wins over cfg.
    assert len(traces) == 2
