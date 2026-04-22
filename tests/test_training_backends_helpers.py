"""Unit tests for pure helpers in reliquary.training.backends.

The train(...) paths on the four backends (Local*, TRLAsyncGRPOFSDP2,
TransformersSFT, NeMoRL) need transformers + trl + ray and are covered
by GPU-gated integration runs. These helpers (_group_operations,
_cluster_profile, _normalize_dense_execution_mode,
_write_backend_bundle, build_*_learner_backend factories) are pure
Python and cheap to pin.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from reliquary.training.backends import (
    LocalGRPOLearnerBackend,
    LocalReasoningStudentBackend,
    NeMoRLBackend,
    TransformersSFTLearnerBackend,
    TRLAsyncGRPOFSDP2Backend,
    _cluster_profile,
    _group_operations,
    _normalize_dense_execution_mode,
    _simulate_reasoning_checkpoint,
    _write_backend_bundle,
    build_distill_learner_backend,
    build_rl_learner_backend,
)


def _groups(*specs: list[str]) -> list[dict[str, Any]]:
    """Build minimal groups with nested sample task_metadata.operation."""
    return [
        {
            "samples": [
                {"task_metadata": {"operation": op}} for op in ops
            ]
        }
        for ops in specs
    ]


# ---------------------------------------------------------------------------
# _group_operations
# ---------------------------------------------------------------------------


def test_group_operations_empty_groups() -> None:
    assert _group_operations([]) == []


def test_group_operations_empty_samples() -> None:
    assert _group_operations([{"samples": []}]) == []


def test_group_operations_dedupes_and_sorts() -> None:
    groups = _groups(["add", "subtract"], ["subtract", "multiply"])
    assert _group_operations(groups) == ["add", "multiply", "subtract"]


def test_group_operations_skips_samples_missing_operation() -> None:
    groups = [
        {
            "samples": [
                {"task_metadata": {"operation": "add"}},
                {"task_metadata": {}},  # no operation
                {},  # no task_metadata
            ]
        }
    ]
    assert _group_operations(groups) == ["add"]


def test_group_operations_returns_sorted_list() -> None:
    groups = _groups(["zeta", "alpha", "mu"])
    assert _group_operations(groups) == ["alpha", "mu", "zeta"]


# ---------------------------------------------------------------------------
# _cluster_profile
# ---------------------------------------------------------------------------


def test_cluster_profile_defaults() -> None:
    profile = _cluster_profile({})
    assert profile["control_plane"] == "ray"
    assert profile["ray_head_address"] == "local"
    assert profile["actor_replicas"] == 1
    assert profile["learner_replicas"] == 1
    assert profile["evaluator_replicas"] == 1
    assert profile["actor_endpoints"] == []


def test_cluster_profile_cfg_overrides() -> None:
    cfg = {
        "ray_head_address": "ray://head:10001",
        "ray_actor_replicas": 4,
        "ray_learner_replicas": 2,
        "ray_evaluator_replicas": 8,
        "rl_vllm_actor_endpoints": "http://actor-1:8000, http://actor-2:8000",
    }
    profile = _cluster_profile(cfg)
    assert profile["ray_head_address"] == "ray://head:10001"
    assert profile["actor_replicas"] == 4
    assert profile["learner_replicas"] == 2
    assert profile["evaluator_replicas"] == 8
    assert profile["actor_endpoints"] == ["http://actor-1:8000", "http://actor-2:8000"]


def test_cluster_profile_endpoints_trim_whitespace_and_skip_empty() -> None:
    cfg = {"rl_vllm_actor_endpoints": " http://a:1 , , http://b:2  "}
    profile = _cluster_profile(cfg)
    assert profile["actor_endpoints"] == ["http://a:1", "http://b:2"]


def test_cluster_profile_numeric_env_as_str_coerced() -> None:
    cfg = {
        "ray_actor_replicas": "3",
        "ray_learner_replicas": "1",
        "ray_evaluator_replicas": "2",
    }
    profile = _cluster_profile(cfg)
    assert profile["actor_replicas"] == 3
    assert profile["learner_replicas"] == 1
    assert profile["evaluator_replicas"] == 2


# ---------------------------------------------------------------------------
# _normalize_dense_execution_mode
# ---------------------------------------------------------------------------


def test_normalize_dense_execution_mode_empty_to_local_compat() -> None:
    assert _normalize_dense_execution_mode("") == "local_compat"


def test_normalize_dense_execution_mode_simulate_to_local_compat() -> None:
    assert _normalize_dense_execution_mode("simulate") == "local_compat"
    assert _normalize_dense_execution_mode(" SIMULATE ") == "local_compat"


def test_normalize_dense_execution_mode_local_compat_passthrough() -> None:
    assert _normalize_dense_execution_mode("local_compat") == "local_compat"


def test_normalize_dense_execution_mode_ray_cluster_passthrough() -> None:
    assert _normalize_dense_execution_mode("ray_cluster") == "ray_cluster"


def test_normalize_dense_execution_mode_case_insensitive() -> None:
    assert _normalize_dense_execution_mode("RAY_CLUSTER") == "ray_cluster"
    assert _normalize_dense_execution_mode("LOCAL_COMPAT") == "local_compat"


def test_normalize_dense_execution_mode_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        _normalize_dense_execution_mode("unknown-mode")
    with pytest.raises(ValueError):
        _normalize_dense_execution_mode("dry_run")


def test_normalize_dense_execution_mode_none_input() -> None:
    assert _normalize_dense_execution_mode(None) == "local_compat"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _write_backend_bundle / _simulate_reasoning_checkpoint
# ---------------------------------------------------------------------------


def test_write_backend_bundle_writes_json(tmp_path: Path) -> None:
    payload = {"backend": "trl", "version": "1.0"}
    path = _write_backend_bundle(tmp_path, "bundle.json", payload)
    assert path == tmp_path / "bundle.json"
    assert path.is_file()
    assert json.loads(path.read_text()) == payload


def test_simulate_reasoning_checkpoint_writes_capabilities(tmp_path: Path) -> None:
    groups = _groups(["add", "subtract"], ["multiply"])
    _simulate_reasoning_checkpoint(tmp_path, groups)
    payload = json.loads((tmp_path / "reasoning_capabilities.json").read_text())
    assert payload["supported_operations"] == ["add", "multiply", "subtract"]
    assert payload["record_count"] == 3


def test_simulate_reasoning_checkpoint_empty_groups(tmp_path: Path) -> None:
    _simulate_reasoning_checkpoint(tmp_path, [])
    payload = json.loads((tmp_path / "reasoning_capabilities.json").read_text())
    assert payload["supported_operations"] == []
    assert payload["record_count"] == 0


# ---------------------------------------------------------------------------
# build_distill_learner_backend
# ---------------------------------------------------------------------------


def test_build_distill_local_reasoning_student() -> None:
    backend = build_distill_learner_backend("local_reasoning_student")
    assert isinstance(backend, LocalReasoningStudentBackend)


def test_build_distill_transformers_sft() -> None:
    backend = build_distill_learner_backend("transformers_sft")
    assert isinstance(backend, TransformersSFTLearnerBackend)


def test_build_distill_unknown_falls_back_to_local() -> None:
    backend = build_distill_learner_backend("not-a-known-name")
    assert isinstance(backend, LocalReasoningStudentBackend)


def test_build_distill_empty_falls_back_to_local() -> None:
    backend = build_distill_learner_backend("")
    assert isinstance(backend, LocalReasoningStudentBackend)


# ---------------------------------------------------------------------------
# build_rl_learner_backend
# ---------------------------------------------------------------------------


def test_build_rl_trl_async_grpo() -> None:
    backend = build_rl_learner_backend("trl_async_grpo")
    assert isinstance(backend, TRLAsyncGRPOFSDP2Backend)


def test_build_rl_trl_async_grpo_fsdp2_alias() -> None:
    backend = build_rl_learner_backend("trl_async_grpo_fsdp2")
    assert isinstance(backend, TRLAsyncGRPOFSDP2Backend)


def test_build_rl_nemo_rl() -> None:
    backend = build_rl_learner_backend("nemo_rl")
    assert isinstance(backend, NeMoRLBackend)


def test_build_rl_unknown_falls_back_to_local_grpo() -> None:
    backend = build_rl_learner_backend("not-a-known-name")
    assert isinstance(backend, LocalGRPOLearnerBackend)


def test_build_rl_empty_falls_back_to_local() -> None:
    backend = build_rl_learner_backend("")
    assert isinstance(backend, LocalGRPOLearnerBackend)


# ---------------------------------------------------------------------------
# Backend name attribute invariant
# ---------------------------------------------------------------------------


def test_backend_name_attributes_match_construction_names() -> None:
    assert TRLAsyncGRPOFSDP2Backend().name == "trl_async_grpo_fsdp2"
    assert TransformersSFTLearnerBackend().name == "transformers_sft"
    assert NeMoRLBackend().name == "nemo_rl"
    assert LocalGRPOLearnerBackend().name == "local_grpo"
    assert LocalReasoningStudentBackend().name == "local_reasoning_student"
