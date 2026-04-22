"""Tests for first-class SAT + arithmetic registry entries.

These dogfood the permissionless env pipeline: the two envs express
their task_generator/reward/evaluator as author-style source strings,
pass through the sandbox on validate, and register into an EnvRegistry
just like any external submission would.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("RestrictedPython", reason="RestrictedPython required")

from reliquary.envs.registry import (
    EnvRegistry,
    register_submission,
    validate_bundle_against_spec,
)
from reliquary.envs.sample_envs import (
    ARITHMETIC_SAMPLE_SOLUTION,
    ARITHMETIC_SAMPLE_TASK,
    SAT_SAMPLE_SOLUTION,
    SAT_SAMPLE_TASK,
    build_arithmetic_bundle,
    build_arithmetic_spec,
    build_sat_bundle,
    build_sat_spec,
)
from reliquary.envs.sandbox import SandboxExecutor
from reliquary.envs.spec import EnvSpecStatus

# ---------------------------------------------------------------------------
# SAT env
# ---------------------------------------------------------------------------


def test_sat_bundle_hashes_match_spec() -> None:
    bundle = build_sat_bundle()
    spec = build_sat_spec()
    hashes = bundle.code_hashes()
    assert hashes["task_generator_hash"] == spec.task_generator_hash
    assert hashes["reward_function_hash"] == spec.reward_function_hash
    assert hashes["evaluator_hash"] == spec.evaluator_hash


def test_sat_sample_task_is_valid_json() -> None:
    parsed_task = json.loads(SAT_SAMPLE_TASK)
    parsed_solution = json.loads(SAT_SAMPLE_SOLUTION)
    assert parsed_task["n"] == 3
    assert parsed_solution["assignment"]["1"] is True


def test_sat_bundle_validates_against_spec() -> None:
    bundle = build_sat_bundle()
    spec = build_sat_spec()
    report = validate_bundle_against_spec(bundle, spec)
    assert report.valid is True
    assert report.observed_reward == pytest.approx(1.0)
    assert report.errors == []


def test_sat_bundle_evaluator_rejects_wrong_assignment() -> None:
    bundle = build_sat_bundle()
    executor = SandboxExecutor()
    wrong_solution = {"assignment": {"1": False, "2": False, "3": True}}
    result = executor.run(
        bundle.evaluator_source,
        entry_point=bundle.evaluator_entry,
        args=[json.loads(SAT_SAMPLE_TASK), wrong_solution],
    )
    assert result.ok
    assert result.value == 0.0


def test_sat_env_registers_at_candidate() -> None:
    registry = EnvRegistry()
    spec = build_sat_spec()
    bundle = build_sat_bundle()
    env_id, report = register_submission(registry, spec, bundle)
    entry = registry.get(env_id)
    assert report.valid is True
    assert entry.status == EnvSpecStatus.CANDIDATE
    assert entry.spec.name == "sat_classic"


def test_sat_spec_author_hotkey_override() -> None:
    spec = build_sat_spec(author_hotkey="B" * 48, stake_lock_tao=250.0)
    assert spec.author_hotkey == "B" * 48
    assert spec.author_stake_lock_tao == 250.0


def test_sat_task_generator_produces_expected_clauses() -> None:
    bundle = build_sat_bundle()
    executor = SandboxExecutor()
    result = executor.run(
        bundle.task_generator_source,
        entry_point=bundle.task_generator_entry,
        args=[0],
    )
    assert result.ok
    assert result.value["n"] == 3
    assert len(result.value["clauses"]) == 3


# ---------------------------------------------------------------------------
# Arithmetic env
# ---------------------------------------------------------------------------


def test_arithmetic_bundle_hashes_match_spec() -> None:
    bundle = build_arithmetic_bundle()
    spec = build_arithmetic_spec()
    hashes = bundle.code_hashes()
    assert hashes["task_generator_hash"] == spec.task_generator_hash
    assert hashes["reward_function_hash"] == spec.reward_function_hash
    assert hashes["evaluator_hash"] == spec.evaluator_hash


def test_arithmetic_sample_task_is_valid_json() -> None:
    parsed_task = json.loads(ARITHMETIC_SAMPLE_TASK)
    parsed_solution = json.loads(ARITHMETIC_SAMPLE_SOLUTION)
    assert parsed_task["answer"] == 17
    assert parsed_solution["final_answer"] == 17


def test_arithmetic_bundle_validates_against_spec() -> None:
    bundle = build_arithmetic_bundle()
    spec = build_arithmetic_spec()
    report = validate_bundle_against_spec(bundle, spec)
    assert report.valid is True
    assert report.observed_reward == pytest.approx(1.0)


def test_arithmetic_evaluator_wrong_answer_zero() -> None:
    bundle = build_arithmetic_bundle()
    executor = SandboxExecutor()
    result = executor.run(
        bundle.evaluator_source,
        entry_point=bundle.evaluator_entry,
        args=[json.loads(ARITHMETIC_SAMPLE_TASK), {"final_answer": 42}],
    )
    assert result.ok
    assert result.value == 0.0


def test_arithmetic_evaluator_tolerates_string_number() -> None:
    bundle = build_arithmetic_bundle()
    executor = SandboxExecutor()
    result = executor.run(
        bundle.evaluator_source,
        entry_point=bundle.evaluator_entry,
        args=[json.loads(ARITHMETIC_SAMPLE_TASK), {"final_answer": "17"}],
    )
    assert result.ok
    assert result.value == 1.0


def test_arithmetic_evaluator_handles_non_numeric() -> None:
    bundle = build_arithmetic_bundle()
    executor = SandboxExecutor()
    result = executor.run(
        bundle.evaluator_source,
        entry_point=bundle.evaluator_entry,
        args=[json.loads(ARITHMETIC_SAMPLE_TASK), {"final_answer": "not-a-number"}],
    )
    assert result.ok
    assert result.value == 0.0


def test_arithmetic_env_registers_at_candidate() -> None:
    registry = EnvRegistry()
    spec = build_arithmetic_spec()
    bundle = build_arithmetic_bundle()
    env_id, report = register_submission(registry, spec, bundle)
    entry = registry.get(env_id)
    assert report.valid is True
    assert entry.status == EnvSpecStatus.CANDIDATE
    assert entry.spec.name == "arithmetic_basic"


def test_arithmetic_task_generator_is_seed_deterministic() -> None:
    bundle = build_arithmetic_bundle()
    executor = SandboxExecutor()
    result_a = executor.run(
        bundle.task_generator_source, entry_point=bundle.task_generator_entry, args=[5]
    )
    result_b = executor.run(
        bundle.task_generator_source, entry_point=bundle.task_generator_entry, args=[5]
    )
    assert result_a.value == result_b.value


def test_arithmetic_task_generator_different_seed_differs() -> None:
    bundle = build_arithmetic_bundle()
    executor = SandboxExecutor()
    result_a = executor.run(
        bundle.task_generator_source, entry_point=bundle.task_generator_entry, args=[0]
    )
    result_b = executor.run(
        bundle.task_generator_source, entry_point=bundle.task_generator_entry, args=[3]
    )
    assert result_a.value != result_b.value


# ---------------------------------------------------------------------------
# Combined: both envs in one registry
# ---------------------------------------------------------------------------


def test_both_envs_coexist_in_registry() -> None:
    registry = EnvRegistry()
    sat_id, _ = register_submission(registry, build_sat_spec(), build_sat_bundle())
    arith_id, _ = register_submission(
        registry, build_arithmetic_spec(), build_arithmetic_bundle()
    )
    assert sat_id != arith_id
    assert len(registry) == 2
    ids = set(registry.list_ids())
    assert ids == {sat_id, arith_id}
