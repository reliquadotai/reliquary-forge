"""Tests for the permissionless env registry glue layer."""

from __future__ import annotations

import hashlib

import pytest

pytest.importorskip("RestrictedPython", reason="RestrictedPython required")

from reliquary.envs.registry import (
    BundleRejected,
    BundleValidationReport,
    EnvBundle,
    EnvRegistry,
    RegistryDuplicate,
    RegistryNotFound,
    register_submission,
    validate_bundle_against_spec,
)
from reliquary.envs.spec import EnvMetadata, EnvSpec, EnvSpecStatus, env_spec_id


_EVALUATOR_SOURCE = (
    "def evaluate(task, solution):\n"
    "    target = task['answer']\n"
    "    guess = solution['guess']\n"
    "    return 1.0 if guess == target else 0.0\n"
)
_TASK_GEN_SOURCE = (
    "def generate(seed):\n"
    "    return {'answer': seed * 2}\n"
)
_REWARD_SOURCE = (
    "def reward(solution, target):\n"
    "    return 1.0 if solution == target else 0.0\n"
)


def _bundle(
    evaluator: str = _EVALUATOR_SOURCE,
    task_gen: str = _TASK_GEN_SOURCE,
    reward: str = _REWARD_SOURCE,
) -> EnvBundle:
    return EnvBundle(
        task_generator_source=task_gen,
        reward_function_source=reward,
        evaluator_source=evaluator,
    )


def _spec_from_bundle(
    bundle: EnvBundle,
    *,
    expected_reward: float = 1.0,
    sample_task: str = '{"answer": 42}',
    sample_solution: str = '{"guess": 42}',
    name: str = "sample_env",
) -> EnvSpec:
    hashes = bundle.code_hashes()
    return EnvSpec(
        name=name,
        version="1.0.0",
        author_hotkey="A" * 48,
        author_stake_lock_tao=100.0,
        task_generator_hash=hashes["task_generator_hash"],
        reward_function_hash=hashes["reward_function_hash"],
        evaluator_hash=hashes["evaluator_hash"],
        sample_task=sample_task,
        sample_solution=sample_solution,
        expected_reward=expected_reward,
        metadata=EnvMetadata(
            context_length=2048,
            timeout_ms=5_000,
            difficulty="easy",
            domain="math",
        ),
    )


# ---------------------------------------------------------------------------
# EnvBundle
# ---------------------------------------------------------------------------


def test_bundle_code_hashes_are_sha256() -> None:
    bundle = _bundle()
    hashes = bundle.code_hashes()
    assert (
        hashes["task_generator_hash"]
        == hashlib.sha256(_TASK_GEN_SOURCE.encode("utf-8")).hexdigest()
    )
    assert (
        hashes["reward_function_hash"]
        == hashlib.sha256(_REWARD_SOURCE.encode("utf-8")).hexdigest()
    )
    assert (
        hashes["evaluator_hash"]
        == hashlib.sha256(_EVALUATOR_SOURCE.encode("utf-8")).hexdigest()
    )


def test_bundle_default_entry_points() -> None:
    bundle = _bundle()
    assert bundle.task_generator_entry == "generate"
    assert bundle.reward_function_entry == "reward"
    assert bundle.evaluator_entry == "evaluate"


# ---------------------------------------------------------------------------
# validate_bundle_against_spec
# ---------------------------------------------------------------------------


def test_validate_happy_path() -> None:
    bundle = _bundle()
    spec = _spec_from_bundle(bundle)
    report = validate_bundle_against_spec(bundle, spec)
    assert report.valid is True
    assert report.observed_reward == pytest.approx(1.0)
    assert report.errors == []


def test_validate_hash_mismatch_raises() -> None:
    bundle = _bundle()
    spec = _spec_from_bundle(bundle)
    # Swap in a different bundle so the hashes no longer match.
    tainted = EnvBundle(
        task_generator_source=_TASK_GEN_SOURCE,
        reward_function_source=_REWARD_SOURCE,
        evaluator_source=_EVALUATOR_SOURCE + "# extra byte",
    )
    with pytest.raises(BundleRejected):
        validate_bundle_against_spec(tainted, spec)


def test_validate_reward_mismatch_flags() -> None:
    bundle = _bundle()
    spec = _spec_from_bundle(bundle, expected_reward=0.5)
    report = validate_bundle_against_spec(bundle, spec)
    assert report.valid is False
    assert report.observed_reward == pytest.approx(1.0)
    assert any("reward mismatch" in e for e in report.errors)


def test_validate_rejects_evaluator_with_import() -> None:
    malicious_evaluator = "import os\n" + _EVALUATOR_SOURCE
    bundle = EnvBundle(
        task_generator_source=_TASK_GEN_SOURCE,
        reward_function_source=_REWARD_SOURCE,
        evaluator_source=malicious_evaluator,
    )
    spec = _spec_from_bundle(bundle)
    report = validate_bundle_against_spec(bundle, spec)
    assert report.valid is False
    assert any("evaluator" in e for e in report.errors)


def test_validate_captures_evaluator_exception() -> None:
    broken_evaluator = "def evaluate(task, solution):\n    raise ValueError('bad')\n"
    bundle = EnvBundle(
        task_generator_source=_TASK_GEN_SOURCE,
        reward_function_source=_REWARD_SOURCE,
        evaluator_source=broken_evaluator,
    )
    spec = _spec_from_bundle(bundle)
    report = validate_bundle_against_spec(bundle, spec)
    assert report.valid is False
    assert any("evaluator failed" in e for e in report.errors)


def test_validate_non_numeric_reward_rejected() -> None:
    string_reward_evaluator = (
        "def evaluate(task, solution):\n"
        "    return 'not-a-number'\n"
    )
    bundle = EnvBundle(
        task_generator_source=_TASK_GEN_SOURCE,
        reward_function_source=_REWARD_SOURCE,
        evaluator_source=string_reward_evaluator,
    )
    spec = _spec_from_bundle(bundle)
    report = validate_bundle_against_spec(bundle, spec)
    assert report.valid is False
    assert any("non-numeric" in e for e in report.errors)


def test_validate_sample_task_bad_json() -> None:
    bundle = _bundle()
    spec = _spec_from_bundle(bundle, sample_task="not-json")
    # The EnvSpec itself catches bad JSON in validate_env_spec, so we
    # construct a spec that has already-valid JSON then overwrite one
    # field via dataclasses.replace to bypass the initial check.
    import dataclasses
    tainted = dataclasses.replace(spec, sample_task="{malformed")
    report = validate_bundle_against_spec(bundle, tainted)
    assert report.valid is False
    assert any("sample_task" in e for e in report.errors)


# ---------------------------------------------------------------------------
# EnvRegistry
# ---------------------------------------------------------------------------


def test_registry_empty_initial_state() -> None:
    registry = EnvRegistry()
    assert len(registry) == 0
    assert registry.list_ids() == []


def test_registry_register_stores_entry() -> None:
    registry = EnvRegistry()
    bundle = _bundle()
    spec = _spec_from_bundle(bundle)
    env_id = registry.register(spec, bundle)
    assert env_id == env_spec_id(spec)
    assert env_id in registry
    assert len(registry) == 1
    assert registry.list_ids() == [env_id]


def test_registry_register_duplicate_raises() -> None:
    registry = EnvRegistry()
    bundle = _bundle()
    spec = _spec_from_bundle(bundle)
    registry.register(spec, bundle)
    with pytest.raises(RegistryDuplicate):
        registry.register(spec, bundle)


def test_registry_get_missing_raises() -> None:
    registry = EnvRegistry()
    with pytest.raises(RegistryNotFound):
        registry.get("nonexistent-id")


def test_registry_register_rejects_hash_mismatch() -> None:
    registry = EnvRegistry()
    bundle = _bundle()
    # Spec records hashes that do NOT match the bundle.
    import dataclasses
    spec = _spec_from_bundle(bundle)
    tainted = dataclasses.replace(spec, evaluator_hash="0" * 64)
    with pytest.raises(BundleRejected):
        registry.register(tainted, bundle)


def test_registry_transition_updates_status() -> None:
    registry = EnvRegistry()
    bundle = _bundle()
    spec = _spec_from_bundle(bundle)
    env_id = registry.register(spec, bundle)
    registry.transition(env_id, EnvSpecStatus.SHADOW, note="quorum reached")
    entry = registry.get(env_id)
    assert entry.status == EnvSpecStatus.SHADOW
    assert any("quorum reached" in note for note in entry.validator_notes)


def test_registry_list_by_status() -> None:
    registry = EnvRegistry()
    bundle_a = _bundle()
    spec_a = _spec_from_bundle(bundle_a, name="env_a")
    bundle_b = EnvBundle(
        task_generator_source=_TASK_GEN_SOURCE + "# variant",
        reward_function_source=_REWARD_SOURCE,
        evaluator_source=_EVALUATOR_SOURCE,
    )
    spec_b = _spec_from_bundle(bundle_b, name="env_b")
    registry.register(spec_a, bundle_a, initial_status=EnvSpecStatus.ACTIVE)
    registry.register(spec_b, bundle_b, initial_status=EnvSpecStatus.CANDIDATE)

    actives = registry.list_by_status(EnvSpecStatus.ACTIVE)
    candidates = registry.list_by_status(EnvSpecStatus.CANDIDATE)
    assert len(actives) == 1
    assert actives[0].spec.name == "env_a"
    assert len(candidates) == 1
    assert candidates[0].spec.name == "env_b"


# ---------------------------------------------------------------------------
# register_submission
# ---------------------------------------------------------------------------


def test_register_submission_happy_path() -> None:
    registry = EnvRegistry()
    bundle = _bundle()
    spec = _spec_from_bundle(bundle)
    env_id, report = register_submission(registry, spec, bundle)
    assert env_id == env_spec_id(spec)
    assert report.valid is True
    assert env_id in registry
    assert registry.get(env_id).status == EnvSpecStatus.CANDIDATE


def test_register_submission_rejects_bad_sample() -> None:
    registry = EnvRegistry()
    bundle = _bundle()
    spec = _spec_from_bundle(bundle, expected_reward=0.5)  # sample yields 1.0
    with pytest.raises(BundleRejected):
        register_submission(registry, spec, bundle)
    assert len(registry) == 0  # registry untouched


def test_register_submission_rejects_malicious_source() -> None:
    malicious = "import os\n" + _EVALUATOR_SOURCE
    bundle = EnvBundle(
        task_generator_source=_TASK_GEN_SOURCE,
        reward_function_source=_REWARD_SOURCE,
        evaluator_source=malicious,
    )
    spec = _spec_from_bundle(bundle)
    registry = EnvRegistry()
    with pytest.raises(BundleRejected):
        register_submission(registry, spec, bundle)
    assert len(registry) == 0
