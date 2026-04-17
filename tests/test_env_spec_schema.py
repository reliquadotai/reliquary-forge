"""Acceptance tests for the permissionless env registry schema layer.

Covers the 20 acceptance cases in
``private/reliquary-plan/notes/spec-env-registry.md``.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from reliquary.envs.spec import (
    ENV_CONTEXT_LENGTH_MAX,
    ENV_SPEC_VERSION,
    ENV_TIMEOUT_MS_MAX,
    EnvMetadata,
    EnvSpec,
    EnvSpecInvalid,
    EnvSpecStatus,
    canonical_env_spec_bytes,
    env_spec_id,
    validate_env_spec,
)


def _valid_spec(**overrides) -> EnvSpec:
    base = dict(
        name="sat_classic",
        version="1.0.0",
        author_hotkey="A" * 48,
        author_stake_lock_tao=100.0,
        task_generator_hash="a" * 64,
        reward_function_hash="b" * 64,
        evaluator_hash="c" * 64,
        sample_task='{"prompt": "solve"}',
        sample_solution='{"answer": 42}',
        expected_reward=1.0,
        metadata=EnvMetadata(
            context_length=2048,
            timeout_ms=30_000,
            difficulty="medium",
            domain="math",
        ),
    )
    base.update(overrides)
    return EnvSpec(**base)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_validates() -> None:
    valid, errors = validate_env_spec(_valid_spec())
    assert valid is True
    assert errors == []


# ---------------------------------------------------------------------------
# Field-level validation failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_name",
    [
        "BadName",           # uppercase
        "a-b-c",             # hyphens not allowed
        "ab",                # too short
        "a" * 65,            # too long
        "1foo",              # must start with letter
    ],
)
def test_invalid_name(bad_name: str) -> None:
    valid, errors = validate_env_spec(_valid_spec(name=bad_name))
    assert valid is False
    assert any("name" in e for e in errors)


@pytest.mark.parametrize(
    "bad_version",
    ["1.0", "v1.0.0", "1.0.0-beta", "1", ""],
)
def test_invalid_version(bad_version: str) -> None:
    valid, errors = validate_env_spec(_valid_spec(version=bad_version))
    assert valid is False
    assert any("version" in e for e in errors)


@pytest.mark.parametrize(
    "bad_hotkey",
    ["", "A" * 10, "A" * 49, "ABC!" + "a" * 44],
)
def test_invalid_hotkey(bad_hotkey: str) -> None:
    valid, errors = validate_env_spec(_valid_spec(author_hotkey=bad_hotkey))
    assert valid is False
    assert any("author_hotkey" in e for e in errors)


@pytest.mark.parametrize(
    "hash_field",
    ["task_generator_hash", "reward_function_hash", "evaluator_hash"],
)
def test_invalid_code_hash_non_hex(hash_field: str) -> None:
    valid, errors = validate_env_spec(_valid_spec(**{hash_field: "Z" * 64}))
    assert valid is False
    assert any(hash_field in e for e in errors)


@pytest.mark.parametrize(
    "hash_field",
    ["task_generator_hash", "reward_function_hash", "evaluator_hash"],
)
def test_invalid_code_hash_wrong_length(hash_field: str) -> None:
    valid, errors = validate_env_spec(_valid_spec(**{hash_field: "a" * 63}))
    assert valid is False
    assert any(hash_field in e for e in errors)


def test_invalid_sample_task_json() -> None:
    valid, errors = validate_env_spec(_valid_spec(sample_task="{not-json"))
    assert valid is False
    assert any("sample_task" in e for e in errors)


def test_invalid_sample_solution_json() -> None:
    valid, errors = validate_env_spec(_valid_spec(sample_solution=""))
    assert valid is False
    assert any("sample_solution" in e for e in errors)


def test_invalid_context_length() -> None:
    bad = dataclasses.replace(
        _valid_spec(),
        metadata=EnvMetadata(
            context_length=0, timeout_ms=30_000, difficulty="easy", domain="math"
        ),
    )
    valid, errors = validate_env_spec(bad)
    assert valid is False
    assert any("context_length" in e for e in errors)


def test_invalid_context_length_over_max() -> None:
    bad = dataclasses.replace(
        _valid_spec(),
        metadata=EnvMetadata(
            context_length=ENV_CONTEXT_LENGTH_MAX + 1,
            timeout_ms=30_000,
            difficulty="easy",
            domain="math",
        ),
    )
    valid, errors = validate_env_spec(bad)
    assert valid is False
    assert any("context_length" in e for e in errors)


def test_invalid_timeout() -> None:
    bad = dataclasses.replace(
        _valid_spec(),
        metadata=EnvMetadata(
            context_length=2048, timeout_ms=0, difficulty="easy", domain="math"
        ),
    )
    valid, errors = validate_env_spec(bad)
    assert valid is False
    assert any("timeout_ms" in e for e in errors)


def test_invalid_timeout_over_max() -> None:
    bad = dataclasses.replace(
        _valid_spec(),
        metadata=EnvMetadata(
            context_length=2048,
            timeout_ms=ENV_TIMEOUT_MS_MAX + 1,
            difficulty="easy",
            domain="math",
        ),
    )
    valid, errors = validate_env_spec(bad)
    assert valid is False
    assert any("timeout_ms" in e for e in errors)


def test_unknown_difficulty() -> None:
    bad = dataclasses.replace(
        _valid_spec(),
        metadata=EnvMetadata(
            context_length=2048, timeout_ms=30_000, difficulty="extreme", domain="math"
        ),
    )
    valid, errors = validate_env_spec(bad)
    assert valid is False
    assert any("difficulty" in e for e in errors)


@pytest.mark.parametrize("bad_domain", ["MATH", "math!", "1domain", "a" * 33])
def test_invalid_domain(bad_domain: str) -> None:
    bad = dataclasses.replace(
        _valid_spec(),
        metadata=EnvMetadata(
            context_length=2048,
            timeout_ms=30_000,
            difficulty="easy",
            domain=bad_domain,
        ),
    )
    valid, errors = validate_env_spec(bad)
    assert valid is False
    assert any("domain" in e for e in errors)


def test_nan_expected_reward() -> None:
    valid, errors = validate_env_spec(_valid_spec(expected_reward=float("nan")))
    assert valid is False
    assert any("expected_reward" in e for e in errors)


def test_inf_expected_reward() -> None:
    valid, errors = validate_env_spec(_valid_spec(expected_reward=float("inf")))
    assert valid is False
    assert any("expected_reward" in e for e in errors)


def test_negative_stake() -> None:
    valid, errors = validate_env_spec(_valid_spec(author_stake_lock_tao=-1.0))
    assert valid is False
    assert any("author_stake_lock_tao" in e for e in errors)


def test_nan_stake() -> None:
    valid, errors = validate_env_spec(_valid_spec(author_stake_lock_tao=float("nan")))
    assert valid is False


def test_zero_stake_allowed() -> None:
    valid, errors = validate_env_spec(_valid_spec(author_stake_lock_tao=0.0))
    assert valid is True


# ---------------------------------------------------------------------------
# Canonical bytes + id
# ---------------------------------------------------------------------------


def test_canonical_env_spec_bytes_deterministic() -> None:
    spec = _valid_spec()
    assert canonical_env_spec_bytes(spec) == canonical_env_spec_bytes(spec)


def test_canonical_bytes_contains_all_fields() -> None:
    spec = _valid_spec()
    payload = json.loads(canonical_env_spec_bytes(spec).decode("utf-8"))
    assert payload["schema_version"] == ENV_SPEC_VERSION
    assert payload["name"] == spec.name
    assert payload["version"] == spec.version
    assert payload["author_hotkey"] == spec.author_hotkey
    assert payload["metadata"]["difficulty"] == spec.metadata.difficulty


def test_env_spec_id_is_hex_sha256() -> None:
    digest = env_spec_id(_valid_spec())
    assert len(digest) == 64
    assert all(ch in "0123456789abcdef" for ch in digest)


def test_env_spec_id_changes_with_any_field_mutation() -> None:
    base = _valid_spec()
    mutated = dataclasses.replace(base, expected_reward=2.0)
    assert env_spec_id(base) != env_spec_id(mutated)


def test_env_spec_id_changes_with_metadata_mutation() -> None:
    base = _valid_spec()
    mutated = dataclasses.replace(
        base,
        metadata=EnvMetadata(
            context_length=4096,
            timeout_ms=30_000,
            difficulty="medium",
            domain="math",
        ),
    )
    assert env_spec_id(base) != env_spec_id(mutated)


# ---------------------------------------------------------------------------
# from_dict
# ---------------------------------------------------------------------------


def _valid_payload() -> dict:
    return {
        "schema_version": ENV_SPEC_VERSION,
        "name": "sat_classic",
        "version": "1.0.0",
        "author_hotkey": "A" * 48,
        "author_stake_lock_tao": 100.0,
        "task_generator_hash": "a" * 64,
        "reward_function_hash": "b" * 64,
        "evaluator_hash": "c" * 64,
        "sample_task": '{"prompt": "solve"}',
        "sample_solution": '{"answer": 42}',
        "expected_reward": 1.0,
        "metadata": {
            "context_length": 2048,
            "timeout_ms": 30_000,
            "difficulty": "medium",
            "domain": "math",
        },
    }


def test_from_dict_happy_path() -> None:
    spec = EnvSpec.from_dict(_valid_payload())
    assert spec.name == "sat_classic"


def test_from_dict_rejects_wrong_schema_version() -> None:
    payload = _valid_payload()
    payload["schema_version"] = "v99"
    with pytest.raises(EnvSpecInvalid):
        EnvSpec.from_dict(payload)


def test_from_dict_rejects_invalid_fields() -> None:
    payload = _valid_payload()
    payload["name"] = "BadName"
    with pytest.raises(EnvSpecInvalid):
        EnvSpec.from_dict(payload)


def test_from_dict_rejects_non_dict() -> None:
    with pytest.raises(EnvSpecInvalid):
        EnvSpec.from_dict("not a dict")  # type: ignore[arg-type]


def test_from_dict_canonical_bytes_round_trip() -> None:
    spec = EnvSpec.from_dict(_valid_payload())
    canonical = canonical_env_spec_bytes(spec)
    parsed_back = json.loads(canonical.decode("utf-8"))
    assert parsed_back == {
        **_valid_payload(),
        # dataclass field order in canonical output: identical keys, values normalized
    }


def test_from_dict_missing_required_field() -> None:
    payload = _valid_payload()
    del payload["name"]
    with pytest.raises(EnvSpecInvalid):
        EnvSpec.from_dict(payload)


# ---------------------------------------------------------------------------
# Immutability + EnvSpecStatus + multi-error reporting
# ---------------------------------------------------------------------------


def test_env_spec_is_frozen() -> None:
    spec = _valid_spec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "mutated"  # type: ignore[misc]


def test_env_metadata_is_frozen() -> None:
    metadata = EnvMetadata(
        context_length=2048, timeout_ms=30_000, difficulty="easy", domain="math"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        metadata.difficulty = "hard"  # type: ignore[misc]


def test_env_spec_status_values_are_lowercase_strings() -> None:
    assert EnvSpecStatus.CANDIDATE.value == "candidate"
    assert EnvSpecStatus.SHADOW.value == "shadow"
    assert EnvSpecStatus.ACTIVE.value == "active"
    assert EnvSpecStatus.REVOKED.value == "revoked"


def test_multiple_errors_reported_simultaneously() -> None:
    bad = _valid_spec(name="BadName", version="not-semver")
    valid, errors = validate_env_spec(bad)
    assert valid is False
    assert len(errors) >= 2
    assert any("name" in e for e in errors)
    assert any("version" in e for e in errors)


def test_invalid_spec_raises_with_error_list_exposed() -> None:
    payload = _valid_payload()
    payload["name"] = "BadName"
    payload["version"] = "not-semver"
    try:
        EnvSpec.from_dict(payload)
    except EnvSpecInvalid as exc:
        assert isinstance(exc.errors, list)
        assert len(exc.errors) >= 2
    else:
        pytest.fail("EnvSpecInvalid not raised")
