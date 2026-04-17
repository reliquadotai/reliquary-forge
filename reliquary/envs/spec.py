"""Permissionless environment registry — schema layer v1.

First deliverable of Tier 3 Epic 2. Defines the structured ``EnvSpec``
dataclass, canonical-JSON serialization, content-addressable id, and
the ``validate_env_spec`` checker. The sandbox executor + onchain
submission flow are separate follow-up deliverables.

Spec: private/reliquary-plan/notes/spec-env-registry.md.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


ENV_SPEC_VERSION = "v1"
ENV_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
ENV_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
HOTKEY_PATTERN = re.compile(r"^[A-Za-z0-9]{48}$")
DOMAIN_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
ENV_DIFFICULTIES = frozenset({"easy", "medium", "hard"})
ENV_CONTEXT_LENGTH_MAX = 131_072
ENV_TIMEOUT_MS_MAX = 600_000
HEX_64_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class EnvSpecError(Exception):
    """Base exception for env-spec failures."""


class EnvSpecInvalid(EnvSpecError):
    """Raised when a spec payload fails to validate."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        message = "; ".join(errors) if errors else "env spec invalid"
        super().__init__(message)


class EnvSpecStatus(str, Enum):
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    ACTIVE = "active"
    REVOKED = "revoked"


@dataclass(frozen=True)
class EnvMetadata:
    context_length: int
    timeout_ms: int
    difficulty: str
    domain: str


@dataclass(frozen=True)
class EnvSpec:
    name: str
    version: str
    author_hotkey: str
    author_stake_lock_tao: float
    task_generator_hash: str
    reward_function_hash: str
    evaluator_hash: str
    sample_task: str
    sample_solution: str
    expected_reward: float
    metadata: EnvMetadata

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "EnvSpec":
        """Build an ``EnvSpec`` from a JSON-decoded dict.

        Enforces ``schema_version == ENV_SPEC_VERSION`` and runs
        ``validate_env_spec`` — raises ``EnvSpecInvalid`` on either failure.
        """
        if not isinstance(payload, dict):
            raise EnvSpecInvalid(["payload is not a dict"])
        schema_version = payload.get("schema_version")
        if schema_version != ENV_SPEC_VERSION:
            raise EnvSpecInvalid(
                [f"schema_version {schema_version!r} != {ENV_SPEC_VERSION!r}"]
            )
        try:
            metadata_payload = payload.get("metadata") or {}
            metadata = EnvMetadata(
                context_length=int(metadata_payload["context_length"]),
                timeout_ms=int(metadata_payload["timeout_ms"]),
                difficulty=str(metadata_payload["difficulty"]),
                domain=str(metadata_payload["domain"]),
            )
            spec = EnvSpec(
                name=str(payload["name"]),
                version=str(payload["version"]),
                author_hotkey=str(payload["author_hotkey"]),
                author_stake_lock_tao=float(payload["author_stake_lock_tao"]),
                task_generator_hash=str(payload["task_generator_hash"]),
                reward_function_hash=str(payload["reward_function_hash"]),
                evaluator_hash=str(payload["evaluator_hash"]),
                sample_task=str(payload["sample_task"]),
                sample_solution=str(payload["sample_solution"]),
                expected_reward=float(payload["expected_reward"]),
                metadata=metadata,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EnvSpecInvalid([f"payload construction failed: {exc}"]) from exc

        is_valid, errors = validate_env_spec(spec)
        if not is_valid:
            raise EnvSpecInvalid(errors)
        return spec


def _check_regex(
    errors: list[str],
    *,
    field_name: str,
    value: str,
    pattern: re.Pattern[str],
) -> None:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        errors.append(
            f"{field_name} {value!r} does not match {pattern.pattern}"
        )


def validate_env_spec(spec: EnvSpec) -> tuple[bool, list[str]]:
    """Return ``(is_valid, errors)`` — never raises.

    Collects every violation so callers can surface them all at once.
    """
    errors: list[str] = []

    _check_regex(errors, field_name="name", value=spec.name, pattern=ENV_NAME_PATTERN)
    _check_regex(errors, field_name="version", value=spec.version, pattern=ENV_VERSION_PATTERN)
    _check_regex(
        errors,
        field_name="author_hotkey",
        value=spec.author_hotkey,
        pattern=HOTKEY_PATTERN,
    )
    _check_regex(
        errors,
        field_name="task_generator_hash",
        value=spec.task_generator_hash,
        pattern=HEX_64_PATTERN,
    )
    _check_regex(
        errors,
        field_name="reward_function_hash",
        value=spec.reward_function_hash,
        pattern=HEX_64_PATTERN,
    )
    _check_regex(
        errors,
        field_name="evaluator_hash",
        value=spec.evaluator_hash,
        pattern=HEX_64_PATTERN,
    )

    if (
        not isinstance(spec.author_stake_lock_tao, (int, float))
        or isinstance(spec.author_stake_lock_tao, bool)
        or math.isnan(spec.author_stake_lock_tao)
        or math.isinf(spec.author_stake_lock_tao)
        or spec.author_stake_lock_tao < 0
    ):
        errors.append(
            f"author_stake_lock_tao must be a finite non-negative float, got "
            f"{spec.author_stake_lock_tao!r}"
        )

    if (
        not isinstance(spec.expected_reward, (int, float))
        or isinstance(spec.expected_reward, bool)
        or math.isnan(spec.expected_reward)
        or math.isinf(spec.expected_reward)
    ):
        errors.append(
            f"expected_reward must be finite, got {spec.expected_reward!r}"
        )

    _validate_json(errors, field_name="sample_task", value=spec.sample_task)
    _validate_json(errors, field_name="sample_solution", value=spec.sample_solution)

    _validate_metadata(errors, metadata=spec.metadata)

    return (not errors, errors)


def _validate_json(errors: list[str], *, field_name: str, value: str) -> None:
    if not isinstance(value, str):
        errors.append(f"{field_name} must be a string, got {type(value).__name__}")
        return
    try:
        json.loads(value)
    except json.JSONDecodeError as exc:
        errors.append(f"{field_name} is not valid JSON: {exc.msg}")


def _validate_metadata(errors: list[str], *, metadata: EnvMetadata) -> None:
    if (
        not isinstance(metadata.context_length, int)
        or isinstance(metadata.context_length, bool)
        or not (1 <= metadata.context_length <= ENV_CONTEXT_LENGTH_MAX)
    ):
        errors.append(
            f"metadata.context_length must be int in [1, {ENV_CONTEXT_LENGTH_MAX}], "
            f"got {metadata.context_length!r}"
        )
    if (
        not isinstance(metadata.timeout_ms, int)
        or isinstance(metadata.timeout_ms, bool)
        or not (1 <= metadata.timeout_ms <= ENV_TIMEOUT_MS_MAX)
    ):
        errors.append(
            f"metadata.timeout_ms must be int in [1, {ENV_TIMEOUT_MS_MAX}], "
            f"got {metadata.timeout_ms!r}"
        )
    if metadata.difficulty not in ENV_DIFFICULTIES:
        errors.append(
            f"metadata.difficulty {metadata.difficulty!r} not in {sorted(ENV_DIFFICULTIES)}"
        )
    if not isinstance(metadata.domain, str) or not DOMAIN_PATTERN.fullmatch(metadata.domain):
        errors.append(
            f"metadata.domain {metadata.domain!r} does not match {DOMAIN_PATTERN.pattern}"
        )


def canonical_env_spec_bytes(spec: EnvSpec) -> bytes:
    """Return the deterministic byte-string used for signing + onchain commit."""
    payload = {
        "schema_version": ENV_SPEC_VERSION,
        "name": spec.name,
        "version": spec.version,
        "author_hotkey": spec.author_hotkey,
        "author_stake_lock_tao": float(spec.author_stake_lock_tao),
        "task_generator_hash": spec.task_generator_hash,
        "reward_function_hash": spec.reward_function_hash,
        "evaluator_hash": spec.evaluator_hash,
        "sample_task": spec.sample_task,
        "sample_solution": spec.sample_solution,
        "expected_reward": float(spec.expected_reward),
        "metadata": {
            "context_length": int(spec.metadata.context_length),
            "timeout_ms": int(spec.metadata.timeout_ms),
            "difficulty": spec.metadata.difficulty,
            "domain": spec.metadata.domain,
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def env_spec_id(spec: EnvSpec) -> str:
    """Content-addressable id: sha256 over the canonical bytes."""
    return hashlib.sha256(canonical_env_spec_bytes(spec)).hexdigest()


__all__ = [
    "ENV_CONTEXT_LENGTH_MAX",
    "ENV_DIFFICULTIES",
    "ENV_NAME_PATTERN",
    "ENV_SPEC_VERSION",
    "ENV_TIMEOUT_MS_MAX",
    "ENV_VERSION_PATTERN",
    "DOMAIN_PATTERN",
    "HOTKEY_PATTERN",
    "HEX_64_PATTERN",
    "EnvMetadata",
    "EnvSpec",
    "EnvSpecError",
    "EnvSpecInvalid",
    "EnvSpecStatus",
    "canonical_env_spec_bytes",
    "env_spec_id",
    "validate_env_spec",
]
