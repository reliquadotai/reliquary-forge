"""Permissionless environment registry glue.

Ties together:
- ``spec.py``: EnvSpec + canonical bytes + content-addressable id.
- ``sandbox.py``: policy-enforced execution of author code.

The registry exposes:
- ``EnvBundle``: source code for task_generator + reward_function +
  evaluator + the spec's sample_task / sample_solution / expected_reward
  as a self-contained unit validators can exercise end-to-end.
- ``EnvBundle.code_hashes()``: sha256 of each source string (matches
  ``EnvSpec.*_hash``) so the spec can pin what bundle it commits to.
- ``validate_bundle_against_spec(bundle, spec)``: runs the sandbox
  re-execution of the sample task through the author code and checks
  reward parity with the declared ``expected_reward``.
- ``EnvRegistry``: in-memory store mapping ``env_spec_id`` → (spec, bundle,
  status). Persistence-agnostic (sqlite / R2 / onchain adapters land as
  separate modules later).
- ``register_submission``: the one-call flow validators use — audit
  bundle through sandbox, run sample, compare reward, transition
  CANDIDATE→SHADOW on success.

Spec: private/reliquary-plan/notes/spec-env-registry.md (schema) +
spec-env-sandbox.md (exec).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .sandbox import (
    SandboxExecutor,
    SandboxPolicy,
    SandboxPolicyViolation,
    SandboxResult,
)
from .spec import (
    EnvSpec,
    EnvSpecStatus,
    env_spec_id,
    validate_env_spec,
)

REWARD_TOLERANCE = 1e-6


class RegistryError(Exception):
    """Base class for registry-layer failures."""


class RegistryDuplicate(RegistryError):
    """Raised when an env_spec_id is registered twice."""


class RegistryNotFound(RegistryError):
    """Raised when an env_spec_id is fetched but not present."""


class BundleRejected(RegistryError):
    """Raised when the sandbox / sample-task check rejects the bundle."""


@dataclass(frozen=True)
class EnvBundle:
    """Three source strings + sample inputs comprising one submission."""

    task_generator_source: str
    reward_function_source: str
    evaluator_source: str
    task_generator_entry: str = "generate"
    reward_function_entry: str = "reward"
    evaluator_entry: str = "evaluate"

    def code_hashes(self) -> dict[str, str]:
        return {
            "task_generator_hash": hashlib.sha256(
                self.task_generator_source.encode("utf-8")
            ).hexdigest(),
            "reward_function_hash": hashlib.sha256(
                self.reward_function_source.encode("utf-8")
            ).hexdigest(),
            "evaluator_hash": hashlib.sha256(
                self.evaluator_source.encode("utf-8")
            ).hexdigest(),
        }


@dataclass
class RegistryEntry:
    spec: EnvSpec
    bundle: EnvBundle
    status: EnvSpecStatus = EnvSpecStatus.CANDIDATE
    validator_notes: list[str] = field(default_factory=list)


@dataclass
class BundleValidationReport:
    """Result of validating a bundle against the spec's sample."""

    valid: bool
    bundle_hashes: dict[str, str]
    observed_reward: float | None
    expected_reward: float
    sandbox_results: dict[str, SandboxResult]
    errors: list[str]


def validate_bundle_against_spec(
    bundle: EnvBundle,
    spec: EnvSpec,
    *,
    policy: SandboxPolicy | None = None,
) -> BundleValidationReport:
    """Re-run the author's sample task through the bundle code.

    Flow:
    1. Compare bundle.code_hashes() against the spec's declared hashes.
    2. Sandbox-audit all three sources.
    3. Call evaluator(sample_task, sample_solution) in the sandbox.
    4. Compare the returned reward to expected_reward within tolerance.

    Returns a structured report; raises ``BundleRejected`` if the spec
    hashes and bundle hashes disagree (nothing useful to validate past
    that point — the bundle is the wrong source for this spec).
    """
    executor = SandboxExecutor(policy or SandboxPolicy(timeout_ms=spec.metadata.timeout_ms))

    hashes = bundle.code_hashes()
    errors: list[str] = []
    sandbox_results: dict[str, SandboxResult] = {}

    if hashes["task_generator_hash"] != spec.task_generator_hash:
        errors.append(
            f"task_generator_hash mismatch: bundle={hashes['task_generator_hash']} "
            f"spec={spec.task_generator_hash}"
        )
    if hashes["reward_function_hash"] != spec.reward_function_hash:
        errors.append(
            f"reward_function_hash mismatch: bundle={hashes['reward_function_hash']} "
            f"spec={spec.reward_function_hash}"
        )
    if hashes["evaluator_hash"] != spec.evaluator_hash:
        errors.append(
            f"evaluator_hash mismatch: bundle={hashes['evaluator_hash']} "
            f"spec={spec.evaluator_hash}"
        )
    if errors:
        raise BundleRejected("; ".join(errors))

    # Audit every source first (cheap pre-check).
    for label, source in (
        ("task_generator", bundle.task_generator_source),
        ("reward_function", bundle.reward_function_source),
        ("evaluator", bundle.evaluator_source),
    ):
        try:
            executor.audit(source)
        except SandboxPolicyViolation as exc:
            errors.append(f"{label}: {exc}")
    if errors:
        return BundleValidationReport(
            valid=False,
            bundle_hashes=hashes,
            observed_reward=None,
            expected_reward=spec.expected_reward,
            sandbox_results={},
            errors=errors,
        )

    try:
        sample_task = json.loads(spec.sample_task)
        sample_solution = json.loads(spec.sample_solution)
    except json.JSONDecodeError as exc:
        errors.append(f"sample_task/sample_solution not parseable JSON: {exc}")
        return BundleValidationReport(
            valid=False,
            bundle_hashes=hashes,
            observed_reward=None,
            expected_reward=spec.expected_reward,
            sandbox_results={},
            errors=errors,
        )

    # Run the evaluator on the sample pair and compare its reward to
    # the declared expected_reward. That's the pre-quorum gate —
    # validators reject the submission before ever voting if the
    # author's claim doesn't hold up.
    evaluator_result = executor.run(
        bundle.evaluator_source,
        entry_point=bundle.evaluator_entry,
        args=[sample_task, sample_solution],
    )
    sandbox_results["evaluator"] = evaluator_result

    if not evaluator_result.ok:
        errors.append(
            f"evaluator failed: type={evaluator_result.error_type} "
            f"timeout={evaluator_result.timeout} oom={evaluator_result.oom} "
            f"message={evaluator_result.error_message}"
        )
        return BundleValidationReport(
            valid=False,
            bundle_hashes=hashes,
            observed_reward=None,
            expected_reward=spec.expected_reward,
            sandbox_results=sandbox_results,
            errors=errors,
        )

    observed = evaluator_result.value
    if not isinstance(observed, (int, float)):
        errors.append(
            f"evaluator returned non-numeric reward {observed!r}"
        )
        return BundleValidationReport(
            valid=False,
            bundle_hashes=hashes,
            observed_reward=None,
            expected_reward=spec.expected_reward,
            sandbox_results=sandbox_results,
            errors=errors,
        )

    observed_reward = float(observed)
    if abs(observed_reward - spec.expected_reward) > REWARD_TOLERANCE:
        errors.append(
            f"reward mismatch: observed={observed_reward} expected={spec.expected_reward} "
            f"(tolerance={REWARD_TOLERANCE})"
        )
        return BundleValidationReport(
            valid=False,
            bundle_hashes=hashes,
            observed_reward=observed_reward,
            expected_reward=spec.expected_reward,
            sandbox_results=sandbox_results,
            errors=errors,
        )

    return BundleValidationReport(
        valid=True,
        bundle_hashes=hashes,
        observed_reward=observed_reward,
        expected_reward=spec.expected_reward,
        sandbox_results=sandbox_results,
        errors=[],
    )


class EnvRegistry:
    """In-memory env registry; persistence-agnostic.

    Operators subclass or compose to add sqlite / R2 / onchain backing.
    The in-memory form is enough for tests + local dev.
    """

    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}

    def __contains__(self, env_id: str) -> bool:
        return env_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def list_ids(self) -> list[str]:
        return sorted(self._entries)

    def list_by_status(self, status: EnvSpecStatus) -> list[RegistryEntry]:
        return [entry for entry in self._entries.values() if entry.status == status]

    def get(self, env_id: str) -> RegistryEntry:
        if env_id not in self._entries:
            raise RegistryNotFound(env_id)
        return self._entries[env_id]

    def register(
        self,
        spec: EnvSpec,
        bundle: EnvBundle,
        *,
        initial_status: EnvSpecStatus = EnvSpecStatus.CANDIDATE,
    ) -> str:
        """Insert a validated spec + bundle. Caller is responsible for
        having already run ``validate_bundle_against_spec`` — ``register``
        does NOT re-execute the sandbox; it just records the entry."""
        is_valid, schema_errors = validate_env_spec(spec)
        if not is_valid:
            raise BundleRejected("; ".join(schema_errors))

        spec_hashes = {
            "task_generator_hash": spec.task_generator_hash,
            "reward_function_hash": spec.reward_function_hash,
            "evaluator_hash": spec.evaluator_hash,
        }
        bundle_hashes = bundle.code_hashes()
        for key, expected in spec_hashes.items():
            if bundle_hashes[key] != expected:
                raise BundleRejected(
                    f"{key} mismatch at register-time: "
                    f"bundle={bundle_hashes[key]} spec={expected}"
                )

        env_id = env_spec_id(spec)
        if env_id in self._entries:
            raise RegistryDuplicate(env_id)
        self._entries[env_id] = RegistryEntry(
            spec=spec, bundle=bundle, status=initial_status
        )
        return env_id

    def transition(
        self,
        env_id: str,
        new_status: EnvSpecStatus,
        *,
        note: str | None = None,
    ) -> None:
        entry = self.get(env_id)
        entry.status = new_status
        if note:
            entry.validator_notes.append(f"{new_status.value}: {note}")


def register_submission(
    registry: EnvRegistry,
    spec: EnvSpec,
    bundle: EnvBundle,
    *,
    policy: SandboxPolicy | None = None,
) -> tuple[str, BundleValidationReport]:
    """Validator's one-call submission flow.

    Validates the bundle through the sandbox; on success, registers the
    spec at CANDIDATE status and returns ``(env_spec_id, report)``. On
    failure, raises ``BundleRejected`` with the report errors attached.
    """
    report = validate_bundle_against_spec(bundle, spec, policy=policy)
    if not report.valid:
        raise BundleRejected("; ".join(report.errors))
    env_id = registry.register(spec, bundle, initial_status=EnvSpecStatus.CANDIDATE)
    return env_id, report


__all__ = [
    "REWARD_TOLERANCE",
    "BundleRejected",
    "BundleValidationReport",
    "EnvBundle",
    "EnvRegistry",
    "RegistryDuplicate",
    "RegistryEntry",
    "RegistryError",
    "RegistryNotFound",
    "register_submission",
    "validate_bundle_against_spec",
]
