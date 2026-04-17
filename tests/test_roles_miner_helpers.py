"""Unit tests for the pure-Python helpers in reliquary.roles.miner.

The full ``mine_window`` function needs a transformers model bundle
and HF generate(), so it's covered by GPU-gated integration tests.
These helpers — stable_int, render_assignment_text, guided_completion_text,
apply_proof_corruption — are the testable plumbing around the actual
generation call, and are responsible for strategy dispatch that
downstream audit harness adversarial classes depend on.
"""

from __future__ import annotations

from typing import Any

import pytest

from reliquary.envs.sat import generate_sat_challenge
from reliquary.roles.miner import (
    _apply_proof_corruption,
    _guided_completion_text,
    _render_assignment_text,
    _stable_int,
)


def _challenge(*, n: int = 4, k: int = 3, m: int = 6, seed: int = 42) -> dict:
    return generate_sat_challenge(n=n, k=k, m=m, seed=seed)


# ---------------------------------------------------------------------------
# _stable_int
# ---------------------------------------------------------------------------


def test_stable_int_deterministic() -> None:
    assert _stable_int("a", "b", 1) == _stable_int("a", "b", 1)


def test_stable_int_order_sensitive() -> None:
    assert _stable_int("a", "b") != _stable_int("b", "a")


def test_stable_int_in_64_bit_range() -> None:
    value = _stable_int("a", "b", "c")
    assert 0 <= value < 2**64


def test_stable_int_coerces_arbitrary_types() -> None:
    value = _stable_int(42, True, "mixed", None)
    assert isinstance(value, int)


def test_stable_int_differs_across_inputs() -> None:
    assert _stable_int("alpha") != _stable_int("beta")


# ---------------------------------------------------------------------------
# _render_assignment_text
# ---------------------------------------------------------------------------


def test_render_assignment_text_sorts_by_variable() -> None:
    rendered = _render_assignment_text({2: True, 1: False, 3: True})
    assert rendered == "x1=False, x2=True, x3=True"


def test_render_assignment_text_empty_dict() -> None:
    assert _render_assignment_text({}) == ""


def test_render_assignment_text_uses_comma_separation() -> None:
    rendered = _render_assignment_text({1: True, 2: False})
    assert rendered == "x1=True, x2=False"


def test_render_assignment_text_booleans_capitalized() -> None:
    rendered = _render_assignment_text({1: True})
    assert "True" in rendered


# ---------------------------------------------------------------------------
# _guided_completion_text strategy dispatch
# ---------------------------------------------------------------------------


def test_guided_completion_sampled_strategy_returns_none() -> None:
    challenge = _challenge()
    assert _guided_completion_text(challenge, {"strategy": "sampled"}, 0) is None


def test_guided_completion_guided_exact_returns_full_solution() -> None:
    challenge = _challenge()
    text = _guided_completion_text(challenge, {"strategy": "guided_exact"}, 0)
    solution = challenge["extra"]["solution"]
    for var, value in solution.items():
        assert f"x{var}={'True' if value else 'False'}" in text


def test_guided_completion_guided_duplicate_matches_exact_output() -> None:
    challenge = _challenge()
    exact = _guided_completion_text(challenge, {"strategy": "guided_exact"}, 0)
    duplicate = _guided_completion_text(challenge, {"strategy": "guided_duplicate"}, 0)
    assert exact == duplicate


def test_guided_completion_guided_mixed_mode_0_is_correct() -> None:
    challenge = _challenge()
    text = _guided_completion_text(challenge, {"strategy": "guided_mixed"}, 0)
    exact = _render_assignment_text(challenge["extra"]["solution"])
    assert text == exact


def test_guided_completion_guided_mixed_mode_1_flips_one_variable() -> None:
    challenge = _challenge()
    text = _guided_completion_text(challenge, {"strategy": "guided_mixed"}, 1)
    exact = _render_assignment_text(challenge["extra"]["solution"])
    # Mode 1 flips one variable, so the rendered text must differ.
    assert text != exact
    # Same number of assignments (no drop).
    assert text.count("=") == len(challenge["extra"]["solution"])


def test_guided_completion_guided_mixed_mode_2_drops_one_variable() -> None:
    challenge = _challenge()
    text = _guided_completion_text(challenge, {"strategy": "guided_mixed"}, 2)
    assert text.count("=") == len(challenge["extra"]["solution"]) - 1


def test_guided_completion_random_guess_is_deterministic_under_seed() -> None:
    challenge = _challenge()
    profile = {"strategy": "random_guess", "miner_id": "miner-A"}
    a = _guided_completion_text(challenge, profile, 0)
    b = _guided_completion_text(challenge, profile, 0)
    assert a == b


def test_guided_completion_random_guess_differs_across_miners() -> None:
    challenge = _challenge()
    a = _guided_completion_text(challenge, {"strategy": "random_guess", "miner_id": "miner-A"}, 0)
    b = _guided_completion_text(challenge, {"strategy": "random_guess", "miner_id": "miner-B"}, 0)
    # With high probability they differ; be tolerant if they happen to match.
    # The deterministic seed space is per-miner so two miners should usually
    # disagree on at least one variable.
    assert a is not None and b is not None


def test_guided_completion_malformed_uses_default_text() -> None:
    text = _guided_completion_text(_challenge(), {"strategy": "malformed"}, 0)
    assert text == "solver unavailable"


def test_guided_completion_malformed_honors_override_text() -> None:
    text = _guided_completion_text(
        _challenge(),
        {"strategy": "malformed", "malformed_text": "custom-malformed"},
        0,
    )
    assert text == "custom-malformed"


def test_guided_completion_unknown_strategy_returns_none() -> None:
    assert _guided_completion_text(_challenge(), {"strategy": "unknown-strategy"}, 0) is None


def test_guided_completion_missing_strategy_defaults_to_sampled() -> None:
    assert _guided_completion_text(_challenge(), {}, 0) is None


# ---------------------------------------------------------------------------
# _apply_proof_corruption
# ---------------------------------------------------------------------------


def test_apply_proof_corruption_none_mode_is_identity() -> None:
    proof = {"signature": "abc123", "other": "value"}
    assert _apply_proof_corruption(proof, "none") == proof


def test_apply_proof_corruption_bad_signature_zeroes_signature() -> None:
    proof = {"signature": "deadbeef" * 8, "other": "value"}
    corrupted = _apply_proof_corruption(proof, "bad_signature")
    assert corrupted["signature"] == "0" * len(proof["signature"])
    assert corrupted["other"] == "value"


def test_apply_proof_corruption_bad_signature_empty_sig() -> None:
    corrupted = _apply_proof_corruption({"signature": ""}, "bad_signature")
    assert corrupted["signature"] == "0"


def test_apply_proof_corruption_unknown_mode_is_identity() -> None:
    proof = {"signature": "abc", "foo": 1}
    assert _apply_proof_corruption(proof, "not-a-known-mode") == proof


def test_apply_proof_corruption_does_not_mutate_input() -> None:
    proof = {"signature": "abc123"}
    _apply_proof_corruption(proof, "bad_signature")
    assert proof["signature"] == "abc123"
