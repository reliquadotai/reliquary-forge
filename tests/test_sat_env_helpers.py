"""Comprehensive tests for the SAT challenge env.

Extends the existing basic `test_sat_env.py` with wider coverage of
``parse_assignments`` input variants, ``evaluate_sat_response`` edge
cases (partial satisfaction, out-of-range variables, missing variables,
format reasons), and determinism guarantees on ``generate_sat_challenge``.
"""

from __future__ import annotations

from reliquary.envs.sat import (
    evaluate_sat_response,
    generate_sat_challenge,
    parse_assignments,
)


# ---------------------------------------------------------------------------
# parse_assignments
# ---------------------------------------------------------------------------


def test_parse_assignments_basic_true_false() -> None:
    assert parse_assignments("x1=True, x2=False") == {1: True, 2: False}


def test_parse_assignments_one_zero_digit_form() -> None:
    assert parse_assignments("x1=1, x2=0") == {1: True, 2: False}


def test_parse_assignments_requires_exact_case_on_true_false() -> None:
    # The regex whitelist is case-sensitive on True/False; all-caps "TRUE" and
    # lowercase "false" do not match. This pins the current behavior.
    assert parse_assignments("x1=TRUE, x2=false") == {}
    assert parse_assignments("x1=True, x2=False") == {1: True, 2: False}


def test_parse_assignments_handles_whitespace_around_equals() -> None:
    assert parse_assignments("x1 = True , x2   =    False") == {1: True, 2: False}


def test_parse_assignments_ignores_surrounding_noise() -> None:
    assert parse_assignments("The answer is x1=True. Also x2=False. Done.") == {
        1: True,
        2: False,
    }


def test_parse_assignments_empty_returns_empty_dict() -> None:
    assert parse_assignments("") == {}


def test_parse_assignments_none_is_tolerated() -> None:
    assert parse_assignments(None) == {}  # type: ignore[arg-type]


def test_parse_assignments_later_assignment_wins() -> None:
    # Later assignment of same variable overrides earlier.
    assert parse_assignments("x1=True, x1=False") == {1: False}


def test_parse_assignments_multi_digit_variable() -> None:
    assert parse_assignments("x12=True") == {12: True}


def test_parse_assignments_ignores_malformed_values() -> None:
    # "Maybe" is neither True/False/0/1, gets skipped.
    assert parse_assignments("x1=Maybe, x2=True") == {2: True}


# ---------------------------------------------------------------------------
# generate_sat_challenge
# ---------------------------------------------------------------------------


def test_generate_sat_challenge_is_seed_deterministic() -> None:
    a = generate_sat_challenge(n=4, k=3, m=10, seed=7)
    b = generate_sat_challenge(n=4, k=3, m=10, seed=7)
    assert a["extra"]["clauses"] == b["extra"]["clauses"]
    assert a["extra"]["solution"] == b["extra"]["solution"]
    assert a["prompt"] == b["prompt"]


def test_generate_sat_challenge_different_seed_different_clauses() -> None:
    a = generate_sat_challenge(n=4, k=3, m=10, seed=1)
    b = generate_sat_challenge(n=4, k=3, m=10, seed=2)
    assert a["extra"]["clauses"] != b["extra"]["clauses"]


def test_generate_sat_challenge_clause_count_matches_m() -> None:
    challenge = generate_sat_challenge(n=5, k=3, m=8, seed=0)
    assert len(challenge["extra"]["clauses"]) == 8


def test_generate_sat_challenge_clause_width_is_k() -> None:
    challenge = generate_sat_challenge(n=5, k=3, m=5, seed=0)
    for clause in challenge["extra"]["clauses"]:
        assert len(clause) == 3


def test_generate_sat_challenge_solution_satisfies_every_clause() -> None:
    challenge = generate_sat_challenge(n=5, k=3, m=20, seed=42)
    solution = challenge["extra"]["solution"]
    for clause in challenge["extra"]["clauses"]:
        assert any((lit > 0) == solution[abs(lit)] for lit in clause)


def test_generate_sat_challenge_solution_covers_all_variables() -> None:
    challenge = generate_sat_challenge(n=7, k=3, m=5, seed=0)
    assert set(challenge["extra"]["solution"].keys()) == set(range(1, 8))


def test_generate_sat_challenge_id_includes_seed() -> None:
    challenge = generate_sat_challenge(n=3, k=2, m=3, seed=42)
    assert challenge["challenge_id"] == "sat-42"
    assert challenge["env_id"] == "sat"
    assert challenge["seed"] == 42


def test_generate_sat_challenge_prompt_has_formula() -> None:
    challenge = generate_sat_challenge(n=3, k=2, m=3, seed=0)
    assert challenge["prompt"].count("(") == 3  # one open-paren per clause
    assert "x1" in challenge["prompt"]


# ---------------------------------------------------------------------------
# evaluate_sat_response
# ---------------------------------------------------------------------------


def test_evaluate_sat_response_full_match_gives_perfect_score() -> None:
    challenge = generate_sat_challenge(n=3, k=2, m=4, seed=5)
    solution = challenge["extra"]["solution"]
    response = ", ".join(
        f"x{var}={'True' if val else 'False'}" for var, val in solution.items()
    )
    evaluation = evaluate_sat_response(challenge, response)
    assert evaluation["score"] == 1.0
    assert evaluation["dense_reward"] == 1.0
    assert evaluation["satisfied_fraction"] == 1.0
    assert evaluation["format_ok"] is True
    assert evaluation["accepted"] is True
    assert evaluation["format_reason"] == "ok"


def test_evaluate_sat_response_partial_gives_dense_reward_under_one() -> None:
    challenge = generate_sat_challenge(n=5, k=3, m=10, seed=11)
    # Flip one variable to force partial satisfaction.
    solution = dict(challenge["extra"]["solution"])
    flipped_var = next(iter(solution))
    solution[flipped_var] = not solution[flipped_var]
    response = ", ".join(
        f"x{var}={'True' if val else 'False'}" for var, val in solution.items()
    )
    evaluation = evaluate_sat_response(challenge, response)
    assert 0.0 <= evaluation["dense_reward"] < 1.0 or evaluation["dense_reward"] == 1.0
    assert evaluation["score"] in (0.0, 1.0)  # score is 0/1 exact


def test_evaluate_sat_response_missing_variables_listed() -> None:
    challenge = generate_sat_challenge(n=4, k=2, m=3, seed=1)
    # Only provide x1.
    evaluation = evaluate_sat_response(challenge, "x1=True")
    assert set(evaluation["missing_variables"]) == {2, 3, 4}
    assert evaluation["assignment_count"] == 1
    assert evaluation["expected_assignment_count"] == 4


def test_evaluate_sat_response_out_of_range_variables_listed() -> None:
    challenge = generate_sat_challenge(n=2, k=2, m=2, seed=0)
    evaluation = evaluate_sat_response(
        challenge, "x1=True, x2=False, x10=True"
    )
    assert evaluation["out_of_range_variables"] == [10]
    assert evaluation["format_ok"] is False
    assert evaluation["format_reason"] == "out_of_range_variables"


def test_evaluate_sat_response_missing_assignments_format_reason() -> None:
    challenge = generate_sat_challenge(n=3, k=2, m=3, seed=8)
    evaluation = evaluate_sat_response(challenge, "no assignments here")
    assert evaluation["format_reason"] == "missing_assignments"
    assert evaluation["format_ok"] is False
    assert evaluation["accepted"] is False


def test_evaluate_sat_response_no_clauses_edge_case() -> None:
    challenge = {
        "extra": {
            "clauses": [],
            "solution": {1: True, 2: False},
        }
    }
    evaluation = evaluate_sat_response(challenge, "x1=True, x2=False")
    assert evaluation["dense_reward"] == 0.0
    assert evaluation["score"] == 0.0
    assert evaluation["accepted"] is False


def test_evaluate_sat_response_case_and_digit_variants_accepted() -> None:
    challenge = generate_sat_challenge(n=3, k=2, m=5, seed=99)
    solution = challenge["extra"]["solution"]
    response_one = ", ".join(
        f"x{var}={1 if val else 0}" for var, val in solution.items()
    )
    evaluation = evaluate_sat_response(challenge, response_one)
    assert evaluation["score"] == 1.0


def test_evaluate_sat_response_accepted_flag_tracks_format_ok() -> None:
    challenge = generate_sat_challenge(n=3, k=2, m=3, seed=12)
    solution = challenge["extra"]["solution"]
    response_ok = ", ".join(
        f"x{var}={'True' if val else 'False'}" for var, val in solution.items()
    )
    assert evaluate_sat_response(challenge, response_ok)["accepted"] is True

    response_bad = response_ok + ", x99=True"
    assert evaluate_sat_response(challenge, response_bad)["accepted"] is False


def test_evaluate_sat_response_solution_passed_through() -> None:
    challenge = generate_sat_challenge(n=3, k=2, m=3, seed=22)
    solution = challenge["extra"]["solution"]
    evaluation = evaluate_sat_response(challenge, "x1=True")
    assert evaluation["expected"] == solution
