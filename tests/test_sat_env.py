from reliquary.envs.sat import evaluate_sat_response, generate_sat_challenge


def test_sat_generation_and_eval() -> None:
    challenge = generate_sat_challenge(n=3, k=2, m=5, seed=123)
    expected = challenge["extra"]["solution"]
    response = ", ".join(f"x{var}={'True' if value else 'False'}" for var, value in expected.items())
    evaluation = evaluate_sat_response(challenge, response)
    assert evaluation["score"] == 1.0
    assert evaluation["dense_reward"] == 1.0
    assert evaluation["format_ok"] is True
    assert evaluation["accepted"] is True


def test_sat_eval_rejects_malformed_response() -> None:
    challenge = generate_sat_challenge(n=3, k=2, m=5, seed=321)
    evaluation = evaluate_sat_response(challenge, "solver unavailable")
    assert evaluation["format_ok"] is False
    assert evaluation["format_reason"] == "missing_assignments"
    assert evaluation["accepted"] is False
