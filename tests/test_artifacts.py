from reliquary.protocol.artifacts import finalize_policy_artifact, make_policy_artifact


def test_policy_artifact_has_stable_id() -> None:
    draft = make_policy_artifact(
        producer_id="trainer",
        window_id=0,
        model_ref="test-model",
        env_id="sat",
        env_config={"n": 4, "k": 3, "m": 17},
        generation_config={"max_new_tokens": 16, "temperature": 1.0, "top_p": 1.0, "top_k": 50, "repetition_penalty": 1.0},
        trainer_config={"lr": 1e-5, "epochs": 1, "clip_eps": 0.2, "kl_coef": 0.0},
    )
    policy = finalize_policy_artifact(draft)
    assert policy["artifact_id"] == policy["policy_id"]
    assert policy["artifact_type"] == "policy"
