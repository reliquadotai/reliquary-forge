from reliquary.protocol.proof import derive_rollout_randomness, indices_from_tokens


def test_indices_are_deterministic() -> None:
    tokens = [1, 2, 3, 4, 5, 6]
    randomness = derive_rollout_randomness("policy", "challenge", "miner", 0)
    indices_a = indices_from_tokens(tokens, randomness, len(tokens), 4)
    indices_b = indices_from_tokens(tokens, randomness, len(tokens), 4)
    assert indices_a == indices_b
