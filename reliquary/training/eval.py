from __future__ import annotations

from typing import Any

from ..envs.sat import evaluate_sat_response, generate_sat_challenge
from ..utils.modeling import decode_completion, load_model_bundle


def evaluate_policy(
    model_ref: str,
    device: str,
    sat_n: int,
    sat_k: int,
    sat_m: int,
    count: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    bundle = load_model_bundle(model_ref, device=device)
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]

    successes = 0.0
    dense_rewards = 0.0

    for offset in range(count):
        challenge = generate_sat_challenge(sat_n, sat_k, sat_m, seed=10_000 + offset)
        inputs = tokenizer(challenge["prompt"], return_tensors="pt")
        generated = model.generate(
            inputs["input_ids"].to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        full_ids = generated[0].detach().cpu().tolist()
        prompt_len = len(inputs["input_ids"][0].tolist())
        completion_ids = full_ids[prompt_len:]
        completion_text = decode_completion(tokenizer, completion_ids)
        evaluation = evaluate_sat_response(challenge, completion_text)
        successes += float(evaluation["score"])
        dense_rewards += float(evaluation["dense_reward"])

    total = float(max(count, 1))
    return {
        "eval_count": count,
        "success_rate": successes / total,
        "mean_dense_reward": dense_rewards / total,
    }
