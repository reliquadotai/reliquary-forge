from __future__ import annotations

import hashlib
import random
from typing import Any

import torch

from ..protocol.artifacts import make_artifact
from ..protocol.constants import MAX_NEW_TOKENS_PROTOCOL_CAP
from ..protocol.proof import build_proof, derive_rollout_randomness
from ..utils.modeling import compute_completion_logprobs, decode_completion, load_model_bundle


def _stable_int(*parts: object) -> int:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(material).hexdigest()[:16], 16)


def _sample_completion(bundle: dict[str, Any], prompt: str, generation_config: dict[str, Any], seed: int) -> dict[str, Any]:
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    device = bundle["device"]

    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_token_ids = inputs["input_ids"][0].tolist()
    input_ids = inputs["input_ids"].to(device)

    torch.manual_seed(seed)
    generated = model.generate(
        input_ids,
        max_new_tokens=min(
            int(generation_config["max_new_tokens"]),
            int(MAX_NEW_TOKENS_PROTOCOL_CAP),
        ),
        do_sample=True,
        temperature=float(generation_config["temperature"]),
        top_p=float(generation_config["top_p"]),
        top_k=int(generation_config["top_k"]),
        repetition_penalty=float(generation_config["repetition_penalty"]),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    full_token_ids = generated[0].detach().cpu().tolist()
    completion_token_ids = full_token_ids[len(prompt_token_ids) :]
    completion_text = decode_completion(tokenizer, completion_token_ids)
    completion_logprobs = compute_completion_logprobs(
        model=model,
        full_token_ids=full_token_ids,
        prompt_length=len(prompt_token_ids),
        device=device,
    )

    return {
        "prompt_token_ids": prompt_token_ids,
        "completion_token_ids": completion_token_ids,
        "sequence_token_ids": full_token_ids,
        "completion_text": completion_text,
        "completion_logprobs": completion_logprobs,
        "sum_logprob": float(sum(completion_logprobs)),
    }


def _forced_completion(
    bundle: dict[str, Any],
    prompt: str,
    completion_text: str,
) -> dict[str, Any]:
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]

    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_token_ids = tokenizer.encode(completion_text, add_special_tokens=False)
    full_token_ids = prompt_token_ids + completion_token_ids
    completion_logprobs = compute_completion_logprobs(
        model=model,
        full_token_ids=full_token_ids,
        prompt_length=len(prompt_token_ids),
        device=device,
    )

    return {
        "prompt_token_ids": prompt_token_ids,
        "completion_token_ids": completion_token_ids,
        "sequence_token_ids": full_token_ids,
        "completion_text": decode_completion(tokenizer, completion_token_ids),
        "completion_logprobs": completion_logprobs,
        "sum_logprob": float(sum(completion_logprobs)),
    }


def _render_assignment_text(assignments: dict[int, bool]) -> str:
    return ", ".join(f"x{var}={'True' if value else 'False'}" for var, value in sorted(assignments.items()))


def _guided_completion_text(
    challenge: dict[str, Any],
    miner_profile: dict[str, Any],
    sample_index: int,
) -> str | None:
    strategy = miner_profile.get("strategy", "sampled")
    solution = dict(challenge["extra"]["solution"])
    variables = sorted(solution)
    miner_id = miner_profile.get("miner_id", "local-miner")
    rng = random.Random(_stable_int(challenge["challenge_id"], miner_id, sample_index))

    if strategy == "sampled":
        return None
    if strategy == "guided_exact":
        return _render_assignment_text(solution)
    if strategy == "guided_duplicate":
        return _render_assignment_text(solution)
    if strategy == "guided_mixed":
        mode = sample_index % 3
        if mode == 0:
            return _render_assignment_text(solution)
        if mode == 1:
            flip_var = variables[sample_index % len(variables)]
            solution[flip_var] = not solution[flip_var]
            return _render_assignment_text(solution)
        drop_var = variables[sample_index % len(variables)]
        solution.pop(drop_var, None)
        return _render_assignment_text(solution)
    if strategy == "random_guess":
        guess = {var: rng.choice([True, False]) for var in variables}
        return _render_assignment_text(guess)
    if strategy == "malformed":
        return miner_profile.get("malformed_text", "solver unavailable")
    return None


def _apply_proof_corruption(proof: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "bad_signature":
        corrupted = dict(proof)
        corrupted["signature"] = ("0" * len(proof["signature"])) if proof["signature"] else "0"
        return corrupted
    return proof


def mine_window(
    cfg: dict[str, Any],
    policy_artifact: dict[str, Any],
    challenge_artifacts: list[dict[str, Any]],
    window_id: int,
    miner_profile: dict[str, Any] | None = None,
    rollouts_per_challenge: int | None = None,
) -> list[dict[str, Any]]:
    generation_config = policy_artifact["payload"]["generation_config"]
    bundle = load_model_bundle(policy_artifact["payload"]["model_ref"], cfg["device"])
    rollouts: list[dict[str, Any]] = []
    active_profile = dict(miner_profile or {})
    miner_id = active_profile.get("miner_id", cfg["miner_id"])
    miner_secret = active_profile.get("miner_secret", cfg["miner_secret"])
    sample_count = int(rollouts_per_challenge or cfg["rollouts_per_challenge"])
    proof_corruption = active_profile.get("proof_corruption", "none")
    archetype = active_profile.get("archetype", "default")

    for challenge_artifact in challenge_artifacts:
        challenge = challenge_artifact["payload"]["challenge"]
        for sample_index in range(sample_count):
            generation_seed = challenge["seed"] * 10_000 + sample_index
            forced_completion = _guided_completion_text(challenge, active_profile, sample_index)
            if forced_completion is None:
                generation = _sample_completion(bundle, challenge["prompt"], generation_config, generation_seed)
            else:
                generation = _forced_completion(bundle, challenge["prompt"], forced_completion)
            randomness = derive_rollout_randomness(
                policy_artifact["artifact_id"],
                challenge["challenge_id"],
                miner_id,
                sample_index,
            )
            proof = build_proof(
                model=bundle["model"],
                full_token_ids=generation["sequence_token_ids"],
                miner_secret=miner_secret,
                randomness_hex=randomness,
            )
            proof = _apply_proof_corruption(proof, proof_corruption)
            payload = {
                "challenge_id": challenge["challenge_id"],
                "challenge": challenge,
                "sample_index": sample_index,
                "generation": generation,
                "proof": proof,
                "rollout_archetype": archetype,
                "miner_profile": {
                    "miner_id": miner_id,
                    "archetype": archetype,
                    "strategy": active_profile.get("strategy", "sampled"),
                    "proof_corruption": proof_corruption,
                },
            }
            artifact = make_artifact(
                artifact_type="rollout",
                producer_id=miner_id,
                policy_id=policy_artifact["artifact_id"],
                window_id=window_id,
                payload=payload,
                signature_bundle={"scheme": "local-hmac", "miner_id": miner_id},
            )
            rollouts.append(artifact)

    return rollouts
