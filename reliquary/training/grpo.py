from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from ..utils.json_io import write_json
from ..utils.modeling import load_model_bundle, save_model_bundle


def build_grpo_groups(verdicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_challenge: dict[str, list[dict[str, Any]]] = {}
    for verdict in verdicts:
        if not verdict["payload"].get("training_eligible", verdict["payload"]["accepted"]):
            continue
        by_challenge.setdefault(verdict["payload"]["challenge_id"], []).append(verdict)

    groups: list[dict[str, Any]] = []
    for challenge_id, items in by_challenge.items():
        rewards = [float(item["payload"]["dense_reward"]) for item in items]
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((reward - mean_reward) ** 2 for reward in rewards) / max(len(rewards), 1)
        std = variance**0.5
        advantages = [
            (reward - mean_reward) / std if std > 1e-6 else reward - mean_reward for reward in rewards
        ]
        groups.append(
            {
                "challenge_id": challenge_id,
                "prompt": items[0]["payload"]["challenge"]["prompt"],
                "samples": [
                    {
                        "full_token_ids": item["payload"]["generation"]["sequence_token_ids"],
                        "prompt_token_ids": item["payload"]["generation"]["prompt_token_ids"],
                        "completion_token_ids": item["payload"]["generation"]["completion_token_ids"],
                        "completion_text": item["payload"]["generation"]["completion_text"],
                        "old_logprobs": item["payload"]["generation"]["completion_logprobs"],
                        "old_sum_logprob": item["payload"]["generation"]["sum_logprob"],
                        "reward": float(item["payload"]["dense_reward"]),
                        "binary_score": float(item["payload"]["score"]),
                        "advantage": float(advantages[idx]),
                    }
                    for idx, item in enumerate(items)
                ],
            }
        )
    return groups


def _sequence_logprob(model, full_token_ids: list[int], prompt_len: int, device: str) -> torch.Tensor:
    input_ids = torch.tensor(full_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    logits = model(input_ids).logits[0]
    log_probs = torch.log_softmax(logits[:-1], dim=-1)
    completion_targets = input_ids[0, prompt_len:]
    start = prompt_len - 1
    end = start + completion_targets.shape[0]
    selected = log_probs[start:end, :].gather(1, completion_targets.unsqueeze(1)).squeeze(1)
    return selected.sum()


def train_grpo_checkpoint(
    model_ref: str,
    groups: list[dict[str, Any]],
    output_dir: Path,
    device: str,
    lr: float,
    clip_eps: float,
    kl_coef: float,
    epochs: int,
) -> dict[str, Any]:
    train_bundle = load_model_bundle(model_ref, device=device, trainable=True)
    ref_bundle = load_model_bundle(model_ref, device=device, trainable=False)
    model = train_bundle["model"]
    tokenizer = train_bundle["tokenizer"]
    ref_model = ref_bundle["model"]
    ref_model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    step_losses: list[float] = []
    accepted_samples = sum(len(group["samples"]) for group in groups)

    for _ in range(max(epochs, 1)):
        for group in groups:
            for sample in group["samples"]:
                advantage = float(sample["advantage"])
                if abs(advantage) < 1e-8:
                    continue
                prompt_len = len(sample["prompt_token_ids"])
                old_sum = torch.tensor(sample["old_sum_logprob"], dtype=torch.float32, device=device)
                current_sum = _sequence_logprob(
                    model=model,
                    full_token_ids=sample["full_token_ids"],
                    prompt_len=prompt_len,
                    device=device,
                )
                with torch.no_grad():
                    ref_sum = _sequence_logprob(
                        model=ref_model,
                        full_token_ids=sample["full_token_ids"],
                        prompt_len=prompt_len,
                        device=device,
                    )

                ratio = torch.exp(current_sum - old_sum)
                unclipped = ratio * advantage
                clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
                policy_loss = -torch.min(unclipped, clipped)

                kl_term = torch.relu(current_sum - ref_sum)
                loss = policy_loss + kl_coef * kl_term

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step_losses.append(float(loss.detach().cpu()))

    save_model_bundle(model, tokenizer, output_dir)
    summary = {
        "accepted_samples": accepted_samples,
        "groups": len(groups),
        "mean_loss": sum(step_losses) / len(step_losses) if step_losses else 0.0,
        "steps": len(step_losses),
    }
    write_json(output_dir / "training_summary.json", summary)
    return summary
