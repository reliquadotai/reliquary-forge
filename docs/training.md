# GRPO Training Pipeline

The trainer implements **Group Relative Policy Optimization (GRPO)** — a variant of PPO adapted for verified rollouts in a decentralized setting. The key insight: because rollouts carry proofs and rewards are validator-computed, the training data is trustworthy enough for on-policy updates.

For the analytical argument on why this drives real learning, see [grpo_analysis.md](grpo_analysis.md).

## Overview

Each training window:

1. Collect accepted verdicts from the validator.
2. Group rollouts by challenge and compute advantages.
3. Run a clipped on-policy update with KL penalty.
4. Evaluate the model before and after training.
5. Save the checkpoint and publish the next policy.

## Group Construction

`build_grpo_groups(verdicts)` transforms verdict artifacts into training groups.

### Algorithm

1. **Filter**: Keep only verdicts with `accepted = true`.
2. **Group by challenge**: Collect all accepted rollouts for the same `challenge_id`.
3. **Compute advantages** within each group:
   - `mean_reward = mean(dense_rewards in group)`
   - `std = standard_deviation(dense_rewards in group)`
   - If `std > 1e-6`: `advantage[i] = (reward[i] - mean_reward) / std`
   - If `std <= 1e-6`: `advantage[i] = reward[i] - mean_reward` (unnormalized fallback)

The per-group normalization is critical: it ensures advantages reflect relative quality within the same challenge, not absolute difficulty differences between challenges.

### Group Structure

Each group contains:

```json
{
  "challenge_id": "sat-42",
  "prompt": "Solve this SAT problem...",
  "samples": [
    {
      "full_token_ids": [101, 205, ...],
      "prompt_token_ids": [101, 205, ...],
      "completion_token_ids": [307, 408, ...],
      "completion_text": "x1=True, x2=False, ...",
      "old_logprobs": [-2.3, -1.1, ...],
      "old_sum_logprob": -12.7,
      "reward": 0.88,
      "binary_score": 0.0,
      "advantage": 1.42
    }
  ]
}
```

The `old_logprobs` and `old_sum_logprob` are the log-probabilities computed by the miner during generation. These serve as the "old policy" baseline for the importance sampling ratio.

## Training Loop

`train_grpo_checkpoint(model_ref, groups, output_dir, device, lr, clip_eps, kl_coef, epochs)`

### Setup

1. Load **two copies** of the model:
   - `model` (trainable): will be updated via gradient descent.
   - `ref_model` (frozen): stays at the window's starting weights for KL penalty computation.
2. Initialize `AdamW(model.parameters(), lr=lr)`.

### Per-Sample Update

For each epoch, for each group, for each sample:

1. **Skip near-zero advantages**: If `|advantage| < 1e-8`, skip this sample (no useful signal).

2. **Compute current log-probability**:
   ```
   current_sum = sum(log_softmax(model(full_ids).logits)[prompt_len:] at completion tokens)
   ```

3. **Compute reference log-probability** (no gradient):
   ```
   ref_sum = sum(log_softmax(ref_model(full_ids).logits)[prompt_len:] at completion tokens)
   ```

4. **Importance sampling ratio**:
   ```
   ratio = exp(current_sum - old_sum_logprob)
   ```

5. **Clipped objective** (PPO/GRPO style):
   ```
   unclipped = ratio * advantage
   clipped   = clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
   policy_loss = -min(unclipped, clipped)
   ```

6. **KL penalty** (one-sided):
   ```
   kl_term = max(current_sum - ref_sum, 0)
   ```
   This only penalizes when the updated policy assigns **more** probability than the reference. It allows the policy to decrease probability freely but resists large increases that might indicate overfitting.

7. **Total loss**:
   ```
   loss = policy_loss + kl_coef * kl_term
   ```

8. **Update**: `loss.backward()` then `optimizer.step()`.

### Output

After all epochs, the model and tokenizer are saved to `output_dir`. The function returns:

```json
{
  "accepted_samples": 12,
  "groups": 4,
  "mean_loss": 0.034,
  "steps": 10
}
```

## Sequence-Level Log-Probability

The helper `_sequence_logprob(model, full_token_ids, prompt_len, device)` computes the sum of per-token log-probabilities for the completion portion:

1. Forward pass: `logits = model(input_ids).logits[0]` — shape `[seq_len, vocab_size]`.
2. Apply `log_softmax` over vocabulary dimension on `logits[:-1]` (shifted by one for next-token prediction).
3. Gather log-probabilities at the actual completion token positions (`input_ids[prompt_len:]`).
4. Return the sum as a scalar tensor.

This is a **sequence-level** metric (total log-likelihood), not token-level. The importance sampling ratio operates on the full completion, not individual tokens.

## Evaluation

`evaluate_policy(model_ref, device, sat_n, sat_k, sat_m, count, max_new_tokens)` measures model quality:

1. Generate `count` fresh SAT challenges using fixed seeds `10_000` through `10_000 + count - 1`.
2. For each challenge, run **greedy decoding** (`do_sample=False`) — deterministic, no temperature.
3. Evaluate each completion via `evaluate_sat_response`.
4. Report:

```json
{
  "eval_count": 8,
  "success_rate": 0.25,
  "mean_dense_reward": 0.72
}
```

Evaluation runs **before and after** each training window to measure the delta. The eval seeds (10,000+) are distinct from training seeds (window * 10,000 + offset) to avoid contamination.

## Hyperparameters

| Parameter | Env Variable | Default | Effect |
|-----------|-------------|---------|--------|
| Learning rate | `RELIQUARY_TRAIN_LR` | `1e-5` | AdamW step size. Lower = more stable, higher = faster convergence |
| Epochs | `RELIQUARY_TRAIN_EPOCHS` | `1` | Passes over the rollout data per window |
| Clip epsilon | `RELIQUARY_CLIP_EPS` | `0.2` | Constrains the ratio to `[0.8, 1.2]`. Prevents catastrophic policy updates |
| KL coefficient | `RELIQUARY_KL_COEF` | `0.0` | Weight of the KL penalty. `0.0` disables it entirely |
| Rollouts per challenge | `RELIQUARY_ROLLOUTS_PER_CHALLENGE` | `4` | More rollouts = richer advantage contrast within groups |

### Tuning Guidance

- **`rollouts_per_challenge >= 4`**: Fewer than 4 samples per group makes advantage normalization noisy.
- **`kl_coef = 0.0`** (default): Appropriate when training windows are short (1 epoch). Increase to `0.01-0.1` if using multiple epochs or seeing reward hacking.
- **`clip_eps = 0.2`**: Standard PPO value. Decrease to `0.1` for more conservative updates.
- **`lr = 1e-5`**: Appropriate for fine-tuning. Increase for smaller models, decrease for larger ones.

## Connection to Verified Rollouts

The training loop's correctness depends on a subtle assumption: **the `old_sum_logprob` stored in the rollout artifact was computed by the same model that generated the completion**.

This is where the proof system becomes critical for training, not just scoring:

1. The miner generates a completion and computes `completion_logprobs` in the same forward pass.
2. The proof commits to the hidden states of that exact forward pass.
3. The validator verifies the proof, confirming the completion came from the policy model.
4. The trainer uses the verified `old_sum_logprob` as the importance sampling baseline.

Without proof verification, a miner could submit fabricated logprobs, causing the importance sampling ratio to be meaningless and training to diverge.

## Window Orchestration

The trainer orchestrates each window via `train_window`:

1. `build_grpo_groups(verdicts)` — organize training data.
2. `evaluate_policy(current_model)` — baseline measurement.
3. `train_grpo_checkpoint(...)` — run the update.
4. `evaluate_policy(checkpoint_model)` — post-training measurement.
5. Create a checkpoint artifact with `{training_summary, eval_before, eval_after}`.
6. `publish_policy(model_ref=checkpoint_path, window_id=next)` — advance the loop.

The checkpoint artifact preserves the full training provenance: which policy it derived from, what the training metrics were, and how the model improved.
