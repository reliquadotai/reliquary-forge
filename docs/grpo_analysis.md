# GRPO Analysis For Reliquary SAT Rollouts

## Goal

Show that Reliquary's verified rollout pipeline produces training targets that can drive genuine on-policy improvement rather than only post-hoc scoring.

## What The Trainer Actually Optimizes

Each accepted rollout contributes:

- the exact prompt,
- the exact generated completion,
- the exact completion token ids,
- the miner-time completion logprobs,
- a validator-confirmed binary `score`,
- a validator-confirmed dense `dense_reward`.

Rollouts are grouped by challenge. For each challenge:

1. The trainer computes group-relative advantages from `dense_reward`.
2. The current policy recomputes the completion logprob.
3. The trainer forms an on-policy ratio `exp(logp_new - logp_old)`.
4. The loss uses a clipped GRPO/PPO-style objective on the verified rollout.

This means the update is not based on self-reported miner reward. It is based on validator-confirmed reward attached to an authenticated rollout.

## Why Binary Reward Alone Is Not Enough

For SAT, a weak model often produces:

- malformed assignments,
- partial assignments,
- wrong but nearly-correct assignments,
- zero full solves.

If the reward were only binary exact solve / fail, most rollout groups would have all-zero reward and therefore near-zero advantage. The trainer would receive little signal.

That is why the environment exposes two values:

- `score`: `1.0` only when all clauses are satisfied,
- `dense_reward`: fraction of clauses satisfied.

This keeps the validator honest about the actual task while giving the trainer a smoother signal.

## Why Dense Clause Reward Is Legitimate Here

For SAT, clause satisfaction is aligned with the task:

- satisfying more clauses is directionally better than satisfying fewer,
- a perfect assignment is still the top reward,
- the signal is computed deterministically by the validator from the actual submitted assignment,
- miners cannot claim a dense reward that the validator did not recompute.

So the dense reward is not a separate heuristic model. It is a deterministic decomposition of the task objective.

## Why Verified Rollouts Matter

The proof layer prevents a miner from cheaply fabricating:

- a completion from the wrong model,
- a completion with altered token ids,
- a replay whose hidden-state sketch does not match the claimed sequence.

That matters because on-policy RL assumes the old logprobs belong to the same policy family that generated the sampled action. The proof system improves confidence that the rollout is genuinely policy-bound.

## Expected Learning Dynamics

The default local model `sshleifer/tiny-gpt2` is intentionally tiny. It is suitable for smoke tests, but it should not be expected to become a strong SAT solver quickly.

For a stronger instruct model, the expected learning path is:

1. Early training improves formatting consistency and variable coverage.
2. Dense reward increases as more clauses are satisfied.
3. Binary solve rate improves after the model reliably emits complete assignments.

This is exactly the kind of trajectory where dense verified reward is useful.

## Risks

1. If all group rewards are identical, advantages collapse.
2. If the base model cannot parse the task at all, improvement may be very slow.
3. If generation temperature is too low, rollout diversity collapses and GRPO has weak contrast.
4. If temperature is too high, proofs still pass but rollout quality may become too noisy.

## Practical Recommendations

1. Keep `rollouts_per_challenge >= 4` so advantages are comparative rather than isolated.
2. Use a stronger instruct checkpoint for real experiments.
3. Start with small SAT instances, then increase `(n, k, m)` as solve rate improves.
4. Track both `mean_dense_reward` and `success_rate`; one without the other is misleading.

## Bottom Line

Yes, the stack is structurally capable of real on-policy learning:

- rewards are validator-computed,
- rollouts are proof-bound,
- old logprobs are preserved,
- advantages are group-relative,
- the trainer performs a clipped on-policy update.

Whether the model actually becomes a good SAT solver depends on model capacity and curriculum, not on a missing signal path in the protocol.
