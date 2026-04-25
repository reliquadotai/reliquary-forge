# Reliquary Forge — How it works

Forge is the trainer-quorum runtime for Reliquary, a proof-carrying
Bittensor subnet that runs two companion protocols under a single netuid.
Where [Ledger](https://github.com/reliquadotai/reliquary-ledger)
produces **verifiable completions**, Forge turns them into **policy
checkpoints** that flow back to Ledger miners through a signed
closed-loop bridge.

## The flywheel in one picture

```
┌───────────────────────────────────────────────────────────────────────┐
│   Ledger (netuid 462)                                                 │
│   Miners run Qwen2.5-3B on Hendrycks MATH with M=8 sampled rollouts   │
│   per prompt. 4-validator mesh replays each forward pass, computes    │
│   GRAIL sketch commitments, evaluates boxed-answer correctness, and   │
│   publishes signed verdicts + scorecards with zone-filter stats.      │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                 rollout_bundle │ (task_batch + verdicts + scorecard)
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│   Forge (this repo)                                                   │
│                                                                       │
│   scripts/run_forge_grpo_live.py                                      │
│     1. Pull zone-filtered in-zone groups from R2 via a local cursor   │
│        (~10 R2 gets/cycle in steady state)                            │
│     2. Join verdicts → completions by completion_id to get per-       │
│        rollout correctness (this is THE signal)                       │
│     3. Compute group-relative advantages: aᵢ = (rᵢ − μ)/σ             │
│     4. Run GRPO: PPO-clipped surrogate + KL k3 vs frozen reference    │
│     5. Freeze everything except attention q/k/v in the first 4 layers │
│        so the delta stays compact (~20 MB)                            │
│     6. Publish DeltaBundle + CheckpointAttestation + PolicyCommitment │
│        at effective_at_ledger_window = current_window + 60 blocks     │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
             CheckpointAttestation│  (HMAC-signed by policy-authority)
             PolicyCommitment   │  (on-chain, netuid 462)
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│   Ledger miner + validators                                           │
│   policy_consumer polls PolicyCommitment every ~10 s, verifies        │
│   attestation signature + smoke hash + reparam-trick sanity guard,    │
│   then hot-swaps the delta into the cached model at the effective     │
│   window. Next mining cycle runs with the updated policy.             │
└───────────────────────────────────────────────────────────────────────┘
```

## What makes this different from most RL subnets

- **Signal is verified, not self-reported.** Every reward that reaches
  Forge has been computed by at least one mesh validator running
  `evaluate_math_trace` on the actual completion text. Miners can't
  lie about their performance.
- **Training is gated on within-group variance.** DAPO's zone filter
  drops groups with σ = 0 (all rollouts same reward — no gradient
  signal) at the validator, so Forge doesn't waste a cycle on them.
- **Deltas are tiny and composable.** Freezing all but a handful of
  attention projections keeps each cycle's delta in the low-tens-of-MB
  range. Policy swaps happen in seconds, not minutes.
- **Provenance is on-chain.** `CheckpointAttestation` records which
  Ledger rollout windows contributed to a specific checkpoint. Any
  third party can walk the chain-of-custody from a policy back to the
  exact mesh-signed rollouts that shaped it.

## Two training lanes

Forge's internal architecture is dual-lane:

| Lane | Input | Output | Status |
|---|---|---|---|
| **RL (live)** | Zone-filtered rollout groups from Ledger | GRPO delta via closed-loop bridge | Live on netuid 462 |
| **Distillation** | Teacher reasoning traces + miner student completions | Packed dataset + student checkpoint | Design + scaffolding in place |

The RL lane is what you see running on testnet today. The distillation
lane is the next major milestone — it turns every accepted Ledger
reasoning trace into a reusable training asset that external model
builders can consume.

## What's it good for

- Turning inference rollouts into policy updates in a cryptographically
  verifiable way.
- Training small models (≤ 8B) on domain-specific reasoning benchmarks
  where correctness is objectively checkable.
- Decentralized RL post-training without trusting any single trainer.

## What it's not (yet)

- A general pretraining competition. That mechanism class lives
  elsewhere; Forge is strictly post-training.
- A model hosting service. Miners + validators each run their own
  copy of the current policy; Forge just publishes the diff.
- A public-facing inference API. Policy deltas are open, the current
  model hash is committed on-chain, but running inference on it is the
  user's / operator's responsibility.
