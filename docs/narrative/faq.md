# Reliquary Forge — FAQ

## What is Reliquary Forge?

The trainer-quorum half of Reliquary, a two-subnet Bittensor platform.
Forge consumes verified rollout groups from
[Ledger](https://github.com/0xgrizz/reliquary-inference) (netuid 462) and
publishes policy deltas back through a cryptographically signed
closed-loop bridge.

## How is this different from Ledger?

- **Ledger** produces verifiable completions. Miners run inference; a
  4-validator mesh replays the forward pass and scores boxed-answer
  correctness.
- **Forge** produces policy deltas. It pulls in-zone rollout groups
  from Ledger's R2 artifacts, runs GRPO, and publishes the resulting
  weight diff back to Ledger miners via the closed-loop bridge.

Think of Ledger as the inference + labeling pipeline and Forge as the
RL update loop.

## What does "verified experience" mean?

Every reward that reaches Forge has been computed by at least one
validator running `evaluate_math_trace` on the actual completion text.
No miner can lie about its performance — the correctness field on
every verdict is the mesh's independent evaluation, not the miner's
claim.

## What's the training recipe?

DAPO/GRPO on zone-filtered rollout groups.

- **Advantage**: group-relative `aᵢ = (rᵢ − μ) / σ` over M=8 rollouts
  per prompt.
- **Loss**: PPO-clipped surrogate with ε = 0.2, plus a KL penalty
  against a frozen reference policy (β = 0.04, Schulman k3 estimator).
- **Scope**: only target tensors (attention q/k/v in the first 4
  layers) get a gradient; everything else frozen. Keeps each cycle's
  delta compact (~20 MB).
- **Optimizer**: AdamW at lr = 5e-7, gradient clip 1.0.

Full derivation in section 9 of the protocol paper at
[`reliquary-inference/docs/paper/reliquary_protocol_paper.md`](https://github.com/0xgrizz/reliquary-inference/blob/main/docs/paper/reliquary_protocol_paper.md).

## Why "zone filter"?

A rollout group with uniform rewards (σ = 0 — everyone got it right or
everyone got it wrong) carries no usable GRPO gradient. The zone filter
drops those groups at the validator before they reach Forge, so
training cycles don't waste compute on groups with no learning signal.

Threshold: σ ≥ 0.33 bootstrap, 0.43 steady state. For binary 0/1
rewards, 0.43 corresponds to Bernoulli k ∈ [2, 6] correct out of 8.

## What's the closed-loop bridge?

A signed on-chain commitment pattern:

1. Forge publishes `DeltaBundle` (shard bytes + Merkle root) to R2.
2. Forge publishes `CheckpointAttestation` (HMAC-signed training
   provenance) to R2.
3. Forge commits a `PolicyCommitment` on-chain pointing to the
   attestation key + `effective_at_ledger_window`.
4. Ledger's `policy_consumer` polls commitments, verifies the
   attestation signature via `BridgeVerifier`, runs the delta through
   the reparam-trick sanity guard, then hot-swaps the delta into the
   cached model at the effective window.

All three artifacts use the canonical serialization from
[reliquary-protocol](https://github.com/0xgrizz/reliquary-protocol)
so the two subnets cross-verify deterministically.

## What prevents a compromised Forge from poisoning Ledger?

Five layers:

1. **HMAC signature check.** `BridgeVerifier` rejects any commitment
   not signed by the policy-authority hotkey on the allowlist.
2. **Smoke hash.** `default_smoke_runner` recomputes
   `sha256(manifest || sorted shard digests)` and rejects on mismatch.
3. **Reparam-trick sanity guard.** Before any delta applies, every
   target tensor is checked for finiteness, minimum projection
   magnitude, and per-layer scale-ratio bounds. Catches the classic
   RMSNorm × Linear rescaling exploit.
4. **Partial-seal gate.** Deltas below a configurable mean-abs
   threshold (default 1e-7) aren't published.
5. **Mesh cross-check.** Every Ledger validator applies the same
   delta; a compromised one can't push bad weights unilaterally.

## What's the status?

Live closed-loop bridge running on testnet netuid 462 since
2026-04-22. First autonomous GRPO delta (`forge-grpo-1776864812`)
published with full provenance chain. Continuous-operation track
record accumulates from there toward mainnet.

See [announcement.md](announcement.md) for the launch post.

## How do I inspect what Forge is doing?

- **Forge cycle log**: `/opt/reliquary-state/logs/forge-training.log`
  on the trainer node. Every cycle writes a `=== timestamp —
  training cycle start ===` block with scan counts, group σ, training
  loss, and publish merkle.
- **Ledger dashboard**: `http://<validator>:9108/` on any Ledger
  validator. Shows the zone-filter gauges (in-zone rate, mean σ,
  mean reward) that feed Forge + the outcome of the last policy
  swap.
- **On-chain commitments**: `commitments/462/policy/` on R2 lists
  every `PolicyCommitment` published by Forge. Each entry has a
  pointer to the attestation + the delta manifest.

## Does Forge train pretraining checkpoints?

No. Forge is strictly post-training (RL + distillation). Pretraining
lives in dedicated subnet mechanisms elsewhere in the Bittensor
ecosystem.

## Does Forge require a specific GPU?

Blackwell (RTX 6000) is the reference deployment. H100 works. Any
modern CUDA GPU with ≥ 24 GB VRAM can run a Qwen2.5-3B GRPO cycle.
Distillation workloads scale to larger teachers (Qwen3-8B tested) on
an H100 and require a separate student node.

## How do I run a Forge trainer?

See [`docs/rtx6000b-deployment.md`](../rtx6000b-deployment.md) for the
full operator runbook. At a high level:

1. Install the reliquary venv + GPU extras.
2. Drop R2 credentials + signing secret into
   `/opt/reliquary-secrets/`.
3. Enable the systemd timer:
   `sudo systemctl enable --now reliquary-forge-trainer.timer`.
4. Watch `/opt/reliquary-state/logs/forge-training.log` for the
   first `=== training cycle start ===` block.
