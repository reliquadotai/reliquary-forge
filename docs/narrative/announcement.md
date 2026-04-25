# Reliquary Forge — launch announcement

**2026-04-22.** Reliquary Forge, the trainer-quorum runtime that runs as a
companion protocol alongside Ledger on a single subnet, is live on the testnet
closed-loop bridge with its companion [Ledger](https://github.com/reliquadotai/reliquary-inference) (netuid 462).

## What Forge does

Forge takes verified rollout groups produced by Ledger's 4-validator
mesh and runs DAPO/GRPO post-training on them. The resulting policy
delta is published back to Ledger miners through an on-chain
`CheckpointAttestation + PolicyCommitment` pair, and Ledger's
`policy_consumer` hot-swaps the delta at the effective-at-window
boundary without service restart.

That's the flywheel:

```
Ledger mine → mesh verify → zone filter (σ ≥ 0.33) →
  Forge GRPO → delta publish → policy_consumer apply → Ledger mine with new policy
```

## What's live right now

- **Live GRPO cycle.** `scripts/run_forge_grpo_live.py` runs
  autonomously every 60 min via systemd timer.
- **First published GRPO delta.** `forge-grpo-1776864812` — merkle
  `87b30c85…`, 5 groups × 8 rollouts, targeting
  `effective_at_ledger_window=6962490`. Downstream `policy_consumer`
  verified signature + smoke hash + reparam-trick sanity guard and
  swapped the delta into the cached model.
- **Cursor-memoised R2 scan.** Per-cycle R2 cost dropped from ~900
  gets to ~10 in steady state (20 min cold → 1.5 min warm).
- **Partial-seal gates.** Groups with σ = 0 (all-wrong or all-right
  rollouts) are dropped before training; deltas below 1e-7 mean-abs
  aren't published. No busy-work on-chain.
- **R2 state mirror with auto-restore.** Node rebuild no longer resets
  `last_trained_window`.

## What's on the other side of the bridge

[Ledger](https://github.com/reliquadotai/reliquary-inference) has the inference
half: GRAIL sketch proofs, 9-stage verifier pipeline, DAPO zone filter,
mesh consensus. See its [launch announcement](https://github.com/reliquadotai/reliquary-inference/blob/main/docs/narrative/announcement.md)
for the Ledger side.

## Roadmap to mainnet

1. **Continuous-operation track record.** Forge needs to land signed
   deltas through the bridge for ≥ 4 weeks with no single-point
   failure. In progress.
2. **Distillation lane live.** Teacher catalog, trace artifacts,
   contamination screening, packer, learner promotion. Scaffolding
   exists; production wiring is the next major epic.
3. **Permissionless environment registry.** Authors can submit a
   sandboxed task-generator + reward-function spec; validators vote
   acceptance; stake-backed sponsors. Design lives in
   [`docs/lightweight-inference-subnet-blueprint.md`](../lightweight-inference-subnet-blueprint.md).
4. **Mainnet Forge at production cadence.** The dual-protocol design
   (Ledger + Forge on one netuid) is the canonical target; the
   production cadence will be announced when the 4-week track record is in.

## The policy on governance

No single entity. The policy-authority hotkey is HMAC-signed and
rotated via the subnet identity commit interface. Upgrades require a
coordinated `reliquary-protocol` version bump + 24h testnet bake +
on-chain commit of upgrade intent before cutover. Anti-centralization
is a protocol invariant enforced by the validator mesh, not a
marketing claim.

## Get involved

- **Run a Ledger miner or validator** on netuid 462: see the
  [miner](https://github.com/reliquadotai/reliquary-inference/blob/main/docs/miner-quickstart.md)
  and [validator](https://github.com/reliquadotai/reliquary-inference/blob/main/docs/validator-quickstart.md)
  quickstarts.
- **Inspect Forge cycles** via the live dashboard on any Ledger
  validator (`:9108/`) or the audit index at
  [`audit/index.html`](https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev/audit/index.html).
- **Read the protocol paper**:
  [`reliquary-inference/docs/paper/reliquary_protocol_paper.md`](https://github.com/reliquadotai/reliquary-inference/blob/main/docs/paper/reliquary_protocol_paper.md).
  Section 9 covers the DAPO zone filter, GRPO update rule, cooldown,
  and reparam-trick guard.
- **Contribute**: PRs, issues, bug reports welcome on either
  [`reliquary`](https://github.com/reliquadotai/reliquary) (this repo) or
  [`reliquary-inference`](https://github.com/reliquadotai/reliquary-inference).
