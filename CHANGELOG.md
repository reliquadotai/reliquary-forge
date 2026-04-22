# Changelog

All notable changes to **Reliquary Forge** (`reliquary`) are documented
here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
we version jointly with
[reliquary-protocol](https://github.com/0xgrizz/reliquary-protocol) so both
subnets bump in lockstep.

## [Unreleased]

### Added
- `scripts/run_forge_grpo_live.py` — autonomous GRPO trainer that
  consumes zone-filtered rollout groups from the Ledger closed-loop
  bridge and publishes policy deltas back. Runs via systemd timer
  (default cadence: every 60 min).
- Local cursor memoisation (`forge-grpo-cursor.json`) for R2 scans.
  Per-cycle R2 cost drops from ~900 gets to ~10 in steady state.
- Self-healing cursor: `used = True` only flipped after σ-filter
  survival, so a correctness-definition bug fix (e.g.,
  `correctness` source migration) doesn't brick the cursor.
- R2 state mirror at `forge/state/forge-trainer-state.json` with
  auto-restore on node rebuild. `last_trained_window` survives disk
  loss.
- Partial-seal gates: drop σ = 0 groups upfront + skip publishing
  deltas with `mean_abs_delta < min_mean_abs_delta` (default 1e-7).
- `verdict_bundles` join layer — pulls per-rollout correctness from
  the validator's verdict payloads (miner payloads don't carry
  correctness).
- Forge narrative docs: `docs/narrative/how_it_works.md`,
  `announcement.md`, `faq.md`.

### Changed
- Forge trainer now reads `completion_logprobs` (not `token_logprobs`)
  from miner payloads — matches the upstream field name. Old name
  supported as a fallback for older miner builds.
- Trainer service drop-in `grpo.conf` points `FORGE_TRAIN_SCRIPT` at
  `run_forge_grpo_live.py` and bumps `TimeoutStartSec` to 30 min for
  cold cursor rebuilds.

### Fixed
- Trainer-cycle correctness: used to read `correctness` from
  `completion_bundle` (where it doesn't exist — miners don't evaluate
  themselves). Now joins on `completion_id` to pull reward from
  `verdict_bundle`.
- Cursor poisoning: pre-fix cycles marked `used = True` on bundles
  whose rollouts all collapsed to σ = 0; fix is a schema migration
  that defers `used = True` until after σ-filter.

### Security
- Every published delta is HMAC-signed via the policy-authority
  hotkey; the Ledger `BridgeVerifier` rejects commitments not on the
  allowlist. Ledger's `PolicyConsumer` applies a reparam-trick sanity
  guard before touching cached models.

## [0.1.0] - 2026-04-18

Initial public release. Dual-lane platform: RL lane (GRPO on verified
rollouts) + distillation lane (teacher trace provenance, contamination
screening, packer, learner). See [`docs/architecture.md`](docs/architecture.md)
and [`docs/dual-lane-subnet-decision-memo.md`](docs/dual-lane-subnet-decision-memo.md).
