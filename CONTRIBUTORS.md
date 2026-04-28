# Contributors

Reliquary Forge is the work of the team maintaining the
[reliquadotai](https://github.com/reliquadotai) organization across three
companion repos:

- [reliquary-ledger](https://github.com/reliquadotai/reliquary-ledger) — inference + verification runtime
- [reliquary-forge](https://github.com/reliquadotai/reliquary-forge) — training runtime (this repo)
- [reliquary-protocol](https://github.com/reliquadotai/reliquary-protocol) — shared protocol package

## Maintainers

- [@0xgrizz](https://github.com/0xgrizz)
- [@0xgrizz](https://github.com/0xgrizz) (formerly active under another GitHub alias on early commits)

## Acknowledgments

### Upstream protocol heritage

The GRPO + PPO-clip + KL-penalty training loop derives from public
literature (DAPO, R1-Zero, DeepSeek GRPO). The hidden-state sketch proof
primitive comes from the upstream
[`grail`](https://github.com/grail-the-game/grail) project. The
distillation lane, permissionless env registry, FSDP2/DiLoCo distributed
training, and signed `PolicyCommitment + CheckpointAttestation` envelopes
are independent Reliquary additions.

### Parallel work — romain13190

[romain13190](https://github.com/romain13190) has been independently
iterating on a parallel SN81 implementation at
[romain13190/reliquary](https://github.com/romain13190/reliquary). Forge
adopts the following ideas from that repo with explicit per-commit
attribution:

- W&B richer training-config logging (lr, kl_beta, ppo_clip, etc.)
  (his [b65085b](https://github.com/romain13190/reliquary/commit/b65085b))
- Continuous HF Hub checkpoint publishing every 10 windows as a
  side-channel beside our delta-checkpoint flow
  (his [0b9795d](https://github.com/romain13190/reliquary/commit/0b9795d))
- Cheater-curve threshold calibration methodology — ported as
  `scripts/cheater_curve_threshold.py` and run on our own staging fleet
  (his [5a7b3eb](https://github.com/romain13190/reliquary/commit/5a7b3eb))
- Sketch-drift measurement tooling
  (his [013763e](https://github.com/romain13190/reliquary/commit/013763e))
- Staleness-drift curve tooling
  (his [12266da](https://github.com/romain13190/reliquary/commit/12266da))
- CUDA 12.8 + flash-attn-2 Docker image baseline
  (his [6ee8592](https://github.com/romain13190/reliquary/commit/6ee8592))

Where empirical thresholds appear in our codebase, the values are
calibrated independently on our cross-GPU staging fleet, not copied.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) (top-level reliquary-ledger repo
hosts the canonical contributor workflow).
