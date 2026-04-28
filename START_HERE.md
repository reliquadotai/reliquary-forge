# Start here

Reliquary Forge is the **training arm** of Reliquary. It consumes verified
rollout groups from the Ledger runtime (sibling repo
[reliquary-ledger](https://github.com/reliquadotai/reliquary-ledger)), runs
PPO-clipped GRPO with KL penalty, and publishes signed delta checkpoints
back to the subnet via signed `PolicyCommitment` envelopes.

Pick your role:

| Role               | Hardware                                  | Quickstart                                              |
|--------------------|-------------------------------------------|---------------------------------------------------------|
| RL trainer (live)  | 1–2× GPU ≥ 40 GB (A100/H100/B200)         | [docs/trainer-quickstart.md](docs/trainer-quickstart.md) |
| Distillation lane  | 1× GPU ≥ 40 GB + a teacher model on vLLM  | [docs/distillation-quickstart.md](docs/distillation-quickstart.md) |
| Local showcase     | CPU is fine                                | `reliquary demo showcase`                              |

## 5-minute path (testnet 462, Docker)

```bash
git clone https://github.com/reliquadotai/reliquary-forge.git
cd reliquary-forge
cp env.testnet.example .env        # edit netuid, R2 creds, wallet path
export BT_WALLETS_DIR="$HOME/.bittensor/wallets"
docker compose up -d trainer
```

## How it fits together

Forge runs **alongside** a Ledger validator on the same netuid. It does
not need its own subnet — both runtimes converge on `netuid=462` in
testnet and (post-cutover) `netuid=81` on mainnet. See
[reliquary-ledger START_HERE.md](https://github.com/reliquadotai/reliquary-ledger/blob/main/START_HERE.md)
for the system-level picture.

## Repos

| Repo                                                                  | Role                                                  |
|-----------------------------------------------------------------------|-------------------------------------------------------|
| [reliquary-ledger](https://github.com/reliquadotai/reliquary-ledger)  | Inference + verification runtime                      |
| [reliquary-forge](https://github.com/reliquadotai/reliquary-forge)    | Training runtime (this repo)                          |
| [reliquary-protocol](https://github.com/reliquadotai/reliquary-protocol) | Shared protocol package                            |

## Next steps

- [docs/trainer-quickstart.md](docs/trainer-quickstart.md) — RL trainer end-to-end.
- [docs/distillation-quickstart.md](docs/distillation-quickstart.md) — distillation lane.
- [docs/architecture.md](docs/architecture.md) — closed-loop bridge spec.
- [CONTRIBUTORS.md](CONTRIBUTORS.md) — credit, attribution, parallel work.
