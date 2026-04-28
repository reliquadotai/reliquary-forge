# Trainer quickstart (testnet 462)

Run a Reliquary Forge GRPO RL trainer that consumes Ledger verdicts from
R2 and publishes signed delta checkpoints back to the subnet. ~15 min
from clone to first cycle on a clean GPU box.

## Prerequisites

- 1× GPU ≥ 40 GB VRAM (A100/H100/B200; 24 GB works for Qwen3-4B but is tight).
- CUDA 12.4+ driver on the host.
- Python 3.12 (Docker path doesn't need this on the host).
- A Bittensor wallet whose hotkey is registered on `netuid=462` testnet.
  - Create: `btcli w new_coldkey`, then `btcli w new_hotkey`.
  - Register: `btcli s register --netuid 462 --network test`.
- Cloudflare R2 bucket + API token (read+write).
- (Optional) HF token — for the Phase 1.4 continuous-publish side-channel.

## Docker path (recommended)

```bash
git clone https://github.com/reliquadotai/reliquary-forge.git
cd reliquary-forge
cp env.testnet.example .env       # then edit, see "What .env needs" below
export BT_WALLETS_DIR="$HOME/.bittensor/wallets"
docker compose up -d trainer
docker compose logs -f trainer
```

## Bare-metal path

```bash
git clone https://github.com/reliquadotai/reliquary-forge.git
cd reliquary-forge
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[ml,gpu,wandb]"
cp env.testnet.example .env       # then edit
bash scripts/launch_trainer.sh
tail -f data/logs/trainer.log
```

## What `.env` needs

Minimum fields (rest have testnet 462 defaults):

```dotenv
# Chain
NETWORK=test
NETUID=462

# Wallet
BT_WALLET_PATH=$HOME/.bittensor/wallets
WALLET_NAME=reliquary-ledger
HOTKEY_NAME=trainer

# Model + GPU
RELIQUARY_DEVICE=cuda:0
RELIQUARY_MODEL_NAME=Qwen/Qwen3-4B-Instruct

# R2 (where Forge reads verdicts + writes deltas)
RELIQUARY_R2_BUCKET=reliquary-forge-testnet
RELIQUARY_R2_ENDPOINT_URL=https://YOUR_ACCOUNT.r2.cloudflarestorage.com
RELIQUARY_R2_ACCESS_KEY_ID=...
RELIQUARY_R2_SECRET_ACCESS_KEY=...

# (Optional) HF Hub side-channel — Phase 1.4
RELIQUARY_HF_REPO_ID=
RELIQUARY_HF_TOKEN=
RELIQUARY_HF_PUBLISH_INTERVAL_WINDOWS=10

# (Optional) W&B telemetry
WANDB_PROJECT=reliquary-forge-testnet
WANDB_API_KEY=
```

## What the trainer does

Each cycle (default 60 min via systemd timer, configurable):

1. **Pull verdicts**: scan R2 `verdicts/<netuid>/<window>/...` for new bundles.
2. **Filter to in-zone groups**: keep groups with σ ≥ 0.33 (bootstrap) or 0.43 (steady).
3. **GRPO step**: PPO-clipped surrogate (ε = 0.2) + KL penalty (β = 0.04) against frozen reference. AdamW lr = 5e-6 with linear warmup + cosine decay.
4. **Publish delta**: `compute_delta(snapshot, base) → Merkle-committed int8 shards → upload to R2`.
5. **Sign + commit**: sign `(checkpoint_n, merkle_root)` with trainer hotkey; emit `PolicyCommitment` artifact + on-chain `subtensor.commit`.
6. **(Phase 1.4 side-channel)** Every 10 windows, also push a full HF-format snapshot to `RELIQUARY_HF_REPO_ID` for ecosystem use. Manifest is HMAC-signed.

Miners running `reliquary-ledger` with `RELIQUARY_INFERENCE_POLICY_CONSUMER_ENABLED=true` automatically apply each new `PolicyCommitment` at the next window boundary.

## Verify

- `docker compose logs trainer` — look for "GRPO step complete, kl=…, loss=…, lr=…"
- `tail -f data/logs/trainer.log` (bare-metal)
- W&B run page (if `WANDB_API_KEY` set) — loss curve, lr schedule, KL.
- Public audit index — your trainer's published `PolicyCommitment` appears under the `forge/` prefix within 10 min.

## Common issues

- **`flash_attn` import error** — Docker image has it pinned; bare-metal needs the prebuilt wheel: `pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`.
- **R2 credentials rejected** — verify the API token has both `R2 Object Read` and `R2 Object Write` scoped to the bucket.
- **No verdicts to consume** — Forge sits idle if no Ledger validators have published recent verdicts. Check the public audit index.

## See also

- [docs/distillation-quickstart.md](distillation-quickstart.md) — distillation lane.
- [docs/architecture.md](architecture.md) — closed-loop bridge spec, plane separation.
- [docs/training.md](training.md) — GRPO loss + scheduler internals.
- [START_HERE.md](../START_HERE.md) — picking your role.
