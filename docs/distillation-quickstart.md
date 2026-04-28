# Distillation quickstart (testnet 462)

Run a distillation lane: a teacher model produces reasoning traces, miners
generate student completions conditioned on the trace, validators verify
provenance + non-contamination, and learners produce student checkpoints.
~15 min from clone to first cycle on a clean GPU box.

## Prerequisites

Same as the [trainer quickstart](trainer-quickstart.md), plus:

- A teacher model running on vLLM (e.g. `Qwen/Qwen3-8B`) with an HTTP endpoint.
- ≥ 24 GB VRAM dedicated to the student model.

## Docker path

```bash
git clone https://github.com/reliquadotai/reliquary-forge.git
cd reliquary-forge
cp env.testnet.example .env        # edit netuid, R2 creds, wallet
export BT_WALLETS_DIR="$HOME/.bittensor/wallets"

# Run with the distill profile (opt-in, uses RELIQUARY_FORGE_ROLE=distill)
docker compose --profile distill up -d distill
docker compose --profile distill logs -f distill
```

## Bare-metal path

```bash
source .venv/bin/activate
reliquary distill-cycle --windows 2
```

## What `.env` needs (additions on top of trainer)

```dotenv
# Teacher backend
RELIQUARY_TEACHER_BACKEND=local_reasoning_teacher
RELIQUARY_MODEL_TEACHER=Qwen/Qwen3-8B
RELIQUARY_VLLM_BASE_URL=http://127.0.0.1:8001
RELIQUARY_VLLM_MODEL=Qwen/Qwen3-8B
RELIQUARY_VLLM_GPU_MEMORY_UTILIZATION=0.65

# Student backend
RELIQUARY_DISTILL_LEARNER_BACKEND=local_reasoning_student
RELIQUARY_MODEL_STUDENT=Qwen/Qwen3-4B

# Distillation cadence + LoRA
RELIQUARY_DISTILL_CYCLE_INTERVAL_SECONDS=900
RELIQUARY_DISTILL_HOLDOUT_COUNT=8
RELIQUARY_DISTILL_TRAIN_TASK_COUNT=6
RELIQUARY_DISTILL_SAMPLES_PER_TASK=1
RELIQUARY_DISTILL_LORA_RANK=16
RELIQUARY_DISTILL_LORA_ALPHA=32
RELIQUARY_DISTILL_LORA_DROPOUT=0.05
```

## What the distillation lane does

Each cycle:

1. **Teacher trace**: teacher generates `RELIQUARY_DISTILL_TRAIN_TASK_COUNT` reasoning traces for fresh prompts (separate from RL prompts). Trace is signed + committed.
2. **Contamination screen**: Forge checks every trace against MMLU / HumanEval / GSM8K / HellaSwag using NFKM canonicalization + 10-gram Jaccard (threshold 0.05). Contaminated traces are dropped.
3. **Final-answer separation**: extract `Final Answer: <X>` via deterministic regex; the answer becomes the supervised target.
4. **Student SFT**: LoRA fine-tune the student on `(prompt, trace + final answer)` pairs with cross-entropy loss. Default 1 epoch, batch 2, grad-accum 8, lr 2e-5.
5. **Eval on holdout**: run the student on `RELIQUARY_DISTILL_HOLDOUT_COUNT` held-out tasks; record pass/fail rate.
6. **Publish checkpoint**: same delta-checkpoint format as the RL lane, separate prefix in R2.

## Verify

- `docker compose --profile distill logs distill` — look for "trace generated, screened, accepted: N/M".
- Trace + commit artifact appears at R2 under `distillation/<netuid>/<window>/`.
- Student checkpoint manifest under `checkpoints/<run_id>/<window>/` with the `distill` lane tag.

## Common issues

- **vLLM endpoint unreachable** — verify with `curl http://127.0.0.1:8001/v1/models`. If using Docker, the endpoint URL must be the host's network address (use `host.docker.internal` on macOS, `--network host` on Linux).
- **Contamination rate spike** — the screener will warn if > 30% of a cycle's traces are dropped; check the trace prompts for benchmark leakage.

## See also

- [docs/trainer-quickstart.md](trainer-quickstart.md) — RL lane.
- [docs/architecture.md](architecture.md) — closed-loop bridge spec.
- [docs/protocol.md](protocol.md) — distillation artifact schema.
- [START_HERE.md](../START_HERE.md) — picking your role.
