# Configuration Reference

Reliquary is configured entirely through environment variables, loaded at startup via `python-dotenv`. Copy `env.example` to `.env` and adjust values as needed.

## Environment Variables

### Core Identity

| Variable | Default | Description |
|----------|---------|-------------|
| `RELIQUARY_STATE_ROOT` | `./data` | Root directory for node-local state |
| `RELIQUARY_ARTIFACT_DIR` | `./data/artifacts` | Root directory for the artifact registry |
| `RELIQUARY_EXPORT_DIR` | `./data/exports` | Export directory for run bundles and dashboards |
| `RELIQUARY_LOG_DIR` | `./data/logs` | Service log directory for remote nodes |
| `RELIQUARY_DEVICE` | `cpu` | PyTorch device (`cpu`, `cuda`, `cuda:0`, `mps`) |
| `RELIQUARY_MODEL_NAME` | `sshleifer/tiny-gpt2` | HuggingFace model identifier or local checkpoint path |
| `RELIQUARY_MODEL_TEACHER` | value of `RELIQUARY_MODEL_NAME` | Teacher model identifier for the distillation lane |
| `RELIQUARY_MODEL_STUDENT` | `reasoning-student://bootstrap` | Student model identifier or local checkpoint path |
| `RELIQUARY_MINER_ID` | `local-miner` | Identity string stamped on rollout artifacts |
| `RELIQUARY_VALIDATOR_ID` | `local-validator` | Identity string stamped on verdict artifacts |
| `RELIQUARY_TRAINER_ID` | `local-trainer` | Identity string stamped on policy and checkpoint artifacts |
| `RELIQUARY_MINER_SECRET` | `local-miner-secret` | HMAC signing key for proof commitments |
| `RELIQUARY_NODE_LABEL` | `local-node` | Human-readable host label written into benchmark artifacts |
| `RELIQUARY_HOST_MODE` | `local` | Host posture (`local`, `durable-single-node`, etc.) |
| `RELIQUARY_SERVICE_VERSION` | `0.1.0` | Deployment version written into manifests |

> **Production note:** In a real subnet each miner has its own secret. The shared default is for local demo only.

### APIs and Serving

| Variable | Default | Description |
|----------|---------|-------------|
| `RELIQUARY_API_BIND` | `127.0.0.1` | Bind address for the FastAPI control plane |
| `RELIQUARY_API_PORT` | `8000` | Control plane port |
| `RELIQUARY_VLLM_BASE_URL` | `""` | Local or remote `vLLM` OpenAI-compatible endpoint |
| `RELIQUARY_VLLM_MODEL` | value of `RELIQUARY_MODEL_TEACHER` | Model served by `vLLM` |
| `RELIQUARY_VLLM_GPU_MEMORY_UTILIZATION` | `0.65` | Fraction of GPU memory reserved by `vLLM`; lower this on mixed train/serve nodes so the student learner can still fit |
| `RELIQUARY_VLLM_TEMPERATURE` | `0.2` | Distillation teacher sampling temperature for `vLLM` |
| `RELIQUARY_VLLM_TIMEOUT_SECONDS` | `120` | Request timeout for `vLLM` teacher calls |
| `RELIQUARY_TEACHER_BACKEND` | `local_reasoning_teacher` | Teacher backend (`local_reasoning_teacher`, `vllm`) |

### Generation

| Variable | Default | Description |
|----------|---------|-------------|
| `RELIQUARY_MAX_NEW_TOKENS` | `48` | Maximum completion length (protocol cap: 256) |
| `RELIQUARY_TEMPERATURE` | `1.0` | Sampling temperature |
| `RELIQUARY_TOP_P` | `0.95` | Nucleus sampling threshold |
| `RELIQUARY_TOP_K` | `50` | Top-k sampling cutoff |
| `RELIQUARY_REPETITION_PENALTY` | `1.0` | Repetition penalty multiplier |

### SAT Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `RELIQUARY_SAT_N` | `4` | Number of Boolean variables |
| `RELIQUARY_SAT_K` | `3` | Literals per clause (k-SAT) |
| `RELIQUARY_SAT_M` | `17` | Number of clauses |

The satisfiability phase transition for random k-SAT is near `m/n ~ 2^k * ln(2)`. The defaults (`m=17, n=4, k=3`) sit above this threshold, making instances challenging but always satisfiable by construction.

### Training

| Variable | Default | Description |
|----------|---------|-------------|
| `RELIQUARY_WINDOW_LENGTH` | `30` | Reserved for future window-duration logic |
| `RELIQUARY_ROLLOUTS_PER_CHALLENGE` | `4` | Completions generated per challenge per miner |
| `RELIQUARY_TRAIN_LR` | `1e-5` | AdamW learning rate |
| `RELIQUARY_TRAIN_EPOCHS` | `1` | Training epochs per window |
| `RELIQUARY_CLIP_EPS` | `0.2` | PPO/GRPO clipping epsilon |
| `RELIQUARY_KL_COEF` | `0.0` | KL divergence penalty coefficient (0 = disabled) |
| `RELIQUARY_RL_ENV_ID` | `reasoning` | Primary RL environment (`reasoning`, `sat`) |
| `RELIQUARY_RL_TASK_SOURCE` | `reasoning_tasks` | Imported task source for the RL lane |
| `RELIQUARY_RL_ACTOR_MODEL_REF` | value of `RELIQUARY_MODEL_NAME` | Actor model used by the trainer quorum |
| `RELIQUARY_RL_REFERENCE_MODEL_REF` | value of `RELIQUARY_MODEL_NAME` | Reference model for RL evaluation and KL |
| `RELIQUARY_RL_LEARNER_BACKEND` | `trl_async_grpo_fsdp2` | RL learner backend (`local_grpo`, `trl_async_grpo_fsdp2`, `nemo_rl`) |
| `RELIQUARY_RL_BACKEND_EXECUTION_MODE` | `simulate` | Dense backend execution mode (`simulate`, `local_compat`) |
| `RELIQUARY_RL_MODEL_FAMILY` | `qwen-dense` | Trainer family label recorded into artifacts |
| `RELIQUARY_RL_REASONING_EVAL_COUNT` | `8` | Holdout reasoning task count for RL evaluator runs |
| `RELIQUARY_RL_HOLDOUT_METRIC_MIN` | `0.0` | Minimum benchmark delta required for promotion |
| `RELIQUARY_RL_CONTRACT_SCORE_MIN` | `0.0` | Optional floor for contract-score gating |
| `RELIQUARY_RL_CYCLE_INTERVAL_SECONDS` | `900` | Poll interval for the autonomous RL worker |
| `RELIQUARY_RL_VLLM_ACTOR_ENDPOINTS` | `""` | Comma-separated vLLM actor endpoints used by the dense backend |
| `RELIQUARY_NEMO_EXECUTION_MODE` | `simulate` | NeMo backend execution mode |

### Inference Import and Trainer Quorum

| Variable | Default | Description |
|----------|---------|-------------|
| `RELIQUARY_INFERENCE_REGISTRY_BACKEND` | `local` | Source backend for finalized inference windows (`local`, `r2`) |
| `RELIQUARY_INFERENCE_REGISTRY_LOCAL_ROOT` | `/srv/reliquary-inference/state/artifacts` | Same-host inference artifact root |
| `RELIQUARY_INFERENCE_REGISTRY_BUCKET` | `""` | R2 bucket for finalized inference windows |
| `RELIQUARY_INFERENCE_REGISTRY_ENDPOINT_URL` | `""` | R2 endpoint URL for finalized inference windows |
| `RELIQUARY_INFERENCE_REGISTRY_ACCESS_KEY_ID` | `""` | Access key for inference registry reads |
| `RELIQUARY_INFERENCE_REGISTRY_SECRET_ACCESS_KEY` | `""` | Secret key for inference registry reads |
| `RELIQUARY_INFERENCE_REGISTRY_REGION` | `auto` | Region label for object-store client construction |
| `RELIQUARY_INFERENCE_IMPORT_CURSOR_NAME` | `reasoning_rl` | Cursor file name used by the autonomous RL worker |
| `RELIQUARY_RAY_CLUSTER_PROFILE` | `single-node` | Cluster profile label exported into artifacts and metrics |
| `RELIQUARY_RAY_HEAD_ADDRESS` | `local` | Ray head address for trainer-quorum workers |
| `RELIQUARY_RAY_ACTOR_REPLICAS` | `1` | Desired actor worker replica count |
| `RELIQUARY_RAY_LEARNER_REPLICAS` | `1` | Desired learner worker replica count |
| `RELIQUARY_RAY_EVALUATOR_REPLICAS` | `1` | Desired evaluator worker replica count |

### Distillation

| Variable | Default | Description |
|----------|---------|-------------|
| `RELIQUARY_DISTILL_LEARNER_BACKEND` | `local_reasoning_student` | Distillation learner backend (`local_reasoning_student`, `transformers_sft`) |
| `RELIQUARY_DISTILL_HOLDOUT_COUNT` | `8` | Holdout reasoning task count for benchmark artifacts |
| `RELIQUARY_DISTILL_TRAIN_TASK_COUNT` | `6` | Training task count per reasoning task batch |
| `RELIQUARY_DISTILL_SAMPLES_PER_TASK` | `1` | Number of trace samples per task and miner |
| `RELIQUARY_DISTILL_PROFILE` | `distill` | Default distillation miner profile |
| `RELIQUARY_DISTILL_GENERATION_MINER_LIMIT` | `0` | Active miner limit for distillation (`0` = full profile) |
| `RELIQUARY_DISTILL_SFT_EPOCHS` | `1` | SFT epochs for the production learner backend |
| `RELIQUARY_DISTILL_SFT_BATCH_SIZE` | `2` | Per-device SFT batch size |
| `RELIQUARY_DISTILL_SFT_GRAD_ACCUM` | `8` | Gradient accumulation steps |
| `RELIQUARY_DISTILL_SFT_LR` | `2e-5` | SFT learning rate |
| `RELIQUARY_DISTILL_SFT_MAX_SEQ_LENGTH` | `1024` | Max sequence length for SFT |
| `RELIQUARY_DISTILL_LORA_RANK` | `16` | LoRA rank |
| `RELIQUARY_DISTILL_LORA_ALPHA` | `32` | LoRA alpha |
| `RELIQUARY_DISTILL_LORA_DROPOUT` | `0.05` | LoRA dropout |
| `RELIQUARY_DISTILL_CYCLE_INTERVAL_SECONDS` | `900` | Poll interval for the remote distillation worker |

### Bittensor Chain

| Variable | Default | Description |
|----------|---------|-------------|
| `NETWORK` | `finney` | Bittensor network name (`finney`, `test`, `local`) |
| `NETUID` | `1` | Subnet UID on the Bittensor network |
| `WALLET_NAME` | `default` | Bittensor wallet name |
| `HOTKEY_NAME` | `default` | Bittensor hotkey name |

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

## Protocol Constants

These are hard-coded in `reliquary/protocol/constants.py` and are not configurable at runtime. They define the proof and scoring protocol.

| Constant | Value | Description |
|----------|-------|-------------|
| `PROTOCOL_VERSION` | `"v1"` | Protocol version tag |
| `POLICY_VERSION` | `"v1"` | Policy artifact version tag |
| `ENV_ID` | `"sat"` | Environment identifier |
| `PRIME_Q` | `2,147,483,647` | Mersenne prime (2^31 - 1) for modular arithmetic in sketch proofs |
| `CHALLENGE_K` | `16` | Number of sequence positions opened during proof verification |
| `PROOF_TOLERANCE` | `4` | Maximum allowed modular distance between committed and recomputed sketch values |
| `LAYER_INDEX` | `-1` | Hidden-state layer used for proof computation (last transformer layer) |
| `RNG_LABEL` | `{"sketch": b"sketch", "open": b"open"}` | PRF domain-separation labels |
| `WINDOW_LENGTH` | `30` | Default window length |
| `ROLLOUTS_PER_CHALLENGE` | `4` | Default rollouts per challenge |
| `MAX_NEW_TOKENS_PROTOCOL_CAP` | `256` | Hard upper bound on completion length |
| `SUPERLINEAR_EXPONENT` | `2.0` | Exponent for miner weight computation |

## Docker Deployment

The project ships with a `Dockerfile` and `docker-compose.yml` for containerized runs.

### Image

- Base: `python:3.12-slim`
- Build tool: `uv` (fast pip alternative)
- Default model: `sshleifer/tiny-gpt2` (set `RELIQUARY_MODEL_NAME` in `.env` for a real model)

### Services

| Service | Command | Purpose |
|---------|---------|---------|
| `miner` | `reliquary mine --window 0` | Generate rollouts for window 0 |
| `validator` | `reliquary validate --window 0` | Validate rollouts for window 0 |
| `trainer` | `reliquary train --window 0` | Train on verdicts for window 0 |
| `demo` | `reliquary demo showcase` | End-to-end showcase export |

All services share a `reliquary-data` Docker volume mounted at `/app/data` and read configuration from `.env`.

```bash
# Quick start with Docker
cp env.example .env
docker compose up demo
```

### Bittensor Integration

The `BittensorChainAdapter` in `reliquary/chain/adapters.py` calls `subtensor.set_weights()` to publish miner weights on-chain. It requires:

- A registered wallet (`WALLET_NAME` + `HOTKEY_NAME`)
- A valid `NETWORK` and `NETUID`
- The `bittensor` and `bittensor-wallet` packages (included in dependencies)

The showcase flow uses the `LocalChainAdapter` by default and records chain-like publication events into the exported run bundle. The `BittensorChainAdapter` remains a scaffold for real subnet deployment.

## Remote Node Notes

The first RTX6000B deployment intentionally uses host-level Python services rather than nested GPU Docker containers. See [rtx6000b-deployment.md](rtx6000b-deployment.md) for the Miniforge bootstrap, remote env template, systemd units, and self-hosted runner setup.
