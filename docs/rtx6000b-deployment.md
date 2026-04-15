# RTX6000B Deployment

This runbook is the first production-oriented deployment target for Reliquary:

- host: `<staging-host>`
- shape: private, single-node, durable
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- runtime: host-level Python via Miniforge

## Host bootstrap

Run these steps as `root` on the node:

```bash
mkdir -p /srv/reliquary/current
cd /srv/reliquary/current
bash deploy/bootstrap-host.sh
cp deploy/reliquary-node.env.example /srv/reliquary/state/reliquary.env
chown reliquary:reliquary /srv/reliquary/state/reliquary.env
```

The bootstrap script installs the base host toolchain required by `vLLM` and Triton on this node, including `build-essential`, `curl`, `git`, and `jq`. It also creates and assigns ownership for the nested runtime paths under `/srv/reliquary/state`, including `artifacts`, `exports`, and `metadata`, so the non-root `reliquary` user can execute distillation runs without manual permission fixes.

Sync the repo into `/srv/reliquary/current`, then install the app and serving environments:

```bash
bash /srv/reliquary/current/deploy/install-environments.sh
bash /srv/reliquary/current/deploy/install-systemd-units.sh
```

## Services

The node uses four long-lived services:

- `reliquary-vllm.service`
- `reliquary-control-plane.service`
- `reliquary-distill-worker.service`
- `reliquary-github-runner.service`

On the reference RTX6000B stack, `reliquary-vllm.service` runs `vLLM` with `--enforce-eager` and a reduced `RELIQUARY_VLLM_GPU_MEMORY_UTILIZATION` budget. This avoids unstable compile / CUDA-graph startup paths on the Blackwell host while leaving enough VRAM for the local LoRA learner to run on the same node.

Enable and start the core application services after the env file is populated:

```bash
systemctl enable reliquary-vllm.service reliquary-control-plane.service
systemctl start reliquary-vllm.service reliquary-control-plane.service
```

Keep `reliquary-distill-worker.service` installed but operator-controlled. Start it only after the teacher footprint and learner path have been validated on the node, otherwise it will immediately begin autonomous distillation cycles:

```bash
systemctl enable reliquary-distill-worker.service
systemctl start reliquary-distill-worker.service
```

The control plane stays private. Reach it over SSH tunneling:

```bash
ssh -L 8000:127.0.0.1:8000 -L 8001:127.0.0.1:8001 <staging-user>@<staging-host>
```

## Self-hosted runner

Create a short-lived registration token from a machine with `gh` access:

```bash
gh api repos/<owner>/<repo>/actions/runners/registration-token -X POST --jq .token
```

Then on the node:

```bash
export GITHUB_RUNNER_TOKEN=...
bash /srv/reliquary/current/deploy/register-github-runner.sh
systemctl enable reliquary-github-runner.service
systemctl start reliquary-github-runner.service
```

## Smoke checks

```bash
bash /srv/reliquary/current/deploy/smoke-distill.sh
systemctl status reliquary-vllm.service
systemctl status reliquary-control-plane.service
systemctl status reliquary-distill-worker.service
```

The smoke script validates all three critical surfaces:

- `nvidia-smi`
- `torch.cuda.is_available()` in the app environment
- `vLLM` model listing on `127.0.0.1:8001`
- control plane health on `127.0.0.1:8000`

The first real run target is:

- `2` windows
- `256` train tasks per window
- `2` samples per task
- `3` active miners
- `128` holdout tasks
- `Qwen/Qwen3-8B` teacher
- `Qwen/Qwen3-4B` LoRA student
