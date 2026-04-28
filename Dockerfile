# syntax=docker/dockerfile:1.6
# Reliquary Forge — trainer / RL-worker / distillation image.
#
# CUDA 12.8 + torch 2.7 + flash-attn 2.8.3 (matches the Ledger image so
# proof bit-equivalence holds across closed-loop bridge boundaries).
#
# Build:
#   docker build -t ghcr.io/reliquadotai/reliquary-forge:latest .
#
# Run RL trainer (testnet 462):
#   docker run --rm --gpus all --env-file .env \
#     -v ~/.bittensor/wallets:/root/.bittensor/wallets:ro \
#     -e RELIQUARY_FORGE_ROLE=trainer \
#     ghcr.io/reliquadotai/reliquary-forge:latest
#
# Parallel-work credit: romain13190/reliquary@6ee8592 (CUDA + flash-attn
# multi-stage shape). Our role taxonomy (miner/validator/trainer/distill/
# demo) is independent.

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update -qq && apt-get install -y -qq \
        python3.12 python3.12-venv python3-pip \
        git build-essential wget curl ca-certificates jq \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv.
RUN python3.12 -m venv /opt/reliquary-venv
ENV PATH="/opt/reliquary-venv/bin:${PATH}"

# torch 2.7.0 + CUDA 12.8 (matches the Ledger image and the cross-GPU audit).
RUN pip install --upgrade pip wheel setuptools \
 && pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# flash-attn prebuilt wheel — same wheel as Ledger so proof determinism is
# preserved across the closed-loop bridge.
ARG FA_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
RUN wget -q "${FA_URL}" -P /tmp/ \
 && pip install /tmp/flash_attn-*.whl \
 && rm /tmp/flash_attn-*.whl

# Source + install (trainer extras: trl, peft, accelerate). The wandb
# extra adds telemetry; hf adds the HF Hub continuous-publish path
# (Phase 1.4).
WORKDIR /opt/reliquary-forge
COPY . /opt/reliquary-forge
RUN pip install -e ".[trainer,wandb]" || pip install -e .

# bittensor 9.x in Forge (not 10.x like Ledger) but the same scalecodec
# pin keeps both images compatible.
RUN pip uninstall -y cyscale 2>/dev/null || true \
 && pip install 'async-substrate-interface<2.0.0' || true \
 && pip install --force-reinstall --no-deps scalecodec==1.2.12 || true

# boto3 for R2 (delta-checkpoint storage + rollout bundle pulls).
RUN pip install boto3 huggingface_hub

# Runtime defaults — match Ledger so cross-runtime proofs verify.
ENV GRAIL_ATTN_IMPL=flash_attention_2
ENV RELIQUARY_REQUIRE_FLASH_ATTENTION=1

COPY docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=4 --start-period=120s \
  CMD curl -fsS http://127.0.0.1:8000/healthz || exit 0

ENTRYPOINT ["/opt/entrypoint.sh"]
