#!/usr/bin/env bash
# Multi-node DiLoCo launcher for the Reliquary Forge training lane.
#
# Invoke on each training host in the cluster:
#
#   RELIQUARY_DILOCO_RUN_ID=run-0001 \
#   RELIQUARY_DILOCO_WINDOW_ID=5 \
#   RELIQUARY_DILOCO_BACKEND_ROOT=/srv/reliquary-forge/checkpoints \
#   RELIQUARY_DILOCO_NNODES=2 \
#   RELIQUARY_DILOCO_NODE_RANK=0 \
#   RELIQUARY_DILOCO_NPROC_PER_NODE=1 \
#   RELIQUARY_DILOCO_MASTER_ADDR=staging1 \
#   RELIQUARY_DILOCO_MASTER_PORT=29500 \
#   RELIQUARY_DILOCO_PUBLISH=1 \
#   RELIQUARY_DILOCO_AUTHORITY_SECRET=<read from secrets store> \
#   RELIQUARY_DIST_BACKEND=nccl \
#   bash deploy/scripts/launch_diloco.sh
#
# Single-node gloo smoke (no GPU needed):
#
#   RELIQUARY_DILOCO_RUN_ID=smoke \
#   RELIQUARY_DILOCO_WINDOW_ID=1 \
#   RELIQUARY_DILOCO_BACKEND_ROOT=/tmp/reliquary-smoke \
#   RELIQUARY_DILOCO_NNODES=1 \
#   RELIQUARY_DILOCO_NODE_RANK=0 \
#   RELIQUARY_DILOCO_NPROC_PER_NODE=2 \
#   RELIQUARY_DILOCO_MASTER_ADDR=127.0.0.1 \
#   RELIQUARY_DILOCO_MASTER_PORT=29500 \
#   RELIQUARY_DIST_BACKEND=gloo \
#   bash deploy/scripts/launch_diloco.sh

set -euo pipefail

: "${RELIQUARY_DILOCO_RUN_ID:?required}"
: "${RELIQUARY_DILOCO_WINDOW_ID:?required}"
: "${RELIQUARY_DILOCO_BACKEND_ROOT:?required}"
: "${RELIQUARY_DILOCO_NNODES:?required}"
: "${RELIQUARY_DILOCO_NODE_RANK:?required}"
: "${RELIQUARY_DILOCO_NPROC_PER_NODE:?required}"
: "${RELIQUARY_DILOCO_MASTER_ADDR:?required}"
: "${RELIQUARY_DILOCO_MASTER_PORT:?required}"

# Optional; defaults baked in to LaunchConfig.from_env when absent.
: "${RELIQUARY_DILOCO_PUBLISH:=0}"
: "${RELIQUARY_DIST_BACKEND:=nccl}"
: "${RELIQUARY_DILOCO_INNER_STEPS:=30}"
: "${RELIQUARY_DILOCO_OUTER_LR:=0.4}"
: "${RELIQUARY_DILOCO_OUTER_MOMENTUM:=0.95}"

export RELIQUARY_DIST_BACKEND

# Operator may override venv / python via env.
RELIQUARY_PYTHON="${RELIQUARY_PYTHON:-python3}"
RELIQUARY_VENV="${RELIQUARY_VENV:-/opt/reliquary-venv}"
if [[ -z "${RELIQUARY_NO_VENV_ACTIVATE:-}" ]] && [[ -f "${RELIQUARY_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${RELIQUARY_VENV}/bin/activate"
  RELIQUARY_PYTHON="python"
fi

RDZV_ENDPOINT="${RELIQUARY_DILOCO_MASTER_ADDR}:${RELIQUARY_DILOCO_MASTER_PORT}"

exec "${RELIQUARY_PYTHON}" -m torch.distributed.run \
  --nnodes "${RELIQUARY_DILOCO_NNODES}" \
  --node_rank "${RELIQUARY_DILOCO_NODE_RANK}" \
  --nproc_per_node "${RELIQUARY_DILOCO_NPROC_PER_NODE}" \
  --rdzv_backend c10d \
  --rdzv_endpoint "${RDZV_ENDPOINT}" \
  -m reliquary.training.distributed_launcher
