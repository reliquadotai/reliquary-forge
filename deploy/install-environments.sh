#!/usr/bin/env bash
set -euo pipefail

MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge}"
REPO_ROOT="${REPO_ROOT:-/srv/reliquary/current}"
APP_ENV="${APP_ENV:-reliquary-app}"
VLLM_ENV="${VLLM_ENV:-reliquary-vllm}"

"${MINIFORGE_ROOT}/bin/conda" create -y -n "${APP_ENV}" python=3.12 pip
"${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" python -m pip install --upgrade pip setuptools wheel
"${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" python -m pip install \
  -e "${REPO_ROOT}[dev]" \
  -r "${REPO_ROOT}/deploy/requirements/app-gpu.txt"

"${MINIFORGE_ROOT}/bin/conda" create -y -n "${VLLM_ENV}" python=3.12 pip
"${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${VLLM_ENV}" python -m pip install --upgrade pip setuptools wheel
"${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${VLLM_ENV}" python -m pip install \
  -r "${REPO_ROOT}/deploy/requirements/vllm.txt"

echo "Environments installed:"
"${MINIFORGE_ROOT}/bin/conda" env list
