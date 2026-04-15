#!/usr/bin/env bash
set -euo pipefail

MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge}"
APP_ENV="${APP_ENV:-reliquary-app}"
VLLM_ENV="${VLLM_ENV:-reliquary-vllm}"

nvidia-smi
"${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
"${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${VLLM_ENV}" python -c "import vllm; print(vllm.__version__)"
curl -fsSL http://127.0.0.1:8000/healthz

for _ in $(seq 1 24); do
  if curl -fsSL http://127.0.0.1:8001/v1/models; then
    exit 0
  fi
  sleep 10
done

echo "vLLM teacher endpoint did not become ready in time" >&2
exit 1
