#!/usr/bin/env bash
set -euo pipefail

MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge}"
APP_ENV="${APP_ENV:-reliquary-app}"
REPO_ROOT="${REPO_ROOT:-/srv/reliquary/current}"

cd "${REPO_ROOT}"

while true; do
  "${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" \
    reliquary distill-cycle \
      --windows "${RELIQUARY_REMOTE_WINDOWS:-2}" \
      --tasks "${RELIQUARY_DISTILL_TRAIN_TASK_COUNT:-256}"
  if [ "${RELIQUARY_WORKER_ONCE:-0}" = "1" ]; then
    exit 0
  fi
  sleep "${RELIQUARY_DISTILL_CYCLE_INTERVAL_SECONDS:-900}"
done
