#!/usr/bin/env bash
set -euo pipefail

MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge}"
APP_ENV="${APP_ENV:-reliquary-app}"
REPO_ROOT="${REPO_ROOT:-/srv/reliquary/current}"

cd "${REPO_ROOT}"

while true; do
  cmd=(
    "${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}"
    reliquary run-rl-worker
    --once
  )
  if [ -n "${RELIQUARY_RL_START_WINDOW:-}" ]; then
    cmd+=(--start-window "${RELIQUARY_RL_START_WINDOW}")
  fi
  if [ -n "${RELIQUARY_RL_SOURCE_WINDOW:-}" ]; then
    cmd+=(--source-window "${RELIQUARY_RL_SOURCE_WINDOW}")
  fi
  "${cmd[@]}"
  if [ "${RELIQUARY_WORKER_ONCE:-0}" = "1" ]; then
    exit 0
  fi
  sleep "${RELIQUARY_RL_CYCLE_INTERVAL_SECONDS:-900}"
done
