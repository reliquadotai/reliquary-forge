#!/usr/bin/env bash
# Launch a Reliquary Forge GRPO trainer in the background.
#
# Quickstart (testnet 462):
#   cp env.testnet.example .env   # edit netuid, R2 creds, wallet
#   bash scripts/launch_trainer.sh
#
# Or with Docker:
#   docker compose up -d trainer
#
# This launches the live closed-loop GRPO RL worker (`reliquary
# run-rl-worker`) which:
#   - polls the Ledger R2 verdict registry every cycle
#   - filters in-zone rollout groups (σ ≥ 0.33 bootstrap)
#   - runs one PPO-clipped GRPO step against the frozen reference
#   - publishes a signed PolicyCommitment + delta checkpoint
#   - (Phase 1.4 follow-up) pushes a full HF snapshot every 10 windows
#
# Variables (from .env):
#   RELIQUARY_INSTALL_DIR — checkout root (default: pwd)
#   RELIQUARY_LOG_DIR — where logs go (default: ./data/logs)
#
# Parallel-work credit: romain13190/reliquary@5f8fbb3 (script shape).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${RELIQUARY_INSTALL_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ -f "${INSTALL_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "${INSTALL_DIR}/.env"
  set +a
fi

LOG_DIR="${RELIQUARY_LOG_DIR:-${INSTALL_DIR}/data/logs}"
LOG_FILE="${LOG_DIR}/trainer.log"
PID_FILE="${LOG_DIR}/trainer.pid"
mkdir -p "${LOG_DIR}"

: "${NETWORK:?NETWORK must be set (e.g. test, finney)}"
: "${NETUID:?NETUID must be set (e.g. 462)}"

cd "${INSTALL_DIR}"

find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

if [[ -f "${PID_FILE}" ]]; then
  prior_pid="$(cat "${PID_FILE}")"
  if kill -0 "${prior_pid}" 2>/dev/null; then
    echo "stopping prior trainer PID=${prior_pid}"
    kill "${prior_pid}" 2>/dev/null || true
    sleep 2
    kill -9 "${prior_pid}" 2>/dev/null || true
  fi
fi
pkill -9 -f 'reliquary run-rl-worker' 2>/dev/null || true
sleep 1
rm -f "${LOG_FILE}" "${PID_FILE}"

PYTHON="${INSTALL_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3)"
fi

echo "launching trainer: network=${NETWORK} netuid=${NETUID} log=${LOG_FILE}"
nohup "${PYTHON}" -m reliquary.cli run-rl-worker \
    > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "trainer PID=$(cat "${PID_FILE}") log=${LOG_FILE}"
