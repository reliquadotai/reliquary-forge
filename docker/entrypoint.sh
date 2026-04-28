#!/usr/bin/env bash
# Entrypoint for the Reliquary Forge image.
#
# Reads RELIQUARY_FORGE_ROLE to dispatch into one of:
#   trainer       → reliquary run-rl-worker (the live closed-loop GRPO worker)
#   distill       → reliquary distill-cycle
#   miner         → reliquary mine
#   validator     → reliquary validate
#   showcase      → reliquary demo showcase
#   control-plane → reliquary serve-control-plane
#
# All other config flows through env vars (RELIQUARY_*, BT_*).
# See env.testnet.example for the full operator contract.
#
# Parallel-work credit: romain13190/reliquary@ed6da6c.
set -euo pipefail

ROLE="${RELIQUARY_FORGE_ROLE:-trainer}"

case "${ROLE}" in
  trainer)
    extra_args=()
    if [[ "${RELIQUARY_FORGE_ONCE:-0}" == "1" ]]; then
      extra_args+=(--once)
    fi
    echo "[entrypoint] launching reliquary run-rl-worker ${extra_args[*]:-}"
    exec reliquary run-rl-worker "${extra_args[@]}"
    ;;
  distill)
    : "${RELIQUARY_DISTILL_WINDOWS:=2}"
    echo "[entrypoint] launching reliquary distill-cycle --windows ${RELIQUARY_DISTILL_WINDOWS}"
    exec reliquary distill-cycle --windows "${RELIQUARY_DISTILL_WINDOWS}"
    ;;
  miner)
    : "${RELIQUARY_FORGE_WINDOW:=0}"
    echo "[entrypoint] launching reliquary mine --window ${RELIQUARY_FORGE_WINDOW}"
    exec reliquary mine --window "${RELIQUARY_FORGE_WINDOW}"
    ;;
  validator)
    : "${RELIQUARY_FORGE_WINDOW:=0}"
    echo "[entrypoint] launching reliquary validate --window ${RELIQUARY_FORGE_WINDOW}"
    exec reliquary validate --window "${RELIQUARY_FORGE_WINDOW}"
    ;;
  showcase)
    echo "[entrypoint] launching reliquary demo showcase"
    exec reliquary demo showcase
    ;;
  control-plane)
    echo "[entrypoint] launching reliquary serve-control-plane"
    exec reliquary serve-control-plane
    ;;
  *)
    echo "[entrypoint] unknown RELIQUARY_FORGE_ROLE=${ROLE}; expected trainer|distill|miner|validator|showcase|control-plane" >&2
    exit 64
    ;;
esac
