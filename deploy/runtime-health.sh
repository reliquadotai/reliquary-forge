#!/usr/bin/env bash
set -euo pipefail

CONTROL_URL="${RELIQUARY_CONTROL_URL:-http://127.0.0.1:8000}"
SERVICES=(
  reliquary-control-plane.service
  reliquary-distill-worker.service
  reliquary-rl-worker.service
  reliquary-vllm.service
)

echo "== reliquary status =="
reliquary status || true
echo

echo "== systemd =="
for service in "${SERVICES[@]}"; do
  if systemctl list-unit-files "${service}" >/dev/null 2>&1; then
    printf "%s\t%s\n" "${service}" "$(systemctl is-active "${service}" 2>/dev/null || true)"
  fi
done
echo

echo "== control plane =="
curl -fsS "${CONTROL_URL}/healthz" || true
echo
curl -fsS "${CONTROL_URL}/status" || true
echo

echo "== metrics sample =="
curl -fsS "${CONTROL_URL}/metrics" | rg '^reliquary_(latest_imported_source_window|latest_rl_queue_depth|rl_worker_|latest_checkpoint_approved)' || true
