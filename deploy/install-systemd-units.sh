#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/srv/reliquary/current}"

install -m 0644 "${REPO_ROOT}/deploy/systemd/reliquary-vllm.service" /etc/systemd/system/reliquary-vllm.service
install -m 0644 "${REPO_ROOT}/deploy/systemd/reliquary-control-plane.service" /etc/systemd/system/reliquary-control-plane.service
install -m 0644 "${REPO_ROOT}/deploy/systemd/reliquary-distill-worker.service" /etc/systemd/system/reliquary-distill-worker.service
install -m 0644 "${REPO_ROOT}/deploy/systemd/reliquary-rl-worker.service" /etc/systemd/system/reliquary-rl-worker.service
install -m 0644 "${REPO_ROOT}/deploy/systemd/reliquary-github-runner.service" /etc/systemd/system/reliquary-github-runner.service
install -m 0644 "${REPO_ROOT}/deploy/systemd/reliquary-backup.service" /etc/systemd/system/reliquary-backup.service
install -m 0644 "${REPO_ROOT}/deploy/systemd/reliquary-backup.timer" /etc/systemd/system/reliquary-backup.timer

systemctl daemon-reload
systemctl enable reliquary-backup.timer

echo "Systemd units installed."
