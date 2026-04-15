#!/usr/bin/env bash
set -euo pipefail

SERVICE_USER="${SERVICE_USER:-reliquary}"
STATE_ROOT="${STATE_ROOT:-/srv/reliquary}"
MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge}"

export DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"

if ! id -u "${SERVICE_USER}" >/dev/null 2>&1; then
  useradd --system --create-home --shell /bin/bash "${SERVICE_USER}"
fi

apt-get update
apt-get install -y \
  build-essential \
  curl \
  git \
  jq

mkdir -p \
  "${STATE_ROOT}/current" \
  "${STATE_ROOT}/state" \
  "${STATE_ROOT}/state/artifacts" \
  "${STATE_ROOT}/state/exports" \
  "${STATE_ROOT}/state/metadata" \
  "${STATE_ROOT}/logs" \
  "${STATE_ROOT}/backups"

chown -R "${SERVICE_USER}:${SERVICE_USER}" "${STATE_ROOT}"

if [ ! -x "${MINIFORGE_ROOT}/bin/conda" ]; then
  case "$(uname -m)" in
    x86_64) installer="Miniforge3-Linux-x86_64.sh" ;;
    aarch64) installer="Miniforge3-Linux-aarch64.sh" ;;
    *)
      echo "Unsupported architecture: $(uname -m)" >&2
      exit 1
      ;;
  esac
  curl -fsSL -o /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/${installer}"
  bash /tmp/miniforge.sh -b -p "${MINIFORGE_ROOT}"
  rm -f /tmp/miniforge.sh
fi

"${MINIFORGE_ROOT}/bin/conda" config --system --set auto_activate_base false || true

echo "Bootstrap complete."
