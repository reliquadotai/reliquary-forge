#!/usr/bin/env bash
set -euo pipefail

: "${GITHUB_RUNNER_TOKEN:?GITHUB_RUNNER_TOKEN is required}"

RUNNER_ROOT="${RUNNER_ROOT:-/srv/reliquary/state/actions-runner}"
REPO_URL="${REPO_URL:-}"
RUNNER_NAME="${RUNNER_NAME:-reliquary-rtx6000b}"
RUNNER_LABELS="${RUNNER_LABELS:-self-hosted,gpu,rtx6000b,reliquary}"
RUNNER_VERSION="${RUNNER_VERSION:-}"

: "${REPO_URL:?REPO_URL is required}"

mkdir -p "${RUNNER_ROOT}"
cd "${RUNNER_ROOT}"

if [ ! -x "${RUNNER_ROOT}/config.sh" ]; then
  if [ -z "${RUNNER_VERSION}" ]; then
    RUNNER_VERSION="$(
      curl -fsSL https://api.github.com/repos/actions/runner/releases/latest \
      | sed -n 's/.*"tag_name": "v\([^"]*\)".*/\1/p' \
      | head -n 1
    )"
  fi
  curl -fsSL -o actions-runner.tar.gz \
    "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
  tar xzf actions-runner.tar.gz
  rm -f actions-runner.tar.gz
fi

export RUNNER_ALLOW_RUNASROOT=1
"${RUNNER_ROOT}/config.sh" \
  --unattended \
  --replace \
  --url "${REPO_URL}" \
  --token "${GITHUB_RUNNER_TOKEN}" \
  --name "${RUNNER_NAME}" \
  --labels "${RUNNER_LABELS}" \
  --work "${RUNNER_ROOT}/_work"

echo "Runner configured at ${RUNNER_ROOT}"
