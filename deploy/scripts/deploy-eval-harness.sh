#!/usr/bin/env bash
# Activate the Forge eval harness on a GPU host.
#
# Run AFTER:
#   - reliquary-forge has been pulled to /opt/reliquary-repos/reliquary
#     (or the configured FORGE_REPO_ROOT) at the desired SHA.
#   - reliquary-protocol >= 0.3.0 is installed in $VENV.
#   - reliquary-ledger is installed in $VENV (the eval harness soft-imports
#     `reliquary_inference.dataset.math_holdout` for the holdout loader).
#   - /opt/reliquary-secrets/{r2-cf-api-token,signing-secret} exist + are
#     readable by root.
#
# What this script does (idempotent — safe to re-run):
#   1. Installs the eval-cycle wrapper to /opt/reliquary-deploy/.
#   2. Installs the systemd service + timer units.
#   3. Reloads the systemd daemon.
#   4. Enables + starts the timer.
#   5. Optionally fires one cycle in --no-publish mode as a smoke test.
#
# Usage:
#   ./deploy-eval-harness.sh              # install + enable + start timer
#   ./deploy-eval-harness.sh --smoke      # also run one --no-publish cycle
#   ./deploy-eval-harness.sh --uninstall  # disable + remove units
#
# Pre-mainnet smoke pattern: run with --smoke, confirm the cycle exits 0,
# inspect /opt/reliquary-state/logs/forge-eval.log, then re-run without
# --smoke to enable the timer for continuous operation.

set -euo pipefail

REPO_ROOT="${FORGE_REPO_ROOT:-/opt/reliquary-repos/reliquary}"
DEPLOY_ROOT="${RELIQUARY_DEPLOY_ROOT:-/opt/reliquary-deploy}"
SYSTEMD_DIR="${SYSTEMD_DIR:-/etc/systemd/system}"
VENV="${RELIQUARY_FORGE_VENV:-/opt/reliquary-venv}"

cmd="${1:-install}"

# Pre-flight checks.
require_path() {
  if [[ ! -e "$1" ]]; then
    echo "ERROR: required path missing: $1" >&2
    exit 1
  fi
}

if [[ "$cmd" != "--uninstall" ]]; then
  require_path "$REPO_ROOT/scripts/run_forge_eval.py"
  require_path "$REPO_ROOT/deploy/scripts/run-forge-eval-cycle.sh"
  require_path "$REPO_ROOT/deploy/systemd/reliquary-forge-eval.service"
  require_path "$REPO_ROOT/deploy/systemd/reliquary-forge-eval.timer"
  require_path "$VENV/bin/python"
  require_path "/opt/reliquary-secrets/r2-cf-api-token"
  require_path "/opt/reliquary-secrets/signing-secret"

  # Sanity: check that the venv actually has reliquary-protocol >= 0.3.0
  # (the EvalBundle artifact landed in 0.3.0).
  if ! "$VENV/bin/python" -c \
      "import reliquary_protocol as p; v = tuple(int(x) for x in p.VERSION.split('.')[:2]); assert v >= (0, 3), p.VERSION" \
      2>/dev/null; then
    PROTO_VERSION=$("$VENV/bin/python" -c "import reliquary_protocol; print(reliquary_protocol.VERSION)" 2>/dev/null || echo "<missing>")
    echo "ERROR: reliquary-protocol >= 0.3.0 required for EvalBundle; got: $PROTO_VERSION" >&2
    echo "  fix: $VENV/bin/pip install --upgrade 'reliquary-protocol>=0.3.0'" >&2
    exit 1
  fi

  # Sanity: confirm reliquary-ledger (Python package: reliquary_inference)
  # is installed — the eval harness needs it for the MATH holdout.
  if ! "$VENV/bin/python" -c \
      "from reliquary_inference.dataset.math_holdout import load_holdout_problems" \
      2>/dev/null; then
    echo "ERROR: reliquary-ledger not installed (or its dataset module is older than the holdout addition)." >&2
    echo "  fix: cd to the reliquary-ledger checkout and: $VENV/bin/pip install -e ." >&2
    exit 1
  fi
fi

case "$cmd" in
  install|--smoke)
    echo "=== installing wrapper ==="
    install -m 0755 -D \
      "$REPO_ROOT/deploy/scripts/run-forge-eval-cycle.sh" \
      "$DEPLOY_ROOT/run-forge-eval-cycle.sh"

    echo "=== installing systemd units ==="
    install -m 0644 -D \
      "$REPO_ROOT/deploy/systemd/reliquary-forge-eval.service" \
      "$SYSTEMD_DIR/reliquary-forge-eval.service"
    install -m 0644 -D \
      "$REPO_ROOT/deploy/systemd/reliquary-forge-eval.timer" \
      "$SYSTEMD_DIR/reliquary-forge-eval.timer"

    systemctl daemon-reload

    if [[ "$cmd" == "--smoke" ]]; then
      echo "=== smoke run (--no-publish) ==="
      "$VENV/bin/python" "$REPO_ROOT/scripts/run_forge_eval.py" \
        --max-problems 4 \
        --no-publish
      echo "=== smoke OK ==="
    fi

    echo "=== enabling + starting timer ==="
    systemctl enable reliquary-forge-eval.timer
    systemctl start reliquary-forge-eval.timer

    echo "=== status ==="
    systemctl status reliquary-forge-eval.timer --no-pager | head -20 || true

    echo
    echo "Done. The first eval cycle will fire ~10 min after timer start"
    echo "(per OnBootSec), then every 120 min thereafter."
    echo
    echo "Watch:  journalctl -u reliquary-forge-eval.service -f"
    echo "       tail -f /opt/reliquary-state/logs/forge-eval.log"
    ;;

  --uninstall)
    echo "=== stopping + disabling timer ==="
    systemctl stop reliquary-forge-eval.timer 2>/dev/null || true
    systemctl disable reliquary-forge-eval.timer 2>/dev/null || true

    echo "=== removing systemd units ==="
    rm -f "$SYSTEMD_DIR/reliquary-forge-eval.service"
    rm -f "$SYSTEMD_DIR/reliquary-forge-eval.timer"
    rm -f "$DEPLOY_ROOT/run-forge-eval-cycle.sh"

    systemctl daemon-reload
    echo "=== uninstalled ==="
    ;;

  *)
    echo "Usage: $0 [install|--smoke|--uninstall]" >&2
    exit 2
    ;;
esac
