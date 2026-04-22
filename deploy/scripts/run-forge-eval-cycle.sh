#!/usr/bin/env bash
# Entry point for the continuous Forge eval timer.
#
# Invoked by reliquary-forge-eval.service (itself triggered every
# ~60 min by reliquary-forge-eval.timer). Sources the forge env,
# runs one eval cycle against the MATH holdout, and logs the outcome.
#
# Produces a signed EvalBundle at eval_bundles/<netuid>/<window>/<run>.json
# in R2 with per-problem correctness + per-level / per-subject accuracy
# + policy hash. Every run is reproducible: same policy + same holdout
# seed + greedy sampling produce the same number.

set -uo pipefail

VENV=/opt/reliquary-venv
STATE_ROOT=/opt/reliquary-state
LOG_FILE="$STATE_ROOT/logs/forge-eval.log"
LOG_MAX_BYTES="${LOG_MAX_BYTES:-10485760}"  # 10 MB rotation threshold

mkdir -p "$STATE_ROOT/logs"

if [[ -f "$LOG_FILE" ]] && [[ "$(stat -c %s "$LOG_FILE" 2>/dev/null || echo 0)" -gt "$LOG_MAX_BYTES" ]]; then
  mv "$LOG_FILE" "$LOG_FILE.1"
fi

EVAL_SCRIPT="${FORGE_EVAL_SCRIPT:-/opt/reliquary-repos/reliquary/scripts/run_forge_eval.py}"
MAX_PROBLEMS="${FORGE_EVAL_MAX_PROBLEMS:-500}"
MAX_NEW_TOKENS="${FORGE_EVAL_MAX_NEW_TOKENS:-1024}"
SAMPLING_MODE="${FORGE_EVAL_SAMPLING_MODE:-greedy}"
SUBTENSOR_NETWORK="${FORGE_SUBTENSOR_NETWORK:-test}"

{
  echo "=== $(date -u +%FT%TZ) — eval cycle start ==="
  "$VENV/bin/python" "$EVAL_SCRIPT" \
    --max-problems "$MAX_PROBLEMS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --sampling-mode "$SAMPLING_MODE" \
    --subtensor-network "$SUBTENSOR_NETWORK"
  RC=$?
  echo "=== $(date -u +%FT%TZ) — exit=$RC ==="
  exit $RC
} >> "$LOG_FILE" 2>&1
