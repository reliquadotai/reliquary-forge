#!/usr/bin/env bash
# Entry point for the continuous Forge training timer.
#
# Invoked by reliquary-forge-trainer.service (itself triggered every
# ~60 min by reliquary-forge-trainer.timer). Sources the forge env,
# runs one training cycle, and logs the outcome.
#
# Safe to re-invoke: the training script tracks last_trained_window in
# /opt/reliquary-state/forge-trainer-state.json and exits 0 without
# publishing a delta if not enough new rollout data has accumulated
# since the previous cycle (default: skip if <16 new pairs).

set -uo pipefail

# The reliquary Forge venv on staging2 has torch + transformers + peft
# + accelerate already installed from the first manual run.
VENV=/opt/reliquary-venv
STATE_ROOT=/opt/reliquary-state
LOG_FILE="$STATE_ROOT/logs/forge-training.log"
LOG_MAX_BYTES="${LOG_MAX_BYTES:-10485760}"  # 10 MB rotation threshold

mkdir -p "$STATE_ROOT/logs"

# Naive size-based rotation so the log doesn't grow unbounded across days
# of continuous operation. systemd's journal already captures the stdout
# for operational visibility — this file is the canonical per-run log.
if [[ -f "$LOG_FILE" ]] && [[ "$(stat -c %s "$LOG_FILE" 2>/dev/null || echo 0)" -gt "$LOG_MAX_BYTES" ]]; then
  mv "$LOG_FILE" "$LOG_FILE.1"
fi

TRAIN_SCRIPT="${FORGE_TRAIN_SCRIPT:-/opt/reliquary-repos/reliquary/scripts/run_forge_train_live.py}"
STATE_FILE="${FORGE_TRAINER_STATE:-$STATE_ROOT/forge-trainer-state.json}"
MIN_NEW_PAIRS="${FORGE_MIN_NEW_PAIRS:-16}"
# Partial-seal floor — single-group GRPO cycles produce updates around
# the 1e-9 mean-abs-delta range (one prompt × M=8 rollouts → one
# AdamW step → ~1e-7 weight movement after Adam scaling on a model
# this size is rare). The 1e-7 default was tuned for full B_BATCH=8
# group cycles and silently filters legitimate updates when the live
# task_count=1 path delivers fewer groups per cycle. Lower by default
# so smaller-cohort cycles still publish; numerical-noise-only updates
# (~1e-12) still get filtered.
MIN_MEAN_ABS_DELTA="${FORGE_MIN_MEAN_ABS_DELTA:-1e-9}"

{
  echo "=== $(date -u +%FT%TZ) — training cycle start ==="
  "$VENV/bin/python" "$TRAIN_SCRIPT" \
    --state-file "$STATE_FILE" \
    --min-new-pairs "$MIN_NEW_PAIRS" \
    --min-mean-abs-delta "$MIN_MEAN_ABS_DELTA"
  RC=$?
  echo "=== $(date -u +%FT%TZ) — exit=$RC ==="
  exit $RC
} >> "$LOG_FILE" 2>&1
