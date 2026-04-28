# Calibration audit

Empirical sketch + logprob drift measurements that drive the constants
in `reliquary-ledger/reliquary_inference/protocol/constants.py`.

## Methodology

For each measurement:

1. Load two copies of the same model. **Honest baseline**: both copies
   identical. **Cheater test**: copy B has a small Gaussian
   perturbation (σ = 1e-4) added to one MLP weight, simulating a
   1-step-stale cheater.
2. Generate a completion with copy B at the production sampler
   settings (T=0.9, top_p=1.0).
3. Forward both copies on the completion at `LAYER_INDEX = -1`.
4. Compute per-position GRAIL sketch (via `SketchProofVerifier`) on
   both copies; record `|sketch_a - sketch_b|`.
5. Compute per-position log-probabilities on both copies; record
   `|log_a[next_token] - log_b[next_token]|`.

The sweep produces ~ N_PROMPTS × CHALLENGE_K = N_PROMPTS × 32
measurements per condition. Results are aggregated to p50/p95/p99/max
and used to set `PROOF_SKETCH_TOLERANCE_BASE` and
`LOGPROB_DRIFT_THRESHOLD` such that:

- Tolerance is well above the honest floor (any honest miner is
  accepted, even on hardware with non-zero drift).
- Tolerance is well below the cheater p95 (a typical 1-step cheater
  is detected at the rollout level via max-of-32-challenged-positions).

## Reports

| Date       | Host       | GPU                    | Rev applied | Report |
| ---------- | ---------- | ---------------------- | ----------- | ------ |
| 2026-04-28 | staging2   | NVIDIA RTX 6000B Blackwell | 9671f9a (ledger), a24d841 (protocol) | [staging2_rtx6000b_2026-04-28.json](staging2_rtx6000b_2026-04-28.json) |

## Re-running

```bash
ssh staging2rtx6000
cd /opt/reliquary-repos/reliquary
/opt/reliquary-venv/bin/python3 scripts/calibration_baseline.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --n-prompts 24 \
    --max-new-tokens 80 \
    --output /tmp/calibration.json
```

For new hardware classes (e.g. H100, A100, L40), run the same command
on a host with that GPU and commit the JSON output here.
