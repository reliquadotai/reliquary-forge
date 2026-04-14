# Showcase PoC

The canonical proof-of-concept path is:

```bash
cp env.example .env
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
reliquary demo showcase
```

This produces a deterministic local run that demonstrates:

1. trainer-published policy artifacts,
2. windowed challenge generation,
3. multiple miner archetypes submitting rollouts,
4. validator proof replay and verdict explanation,
5. scorecard and weight publication,
6. GRPO training over verified and uniqueness-eligible experience,
7. checkpoint and next-policy publication,
8. a static export for audit and sharing.

## Built-in Profiles

- `showcase`: full demo with five miner archetypes.
- `recording`: shorter deterministic profile for screen captures and quick walkthroughs.

## Export Bundle

Each run is written under:

```text
./data/runs/<run-id>/
  artifacts/
  chain-events.json
  run-manifest.json
  SUMMARY.md
  dashboard/
    dashboard-data.json
    index.html
```

Open `dashboard/index.html` in a browser after the run. The dashboard is static and can be shared or archived without rerunning the demo.

## Miner Archetypes

- `honest_high_quality`: uses exact SAT assignments and produces strong accepted rollouts.
- `honest_average`: mixes exact, noisy, and partial assignments.
- `low_diversity`: repeatedly submits the same valid completion and is penalized by uniqueness filtering.
- `weak_policy`: emits deterministic random guesses.
- `malformed_proof_failing`: emits malformed text and a corrupted proof to demonstrate validator rejection.

## Reading the Dashboard

The dashboard is intentionally read-only. It should answer:

- which policy was active for a given window,
- which miners participated,
- which rollouts were accepted or rejected and why,
- which rollouts remained training-eligible after uniqueness filtering,
- what scorecard or weights were published,
- whether the checkpoint improved dense reward or success rate.
