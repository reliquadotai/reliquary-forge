# Reasoning RL

Reliquary’s RL lane now treats finalized reasoning trajectories as the primary commodity for off-chain training.

## Data Path

1. `reliquary-ledger` finalizes a reasoning window.
2. The finalized `window_manifest`, `task_batch`, completion bundles, verdict bundle, and scorecard remain the source of truth.
3. `InferenceRegistryAdapter` reads those artifacts from either:
   - a same-host local registry root
   - an R2-backed registry
4. Imported reasoning trajectories materialize into Reliquary `rollout` and `verdict` artifacts with:
   - source window lineage
   - source artifact ids
   - deterministic reward features
   - a reusable `training_batch`

The import contract is artifact-level only. Reliquary does not depend on the inference runtime’s Python internals.

## Reward Contract

The default reasoning RL reward is deterministic and additive:

- correctness
- policy compliance
- format compliance
- acceptance
- difficulty scaling

These signals are computed from finalized reasoning verdict payloads and preserved in the imported RL artifacts. The reward contract is stored in the active policy and copied into training batches and checkpoints.

## Trainer Quorum

The trainer quorum is managed infrastructure, not decentralized learner execution. The current control-plane shape is:

- importer
- RL batch builder
- actor pool
- learner
- evaluator
- policy authority

Ray is the intended cluster control plane. Reliquary ships concrete manifests for:

- head node
- actor workers
- learner workers
- evaluator workers

See `/Users/malouk/GRAIL/reliquary/deploy/ray` for the baseline manifests.

## Dense and Frontier Backends

Two learner backends share the same `training_batch` contract:

- `trl_async_grpo_fsdp2`
  - default dense path
  - TRL/vLLM/FSDP2-oriented run spec
  - `local_compat` mode for smoke and single-node validation
  - `ray_cluster` mode for the intended multi-node trainer quorum
- `nemo_rl`
  - MoE-ready path
  - NeMo-oriented execution manifest
  - artifact-compatible with the dense path
  - kept in compatibility mode until the dense backend has repeated stable cycles

Both backends emit:

- checkpoint artifacts
- benchmark result artifacts
- backend run specs
- cluster metadata

## Promotion Gate

Checkpoint promotion is conservative:

- evaluator suite runs before and after training
- benchmark deltas become `benchmark_result` artifacts
- policy authority records a decision artifact
- only approved checkpoints become the next policy
- failed candidates publish a safe fallback policy that preserves the previous active model

## Operator Path

Use this sequence for an off-chain reasoning RL cycle:

```bash
reliquary import-inference-window --window 0 --source-window 6911280
reliquary run-rl-cycle --window 0 --source-window 6911280
```

For live nodes, configure:

- `RELIQUARY_INFERENCE_REGISTRY_*`
- `RELIQUARY_RL_*`
- `RELIQUARY_RAY_*`

in `/Users/malouk/GRAIL/reliquary/deploy/reliquary-node.env.example`.

For the dense trainer quorum, the sober operator sequence is:

1. point `RELIQUARY_INFERENCE_REGISTRY_*` at the finalized inference registry
2. set `RELIQUARY_RL_BACKEND_EXECUTION_MODE=ray_cluster`
3. configure `RELIQUARY_RAY_HEAD_ADDRESS` and the Ray replica counts
4. set `RELIQUARY_RL_VLLM_ACTOR_ENDPOINTS` to the serving endpoints used by the actor pool
5. run one import and one RL cycle locally
6. enable the continuous RL worker only after the single-cycle path is stable
