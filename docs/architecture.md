# System Architecture

Reliquary is a verified decentralized on-policy RL system. Three roles — Trainer, Miner, and Validator — exchange content-addressed artifacts in a windowed loop. Each window produces a training step that advances the policy.

## Roles

### Trainer

The Trainer is the loop driver. It:

1. **Publishes the policy** — creates a policy artifact specifying the model, environment config, generation parameters, and training hyperparameters. The policy ID is deterministic: `SHA-256` of the canonical policy content.

2. **Generates challenges** — creates SAT challenge artifacts with deterministic seeds (`window_id * 10_000 + offset`). Each challenge contains a guaranteed-satisfiable k-SAT formula.

3. **Collects verdicts** — reads validator-produced verdict artifacts from the registry.

4. **Trains** — builds GRPO groups from training-eligible verdicts, emits a training-batch artifact, runs a clipped on-policy update with before/after evaluation, saves the checkpoint.

5. **Advances** — publishes the next policy pointing to the new checkpoint, starting the next window.

**Source**: `reliquary/roles/trainer.py`

### Miner

The Miner generates completions (rollouts) for challenges. For each challenge:

1. Load the model specified in the policy artifact.
2. Generate `rollouts_per_challenge` samples using the policy's generation config.
3. Compute per-token log-probabilities for each completion (reused during training).
4. Build a sketch proof binding the completion to the model's hidden states.
5. Package everything into a rollout artifact with HMAC signature.

The miner's generation seed is deterministic: `challenge_seed * 10_000 + sample_index`. This makes rollouts reproducible from the artifact data.

**Source**: `reliquary/roles/miner.py`

### Validator

The Validator verifies rollouts and produces verdicts. For each rollout:

1. Verify the sketch proof (HMAC signature, index consistency, hidden-state commitment).
2. Evaluate the SAT response (parse assignments, count satisfied clauses).
3. Determine acceptance: `accepted = proof_valid AND evaluation_accepted`.
4. Compute a uniqueness key (hash of completion tokens) to detect duplicates.

After processing all rollouts, the validator:

5. Marks uniqueness across all verdicts for the same challenge.
6. Computes miner weights using superlinear scoring and emits a scorecard artifact.

**Source**: `reliquary/roles/validator.py`

## Window Model

Time is divided into discrete **windows**. Each window is one complete cycle:

```
Window N:
  Policy(N) → Challenges → Rollouts → Verdicts → Training → Checkpoint → Policy(N+1)
```

Window IDs are integers starting at 0. Artifacts carry their `window_id`, enabling the registry to filter by window. The trainer increments the window ID when publishing the next policy.

## Data Flow

```
                    ┌─────────────┐
                    │   Trainer   │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            │
        ┌──────────┐ ┌──────────┐      │
        │  Policy  │ │Challenges│      │
        └─────┬────┘ └────┬─────┘      │
              │            │            │
              ▼            ▼            │
        ┌─────────────────────┐        │
        │       Miner         │        │
        │  load model         │        │
        │  generate samples   │        │
        │  compute logprobs   │        │
        │  build proofs       │        │
        └──────────┬──────────┘        │
                   │                   │
                   ▼                   │
             ┌──────────┐             │
             │ Rollouts  │             │
             └─────┬─────┘             │
                   │                   │
                   ▼                   │
        ┌─────────────────────┐        │
        │     Validator       │        │
        │  verify proofs      │        │
        │  evaluate SAT       │        │
        │  mark uniqueness    │        │
        │  compute weights    │        │
        └──────────┬──────────┘        │
                   │                   │
                   ▼                   │
             ┌──────────┐             │
             │ Verdicts  │             │
             └─────┬─────┘             │
                   │                   │
                   └───────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  GRPO Train │
                    │  eval before│
                    │  update     │
                    │  eval after │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Checkpoint  │
                    │ Next Policy │
                    └─────────────┘
```

## Artifact Registry

All artifacts are persisted by the `LocalArtifactRegistry`, a file-based store under `RELIQUARY_DATA_DIR`:

```
data/
  policies/          # Policy artifacts (one per window)
  challenges/        # SAT challenge artifacts
  rollouts/          # Miner-produced rollouts with proofs
  verdicts/          # Validator-produced verdicts with explanations
  training_batches/  # Trainer-selected verified experience
  scorecards/        # Per-window miner weights and summaries
  checkpoints/       # Saved model + tokenizer per training step
  window_manifests/  # Window-level lineage and metrics
  run_manifests/     # Run-level export metadata
  runs/              # Copied artifacts, dashboard, summaries
  training/          # GRPO groups, window summaries
```

Each artifact is stored as `{artifact_type}s/{artifact_id}.json`. Artifacts are content-addressed: the filename is the SHA-256 of the artifact's contents.

The registry supports:
- `put_artifact(artifact)` — write to disk.
- `list_artifacts(type, window_id=None)` — retrieve by type, optionally filtered.
- `latest_policy()` — get the most recent policy by `(window_id, created_at)`.

See [artifacts.md](artifacts.md) for the full schema reference.

## Chain Adapters

The chain layer abstracts publication events:

| Adapter | Backend | Use Case |
|---------|---------|----------|
| `LocalChainAdapter` | In-memory event log | Local testing, showcase exports |
| `BittensorChainAdapter` | `subtensor.set_weights()` plus reference publishing | Real subnet deployment |

The local adapter records policy publication, window metadata, miner result references, scorecards, and checkpoint handoffs into the exported run bundle. The Bittensor adapter mirrors those seams so later integration is wiring rather than redesign.

See [configuration.md](configuration.md) for Bittensor setup.

## Configuration

All configuration flows through environment variables loaded by `load_runtime_config()`. There is no YAML, no config files, no Pydantic models. This keeps the system inspectable and Docker-friendly.

26 env vars cover identity, generation, SAT parameters, training hyperparameters, chain config, and logging. See [configuration.md](configuration.md) for the full reference.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Dict-based artifacts** | Plain dicts are inspectable, hashable, and serializable without a schema layer. Content-addressing via `sha256_json` ensures deterministic IDs. |
| **No Pydantic** | Avoids version coupling, validation overhead, and non-deterministic serialization. The protocol is simple enough that schema validation is unnecessary. |
| **Proof-carrying rollouts** | On-policy RL requires that training data came from the current policy. In a decentralized setting, miners can fabricate data. Sketch proofs bind rollouts to model hidden states. See [proof-system.md](proof-system.md). |
| **Dense + binary reward** | Binary SAT scoring gives no gradient signal for partial solutions. Dense clause reward decomposes the task into a smooth objective that GRPO can optimize. See [sat-environment.md](sat-environment.md). |
| **Superlinear scoring** | Quadratic weight scaling concentrates incentives on top performers, discouraging quantity-over-quality strategies. See [scoring.md](scoring.md). |
| **Windowed loop** | Each window is a self-contained training iteration with clear artifact boundaries. Windows can be replayed from artifacts for debugging or auditing. |
| **Local-first** | The demo runs entirely on localhost with a tiny model. The Bittensor adapter is a scaffold, not a hard dependency. This makes development and testing fast. |

## Showcase

The `reliquary demo showcase` command runs a deterministic local demonstration with five differentiated miner archetypes (honest high-quality, honest average, low-diversity, weak policy, malformed proof). It produces a static HTML dashboard for auditing the full pipeline.

See [showcase.md](showcase.md) for the PoC walkthrough, miner archetypes, and dashboard output.

## Documentation Index

| Document | Contents |
|----------|----------|
| [artifacts.md](artifacts.md) | Artifact protocol and schema reference |
| [proof-system.md](proof-system.md) | Sketch proof construction and verification |
| [sat-environment.md](sat-environment.md) | SAT challenge generation and evaluation |
| [scoring.md](scoring.md) | Uniqueness, scoring, and incentive mechanism |
| [training.md](training.md) | GRPO training pipeline |
| [grpo_analysis.md](grpo_analysis.md) | Analytical argument for why GRPO works here |
| [showcase.md](showcase.md) | PoC walkthrough, miner archetypes, dashboard |
| [configuration.md](configuration.md) | Environment variables and deployment |
