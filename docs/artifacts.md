# Artifact Protocol Reference

Every significant piece of data in Reliquary flows through immutable, content-addressed **artifacts**. Artifacts are plain Python dicts serialized as JSON. There is no Pydantic layer: the protocol is inspectable, hashable, and straightforward.

## Common Envelope

Every artifact shares this top-level structure:

```json
{
  "artifact_type": "policy | challenge | rollout | verdict | training_batch | scorecard | checkpoint | window_manifest | run_manifest",
  "producer_id":   "string — identity of the role that created this artifact",
  "policy_id":     "string — SHA-256 hex linking this artifact to its parent policy",
  "window_id":     "int — training window this artifact belongs to",
  "created_at":    "string — ISO 8601 UTC timestamp",
  "payload":       "dict — artifact-type-specific content (see below)",
  "signature_bundle": "dict — authentication metadata (default: {\"scheme\": \"none\"})",
  "artifact_id":   "string — SHA-256 hex of the entire artifact (see ID Computation)"
}
```

## ID Computation

Artifact IDs are deterministic and content-addressed:

1. All fields except `artifact_id` are assembled into a dict.
2. The dict is serialized with `stable_json_dumps`: `sort_keys=True`, `ensure_ascii=True`, `separators=(",", ":")`.
3. The SHA-256 hex digest of the UTF-8 encoded JSON string becomes the `artifact_id`.

This means two artifacts with identical content always produce the same ID, regardless of creation order or machine.

**Policy artifacts** are special: after finalization, `policy_id == artifact_id`. This is computed by `finalize_policy_artifact`, which hashes the policy excluding both `artifact_id` and the temporary `policy_id` placeholder.

## Artifact Types

### Policy

Created by the **Trainer** at the start of each window. Defines the model, environment, and hyperparameters for the window.

```
artifact_type: "policy"
producer_id:   trainer_id
policy_id:     self (policy_id == artifact_id after finalization)
```

**Payload:**

| Field | Type | Description |
|-------|------|-------------|
| `policy_version` | `"v1"` | Protocol version |
| `model_ref` | string | HuggingFace model name or local checkpoint path |
| `env_id` | `"sat"` | Environment identifier |
| `env_config` | dict | `{n, k, m}` — SAT problem parameters |
| `generation_config` | dict | `{max_new_tokens, temperature, top_p, top_k, repetition_penalty}` |
| `trainer_config` | dict | `{lr, epochs, clip_eps, kl_coef}` |
| `allowed_artifact_types` | list | Artifact families accepted under this policy contract |

### Challenge

Created by the **Trainer**. Defines a specific SAT instance for miners to solve.

```
artifact_type: "challenge"
producer_id:   trainer_id
policy_id:     current policy's ID
```

**Payload:**

| Field | Type | Description |
|-------|------|-------------|
| `challenge` | dict | Full challenge object (see below) |

The nested `challenge` dict contains:

| Field | Type | Description |
|-------|------|-------------|
| `challenge_id` | string | `"sat-{seed}"` |
| `env_id` | `"sat"` | Environment identifier |
| `seed` | int | Deterministic seed: `window_id * 10_000 + offset` |
| `prompt` | string | Human-readable SAT formula with instructions |
| `extra.n` | int | Number of variables |
| `extra.k` | int | Literals per clause |
| `extra.m` | int | Number of clauses |
| `extra.solution` | dict | Ground-truth satisfying assignment `{var_index: bool}` |
| `extra.clauses` | list | List of clauses, each a list of signed literal integers |
| `extra.formula` | string | Formula in `(x1 v !x2) ^ ...` notation |

### Rollout

Created by the **Miner**. Contains a model completion, its logprobs, and a sketch proof.

```
artifact_type: "rollout"
producer_id:   miner_id
policy_id:     current policy's ID
signature_bundle: {"scheme": "local-hmac", "miner_id": "..."}
```

**Payload:**

| Field | Type | Description |
|-------|------|-------------|
| `challenge_id` | string | ID of the challenge being solved |
| `challenge` | dict | Full challenge object (denormalized) |
| `sample_index` | int | Index within this miner's rollouts for this challenge |
| `generation` | dict | Completion data (see below) |
| `proof` | dict | Sketch proof (see [proof-system.md](proof-system.md)) |

The `generation` dict contains:

| Field | Type | Description |
|-------|------|-------------|
| `prompt_token_ids` | list[int] | Tokenized prompt |
| `completion_token_ids` | list[int] | Generated token IDs (completion only) |
| `sequence_token_ids` | list[int] | Full sequence (prompt + completion) |
| `completion_text` | string | Decoded completion text |
| `completion_logprobs` | list[float] | Per-token log-probabilities for the completion |
| `sum_logprob` | float | Sum of completion log-probabilities |

### Verdict

Created by the **Validator**. Records proof verification and SAT evaluation results for a single rollout.

```
artifact_type: "verdict"
producer_id:   validator_id
policy_id:     current policy's ID
```

**Payload:**

| Field | Type | Description |
|-------|------|-------------|
| `challenge_id` | string | Challenge this verdict evaluates |
| `rollout_artifact_id` | string | ID of the evaluated rollout |
| `rollout_producer_id` | string | Miner who produced the rollout |
| `proof_valid` | bool | Whether the sketch proof passed verification |
| `proof_reason` | string | `"ok"`, `"bad_signature"`, `"index_mismatch"`, or `"sketch_mismatch:{idx}"` |
| `score` | float | Binary: `1.0` if all clauses satisfied, `0.0` otherwise |
| `dense_reward` | float | Fraction of clauses satisfied (0.0 to 1.0) |
| `satisfied_fraction` | float | Same as `dense_reward` |
| `accepted` | bool | `proof_valid AND evaluation_accepted` |
| `uniqueness_key` | string | SHA-256 of comma-joined `completion_token_ids` |
| `is_unique` | bool | Added by post-processing; `false` if another rollout for the same challenge has the same completion |
| `format_ok` | bool | Whether the response was parseable as a variable assignment |
| `format_reason` | string | `"ok"` or description of parsing failure |
| `assignment_count` | int | Number of variables assigned in the response |
| `expected_assignment_count` | int | Number of variables in the formula |
| `missing_variables` | list[int] | Variables not assigned |
| `out_of_range_variables` | list[int] | Variables assigned but not in the formula |
| `rollout_archetype` | string | Miner archetype label (from showcase profiles) |
| `miner_profile` | dict | Optional miner metadata from the rollout |
| `verdict_explanation` | dict | Structured explanation (see below) |
| `challenge` | dict | Denormalized challenge |
| `generation` | dict | Denormalized generation data |

The `verdict_explanation` dict provides a human-readable breakdown:

```json
{
  "summary": "accepted_for_scoring | rejected:{reason}",
  "proof": {"ok": true, "reason": "ok"},
  "format": {"ok": true, "reason": "ok", "assignment_count": 4, ...},
  "reward": {"score": 1.0, "dense_reward": 0.88, "quality_band": "strong"},
  "uniqueness": {"ok": true, "key": "...", "reason": "unique"},
  "final_decision": {"accepted": true, "training_eligible": true, "status": "accepted"}
}
```

### Scorecard

Created by the **Validator** after processing all rollouts for a window. Summarizes miner weights.

```
artifact_type: "scorecard"
producer_id:   validator_id
policy_id:     current policy's ID
```

**Payload:**

| Field | Type | Description |
|-------|------|-------------|
| `totals` | dict[string, float] | Raw accumulated `dense_reward` per miner |
| `weights` | dict[string, float] | Normalized weights after superlinear scaling (sum to 1.0) |
| `miners` | dict[string, dict] | Per-miner counts, rewards, failures, and published weight |
| `window_metrics` | dict | Aggregate accepted, rejected, proof-failure, format-failure, and non-unique counts |

### Training Batch

Created by the **Trainer** before GRPO runs. Captures exactly which verified verdicts are allowed into training.

```
artifact_type: "training_batch"
producer_id:   trainer_id
policy_id:     current policy's ID
```

**Payload:**

| Field | Type | Description |
|-------|------|-------------|
| `source_verdict_ids` | list[string] | Verdict artifacts selected for training |
| `challenge_ids` | list[string] | Challenges represented in the batch |
| `group_count` | int | Number of GRPO groups |
| `accepted_verdict_count` | int | Number of training-eligible verdicts |
| `mean_dense_reward` | float | Mean dense reward of the batch |
| `mean_binary_score` | float | Mean binary score of the batch |

### Checkpoint

Created by the **Trainer** after a successful training step. Links the trained model to its evaluation metrics.

```
artifact_type: "checkpoint"
producer_id:   trainer_id
policy_id:     next policy's ID (window_id + 1)
window_id:     next window (window_id + 1)
```

**Payload:**

| Field | Type | Description |
|-------|------|-------------|
| `checkpoint_id` | string | `"checkpoint-window-{NNNN}"` |
| `checkpoint_path` | string | Filesystem path to saved model + tokenizer |
| `parent_policy_id` | string | Policy ID of the window that produced training data |
| `training_summary` | dict | `{accepted_samples, groups, mean_loss, steps}` |
| `eval_before` | dict | `{eval_count, success_rate, mean_dense_reward}` |
| `eval_after` | dict | Same structure, measured after training |

### Window Manifest

Created by the **Trainer** at the end of every window. Captures the full lineage for a single step of the subnet loop.

**Payload highlights:**

- `run_id`
- `policy_id`
- `challenge_ids`
- `rollout_artifact_ids`
- `verdict_artifact_ids`
- `scorecard_artifact_id`
- `training_batch_artifact_id`
- `checkpoint_artifact_id`
- `next_policy_artifact_id`
- `participating_miners`
- `accepted_count`
- `training_eligible_count`
- `rejected_count`
- `weights`
- `eval_before`
- `eval_after`

### Run Manifest

Created by the **Trainer** at the end of a showcase run. Captures top-level export metadata.

**Payload highlights:**

- `run_id`
- `profile`
- `profile_label`
- `description`
- `initial_policy_id`
- `final_policy_id`
- `window_manifest_ids`
- `aggregate_metrics`
- `output_assets`

## Storage Layout

The `LocalArtifactRegistry` persists artifacts as JSON files under `RELIQUARY_DATA_DIR`:

```
data/
  policies/
    {artifact_id}.json
  challenges/
    {artifact_id}.json
  rollouts/
    {artifact_id}.json
  verdicts/
    {artifact_id}.json
    weights-window-{NNNN}.json    # compatibility summary emitted by CLI validate
  training_batches/
    {artifact_id}.json
  scorecards/
    {artifact_id}.json
  checkpoints/
    checkpoint-window-{NNNN}/     # saved model + tokenizer
  window_manifests/
    {artifact_id}.json
  run_manifests/
    {artifact_id}.json
  runs/
    <run-id>/
      artifacts/
      dashboard/
      SUMMARY.md
      chain-events.json
      run-manifest.json
  training/
    window_{NNNN}/
      grpo_groups.json
      window_summary.json
```

Artifacts are retrieved by type and optionally filtered by `window_id`. The `latest_policy()` method returns the policy with the highest `(window_id, created_at)`.
