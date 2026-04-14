# Scoring and Incentive Mechanism

The scoring system converts verified rollouts into miner weights. It is designed to reward quality over quantity, prevent duplication exploits, and concentrate incentives on top performers.

## Pipeline

```
Rollouts → Verdicts → Uniqueness Dedup → Score Aggregation → Superlinear Weighting → Normalized Weights
```

## Uniqueness Deduplication

Before scoring, verdicts are checked for duplicate completions. A miner who submits the same completion multiple times for the same challenge should not receive multiple rewards.

### How It Works

1. **Compute uniqueness key**: For each rollout, hash the comma-joined `completion_token_ids` with SHA-256:
   ```
   uniqueness_key = SHA-256("tok1,tok2,tok3,...")
   ```

2. **Mark uniqueness**: Process verdicts in order. Track seen `(challenge_id, uniqueness_key)` pairs:
   - First occurrence: `is_unique = true`
   - Subsequent occurrences: `is_unique = false`

3. **Recompute IDs**: After adding the `is_unique` field, each verdict's `artifact_id` is recomputed to maintain content-addressing integrity.

Non-unique verdicts are excluded from all reward calculations.

## Score Aggregation

Only verdicts that are both **accepted** and **unique** contribute to miner scores.

A verdict is `accepted` when:
- The sketch proof passed verification (`proof_valid = true`), AND
- The SAT evaluation completed successfully (`evaluation.accepted = true`).

For each qualifying verdict, the miner's total is incremented by the `dense_reward` (fraction of clauses satisfied):

```
totals[miner] = sum(dense_reward for each accepted, unique verdict by that miner)
```

## Superlinear Weighting

Raw totals are raised to a superlinear exponent before normalization:

```
powered[miner] = totals[miner] ^ SUPERLINEAR_EXPONENT
```

The default `SUPERLINEAR_EXPONENT = 2.0` means:

- A miner with 2x the raw reward gets **4x** the weight.
- A miner with 3x the raw reward gets **9x** the weight.

This disproportionately rewards top performers and incentivizes solving harder instances (high dense_reward) rather than mass-producing mediocre rollouts (many low dense_reward).

## Weight Normalization

Powered scores are normalized to sum to 1.0:

```
weight[miner] = powered[miner] / sum(powered[all miners])
```

If no miners have qualifying verdicts, division by zero is prevented by defaulting the denominator to 1.0.

## Output

The scoring function returns:

```json
{
  "totals": {
    "miner-A": 2.4,
    "miner-B": 1.1,
    "miner-C": 0.5
  },
  "weights": {
    "miner-A": 0.799,
    "miner-B": 0.168,
    "miner-C": 0.033
  }
}
```

## Worked Example

Three miners submit rollouts for a window with 4 challenges:

| Miner | Accepted + Unique Verdicts | Dense Rewards | Total |
|-------|---------------------------|---------------|-------|
| Alice | 3 | 0.8, 0.9, 0.7 | 2.4 |
| Bob | 2 | 0.6, 0.5 | 1.1 |
| Carol | 1 | 0.5 | 0.5 |

**Superlinear weighting** (exponent = 2.0):

```
powered[Alice] = 2.4^2 = 5.76
powered[Bob]   = 1.1^2 = 1.21
powered[Carol] = 0.5^2 = 0.25
total_power    = 7.22
```

**Normalized weights:**

```
weight[Alice] = 5.76 / 7.22 = 0.798
weight[Bob]   = 1.21 / 7.22 = 0.168
weight[Carol] = 0.25 / 7.22 = 0.035
```

Alice has 2.18x Bob's raw reward but receives 4.75x the weight. The superlinear exponent amplifies quality differences.

## Weight Publication

Weights flow to the chain adapter:

- **Local mode** (`LocalChainAdapter`): Stores weights in an in-memory dict. Used for local testing and the demo flow.
- **Bittensor mode** (`BittensorChainAdapter`): Calls `subtensor.set_weights(wallet, netuid, uids, weights)` to publish on-chain. Maps miner identity strings to Bittensor UIDs.

The chain adapter is a scaffold in the current version. The CLI demo flow uses the local adapter; real subnet deployment would wire up the Bittensor adapter in the validation step.

## Design Rationale

| Choice | Why |
|--------|-----|
| Dense reward, not binary | Binary scoring produces all-zero groups when the model can't fully solve instances. Dense reward gives the trainer signal even for partial solutions. |
| Uniqueness dedup | Prevents miners from submitting identical completions to farm rewards. Checked per-challenge, not globally. |
| Superlinear weighting | Linear weights give insufficient incentive to solve harder instances. Quadratic scaling makes quality dramatically more valuable than quantity. |
| Validator-computed scores | Miners cannot self-report rewards. The dense_reward is recomputed by the validator from the actual clause evaluation, and proof verification ensures the completion is authentic. |
