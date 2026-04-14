# SAT Environment

Reliquary's first environment is **k-SAT**: miners receive a Boolean satisfiability formula and must produce a satisfying assignment. SAT was chosen because it is:

- **Verifiable**: an assignment can be checked in linear time.
- **Decomposable**: partial solutions (fraction of clauses satisfied) provide dense gradient signal.
- **Scalable**: difficulty is tunable via `(n, k, m)` parameters.
- **Well-studied**: the satisfiability phase transition provides principled difficulty curves.

## Challenge Generation

`generate_sat_challenge(n, k, m, seed)` produces a random k-SAT instance that is **guaranteed satisfiable by construction**.

### Algorithm

1. Seed a deterministic RNG with `seed`.
2. Generate a random ground-truth solution: `solution[i] = random(True, False)` for `i` in `1..n`.
3. For each of the `m` clauses:
   - Sample `k` distinct variables uniformly from `1..n`.
   - Pick one variable as the "satisfying literal": set its sign to match the ground-truth solution.
   - For the remaining `k-1` variables: assign signs randomly.
   - Append the clause (list of signed literal integers; negative = negated).
4. Format the formula in human-readable infix notation: `(x1 v !x2 v x3) ^ (x4 v x1 v !x5) ^ ...`
5. Construct the prompt with the formula and expected response format.

### Prompt Format

The generated prompt looks like:

```
Solve this SAT problem. Given the formula:
(x1 v !x2 v x3) ^ (!x1 v x4 v x2) ^ ...

Provide a satisfying assignment in the format:
x1=True, x2=False, x3=True, ...
```

### Challenge Seed Convention

Seeds are deterministic per window and offset:

- **Training challenges**: `seed = window_id * 10_000 + offset`
- **Evaluation challenges**: `seed = 10_000 + offset` (fixed range, independent of window)

This ensures challenges are reproducible and non-overlapping between windows.

### Output Structure

```json
{
  "challenge_id": "sat-{seed}",
  "env_id": "sat",
  "seed": 42,
  "prompt": "Solve this SAT problem...",
  "extra": {
    "n": 4,
    "k": 3,
    "m": 17,
    "solution": {"1": true, "2": false, "3": true, "4": true},
    "clauses": [[1, -2, 3], [-1, 4, 2], ...],
    "formula": "(x1 v !x2 v x3) ^ (!x1 v x4 v x2) ^ ..."
  }
}
```

## Response Parsing

`parse_assignments(text)` extracts variable assignments from free-form text using the regex pattern:

```
x(\d+)\s*=\s*(True|False|1|0)
```

- Case-insensitive matching.
- Accepts both `True/False` and `1/0`.
- Returns a dict mapping variable index (int) to boolean value.
- Missing variables are treated as unassigned (clauses referencing them will not be satisfied).

## Evaluation

`evaluate_sat_response(challenge, response_text)` scores a miner's response:

1. Parse assignments from the response text.
2. For each clause in the formula:
   - A clause is **satisfied** if at least one of its literals evaluates to `True`.
   - Positive literal `x_i`: satisfied if `assignment[i] == True`.
   - Negative literal `!x_i` (represented as `-i`): satisfied if `assignment[i] == False`.
3. Compute metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `score` | float | `1.0` if ALL clauses satisfied, `0.0` otherwise |
| `dense_reward` | float | Fraction of clauses satisfied (0.0 to 1.0) |
| `satisfied_fraction` | float | Same as `dense_reward` |
| `accepted` | bool | `True` if clauses exist (schema validity, not correctness) |
| `got` | dict | Parsed assignment `{var: bool}` |
| `expected` | dict | Ground-truth solution |

### Why Two Reward Signals

- **`score` (binary)**: Preserves strict task semantics. A partial solution is not a real solution.
- **`dense_reward` (continuous)**: Provides gradient signal for GRPO training even when the model cannot yet solve the full instance. Satisfying 15 out of 17 clauses is better than satisfying 5, and the training loop can use this to improve the policy incrementally.

Both values are computed by the validator, not self-reported by the miner.

## Configuration

| Variable | Default | Effect |
|----------|---------|--------|
| `RELIQUARY_SAT_N` | `4` | Number of Boolean variables |
| `RELIQUARY_SAT_K` | `3` | Literals per clause (k-SAT) |
| `RELIQUARY_SAT_M` | `17` | Number of clauses |

### Difficulty Scaling

For random k-SAT, the satisfiability phase transition occurs near the **clause-to-variable ratio** `m/n ~ 2^k * ln(2)`:

| k | Threshold m/n |
|---|---------------|
| 2 | ~1.39 |
| 3 | ~5.54 |
| 4 | ~11.09 |

The defaults (`n=4, k=3, m=17`) give a ratio of `17/4 = 4.25`, which is below the k=3 threshold. This means most random instances are satisfiable (and all Reliquary instances are satisfiable by construction). Increasing `m` or decreasing `n` makes instances harder to solve.

**Recommended curriculum**: Start with small `(n=4, k=3, m=10)` for initial training, then gradually increase `m` and `n` as the model's solve rate improves.

## Extending to New Environments

The artifact protocol is environment-agnostic. To add a new environment:

1. Create a new file in `reliquary/envs/` (e.g., `math_env.py`).
2. Implement two functions:
   - `generate_*_challenge(seed, ...) -> dict` — must return a dict with at least `challenge_id`, `env_id`, `seed`, `prompt`, and `extra`.
   - `evaluate_*_response(challenge, response_text) -> dict` — must return at least `score` (binary), `dense_reward` (continuous), and `accepted` (bool).
3. Update the trainer to call the new generation function when `env_id` matches.
4. Update the validator to call the new evaluation function.

The proof system, artifact protocol, scoring, and training pipeline work unchanged with any environment that provides these two interfaces.
