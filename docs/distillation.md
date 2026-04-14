# Distillation Lane

Reliquary's first launch commodity is **verified reasoning traces**.

## Flow

1. `PolicyAuthority` publishes a reasoning policy.
2. `TaskBatchArtifact` defines deterministic reasoning tasks.
3. Generation miners emit `DistillTraceArtifact`s with teacher provenance.
4. Validator mesh checks:
   - provenance signature
   - final-answer format
   - task correctness
   - contamination flags
   - uniqueness
5. `DistillPackArtifact` exports accepted traces as deterministic `jsonl` and `parquet`.
6. Learner backends produce:
   - `CheckpointArtifact`
   - `BenchmarkResultArtifact`
   - `PolicyAuthorityDecisionArtifact`

## Current implementation

- Local deterministic teacher backend for demo and tests
- `vLLM` teacher backend for real remote serving
- Local reasoning-student backend for measurable holdout lift in the demo
- `TransformersSFTLearnerBackend` for the first real `Qwen3-4B` LoRA student path
- Policy promotion through a trusted quorum decision artifact

## CLI

```bash
reliquary publish-distill-policy
reliquary generate-task-batch
reliquary distill-generate
reliquary distill-validate
reliquary pack-distill
reliquary distill-train
reliquary distill-cycle
reliquary run-distill-worker
reliquary demo distill
```

## Acceptance gates

A trace is training-eligible only if all of these are true:

- provenance valid
- format valid
- correctness positive
- contamination absent
- uniqueness preserved
