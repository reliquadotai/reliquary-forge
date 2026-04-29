# Distillation Lane — Live Audit Artifacts

Real traces + student completions produced by running the distillation lane (`reliquary/distill/lane.py`) against a live deployed vLLM endpoint, stored as structured JSON for external audit.

## Why these are here

The distillation lane's pure-Python core is unit-tested (16 tests in `tests/test_distill_lane.py`). These artifacts additionally document **what the lane produces on a real deployed teacher** — concrete evidence that the artifact schema, contamination screening, and final-answer extraction operate correctly against a production vLLM at the other end of the OpenAI chat-completions wire protocol.

## Committed runs

| File | Role | Model | Hardware | Tasks | Date |
|---|---|---|---|---|---|
| [teacher_traces_qwen3_8b.json](teacher_traces_qwen3_8b.json) | teacher | `Qwen/Qwen3-8B` (vLLM at `127.0.0.1:8001`, `gpu-memory-utilization=0.65`, `enforce-eager`) | NVIDIA RTX PRO 6000 Blackwell (dev server) | 4 | 2026-04-17 |
| [student_completions_qwen3_8b.json](student_completions_qwen3_8b.json) | student (self-distill) | same vLLM endpoint, conditioned on each teacher trace | same hardware | 4 | 2026-04-17 |
| [student_completions_qwen3_4b_staging1.json](student_completions_qwen3_4b_staging1.json) | **student (asymmetric)** | `Qwen/Qwen3-4B-Instruct-2507` via `transformers` in-process (`run_student_cycle_local`) | NVIDIA RTX PRO 6000 Blackwell (**staging1**, distinct node from the teacher) | 4 | 2026-04-17 |

## Run summary (2026-04-17)

Sampling params: `temperature=0.5, top_p=0.9, max_tokens=2048`. All four traces parsed correctly after `<think>...</think>` stripping; all four passed contamination screening; all final answers are mathematically correct:

| Task | Prompt (abbreviated) | Extracted final answer | Verified |
|---|---|---|---|
| t1 | "7th prime number?" | `17` | ✓ |
| t2 | "280km apart, 60+80 km/h, when meet?" | `2 hours.` | ✓ |
| t3 | "argument for deterministic verification" | `Deterministic verification ensures consistency, security, and trust in decentralized...` | qualitative pass |
| t4 | "f(x) = 3x+7, f(a)=22" | `$ \boxed{5} $` | ✓ |

Trace lengths: 2065 – 3681 characters each; all fit inside `max_tokens=2048`. Per-token logprobs captured from vLLM and stored in the artifact so downstream learners can consume them without re-running the teacher.

**Student cycle (2026-04-17, same session):** same Qwen3-8B endpoint served as the student, conditioned on each teacher trace via `run_student_cycle`. All 4 student completions carry `trace_artifact_id` pointing to the corresponding teacher artifact (provenance binding verified). Student answers:

| Task | Student final-answer tail | Bound to teacher |
|---|---|---|
| t1 | `... Final Answer: 17` | ✓ |
| t2 | `... = 2 hours. Final Answer: 2 hours.` | ✓ |
| t3 | `... essential for accurate model outputs and secure collaboration.` | ✓ |
| t4 | `... Final Answer: $ \boxed{5} $` | ✓ |

Using the same model family as teacher and student is a self-distillation configuration used to exercise the artifact pipeline end-to-end; production substitutes a smaller student (e.g. Qwen3-4B) on a different GPU.

**Asymmetric run (2026-04-17, dedicated smaller student on a separate node):** Qwen3-4B-Instruct-2507 loaded in-process via `transformers` on staging1 Blackwell, conditioned on the same teacher trace JSON as above. All 4 student completions produced correct answers; all 4 provenance bindings resolve against the teacher catalog.

| Task | Asymmetric student final answer | Bound |
|---|---|---|
| t1 | `... Final Answer: 17` | ✓ |
| t2 | `... = 2 hours. Final Answer: 2 hours` | ✓ |
| t3 | `... critical for reliable and secure model deployment.` | ✓ |
| t4 | `f(5) = 22 Correct. Final Answer: 5` | ✓ |

This exercises the configuration Tier 3 Epic 1 actually ships to production: teacher on a larger node produces reasoning traces; smaller student on commodity hardware consumes them. The artifact-pipeline code path is the same as the self-distillation run (`StudentCompletionArtifact` shape, `trace_artifact_id` binding, `miner_hotkey` tagging); only the student-side runtime changes (`run_student_cycle_local` instead of `run_student_cycle`).

## Reproducing

On any host with reliquary + vLLM + Qwen3-8B available:

```bash
PYTHONPATH=/path/to/reliquary python3 <<'PY'
from reliquary.distill.lane import (
    DistillTask, TeacherModelRef, run_teacher_cycle, default_benchmark_ngrams,
)

tasks = [ DistillTask(task_id="t1", prompt="…", source="math"), ... ]
teacher = TeacherModelRef(
    model_name="Qwen/Qwen3-8B",
    endpoint_url="http://127.0.0.1:8001/v1",
)
traces = run_teacher_cycle(
    teacher_model=teacher, tasks=tasks,
    teacher_hotkey="dev-server-teacher",
    benchmark_ngrams=default_benchmark_ngrams(),
    sampling_params={"temperature": 0.5, "top_p": 0.9, "max_tokens": 2048},
)
PY
```

## Open items

- **Larger task batch**: 4 tasks is a pilot. Production runs use ≥ 32 per window (matching the nine-stage verifier's CHALLENGE_K).
- **Contamination benchmark expansion**: `default_benchmark_ngrams()` ships a small hardcoded seed set. Production substitutes a full MMLU / GSM8K / HumanEval ngram store.
- **Final-answer extraction robustness**: the updated regex correctly handles Qwen3's `<think>...</think>` format. If other teacher families (Llama-3, Mistral) emit different anchors, add patterns to `DEFAULT_FINAL_ANSWER_REGEX` in `lane.py`.
- **Teacher catalog onchain commitment**: the teacher model allowlist needs to land as a `commit_policy_metadata` call once Forge runtime ownership lands. Uses the existing chain adapter primitive from the reliquary-inference Tier 1 Epic 5.
