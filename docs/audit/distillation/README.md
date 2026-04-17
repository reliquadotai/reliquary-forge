# Distillation Lane — Live Audit Artifacts

Real traces + student completions produced by running the distillation lane (`reliquary/distill/lane.py`) against a live deployed vLLM endpoint, stored as structured JSON for external audit.

## Why these are here

The distillation lane's pure-Python core is unit-tested (16 tests in `tests/test_distill_lane.py`). These artifacts additionally document **what the lane produces on a real deployed teacher** — concrete evidence that the artifact schema, contamination screening, and final-answer extraction operate correctly against a production vLLM at the other end of the OpenAI chat-completions wire protocol.

## Committed runs

| File | Teacher | Hardware | Tasks | Date |
|---|---|---|---|---|
| [teacher_traces_qwen3_8b.json](teacher_traces_qwen3_8b.json) | `Qwen/Qwen3-8B` (vLLM at `127.0.0.1:8001`, `gpu-memory-utilization=0.65`, `enforce-eager`) | NVIDIA RTX PRO 6000 Blackwell (dev server, `wrk-mnmhkheic6r7`) | 4 (2 math, 1 reasoning, 1 algebra) | 2026-04-17 |

## Run summary (2026-04-17)

Sampling params: `temperature=0.5, top_p=0.9, max_tokens=2048`. All four traces parsed correctly after `<think>...</think>` stripping; all four passed contamination screening; all final answers are mathematically correct:

| Task | Prompt (abbreviated) | Extracted final answer | Verified |
|---|---|---|---|
| t1 | "7th prime number?" | `17` | ✓ |
| t2 | "280km apart, 60+80 km/h, when meet?" | `2 hours.` | ✓ |
| t3 | "argument for deterministic verification" | `Deterministic verification ensures consistency, security, and trust in decentralized...` | qualitative pass |
| t4 | "f(x) = 3x+7, f(a)=22" | `$ \boxed{5} $` | ✓ |

Trace lengths: 2065 – 3681 characters each; all fit inside `max_tokens=2048`. Per-token logprobs captured from vLLM and stored in the artifact so downstream learners can consume them without re-running the teacher.

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

- **Student cycle**: not included in this run; the lane includes `run_student_cycle` that takes a list of teacher traces and produces conditioned student completions. A follow-up run will wire Qwen3-4B on a Blackwell staging host (staging1 or staging2) as the student endpoint once a vLLM or transformers-based inference server is configured there.
- **Larger task batch**: 4 tasks is a pilot. Production runs use ≥ 32 per window (matching the nine-stage verifier's CHALLENGE_K).
- **Final-answer extraction robustness**: the updated regex correctly handles Qwen3's `<think>...</think>` format. If other teacher families (Llama-3, Mistral) emit different anchors, add patterns to `DEFAULT_FINAL_ANSWER_REGEX` in `lane.py`.
