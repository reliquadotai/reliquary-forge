"""End-to-end distillation lane driver.

Orchestrates a mini distillation cycle:
  1. For each task, hit a teacher chat-completion endpoint and capture
     the reasoning trace, final answer, and per-token logprobs.
  2. Screen the trace for benchmark contamination.
  3. Build a conditioning prompt that blends the task + teacher trace.
  4. Hit a (possibly different) student endpoint with the conditioning
     prompt; capture the student output.
  5. Emit artifacts for teacher traces + student completions with a
     provenance binding (student.trace_artifact_id -> teacher.artifact_id).

Both endpoints use the OpenAI chat-completions wire format so this works
with vLLM, text-generation-webui, or any drop-in.

Spec: private/reliquary-plan/notes/spec-distillation-lane.md.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable
from urllib import error as urlerror
from urllib import request


DEFAULT_CONTAMINATION_NGRAM = 10
DEFAULT_CONTAMINATION_THRESHOLD = 0.05
DEFAULT_MIN_TRACE_LENGTH_CHARS = 40
DEFAULT_MAX_TRACE_LENGTH_CHARS = 8192
DEFAULT_FINAL_ANSWER_REGEX = re.compile(
    r"(?:^|\n|\*\*|\.\s+)\s*"
    r"(?:final[\s-]*answer|answer|the answer is|so[,]?\s+the\s+answer\s+is|therefore[,]?)"
    r"\s*[:=]\s+(.+?)(?:\n|$)",
    re.IGNORECASE,
)


def _strip_think_blocks(text: str) -> str:
    """Qwen3 and similar models wrap internal reasoning in ``<think>...</think>``.

    The final answer typically appears AFTER the closing tag. This helper
    returns whatever appears outside the think block; empty string if the
    whole output was inside an unclosed think block (usually means the
    model ran out of tokens before finalizing).
    """
    # Drop any explicit <think>...</think> blocks.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # If a think block was opened and never closed, drop from the open tag onward
    # so the answer extractor doesn't see the half-finished reasoning.
    open_match = re.search(r"<think>", text, flags=re.IGNORECASE)
    if open_match:
        text = text[: open_match.start()]
    return text.strip()


@dataclass(frozen=True)
class TeacherModelRef:
    model_name: str
    model_revision: str = ""
    endpoint_url: str = "http://127.0.0.1:8001/v1"


@dataclass(frozen=True)
class StudentModelRef:
    model_name: str
    model_revision: str = ""
    endpoint_url: str = "http://127.0.0.1:8001/v1"


@dataclass(frozen=True)
class DistillTask:
    task_id: str
    prompt: str
    source: str
    difficulty: float = 0.5
    expected_answer_hint: str | None = None


@dataclass
class ScreenResult:
    flagged: bool
    overlap: float
    matched_ngrams: list[str] = field(default_factory=list)
    reason: str | None = None


@dataclass
class TeacherTraceArtifact:
    artifact_id: str
    teacher_model: TeacherModelRef
    task: DistillTask
    reasoning_trace: str
    final_answer: str
    logprobs: list[float] | None
    sampling_params: dict
    teacher_hotkey: str
    created_at: str
    contamination_flag: bool
    contamination_reason: str | None

    def to_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "teacher_model": asdict(self.teacher_model),
            "task": asdict(self.task),
            "reasoning_trace": self.reasoning_trace,
            "final_answer": self.final_answer,
            "logprobs": self.logprobs,
            "sampling_params": dict(self.sampling_params),
            "teacher_hotkey": self.teacher_hotkey,
            "created_at": self.created_at,
            "contamination_flag": self.contamination_flag,
            "contamination_reason": self.contamination_reason,
        }


@dataclass
class StudentCompletionArtifact:
    artifact_id: str
    trace_artifact_id: str
    student_model: StudentModelRef
    miner_hotkey: str
    student_output: str
    student_logprobs: list[float] | None
    sampling_params: dict
    created_at: str

    def to_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "trace_artifact_id": self.trace_artifact_id,
            "student_model": asdict(self.student_model),
            "miner_hotkey": self.miner_hotkey,
            "student_output": self.student_output,
            "student_logprobs": self.student_logprobs,
            "sampling_params": dict(self.sampling_params),
            "created_at": self.created_at,
        }


def _canonicalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", normalized).strip()


def _shingle(text: str, n: int) -> set[str]:
    tokens = re.findall(r"\w+", text)
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def contamination_screen(
    text: str,
    benchmark_ngrams: set[str],
    *,
    ngram: int = DEFAULT_CONTAMINATION_NGRAM,
    threshold: float = DEFAULT_CONTAMINATION_THRESHOLD,
) -> ScreenResult:
    canonical = _canonicalize(text)
    text_ngrams = _shingle(canonical, ngram)
    if not text_ngrams:
        return ScreenResult(flagged=False, overlap=0.0, matched_ngrams=[], reason="text_too_short")
    overlap = len(text_ngrams & benchmark_ngrams) / max(1, len(text_ngrams))
    return ScreenResult(
        flagged=overlap > threshold,
        overlap=overlap,
        matched_ngrams=sorted(text_ngrams & benchmark_ngrams)[:10],
        reason="benchmark_overlap" if overlap > threshold else None,
    )


def extract_final_answer(trace: str) -> str | None:
    """Return the trace's final-answer substring, or None if absent.

    Strategy: strip ``<think>...</think>`` reasoning blocks first (Qwen3-style)
    so the answer regex doesn't match an interim guess inside the internal
    reasoning. Then look for a canonical anchor (``Final Answer:``,
    ``The answer is:`` and the like) that MUST have the ``:`` or ``=``
    separator so unrelated prose about "the answer" does not fire.
    Returns None if no anchor matches.
    """
    outer = _strip_think_blocks(trace)
    match = DEFAULT_FINAL_ANSWER_REGEX.search(outer or trace)
    if match is not None:
        return match.group(1).strip()
    return None


def _artifact_id(payload: dict) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _chat_completion(
    *,
    endpoint_url: str,
    model: str,
    messages: list[dict],
    sampling_params: dict,
    logprobs: bool = False,
    timeout: float = 120.0,
) -> dict:
    body = {
        "model": model,
        "messages": messages,
        "temperature": sampling_params.get("temperature", 0.7),
        "top_p": sampling_params.get("top_p", 0.9),
        "max_tokens": sampling_params.get("max_tokens", 512),
    }
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = sampling_params.get("top_logprobs", 1)
    url = endpoint_url.rstrip("/") + "/chat/completions"
    req = request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urlerror.HTTPError as exc:
        body_bytes = exc.read() if hasattr(exc, "read") else b""
        raise RuntimeError(
            f"chat completion HTTP {exc.code} at {url}: {body_bytes[:400].decode(errors='replace')}"
        ) from exc


def run_teacher_cycle(
    teacher_model: TeacherModelRef,
    tasks: Iterable[DistillTask],
    *,
    teacher_hotkey: str,
    benchmark_ngrams: set[str] | None = None,
    sampling_params: dict | None = None,
) -> list[TeacherTraceArtifact]:
    benchmark_ngrams = benchmark_ngrams or set()
    sampling_params = sampling_params or {"temperature": 0.7, "top_p": 0.9, "max_tokens": 512}
    out: list[TeacherTraceArtifact] = []
    for task in tasks:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful reasoning assistant. Work step by step. "
                    "End your response with a line of the form `Final Answer: <answer>`."
                ),
            },
            {"role": "user", "content": task.prompt},
        ]
        response = _chat_completion(
            endpoint_url=teacher_model.endpoint_url,
            model=teacher_model.model_name,
            messages=messages,
            sampling_params=sampling_params,
            logprobs=True,
        )
        choice = response["choices"][0]
        trace = choice["message"]["content"]
        final = extract_final_answer(trace) or ""
        logprobs_data = choice.get("logprobs") or {}
        raw_logprobs = logprobs_data.get("content") if isinstance(logprobs_data, dict) else None
        logprobs = [float(tok["logprob"]) for tok in (raw_logprobs or [])] or None
        screen = contamination_screen(trace, benchmark_ngrams)
        payload = {
            "teacher": asdict(teacher_model),
            "task": asdict(task),
            "trace": trace,
            "final": final,
        }
        artifact_id = _artifact_id(payload)
        out.append(
            TeacherTraceArtifact(
                artifact_id=artifact_id,
                teacher_model=teacher_model,
                task=task,
                reasoning_trace=trace,
                final_answer=final,
                logprobs=logprobs,
                sampling_params=dict(sampling_params),
                teacher_hotkey=teacher_hotkey,
                created_at=_iso_now(),
                contamination_flag=screen.flagged,
                contamination_reason=screen.reason,
            )
        )
    return out


def run_student_cycle(
    student_model: StudentModelRef,
    traces: Iterable[TeacherTraceArtifact],
    *,
    miner_hotkey: str,
    sampling_params: dict | None = None,
) -> list[StudentCompletionArtifact]:
    sampling_params = sampling_params or {"temperature": 0.7, "top_p": 0.9, "max_tokens": 256}
    out: list[StudentCompletionArtifact] = []
    for trace in traces:
        if trace.contamination_flag:
            continue
        conditioning = (
            f"{trace.task.prompt}\n\n"
            f"Reference reasoning (for guidance; follow its approach):\n{trace.reasoning_trace}\n\n"
            "Now produce your own concise answer following the same approach. "
            "End with `Final Answer: <answer>`."
        )
        messages = [
            {"role": "user", "content": conditioning},
        ]
        response = _chat_completion(
            endpoint_url=student_model.endpoint_url,
            model=student_model.model_name,
            messages=messages,
            sampling_params=sampling_params,
            logprobs=True,
        )
        choice = response["choices"][0]
        output = choice["message"]["content"]
        logprobs_data = choice.get("logprobs") or {}
        raw_logprobs = logprobs_data.get("content") if isinstance(logprobs_data, dict) else None
        logprobs = [float(tok["logprob"]) for tok in (raw_logprobs or [])] or None
        payload = {
            "student": asdict(student_model),
            "parent": trace.artifact_id,
            "output": output,
        }
        artifact_id = _artifact_id(payload)
        out.append(
            StudentCompletionArtifact(
                artifact_id=artifact_id,
                trace_artifact_id=trace.artifact_id,
                student_model=student_model,
                miner_hotkey=miner_hotkey,
                student_output=output,
                student_logprobs=logprobs,
                sampling_params=dict(sampling_params),
                created_at=_iso_now(),
            )
        )
    return out


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def run_student_cycle_local(
    hf_model_name: str,
    traces: Iterable[TeacherTraceArtifact],
    *,
    miner_hotkey: str,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    sampling_params: dict | None = None,
) -> list[StudentCompletionArtifact]:
    """Run the student cycle against a locally-loaded HuggingFace model.

    Unlike :func:`run_student_cycle`, this path does not require an HTTP
    endpoint. It loads the model in-process with ``transformers`` and runs
    inference directly. Intended for small students (Qwen3-4B class) where
    the overhead of running a vLLM server per validator is not justified,
    and for the asymmetric-hardware distillation configuration where a
    Blackwell staging node produces student completions conditioned on
    teacher traces sourced from a separate host.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sampling_params = sampling_params or {"temperature": 0.7, "top_p": 0.9, "max_tokens": 256}
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    student_ref = StudentModelRef(
        model_name=hf_model_name,
        endpoint_url=f"local://{device}",
    )

    out: list[StudentCompletionArtifact] = []
    for trace in traces:
        if trace.contamination_flag:
            continue
        conditioning = (
            f"{trace.task.prompt}\n\n"
            f"Reference reasoning (for guidance; follow its approach):\n"
            f"{trace.reasoning_trace}\n\n"
            "Now produce your own concise answer following the same approach. "
            "End with `Final Answer: <answer>`."
        )
        try:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": conditioning}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:  # models without a chat template
            prompt_text = conditioning
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=int(sampling_params.get("max_tokens", 256)),
                temperature=float(sampling_params.get("temperature", 0.7)),
                top_p=float(sampling_params.get("top_p", 0.9)),
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion_ids = generated[0][inputs["input_ids"].shape[1]:]
        output = tokenizer.decode(completion_ids, skip_special_tokens=True)
        payload = {
            "student": asdict(student_ref),
            "parent": trace.artifact_id,
            "output": output,
        }
        artifact_id = _artifact_id(payload)
        out.append(
            StudentCompletionArtifact(
                artifact_id=artifact_id,
                trace_artifact_id=trace.artifact_id,
                student_model=student_ref,
                miner_hotkey=miner_hotkey,
                student_output=output,
                student_logprobs=None,
                sampling_params=dict(sampling_params),
                created_at=_iso_now(),
            )
        )
    return out


def default_benchmark_ngrams() -> set[str]:
    """Small hardcoded benchmark ngram set for initial contamination screening.

    Real deployments substitute a benchmark-specific pre-computed ngram store;
    this default exists so the screener always has something to compare
    against, producing non-empty results for sanity.
    """
    stems = [
        "Let us think step by step about this problem and reason it through from first principles carefully.",
        "The correct answer to this multiple choice question is exactly one of the four provided options below.",
        "We can solve this by plugging the given values directly into the equation and simplifying both sides.",
        "Choose the best answer: option A stands for zero option B stands for one option C stands for two option D stands for three.",
        "Write a Python function named solution that takes a list of integers and returns the sum of all of them efficiently.",
        "Given the following premises derive the conclusion using standard rules of propositional and predicate logic only.",
    ]
    ngrams: set[str] = set()
    for stem in stems:
        canonical = _canonicalize(stem)
        ngrams |= _shingle(canonical, DEFAULT_CONTAMINATION_NGRAM)
    return ngrams
