"""Tests for the distillation lane core (pure python paths).

Actual teacher/student endpoint hits live in the integration harness
(``reliquary.distill.run_lane_cli``) and the committed audit artifacts;
these tests cover the determinism and screening properties only.
"""

from __future__ import annotations

from reliquary.distill.lane import (
    DEFAULT_FINAL_ANSWER_REGEX,
    DistillTask,
    ScreenResult,
    StudentCompletionArtifact,
    StudentModelRef,
    TeacherModelRef,
    TeacherTraceArtifact,
    _artifact_id,
    _canonicalize,
    _shingle,
    contamination_screen,
    default_benchmark_ngrams,
    extract_final_answer,
)


def test_canonicalize_collapses_whitespace_and_lowercases() -> None:
    assert _canonicalize("  Hello\nWORLD  ") == "hello world"


def test_canonicalize_handles_unicode_normalization() -> None:
    # NFKC normalizes decomposed characters.
    s = "caf\u00e9"  # 'café'
    decomposed = "cafe\u0301"  # 'cafe' + combining acute
    assert _canonicalize(s) == _canonicalize(decomposed)


def test_shingle_returns_ngrams_of_requested_size() -> None:
    text = "the quick brown fox jumps over the lazy dog"
    result = _shingle(text, 3)
    assert "the quick brown" in result
    assert "jumps over the" in result
    assert len(result) == 7


def test_shingle_handles_short_text() -> None:
    assert _shingle("hi", 3) == set()


def test_contamination_screen_flags_obvious_benchmark_overlap() -> None:
    benchmark_text = "Let's think step by step about this problem carefully right now today"
    benchmark_ngrams = _shingle(_canonicalize(benchmark_text), 10)
    result = contamination_screen(benchmark_text, benchmark_ngrams)
    assert result.flagged is True
    assert result.overlap > 0.5
    assert result.reason == "benchmark_overlap"


def test_contamination_screen_passes_clean_text() -> None:
    clean_text = (
        "I consider the problem fresh this morning and work through each subcomponent "
        "with fresh reasoning unlinked to any canonical benchmark phrasing I have seen before"
    )
    result = contamination_screen(clean_text, default_benchmark_ngrams())
    assert result.flagged is False
    assert result.reason is None


def test_contamination_screen_handles_too_short() -> None:
    result = contamination_screen("tiny", default_benchmark_ngrams())
    assert result.flagged is False
    assert result.reason == "text_too_short"


def test_extract_final_answer_parses_canonical_form() -> None:
    trace = "Step 1: compute.\nStep 2: verify.\nFinal Answer: 42"
    assert extract_final_answer(trace) == "42"


def test_extract_final_answer_handles_case_and_separator() -> None:
    trace = "thinking\nfinal answer = seventeen"
    assert extract_final_answer(trace) == "seventeen"


def test_extract_final_answer_missing_returns_none() -> None:
    assert extract_final_answer("I have no final answer here.") is None


def test_artifact_id_is_deterministic() -> None:
    payload = {"a": 1, "b": "x", "nested": {"k": [1, 2]}}
    assert _artifact_id(payload) == _artifact_id(dict(payload))


def test_artifact_id_differs_on_content_change() -> None:
    a = _artifact_id({"a": 1})
    b = _artifact_id({"a": 2})
    assert a != b


def test_trace_artifact_to_dict_roundtrip() -> None:
    teacher = TeacherModelRef(model_name="Qwen/Qwen3-8B", model_revision="abc123")
    task = DistillTask(task_id="t1", prompt="What is 2+2?", source="arith")
    artifact = TeacherTraceArtifact(
        artifact_id="0xdeadbeef",
        teacher_model=teacher,
        task=task,
        reasoning_trace="2+2 equals 4.",
        final_answer="4",
        logprobs=[-0.1, -0.2],
        sampling_params={"temperature": 0.7, "top_p": 0.9},
        teacher_hotkey="teacher-hk",
        created_at="2026-04-17T00:00:00Z",
        contamination_flag=False,
        contamination_reason=None,
    )
    data = artifact.to_dict()
    assert data["artifact_id"] == "0xdeadbeef"
    assert data["teacher_model"]["model_name"] == "Qwen/Qwen3-8B"
    assert data["task"]["prompt"] == "What is 2+2?"
    assert data["final_answer"] == "4"


def test_student_artifact_carries_parent_trace_id() -> None:
    student = StudentModelRef(model_name="Qwen/Qwen3-4B-Instruct")
    artifact = StudentCompletionArtifact(
        artifact_id="0xa",
        trace_artifact_id="0xparent",
        student_model=student,
        miner_hotkey="miner-hk",
        student_output="Final Answer: 4",
        student_logprobs=[-0.5],
        sampling_params={"temperature": 0.7, "top_p": 0.9},
        created_at="2026-04-17T00:00:00Z",
    )
    assert artifact.to_dict()["trace_artifact_id"] == "0xparent"


def test_default_benchmark_ngrams_non_empty() -> None:
    assert len(default_benchmark_ngrams()) > 0


def test_screen_result_carries_overlap_value() -> None:
    result = ScreenResult(flagged=False, overlap=0.02)
    assert result.overlap == 0.02
    assert result.matched_ngrams == []
