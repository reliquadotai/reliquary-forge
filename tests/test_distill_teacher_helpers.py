"""Expanded tests for distill.teacher (sign/verify + LocalReasoningTeacherBackend).

Pins the HMAC trace-signature invariants + the strategy-dispatch logic in
``_local_trace_text`` / ``LocalReasoningTeacherBackend.generate_trace`` so
a silent change to strategy branching doesn't corrupt the distillation
signal.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from reliquary.distill.teacher import (
    LocalReasoningTeacherBackend,
    VLLMTeacherBackend,
    _local_trace_text,
    _sign_trace,
    build_teacher_backend,
    verify_trace_signature,
)
from reliquary.envs.reasoning import generate_reasoning_tasks


def _example_task() -> dict:
    return generate_reasoning_tasks(count=1, seed=7, split="train")[0]


# ---------------------------------------------------------------------------
# _sign_trace / verify_trace_signature
# ---------------------------------------------------------------------------


def test_sign_trace_is_hex_sha256() -> None:
    sig = _sign_trace("task-1", "reasoning...", "42", "secret")
    assert len(sig) == 64
    assert all(ch in "0123456789abcdef" for ch in sig)


def test_sign_trace_deterministic() -> None:
    a = _sign_trace("task-1", "reasoning...", "42", "secret")
    b = _sign_trace("task-1", "reasoning...", "42", "secret")
    assert a == b


def test_sign_trace_matches_manual_hmac_sha256() -> None:
    material = json.dumps(
        {"task_id": "t", "trace_text": "body", "final_answer": "42"},
        sort_keys=True,
    ).encode("utf-8")
    expected = hmac.new(b"secret", material, hashlib.sha256).hexdigest()
    assert _sign_trace("t", "body", "42", "secret") == expected


def test_sign_trace_changes_with_task_id() -> None:
    a = _sign_trace("task-A", "body", "1", "secret")
    b = _sign_trace("task-B", "body", "1", "secret")
    assert a != b


def test_sign_trace_changes_with_trace_text() -> None:
    a = _sign_trace("t", "trace-A", "1", "secret")
    b = _sign_trace("t", "trace-B", "1", "secret")
    assert a != b


def test_sign_trace_changes_with_final_answer() -> None:
    a = _sign_trace("t", "body", "1", "secret")
    b = _sign_trace("t", "body", "2", "secret")
    assert a != b


def test_sign_trace_changes_with_secret() -> None:
    a = _sign_trace("t", "body", "1", "secret-A")
    b = _sign_trace("t", "body", "1", "secret-B")
    assert a != b


def test_verify_trace_signature_accepts_matching() -> None:
    sig = _sign_trace("task-1", "body", "42", "secret")
    result = verify_trace_signature(
        task_id="task-1",
        trace_text="body",
        final_answer="42",
        secret="secret",
        signature=sig,
    )
    assert result["valid"] is True
    assert result["reason"] == "ok"


def test_verify_trace_signature_rejects_tampered() -> None:
    sig = _sign_trace("task-1", "body", "42", "secret")
    tampered = sig[:-1] + ("0" if sig[-1] != "0" else "1")
    result = verify_trace_signature(
        task_id="task-1",
        trace_text="body",
        final_answer="42",
        secret="secret",
        signature=tampered,
    )
    assert result["valid"] is False
    assert result["reason"] == "bad_trace_signature"


def test_verify_trace_signature_rejects_wrong_secret() -> None:
    sig = _sign_trace("task-1", "body", "42", "secret-A")
    result = verify_trace_signature(
        task_id="task-1",
        trace_text="body",
        final_answer="42",
        secret="secret-B",
        signature=sig,
    )
    assert result["valid"] is False


def test_verify_trace_signature_rejects_tampered_body() -> None:
    sig = _sign_trace("task-1", "body", "42", "secret")
    result = verify_trace_signature(
        task_id="task-1",
        trace_text="DIFFERENT body",
        final_answer="42",
        secret="secret",
        signature=sig,
    )
    assert result["valid"] is False


# ---------------------------------------------------------------------------
# _local_trace_text strategy dispatch
# ---------------------------------------------------------------------------


def test_local_trace_text_default_strategy_includes_final_answer() -> None:
    task = _example_task()
    trace, answer = _local_trace_text(task, {"strategy": "reasoned_exact"}, 0)
    assert "Final Answer:" in trace
    assert answer == task["reference_answer"]


def test_local_trace_text_reasoned_duplicate_uses_correct_answer() -> None:
    task = _example_task()
    trace, answer = _local_trace_text(task, {"strategy": "reasoned_duplicate"}, 0)
    assert answer == task["reference_answer"]
    assert "canonical answer style" in trace


def test_local_trace_text_reasoned_mixed_follows_sample_index() -> None:
    task = _example_task()
    # sample_index 0 (multiple of 3) → correct answer.
    _, answer_0 = _local_trace_text(task, {"strategy": "reasoned_mixed"}, 0)
    assert answer_0 == task["reference_answer"]

    # sample_index 1 (not multiple of 3) → wrong answer (+1).
    _, answer_1 = _local_trace_text(task, {"strategy": "reasoned_mixed"}, 1)
    assert int(answer_1) == int(task["reference_answer"]) + 1


def test_local_trace_text_reasoned_contaminated_marks_leak() -> None:
    task = _example_task()
    trace, _ = _local_trace_text(task, {"strategy": "reasoned_contaminated"}, 0)
    assert "benchmark leak" in trace


def test_local_trace_text_reasoned_malformed_empty_answer() -> None:
    task = _example_task()
    trace, answer = _local_trace_text(task, {"strategy": "reasoned_malformed"}, 0)
    assert answer == ""
    assert "Final Answer" not in trace


def test_local_trace_text_default_for_unknown_strategy() -> None:
    task = _example_task()
    _, answer = _local_trace_text(task, {"strategy": "unknown-one"}, 0)
    assert answer == task["reference_answer"]


# ---------------------------------------------------------------------------
# LocalReasoningTeacherBackend.generate_trace
# ---------------------------------------------------------------------------


def test_teacher_backend_generate_trace_returns_contract_fields() -> None:
    backend = LocalReasoningTeacherBackend()
    task = _example_task()
    trace = backend.generate_trace(
        task=task,
        miner_profile={"strategy": "reasoned_exact"},
        sample_index=0,
        miner_secret="secret",
    )
    for key in (
        "trace_text",
        "final_answer",
        "teacher_model",
        "teacher_signature",
        "teacher_backend",
        "contamination_source",
        "reasoning_contract",
    ):
        assert key in trace


def test_teacher_backend_signature_verifies_against_same_secret() -> None:
    backend = LocalReasoningTeacherBackend()
    task = _example_task()
    trace = backend.generate_trace(
        task=task,
        miner_profile={"strategy": "reasoned_exact"},
        sample_index=0,
        miner_secret="secret",
    )
    result = verify_trace_signature(
        task_id=task["task_id"],
        trace_text=trace["trace_text"],
        final_answer=trace["final_answer"],
        secret="secret",
        signature=trace["teacher_signature"],
    )
    assert result["valid"] is True


def test_teacher_backend_signature_fails_under_different_secret() -> None:
    backend = LocalReasoningTeacherBackend()
    task = _example_task()
    trace = backend.generate_trace(
        task=task,
        miner_profile={"strategy": "reasoned_exact"},
        sample_index=0,
        miner_secret="secret-A",
    )
    result = verify_trace_signature(
        task_id=task["task_id"],
        trace_text=trace["trace_text"],
        final_answer=trace["final_answer"],
        secret="secret-B",
        signature=trace["teacher_signature"],
    )
    assert result["valid"] is False


def test_teacher_backend_reasoned_contaminated_sets_source_tag() -> None:
    backend = LocalReasoningTeacherBackend()
    task = _example_task()
    trace = backend.generate_trace(
        task=task,
        miner_profile={"strategy": "reasoned_contaminated"},
        sample_index=0,
        miner_secret="secret",
    )
    assert trace["contamination_source"] == "benchmark_holdout"


def test_teacher_backend_default_strategy_contamination_is_none() -> None:
    backend = LocalReasoningTeacherBackend()
    task = _example_task()
    trace = backend.generate_trace(
        task=task,
        miner_profile={"strategy": "reasoned_exact"},
        sample_index=0,
        miner_secret="secret",
    )
    assert trace["contamination_source"] is None


def test_teacher_backend_reports_local_reasoning_teacher_tag() -> None:
    backend = LocalReasoningTeacherBackend()
    task = _example_task()
    trace = backend.generate_trace(
        task=task,
        miner_profile={},
        sample_index=0,
        miner_secret="s",
    )
    assert trace["teacher_backend"] == "local_reasoning_teacher"


# ---------------------------------------------------------------------------
# build_teacher_backend factory
# ---------------------------------------------------------------------------


def test_build_teacher_backend_local_fallback_without_vllm_config() -> None:
    backend = build_teacher_backend({})
    assert isinstance(backend, LocalReasoningTeacherBackend)


def test_build_teacher_backend_prefers_vllm_when_configured() -> None:
    backend = build_teacher_backend(
        {
            "teacher_backend": "vllm",
            "vllm_base_url": "http://127.0.0.1:8000",
            "vllm_model": "qwen3-8b",
            "max_new_tokens": 48,
            "vllm_timeout_seconds": 30,
            "vllm_temperature": 0.1,
        }
    )
    assert isinstance(backend, VLLMTeacherBackend)
    assert backend.base_url == "http://127.0.0.1:8000"
    assert backend.model == "qwen3-8b"
    assert backend.temperature == pytest.approx(0.1)


def test_build_teacher_backend_falls_back_if_vllm_url_missing() -> None:
    backend = build_teacher_backend({"teacher_backend": "vllm"})
    assert isinstance(backend, LocalReasoningTeacherBackend)


def test_build_teacher_backend_honors_model_teacher_for_local() -> None:
    backend = build_teacher_backend({"model_teacher": "custom-teacher"})
    assert isinstance(backend, LocalReasoningTeacherBackend)
    assert backend.teacher_model == "custom-teacher"
