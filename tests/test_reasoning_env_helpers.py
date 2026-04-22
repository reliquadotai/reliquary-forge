"""Expanded unit tests for reliquary.envs.reasoning pure helpers.

The existing test_reasoning_env.py covers the teacher + trace happy
path. This suite pins the pure-Python parser / solver / generator surface
(generate_reasoning_tasks, parse_final_answer, parse_candidate_answer,
evaluate_reasoning_trace, summarize_reasoning_evaluations,
answer_reasoning_task, build_reasoning_messages) so regressions surface
at CI time.
"""

from __future__ import annotations

import pytest

from reliquary.envs.reasoning import (
    OPS,
    PROMPT_TEMPLATES,
    REASONING_SYSTEM_PROMPT,
    _solve,
    _task_difficulty,
    answer_reasoning_task,
    build_reasoning_messages,
    evaluate_reasoning_trace,
    generate_reasoning_tasks,
    parse_candidate_answer,
    parse_final_answer,
    render_reasoning_conversation,
    summarize_reasoning_evaluations,
)

# ---------------------------------------------------------------------------
# _solve
# ---------------------------------------------------------------------------


def test_solve_add_matches_sum() -> None:
    assert _solve("add", {"left": 13, "right": 27}) == 40


def test_solve_subtract() -> None:
    assert _solve("subtract", {"left": 100, "right": 17}) == 83


def test_solve_multiply() -> None:
    assert _solve("multiply", {"left": 11, "right": 13}) == 143


def test_solve_maximum_and_minimum() -> None:
    inputs = {"left": 5, "right": 9}
    assert _solve("maximum", inputs) == 9
    assert _solve("minimum", inputs) == 5


def test_solve_absolute_difference() -> None:
    assert _solve("absolute_difference", {"left": 3, "right": 11}) == 8


def test_solve_multiply_add() -> None:
    assert _solve("multiply_add", {"left": 3, "right": 4, "extra": 5}) == 17


def test_solve_add_then_multiply() -> None:
    assert _solve("add_then_multiply", {"left": 3, "right": 4, "extra": 5}) == 35


def test_solve_unknown_operation_raises() -> None:
    with pytest.raises(ValueError):
        _solve("exponentiate", {"left": 2, "right": 3})


# ---------------------------------------------------------------------------
# _task_difficulty
# ---------------------------------------------------------------------------


def test_task_difficulty_bounded_to_one() -> None:
    for op in OPS:
        diff = _task_difficulty(op, 1000, 1000)
        assert 0.0 <= diff <= 1.0


def test_task_difficulty_magnitude_bonus_monotone() -> None:
    low = _task_difficulty("add", 1, 1)
    high = _task_difficulty("add", 99, 99)
    assert high >= low


def test_task_difficulty_unknown_op_has_base() -> None:
    # Unknown ops fall through to base 0.50.
    assert _task_difficulty("no-such-op", 0, 0) == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# generate_reasoning_tasks
# ---------------------------------------------------------------------------


def test_generate_reasoning_tasks_count_matches() -> None:
    tasks = generate_reasoning_tasks(count=5, seed=0, split="train")
    assert len(tasks) == 5


def test_generate_reasoning_tasks_deterministic_under_seed() -> None:
    a = generate_reasoning_tasks(count=3, seed=42, split="train")
    b = generate_reasoning_tasks(count=3, seed=42, split="train")
    assert a == b


def test_generate_reasoning_tasks_different_seed_differs() -> None:
    a = generate_reasoning_tasks(count=3, seed=1, split="train")
    b = generate_reasoning_tasks(count=3, seed=2, split="train")
    assert a != b


def test_generate_reasoning_tasks_id_has_split_and_seed() -> None:
    task = generate_reasoning_tasks(count=1, seed=17, split="holdout")[0]
    assert task["task_id"].startswith("reasoning-holdout-17-")
    assert task["split"] == "holdout"
    assert task["seed"] == 17


def test_generate_reasoning_tasks_operation_cycles_through_ops() -> None:
    tasks = generate_reasoning_tasks(count=len(OPS), seed=0, split="train")
    observed = {task["operation"] for task in tasks}
    assert observed == set(OPS)


def test_generate_reasoning_tasks_reference_answer_matches_solve() -> None:
    tasks = generate_reasoning_tasks(count=8, seed=99, split="train")
    for task in tasks:
        expected = _solve(task["operation"], task["inputs"])
        assert task["reference_answer"] == str(expected)


def test_generate_reasoning_tasks_template_id_in_range() -> None:
    tasks = generate_reasoning_tasks(count=8, seed=0, split="train")
    for task in tasks:
        index = int(task["template_id"].rsplit("-", 1)[-1])
        assert 0 <= index < len(PROMPT_TEMPLATES)


def test_generate_reasoning_tasks_tags_include_operation_and_split() -> None:
    task = generate_reasoning_tasks(count=1, seed=0, split="train")[0]
    assert f"operation:{task['operation']}" in task["tags"]
    assert "split:train" in task["tags"]


# ---------------------------------------------------------------------------
# parse_final_answer
# ---------------------------------------------------------------------------


def test_parse_final_answer_canonical() -> None:
    assert parse_final_answer("Reasoning: x\nFinal Answer: 42") == "42"


def test_parse_final_answer_with_trailing_period() -> None:
    assert parse_final_answer("Final Answer: 7.") == "7"


def test_parse_final_answer_with_trailing_paren() -> None:
    assert parse_final_answer("Final Answer: 7)") == "7"


def test_parse_final_answer_case_insensitive() -> None:
    assert parse_final_answer("final answer: 5") == "5"


def test_parse_final_answer_negative_allowed() -> None:
    assert parse_final_answer("Final Answer: -13") == "-13"


def test_parse_final_answer_missing_returns_none() -> None:
    assert parse_final_answer("no answer provided") is None


def test_parse_final_answer_must_be_line_anchored() -> None:
    # "some text Final Answer: 5" does not match: Final Answer must anchor.
    assert parse_final_answer("some text Final Answer: 5") is None


def test_parse_final_answer_handles_empty() -> None:
    assert parse_final_answer("") is None
    assert parse_final_answer(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# parse_candidate_answer (fallback paths)
# ---------------------------------------------------------------------------


def test_parse_candidate_answer_uses_strict_first() -> None:
    assert parse_candidate_answer("Final Answer: 3") == "3"


def test_parse_candidate_answer_answer_line_fallback() -> None:
    assert parse_candidate_answer("Answer: 9") == "9"


def test_parse_candidate_answer_final_answer_is_inline() -> None:
    assert parse_candidate_answer("the final answer is 12") == "12"


def test_parse_candidate_answer_boxed_form() -> None:
    assert parse_candidate_answer("result: \\boxed{41}") == "41"


def test_parse_candidate_answer_falls_back_to_last_number() -> None:
    assert parse_candidate_answer("I think 5 is close but 7 is better") == "7"


def test_parse_candidate_answer_returns_none_when_no_digits() -> None:
    assert parse_candidate_answer("all letters and words only") is None


# ---------------------------------------------------------------------------
# evaluate_reasoning_trace
# ---------------------------------------------------------------------------


def test_evaluate_reasoning_trace_accepts_canonical_correct() -> None:
    task = generate_reasoning_tasks(count=1, seed=5, split="train")[0]
    trace = f"Reasoning: steps\nFinal Answer: {task['reference_answer']}"
    evaluation = evaluate_reasoning_trace(task, trace)
    assert evaluation["accepted"] is True
    assert evaluation["format_ok"] is True
    assert evaluation["correctness_or_judge"] == 1.0
    assert evaluation["policy_compliance"] == 1.0
    assert evaluation["format_reason"] == "ok"


def test_evaluate_reasoning_trace_canonical_but_wrong_answer() -> None:
    task = generate_reasoning_tasks(count=1, seed=5, split="train")[0]
    wrong = str(int(task["reference_answer"]) + 1)
    trace = f"Reasoning: wrong\nFinal Answer: {wrong}"
    evaluation = evaluate_reasoning_trace(task, trace)
    assert evaluation["format_ok"] is True
    assert evaluation["correctness_or_judge"] == 0.0
    assert evaluation["accepted"] is False


def test_evaluate_reasoning_trace_missing_final_answer() -> None:
    task = generate_reasoning_tasks(count=1, seed=5, split="train")[0]
    evaluation = evaluate_reasoning_trace(task, "all prose no answer")
    assert evaluation["format_ok"] is False
    assert evaluation["format_reason"] == "missing_final_answer"


def test_evaluate_reasoning_trace_noncanonical_format_reason() -> None:
    task = generate_reasoning_tasks(count=1, seed=5, split="train")[0]
    trace = f"Answer: {task['reference_answer']}"
    evaluation = evaluate_reasoning_trace(task, trace)
    assert evaluation["format_ok"] is False
    assert evaluation["format_reason"] == "noncanonical_answer_format"


def test_evaluate_reasoning_trace_policy_compliance_band() -> None:
    task = generate_reasoning_tasks(count=1, seed=5, split="train")[0]
    # canonical + reason → 1.0
    assert evaluate_reasoning_trace(task, "Reasoning: x\nFinal Answer: 0")["policy_compliance"] == pytest.approx(
        1.0 if "Reasoning:" else 0.75
    )
    # canonical without "reason"/"step" → 0.75
    assert evaluate_reasoning_trace(task, "Final Answer: 0")["policy_compliance"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# summarize_reasoning_evaluations
# ---------------------------------------------------------------------------


def test_summarize_reasoning_evaluations_basic() -> None:
    evals = [
        {"correctness_or_judge": 1.0, "accepted": True, "format_ok": True, "policy_compliance": 1.0},
        {"correctness_or_judge": 0.0, "accepted": False, "format_ok": True, "policy_compliance": 0.75},
    ]
    summary = summarize_reasoning_evaluations(evals)
    assert summary["eval_count"] == 2
    assert summary["answer_accuracy"] == pytest.approx(0.5)
    assert summary["accepted_rate"] == pytest.approx(0.5)
    assert summary["format_rate"] == pytest.approx(1.0)
    assert summary["mean_policy_compliance"] == pytest.approx((1.0 + 0.75) / 2)
    expected_contract = 0.5 * 0.5 + 0.3 * 0.5 + 0.2 * (1.75 / 2)
    assert summary["contract_score"] == pytest.approx(expected_contract)


def test_summarize_reasoning_evaluations_empty_list_uses_1_denominator() -> None:
    summary = summarize_reasoning_evaluations([])
    assert summary["eval_count"] == 0
    assert summary["answer_accuracy"] == 0.0
    assert summary["accepted_rate"] == 0.0
    assert summary["format_rate"] == 0.0


def test_summarize_reasoning_evaluations_model_ref_passthrough() -> None:
    summary = summarize_reasoning_evaluations([], model_ref="qwen3-4b")
    assert summary["model_ref"] == "qwen3-4b"


# ---------------------------------------------------------------------------
# answer_reasoning_task + build_reasoning_messages + render_reasoning_conversation
# ---------------------------------------------------------------------------


def test_answer_reasoning_task_with_learned_op() -> None:
    task = generate_reasoning_tasks(count=1, seed=0, split="train")[0]
    answer_text = answer_reasoning_task(task, {"supported_operations": [task["operation"]]})
    assert "Final Answer:" in answer_text
    expected = _solve(task["operation"], task["inputs"])
    assert str(expected) in answer_text


def test_answer_reasoning_task_unlearned_op_returns_zero() -> None:
    task = generate_reasoning_tasks(count=1, seed=0, split="train")[0]
    answer_text = answer_reasoning_task(task, {"supported_operations": []})
    assert "Final Answer: 0" in answer_text
    assert "I cannot confidently solve" in answer_text


def test_build_reasoning_messages_structure() -> None:
    messages = build_reasoning_messages("prompt text")
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == REASONING_SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    assert "/no_think" in messages[1]["content"]
    assert "prompt text" in messages[1]["content"]
    assert len(messages) == 2


def test_build_reasoning_messages_with_assistant() -> None:
    messages = build_reasoning_messages("prompt", assistant_text="reply")
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == "reply"
    assert len(messages) == 3


def test_render_reasoning_conversation_fallback_without_tokenizer() -> None:
    rendered = render_reasoning_conversation("prompt", assistant_text="reply")
    assert "System:" in rendered
    assert "/no_think" in rendered
    assert "Assistant: reply" in rendered


def test_render_reasoning_conversation_generation_prompt() -> None:
    rendered = render_reasoning_conversation("prompt", add_generation_prompt=True)
    assert rendered.strip().endswith("Assistant:")
