from reliquary.distill.teacher import LocalReasoningTeacherBackend, verify_trace_signature
from reliquary.envs.reasoning import (
    evaluate_reasoning_trace,
    generate_reasoning_tasks,
    render_reasoning_conversation,
)


def test_reasoning_task_generation_and_trace_eval() -> None:
    task = generate_reasoning_tasks(count=1, seed=42, split="train")[0]
    backend = LocalReasoningTeacherBackend()
    trace = backend.generate_trace(
        task=task,
        miner_profile={"strategy": "reasoned_exact"},
        sample_index=0,
        miner_secret="secret",
    )
    provenance = verify_trace_signature(
        task_id=task["task_id"],
        trace_text=trace["trace_text"],
        final_answer=trace["final_answer"],
        secret="secret",
        signature=trace["teacher_signature"],
    )
    evaluation = evaluate_reasoning_trace(task, trace["trace_text"])
    assert provenance["valid"] is True
    assert evaluation["accepted"] is True
    assert evaluation["correctness_or_judge"] == 1.0


def test_reasoning_eval_tracks_noncanonical_correct_answers() -> None:
    task = generate_reasoning_tasks(count=1, seed=7, split="holdout")[0]
    trace_text = f"The correct value is {task['reference_answer']}"
    evaluation = evaluate_reasoning_trace(task, trace_text)
    assert evaluation["correctness_or_judge"] == 1.0
    assert evaluation["format_ok"] is False
    assert evaluation["accepted"] is False
    assert evaluation["format_reason"] == "noncanonical_answer_format"


def test_reasoning_conversation_fallback_renders_contract() -> None:
    task = generate_reasoning_tasks(count=1, seed=11, split="train")[0]
    rendered = render_reasoning_conversation(
        task["prompt"],
        assistant_text="Reasoning: Test\nFinal Answer: 1",
    )
    assert "System:" in rendered
    assert "/no_think" in rendered
    assert "Final Answer:" in rendered
