from __future__ import annotations

import random
import re
from typing import Any


OPS = (
    "add",
    "subtract",
    "multiply",
    "maximum",
    "minimum",
    "absolute_difference",
    "multiply_add",
    "add_then_multiply",
)
PROMPT_TEMPLATES = (
    "Solve the task carefully. Task family: {task_family}. Instruction: {instruction} Provide short reasoning and end with `Final Answer: <value>`.",
    "Arithmetic audit. Follow this instruction: {instruction} Explain briefly, then finish with `Final Answer: <value>`.",
    "Compute the result for this reasoning task. Instruction: {instruction} Return exactly two lines ending with `Final Answer: <value>`.",
    "Work through this compact arithmetic problem. {instruction} Show short reasoning and close with `Final Answer: <value>`.",
)
REASONING_SYSTEM_PROMPT = (
    "Thinking mode is disabled. Respond in exactly two lines:\n"
    "Reasoning: <short reasoning>\n"
    "Final Answer: <integer>"
)


def _task_difficulty(operation: str, left: int, right: int) -> float:
    base = {
        "add": 0.55,
        "subtract": 0.60,
        "multiply": 0.78,
        "maximum": 0.65,
        "minimum": 0.66,
        "absolute_difference": 0.72,
        "multiply_add": 0.88,
        "add_then_multiply": 0.90,
    }.get(operation, 0.50)
    magnitude_bonus = min((abs(left) + abs(right)) / 200.0, 0.20)
    return min(base + magnitude_bonus, 1.0)


def _solve(operation: str, inputs: dict[str, int]) -> int:
    left = int(inputs["left"])
    right = int(inputs["right"])
    extra = int(inputs.get("extra", 0))
    if operation == "add":
        return left + right
    if operation == "subtract":
        return left - right
    if operation == "multiply":
        return left * right
    if operation == "maximum":
        return max(left, right)
    if operation == "minimum":
        return min(left, right)
    if operation == "absolute_difference":
        return abs(left - right)
    if operation == "multiply_add":
        return (left * right) + extra
    if operation == "add_then_multiply":
        return (left + right) * extra
    raise ValueError(f"Unsupported reasoning operation: {operation}")


def _sample_inputs(operation: str, rng: random.Random) -> dict[str, int]:
    if operation in {"add", "subtract", "maximum", "minimum", "absolute_difference"}:
        return {
            "left": rng.randint(11, 99),
            "right": rng.randint(4, 41),
        }
    if operation == "multiply":
        return {
            "left": rng.randint(8, 29),
            "right": rng.randint(3, 17),
        }
    return {
        "left": rng.randint(7, 31),
        "right": rng.randint(3, 19),
        "extra": rng.randint(2, 9),
    }


def _inputs_phrase(inputs: dict[str, int]) -> str:
    values = [str(inputs["left"]), str(inputs["right"])]
    if "extra" in inputs:
        values.append(str(inputs["extra"]))
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{values[0]}, {values[1]}, and {values[2]}"


def _task_instruction(operation: str, inputs: dict[str, int]) -> str:
    left = inputs["left"]
    right = inputs["right"]
    extra = inputs.get("extra")
    if operation == "add":
        return f"Add {left} and {right}."
    if operation == "subtract":
        return f"Subtract {right} from {left}."
    if operation == "multiply":
        return f"Multiply {left} by {right}."
    if operation == "maximum":
        return f"Return the larger of {left} and {right}."
    if operation == "minimum":
        return f"Return the smaller of {left} and {right}."
    if operation == "absolute_difference":
        return f"Compute the absolute difference between {left} and {right}."
    if operation == "multiply_add" and extra is not None:
        return f"First multiply {left} by {right}, then add {extra}."
    if operation == "add_then_multiply" and extra is not None:
        return f"First add {left} and {right}, then multiply the result by {extra}."
    return f"Use the provided inputs { _inputs_phrase(inputs) }."


def generate_reasoning_tasks(
    *,
    count: int,
    seed: int,
    split: str,
    task_family: str = "reasoning_traces",
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []
    for index in range(count):
        operation = OPS[(seed + index) % len(OPS)]
        inputs = _sample_inputs(operation, rng)
        answer = _solve(operation, inputs)
        difficulty = _task_difficulty(operation, inputs["left"], inputs["right"])
        template = PROMPT_TEMPLATES[(seed + index) % len(PROMPT_TEMPLATES)]
        prompt = template.format(
            task_family=task_family,
            operation=operation,
            left=inputs["left"],
            right=inputs["right"],
            input_text=_inputs_phrase(inputs),
            instruction=_task_instruction(operation, inputs),
        )
        tasks.append(
            {
                "task_id": f"reasoning-{split}-{seed}-{index}",
                "task_family": task_family,
                "split": split,
                "seed": seed,
                "prompt": prompt,
                "operation": operation,
                "inputs": inputs,
                "reference_answer": str(answer),
                "difficulty": difficulty,
                "template_id": f"reasoning-template-{(seed + index) % len(PROMPT_TEMPLATES)}",
                "evaluation_policy": {"mode": "exact_final_answer"},
                "contamination_policy": {"forbidden_overlap_tags": ["benchmark_holdout"]},
                "tags": [f"operation:{operation}", f"split:{split}"],
            }
        )
    return tasks


def build_reasoning_messages(prompt: str, assistant_text: str | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": REASONING_SYSTEM_PROMPT},
        {"role": "user", "content": f"/no_think\n{prompt}"},
    ]
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": assistant_text})
    return messages


def render_reasoning_conversation(
    prompt: str,
    *,
    assistant_text: str | None = None,
    tokenizer: Any | None = None,
    add_generation_prompt: bool = False,
) -> str:
    messages = build_reasoning_messages(prompt, assistant_text=assistant_text)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        attempts = (
            {"tokenize": False, "add_generation_prompt": add_generation_prompt, "enable_thinking": False},
            {
                "tokenize": False,
                "add_generation_prompt": add_generation_prompt,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            {"tokenize": False, "add_generation_prompt": add_generation_prompt},
        )
        for kwargs in attempts:
            try:
                rendered = tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError:
                continue
            if isinstance(rendered, str):
                return rendered
    lines = [
        f"System: {REASONING_SYSTEM_PROMPT}",
        f"User: /no_think\n{prompt}",
    ]
    if assistant_text is not None:
        lines.append(f"Assistant: {assistant_text}")
    elif add_generation_prompt:
        lines.append("Assistant:")
    return "\n\n".join(lines)


def parse_final_answer(text: str) -> str | None:
    match = re.search(
        r"(?im)^\s*Final Answer:\s*([-+]?\d+)\s*[\.\)]?\s*$",
        text or "",
    )
    return match.group(1) if match else None


def parse_candidate_answer(text: str) -> str | None:
    strict_answer = parse_final_answer(text)
    if strict_answer is not None:
        return strict_answer
    fallback_patterns = (
        r"(?im)^\s*Answer:\s*([-+]?\d+)\s*[\.\)]?\s*$",
        r"(?i)final answer is\s*([-+]?\d+)",
        r"\\boxed\{([-+]?\d+)\}",
    )
    for pattern in fallback_patterns:
        match = re.search(pattern, text or "")
        if match:
            return match.group(1)
    all_numbers = re.findall(r"[-+]?\d+", text or "")
    return all_numbers[-1] if all_numbers else None


def evaluate_reasoning_trace(task: dict[str, Any], trace_text: str) -> dict[str, Any]:
    strict_final_answer = parse_final_answer(trace_text)
    final_answer = strict_final_answer or parse_candidate_answer(trace_text)
    has_reasoning = "reason" in (trace_text or "").lower() or "step" in (trace_text or "").lower()
    format_ok = strict_final_answer is not None
    if format_ok:
        format_reason = "ok"
    elif final_answer is not None:
        format_reason = "noncanonical_answer_format"
    else:
        format_reason = "missing_final_answer"
    correctness = 1.0 if final_answer == str(task["reference_answer"]) else 0.0
    if format_ok and has_reasoning:
        policy_compliance = 1.0
    elif format_ok:
        policy_compliance = 0.75
    elif final_answer is not None and has_reasoning:
        policy_compliance = 0.50
    elif final_answer is not None:
        policy_compliance = 0.25
    else:
        policy_compliance = 0.0
    accepted = bool(format_ok and correctness > 0.0)
    return {
        "final_answer": final_answer,
        "strict_final_answer": strict_final_answer,
        "format_ok": format_ok,
        "format_reason": format_reason,
        "correctness_or_judge": correctness,
        "policy_compliance": policy_compliance,
        "accepted": accepted,
    }


def summarize_reasoning_evaluations(
    evaluations: list[dict[str, Any]],
    *,
    model_ref: str | None = None,
    supported_operations: list[str] | None = None,
) -> dict[str, Any]:
    total = float(max(len(evaluations), 1))
    answer_accuracy = sum(float(item["correctness_or_judge"]) for item in evaluations) / total
    accepted_rate = sum(1.0 for item in evaluations if item["accepted"]) / total
    format_rate = sum(1.0 for item in evaluations if item["format_ok"]) / total
    mean_policy_compliance = sum(float(item["policy_compliance"]) for item in evaluations) / total
    contract_score = (
        0.5 * answer_accuracy
        + 0.3 * accepted_rate
        + 0.2 * mean_policy_compliance
    )
    summary = {
        "eval_count": len(evaluations),
        "success_rate": answer_accuracy,
        "answer_accuracy": answer_accuracy,
        "accepted_rate": accepted_rate,
        "format_rate": format_rate,
        "mean_policy_compliance": mean_policy_compliance,
        "contract_score": contract_score,
        "supported_operations": supported_operations or [],
    }
    if model_ref is not None:
        summary["model_ref"] = model_ref
    return summary


def answer_reasoning_task(task: dict[str, Any], capabilities: dict[str, Any]) -> str:
    learned_ops = set(capabilities.get("supported_operations", []))
    operation = task["operation"]
    if operation not in learned_ops:
        return "Reasoning: I cannot confidently solve this pattern yet.\nFinal Answer: 0"
    result = _solve(operation, task["inputs"])
    return (
        f"Reasoning: I recognized the {operation} pattern and solved it deterministically.\n"
        f"Final Answer: {result}"
    )
