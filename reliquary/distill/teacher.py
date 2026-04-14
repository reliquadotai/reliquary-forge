from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any
from urllib import request

from ..envs.reasoning import REASONING_SYSTEM_PROMPT, build_reasoning_messages


def _sign_trace(task_id: str, trace_text: str, final_answer: str, secret: str) -> str:
    material = json.dumps(
        {
            "task_id": task_id,
            "trace_text": trace_text,
            "final_answer": final_answer,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hmac.new(secret.encode("utf-8"), material, hashlib.sha256).hexdigest()


def verify_trace_signature(
    *,
    task_id: str,
    trace_text: str,
    final_answer: str,
    secret: str,
    signature: str,
) -> dict[str, Any]:
    expected = _sign_trace(task_id, trace_text, final_answer, secret)
    valid = hmac.compare_digest(expected, signature)
    return {
        "valid": valid,
        "reason": "ok" if valid else "bad_trace_signature",
    }


def _local_trace_text(task: dict[str, Any], miner_profile: dict[str, Any], sample_index: int) -> tuple[str, str]:
    operation = task["operation"]
    left = task["inputs"]["left"]
    right = task["inputs"]["right"]
    correct_answer = task["reference_answer"]
    strategy = miner_profile.get("strategy", "reasoned_exact")

    if strategy == "reasoned_duplicate":
        trace_text = (
            f"Reasoning: I apply {operation} to the provided inputs and reuse a canonical answer style.\n"
            f"Final Answer: {correct_answer}"
        )
        return trace_text, correct_answer

    if strategy == "reasoned_mixed":
        if sample_index % 3 == 0:
            trace_text = (
                f"Reasoning: I solved the {operation} step carefully.\n"
                f"Final Answer: {correct_answer}"
            )
            return trace_text, correct_answer
        wrong_answer = str(int(correct_answer) + 1)
        trace_text = f"Reasoning: I may have drifted slightly in the arithmetic.\nFinal Answer: {wrong_answer}"
        return trace_text, wrong_answer

    if strategy == "reasoned_contaminated":
        trace_text = (
            "Reasoning: I used an external benchmark leak reference to answer this task.\n"
            f"Final Answer: {correct_answer}"
        )
        return trace_text, correct_answer

    if strategy == "reasoned_malformed":
        trace_text = "I think the pattern is solvable but I am not returning the required final answer field."
        return trace_text, ""

    trace_text = (
        f"Reasoning: I read the task, identified the {operation} operation, "
        f"applied it to {left} and {right}, and verified the result.\n"
        f"Final Answer: {correct_answer}"
    )
    return trace_text, correct_answer


@dataclass
class LocalReasoningTeacherBackend:
    teacher_model: str = "local-reasoning-teacher-v1"

    def generate_trace(
        self,
        *,
        task: dict[str, Any],
        miner_profile: dict[str, Any],
        sample_index: int,
        miner_secret: str,
    ) -> dict[str, Any]:
        trace_text, final_answer = _local_trace_text(task, miner_profile, sample_index)
        signature = _sign_trace(task["task_id"], trace_text, final_answer, miner_secret)
        contamination_source = (
            "benchmark_holdout"
            if miner_profile.get("strategy") == "reasoned_contaminated"
            else None
        )
        return {
            "trace_text": trace_text,
            "final_answer": final_answer,
            "teacher_model": self.teacher_model,
            "teacher_signature": signature,
            "teacher_backend": "local_reasoning_teacher",
            "contamination_source": contamination_source,
            "reasoning_contract": REASONING_SYSTEM_PROMPT,
        }


@dataclass
class VLLMTeacherBackend:
    base_url: str
    model: str
    max_tokens: int
    timeout_seconds: int
    temperature: float = 0.2

    def generate_trace(
        self,
        *,
        task: dict[str, Any],
        miner_profile: dict[str, Any],
        sample_index: int,
        miner_secret: str,
    ) -> dict[str, Any]:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": build_reasoning_messages(task["prompt"]),
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode("utf-8")
        req = request.Request(
            f"{self.base_url.rstrip('/')}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        trace_text = data["choices"][0]["message"]["content"]
        final_answer = trace_text.split("Final Answer:")[-1].strip()
        signature = _sign_trace(task["task_id"], trace_text, final_answer, miner_secret)
        return {
            "trace_text": trace_text,
            "final_answer": final_answer,
            "teacher_model": self.model,
            "teacher_signature": signature,
            "teacher_backend": "vllm",
            "contamination_source": None,
            "reasoning_contract": REASONING_SYSTEM_PROMPT,
        }


def build_teacher_backend(cfg: dict[str, Any]):
    if cfg.get("teacher_backend") == "vllm" and cfg.get("vllm_base_url"):
        return VLLMTeacherBackend(
            base_url=cfg["vllm_base_url"],
            model=cfg.get("vllm_model") or cfg.get("model_teacher") or cfg.get("model_name"),
            max_tokens=int(cfg.get("max_new_tokens", 48)),
            timeout_seconds=int(cfg.get("vllm_timeout_seconds", 120)),
            temperature=float(cfg.get("vllm_temperature", 0.2)),
        )
    return LocalReasoningTeacherBackend(
        teacher_model=cfg.get("model_teacher", "local-reasoning-teacher-v1")
    )
