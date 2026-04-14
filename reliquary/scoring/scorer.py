from __future__ import annotations

import hashlib
from typing import Any

from ..utils.json_io import sha256_json


def uniqueness_key_for_rollout(rollout_artifact: dict[str, Any]) -> str:
    completion_ids = rollout_artifact["payload"]["generation"]["completion_token_ids"]
    raw = ",".join(str(token) for token in completion_ids).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def uniqueness_key_for_trace(trace_artifact: dict[str, Any]) -> str:
    trace_text = trace_artifact["payload"]["trace"]["trace_text"]
    return hashlib.sha256(trace_text.encode("utf-8")).hexdigest()


def mark_uniqueness(verdicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str], str] = {}
    marked: list[dict[str, Any]] = []
    for verdict in verdicts:
        updated = dict(verdict)
        updated["payload"] = dict(verdict["payload"])
        explanation = dict(updated["payload"].get("verdict_explanation", {}))
        subject_id = (
            verdict["payload"].get("challenge_id")
            or verdict["payload"].get("task_id")
            or verdict["payload"].get("subject_id")
            or "unknown-subject"
        )
        uniqueness_key = verdict["payload"]["uniqueness_key"]
        key = (subject_id, uniqueness_key)
        duplicate_of_artifact_id = seen.get(key)
        unique = duplicate_of_artifact_id is None
        if unique:
            seen[key] = verdict["payload"].get("rollout_artifact_id") or verdict["payload"].get(
                "trace_artifact_id"
            )
        updated["payload"]["is_unique"] = unique
        updated["payload"]["duplicate_of_rollout_id"] = duplicate_of_artifact_id
        updated["payload"]["training_eligible"] = bool(updated["payload"]["accepted"] and unique)
        explanation["uniqueness"] = {
            "ok": unique,
            "key": uniqueness_key,
            "duplicate_of_rollout_id": duplicate_of_artifact_id,
            "reason": "ok" if unique else "duplicate_completion_for_challenge",
        }
        explanation["final_decision"] = {
            "accepted": bool(updated["payload"]["accepted"]),
            "training_eligible": bool(updated["payload"]["training_eligible"]),
            "status": (
                "accepted"
                if updated["payload"]["training_eligible"]
                else "accepted_not_eligible"
                if updated["payload"]["accepted"]
                else "rejected"
            ),
        }
        updated["payload"]["verdict_explanation"] = explanation
        updated["artifact_id"] = sha256_json({k: v for k, v in updated.items() if k != "artifact_id"})
        marked.append(updated)
    return marked


def compute_miner_scores(verdicts: list[dict[str, Any]], exponent: float = 2.0) -> dict[str, Any]:
    totals: dict[str, float] = {}
    miner_summaries: dict[str, dict[str, Any]] = {}
    window_metrics = {
        "rollouts_seen": len(verdicts),
        "accepted_count": 0,
        "training_eligible_count": 0,
        "rejected_count": 0,
        "proof_failures": 0,
        "format_failures": 0,
        "non_unique_count": 0,
        "contamination_failures": 0,
        "judge_failures": 0,
    }
    for verdict in verdicts:
        producer_id = verdict["payload"].get("rollout_producer_id") or verdict["payload"].get(
            "trace_producer_id",
            "unknown-producer",
        )
        summary = miner_summaries.setdefault(
            producer_id,
            {
                "rollouts_seen": 0,
                "accepted_count": 0,
                "training_eligible_count": 0,
                "rejected_count": 0,
                "proof_failures": 0,
                "format_failures": 0,
                "non_unique_count": 0,
                "contamination_failures": 0,
                "judge_failures": 0,
                "total_dense_reward": 0.0,
                "total_score": 0.0,
                "total_weight_score": 0.0,
                "archetype": verdict["payload"].get("rollout_archetype")
                or verdict["payload"].get("trace_archetype", "unknown"),
            },
        )
        summary["rollouts_seen"] += 1
        summary["total_dense_reward"] += float(verdict["payload"].get("dense_reward", 0.0))
        summary["total_score"] += float(verdict["payload"].get("score", 0.0))
        summary["total_weight_score"] += float(
            verdict["payload"].get("weight_score", verdict["payload"].get("dense_reward", 0.0))
        )

        if verdict["payload"]["accepted"]:
            window_metrics["accepted_count"] += 1
            summary["accepted_count"] += 1
        else:
            window_metrics["rejected_count"] += 1
            summary["rejected_count"] += 1

        if not verdict["payload"]["proof_valid"]:
            window_metrics["proof_failures"] += 1
            summary["proof_failures"] += 1

        if not verdict["payload"].get("format_ok", True):
            window_metrics["format_failures"] += 1
            summary["format_failures"] += 1

        if not verdict["payload"].get("is_unique", True):
            window_metrics["non_unique_count"] += 1
            summary["non_unique_count"] += 1

        if not verdict["payload"].get("contamination_ok", True):
            window_metrics["contamination_failures"] += 1
            summary["contamination_failures"] += 1

        if float(verdict["payload"].get("correctness_or_judge", 1.0)) <= 0.0:
            window_metrics["judge_failures"] += 1
            summary["judge_failures"] += 1

        if not verdict["payload"].get("training_eligible", False):
            continue

        window_metrics["training_eligible_count"] += 1
        summary["training_eligible_count"] += 1
        reward = float(verdict["payload"].get("weight_score", verdict["payload"].get("dense_reward", 0.0)))
        totals[producer_id] = totals.get(producer_id, 0.0) + reward

    powered = {producer: score**exponent for producer, score in totals.items()}
    total_power = sum(powered.values()) or 1.0
    weights = {producer: score / total_power for producer, score in powered.items()}
    for producer_id, summary in miner_summaries.items():
        count = max(summary["rollouts_seen"], 1)
        summary["mean_dense_reward"] = summary["total_dense_reward"] / count
        summary["mean_score"] = summary["total_score"] / count
        summary["mean_weight_score"] = summary["total_weight_score"] / count
        summary["weight"] = weights.get(producer_id, 0.0)

    return {
        "totals": totals,
        "weights": weights,
        "miners": miner_summaries,
        "window_metrics": window_metrics,
    }
