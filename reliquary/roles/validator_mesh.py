from __future__ import annotations

from typing import Any

from ..distill.teacher import verify_trace_signature
from ..envs.reasoning import evaluate_reasoning_trace
from ..envs.sat import evaluate_sat_response
from ..protocol.artifacts import make_artifact
from ..scoring.scorer import (
    compute_miner_scores,
    mark_uniqueness,
    uniqueness_key_for_rollout,
    uniqueness_key_for_trace,
)


def _sat_difficulty(challenge: dict[str, Any]) -> float:
    extra = challenge.get("extra", {})
    n = max(int(extra.get("n", 1)), 1)
    m = max(int(extra.get("m", 1)), 1)
    ratio = m / n
    return min(0.35 + ratio / 10.0, 1.0)


def _rl_verdict_explanation(
    proof_result: dict[str, Any],
    evaluation: dict[str, Any],
    accepted: bool,
) -> dict[str, Any]:
    if accepted:
        summary = "accepted_for_scoring"
    elif not proof_result["valid"]:
        summary = f"rejected:{proof_result['reason']}"
    else:
        summary = f"rejected:{evaluation['format_reason']}"
    return {
        "summary": summary,
        "proof": {"ok": bool(proof_result["valid"]), "reason": proof_result["reason"]},
        "format": {
            "ok": bool(evaluation["format_ok"]),
            "reason": evaluation["format_reason"],
            "assignment_count": evaluation["assignment_count"],
            "expected_assignment_count": evaluation["expected_assignment_count"],
            "missing_variables": evaluation["missing_variables"],
            "out_of_range_variables": evaluation["out_of_range_variables"],
        },
        "reward": {
            "score": float(evaluation["score"]),
            "dense_reward": float(evaluation["dense_reward"]),
            "satisfied_fraction": float(evaluation["satisfied_fraction"]),
        },
        "uniqueness": {"ok": None, "key": None, "duplicate_of_rollout_id": None, "reason": "pending"},
        "final_decision": {
            "accepted": bool(accepted),
            "training_eligible": None,
            "status": "accepted" if accepted else "rejected",
        },
    }


def _distill_verdict_explanation(
    *,
    provenance_result: dict[str, Any],
    evaluation: dict[str, Any],
    contamination_ok: bool,
    accepted: bool,
) -> dict[str, Any]:
    if accepted:
        summary = "accepted_for_packing"
    elif not provenance_result["valid"]:
        summary = f"rejected:{provenance_result['reason']}"
    elif not contamination_ok:
        summary = "rejected:contaminated_trace"
    else:
        summary = f"rejected:{evaluation['format_reason']}"
    return {
        "summary": summary,
        "provenance": {"ok": provenance_result["valid"], "reason": provenance_result["reason"]},
        "format": {"ok": evaluation["format_ok"], "reason": evaluation["format_reason"]},
        "judge": {
            "correctness_or_judge": float(evaluation["correctness_or_judge"]),
            "policy_compliance": float(evaluation["policy_compliance"]),
        },
        "contamination": {
            "ok": contamination_ok,
            "reason": "ok" if contamination_ok else "benchmark_overlap",
        },
        "uniqueness": {"ok": None, "key": None, "duplicate_of_rollout_id": None, "reason": "pending"},
        "final_decision": {
            "accepted": bool(accepted),
            "training_eligible": None,
            "status": "accepted" if accepted else "rejected",
        },
    }


def _apply_rl_scores(verdicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updated_verdicts: list[dict[str, Any]] = []
    for verdict in mark_uniqueness(verdicts):
        updated = dict(verdict)
        updated["payload"] = dict(verdict["payload"])
        explanation = dict(updated["payload"]["verdict_explanation"])
        novelty = 1.0 if updated["payload"].get("is_unique", False) else 0.0
        difficulty = float(updated["payload"]["difficulty"])
        dense_reward = float(updated["payload"]["dense_reward"])
        weight_score = (
            0.70 * dense_reward
            + 0.15 * novelty
            + 0.15 * difficulty
        ) if updated["payload"].get("training_eligible", False) else 0.0
        updated["payload"]["novelty"] = novelty
        updated["payload"]["weight_score"] = weight_score
        explanation["reward"]["novelty"] = novelty
        explanation["reward"]["difficulty"] = difficulty
        explanation["reward"]["weight_score"] = weight_score
        updated["payload"]["verdict_explanation"] = explanation
        updated_verdicts.append(updated)
    return updated_verdicts


def _apply_distill_scores(verdicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updated_verdicts: list[dict[str, Any]] = []
    for verdict in mark_uniqueness(verdicts):
        updated = dict(verdict)
        updated["payload"] = dict(verdict["payload"])
        explanation = dict(updated["payload"]["verdict_explanation"])
        novelty = 1.0 if updated["payload"].get("is_unique", False) else 0.0
        weight_score = (
            0.40 * float(updated["payload"]["correctness_or_judge"])
            + 0.20 * novelty
            + 0.20 * float(updated["payload"]["difficulty"])
            + 0.20 * float(updated["payload"]["policy_compliance"])
        ) if updated["payload"].get("training_eligible", False) else 0.0
        updated["payload"]["novelty"] = novelty
        updated["payload"]["weight_score"] = weight_score
        explanation["judge"]["novelty"] = novelty
        explanation["judge"]["difficulty"] = float(updated["payload"]["difficulty"])
        explanation["judge"]["weight_score"] = weight_score
        updated["payload"]["verdict_explanation"] = explanation
        updated_verdicts.append(updated)
    return updated_verdicts


def validate_rl_window(
    cfg: dict[str, Any],
    policy_artifact: dict[str, Any],
    rollout_artifacts: list[dict[str, Any]],
    *,
    window_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from ..protocol.proof import verify_proof
    from ..utils.modeling import load_model_bundle

    bundle = load_model_bundle(policy_artifact["payload"]["model_ref"], cfg["device"])
    verdicts: list[dict[str, Any]] = []
    miner_secrets = cfg.get("miner_secrets", {})
    for rollout in rollout_artifacts:
        generation = rollout["payload"]["generation"]
        challenge = rollout["payload"]["challenge"]
        proof = rollout["payload"]["proof"]
        miner_secret = miner_secrets.get(rollout["producer_id"], cfg["miner_secret"])
        proof_result = verify_proof(
            model=bundle["model"],
            full_token_ids=generation["sequence_token_ids"],
            proof=proof,
            miner_secret=miner_secret,
        )
        evaluation = evaluate_sat_response(challenge, generation["completion_text"])
        accepted = bool(proof_result["valid"] and evaluation["accepted"])
        verdict = make_artifact(
            artifact_type="verdict",
            producer_id=cfg["validator_id"],
            producer_role="validator_mesh",
            policy_id=policy_artifact["artifact_id"],
            window_id=window_id,
            lane="rl",
            commodity_use="rl_only",
            parent_ids=[policy_artifact["artifact_id"], rollout["artifact_id"]],
            payload={
                "challenge_id": challenge["challenge_id"],
                "rollout_artifact_id": rollout["artifact_id"],
                "rollout_producer_id": rollout["producer_id"],
                "rollout_archetype": rollout["payload"].get("rollout_archetype", "unknown"),
                "proof_valid": proof_result["valid"],
                "proof_reason": proof_result["reason"],
                "score": evaluation["score"],
                "dense_reward": evaluation["dense_reward"],
                "satisfied_fraction": evaluation["satisfied_fraction"],
                "difficulty": _sat_difficulty(challenge),
                "format_ok": evaluation["format_ok"],
                "format_reason": evaluation["format_reason"],
                "assignment_count": evaluation["assignment_count"],
                "expected_assignment_count": evaluation["expected_assignment_count"],
                "missing_variables": evaluation["missing_variables"],
                "out_of_range_variables": evaluation["out_of_range_variables"],
                "accepted": accepted,
                "uniqueness_key": uniqueness_key_for_rollout(rollout),
                "challenge": challenge,
                "generation": generation,
                "miner_profile": rollout["payload"].get("miner_profile", {}),
                "verdict_explanation": _rl_verdict_explanation(proof_result, evaluation, accepted),
            },
        )
        verdicts.append(verdict)
    scored_verdicts = _apply_rl_scores(verdicts)
    scorecard = make_artifact(
        artifact_type="scorecard",
        producer_id=cfg["validator_id"],
        producer_role="validator_mesh",
        policy_id=policy_artifact["artifact_id"],
        window_id=window_id,
        lane="rl",
        commodity_use="rl_only",
        parent_ids=[policy_artifact["artifact_id"], *[item["artifact_id"] for item in scored_verdicts]],
        payload=compute_miner_scores(scored_verdicts),
    )
    return scored_verdicts, scorecard


def validate_distill_window(
    cfg: dict[str, Any],
    policy_artifact: dict[str, Any],
    trace_artifacts: list[dict[str, Any]],
    *,
    window_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    verdicts: list[dict[str, Any]] = []
    miner_secrets = cfg.get("miner_secrets", {})
    for trace_artifact in trace_artifacts:
        task = trace_artifact["payload"]["task"]
        trace = trace_artifact["payload"]["trace"]
        miner_secret = miner_secrets.get(trace_artifact["producer_id"], cfg["miner_secret"])
        provenance_result = verify_trace_signature(
            task_id=task["task_id"],
            trace_text=trace["trace_text"],
            final_answer=trace["final_answer"],
            secret=miner_secret,
            signature=trace["teacher_signature"],
        )
        evaluation = evaluate_reasoning_trace(task, trace["trace_text"])
        contamination_ok = trace.get("contamination_source") is None
        accepted = bool(
            provenance_result["valid"]
            and evaluation["accepted"]
            and contamination_ok
        )
        verdict = make_artifact(
            artifact_type="verdict",
            producer_id=cfg["validator_id"],
            producer_role="validator_mesh",
            policy_id=policy_artifact["artifact_id"],
            window_id=window_id,
            lane="distill",
            commodity_use="dual_use",
            parent_ids=[policy_artifact["artifact_id"], trace_artifact["artifact_id"]],
            payload={
                "task_id": task["task_id"],
                "trace_artifact_id": trace_artifact["artifact_id"],
                "trace_producer_id": trace_artifact["producer_id"],
                "trace_archetype": trace_artifact["payload"].get("trace_archetype", "unknown"),
                "provenance_valid": provenance_result["valid"],
                "proof_valid": provenance_result["valid"],
                "proof_reason": provenance_result["reason"],
                "correctness_or_judge": evaluation["correctness_or_judge"],
                "policy_compliance": evaluation["policy_compliance"],
                "difficulty": float(task["difficulty"]),
                "format_ok": evaluation["format_ok"],
                "format_reason": evaluation["format_reason"],
                "accepted": accepted,
                "contamination_ok": contamination_ok,
                "contamination_reason": "ok" if contamination_ok else "benchmark_overlap",
                "uniqueness_key": uniqueness_key_for_trace(trace_artifact),
                "task": task,
                "trace": trace,
                "score": evaluation["correctness_or_judge"],
                "dense_reward": evaluation["correctness_or_judge"],
                "miner_profile": trace_artifact["payload"].get("miner_profile", {}),
                "verdict_explanation": _distill_verdict_explanation(
                    provenance_result=provenance_result,
                    evaluation=evaluation,
                    contamination_ok=contamination_ok,
                    accepted=accepted,
                ),
            },
        )
        verdicts.append(verdict)
    scored_verdicts = _apply_distill_scores(verdicts)
    scorecard = make_artifact(
        artifact_type="scorecard",
        producer_id=cfg["validator_id"],
        producer_role="validator_mesh",
        policy_id=policy_artifact["artifact_id"],
        window_id=window_id,
        lane="distill",
        commodity_use="dual_use",
        parent_ids=[policy_artifact["artifact_id"], *[item["artifact_id"] for item in scored_verdicts]],
        payload=compute_miner_scores(scored_verdicts),
    )
    return scored_verdicts, scorecard
