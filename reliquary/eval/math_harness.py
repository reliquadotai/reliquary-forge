"""Forge eval harness — held-out MATH benchmark for the live policy.

Purpose:
  Runs the deterministic MATH holdout against the current Forge policy
  and publishes a signed :class:`~reliquary_protocol.EvalBundle` every
  cycle. The signed time-series of accuracy numbers turns
  "the flywheel runs" into "the flywheel measurably improves the
  model" — the single piece of evidence that lets third parties
  verify Reliquary's core product claim.

Design:
  - Holdout is derived via :mod:`reliquary_protocol.eval_holdout` so
    Ledger and Forge pick up byte-identical indices without any
    shared state.
  - Problems are loaded through :mod:`reliquary_inference.dataset.math_holdout`
    (soft-imported; Forge must have reliquary-inference co-installed).
  - Scoring is reused from
    :mod:`reliquary_inference.dataset.task_sources.math_env.evaluate_math_trace`
    so the eval's definition of "correct" matches the mesh validators'
    definition byte-for-byte.
  - Sampling defaults to greedy (T=0, n=1) for reproducibility —
    every eval on the same policy produces the same number. Pass@k
    mode with sampling is an opt-in.
  - Results → :class:`EvalBundle` via ``build_eval_bundle`` →
    ``sign_envelope(EVAL_BUNDLE_TYPE, ...)`` → published to R2 under
    ``eval_bundles/<training_netuid>/<eval_window_id>/<run_id>.json``.

The harness accepts ``model`` + ``tokenizer`` as pre-loaded inputs so a
caller with a hot model (e.g. directly after a GRPO cycle) can pass
them in without re-loading. A thin wrapper script handles the
from-scratch case.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from reliquary_protocol import (
    DEFAULT_HOLDOUT_SEED,
    EVAL_BUNDLE_TYPE,
    HOLDOUT_LABEL_VERSION,
    BridgeEnvelope,
    BridgeSigner,
    EvalBundle,
    build_eval_bundle,
    sign_envelope,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from reliquary_protocol.storage import R2ObjectBackend

logger = logging.getLogger(__name__)


# The system prompt the miner sees on Ledger — mirrored here so the
# eval and mining environment agree on the answer-format contract.
# Changing this without Ledger agreement invalidates eval comparability
# against live rollouts.
EVAL_SYSTEM_PROMPT: str = (
    "You are a mathematics expert. Solve the problem step by step. "
    "Show your reasoning, then give the final answer enclosed in "
    "\\boxed{...}. The reward depends on the boxed answer matching "
    "the ground truth after LaTeX normalization, so only the content "
    "inside the FINAL \\boxed{...} is graded."
)


@dataclass(frozen=True)
class EvalCycleResult:
    """Return value of :func:`run_eval_cycle`.

    The bundle + envelope are exposed separately so the caller can
    decide what to persist: the bundle alone is the content-addressable
    artifact; the envelope is the signed transport form that was
    actually written to R2.
    """

    bundle: EvalBundle
    envelope: BridgeEnvelope
    storage_key: str
    duration_seconds: float


def render_eval_prompt(problem_text: str, tokenizer: Any) -> str:
    """Wrap a MATH problem in the model's chat template + eval system prompt.

    Matches the renderer used on the miner side
    (:func:`reliquary_inference.dataset.task_sources._render_math_conversation`)
    so the eval prompt byte-form is identical to what the mining path
    already produced for this problem — if the mining path were to
    hit this problem at any point, the token sequence would match.
    """
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return f"{EVAL_SYSTEM_PROMPT}\n\nProblem: {problem_text}\n\nSolution:"
    try:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": problem_text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:  # pragma: no cover - tokenizer fallback
        return f"{EVAL_SYSTEM_PROMPT}\n\nProblem: {problem_text}\n\nSolution:"


def run_eval_cycle(
    *,
    model: Any,
    tokenizer: Any,
    backend: "R2ObjectBackend | None",
    signer: BridgeSigner,
    training_netuid: int,
    inference_netuid: int,
    model_ref: str,
    eval_window_id: int,
    policy_artifact_id: str,
    policy_merkle_root_hex: str,
    max_new_tokens: int = 1024,
    max_problems: int | None = None,
    sampling_mode: str = "greedy",
    temperature: float = 0.7,
    top_p: float = 1.0,
    n_samples_per_problem: int = 1,
    evaluator_hotkey: str = "reliquary-forge-evaluator-v1",
    eval_run_id: str | None = None,
    holdout_seed: str = DEFAULT_HOLDOUT_SEED,
    publish: bool = True,
) -> EvalCycleResult:
    """Run the MATH holdout against ``model`` and publish an EvalBundle.

    Parameters
    ----------
    model, tokenizer
        Pre-loaded HF model + tokenizer. Caller owns their lifecycle.
    backend
        R2 backend to publish the signed envelope into. ``None`` to
        skip the publish step (local smoke test).
    signer
        :class:`BridgeSigner` used to sign the envelope. Must match
        the signer allowlisted by the Ledger policy consumer.
    training_netuid, inference_netuid
        Usually equal under the single-subnet model, but kept separate
        for parity with :class:`CheckpointAttestation`.
    model_ref
        HuggingFace model identifier (e.g. ``"Qwen/Qwen2.5-3B-Instruct"``);
        recorded in the bundle so third parties know which model weights
        were evaluated.
    eval_window_id
        Ledger window at which the eval was run.
    policy_artifact_id, policy_merkle_root_hex
        Provenance tags — the Ledger policy artifact id (if any) and
        the Forge delta merkle (if this policy was produced by an
        already-published GRPO cycle). Pass empty strings to mark
        the base model with no applied deltas.
    max_problems
        Optional cap on the number of holdout problems evaluated this
        cycle. ``None`` runs the full 500-problem slice. Caps < 500
        still produce valid bundles but flag the eval as partial via
        the ``n_problems`` field.
    sampling_mode
        ``"greedy"`` (T=0, n=1, reproducible pass@1) or
        ``"temperature"`` (T>0, pass@k-style; callers tune ``temperature``
        / ``top_p`` / ``n_samples_per_problem``).
    eval_run_id
        Defaults to ``f"forge-eval-{int(time.time())}"``.
    publish
        When ``False``, skips the R2 put. Still builds + signs the
        envelope so callers can inspect / archive locally.
    """
    # Soft-import the holdout loader + scorer so reliquary can install
    # without reliquary-inference in environments that don't need eval
    # (CPU CI without datasets, for instance). A production Forge node
    # always has both installed.
    try:
        from reliquary_inference.dataset.math_holdout import (
            HOLDOUT_DATASET,
            HOLDOUT_DATASET_SIZE,
            load_holdout_problems,
        )
        from reliquary_inference.dataset.task_sources.math_env import (
            evaluate_math_trace,
        )
    except ImportError as exc:  # pragma: no cover - hard-miss path
        raise RuntimeError(
            "reliquary.eval requires reliquary-inference to be installed "
            "(for the MATH holdout + scoring functions). Install with "
            "`uv pip install -e /path/to/reliquary-inference`."
        ) from exc

    if eval_run_id is None:
        eval_run_id = f"forge-eval-{int(time.time())}"

    if sampling_mode not in ("greedy", "temperature"):
        raise ValueError(
            f"sampling_mode must be 'greedy' or 'temperature'; got {sampling_mode!r}"
        )

    started = time.perf_counter()
    problems = load_holdout_problems(
        dataset_size=HOLDOUT_DATASET_SIZE,
        seed=holdout_seed,
    )
    if max_problems is not None:
        problems = problems[: int(max_problems)]

    holdout_task_ids = [p["problem_id"] for p in problems]
    logger.info(
        "forge.eval.run_eval_cycle start n=%d model=%s window=%d policy_id=%s",
        len(problems),
        model_ref,
        eval_window_id,
        policy_artifact_id[:12] if policy_artifact_id else "<base>",
    )

    # Run generation loop — per-problem (no batching in v1; batched
    # inference is a follow-up optimization once single-problem latency
    # is profiled).
    import torch  # type: ignore[import-untyped]

    device = next(model.parameters()).device

    per_problem_results: list[dict[str, Any]] = []
    for problem in problems:
        prompt = render_eval_prompt(problem["prompt"], tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = int(inputs["input_ids"].shape[1])

        with torch.no_grad():
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": int(max_new_tokens),
                "pad_token_id": tokenizer.pad_token_id,
            }
            if sampling_mode == "greedy":
                generate_kwargs["do_sample"] = False
            else:
                generate_kwargs.update(
                    {
                        "do_sample": True,
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                        "num_return_sequences": int(n_samples_per_problem),
                    }
                )
            outputs = model.generate(**inputs, **generate_kwargs)

        # For pass@k mode we take "any correct" across samples as the
        # per-problem reward; greedy collapses to 1 sample.
        sample_rewards: list[float] = []
        sample_answers: list[str] = []
        for seq in outputs:
            completion = tokenizer.decode(
                seq[prompt_length:], skip_special_tokens=True
            )
            scored = evaluate_math_trace(
                {"reference_answer": problem["reference_answer"]},
                completion,
            )
            sample_rewards.append(float(scored.get("correctness_or_judge", 0.0)))
            sample_answers.append(str(scored.get("final_answer", "")))

        # Per-problem correctness = 1.0 iff any sample is correct
        # (pass@k semantics). For n=1 greedy, this is just the sample's
        # reward.
        problem_correctness = max(sample_rewards) if sample_rewards else 0.0
        final_answer = sample_answers[0] if sample_answers else ""

        per_problem_results.append(
            {
                "task_id": problem["problem_id"],
                "dataset_index": int(problem["dataset_index"]),
                "level": problem["level"],
                "subject": problem["subject"],
                "correctness": float(problem_correctness),
                "format_ok": problem_correctness >= 1.0
                or any(s != "" for s in sample_answers),
                "final_answer": final_answer,
            }
        )

    bundle = build_eval_bundle(
        training_netuid=int(training_netuid),
        inference_netuid=int(inference_netuid),
        eval_run_id=str(eval_run_id),
        eval_window_id=int(eval_window_id),
        policy_artifact_id=str(policy_artifact_id),
        policy_merkle_root_hex=str(policy_merkle_root_hex),
        model_ref=str(model_ref),
        holdout_version=HOLDOUT_LABEL_VERSION,
        holdout_seed=holdout_seed,
        holdout_dataset=HOLDOUT_DATASET,
        holdout_size=len(problems),
        holdout_task_ids=holdout_task_ids,
        n_samples_per_problem=int(n_samples_per_problem),
        sampling_mode=sampling_mode,
        per_problem_results=per_problem_results,
        evaluator_hotkey=evaluator_hotkey,
    )

    envelope = sign_envelope(EVAL_BUNDLE_TYPE, bundle, signer)
    storage_key = bundle.storage_key()

    if publish:
        if backend is None:
            raise ValueError("publish=True requires a non-None backend")
        backend.put(storage_key, envelope.canonical_bytes())
        logger.info(
            "forge.eval.run_eval_cycle published key=%s accuracy=%.4f n=%d",
            storage_key,
            bundle.accuracy,
            bundle.n_problems,
        )

    duration = time.perf_counter() - started
    return EvalCycleResult(
        bundle=bundle,
        envelope=envelope,
        storage_key=storage_key,
        duration_seconds=duration,
    )


def aggregate_summary(bundle: EvalBundle) -> dict[str, Any]:
    """Return a compact summary dict for logs / dashboards.

    Excludes ``per_problem_results`` + ``holdout_task_ids`` (full lists
    are for the R2 bundle, not for log lines) but keeps the top-line
    numbers plus the policy hash so the dashboard can build an accuracy
    time-series.
    """
    return {
        "eval_run_id": bundle.eval_run_id,
        "eval_window_id": int(bundle.eval_window_id),
        "policy_artifact_id": bundle.policy_artifact_id,
        "policy_merkle_root_hex": bundle.policy_merkle_root_hex,
        "model_ref": bundle.model_ref,
        "holdout_version": bundle.holdout_version,
        "holdout_size": int(bundle.holdout_size),
        "n_problems": int(bundle.n_problems),
        "sampling_mode": bundle.sampling_mode,
        "n_samples_per_problem": int(bundle.n_samples_per_problem),
        "accuracy": float(bundle.accuracy),
        "mean_reward": float(bundle.mean_reward),
        "accuracy_by_level": dict(bundle.accuracy_by_level),
        "accuracy_by_subject": dict(bundle.accuracy_by_subject),
        "evaluator_hotkey": bundle.evaluator_hotkey,
        "evaluated_at": bundle.evaluated_at,
    }


__all__ = [
    "EVAL_SYSTEM_PROMPT",
    "EvalCycleResult",
    "aggregate_summary",
    "render_eval_prompt",
    "run_eval_cycle",
]
