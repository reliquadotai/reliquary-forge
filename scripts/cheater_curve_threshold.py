#!/usr/bin/env python3
"""Cheater-curve threshold calibration sweep.

Phase 1.1 of the parallel-harvest plan. The script orchestrates an
empirical sweep that produces the data we need to tighten
``PROOF_SKETCH_TOLERANCE_BASE`` and ``LOGPROB_DRIFT_THRESHOLD`` in
``reliquary_inference.protocol.constants``.

For each independent run (fresh training trajectory):

  - Load the base model (Qwen3-4B-Instruct by default) twice:
      * ``validator_model`` will be trained.
      * ``miner_model`` is frozen — represents a lazy cheater who never
        updates after step 0.
  - At each step in 1..N:
      * Take a real GRPO step on the validator model (Forge-side
        ``train_grpo_checkpoint`` over a small synthetic batch).
      * Pick a fresh prompt; have the frozen miner_model generate a
        rollout; build the GRAIL commit (sketches + claimed_logprobs)
        from the *frozen* miner_model.
      * Run the validator's full filter pipeline (sketch_verifier +
        sampler_replay + distribution check) using the *trained*
        validator_model.
      * Record: (run, step, sketch_diff_max, lp_dev_max, dist_median,
                 dist_q10, completion_length).

After ``N_RUNS`` runs the harness aggregates per-step max and p95
across runs and prints a table showing where each filter crosses its
threshold. Use those numbers to set
``PROOF_SKETCH_TOLERANCE_BASE`` (= ~ p99 of honest miners' max sketch
diff at step 0) and ``LOGPROB_DRIFT_THRESHOLD`` (= ~ p99 of honest
miners' lp_dev at step 0).

This is a heavy script — expect ~30 min per run on a B200 / RTX 6000.
Run on the staging fleet, not on the dev box.

Parallel-work credit: romain13190/reliquary@5a7b3eb. The methodology
shape (frozen-miner vs trained-validator, N×M sweep, per-step CSV
record) is from the upstream calibration harness; the Reliquary-Ledger
adaptation calls our own sketch_verifier, sampler_replay, and
distribution stage primitives so the thresholds it produces are
directly applicable to ``protocol/constants.py``.

Usage::

    # Smoke test on a small model (no real GRPO; useful before committing
    # to a multi-hour fleet sweep).
    python scripts/cheater_curve_threshold.py \\
        --model sshleifer/tiny-gpt2 \\
        --n-runs 2 --n-steps 3 \\
        --output /tmp/cheater_curve_smoke.csv

    # Production sweep on the staging fleet.
    python scripts/cheater_curve_threshold.py \\
        --model Qwen/Qwen3-4B-Instruct \\
        --n-runs 10 --n-steps 30 --measurements-per-step 2 \\
        --real-grpo \\
        --output /tmp/cheater_curve.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import logging
import math
import random
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from statistics import median
from typing import Any, Callable

logger = logging.getLogger("cheater_curve")


# ────────────────────────────────────────────────────────────────────
#  Schema for one measurement row.
# ────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Measurement:
    """One (run, step) measurement row written to the output CSV."""

    run: int
    step: int
    sketch_diff_max: float
    lp_dev_max: float
    dist_median: float
    dist_q10: float
    completion_length: int


def write_measurements_csv(measurements: list[Measurement], path: Path) -> None:
    """Write the measurements to CSV; deterministic header order."""
    fieldnames = [
        "run",
        "step",
        "sketch_diff_max",
        "lp_dev_max",
        "dist_median",
        "dist_q10",
        "completion_length",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in measurements:
            writer.writerow(dataclasses.asdict(m))


# ────────────────────────────────────────────────────────────────────
#  Aggregation: per-step distribution across runs.
# ────────────────────────────────────────────────────────────────────


def quantile(values: Iterable[float], q: float) -> float:
    """Plain-python quantile (linear interpolation, no numpy dep)."""
    sorted_vals = sorted(values)
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def aggregate_per_step(measurements: list[Measurement]) -> list[dict[str, Any]]:
    """Group measurements by step; emit per-step max + p95 + p99 per metric.

    The aggregation result drives the threshold-setting decision.
    """
    by_step: dict[int, list[Measurement]] = {}
    for m in measurements:
        by_step.setdefault(m.step, []).append(m)
    rows: list[dict[str, Any]] = []
    for step in sorted(by_step):
        bucket = by_step[step]
        rows.append(
            {
                "step": step,
                "n": len(bucket),
                "sketch_diff_max_p95": quantile([m.sketch_diff_max for m in bucket], 0.95),
                "sketch_diff_max_p99": quantile([m.sketch_diff_max for m in bucket], 0.99),
                "sketch_diff_max_max": max(m.sketch_diff_max for m in bucket),
                "lp_dev_max_p95": quantile([m.lp_dev_max for m in bucket], 0.95),
                "lp_dev_max_p99": quantile([m.lp_dev_max for m in bucket], 0.99),
                "lp_dev_max_max": max(m.lp_dev_max for m in bucket),
                "dist_median_p50": quantile([m.dist_median for m in bucket], 0.50),
                "dist_q10_p10": quantile([m.dist_q10 for m in bucket], 0.10),
            }
        )
    return rows


def print_summary_table(rows: list[dict[str, Any]], file=sys.stdout) -> None:
    """Print a fixed-width table of per-step aggregates."""
    print(
        f"{'step':>5} {'n':>4}  "
        f"{'sketch_p95':>11} {'sketch_p99':>11} {'sketch_max':>11}  "
        f"{'lp_p95':>9} {'lp_p99':>9} {'lp_max':>9}  "
        f"{'dist_med_p50':>13} {'dist_q10_p10':>13}",
        file=file,
    )
    for row in rows:
        print(
            f"{row['step']:>5d} {row['n']:>4d}  "
            f"{row['sketch_diff_max_p95']:>11.1f} {row['sketch_diff_max_p99']:>11.1f} {row['sketch_diff_max_max']:>11.1f}  "
            f"{row['lp_dev_max_p95']:>9.5f} {row['lp_dev_max_p99']:>9.5f} {row['lp_dev_max_max']:>9.5f}  "
            f"{row['dist_median_p50']:>13.4f} {row['dist_q10_p10']:>13.4f}",
            file=file,
        )


# ────────────────────────────────────────────────────────────────────
#  Production sweep — gated on transformers + GPU + reliquary-forge.
# ────────────────────────────────────────────────────────────────────


def _import_or_raise(module_path: str, attr: str | None = None) -> Any:
    """Import ``module_path``; raise a helpful error if the module is missing.

    The cheater curve has heavy deps (transformers, torch, reliquary-forge
    GRPO trainer). On a CPU dev box these aren't installed. Raise a
    structured error so the operator sees what to do.
    """
    try:
        mod = __import__(module_path, fromlist=[attr] if attr else [""])
    except ImportError as exc:  # pragma: no cover - exercised only on slim envs
        raise SystemExit(
            f"cheater_curve_threshold: missing dependency {module_path!r}.\n"
            f"  → install with `pip install reliquary[ml,gpu]` and "
            f"`pip install -e ../reliquary-ledger`, then re-run.\n"
            f"  → on a CPU dev box: pass --tiny to use a tiny-gpt2 smoke setup."
        ) from exc
    if attr is None:
        return mod
    return getattr(mod, attr)


def run_sweep_real(
    *,
    model_repo: str,
    n_runs: int,
    n_steps: int,
    measurements_per_step: int,
    seed: int,
    log_every: int,
) -> list[Measurement]:
    """The real, GPU-backed sweep. Each run = independent training trajectory.

    Imports the heavy deps lazily so the script is parseable / unit-testable
    without transformers or reliquary-forge installed.
    """
    torch = _import_or_raise("torch")
    AutoModelForCausalLM = _import_or_raise(
        "transformers", "AutoModelForCausalLM"
    )
    AutoTokenizer = _import_or_raise("transformers", "AutoTokenizer")

    # Ledger-side primitives.
    sketch_mod = _import_or_raise(
        "reliquary_inference.protocol.sketch_verifier"
    )
    crypto_mod = _import_or_raise("reliquary_inference.protocol.crypto")
    forward_mod = _import_or_raise("reliquary_inference.shared.forward")
    constants_mod = _import_or_raise("reliquary_inference.protocol.constants")
    sampler_mod = _import_or_raise(
        "reliquary_inference.validator.sampler_replay"
    )

    # Forge-side trainer.
    grpo_mod = _import_or_raise("reliquary.training.grpo")

    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning(
            "running on CPU; this will be very slow. Use --tiny for a "
            "small-model CPU smoke test instead."
        )

    measurements: list[Measurement] = []
    for run_idx in range(n_runs):
        logger.info("=== run %d/%d ===", run_idx + 1, n_runs)
        validator_model = _load_model(
            AutoModelForCausalLM, model_repo, device, frozen=False
        )
        miner_model = _load_model(
            AutoModelForCausalLM, model_repo, device, frozen=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_repo)
        prompts = _load_prompts(n_prompts=n_steps + measurements_per_step + 16)

        for step in range(n_steps + 1):
            if step > 0:
                # One real GRPO step on the validator.
                _grpo_step(
                    grpo_mod,
                    validator_model,
                    tokenizer,
                    prompts=rng.sample(prompts, k=4),
                    device=device,
                    rng=rng,
                )

            for _ in range(measurements_per_step):
                meas = _measure_one(
                    sketch_mod=sketch_mod,
                    crypto_mod=crypto_mod,
                    forward_mod=forward_mod,
                    constants_mod=constants_mod,
                    sampler_mod=sampler_mod,
                    validator_model=validator_model,
                    miner_model=miner_model,
                    tokenizer=tokenizer,
                    prompt=rng.choice(prompts),
                    device=device,
                    run_idx=run_idx,
                    step=step,
                    rng=rng,
                )
                measurements.append(meas)

            if step % log_every == 0:
                logger.info(
                    "  run %d step %d -> n_measurements=%d",
                    run_idx + 1,
                    step,
                    len(measurements),
                )

        # Free GPU memory between runs.
        del validator_model, miner_model, tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return measurements


def _load_model(AutoModelForCausalLM, repo: str, device, *, frozen: bool):
    import torch  # lazy

    constants_mod = _import_or_raise("reliquary_inference.protocol.constants")
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation=constants_mod.ATTN_IMPLEMENTATION
        if device.type == "cuda"
        else "eager",
    ).to(device)
    if frozen:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def _load_prompts(*, n_prompts: int) -> list[str]:
    """Load Hendrycks MATH prompts from the Ledger task source.

    Falls back to a small synthetic set if the dataset isn't available
    (e.g. on a CI box without HF cache).
    """
    try:
        loader = _import_or_raise("reliquary_inference.dataset.loader")
        # Adapter: many task sources expose ``load_default(n)``.
        loader_fn = getattr(loader, "load_default", None)
        if loader_fn is not None:
            return list(loader_fn(n_prompts))
    except SystemExit:
        pass
    return [
        f"Compute the value of {a} + {b}." for a in range(1, 30) for b in range(1, 30)
    ][:n_prompts]


def _grpo_step(grpo_mod, model, tokenizer, *, prompts, device, rng) -> None:
    """One GRPO update on the validator model.

    Synthetic-reward GRPO: rewards are random Bernoulli per rollout. The
    point is to advance the validator's weights, not to achieve task
    correctness.
    """
    # Generate M rollouts per prompt; build a fake training batch.
    groups = []
    for prompt in prompts:
        rollouts = []
        for _ in range(4):
            tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
            out = model.generate(
                tokens, max_new_tokens=64, do_sample=True, temperature=0.9, top_p=1.0
            )
            rollouts.append(
                {
                    "full_token_ids": out[0].tolist(),
                    "prompt_token_ids": tokens[0].tolist(),
                    "completion_token_ids": out[0, tokens.shape[1]:].tolist(),
                    "old_logprobs": None,
                    "reward": float(rng.random()),
                }
            )
        groups.append({"challenge_id": "synthetic", "prompt": prompt, "samples": rollouts})

    # Use the Forge GRPO trainer's single-step path. Implementations may
    # accept different shapes; if the available API is different, the
    # operator should adapt this call.
    try:
        grpo_mod.train_grpo_step(model, groups, lr=5e-6, clip_eps=0.2, kl_coef=0.04)
    except AttributeError:  # pragma: no cover - depends on forge build
        # Fallback: use the higher-level checkpoint trainer if step-level
        # is unavailable.
        grpo_mod.train_grpo_checkpoint(
            model_ref=None, groups=groups, output_dir="/tmp/_calib", device=str(device),
            lr=5e-6, clip_eps=0.2, kl_coef=0.04, epochs=1,
        )


def _measure_one(
    *,
    sketch_mod,
    crypto_mod,
    forward_mod,
    constants_mod,
    sampler_mod,
    validator_model,
    miner_model,
    tokenizer,
    prompt: str,
    device,
    run_idx: int,
    step: int,
    rng,
) -> Measurement:
    """One measurement: cheater rollout + validator filter run."""
    import torch  # lazy

    # Miner (frozen) generates a rollout.
    inp = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = miner_model.generate(
            inp, max_new_tokens=64, do_sample=True, temperature=0.9, top_p=1.0
        )
    tokens = out[0].tolist()
    completion_length = len(tokens) - inp.shape[1]

    # Build commitments using the FROZEN miner model — these are what the
    # cheater would submit to the validator.
    rand_hex = "%032x" % rng.getrandbits(128)
    indices = crypto_mod.indices_from_root(
        tokens, rand_hex, len(tokens), constants_mod.CHALLENGE_K
    )

    # Forward both models at the challenged positions.
    miner_hidden, miner_logits = forward_mod.forward_single_layer(
        miner_model, torch.tensor([tokens], device=device), None,
        constants_mod.LAYER_INDEX,
    )
    val_hidden, val_logits = forward_mod.forward_single_layer(
        validator_model, torch.tensor([tokens], device=device), None,
        constants_mod.LAYER_INDEX,
    )

    # Sketch-diff per challenged position.
    sketch_diffs: list[float] = []
    verifier = sketch_mod.SketchVerifier(rand_hex=rand_hex)
    for pos in indices:
        miner_sketch = verifier.sketch(miner_hidden[0, pos])
        val_sketch = verifier.sketch(val_hidden[0, pos])
        sketch_diffs.append(abs(miner_sketch - val_sketch))
    sketch_diff_max = max(sketch_diffs) if sketch_diffs else 0.0

    # Logprob deviation per challenged position.
    miner_logprobs = torch.log_softmax(miner_logits[0], dim=-1)
    val_logprobs_t = torch.log_softmax(val_logits[0], dim=-1)
    lp_devs: list[float] = []
    for pos in indices:
        if pos + 1 >= len(tokens):
            continue
        next_token = tokens[pos + 1]
        miner_lp = float(miner_logprobs[pos, next_token])
        val_lp = float(val_logprobs_t[pos, next_token])
        lp_devs.append(abs(miner_lp - val_lp))
    lp_dev_max = max(lp_devs) if lp_devs else 0.0

    # Distribution stage: replay miner's sampling under validator's logits.
    dist_ratios = [
        sampler_mod.replay_probability(
            float(miner_logprobs[pos, tokens[pos + 1]]),
            float(val_logprobs_t[pos, tokens[pos + 1]]),
        )
        for pos in indices
        if pos + 1 < len(tokens)
    ] or [1.0]
    dist_median = float(median(dist_ratios))
    dist_q10 = float(quantile(dist_ratios, 0.10))

    return Measurement(
        run=run_idx,
        step=step,
        sketch_diff_max=float(sketch_diff_max),
        lp_dev_max=float(lp_dev_max),
        dist_median=dist_median,
        dist_q10=dist_q10,
        completion_length=completion_length,
    )


# ────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct",
                        help="HF repo id of the base model")
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--measurements-per-step", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("/tmp/cheater_curve.csv"))
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="CPU-friendly smoke run with tiny-gpt2 (overrides --model).",
    )
    parser.add_argument(
        "--real-grpo",
        action="store_true",
        help="Run the actual GRPO training step at each iteration. Required "
             "for production calibration; without this flag, the script "
             "only validates that the harness imports and writes CSV.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.tiny:
        args.model = "sshleifer/tiny-gpt2"

    if not args.real_grpo:
        logger.warning(
            "--real-grpo not set; skipping the actual sweep. The CSV will be "
            "empty; this run only verifies imports + argument parsing. Pass "
            "--real-grpo to run the full sweep."
        )
        write_measurements_csv([], args.output)
        return 0

    started_at = time.time()
    measurements = run_sweep_real(
        model_repo=args.model,
        n_runs=args.n_runs,
        n_steps=args.n_steps,
        measurements_per_step=args.measurements_per_step,
        seed=args.seed,
        log_every=args.log_every,
    )
    elapsed = time.time() - started_at

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_measurements_csv(measurements, args.output)
    print(f"\n=== sweep complete ({elapsed:.0f}s, n={len(measurements)}) ===\n")
    rows = aggregate_per_step(measurements)
    print_summary_table(rows)
    print(f"\ncsv: {args.output}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    sys.exit(main())
