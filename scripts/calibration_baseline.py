"""Step-0 baseline + synthetic-perturbation calibration sweep.

Phase 1.2 fast-path: instead of running 10×30 real GRPO trajectories
(which costs hours and many GB of intermediate checkpoints), measure
two empirical distributions on the staging fleet's running model:

1. **Honest baseline (step 0)**: miner == validator, identical weights.
   The sketch_diff_max + lp_dev_max here are the noise floor — what
   any honest miner produces when bit-equivalent to the validator.
   This sets the *lower bound* for tolerance: we cannot tighten below.

2. **Synthetic 1-step drift**: add Gaussian perturbation
   (sigma=1e-4 by default) to one transformer block's MLP weight.
   Mimics a "lazy cheater" who's one training step stale. The
   sketch_diff_max + lp_dev_max here are what a real cheater looks
   like; this sets the *upper bound* a tightened tolerance must
   still detect.

Output: empirical p50/p95/p99/max of both distributions across N
prompts. Use the gap between (p99 honest) and (p50 cheater) as the
target tolerance window.

Run on staging2 (RTX 6000B, 96GB) — sufficient for Qwen2.5-3B in
bf16 with room for the perturbation copy.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reliquary_inference.protocol.constants import (
    ATTN_IMPLEMENTATION,
    CHALLENGE_K,
    LAYER_INDEX,
    PROOF_SKETCH_TOLERANCE_BASE,
    PROOF_SKETCH_TOLERANCE_GROWTH,
)
from reliquary_inference.protocol import crypto
from reliquary_inference.protocol import sketch_verifier
from reliquary_inference.shared.forward import forward_single_layer


def quantile(values, q):
    sv = sorted(values)
    if not sv:
        return float("nan")
    if len(sv) == 1:
        return float(sv[0])
    pos = q * (len(sv) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return float(sv[lo])
    return float(sv[lo] + (sv[hi] - sv[lo]) * (pos - lo))


def measure_sketch_lp_diff(
    *,
    model_a,
    model_b,
    tokenizer,
    prompt: str,
    device,
    rng,
    num_completions: int = 1,
    max_new_tokens: int = 64,
) -> tuple[list[float], list[float]]:
    """Generate a completion with model_b (the "miner"), then compare
    sketches + logprobs at challenged positions vs model_a (the
    "validator"). Returns (sketch_diffs, lp_diffs) as lists.
    """
    inp = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model_b.generate(
            inp,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id or 0,
        )
    tokens = out[0].tolist()
    seq_len = len(tokens)
    if seq_len < CHALLENGE_K:
        return [], []  # skip this prompt; sequence too short to challenge
    rand_hex = "%032x" % rng.getrandbits(128)
    indices = crypto.indices_from_root(tokens, rand_hex, seq_len, CHALLENGE_K)

    tok_t = torch.tensor([tokens], device=device)
    with torch.no_grad():
        ha, la = forward_single_layer(model_a, tok_t, None, LAYER_INDEX)
        hb, lb = forward_single_layer(model_b, tok_t, None, LAYER_INDEX)

    # Generate the deterministic r-vector once for this challenge.
    d_model = ha.shape[-1]
    verifier = sketch_verifier.SketchProofVerifier(hidden_dim=d_model)
    r_vec = verifier.generate_r_vec(rand_hex)

    sketch_diffs: list[float] = []
    lp_diffs: list[float] = []

    log_a = torch.log_softmax(la[0], dim=-1)
    log_b = torch.log_softmax(lb[0], dim=-1)

    for pos in indices:
        if pos >= seq_len:
            continue
        sa = verifier.create_commitment(ha[0, pos], r_vec)["sketch"]
        sb = verifier.create_commitment(hb[0, pos], r_vec)["sketch"]
        sketch_diffs.append(abs(int(sa) - int(sb)))
        if pos + 1 < seq_len:
            tok = tokens[pos + 1]
            lp_diffs.append(abs(float(log_a[pos, tok]) - float(log_b[pos, tok])))

    return sketch_diffs, lp_diffs


def make_perturbed_copy(model, *, sigma: float, seed: int):
    """Return a fresh copy of `model` with a small Gaussian
    perturbation added to one MLP weight. Mimics a 1-step cheater.
    """
    import copy

    perturbed = copy.deepcopy(model)
    perturbed.eval()
    for p in perturbed.parameters():
        p.requires_grad_(False)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    # Pick the first MLP-ish weight we can find.
    target = None
    for name, p in perturbed.named_parameters():
        if "mlp" in name and "weight" in name and p.dim() == 2:
            target = (name, p)
            break
    if target is None:
        raise RuntimeError("could not locate an MLP weight to perturb")
    name, p = target
    noise = (
        torch.randn(p.shape, generator=rng, dtype=torch.float32) * sigma
    ).to(p.dtype).to(p.device)
    with torch.no_grad():
        p.add_(noise)
    return perturbed, name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--n-prompts", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--sigma", type=float, default=1e-4,
                        help="Gaussian std-dev applied to MLP weight for 1-step drift.")
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--output", type=Path, default=Path("/tmp/calibration_baseline.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[calibration] device={device} model={args.model}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Note: we deliberately use sdpa (PyTorch built-in) here, not the
    # production ATTN_IMPLEMENTATION (flash_attention_2). Both compared
    # models (model_a vs model_a, then model_a vs model_b) use the same
    # impl so attention-kernel choice doesn't affect the diff
    # measurement. This avoids a flash_attn install just for calibration.
    attn_impl = "sdpa" if device.type == "cuda" else "eager"
    print(f"[calibration] loading honest model A (attn={attn_impl})...", flush=True)
    model_a = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation=attn_impl,
    ).to(device).eval()
    for p in model_a.parameters():
        p.requires_grad_(False)

    rng = random.Random(args.seed)
    prompts = [
        "Compute the value of {} + {}.".format(a, b)
        for a in range(2, 30) for b in range(2, 30)
    ]
    rng.shuffle(prompts)
    prompts = prompts[:args.n_prompts]

    started = time.time()

    # 1. HONEST BASELINE: compare model_a to itself.
    print(f"[calibration] honest baseline ({args.n_prompts} prompts)...", flush=True)
    honest_sketch: list[float] = []
    honest_lp: list[float] = []
    for prompt in prompts:
        s, lp = measure_sketch_lp_diff(
            model_a=model_a, model_b=model_a, tokenizer=tokenizer,
            prompt=prompt, device=device, rng=rng,
            max_new_tokens=args.max_new_tokens,
        )
        honest_sketch.extend(s)
        honest_lp.extend(lp)

    elapsed_honest = time.time() - started

    # 2. SYNTHETIC 1-STEP DRIFT: perturb one MLP weight, then compare model_a (honest)
    #    to model_b (perturbed). model_b plays "cheater" who's 1 step stale.
    print(f"[calibration] perturbing model with sigma={args.sigma}...", flush=True)
    model_b, target_name = make_perturbed_copy(model_a, sigma=args.sigma, seed=args.seed + 1)
    print(f"[calibration] perturbed weight: {target_name}", flush=True)

    print("[calibration] cheater sweep...", flush=True)
    cheater_sketch: list[float] = []
    cheater_lp: list[float] = []
    for prompt in prompts:
        s, lp = measure_sketch_lp_diff(
            model_a=model_a, model_b=model_b, tokenizer=tokenizer,
            prompt=prompt, device=device, rng=rng,
            max_new_tokens=args.max_new_tokens,
        )
        cheater_sketch.extend(s)
        cheater_lp.extend(lp)

    elapsed_total = time.time() - started

    # Aggregate
    summary = {
        "model": args.model,
        "n_prompts": args.n_prompts,
        "max_new_tokens": args.max_new_tokens,
        "perturbation_sigma": args.sigma,
        "perturbed_weight": target_name,
        "elapsed_honest_seconds": round(elapsed_honest, 1),
        "elapsed_total_seconds": round(elapsed_total, 1),
        "current_constants": {
            "PROOF_SKETCH_TOLERANCE_BASE": PROOF_SKETCH_TOLERANCE_BASE,
            "PROOF_SKETCH_TOLERANCE_GROWTH": PROOF_SKETCH_TOLERANCE_GROWTH,
        },
        "honest_sketch_diff": {
            "n": len(honest_sketch),
            "max": float(max(honest_sketch) if honest_sketch else 0.0),
            "p99": quantile(honest_sketch, 0.99),
            "p95": quantile(honest_sketch, 0.95),
            "p50": quantile(honest_sketch, 0.50),
            "median": float(statistics.median(honest_sketch)) if honest_sketch else 0.0,
        },
        "honest_lp_diff": {
            "n": len(honest_lp),
            "max": float(max(honest_lp) if honest_lp else 0.0),
            "p99": quantile(honest_lp, 0.99),
            "p95": quantile(honest_lp, 0.95),
            "p50": quantile(honest_lp, 0.50),
        },
        "cheater_sketch_diff": {
            "n": len(cheater_sketch),
            "max": float(max(cheater_sketch) if cheater_sketch else 0.0),
            "p99": quantile(cheater_sketch, 0.99),
            "p95": quantile(cheater_sketch, 0.95),
            "p50": quantile(cheater_sketch, 0.50),
            "median": float(statistics.median(cheater_sketch)) if cheater_sketch else 0.0,
        },
        "cheater_lp_diff": {
            "n": len(cheater_lp),
            "max": float(max(cheater_lp) if cheater_lp else 0.0),
            "p99": quantile(cheater_lp, 0.99),
            "p95": quantile(cheater_lp, 0.95),
            "p50": quantile(cheater_lp, 0.50),
        },
    }

    # Recommended new constants (with safety factor).
    #
    # Strategy: we operate at the rollout level (not per-position). A
    # rollout is rejected iff MAX(sketch_diff) across the 32 challenged
    # positions exceeds tolerance. So:
    #   floor: max(honest_max) — must accept this with margin
    #   ceil:  cheater_p95 of *position-level* sketch diff — over a 32-
    #          challenge rollout, P(at least one position exceeds tol)
    #          is ~1 - (1 - 0.05)^32 ≈ 80% catch rate
    # Tolerance picked at floor*N + safety_margin where N=4. Must be at
    # least the floor (1 if floor=0) to handle any unobserved drift on
    # untested hardware.
    h_sk = summary["honest_sketch_diff"]
    c_sk = summary["cheater_sketch_diff"]
    h_lp = summary["honest_lp_diff"]
    c_lp = summary["cheater_lp_diff"]
    sketch_floor = max(h_sk["max"], h_sk["p99"])
    sketch_ceil = c_sk["p95"]  # below this catches ~80% of cheater rollouts
    lp_floor = max(h_lp["max"], h_lp["p99"])
    lp_ceil = c_lp["p95"]
    # Safety: tolerance = max(4*floor + epsilon, 100 for sketch, 0.005 for lp).
    suggested_sketch_base = int(max(int(sketch_floor) * 4 + 100, 100))
    if suggested_sketch_base > sketch_ceil and sketch_ceil > 0:
        # If our rule produces a number above the cheater p95, cap at half
        # the cheater p95 — gives margin without over-tightening.
        suggested_sketch_base = max(int(sketch_ceil / 2), 100)
    suggested_lp_threshold = round(max(lp_floor * 4 + 0.005, 0.005), 4)
    if suggested_lp_threshold > lp_ceil and lp_ceil > 0:
        suggested_lp_threshold = round(max(lp_ceil / 2, 0.005), 4)
    summary["recommended"] = {
        "PROOF_SKETCH_TOLERANCE_BASE": suggested_sketch_base,
        "LOGPROB_DRIFT_THRESHOLD": suggested_lp_threshold,
        "sketch_floor_honest_max": sketch_floor,
        "sketch_ceil_cheater_p95": sketch_ceil,
        "lp_floor_honest_max": lp_floor,
        "lp_ceil_cheater_p95": lp_ceil,
        "rule": "max(4 * honest_max + 100, 100); cap at cheater_p95/2 to ensure cheater detection",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\n[calibration] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
