"""Forge eval cycle — held-out MATH benchmark for the live policy.

Runs periodically (systemd timer :file:`reliquary-forge-eval.timer`) to
produce the time-series of accuracy numbers that proves GRPO deltas
actually move the model. Each invocation:

1. Loads the current Qwen2.5-3B policy from HF cache (same model_ref
   the GRPO trainer uses).
2. Resolves the current Ledger window via the subtensor (so the
   EvalBundle can be joined to the exact rollout windows around it).
3. Runs the deterministic MATH holdout (500 problems by default; cap
   with ``--max-problems`` for faster cycles).
4. Signs + publishes the resulting EvalBundle to R2 under
   ``eval_bundles/<netuid>/<window>/<run_id>.json``.

The published bundle carries the full per-problem result list so
third parties can re-run the holdout on their own hardware and verify
the accuracy number bit-for-bit (modulo model-weight download).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("HF_HOME", "/opt/reliquary-state/hf-cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/opt/reliquary-state/hf-cache/hub")

import bittensor as bt  # noqa: E402
import torch  # noqa: E402
from reliquary_protocol.storage import R2ObjectBackend  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from reliquary.eval.math_harness import aggregate_summary, run_eval_cycle  # noqa: E402
from reliquary.training.policy_attestation import make_hmac_authority  # noqa: E402

# Match the GRPO trainer's model + netuid pins so eval and training
# operate on the same policy surface.
MODEL_REF_DEFAULT = "Qwen/Qwen2.5-3B-Instruct"
INFERENCE_NETUID_DEFAULT = 462
TRAINING_NETUID_DEFAULT = 462
POLICY_AUTHORITY_HOTKEY_DEFAULT = "reliquary-policy-authority-v1"
EVALUATOR_HOTKEY_DEFAULT = "reliquary-forge-evaluator-v1"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one Forge eval cycle against the MATH holdout."
    )
    parser.add_argument("--model-ref", default=MODEL_REF_DEFAULT)
    parser.add_argument("--inference-netuid", type=int, default=INFERENCE_NETUID_DEFAULT)
    parser.add_argument("--training-netuid", type=int, default=TRAINING_NETUID_DEFAULT)
    parser.add_argument(
        "--policy-authority-hotkey", default=POLICY_AUTHORITY_HOTKEY_DEFAULT
    )
    parser.add_argument("--evaluator-hotkey", default=EVALUATOR_HOTKEY_DEFAULT)
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help=(
            "Cap the number of holdout problems evaluated this cycle. "
            "Defaults to the full 500 (~15-25 min on Blackwell)."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help=(
            "Per-problem generation budget. 1024 matches miner MAX_NEW_TOKENS."
        ),
    )
    parser.add_argument(
        "--sampling-mode",
        choices=("greedy", "temperature"),
        default="greedy",
        help=(
            "Greedy gives reproducible pass@1 (same number every cycle "
            "on same policy). Temperature gives pass@k-style sampling."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--n-samples-per-problem",
        type=int,
        default=1,
        help="Pass@k sample count. Ignored in greedy mode.",
    )
    parser.add_argument(
        "--policy-artifact-id",
        default="",
        help=(
            "Optional: the Ledger policy artifact id the evaluated model "
            "was published under. Empty string = base model / unattributed."
        ),
    )
    parser.add_argument(
        "--policy-merkle-root-hex",
        default="",
        help=(
            "Optional: the Forge DeltaBundle merkle root that this "
            "policy represents. Empty string = base model."
        ),
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Build + sign but skip R2 publish (local smoke test).",
    )
    parser.add_argument(
        "--r2-token-path",
        default="/opt/reliquary-secrets/r2-cf-api-token",
        help="Path to the R2 CF API token file.",
    )
    parser.add_argument(
        "--signing-secret-path",
        default="/opt/reliquary-secrets/signing-secret",
        help="Path to the HMAC signing secret for the policy authority.",
    )
    parser.add_argument(
        "--subtensor-network",
        default="test",
        help="Bittensor subtensor network ('test' / 'finney').",
    )
    args = parser.parse_args()

    # Resolve current Ledger window via the subtensor so the eval
    # bundle carries the exact window id the eval ran under. Windows
    # are 30 blocks each (6 min at 12s block time), matching the
    # cadence used in GRPO scripts.
    log(f"resolving current window via subtensor.{args.subtensor_network}")
    subt = bt.subtensor(args.subtensor_network)
    current_block = subt.get_current_block()
    current_window = (current_block // 30) * 30
    log(f"  current_block={current_block} current_window={current_window}")

    log(f"loading model {args.model_ref}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_ref)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_ref, torch_dtype=torch.bfloat16,
    ).cuda()
    model.eval()

    if args.no_publish:
        backend = None
        log("  publish disabled (--no-publish)")
    else:
        r2_token = open(args.r2_token_path).read().strip()
        backend = R2ObjectBackend(
            account_id="d5332aea7e3780d0f2391a4e4f6ddfbc",
            bucket="reliquary",
            cf_api_token=r2_token,
            public_url="https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev",
        )

    signing_secret = open(args.signing_secret_path).read().strip()
    signer = make_hmac_authority(
        args.policy_authority_hotkey, signing_secret
    )

    log(
        f"running eval: max_problems={args.max_problems or 'all'} "
        f"sampling={args.sampling_mode} max_new_tokens={args.max_new_tokens}"
    )
    result = run_eval_cycle(
        model=model,
        tokenizer=tokenizer,
        backend=backend,
        signer=signer,
        training_netuid=args.training_netuid,
        inference_netuid=args.inference_netuid,
        model_ref=args.model_ref,
        eval_window_id=current_window,
        policy_artifact_id=args.policy_artifact_id,
        policy_merkle_root_hex=args.policy_merkle_root_hex,
        max_new_tokens=args.max_new_tokens,
        max_problems=args.max_problems,
        sampling_mode=args.sampling_mode,
        temperature=args.temperature,
        top_p=args.top_p,
        n_samples_per_problem=args.n_samples_per_problem,
        evaluator_hotkey=args.evaluator_hotkey,
        publish=not args.no_publish,
    )

    summary = aggregate_summary(result.bundle)
    log(
        f"eval done: accuracy={summary['accuracy']:.4f} "
        f"n_problems={summary['n_problems']} "
        f"mean_reward={summary['mean_reward']:.4f} "
        f"duration={result.duration_seconds:.1f}s"
    )
    for level, acc in sorted(summary["accuracy_by_level"].items()):
        log(f"  {level}: {acc:.4f}")
    if not args.no_publish:
        log(f"published: s3://reliquary/{result.storage_key}")
    else:
        log(f"(skipped publish) would-be key: {result.storage_key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
