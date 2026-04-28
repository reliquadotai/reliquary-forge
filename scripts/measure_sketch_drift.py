#!/usr/bin/env python3
"""Sketch-drift measurement between two HF revisions.

Phase 1.1 companion to ``cheater_curve_threshold.py``. Where the sweep
generates synthetic training trajectories, this script measures the
real drift between two checkpoints already published to the HF Hub
trainer repo (e.g. ``checkpoint_n=10`` vs ``checkpoint_n=20``). The
output is the empirical envelope a miner running ``checkpoint_n``
needs the validator to tolerate against ``checkpoint_n+k``.

Use it to:

  * Verify that a tightened ``PROOF_SKETCH_TOLERANCE_BASE`` still
    accommodates legitimate ~k-step drift between two recent
    checkpoints.
  * Detect mid-deployment drift anomalies (e.g. a checkpoint published
    with a different attention impl than the rest).

Output: CSV with one row per challenged position, plus a stdout
summary showing max + p99 + median sketch diff over a fixed prompt
set.

Parallel-work credit: romain13190/reliquary@013763e. Same goal, our
APIs (sketch_verifier, indices_from_root, forward_single_layer).

Usage::

    python scripts/measure_sketch_drift.py \\
        --base-model Qwen/Qwen3-4B-Instruct \\
        --rev-a 1234abc \\
        --rev-b 5678def \\
        --n-prompts 32 \\
        --output /tmp/sketch_drift.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import sys
from pathlib import Path
from statistics import median
from typing import Any

logger = logging.getLogger("sketch_drift")


@dataclasses.dataclass
class DriftRow:
    prompt_idx: int
    position: int
    sketch_a: int
    sketch_b: int
    diff: int


def write_drift_csv(rows: list[DriftRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["prompt_idx", "position", "sketch_a", "sketch_b", "diff"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(dataclasses.asdict(r))


def quantile(values, q):
    sv = sorted(values)
    if not sv:
        return float("nan")
    if len(sv) == 1:
        return float(sv[0])
    pos = q * (len(sv) - 1)
    import math

    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sv[lo])
    return float(sv[lo] + (sv[hi] - sv[lo]) * (pos - lo))


def summarize_drift(rows: list[DriftRow]) -> dict[str, float]:
    if not rows:
        return {"n": 0, "max": 0.0, "p99": 0.0, "p95": 0.0, "median": 0.0}
    diffs = [abs(r.diff) for r in rows]
    return {
        "n": len(diffs),
        "max": float(max(diffs)),
        "p99": quantile(diffs, 0.99),
        "p95": quantile(diffs, 0.95),
        "median": float(median(diffs)),
    }


def _import_or_raise(path, attr=None):
    try:
        m = __import__(path, fromlist=[attr] if attr else [""])
    except ImportError as exc:
        raise SystemExit(
            f"measure_sketch_drift: missing dep {path!r}. "
            f"Install reliquary[ml,gpu] + reliquary-ledger. ({exc})"
        ) from exc
    return getattr(m, attr) if attr else m


def measure(
    *,
    base_model: str,
    rev_a: str,
    rev_b: str,
    n_prompts: int,
    seed: int,
    output_csv: Path,
) -> dict[str, float]:
    """Load two revisions of ``base_model``, run them on ``n_prompts``
    fixed prompts, record per-position sketch values, return summary.

    The two revisions must share tokenizer + LAYER_INDEX. If they don't
    (e.g. different vocab sizes), the script aborts with a clear error.
    """
    torch = _import_or_raise("torch")
    AutoModelForCausalLM = _import_or_raise("transformers", "AutoModelForCausalLM")
    AutoTokenizer = _import_or_raise("transformers", "AutoTokenizer")
    sketch_mod = _import_or_raise("reliquary_inference.protocol.sketch_verifier")
    crypto_mod = _import_or_raise("reliquary_inference.protocol.crypto")
    forward_mod = _import_or_raise("reliquary_inference.shared.forward")
    constants_mod = _import_or_raise("reliquary_inference.protocol.constants")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def load(rev: str):
        return AutoModelForCausalLM.from_pretrained(
            base_model,
            revision=rev,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            attn_implementation=constants_mod.ATTN_IMPLEMENTATION
            if device.type == "cuda"
            else "eager",
        ).to(device).eval()

    logger.info("loading rev_a=%s ...", rev_a)
    model_a = load(rev_a)
    logger.info("loading rev_b=%s ...", rev_b)
    model_b = load(rev_b)
    if model_a.config.vocab_size != model_b.config.vocab_size:
        raise SystemExit(
            f"vocab mismatch between revisions: a={model_a.config.vocab_size}, "
            f"b={model_b.config.vocab_size}; cannot compare sketches."
        )

    import random
    rng = random.Random(seed)
    prompts = [
        f"Compute the value of {a} + {b}." for a in range(1, 30) for b in range(1, 30)
    ][:n_prompts]
    rows: list[DriftRow] = []
    for prompt_idx, prompt in enumerate(prompts):
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        rand_hex = "%032x" % rng.getrandbits(128)
        seq_len = tokens.shape[1]
        indices = crypto_mod.indices_from_root(
            tokens[0].tolist(), rand_hex, seq_len, constants_mod.CHALLENGE_K
        )
        with torch.no_grad():
            ha, _ = forward_mod.forward_single_layer(model_a, tokens, None, constants_mod.LAYER_INDEX)
            hb, _ = forward_mod.forward_single_layer(model_b, tokens, None, constants_mod.LAYER_INDEX)
        verifier = sketch_mod.SketchVerifier(rand_hex=rand_hex)
        for pos in indices:
            if pos >= seq_len:
                continue
            sa = int(verifier.sketch(ha[0, pos]))
            sb = int(verifier.sketch(hb[0, pos]))
            rows.append(DriftRow(prompt_idx, int(pos), sa, sb, sa - sb))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_drift_csv(rows, output_csv)
    summary = summarize_drift(rows)
    print(
        f"sketch drift across {summary['n']} positions:\n"
        f"  max     = {summary['max']:>10.1f}\n"
        f"  p99     = {summary['p99']:>10.1f}\n"
        f"  p95     = {summary['p95']:>10.1f}\n"
        f"  median  = {summary['median']:>10.1f}\n"
        f"\n  csv: {output_csv}"
    )
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct")
    p.add_argument("--rev-a", required=True)
    p.add_argument("--rev-b", required=True)
    p.add_argument("--n-prompts", type=int, default=32)
    p.add_argument("--seed", type=int, default=20260428)
    p.add_argument("--output", type=Path, default=Path("/tmp/sketch_drift.csv"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    measure(
        base_model=args.base_model,
        rev_a=args.rev_a,
        rev_b=args.rev_b,
        n_prompts=args.n_prompts,
        seed=args.seed,
        output_csv=args.output,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    sys.exit(main())
