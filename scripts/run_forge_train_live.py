"""Real Forge cycle — SFT-LoRA training on accumulated rollouts, publishes
a semantically-meaningful delta into the live closed-loop bridge.

What this does:
  1. Pull every recent (prompt, completion) pair from R2 where the
     validator mesh marked the completion accepted + correctness=1.0.
     Completions come from completion_bundles/; prompts via task_batches/
     joined on task_id.
  2. Load Qwen2.5-3B-Instruct base weights. Snapshot the target-tensor
     parameters (attention q/k/v projections in the first few layers)
     before training — we need these as the "base" for compute_delta.
  3. Attach a LoRA adapter (rank 8, alpha 16) to q_proj/k_proj/v_proj.
     Train ~20 steps with AdamW at 2e-4 over the collected pairs.
  4. Merge the LoRA adapter into the base weights (peft.merge_and_unload).
     Now the post-training weights live directly in the model's
     state_dict, and we can diff vs the pre-training snapshot.
  5. Build DeltaBundle over target tensors, publish_bundle + sign +
     publish_attestation + PolicyCommitment at effective_at_ledger_window
     = current_window + 2 windows.

Deterministic seeds + tight step count keep the delta compact (~few MB
quantized int8) so it applies fast in the miner/validator hot-swap
hooks. Semantic direction: nudges Q/K/V toward the reasoning-format
completions the mesh has been accepting, so subsequent mining should
produce more-consistent "Reasoning: X \\nFinal Answer: Y" outputs.
"""

from __future__ import annotations

import gzip
import json
import os
import random
import time
from collections import defaultdict

os.environ.setdefault("HF_HOME", "/opt/reliquary-state/hf-cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/opt/reliquary-state/hf-cache/hub")

import torch
import bittensor as bt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from reliquary.training.checkpoint_storage import publish_bundle
from reliquary.training.delta_checkpoints import compute_delta
from reliquary.training.policy_attestation import (
    make_hmac_authority,
    publish_attestation,
)
from reliquary_protocol.storage import R2ObjectBackend


MODEL_REF = "Qwen/Qwen2.5-3B-Instruct"
INFERENCE_NETUID = 462
TRAINING_NETUID = 462
POLICY_AUTHORITY_HOTKEY = "reliquary-policy-authority-v1"

# Only train on post-v5-cutover data (pre-cutover had all weights=0 anyway)
MIN_LIVE_WINDOW = 6956700

# LoRA target — attention projections, first 4 layers only to keep the
# delta compact + iteration time tight.
LORA_TARGET_LAYERS = [0, 1, 2, 3]
LORA_TARGET_PROJS = ("q_proj", "k_proj", "v_proj")
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0

TRAIN_STEPS = 20
TRAIN_LR = 2e-4
TRAIN_BATCH = 1  # Qwen 3B bf16 fits easily but per-batch full-sequence fwd+bwd is expensive
MAX_SEQ_LEN = 256

RUN_ID = f"forge-sft-{int(time.time())}"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------


def pull_training_pairs(backend: R2ObjectBackend) -> list[tuple[str, str]]:
    """Fetch + join task_batches (for prompts) with verdict_bundles + completion_bundles
    (for accepted completions). Returns a list of (prompt, completion) tuples."""

    # 1. Task batches → {task_id: prompt}
    tb_keys = backend.list("task_batches/")
    task_id_to_prompt: dict[str, str] = {}
    task_id_to_reference: dict[str, str] = {}
    tb_checked = 0
    for key in tb_keys:
        raw = backend.get(key)
        if raw is None:
            continue
        try:
            tb = json.loads(raw.decode())
        except Exception:
            continue
        tb_checked += 1
        window_id = int(tb["payload"].get("window_id", 0))
        if window_id < MIN_LIVE_WINDOW:
            continue
        for task in tb["payload"].get("tasks", []):
            tid = task.get("task_id")
            prompt = task.get("prompt", "")
            if tid and prompt:
                task_id_to_prompt[tid] = prompt
                task_id_to_reference[tid] = task.get("reference_answer", "")
    log(f"  task_batches scanned={tb_checked} prompts_indexed={len(task_id_to_prompt)}")

    # 2. Verdict bundles → {completion_id: accepted}
    vb_keys = [k for k in backend.list("verdict_bundles/") if k.endswith(".json.gz")]
    accepted_completion_ids: set[str] = set()
    for key in vb_keys:
        try:
            window_id = int(key.split("/")[1].split("-")[1])
        except (IndexError, ValueError):
            continue
        if window_id < MIN_LIVE_WINDOW:
            continue
        raw = backend.get(key)
        if raw is None:
            continue
        try:
            verdicts = json.loads(gzip.decompress(raw).decode())
        except Exception:
            continue
        for v in verdicts:
            p = v["payload"]
            if p.get("accepted") and float(p.get("correctness", 0)) >= 0.99:
                accepted_completion_ids.add(p["completion_id"])
    log(f"  verdict_bundles scanned={len(vb_keys)} accepted={len(accepted_completion_ids)}")

    # 3. Completion bundles → (prompt, completion) pairs where accepted
    cb_keys = [k for k in backend.list("completion_bundles/") if k.endswith(".json.gz")]
    pairs: list[tuple[str, str]] = []
    for key in cb_keys:
        try:
            window_id = int(key.split("/")[1].split("-")[1])
        except (IndexError, ValueError):
            continue
        if window_id < MIN_LIVE_WINDOW:
            continue
        raw = backend.get(key)
        if raw is None:
            continue
        try:
            comps = json.loads(gzip.decompress(raw).decode())
        except Exception:
            continue
        for c in comps:
            if c["artifact_id"] not in accepted_completion_ids:
                continue
            task_id = c["payload"].get("task_id")
            prompt = task_id_to_prompt.get(task_id)
            completion = c["payload"].get("completion_text", "")
            if not prompt or not completion:
                continue
            pairs.append((prompt, completion))
    log(f"  completion_bundles scanned={len(cb_keys)} training_pairs={len(pairs)}")
    return pairs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def make_target_tensor_names() -> list[str]:
    """Return the flat names of target-layer q/k/v projections."""
    return [
        f"model.layers.{layer}.self_attn.{proj}.weight"
        for layer in LORA_TARGET_LAYERS
        for proj in LORA_TARGET_PROJS
    ]


def run_sft_lora(
    model, tokenizer, pairs: list[tuple[str, str]]
) -> None:
    """Attach LoRA, train TRAIN_STEPS steps, merge back into base."""
    device = next(model.parameters()).device

    target_modules = list(LORA_TARGET_PROJS)  # PEFT matches by module suffix
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=target_modules,
        layers_to_transform=LORA_TARGET_LAYERS,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        (p for p in peft_model.parameters() if p.requires_grad),
        lr=TRAIN_LR,
    )

    peft_model.train()
    random.seed(0)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for step in range(TRAIN_STEPS):
        batch = random.sample(pairs, min(TRAIN_BATCH, len(pairs)))
        losses = []
        for prompt, completion in batch:
            full = prompt + completion + tokenizer.eos_token
            enc = tokenizer(
                full,
                return_tensors="pt",
                max_length=MAX_SEQ_LEN,
                truncation=True,
            ).to(device)

            # Mask loss to only the completion tokens (not the prompt)
            prompt_enc = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=MAX_SEQ_LEN,
                truncation=True,
            )
            prompt_len = int(prompt_enc["input_ids"].size(1))
            labels = enc["input_ids"].clone()
            labels[0, :prompt_len] = -100  # ignore prompt loss
            out = peft_model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=labels,
            )
            losses.append(out.loss)

        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log(f"  step {step+1}/{TRAIN_STEPS} loss={loss.item():.4f}")

    peft_model.eval()
    # Merge LoRA into base weights in-place, drop adapter.
    merged = peft_model.merge_and_unload()
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    r2_token = open("/opt/reliquary-secrets/r2-cf-api-token").read().strip()
    signing_secret = open("/opt/reliquary-secrets/signing-secret").read().strip()
    backend = R2ObjectBackend(
        account_id="d5332aea7e3780d0f2391a4e4f6ddfbc",
        bucket="reliquary",
        cf_api_token=r2_token,
        public_url="https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev",
    )

    log(f"building training set from R2 (window >= {MIN_LIVE_WINDOW})")
    pairs = pull_training_pairs(backend)
    if len(pairs) < 5:
        log(f"FATAL: only {len(pairs)} training pairs available — need more live windows")
        return 2
    log(f"  training set: {len(pairs)} pairs")
    # Sample one
    sp, sc = pairs[0]
    log(f"  sample prompt (first 80): {sp[:80]!r}")
    log(f"  sample completion (first 80): {sc[:80]!r}")

    log(f"loading {MODEL_REF}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REF)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REF,
        torch_dtype=torch.bfloat16,
    ).cuda()
    model.eval()

    target_names = make_target_tensor_names()
    log(f"  target tensors (pre-snapshot): {len(target_names)} ({target_names[:3]}...)")

    pre_params = {
        n: p.detach().clone().cpu().float()
        for n, p in model.named_parameters()
        if n in target_names
    }
    assert len(pre_params) == len(target_names), \
        f"missing target tensors in model: have {set(pre_params)} ; want {set(target_names)}"
    log(f"  snapshotted {len(pre_params)} tensors pre-training")

    log(f"SFT-LoRA training: {TRAIN_STEPS} steps, rank={LORA_RANK}, alpha={LORA_ALPHA}")
    merged = run_sft_lora(model, tokenizer, pairs)

    post_params = {
        n: p.detach().clone().cpu().float()
        for n, p in merged.named_parameters()
        if n in target_names
    }
    assert len(post_params) == len(target_names)

    # Sanity: show a delta magnitude
    sample_name = target_names[0]
    d = (post_params[sample_name] - pre_params[sample_name]).abs().mean().item()
    log(f"  mean abs delta on {sample_name}: {d:.6f}")

    log("computing delta bundle")
    bundle = compute_delta(pre_params, post_params)
    log(f"  shards: {len(bundle.shards)}")
    log(f"  merkle_root: {bundle.merkle_root_hex}")
    log(f"  total delta bytes: {sum(len(s.data_bytes) for s in bundle.shards)}")

    subt = bt.subtensor("test")
    current_block = subt.get_current_block()
    current_window = (current_block // 30) * 30
    effective_at = current_window + 60

    log(f"publishing delta bundle to checkpoints/{RUN_ID}/{current_window}/")
    manifest = publish_bundle(
        bundle, run_id=RUN_ID, window_id=current_window, backend=backend
    )

    log("publishing CheckpointAttestation + PolicyCommitment")
    signer = make_hmac_authority(POLICY_AUTHORITY_HOTKEY, signing_secret)
    pub = publish_attestation(
        backend=backend,
        signer=signer,
        training_netuid=TRAINING_NETUID,
        inference_netuid=INFERENCE_NETUID,
        checkpoint_manifest=manifest,
        base_snapshot_window_id=0,
        consumed_rollout_windows=[],  # future: pass the actual trained-on windows
        consumed_rollout_keys=[],
        effective_at_ledger_window=effective_at,
    )
    log(f"  attestation_key: {pub.attestation_key}")
    log(f"  commitment_key:  {pub.commitment_key}")
    log(f"  smoke_hash:      {pub.smoke_hash_hex}")
    log(f"  effective_at_ledger_window: {effective_at} (current: {current_window})")

    print("SUMMARY:", flush=True)
    print(json.dumps({
        "run_id": RUN_ID,
        "training_pairs": len(pairs),
        "training_steps": TRAIN_STEPS,
        "target_tensor_count": len(target_names),
        "delta_bytes": sum(len(s.data_bytes) for s in bundle.shards),
        "merkle_root_hex": bundle.merkle_root_hex,
        "effective_at_ledger_window": effective_at,
        "current_window": current_window,
    }, indent=2), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
