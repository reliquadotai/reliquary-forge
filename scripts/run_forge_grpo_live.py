"""Forge GRPO cycle — DAPO-style group-relative policy optimization on
zone-filtered rollout groups from the live mesh, publishing a real RL
delta into the closed-loop bridge.

Why this replaces SFT:
  - SFT on only-correct completions teaches the model to imitate already-
    correct behaviour. It can't push toward any completion the model has
    never produced, and it's wasted compute on prompts where every rollout
    is already correct (no learning signal anyway).
  - GRPO feeds on CONTRAST: within a group of M rollouts on the same prompt,
    correct rollouts get positive advantage, incorrect get negative. The
    policy shifts toward the direction of the correct ones relative to its
    own average. DAPO's zone filter σ ≥ 0.43 ensures every group we train
    on has real contrast (i.e. 2-6 correct out of 8 for binary rewards).

Pipeline:
  1. Pull zone-filtered rollout groups from R2. A group is ``(miner_id,
     task_id)`` whose ``scorecard.window_metrics.zone_filter.groups`` entry
     has ``in_zone == True``. Each group contributes up to M=8 rollouts —
     prompt, completion text, token_logprobs (π_old), correctness.
  2. Load base model + clone a frozen reference for KL. Snapshot the
     target tensors before training so we can diff post-training.
  3. For each window's GRPO step:
       advantage_i = (reward_i - mean_group) / std_group   # within-group
       L_ppo      = -min(r_i A_i, clip(r_i, 1±ε) A_i)       # PPO-clipped
       L_kl       = exp(log_pi_ref - log_pi_new) - 1
                    - (log_pi_ref - log_pi_new)             # k3 estimator
       L          = mean(L_ppo + β·L_kl)                    # β = 0.04
     One optimiser step per train_step call; LR follows linear-warmup +
     cosine (same schedule as SFT run so the delta magnitude stays in
     the band Forge's checkpoint-delta pipeline handles well).
  4. Merge post-training weights back; compute_delta vs the pre-training
     snapshot; publish bundle + attestation + PolicyCommitment — the
     same bridge the SFT trainer uses, so the Ledger hot-swap hooks are
     unchanged.

Algorithm lineage: GRPO from DeepSeek-Math; zone filter / cooldown from
ByteDance DAPO (March 2025). Concrete code adapted from romain13190/
reliquary (MIT) — core loss math preserved, I/O layer redone against
our R2 artifact shape.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

os.environ.setdefault("HF_HOME", "/opt/reliquary-state/hf-cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/opt/reliquary-state/hf-cache/hub")

import torch
import torch.nn.functional as F
import bittensor as bt
from transformers import AutoModelForCausalLM, AutoTokenizer

from reliquary.training.checkpoint_storage import publish_bundle
from reliquary.training.delta_checkpoints import compute_delta
from reliquary.training.policy_attestation import (
    make_hmac_authority,
    publish_attestation,
)
from reliquary_protocol.storage import R2ObjectBackend


# ---------------------------------------------------------------------------
# Config — GRPO canonical hyperparameters + Qwen2.5-3B model details
# ---------------------------------------------------------------------------

MODEL_REF = "Qwen/Qwen2.5-3B-Instruct"
INFERENCE_NETUID = 462
TRAINING_NETUID = 462
POLICY_AUTHORITY_HOTKEY = "reliquary-policy-authority-v1"

# Post-v5-cutover data only.
MIN_LIVE_WINDOW = 6956700

# Target tensors: same as SFT cycle — attention q/k/v in the first 4
# layers. Keeps delta compact + hot-swap fast.
TARGET_LAYERS = [0, 1, 2, 3]
TARGET_PROJS = ("q_proj", "k_proj", "v_proj")

# GRPO optimisation (DAPO / DeepSeek GRPO canonical settings).
LEARNING_RATE = 5e-7       # small — RL fine-tuning on pretrained LLMs collapses at higher LR.
PPO_CLIP_EPSILON = 0.2     # standard PPO clip.
KL_BETA = 0.04             # DeepSeek GRPO default.
GRAD_CLIP_NORM = 1.0
MAX_SEQ_LEN = 512

# Training batch = B_BATCH distinct groups per optimiser step.
B_BATCH = 8
M_ROLLOUTS = 8

RUN_ID = f"forge-grpo-{int(time.time())}"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data — pull zone-filtered groups from R2
# ---------------------------------------------------------------------------


@dataclass
class Rollout:
    completion_text: str
    prompt: str
    correctness: float
    tokens: list[int] = field(default_factory=list)
    prompt_length: int = 0
    token_logprobs: list[float] = field(default_factory=list)


@dataclass
class RolloutGroup:
    miner_id: str
    task_id: str
    window_id: int
    sigma: float
    mean_reward: float
    rollouts: list[Rollout] = field(default_factory=list)


def pull_training_groups(
    backend: R2ObjectBackend,
    *,
    min_window: int,
) -> tuple[list[RolloutGroup], list[int], list[str]]:
    """Fetch ``(miner, task)`` rollout groups that passed the validator's
    zone filter (σ ≥ 0.43 for binary rewards).

    We walk every scorecard.payload.window_metrics.zone_filter.groups entry
    where ``in_zone == True``, then join to the matching task prompt (from
    task_batches) and completion rollouts (from completion_bundles).

    Returns ``(groups, consumed_windows, consumed_keys)`` — same shape
    as the SFT pull_training_pairs helper so the attestation step can
    use it without modification.
    """
    # 1. Task batches → {task_id: (prompt, window_id)}
    tb_keys = backend.list("task_batches/")
    task_index: dict[str, tuple[str, int]] = {}
    tb_errors = 0
    for key in tb_keys:
        try:
            raw = backend.get(key)
        except Exception:
            tb_errors += 1
            continue
        if raw is None:
            continue
        try:
            tb = json.loads(raw.decode())
        except Exception:
            continue
        window_id = int(tb["payload"].get("window_id", 0))
        if window_id < min_window:
            continue
        for task in tb["payload"].get("tasks", []):
            tid = task.get("task_id")
            prompt = task.get("prompt", "")
            if tid and prompt:
                task_index[tid] = (prompt, window_id)
    log(
        f"  task_batches scanned={len(tb_keys)} skipped_errors={tb_errors} "
        f"prompts_indexed={len(task_index)}"
    )

    # 2. Scorecards → set of (miner_id, task_id) pairs flagged in_zone.
    # Under heavy mesh load a single scorecard GET can 429 despite the
    # R2 backend's retry loop; don't crash the whole cycle on one bad
    # fetch — count skips separately and proceed with what we got. The
    # next cycle will retry the missed ones (R2 keys are stable).
    sc_keys = backend.list("scorecards/")
    in_zone_keys: set[tuple[str, str]] = set()
    consumed_windows: set[int] = set()
    sc_errors = 0
    for key in sc_keys:
        try:
            raw = backend.get(key)
        except Exception as exc:
            sc_errors += 1
            if sc_errors <= 3:
                log(f"  scorecard fetch error ({sc_errors}/?): {key} → {type(exc).__name__}")
            continue
        if raw is None:
            continue
        try:
            sc = json.loads(raw.decode())
        except Exception:
            continue
        window_id = int(sc["payload"].get("window_id", 0))
        if window_id < min_window:
            continue
        zone = sc["payload"].get("window_metrics", {}).get("zone_filter") or {}
        for group in zone.get("groups", []):
            if not group.get("in_zone"):
                continue
            miner = str(group.get("miner_id") or "")
            task_id = str(group.get("task_id") or "")
            if miner and task_id:
                in_zone_keys.add((miner, task_id))
                consumed_windows.add(window_id)
    log(
        f"  scorecards scanned={len(sc_keys)} skipped_errors={sc_errors} "
        f"in_zone_groups={len(in_zone_keys)} consumed_windows={len(consumed_windows)}"
    )

    # 3. Completion bundles → rollouts, grouped by (miner, task) where in-zone
    cb_keys = [k for k in backend.list("completion_bundles/") if k.endswith(".json.gz")]
    bucket: dict[tuple[str, str], list[Rollout]] = defaultdict(list)
    group_window: dict[tuple[str, str], int] = {}
    consumed_keys: list[str] = []
    cb_errors = 0
    for key in cb_keys:
        try:
            window_id = int(key.split("/")[1].split("-")[1])
        except (IndexError, ValueError):
            continue
        if window_id < min_window:
            continue
        try:
            raw = backend.get(key)
        except Exception:
            cb_errors += 1
            continue
        if raw is None:
            continue
        try:
            comps = json.loads(gzip.decompress(raw).decode())
        except Exception:
            continue
        used_any = False
        for c in comps:
            payload = c["payload"]
            miner = str(c.get("producer_id") or payload.get("miner_id") or "")
            task_id = str(payload.get("task_id") or "")
            if (miner, task_id) not in in_zone_keys:
                continue
            prompt, _ = task_index.get(task_id, ("", 0))
            if not prompt:
                continue
            rollout = Rollout(
                completion_text=str(payload.get("completion_text", "")),
                prompt=prompt,
                correctness=float(payload.get("correctness", 0.0)),
                tokens=list(payload.get("tokens") or []),
                prompt_length=int(payload.get("prompt_length") or 0),
                token_logprobs=list(payload.get("token_logprobs") or []),
            )
            bucket[(miner, task_id)].append(rollout)
            group_window[(miner, task_id)] = window_id
            used_any = True
        if used_any:
            consumed_keys.append(key)

    groups: list[RolloutGroup] = []
    for (miner, task_id), rollouts in bucket.items():
        if len(rollouts) < 2:
            continue  # need n≥2 for a meaningful σ
        rewards = [r.correctness for r in rollouts]
        mean = sum(rewards) / len(rewards)
        std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
        groups.append(
            RolloutGroup(
                miner_id=miner,
                task_id=task_id,
                window_id=group_window.get((miner, task_id), 0),
                sigma=std,
                mean_reward=mean,
                rollouts=rollouts,
            )
        )
    log(
        f"  completion_bundles scanned={len(cb_keys)} skipped_errors={cb_errors} "
        f"assembled_groups={len(groups)} "
        f"total_rollouts={sum(len(g.rollouts) for g in groups)}"
    )
    return groups, sorted(consumed_windows), consumed_keys


# ---------------------------------------------------------------------------
# GRPO step — pure math, adapted from romain13190/reliquary (MIT)
# ---------------------------------------------------------------------------


def _compute_advantages(rewards: list[float]) -> list[float]:
    """Group-relative normalised advantages.

    advantage_i = (r_i − mean) / std
    Degenerate groups (std ≈ 0) return zeros; callers should skip them.
    """
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    std = variance ** 0.5
    if std < 1e-8:
        return [0.0] * n
    return [(r - mean) / std for r in rewards]


def _rollout_loss(
    model,
    ref_model,
    rollout: Rollout,
    advantage: float,
    device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Compute (ppo_loss, kl_term) for one rollout.

    Returns None if the rollout is unusable (no tokens / no logprobs).
    π_old comes from the miner's token_logprobs — saves one forward pass
    per rollout (miner already computed them to sign the GRAIL proof).
    """
    tokens_list = rollout.tokens
    prompt_length = rollout.prompt_length
    old_logprobs_list = rollout.token_logprobs

    if prompt_length <= 0 or not old_logprobs_list or not tokens_list:
        return None
    if prompt_length >= len(tokens_list):
        return None  # no completion tokens to learn from

    tokens = torch.tensor([tokens_list], device=device)

    dtype_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )
    with dtype_ctx:
        logits = model(tokens).logits[0]

    log_probs_all = F.log_softmax(logits[:-1].float(), dim=-1)
    next_tokens = tokens[0, 1:]
    new_logprobs = log_probs_all.gather(1, next_tokens.unsqueeze(1)).squeeze(1)
    new_logprobs_c = new_logprobs[prompt_length - 1:]

    with torch.no_grad():
        with dtype_ctx:
            ref_logits = ref_model(tokens).logits[0]
        ref_log_probs_all = F.log_softmax(ref_logits[:-1].float(), dim=-1)
        ref_logprobs = ref_log_probs_all.gather(1, next_tokens.unsqueeze(1)).squeeze(1)
    ref_logprobs_c = ref_logprobs[prompt_length - 1:]

    old_logprobs = torch.tensor(
        old_logprobs_list, device=device, dtype=new_logprobs_c.dtype,
    )
    # Miner may have truncated at max_new_tokens — align lengths defensively.
    n = min(len(old_logprobs), len(new_logprobs_c), len(ref_logprobs_c))
    if n == 0:
        return None
    old_logprobs = old_logprobs[:n]
    new_logprobs_c = new_logprobs_c[:n]
    ref_logprobs_c = ref_logprobs_c[:n]

    log_ratio = new_logprobs_c - old_logprobs
    ratio = torch.exp(log_ratio)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1 - PPO_CLIP_EPSILON, 1 + PPO_CLIP_EPSILON) * advantage
    ppo_loss = -torch.min(surr1, surr2).mean()

    # KL(π_new || π_ref) via Schulman's k3: unbiased, low-variance, always ≥ 0.
    kl_log_ratio = ref_logprobs_c - new_logprobs_c
    kl = (torch.exp(kl_log_ratio) - 1 - kl_log_ratio).mean()

    return ppo_loss, kl


def grpo_train(
    model,
    ref_model,
    tokenizer,
    groups: list[RolloutGroup],
    *,
    steps: int,
    device,
) -> None:
    """Run *steps* GRPO optimiser steps over *groups*.

    Each step picks up to B_BATCH groups (FIFO, distinct prompts). Within
    each group, advantages are computed from rollout rewards; PPO-clipped
    surrogate summed over all rollouts; KL against ref added; one backward
    + optimiser.step per optimisation step.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
    )

    # Dedupe groups by prompt — prefer first-seen (= cheapest arrived; the
    # validator's FIFO order is preserved).
    seen_tasks: set[str] = set()
    deduped: list[RolloutGroup] = []
    for g in groups:
        if g.task_id in seen_tasks:
            continue
        seen_tasks.add(g.task_id)
        deduped.append(g)
    if not deduped:
        log("grpo_train: no usable groups, skipping")
        return

    model.train()
    step_idx = 0
    batch_start = 0
    while step_idx < steps and batch_start < len(deduped):
        batch = deduped[batch_start: batch_start + B_BATCH]
        batch_start += B_BATCH
        n_total = sum(len(g.rollouts) for g in batch)
        if n_total == 0:
            continue

        optimizer.zero_grad()
        total_ppo = 0.0
        total_kl = 0.0
        n_processed = 0

        for group in batch:
            advantages = _compute_advantages([r.correctness for r in group.rollouts])
            if all(a == 0.0 for a in advantages):
                continue
            for rollout, adv in zip(group.rollouts, advantages):
                result = _rollout_loss(
                    model=model, ref_model=ref_model,
                    rollout=rollout, advantage=adv, device=device,
                )
                if result is None:
                    continue
                ppo_loss, kl = result
                loss = (ppo_loss + KL_BETA * kl) / n_total
                loss.backward()
                total_ppo += float(ppo_loss.item())
                total_kl += float(kl.item())
                n_processed += 1

        if n_processed == 0:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        step_idx += 1
        log(
            f"  step {step_idx}/{steps} "
            f"ppo={total_ppo/n_processed:.4f} kl={total_kl/n_processed:.4f} "
            f"grad={float(grad_norm):.3f} rollouts={n_processed}/{n_total}"
        )

    model.eval()


# ---------------------------------------------------------------------------
# Helpers shared with SFT script
# ---------------------------------------------------------------------------


def make_target_tensor_names() -> list[str]:
    return [
        f"model.layers.{layer}.self_attn.{proj}.weight"
        for layer in TARGET_LAYERS
        for proj in TARGET_PROJS
    ]


def _load_state(path: str) -> dict:
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return {}


def _save_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Forge GRPO cycle against live mesh rollout groups"
    )
    parser.add_argument(
        "--state-file",
        default="/opt/reliquary-state/forge-trainer-state.json",
        help="Persistent state — same file as SFT cycle so continuous runs skip re-training.",
    )
    parser.add_argument(
        "--min-new-groups",
        "--min-new-pairs",  # wrapper-compat alias — the shared run-forge-training-cycle.sh passes --min-new-pairs.
        type=int,
        default=4,
        dest="min_new_groups",
        help="Skip the cycle if fewer than this many new in-zone groups since last run.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="GRPO optimiser steps this cycle (each step = B_BATCH groups).",
    )
    parser.add_argument(
        "--effective-at-offset-blocks",
        type=int,
        default=60,
        help="Blocks past current_window to schedule the hot-swap at.",
    )
    parser.add_argument(
        "--min-mean-abs-delta",
        type=float,
        default=1e-7,
        help=(
            "Partial-seal gate: skip publishing a delta when the mean abs "
            "parameter change is below this threshold. Prevents zero-delta "
            "commitments from burning R2 bandwidth + chain commitment slots."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-train on all post-v5 data regardless of state file.",
    )
    args = parser.parse_args()

    r2_token = open("/opt/reliquary-secrets/r2-cf-api-token").read().strip()
    signing_secret = open("/opt/reliquary-secrets/signing-secret").read().strip()
    backend = R2ObjectBackend(
        account_id="d5332aea7e3780d0f2391a4e4f6ddfbc",
        bucket="reliquary",
        cf_api_token=r2_token,
        public_url="https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev",
    )

    state = _load_state(args.state_file)
    if args.force:
        last_trained_window = MIN_LIVE_WINDOW
        log(f"--force: re-training on all data from window {last_trained_window}")
    else:
        last_trained_window = int(state.get("last_trained_window", MIN_LIVE_WINDOW))
    min_window_for_training = max(last_trained_window, MIN_LIVE_WINDOW)

    log(f"building zone-filtered group set from R2 (window >= {min_window_for_training})")
    groups, consumed_windows, consumed_keys = pull_training_groups(
        backend, min_window=min_window_for_training
    )
    if len(groups) < args.min_new_groups:
        log(
            f"only {len(groups)} in-zone groups since window {min_window_for_training} — "
            f"need {args.min_new_groups}. Skipping cycle (exit 0, not an error)."
        )
        return 0

    # Partial-seal gate: drop groups with σ=0 upfront. The zone filter
    # already rejects σ < σ_min at the validator, but old scorecards
    # from before miner_id got embedded in the verdict payload can still
    # surface σ=0 groups here. Training on them wastes a forge cycle —
    # GRPO advantage = (r − μ)/σ → 0 → zero gradient → zero delta.
    nonzero_groups = [g for g in groups if g.sigma > 1e-8]
    dropped = len(groups) - len(nonzero_groups)
    if dropped:
        log(f"  dropped {dropped}/{len(groups)} groups with σ=0 (no within-group contrast)")
    groups = nonzero_groups
    if len(groups) < args.min_new_groups:
        log(
            f"after σ=0 filter: only {len(groups)} trainable groups — need "
            f"{args.min_new_groups}. Skipping cycle (exit 0, not an error)."
        )
        return 0
    log(
        f"  training set: {len(groups)} groups, "
        f"{sum(len(g.rollouts) for g in groups)} total rollouts"
    )
    sample = groups[0]
    log(f"  sample group: task={sample.task_id} σ={sample.sigma:.3f} μ={sample.mean_reward:.3f}")

    log(f"loading {MODEL_REF}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REF)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REF, torch_dtype=torch.bfloat16,
    ).cuda()
    model.eval()

    target_names = make_target_tensor_names()
    pre_params = {
        n: p.detach().clone().cpu().float()
        for n, p in model.named_parameters()
        if n in target_names
    }
    assert len(pre_params) == len(target_names), (
        f"missing target tensors: have {set(pre_params)} want {set(target_names)}"
    )
    log(f"  snapshotted {len(pre_params)} target tensors pre-training")

    log("cloning frozen reference model for KL term")
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Only the target tensors require grad so we don't update the whole
    # 3B model — this mirrors the SFT LoRA scope while keeping the delta
    # computation identical between the two training paths.
    for n, p in model.named_parameters():
        p.requires_grad = n in target_names
    log(f"  frozen all params except {sum(1 for p in model.parameters() if p.requires_grad)} target tensors")

    log(f"GRPO training: ≤{args.steps} steps, B_BATCH={B_BATCH}, "
        f"lr={LEARNING_RATE}, clip={PPO_CLIP_EPSILON}, β={KL_BETA}")
    device = next(model.parameters()).device
    grpo_train(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        groups=groups,
        steps=args.steps,
        device=device,
    )

    post_params = {
        n: p.detach().clone().cpu().float()
        for n, p in model.named_parameters()
        if n in target_names
    }
    assert len(post_params) == len(target_names)

    sample_name = target_names[0]
    d = (post_params[sample_name] - pre_params[sample_name]).abs().mean().item()
    log(f"  mean abs delta on {sample_name}: {d:.6e}")

    # Partial-seal: don't commit a zero-delta policy to the bridge. Happens
    # when all groups had σ≈0 at training time (advantage=0 → gradient=0 →
    # no parameter change). Publishing would waste an R2 upload + a chain
    # commitment slot, and miner policy_consumer applies a no-op delta.
    min_delta = float(args.min_mean_abs_delta)
    if d < min_delta:
        log(
            f"mean abs delta {d:.6e} < min_mean_abs_delta {min_delta:.6e} — "
            f"policy didn't move; skipping publish (exit 0)."
        )
        # Advance last_trained_window anyway so we don't re-train on the
        # same dead groups next cycle.
        if consumed_windows:
            state["last_trained_window"] = int(max(consumed_windows)) + 1
            _save_state(args.state_file, state)
        return 0

    log("computing delta bundle")
    bundle = compute_delta(pre_params, post_params)
    log(f"  shards: {len(bundle.shards)}")
    log(f"  merkle_root: {bundle.merkle_root_hex}")
    log(f"  total delta bytes: {sum(len(s.data_bytes) for s in bundle.shards)}")

    subt = bt.subtensor("test")
    current_block = subt.get_current_block()
    current_window = (current_block // 30) * 30
    effective_at = current_window + int(args.effective_at_offset_blocks)

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
        consumed_rollout_windows=consumed_windows,
        consumed_rollout_keys=consumed_keys,
        effective_at_ledger_window=effective_at,
    )
    log(f"  attestation_key: {pub.attestation_key}")
    log(f"  commitment_key:  {pub.commitment_key}")
    log(f"  smoke_hash:      {pub.smoke_hash_hex}")
    log(f"  effective_at_ledger_window: {effective_at} (current: {current_window})")

    if consumed_windows:
        state["last_trained_window"] = int(max(consumed_windows)) + 1
    state.setdefault("runs", []).append(
        {
            "run_id": RUN_ID,
            "at_window": current_window,
            "effective_at_window": effective_at,
            "merkle_root": bundle.merkle_root_hex,
            "smoke_hash": pub.smoke_hash_hex,
            "groups": len(groups),
            "rollouts": sum(len(g.rollouts) for g in groups),
            "consumed_windows": consumed_windows,
            "trainer": "grpo",
        }
    )
    state["runs"] = state["runs"][-20:]
    _save_state(args.state_file, state)
    log(f"  state written: last_trained_window={state.get('last_trained_window')}")

    print("SUMMARY:", flush=True)
    print(json.dumps({
        "run_id": RUN_ID,
        "trainer": "grpo",
        "groups": len(groups),
        "rollouts": sum(len(g.rollouts) for g in groups),
        "steps_scheduled": args.steps,
        "target_tensor_count": len(target_names),
        "delta_bytes": sum(len(s.data_bytes) for s in bundle.shards),
        "merkle_root_hex": bundle.merkle_root_hex,
        "effective_at_ledger_window": effective_at,
        "current_window": current_window,
    }, indent=2), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
