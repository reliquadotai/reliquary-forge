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
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

os.environ.setdefault("HF_HOME", "/opt/reliquary-state/hf-cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/opt/reliquary-state/hf-cache/hub")

import bittensor as bt
import torch
import torch.nn.functional as F
from reliquary_protocol.storage import R2ObjectBackend
from transformers import AutoModelForCausalLM, AutoTokenizer

from reliquary.training.checkpoint_storage import publish_bundle
from reliquary.training.delta_checkpoints import compute_delta
from reliquary.training.policy_attestation import (
    make_hmac_authority,
    publish_attestation,
)


def _maybe_hf_publish(
    *,
    model: Any,
    tokenizer: Any,
    current_window: int,
    signing_secret: str,
) -> None:
    """Side-channel snapshot push to Hugging Face Hub (Phase 1.4).

    Reads ``RELIQUARY_HF_REPO_ID``, ``RELIQUARY_HF_TOKEN``, and
    ``RELIQUARY_HF_PUBLISH_INTERVAL_WINDOWS`` from env. When the
    publish-interval boundary aligns with the current window, the
    in-memory post-trained model is dumped to a temp dir and pushed
    asynchronously. Never blocks the trainer: ``maybe_publish``
    returns a Future, and on any exception we log + continue.
    """
    repo_id = os.environ.get("RELIQUARY_HF_REPO_ID", "").strip()
    if not repo_id:
        return
    interval_raw = os.environ.get("RELIQUARY_HF_PUBLISH_INTERVAL_WINDOWS", "10").strip()
    try:
        interval = max(1, int(interval_raw))
    except ValueError:
        log(f"  hf_hub_publisher: invalid interval {interval_raw!r}, skipping")
        return
    window_n = max(0, int(current_window))
    if window_n == 0 or window_n % interval != 0:
        return
    token = os.environ.get("RELIQUARY_HF_TOKEN") or None
    try:
        from reliquary.training.hf_hub_publisher import HfHubPublisher
    except Exception as exc:  # missing huggingface_hub or import error
        log(f"  hf_hub_publisher: import failed ({exc}); skipping")
        return
    import tempfile
    from pathlib import Path

    snapshot_dir = Path(tempfile.mkdtemp(prefix="reliquary-hf-snap-"))
    try:
        log(f"  hf_hub_publisher: serialising snapshot to {snapshot_dir}")
        model.save_pretrained(str(snapshot_dir))
        tokenizer.save_pretrained(str(snapshot_dir))
        publisher = HfHubPublisher(
            repo_id=repo_id,
            token=token,
            signing_secret=signing_secret,
            interval_windows=interval,
        )
        future = publisher.maybe_publish(
            window_n=window_n,
            snapshot_dir=snapshot_dir,
            commit_message=f"reliquary-forge sn462 trainer cycle (window {window_n})",
        )
        if future is None:
            log(
                "  hf_hub_publisher: scheduling skipped "
                f"(window={window_n} interval={interval})"
            )
            return
        # Block briefly so the operator log shows success/failure inline;
        # the executor still runs the upload off the main path so the
        # trainer's exit isn't gated by network latency beyond this wait.
        try:
            result = future.result(timeout=600)
            log(
                f"  hf_hub_publisher: published checkpoint_n={result.checkpoint_n} "
                f"revision={result.revision[:12]}"
            )
        except Exception as exc:
            log(f"  hf_hub_publisher: publish failed ({exc})")
        finally:
            publisher.shutdown(wait=False)
    finally:
        # Best-effort cleanup. The HF SDK has already uploaded the
        # bytes (or has failed); we don't need this dir any more.
        try:
            import shutil

            shutil.rmtree(snapshot_dir, ignore_errors=True)
        except Exception:
            pass

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


# Cursor file format (stored under the forge state dir):
#   {
#     "scorecards": {"<r2_key>": {"window_id": int, "task_prompts_present": bool, "in_zone_entries": [...], "seen_at": "iso"}},
#     "task_batches": {"<r2_key>": {"window_id": int, "seen_at": "iso"}},
#     "completion_bundles": {"<r2_key>": {"window_id": int, "miner_id": str, "seen_at": "iso"}},
#   }
# The cursor lets each cycle do O(new_keys) gets instead of O(all_keys).
FORGE_CURSOR_PATH_DEFAULT = "/opt/reliquary-state/forge-grpo-cursor.json"


def _load_cursor(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _save_cursor(path: str, cursor: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cursor, f)
    os.replace(tmp, path)


def pull_training_groups(
    backend: R2ObjectBackend,
    *,
    min_window: int,
    cursor_path: str = FORGE_CURSOR_PATH_DEFAULT,
) -> tuple[list[RolloutGroup], list[int], list[str]]:
    """Fetch ``(miner, task)`` rollout groups that passed the validator's
    zone filter (σ ≥ 0.43 for binary rewards).

    We walk every scorecard.payload.window_metrics.zone_filter.groups entry
    where ``in_zone == True``, then join to the matching task prompt (from
    task_batches) and completion rollouts (from completion_bundles).

    Uses a local cursor file so each cycle only fetches NEW R2 keys;
    previously-seen keys are served from the cursor's memoised summary.
    This takes per-cycle time from ~10 min (full 700+ scorecard rescan)
    to ~30 s in steady state.

    Returns ``(groups, consumed_windows, consumed_keys)`` — same shape
    as the SFT pull_training_pairs helper so the attestation step can
    use it without modification.
    """
    cursor = _load_cursor(cursor_path)
    cursor.setdefault("scorecards", {})
    cursor.setdefault("task_batches", {})
    cursor.setdefault("completion_bundles", {})
    iso_now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # 1. Task batches → {task_id: (prompt, window_id)}
    tb_keys = backend.list("task_batches/")
    task_index: dict[str, tuple[str, int]] = {}
    tb_errors = 0
    tb_new = 0
    for key in tb_keys:
        entry = cursor["task_batches"].get(key)
        if entry is not None:
            # Memoised: the cursor carries the tasks list so we don't
            # refetch. Rebuild task_index from the cached prompts.
            for tid, prompt in (entry.get("tasks") or {}).items():
                task_index[tid] = (prompt, int(entry["window_id"]))
            continue
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
        cached_tasks: dict[str, str] = {}
        for task in tb["payload"].get("tasks", []):
            tid = task.get("task_id")
            prompt = task.get("prompt", "")
            if tid and prompt:
                cached_tasks[tid] = prompt
                if window_id >= min_window:
                    task_index[tid] = (prompt, window_id)
        cursor["task_batches"][key] = {
            "window_id": window_id,
            "tasks": cached_tasks,
            "seen_at": iso_now,
        }
        tb_new += 1
    log(
        f"  task_batches scanned={len(tb_keys)} new={tb_new} skipped_errors={tb_errors} "
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
    sc_new = 0
    for key in sc_keys:
        entry = cursor["scorecards"].get(key)
        if entry is not None:
            # Memoised: use the cached zone_filter summary to rebuild
            # in-zone keys without refetching the scorecard.
            w = int(entry["window_id"])
            if w >= min_window:
                for k in entry.get("in_zone_keys", []):
                    in_zone_keys.add(tuple(k))
                if entry.get("in_zone_keys"):
                    consumed_windows.add(w)
            continue
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
        zone = sc["payload"].get("window_metrics", {}).get("zone_filter") or {}
        cached_in_zone: list[list[str]] = []
        for group in zone.get("groups", []):
            if not group.get("in_zone"):
                continue
            miner = str(group.get("miner_id") or "")
            task_id = str(group.get("task_id") or "")
            if miner and task_id:
                cached_in_zone.append([miner, task_id])
                if window_id >= min_window:
                    in_zone_keys.add((miner, task_id))
                    consumed_windows.add(window_id)
        cursor["scorecards"][key] = {
            "window_id": window_id,
            "in_zone_keys": cached_in_zone,
            "seen_at": iso_now,
        }
        sc_new += 1
    log(
        f"  scorecards scanned={len(sc_keys)} new={sc_new} skipped_errors={sc_errors} "
        f"in_zone_groups={len(in_zone_keys)} consumed_windows={len(consumed_windows)}"
    )

    # 3a. Verdict bundles → {completion_id: correctness}.
    # CRITICAL: miners don't evaluate their own completions — completion
    # payloads DON'T carry a `correctness` field. Only validators compute
    # it (via evaluate_math_trace) and embed it in verdict payloads.
    # Without this lookup the trainer reads correctness=0 for every
    # rollout → σ=0 → partial-seal gate drops every group.
    #
    # We only need verdicts for the windows that had in-zone groups, so
    # we scope the scan to consumed_windows. Memoise fetched keys in the
    # cursor so subsequent cycles skip already-read verdict bundles.
    cursor.setdefault("verdict_bundles", {})
    vb_keys = [k for k in backend.list("verdict_bundles/") if k.endswith(".json.gz")]
    correctness_by_cid: dict[str, float] = {}
    vb_errors = 0
    vb_new = 0
    for key in vb_keys:
        try:
            window_id = int(key.split("/")[1].split("-")[1])
        except (IndexError, ValueError):
            continue
        if window_id not in consumed_windows:
            continue
        cached = cursor["verdict_bundles"].get(key)
        if cached and isinstance(cached.get("correctness"), dict):
            # Replay from cursor — no R2 fetch needed.
            for cid, reward in cached["correctness"].items():
                correctness_by_cid[str(cid)] = float(reward)
            continue
        try:
            raw = backend.get(key)
        except Exception:
            vb_errors += 1
            continue
        if raw is None:
            continue
        try:
            verdicts = json.loads(gzip.decompress(raw).decode())
        except Exception:
            continue
        vb_new += 1
        fresh: dict[str, float] = {}
        for v in verdicts:
            p = v.get("payload", {})
            cid = str(p.get("completion_id") or "")
            if not cid:
                continue
            try:
                reward = float(p.get("correctness", 0.0))
            except (TypeError, ValueError):
                reward = 0.0
            fresh[cid] = reward
            correctness_by_cid[cid] = reward
        cursor["verdict_bundles"][key] = {
            "window_id": window_id,
            "seen_at": iso_now,
            "correctness": fresh,
        }
    log(
        f"  verdict_bundles scanned={len(vb_keys)} new={vb_new} skipped_errors={vb_errors} "
        f"correctness_map_size={len(correctness_by_cid)}"
    )

    # 3b. Completion bundles → rollouts, grouped by (miner, task) where in-zone.
    # Only fetch bundles whose window has at least one in-zone group.
    # Bundles without any in-zone group can't contribute to training,
    # so skipping them saves the biggest single category of R2 gets.
    # Also memoise `used` state so post-training we don't reprocess
    # bundles we already pulled rollouts from.
    cb_keys = [k for k in backend.list("completion_bundles/") if k.endswith(".json.gz")]
    bucket: dict[tuple[str, str], list[Rollout]] = defaultdict(list)
    group_window: dict[tuple[str, str], int] = {}
    consumed_keys: list[str] = []
    cb_errors = 0
    cb_new = 0
    for key in cb_keys:
        try:
            window_id = int(key.split("/")[1].split("-")[1])
        except (IndexError, ValueError):
            continue
        if window_id < min_window or window_id not in consumed_windows:
            # Either below horizon OR this window had no in-zone groups →
            # no rollouts worth fetching. Record-and-skip.
            cursor["completion_bundles"].setdefault(
                key, {"window_id": window_id, "seen_at": iso_now, "used": False}
            )
            continue
        entry = cursor["completion_bundles"].get(key)
        if entry is not None and entry.get("used") is True:
            # Already consumed in a prior cycle — nothing new here.
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
        cb_new += 1
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
            # Correctness lives in the validator's verdict, not the miner's
            # completion. Look it up in the map we built from verdict_bundles.
            completion_id = str(c.get("artifact_id") or "")
            reward = correctness_by_cid.get(completion_id, 0.0)
            # Miner's payload uses `completion_logprobs` (see
            # reliquary_inference/miner/engine.py:291). Older
            # upstream-derived code called it `token_logprobs`;
            # support both so we're compatible with either miner build.
            logprobs = (
                payload.get("completion_logprobs")
                or payload.get("token_logprobs")
                or []
            )
            rollout = Rollout(
                completion_text=str(payload.get("completion_text", "")),
                prompt=prompt,
                correctness=reward,
                tokens=list(payload.get("tokens") or []),
                prompt_length=int(payload.get("prompt_length") or 0),
                token_logprobs=list(logprobs),
            )
            bucket[(miner, task_id)].append(rollout)
            group_window[(miner, task_id)] = window_id
            used_any = True
        if used_any:
            consumed_keys.append(key)
        # NOTE: we defer marking `used=True` until AFTER σ-filter downstream.
        # A bundle whose rollouts all collapsed to σ=0 shouldn't be
        # locked in as "consumed" — that's how the pre-correctness-fix
        # cycles bricked the cursor (see /opt/reliquary-state/forge-grpo-
        # cursor.json.bak.pre-correctness-fix). Mark `seen=True` only so
        # we don't repeatedly scan the key each cycle.
        cursor["completion_bundles"][key] = {
            "window_id": window_id,
            "seen_at": iso_now,
            "used": False,  # overwritten below if any group survives σ-filter
            "keys_extracted": used_any,
        }

    groups: list[RolloutGroup] = []
    productive_windows: set[int] = set()
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
        if std > 1e-8:
            productive_windows.add(group_window.get((miner, task_id), 0))
    # Now commit `used=True` on the completion_bundle cursor entries
    # whose window produced at least one σ>0 group. Bundles that
    # extracted rollouts but collapsed to σ=0 stay `used=False` so a
    # future correctness re-check (e.g. bug fix, validator update) can
    # re-pull them without needing a cursor wipe.
    for key, entry in cursor["completion_bundles"].items():
        if entry.get("keys_extracted") and entry.get("window_id") in productive_windows:
            entry["used"] = True
    log(
        f"  completion_bundles scanned={len(cb_keys)} new_fetched={cb_new} skipped_errors={cb_errors} "
        f"assembled_groups={len(groups)} "
        f"total_rollouts={sum(len(g.rollouts) for g in groups)}"
    )
    # Save the cursor so the next cycle can skip keys we've already seen.
    # Safe to fail: the worst outcome is a slower next cycle, not a
    # correctness issue (the cursor is purely a perf optimisation).
    try:
        _save_cursor(cursor_path, cursor)
        cursor_size = (
            len(cursor.get("scorecards") or {})
            + len(cursor.get("task_batches") or {})
            + len(cursor.get("completion_bundles") or {})
        )
        log(f"  cursor written: {cursor_size} keys memoised at {cursor_path}")
    except Exception as exc:
        log(f"  cursor write failed (non-fatal): {type(exc).__name__}: {exc}")
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
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _save_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


# R2 backup key for the forge trainer state. Mirrors the local file so a
# node rebuild / disk-loss doesn't reset the trainer to MIN_LIVE_WINDOW
# and re-process already-trained rollouts (which would cause dup deltas +
# thrash the miner policy).
FORGE_STATE_R2_KEY_DEFAULT = "forge/state/forge-trainer-state.json"


def _backup_state_to_r2(backend: R2ObjectBackend, state: dict, *, r2_key: str) -> None:
    """Best-effort R2 backup of the forge trainer state.

    Swallows exceptions — the local file is the authoritative copy,
    R2 is the disaster-recovery mirror. A transient R2 error should
    not fail the training cycle.
    """
    try:
        backend.put(r2_key, json.dumps(state, indent=2, sort_keys=True).encode("utf-8"))
    except Exception as exc:
        log(f"  state r2 backup failed (non-fatal): {type(exc).__name__}: {exc}")


def _restore_state_from_r2(backend: R2ObjectBackend, *, r2_key: str) -> dict | None:
    """Pull the state JSON from R2. Returns None if not found."""
    try:
        raw = backend.get(r2_key)
        if raw is None:
            return None
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


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
        "--min-distinct-miners",
        type=int,
        default=int(os.environ.get("RELIQUARY_FORGE_MIN_DISTINCT_MINERS", "2")),
        help=(
            "Anti-poison gate: refuse the cycle if fewer than N distinct "
            "miner_ids appear across the assembled training set. Closes the "
            "sybil + stake-majority trainer-poisoning attack vector — a "
            "single miner crafting σ≈0.5 with [1,1,1,1,0,0,0,0] rewards can "
            "pass the σ ≥ 0.43 zone filter and feed the trainer "
            "half-poisoned data unless this gate trips. Default 2 (env: "
            "RELIQUARY_FORGE_MIN_DISTINCT_MINERS); set 1 only on bootstrapping "
            "windows with a single live miner — only safe when the operator "
            "controls the miner."
        ),
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
        "--cursor-path",
        default=FORGE_CURSOR_PATH_DEFAULT,
        help=(
            "Local JSON cursor memoising (r2_key → summary) for already-"
            "scanned task_batches / scorecards / completion_bundles. Cuts "
            "per-cycle R2 get count from ~900 to ~10 in steady state."
        ),
    )
    parser.add_argument(
        "--state-r2-key",
        default=FORGE_STATE_R2_KEY_DEFAULT,
        help=(
            "R2 key to mirror the trainer state to on each save. If the "
            "local file is missing at startup (fresh node build / disk "
            "loss), the trainer pulls from this key as disaster recovery. "
            "Set to '' to disable R2 mirroring."
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
    # If the local state file is missing / empty (fresh node build, disk
    # loss), transparently restore from R2 so the trainer doesn't reset
    # last_trained_window to MIN_LIVE_WINDOW and thrash the policy.
    if not state:
        restored = _restore_state_from_r2(backend, r2_key=args.state_r2_key)
        if restored:
            state = restored
            log(f"  restored state from R2 key {args.state_r2_key}")
    if args.force:
        last_trained_window = MIN_LIVE_WINDOW
        log(f"--force: re-training on all data from window {last_trained_window}")
    else:
        last_trained_window = int(state.get("last_trained_window", MIN_LIVE_WINDOW))
    min_window_for_training = max(last_trained_window, MIN_LIVE_WINDOW)

    log(f"building zone-filtered group set from R2 (window >= {min_window_for_training})")
    groups, consumed_windows, consumed_keys = pull_training_groups(
        backend, min_window=min_window_for_training, cursor_path=args.cursor_path,
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

    # Anti-poison gate: require ≥ N distinct miner_ids in the training
    # set. Closes the sybil + stake-majority trainer-poisoning vector
    # documented in the pre-cutover security audit (finding #7). A
    # single miner crafting σ≈0.5 rewards can pass the zone filter and
    # feed the trainer half-poisoned data — but if the trainer also
    # requires multiple distinct submitters per cycle, the attacker
    # has to coordinate enough hotkeys to also pass the validator
    # mesh independently, which is the same hard problem mesh
    # consensus already imposes. Skip here is a deliberate non-error.
    distinct_miners = {g.miner_id for g in groups}
    min_miners = max(1, int(args.min_distinct_miners))
    if len(distinct_miners) < min_miners:
        log(
            f"  anti-poison gate: only {len(distinct_miners)} distinct "
            f"miner(s) in {len(groups)} groups — need {min_miners}. "
            f"Skipping cycle (exit 0, not an error)."
        )
        return 0
    log(
        f"  anti-poison gate: {len(distinct_miners)} distinct miner(s) "
        f"across {len(groups)} groups (≥ {min_miners} required)"
    )

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
            _backup_state_to_r2(backend, state, r2_key=args.state_r2_key)
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
    _backup_state_to_r2(backend, state, r2_key=args.state_r2_key)
    log(
        f"  state written: last_trained_window={state.get('last_trained_window')} "
        f"(r2 backup: {args.state_r2_key})"
    )

    _maybe_hf_publish(
        model=model,
        tokenizer=tokenizer,
        current_window=current_window,
        signing_secret=signing_secret,
    )

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
