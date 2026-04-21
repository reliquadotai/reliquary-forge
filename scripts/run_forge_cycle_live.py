"""Run one simulated Forge cycle end-to-end, real Qwen2.5-3B weights over R2.

What this proves (the bridge is fully live against real state):

  1. Load base Qwen2.5-3B-Instruct weights from HF cache.
  2. Apply a small deterministic perturbation to a handful of tensors
     (simulates a post-training LoRA-like delta — we don't actually
     backprop here to keep the wallclock short, but the bytes are real
     3B weights, so the delta bundle is a real DeltaBundle over real
     tensors).
  3. compute_delta(pre, post) → DeltaBundle (int8 quantized + Merkle root).
  4. publish_bundle() → writes shards + manifest to real R2 under
     checkpoints/forge-closed-loop-<ts>/<window>/.
  5. publish_attestation() → signs + writes CheckpointAttestation +
     PolicyCommitment. Commitment is keyed by effective_at_window so
     the Ledger policy_consumer can find it.
  6. consumed_rollout_windows + consumed_rollout_keys reference the
     ACTUAL rollout_bundles that our 4-validator mesh has been
     publishing for netuid 462 this session.

The produced artifacts are CONTENT-ADDRESSABLE + SIGNED and live in the
live reliquary R2 bucket. The Ledger policy_consumer can fetch them
and verify the signature chain end-to-end.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time

# The reliquary-inference + reliquary packages are on staging2's venv via
# `pip install -e`. This script runs out of /opt/reliquary-venv.

os.environ.setdefault("HF_HOME", "/opt/reliquary-state/hf-cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/opt/reliquary-state/hf-cache/hub")

import torch
from transformers import AutoModelForCausalLM
from collections import defaultdict

from reliquary.training.checkpoint_storage import publish_bundle
from reliquary.training.delta_checkpoints import compute_delta
from reliquary.training.policy_attestation import (
    make_hmac_authority,
    publish_attestation,
)
from reliquary_protocol.storage import R2ObjectBackend


MODEL_REF = "Qwen/Qwen2.5-3B-Instruct"
INFERENCE_NETUID = 462
TRAINING_NETUID = 462  # per project_forge_single_subnet.md — same subnet
RUN_ID = f"forge-closedloop-{int(time.time())}"
POLICY_AUTHORITY_HOTKEY = "reliquary-policy-authority-v1"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    # --- Secrets + R2 backend ---
    r2_token = open("/opt/reliquary-secrets/r2-cf-api-token").read().strip()
    signing_secret = open("/opt/reliquary-secrets/signing-secret").read().strip()

    backend = R2ObjectBackend(
        account_id="d5332aea7e3780d0f2391a4e4f6ddfbc",
        bucket="reliquary",
        cf_api_token=r2_token,
        public_url="https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev",
    )

    # --- Discover what the Ledger mesh has been publishing ---
    log(f"discovering rollout_bundles under rollouts/{INFERENCE_NETUID}/")
    rollout_keys = [k for k in backend.list(f"rollouts/{INFERENCE_NETUID}/") if k.endswith(".json")]
    if not rollout_keys:
        log(f"NOTE: no rollout_bundles at rollouts/{INFERENCE_NETUID}/ yet")
        log("  falling back to empty consumed-windows list (bridge still works)")
        consumed_windows: list[int] = []
        consumed_keys: list[str] = []
    else:
        rollout_keys = sorted(rollout_keys)[-5:]  # last 5
        consumed_windows = []
        consumed_keys = list(rollout_keys)
        for k in rollout_keys:
            # keys look like rollouts/462/<win>-<win>/<validator>.json
            parts = k.split("/")
            if len(parts) >= 3 and "-" in parts[2]:
                try:
                    consumed_windows.append(int(parts[2].split("-")[0]))
                except ValueError:
                    pass
        consumed_windows = sorted(set(consumed_windows))
        log(f"  consumed {len(consumed_keys)} rollout keys across windows {consumed_windows}")

    # --- Load Qwen2.5-3B-Instruct ---
    log(f"loading {MODEL_REF} (this reads from /opt/reliquary-state/hf-cache)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REF, torch_dtype=torch.bfloat16
    )
    model.eval()
    log(f"  params = {sum(p.numel() for p in model.parameters()):,}")

    # --- Snapshot pre-training weights for a SUBSET of tensors ---
    # Full model is 3B params; publishing a delta for every tensor is
    # gigabytes + 100+ shards. Pick a small representative subset:
    # the first attention block's q/k/v projections. This is exactly
    # the shape a targeted LoRA-merged delta would take.
    target_names = []
    for name, _ in model.named_parameters():
        if ".0." in name and ("q_proj" in name or "k_proj" in name or "v_proj" in name) and "weight" in name:
            target_names.append(name)
        if len(target_names) >= 3:
            break
    log(f"  target tensors for delta: {target_names}")

    pre_params = {
        n: p.detach().clone().cpu().float()
        for n, p in model.named_parameters()
        if n in target_names
    }

    # --- Apply a deterministic small perturbation (simulating training) ---
    # 0.002 × bfloat16-normal — below sketch tolerance, mimics a
    # few-hundred-step LoRA fine-tune.
    torch.manual_seed(0xC10DE100)
    for name, p in model.named_parameters():
        if name in target_names:
            with torch.no_grad():
                delta = torch.randn_like(p, dtype=torch.float32) * 0.002
                p.data.add_(delta.to(p.dtype))

    post_params = {
        n: p.detach().clone().cpu().float()
        for n, p in model.named_parameters()
        if n in target_names
    }

    # --- compute_delta: int8 quantize + Merkle commit ---
    log("computing delta bundle (int8 quantized + Merkle-committed)")
    bundle = compute_delta(pre_params, post_params)
    log(f"  shards: {len(bundle.shards)}")
    log(f"  merkle_root: {bundle.merkle_root_hex}")
    log(f"  total delta bytes: {sum(len(s.data_bytes) for s in bundle.shards)}")

    # --- Pick a checkpoint window + effective ledger window ---
    # Find current Ledger block so the commitment's effective_at_window
    # is meaningfully "soon".
    import bittensor as bt

    subt = bt.subtensor("test")
    current_block = subt.get_current_block()
    current_window = (current_block // 30) * 30
    effective_at = current_window + 60  # 2 windows out
    checkpoint_window = current_window
    log(f"  current block={current_block} window={current_window}")
    log(f"  effective_at_ledger_window={effective_at}")

    # --- publish_bundle → R2 ---
    log(f"publishing delta bundle to checkpoints/{RUN_ID}/{checkpoint_window}/")
    manifest = publish_bundle(
        bundle, run_id=RUN_ID, window_id=checkpoint_window, backend=backend
    )
    log(f"  manifest_key: {manifest.manifest_key}")

    # --- publish_attestation + PolicyCommitment → R2 ---
    log("publishing CheckpointAttestation + PolicyCommitment")
    signer = make_hmac_authority(POLICY_AUTHORITY_HOTKEY, signing_secret)
    publication = publish_attestation(
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
    log(f"  attestation_key: {publication.attestation_key}")
    log(f"  attestation_id:  {publication.attestation.artifact_id()}")
    log(f"  commitment_key:  {publication.commitment_key}")
    log(f"  smoke_hash:      {publication.smoke_hash_hex}")

    # --- Emit a summary JSON for the Ledger side to consume ---
    summary = {
        "run_id": RUN_ID,
        "checkpoint_window_id": checkpoint_window,
        "effective_at_ledger_window": effective_at,
        "attestation_key": publication.attestation_key,
        "attestation_id": publication.attestation.artifact_id(),
        "commitment_key": publication.commitment_key,
        "manifest_key": manifest.manifest_key,
        "merkle_root_hex": bundle.merkle_root_hex,
        "smoke_hash_hex": publication.smoke_hash_hex,
        "shard_count": len(bundle.shards),
        "total_delta_bytes": sum(len(s.data_bytes) for s in bundle.shards),
        "consumed_rollout_windows": consumed_windows,
    }
    print("SUMMARY:", flush=True)
    import json

    print(json.dumps(summary, indent=2), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
