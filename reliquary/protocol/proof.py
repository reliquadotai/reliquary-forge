from __future__ import annotations

import hashlib
import hmac
import random
import struct
from typing import Any

from .constants import CHALLENGE_K, LAYER_INDEX, PRIME_Q, PROOF_TOLERANCE, RNG_LABEL


def int_to_bytes(value: int) -> bytes:
    return struct.pack(">I", value & 0xFFFFFFFF)


def prf(label: bytes, *parts: bytes, out_bytes: int) -> bytes:
    digest = hashlib.sha256(label + b"||" + b"||".join(parts)).digest()
    while len(digest) < out_bytes:
        digest += hashlib.sha256(digest).digest()
    return digest[:out_bytes]


def r_vec_from_randomness(rand_hex: str, d_model: int) -> torch.Tensor:
    import torch

    clean_hex = rand_hex.replace("0x", "").replace("0X", "")
    raw = prf(RNG_LABEL["sketch"], bytes.fromhex(clean_hex), out_bytes=4 * d_model)
    ints = struct.unpack(">" + "i" * d_model, raw)
    return torch.tensor(ints, dtype=torch.int32)


def indices_from_tokens(tokens: list[int], rand_hex: str, seq_len: int, k: int) -> list[int]:
    token_bytes = b"".join(int_to_bytes(token) for token in tokens)
    token_hash = hashlib.sha256(token_bytes).digest()
    clean_hex = rand_hex.replace("0x", "").replace("0X", "")
    material = prf(RNG_LABEL["open"], token_hash, bytes.fromhex(clean_hex), out_bytes=32)
    rng = random.Random(material)
    return sorted(rng.sample(range(seq_len), min(k, seq_len)))


def dot_mod_q(hidden: torch.Tensor, r_vec: torch.Tensor) -> int:
    import torch

    scaled = torch.round(hidden * 1024).to(torch.int64)
    prod = torch.dot(scaled, r_vec.to(torch.int64))
    return int(prod.item() % PRIME_Q)


def sign_s_vals(s_vals: list[int], secret_key: str) -> str:
    payload = b"".join(int_to_bytes(value) for value in s_vals)
    signature = hmac.new(secret_key.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return signature


def verify_s_vals_signature(s_vals: list[int], signature_hex: str, secret_key: str) -> bool:
    expected = sign_s_vals(s_vals, secret_key)
    return hmac.compare_digest(signature_hex, expected)


def derive_rollout_randomness(policy_id: str, challenge_id: str, miner_id: str, sample_index: int) -> str:
    material = f"{policy_id}|{challenge_id}|{miner_id}|{sample_index}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def build_proof(
    model,
    full_token_ids: list[int],
    miner_secret: str,
    randomness_hex: str,
) -> dict[str, Any]:
    import torch

    device = next(model.parameters()).device
    full_ids = torch.tensor(full_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        outputs = model(full_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[LAYER_INDEX][0]

    r_vec = r_vec_from_randomness(randomness_hex, hidden.shape[-1]).to(device)
    s_vals = [dot_mod_q(hidden[idx], r_vec) for idx in range(hidden.shape[0])]
    signature = sign_s_vals(s_vals, miner_secret)

    open_randomness = hashlib.sha256((randomness_hex + ":open").encode("utf-8")).hexdigest()
    indices = indices_from_tokens(full_token_ids, open_randomness, len(full_token_ids), CHALLENGE_K)

    return {
        "proof_version": "v1",
        "layer_index": LAYER_INDEX,
        "randomness": randomness_hex,
        "open_randomness": open_randomness,
        "s_vals": s_vals,
        "indices": indices,
        "signature": signature,
    }


def verify_proof(
    model,
    full_token_ids: list[int],
    proof: dict[str, Any],
    miner_secret: str,
) -> dict[str, Any]:
    import torch

    if not verify_s_vals_signature(proof["s_vals"], proof["signature"], miner_secret):
        return {"valid": False, "reason": "bad_signature"}

    expected_indices = indices_from_tokens(
        full_token_ids,
        proof["open_randomness"],
        len(full_token_ids),
        len(proof["indices"]),
    )
    if expected_indices != proof["indices"]:
        return {"valid": False, "reason": "index_mismatch"}

    device = next(model.parameters()).device
    full_ids = torch.tensor(full_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        outputs = model(full_ids, output_hidden_states=True)
    hidden = outputs.hidden_states[LAYER_INDEX][0]
    r_vec = r_vec_from_randomness(proof["randomness"], hidden.shape[-1]).to(device)

    for idx in proof["indices"]:
        local = dot_mod_q(hidden[idx], r_vec)
        committed = proof["s_vals"][idx]
        diff = abs(local - committed)
        mod_diff = min(diff, PRIME_Q - diff)
        if mod_diff > PROOF_TOLERANCE:
            return {"valid": False, "reason": f"sketch_mismatch:{idx}"}

    return {"valid": True, "reason": "ok"}
