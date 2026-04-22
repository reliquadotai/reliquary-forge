from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


def _resolve_pretrained_ref(model_ref: str) -> tuple[str, dict[str, Any]]:
    path_obj = Path(model_ref).expanduser()
    if path_obj.exists():
        return str(path_obj.resolve()), {"local_files_only": True}
    return model_ref, {}


@lru_cache(maxsize=8)
def _load_tokenizer(model_ref: str):
    from transformers import AutoTokenizer

    resolved_ref, kwargs = _resolve_pretrained_ref(model_ref)
    tokenizer = AutoTokenizer.from_pretrained(resolved_ref, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_bundle(model_ref: str, device: str = "cpu", trainable: bool = False) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM

    resolved_ref, kwargs = _resolve_pretrained_ref(model_ref)
    tokenizer = _load_tokenizer(model_ref)
    model = AutoModelForCausalLM.from_pretrained(resolved_ref, **kwargs)
    if not trainable:
        model.eval()
    model.to(device)
    return {"model": model, "tokenizer": tokenizer, "device": device, "model_ref": resolved_ref}


def save_model_bundle(model, tokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def compute_completion_logprobs(
    model,
    full_token_ids: list[int],
    prompt_length: int,
    device: str,
) -> list[float]:
    import torch

    if len(full_token_ids) <= prompt_length:
        return []

    input_ids = torch.tensor(full_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids).logits[0]

    log_probs = torch.log_softmax(logits[:-1], dim=-1)
    completion_target_ids = input_ids[0, prompt_length:]
    start = prompt_length - 1
    end = start + completion_target_ids.shape[0]
    selected = log_probs[start:end, :].gather(1, completion_target_ids.unsqueeze(1)).squeeze(1)
    return [float(x) for x in selected.detach().cpu().tolist()]


def decode_completion(tokenizer, completion_token_ids: list[int]) -> str:
    return tokenizer.decode(completion_token_ids, skip_special_tokens=True)
