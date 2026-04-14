from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..envs.reasoning import (
    answer_reasoning_task,
    evaluate_reasoning_trace,
    generate_reasoning_tasks,
    render_reasoning_conversation,
    summarize_reasoning_evaluations,
)
from ..protocol.artifacts import make_artifact
from ..storage.registry import ArtifactRegistry
from ..utils.json_io import read_json, write_json


def _pack_rows(pack_artifact: dict[str, Any]) -> list[dict[str, Any]]:
    jsonl_path = Path(pack_artifact["payload"]["export_refs"]["jsonl"])
    return [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _reasoning_capabilities_from_pack(pack_artifact: dict[str, Any]) -> dict[str, Any]:
    rows = _pack_rows(pack_artifact)
    supported_operations = sorted({eval_row["operation"] for eval_row in rows})
    return {
        "supported_operations": supported_operations,
        "record_count": len(rows),
    }


def evaluate_reasoning_checkpoint(
    *,
    model_ref: str,
    holdout_count: int,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    capabilities = {"supported_operations": []}
    if Path(model_ref).exists():
        capability_path = Path(model_ref) / "reasoning_capabilities.json"
        if capability_path.exists():
            capabilities = read_json(capability_path)
    if capabilities.get("supported_operations"):
        tasks = generate_reasoning_tasks(count=holdout_count, seed=20_000, split="holdout")
        evaluations = []
        for task in tasks:
            trace_text = answer_reasoning_task(task, capabilities)
            evaluations.append(evaluate_reasoning_trace(task, trace_text))
        return summarize_reasoning_evaluations(
            evaluations,
            supported_operations=capabilities.get("supported_operations", []),
        )
    if cfg is not None and not str(model_ref).startswith("reasoning-student://"):
        return _evaluate_transformer_reasoning_checkpoint(
            model_ref=model_ref,
            holdout_count=holdout_count,
            cfg=cfg,
        )
    tasks = generate_reasoning_tasks(count=holdout_count, seed=20_000, split="holdout")
    evaluations = []
    for task in tasks:
        trace_text = answer_reasoning_task(task, capabilities)
        evaluations.append(evaluate_reasoning_trace(task, trace_text))
    return summarize_reasoning_evaluations(
        evaluations,
        supported_operations=capabilities.get("supported_operations", []),
    )


def _evaluate_transformer_reasoning_checkpoint(
    *,
    model_ref: str,
    holdout_count: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    path_obj = Path(model_ref)
    if path_obj.exists():
        model_ref = str(path_obj.resolve())
        model_kwargs["local_files_only"] = True
    if torch.cuda.is_available():
        model_kwargs.update({"device_map": "auto", "torch_dtype": dtype})
    model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)
    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if path_obj.exists():
        tokenizer_kwargs["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_ref, **tokenizer_kwargs)
    result = _evaluate_loaded_transformer_reasoning_checkpoint(
        model=model,
        tokenizer=tokenizer,
        holdout_count=holdout_count,
        cfg=cfg,
        model_ref=model_ref,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _evaluate_loaded_transformer_reasoning_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    holdout_count: int,
    cfg: dict[str, Any],
    model_ref: str,
) -> dict[str, Any]:
    import torch

    tasks = generate_reasoning_tasks(count=holdout_count, seed=20_000, split="holdout")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    evaluations: list[dict[str, Any]] = []
    for task in tasks:
        prompt_text = render_reasoning_conversation(
            task["prompt"],
            tokenizer=tokenizer,
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt_text, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=int(cfg.get("max_new_tokens", 48)),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion_ids = outputs[0][encoded["input_ids"].shape[1] :]
        trace_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        evaluations.append(evaluate_reasoning_trace(task, trace_text))
    result = summarize_reasoning_evaluations(
        evaluations,
        model_ref=model_ref,
        supported_operations=[],
    )
    return result


class LocalReasoningStudentBackend:
    name = "local_reasoning_student"

    def train(
        self,
        *,
        registry: ArtifactRegistry,
        cfg: dict[str, Any],
        policy_artifact: dict[str, Any],
        pack_artifact: dict[str, Any],
        window_id: int,
    ) -> dict[str, Any]:
        before_eval = evaluate_reasoning_checkpoint(
            model_ref=policy_artifact["payload"]["model_ref"],
            holdout_count=cfg["distill_holdout_count"],
            cfg=cfg,
        )
        checkpoint_id = f"distill-checkpoint-window-{window_id + 1:04d}"
        checkpoint_dir = registry.checkpoint_dir(checkpoint_id)
        capabilities = _reasoning_capabilities_from_pack(pack_artifact)
        write_json(checkpoint_dir / "reasoning_capabilities.json", capabilities)
        write_json(
            checkpoint_dir / "training_summary.json",
            {
                "backend": self.name,
                "record_count": capabilities["record_count"],
                "supported_operations": capabilities["supported_operations"],
            },
        )
        after_eval = evaluate_reasoning_checkpoint(
            model_ref=str(checkpoint_dir),
            holdout_count=cfg["distill_holdout_count"],
            cfg=cfg,
        )
        benchmark_artifact = make_artifact(
            artifact_type="benchmark_result",
            producer_id=cfg["validator_id"],
            producer_role="benchmark_validator",
            policy_id=policy_artifact["artifact_id"],
            window_id=window_id,
            lane="distill",
            commodity_use="dual_use",
            parent_ids=[policy_artifact["artifact_id"], pack_artifact["artifact_id"]],
            payload={
                "backend": self.name,
                "model_id": checkpoint_id,
                "eval_set_version": "reasoning-holdout-v1",
                "node_label": cfg.get("node_label", "local-node"),
                "git_sha": cfg.get("git_sha", "unknown"),
                "eval_before": before_eval,
                "eval_after": after_eval,
                "holdout_metric": after_eval["contract_score"] - before_eval["contract_score"],
                "metric_summary": {
                    "before_success_rate": before_eval["success_rate"],
                    "after_success_rate": after_eval["success_rate"],
                    "before_accepted_rate": before_eval["accepted_rate"],
                    "after_accepted_rate": after_eval["accepted_rate"],
                    "before_contract_score": before_eval["contract_score"],
                    "after_contract_score": after_eval["contract_score"],
                },
            },
        )
        checkpoint_artifact = make_artifact(
            artifact_type="checkpoint",
            producer_id=cfg["trainer_id"],
            producer_role="learner",
            policy_id=policy_artifact["artifact_id"],
            window_id=window_id + 1,
            lane="distill",
            commodity_use="dual_use",
            parent_ids=[policy_artifact["artifact_id"], pack_artifact["artifact_id"]],
            payload={
                "checkpoint_id": checkpoint_id,
                "checkpoint_path": str(checkpoint_dir),
                "parent_policy_id": policy_artifact["artifact_id"],
                "base_model": policy_artifact["payload"]["model_ref"],
                "adapter_type": "symbolic_capabilities",
                "train_recipe": {
                    "backend": self.name,
                    "record_count": capabilities["record_count"],
                    "supported_operations": capabilities["supported_operations"],
                },
                "git_sha": cfg.get("git_sha", "unknown"),
                "window_id": window_id + 1,
                "training_backend": self.name,
                "pack_artifact_id": pack_artifact["artifact_id"],
                "eval_before": before_eval,
                "eval_after": after_eval,
            },
        )
        registry.put_artifact(benchmark_artifact)
        registry.put_artifact(checkpoint_artifact)
        return {
            "checkpoint_artifact": checkpoint_artifact,
            "benchmark_result_artifact": benchmark_artifact,
            "eval_before": before_eval,
            "eval_after": after_eval,
        }


class LocalGRPOLearnerBackend:
    name = "local_grpo"

    def train(
        self,
        *,
        model_ref: str,
        groups: list[dict[str, Any]],
        output_dir: Path,
        device: str,
        lr: float,
        clip_eps: float,
        kl_coef: float,
        epochs: int,
    ) -> dict[str, Any]:
        from ..training.grpo import train_grpo_checkpoint

        return train_grpo_checkpoint(
            model_ref=model_ref,
            groups=groups,
            output_dir=output_dir,
            device=device,
            lr=lr,
            clip_eps=clip_eps,
            kl_coef=kl_coef,
            epochs=epochs,
        )


class AsyncGRPOBackend:
    name = "trl_async_grpo"

    def train(self, **kwargs) -> dict[str, Any]:
        return {
            "backend": self.name,
            "status": "planned",
            "detail": "External TRL AsyncGRPO execution must be configured with vLLM before use.",
        }


class TransformersSFTLearnerBackend:
    name = "transformers_sft"

    def train(
        self,
        *,
        registry: ArtifactRegistry,
        cfg: dict[str, Any],
        policy_artifact: dict[str, Any],
        pack_artifact: dict[str, Any],
        window_id: int,
    ) -> dict[str, Any]:
        try:
            import inspect
            import torch
            from datasets import Dataset
            from peft import LoraConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from trl import SFTConfig, SFTTrainer
        except ImportError as exc:
            raise RuntimeError(
                "Transformers SFT backend requires the optional ml and gpu extras."
            ) from exc

        base_model = policy_artifact["payload"]["model_ref"]
        before_eval = evaluate_reasoning_checkpoint(
            model_ref=base_model,
            holdout_count=cfg["distill_holdout_count"],
            cfg=cfg,
        )
        checkpoint_id = f"distill-checkpoint-window-{window_id + 1:04d}"
        checkpoint_dir = registry.checkpoint_dir(checkpoint_id)
        adapter_dir = checkpoint_dir / "adapter"
        merged_dir = checkpoint_dir / "merged"
        rows = _pack_rows(pack_artifact)
        if not rows:
            raise RuntimeError("Distillation pack contains zero accepted rows; skipping learner run.")
        tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": True}
        path_obj = Path(base_model)
        if path_obj.exists():
            base_model = str(path_obj.resolve())
            tokenizer_kwargs["local_files_only"] = True
        tokenizer = AutoTokenizer.from_pretrained(base_model, **tokenizer_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dataset = Dataset.from_list(
            [
                {
                    "text": render_reasoning_conversation(
                        row["prompt"],
                        assistant_text=row["trace_text"],
                        tokenizer=tokenizer,
                    )
                }
                for row in rows
            ]
        )
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        if path_obj.exists():
            model_kwargs["local_files_only"] = True
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        if torch.cuda.is_available():
            model = model.to("cuda")
        if hasattr(model, "config"):
            model.config.use_cache = False
        peft_config = LoraConfig(
            r=int(cfg["distill_lora_rank"]),
            lora_alpha=int(cfg["distill_lora_alpha"]),
            lora_dropout=float(cfg["distill_lora_dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
        train_recipe = {
            "backend": self.name,
            "epochs": float(cfg["distill_sft_epochs"]),
            "per_device_batch_size": int(cfg["distill_sft_batch_size"]),
            "gradient_accumulation_steps": int(cfg["distill_sft_grad_accum"]),
            "learning_rate": float(cfg["distill_sft_learning_rate"]),
            "max_seq_length": int(cfg["distill_sft_max_seq_length"]),
            "gradient_checkpointing": True,
            "bf16": bool(torch.cuda.is_available()),
            "peft": {
                "r": int(cfg["distill_lora_rank"]),
                "alpha": int(cfg["distill_lora_alpha"]),
                "dropout": float(cfg["distill_lora_dropout"]),
            },
        }
        sft_config_kwargs: dict[str, Any] = {
            "output_dir": str(adapter_dir),
            "num_train_epochs": float(cfg["distill_sft_epochs"]),
            "learning_rate": float(cfg["distill_sft_learning_rate"]),
            "per_device_train_batch_size": int(cfg["distill_sft_batch_size"]),
            "gradient_accumulation_steps": int(cfg["distill_sft_grad_accum"]),
            "bf16": bool(torch.cuda.is_available()),
            "gradient_checkpointing": True,
            "save_strategy": "no",
            "logging_steps": 10,
            "report_to": [],
        }
        length_key = (
            "max_length"
            if "max_length" in inspect.signature(SFTConfig.__init__).parameters
            else "max_seq_length"
        )
        sft_config_kwargs[length_key] = int(cfg["distill_sft_max_seq_length"])
        trainer = SFTTrainer(
            model=model,
            args=SFTConfig(**sft_config_kwargs),
            train_dataset=dataset,
            processing_class=tokenizer,
            formatting_func=lambda sample: sample["text"],
            peft_config=peft_config,
        )
        trainer.train()
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        merged_model = trainer.model
        if hasattr(merged_model, "merge_and_unload"):
            merged_model = merged_model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        del merged_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        after_eval = evaluate_reasoning_checkpoint(
            model_ref=str(merged_dir),
            holdout_count=cfg["distill_holdout_count"],
            cfg=cfg,
        )
        benchmark_artifact = make_artifact(
            artifact_type="benchmark_result",
            producer_id=cfg["validator_id"],
            producer_role="benchmark_validator",
            policy_id=policy_artifact["artifact_id"],
            window_id=window_id,
            lane="distill",
            commodity_use="dual_use",
            parent_ids=[policy_artifact["artifact_id"], pack_artifact["artifact_id"]],
            payload={
                "backend": self.name,
                "model_id": checkpoint_id,
                "eval_set_version": "reasoning-holdout-v1",
                "node_label": cfg.get("node_label", "local-node"),
                "git_sha": cfg.get("git_sha", "unknown"),
                "eval_before": before_eval,
                "eval_after": after_eval,
                "holdout_metric": after_eval["contract_score"] - before_eval["contract_score"],
                "metric_summary": {
                    "before_success_rate": before_eval["success_rate"],
                    "after_success_rate": after_eval["success_rate"],
                    "before_accepted_rate": before_eval["accepted_rate"],
                    "after_accepted_rate": after_eval["accepted_rate"],
                    "before_contract_score": before_eval["contract_score"],
                    "after_contract_score": after_eval["contract_score"],
                },
            },
        )
        checkpoint_artifact = make_artifact(
            artifact_type="checkpoint",
            producer_id=cfg["trainer_id"],
            producer_role="learner",
            policy_id=policy_artifact["artifact_id"],
            window_id=window_id + 1,
            lane="distill",
            commodity_use="dual_use",
            parent_ids=[policy_artifact["artifact_id"], pack_artifact["artifact_id"]],
            payload={
                "checkpoint_id": checkpoint_id,
                "checkpoint_path": str(merged_dir),
                "adapter_path": str(adapter_dir),
                "parent_policy_id": policy_artifact["artifact_id"],
                "base_model": base_model,
                "adapter_type": "lora",
                "train_recipe": train_recipe,
                "git_sha": cfg.get("git_sha", "unknown"),
                "window_id": window_id + 1,
                "training_backend": self.name,
                "pack_artifact_id": pack_artifact["artifact_id"],
                "eval_before": before_eval,
                "eval_after": after_eval,
            },
        )
        registry.put_artifact(benchmark_artifact)
        registry.put_artifact(checkpoint_artifact)
        write_json(
            checkpoint_dir / "training_summary.json",
            {
                "backend": self.name,
                "base_model": base_model,
                "rows": len(rows),
                "train_recipe": train_recipe,
            },
        )
        return {
            "checkpoint_artifact": checkpoint_artifact,
            "benchmark_result_artifact": benchmark_artifact,
            "eval_before": before_eval,
            "eval_after": after_eval,
        }


class NeMoLearnerBackend:
    name = "nemo_rl"

    def train(self, **kwargs) -> dict[str, Any]:
        return {
            "backend": self.name,
            "status": "planned",
            "detail": "NeMo RL backend is reserved for multi-node frontier integration milestones.",
        }


def build_distill_learner_backend(name: str):
    if name == "local_reasoning_student":
        return LocalReasoningStudentBackend()
    if name == "transformers_sft":
        return TransformersSFTLearnerBackend()
    return LocalReasoningStudentBackend()


def build_rl_learner_backend(name: str):
    if name == "trl_async_grpo":
        return AsyncGRPOBackend()
    if name == "nemo_rl":
        return NeMoLearnerBackend()
    return LocalGRPOLearnerBackend()
