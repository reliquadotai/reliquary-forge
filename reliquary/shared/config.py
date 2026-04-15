from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _git_sha() -> str:
    env_value = os.getenv("RELIQUARY_GIT_SHA", "").strip()
    if env_value:
        return env_value
    package_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=package_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def load_runtime_config() -> dict:
    state_root = _env_str(
        "RELIQUARY_STATE_ROOT",
        _env_str("RELIQUARY_DATA_DIR", "./data"),
    )
    artifact_dir = _env_str("RELIQUARY_ARTIFACT_DIR", f"{state_root}/artifacts")
    export_dir = _env_str("RELIQUARY_EXPORT_DIR", f"{state_root}/exports")
    log_dir = _env_str("RELIQUARY_LOG_DIR", f"{state_root}/logs")
    model_name = _env_str("RELIQUARY_MODEL_NAME", "sshleifer/tiny-gpt2")
    model_teacher = _env_str("RELIQUARY_MODEL_TEACHER", model_name)
    model_student = _env_str(
        "RELIQUARY_MODEL_STUDENT",
        _env_str("RELIQUARY_DISTILL_STUDENT_MODEL_REF", "reasoning-student://bootstrap"),
    )
    control_api_host = _env_str(
        "RELIQUARY_API_BIND",
        _env_str("RELIQUARY_CONTROL_API_HOST", "127.0.0.1"),
    )
    control_api_port = _env_int(
        "RELIQUARY_API_PORT",
        _env_int("RELIQUARY_CONTROL_API_PORT", 8000),
    )
    return {
        "state_root": state_root,
        "artifact_dir": artifact_dir,
        "export_dir": export_dir,
        "log_dir": log_dir,
        "data_dir": artifact_dir,
        "device": _env_str("RELIQUARY_DEVICE", "cpu"),
        "model_name": model_name,
        "model_teacher": model_teacher,
        "model_student": model_student,
        "control_api_host": control_api_host,
        "control_api_port": control_api_port,
        "miner_id": _env_str("RELIQUARY_MINER_ID", "local-miner"),
        "validator_id": _env_str("RELIQUARY_VALIDATOR_ID", "local-validator"),
        "trainer_id": _env_str("RELIQUARY_TRAINER_ID", "local-trainer"),
        "miner_secret": _env_str("RELIQUARY_MINER_SECRET", "local-miner-secret"),
        "artifact_registry_backend": _env_str("RELIQUARY_ARTIFACT_REGISTRY_BACKEND", "local"),
        "metadata_index_backend": _env_str("RELIQUARY_METADATA_INDEX_BACKEND", "sqlite"),
        "metadata_index_dsn": _env_str("RELIQUARY_METADATA_INDEX_DSN", ""),
        "r2_bucket": _env_str("RELIQUARY_R2_BUCKET", ""),
        "r2_prefix": _env_str("RELIQUARY_R2_PREFIX", "reliquary"),
        "r2_region": _env_str("RELIQUARY_R2_REGION", "auto"),
        "r2_endpoint_url": _env_str("RELIQUARY_R2_ENDPOINT_URL", ""),
        "r2_access_key_id": _env_str("RELIQUARY_R2_ACCESS_KEY_ID", ""),
        "r2_secret_access_key": _env_str("RELIQUARY_R2_SECRET_ACCESS_KEY", ""),
        "redis_url": _env_str("RELIQUARY_REDIS_URL", "redis://127.0.0.1:6379/0"),
        "policy_authority_mode": _env_str("RELIQUARY_POLICY_AUTHORITY_MODE", "trusted_quorum"),
        "policy_authority_quorum": _env_str(
            "RELIQUARY_POLICY_AUTHORITY_QUORUM",
            "local-trainer,local-validator",
        ),
        "policy_authority_min_signers": _env_int("RELIQUARY_POLICY_AUTHORITY_MIN_SIGNERS", 2),
        "window_length": _env_int("RELIQUARY_WINDOW_LENGTH", 30),
        "rollouts_per_challenge": _env_int("RELIQUARY_ROLLOUTS_PER_CHALLENGE", 4),
        "max_new_tokens": _env_int("RELIQUARY_MAX_NEW_TOKENS", 48),
        "temperature": _env_float("RELIQUARY_TEMPERATURE", 1.0),
        "top_p": _env_float("RELIQUARY_TOP_P", 0.95),
        "top_k": _env_int("RELIQUARY_TOP_K", 50),
        "repetition_penalty": _env_float("RELIQUARY_REPETITION_PENALTY", 1.0),
        "sat_n": _env_int("RELIQUARY_SAT_N", 4),
        "sat_k": _env_int("RELIQUARY_SAT_K", 3),
        "sat_m": _env_int("RELIQUARY_SAT_M", 17),
        "train_lr": _env_float("RELIQUARY_TRAIN_LR", 1e-5),
        "train_epochs": _env_int("RELIQUARY_TRAIN_EPOCHS", 1),
        "clip_eps": _env_float("RELIQUARY_CLIP_EPS", 0.2),
        "kl_coef": _env_float("RELIQUARY_KL_COEF", 0.0),
        "rl_env_id": _env_str("RELIQUARY_RL_ENV_ID", "reasoning"),
        "rl_task_source": _env_str("RELIQUARY_RL_TASK_SOURCE", "reasoning_tasks"),
        "rl_actor_model_ref": _env_str("RELIQUARY_RL_ACTOR_MODEL_REF", model_name),
        "rl_reference_model_ref": _env_str("RELIQUARY_RL_REFERENCE_MODEL_REF", model_name),
        "rl_learner_backend": _env_str("RELIQUARY_RL_LEARNER_BACKEND", "trl_async_grpo_fsdp2"),
        "rl_backend_execution_mode": _env_str("RELIQUARY_RL_BACKEND_EXECUTION_MODE", "simulate"),
        "rl_model_family": _env_str("RELIQUARY_RL_MODEL_FAMILY", "qwen-dense"),
        "rl_reasoning_eval_count": _env_int("RELIQUARY_RL_REASONING_EVAL_COUNT", 8),
        "rl_holdout_metric_min": _env_float("RELIQUARY_RL_HOLDOUT_METRIC_MIN", 0.0),
        "rl_contract_score_min": _env_float("RELIQUARY_RL_CONTRACT_SCORE_MIN", 0.0),
        "rl_reward_weight_correctness": _env_float("RELIQUARY_RL_REWARD_WEIGHT_CORRECTNESS", 0.55),
        "rl_reward_weight_policy_compliance": _env_float("RELIQUARY_RL_REWARD_WEIGHT_POLICY_COMPLIANCE", 0.20),
        "rl_reward_weight_format_ok": _env_float("RELIQUARY_RL_REWARD_WEIGHT_FORMAT_OK", 0.15),
        "rl_reward_weight_accepted": _env_float("RELIQUARY_RL_REWARD_WEIGHT_ACCEPTED", 0.10),
        "inference_registry_backend": _env_str("RELIQUARY_INFERENCE_REGISTRY_BACKEND", "local"),
        "inference_registry_local_root": _env_str(
            "RELIQUARY_INFERENCE_REGISTRY_LOCAL_ROOT",
            "/srv/reliquary-inference/state/artifacts",
        ),
        "inference_registry_bucket": _env_str("RELIQUARY_INFERENCE_REGISTRY_BUCKET", ""),
        "inference_registry_endpoint_url": _env_str("RELIQUARY_INFERENCE_REGISTRY_ENDPOINT_URL", ""),
        "inference_registry_access_key_id": _env_str("RELIQUARY_INFERENCE_REGISTRY_ACCESS_KEY_ID", ""),
        "inference_registry_secret_access_key": _env_str("RELIQUARY_INFERENCE_REGISTRY_SECRET_ACCESS_KEY", ""),
        "inference_registry_region": _env_str("RELIQUARY_INFERENCE_REGISTRY_REGION", "auto"),
        "inference_import_cursor_name": _env_str("RELIQUARY_INFERENCE_IMPORT_CURSOR_NAME", "reasoning_rl"),
        "ray_cluster_profile": _env_str("RELIQUARY_RAY_CLUSTER_PROFILE", "single-node"),
        "ray_head_address": _env_str("RELIQUARY_RAY_HEAD_ADDRESS", "local"),
        "ray_actor_replicas": _env_int("RELIQUARY_RAY_ACTOR_REPLICAS", 1),
        "ray_learner_replicas": _env_int("RELIQUARY_RAY_LEARNER_REPLICAS", 1),
        "ray_evaluator_replicas": _env_int("RELIQUARY_RAY_EVALUATOR_REPLICAS", 1),
        "rl_vllm_actor_endpoints": _env_str("RELIQUARY_RL_VLLM_ACTOR_ENDPOINTS", ""),
        "nemo_execution_mode": _env_str("RELIQUARY_NEMO_EXECUTION_MODE", "simulate"),
        "rl_cycle_interval_seconds": _env_int("RELIQUARY_RL_CYCLE_INTERVAL_SECONDS", 900),
        "distill_learner_backend": _env_str(
            "RELIQUARY_DISTILL_LEARNER_BACKEND",
            "local_reasoning_student",
        ),
        "vllm_base_url": _env_str("RELIQUARY_VLLM_BASE_URL", ""),
        "vllm_model": _env_str("RELIQUARY_VLLM_MODEL", model_teacher),
        "vllm_gpu_memory_utilization": _env_float(
            "RELIQUARY_VLLM_GPU_MEMORY_UTILIZATION",
            0.65,
        ),
        "vllm_temperature": _env_float("RELIQUARY_VLLM_TEMPERATURE", 0.2),
        "vllm_timeout_seconds": _env_int("RELIQUARY_VLLM_TIMEOUT_SECONDS", 120),
        "teacher_backend": _env_str("RELIQUARY_TEACHER_BACKEND", "local_reasoning_teacher"),
        "distill_student_model_ref": model_student,
        "distill_holdout_count": _env_int("RELIQUARY_DISTILL_HOLDOUT_COUNT", 8),
        "distill_train_task_count": _env_int("RELIQUARY_DISTILL_TRAIN_TASK_COUNT", 6),
        "distill_samples_per_task": _env_int("RELIQUARY_DISTILL_SAMPLES_PER_TASK", 1),
        "distill_profile": _env_str("RELIQUARY_DISTILL_PROFILE", "distill"),
        "distill_generation_miner_limit": _env_int("RELIQUARY_DISTILL_GENERATION_MINER_LIMIT", 0),
        "distill_pack_format": _env_str("RELIQUARY_DISTILL_PACK_FORMAT", "jsonl,parquet"),
        "distill_sft_epochs": _env_float("RELIQUARY_DISTILL_SFT_EPOCHS", 1.0),
        "distill_sft_batch_size": _env_int("RELIQUARY_DISTILL_SFT_BATCH_SIZE", 2),
        "distill_sft_grad_accum": _env_int("RELIQUARY_DISTILL_SFT_GRAD_ACCUM", 8),
        "distill_sft_learning_rate": _env_float("RELIQUARY_DISTILL_SFT_LR", 2e-5),
        "distill_sft_max_seq_length": _env_int("RELIQUARY_DISTILL_SFT_MAX_SEQ_LENGTH", 1024),
        "distill_lora_rank": _env_int("RELIQUARY_DISTILL_LORA_RANK", 16),
        "distill_lora_alpha": _env_int("RELIQUARY_DISTILL_LORA_ALPHA", 32),
        "distill_lora_dropout": _env_float("RELIQUARY_DISTILL_LORA_DROPOUT", 0.05),
        "distill_cycle_interval_seconds": _env_int("RELIQUARY_DISTILL_CYCLE_INTERVAL_SECONDS", 900),
        "node_label": _env_str("RELIQUARY_NODE_LABEL", "local-node"),
        "host_mode": _env_str("RELIQUARY_HOST_MODE", "local"),
        "service_version": _env_str("RELIQUARY_SERVICE_VERSION", "0.1.0"),
        "git_sha": _git_sha(),
        "network": _env_str("NETWORK", "finney"),
        "netuid": _env_int("NETUID", 1),
        "wallet_name": _env_str("WALLET_NAME", "default"),
        "hotkey_name": _env_str("HOTKEY_NAME", "default"),
        "log_level": _env_str("LOG_LEVEL", "INFO"),
    }
