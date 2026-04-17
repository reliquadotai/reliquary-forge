"""Unit tests for reliquary.shared.config.load_runtime_config.

The loader reads env vars and returns a dict of runtime knobs. These
tests pin the default values + env-var override behavior by
monkeypatching os.environ so an accidental rename or default drift is
caught at CI.
"""

from __future__ import annotations

from typing import Any

import pytest

from reliquary.shared.config import load_runtime_config


def _clear_reliquary_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear every RELIQUARY_* + NETWORK/NETUID/WALLET/HOTKEY/LOG_LEVEL
    env var so defaults are exercised deterministically."""
    import os

    for key in list(os.environ):
        if key.startswith("RELIQUARY_") or key in {
            "NETWORK",
            "NETUID",
            "WALLET_NAME",
            "HOTKEY_NAME",
            "LOG_LEVEL",
        }:
            monkeypatch.delenv(key, raising=False)


def test_defaults_apply_when_nothing_set(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reliquary_env(monkeypatch)
    config = load_runtime_config()
    assert config["device"] == "cpu"
    assert config["state_root"] == "./data"
    assert config["miner_id"] == "local-miner"
    assert config["validator_id"] == "local-validator"
    assert config["trainer_id"] == "local-trainer"
    assert config["network"] == "finney"
    assert config["netuid"] == 1


def test_env_overrides_state_root(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reliquary_env(monkeypatch)
    monkeypatch.setenv("RELIQUARY_STATE_ROOT", "/custom/root")
    config = load_runtime_config()
    assert config["state_root"] == "/custom/root"
    assert config["artifact_dir"] == "/custom/root/artifacts"
    assert config["export_dir"] == "/custom/root/exports"
    assert config["log_dir"] == "/custom/root/logs"


def test_env_data_dir_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reliquary_env(monkeypatch)
    monkeypatch.setenv("RELIQUARY_DATA_DIR", "/fallback/data")
    config = load_runtime_config()
    assert config["state_root"] == "/fallback/data"


def test_artifact_dir_env_overrides_state_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_reliquary_env(monkeypatch)
    monkeypatch.setenv("RELIQUARY_STATE_ROOT", "/custom/root")
    monkeypatch.setenv("RELIQUARY_ARTIFACT_DIR", "/special/artifacts")
    config = load_runtime_config()
    assert config["state_root"] == "/custom/root"
    assert config["artifact_dir"] == "/special/artifacts"


def test_model_name_default_and_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_reliquary_env(monkeypatch)
    default_config = load_runtime_config()
    assert default_config["model_name"] == "sshleifer/tiny-gpt2"

    monkeypatch.setenv("RELIQUARY_MODEL_NAME", "qwen3-4b")
    config = load_runtime_config()
    assert config["model_name"] == "qwen3-4b"


def test_model_teacher_defaults_to_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_reliquary_env(monkeypatch)
    monkeypatch.setenv("RELIQUARY_MODEL_NAME", "base-model")
    config = load_runtime_config()
    assert config["model_teacher"] == "base-model"


def test_model_teacher_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reliquary_env(monkeypatch)
    monkeypatch.setenv("RELIQUARY_MODEL_NAME", "base-model")
    monkeypatch.setenv("RELIQUARY_MODEL_TEACHER", "teacher-override")
    config = load_runtime_config()
    assert config["model_teacher"] == "teacher-override"


def test_model_student_default_is_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_reliquary_env(monkeypatch)
    config = load_runtime_config()
    assert config["model_student"] == "reasoning-student://bootstrap"


def test_control_api_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reliquary_env(monkeypatch)
    config = load_runtime_config()
    assert config["control_api_host"] == "127.0.0.1"
    assert config["control_api_port"] == 8000


def test_control_api_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reliquary_env(monkeypatch)
    monkeypatch.setenv("RELIQUARY_API_BIND", "0.0.0.0")
    monkeypatch.setenv("RELIQUARY_API_PORT", "9000")
    config = load_runtime_config()
    assert config["control_api_host"] == "0.0.0.0"
    assert config["control_api_port"] == 9000


def test_r2_defaults_are_empty_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reliquary_env(monkeypatch)
    config = load_runtime_config()
    for key in (
        "r2_bucket",
        "r2_endpoint_url",
        "r2_access_key_id",
        "r2_secret_access_key",
    ):
        assert config[key] == ""


def test_r2_region_default_is_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reliquary_env(monkeypatch)
    config = load_runtime_config()
    assert config["r2_region"] == "auto"


def test_r2_prefix_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reliquary_env(monkeypatch)
    config = load_runtime_config()
    assert config["r2_prefix"] == "reliquary"


def test_int_env_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reliquary_env(monkeypatch)
    monkeypatch.setenv("RELIQUARY_WINDOW_LENGTH", "60")
    monkeypatch.setenv("RELIQUARY_MAX_NEW_TOKENS", "128")
    config = load_runtime_config()
    assert config["window_length"] == 60
    assert config["max_new_tokens"] == 128


def test_float_env_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reliquary_env(monkeypatch)
    monkeypatch.setenv("RELIQUARY_TEMPERATURE", "0.7")
    monkeypatch.setenv("RELIQUARY_TOP_P", "0.9")
    config = load_runtime_config()
    assert config["temperature"] == pytest.approx(0.7)
    assert config["top_p"] == pytest.approx(0.9)


def test_policy_authority_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reliquary_env(monkeypatch)
    config = load_runtime_config()
    assert config["policy_authority_mode"] == "trusted_quorum"
    assert config["policy_authority_quorum"] == "local-trainer,local-validator"
    assert config["policy_authority_min_signers"] == 2


def test_network_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reliquary_env(monkeypatch)
    config = load_runtime_config()
    assert config["network"] == "finney"
    assert config["netuid"] == 1
    assert config["wallet_name"] == "default"
    assert config["hotkey_name"] == "default"


def test_network_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reliquary_env(monkeypatch)
    monkeypatch.setenv("NETWORK", "test")
    monkeypatch.setenv("NETUID", "81")
    monkeypatch.setenv("WALLET_NAME", "custom-wallet")
    monkeypatch.setenv("HOTKEY_NAME", "custom-hotkey")
    config = load_runtime_config()
    assert config["network"] == "test"
    assert config["netuid"] == 81
    assert config["wallet_name"] == "custom-wallet"
    assert config["hotkey_name"] == "custom-hotkey"


def test_log_level_env_and_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reliquary_env(monkeypatch)
    assert load_runtime_config()["log_level"] == "INFO"
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    assert load_runtime_config()["log_level"] == "DEBUG"


def test_git_sha_env_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reliquary_env(monkeypatch)
    monkeypatch.setenv("RELIQUARY_GIT_SHA", "deadbeef" * 5)
    config = load_runtime_config()
    assert config["git_sha"] == "deadbeef" * 5


def test_config_has_rl_backend_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reliquary_env(monkeypatch)
    config = load_runtime_config()
    assert config["rl_env_id"] == "reasoning"
    assert config["rl_learner_backend"] == "trl_async_grpo_fsdp2"
    assert config["rl_backend_execution_mode"] == "local_compat"


def test_config_has_distill_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reliquary_env(monkeypatch)
    config = load_runtime_config()
    assert config["distill_profile"] == "distill"
    assert config["distill_pack_format"] == "jsonl,parquet"
    assert config["distill_lora_rank"] == 16
    assert config["distill_lora_alpha"] == 32


def test_config_has_ray_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reliquary_env(monkeypatch)
    config = load_runtime_config()
    assert config["ray_cluster_profile"] == "single-node"
    assert config["ray_head_address"] == "local"
    assert config["ray_actor_replicas"] == 1


def test_config_returns_flat_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_reliquary_env(monkeypatch)
    config = load_runtime_config()
    assert isinstance(config, dict)
    # All values are simple scalars (str/int/float).
    for key, value in config.items():
        assert isinstance(value, (str, int, float)), f"{key!r} -> {type(value)!r}"
