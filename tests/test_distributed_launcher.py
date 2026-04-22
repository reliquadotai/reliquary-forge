"""Tests for :mod:`reliquary.training.distributed_launcher`."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from reliquary.training.checkpoint_storage import LocalFilesystemBackend
from reliquary.training.distributed import DistributedConfig
from reliquary.training.distributed_launcher import (
    LaunchConfig,
    LaunchResult,
    main,
    run_diloco_harness,
)

# ---------------------------------------------------------------------------
# LaunchConfig env parsing
# ---------------------------------------------------------------------------


def _base_env(backend_root: str):
    return {
        "RELIQUARY_DILOCO_RUN_ID": "run-0001",
        "RELIQUARY_DILOCO_WINDOW_ID": "5",
        "RELIQUARY_DILOCO_BACKEND_ROOT": backend_root,
    }


def test_launch_config_from_env_minimal(tmp_path):
    cfg = LaunchConfig.from_env(_base_env(str(tmp_path)))
    assert cfg.run_id == "run-0001"
    assert cfg.window_id == 5
    assert cfg.backend_root == str(tmp_path)
    assert cfg.publish is False
    # Defaults:
    assert cfg.inner_steps == 30
    assert cfg.outer_lr == pytest.approx(0.4)
    assert cfg.outer_momentum == pytest.approx(0.95)
    assert cfg.inference_netuid == 81
    assert cfg.training_netuid == 3


def test_launch_config_from_env_publish_and_consumed(tmp_path):
    env = dict(_base_env(str(tmp_path)))
    env.update(
        {
            "RELIQUARY_DILOCO_PUBLISH": "1",
            "RELIQUARY_DILOCO_CONSUMED_WINDOWS": "10,11,12",
            "RELIQUARY_DILOCO_CONSUMED_KEYS": "k1,k2,k3",
            "RELIQUARY_DILOCO_EFFECTIVE_AT_WINDOW": "100",
            "RELIQUARY_DILOCO_AUTHORITY_HOTKEY": "authority",
            "RELIQUARY_DILOCO_AUTHORITY_SECRET": "secret",
        }
    )
    cfg = LaunchConfig.from_env(env)
    assert cfg.publish is True
    assert cfg.consumed_rollout_windows == (10, 11, 12)
    assert cfg.consumed_rollout_keys == ("k1", "k2", "k3")
    assert cfg.effective_at_ledger_window == 100


def test_launch_config_requires_run_id(tmp_path):
    env = {
        "RELIQUARY_DILOCO_WINDOW_ID": "5",
        "RELIQUARY_DILOCO_BACKEND_ROOT": str(tmp_path),
    }
    with pytest.raises(RuntimeError, match="RELIQUARY_DILOCO_RUN_ID"):
        LaunchConfig.from_env(env)


def test_launch_config_empty_consumed_lists_ok(tmp_path):
    env = dict(_base_env(str(tmp_path)))
    env["RELIQUARY_DILOCO_CONSUMED_WINDOWS"] = ""
    cfg = LaunchConfig.from_env(env)
    assert cfg.consumed_rollout_windows == ()
    assert cfg.consumed_rollout_keys == ()


# ---------------------------------------------------------------------------
# run_diloco_harness — single-process smoke
# ---------------------------------------------------------------------------


def _single_proc_dist():
    return DistributedConfig(
        world_size=1, rank=0, local_rank=0, backend="gloo"
    )


def _linear_factory(*, config):
    torch.manual_seed(0)
    return torch.nn.Linear(4, 2, bias=False)


def _add_ones_trainer(*, model, rank, step):
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.full_like(p, 0.01))
    return 1.0 / (step + 1)


def test_harness_runs_inner_steps_single_process(tmp_path):
    launch = LaunchConfig.from_env(_base_env(str(tmp_path)))
    launch = _with_overrides(launch, inner_steps=5)

    result = run_diloco_harness(
        launch=launch,
        model_factory=_linear_factory,
        inner_trainer=_add_ones_trainer,
        distributed=_single_proc_dist(),
    )
    assert isinstance(result, LaunchResult)
    assert result.inner_steps_executed == 5
    assert result.outer_applied is True
    assert result.publication is None  # publish=False by default
    assert result.delta_bundle is None  # publish=False
    assert result.metrics_counters["reliquary_launcher_outer_applies_total"] == 1


def test_harness_publishes_and_attests_when_enabled(tmp_path):
    """Publish + attestation wired end-to-end on rank 0."""
    launch = LaunchConfig.from_env(_base_env(str(tmp_path / "backend_root")))
    launch = _with_overrides(
        launch,
        inner_steps=3,
        publish=True,
        attestation_authority_hotkey="forge-authority",
        attestation_authority_secret="authority-secret",
        effective_at_ledger_window=100,
        consumed_rollout_windows=(10, 11),
        consumed_rollout_keys=(
            "rollouts/81/10-10/val.json",
            "rollouts/81/11-11/val.json",
        ),
    )
    result = run_diloco_harness(
        launch=launch,
        model_factory=_linear_factory,
        inner_trainer=_add_ones_trainer,
        distributed=_single_proc_dist(),
    )
    assert result.publication is not None
    assert result.delta_bundle is not None
    assert (
        result.metrics_counters["reliquary_launcher_publish_ok_total"] == 1
    )
    # Published artifacts are on disk.
    backend = LocalFilesystemBackend(tmp_path / "backend_root")
    keys = backend.list("")
    assert any(k.startswith(f"checkpoints/{launch.run_id}/") for k in keys)
    assert any(k.startswith(f"attestations/{launch.training_netuid}/") for k in keys)
    assert any(k.startswith(f"commitments/{launch.inference_netuid}/") for k in keys)


def test_harness_delta_is_non_zero(tmp_path):
    """The synthetic trainer adds 0.01 each step → post-params differ from pre."""
    launch = LaunchConfig.from_env(_base_env(str(tmp_path)))
    launch = _with_overrides(launch, inner_steps=4, publish=True)

    result = run_diloco_harness(
        launch=launch,
        model_factory=_linear_factory,
        inner_trainer=_add_ones_trainer,
        distributed=_single_proc_dist(),
    )
    assert result.delta_bundle is not None
    assert result.metrics_counters["reliquary_launcher_delta_bytes_last"] > 0


def test_harness_skips_attestation_when_secret_missing(tmp_path):
    """publish=True but no authority_secret → delta is published, attestation is not."""
    launch = LaunchConfig.from_env(_base_env(str(tmp_path)))
    launch = _with_overrides(launch, publish=True, inner_steps=2)
    # attestation_authority_secret defaults to empty.
    result = run_diloco_harness(
        launch=launch,
        model_factory=_linear_factory,
        inner_trainer=_add_ones_trainer,
        distributed=_single_proc_dist(),
    )
    assert result.delta_bundle is not None
    assert result.publication is None


def test_harness_records_last_loss(tmp_path):
    launch = LaunchConfig.from_env(_base_env(str(tmp_path)))
    launch = _with_overrides(launch, inner_steps=3)
    result = run_diloco_harness(
        launch=launch,
        model_factory=_linear_factory,
        inner_trainer=lambda *, model, rank, step: float(step) * 0.5,
        distributed=_single_proc_dist(),
    )
    # Trainer returned 0, 0.5, 1.0 across 3 steps → last_loss is 1.0.
    assert result.last_loss == pytest.approx(1.0)


def test_harness_rank_propagation(tmp_path):
    """Rank in LaunchResult matches the distributed config we passed."""
    launch = LaunchConfig.from_env(_base_env(str(tmp_path)))
    launch = _with_overrides(launch, inner_steps=1)
    dist_cfg = DistributedConfig(
        world_size=1, rank=0, local_rank=0, backend="gloo"
    )
    result = run_diloco_harness(
        launch=launch,
        model_factory=_linear_factory,
        inner_trainer=_add_ones_trainer,
        distributed=dist_cfg,
    )
    assert result.rank == 0
    assert result.world_size == 1


# ---------------------------------------------------------------------------
# main() CLI — dry run path
# ---------------------------------------------------------------------------


def test_main_dry_run_exits_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("RELIQUARY_DILOCO_RUN_ID", "run-0001")
    monkeypatch.setenv("RELIQUARY_DILOCO_WINDOW_ID", "5")
    monkeypatch.setenv("RELIQUARY_DILOCO_BACKEND_ROOT", str(tmp_path))
    exit_code = main(["--dry-run"])
    assert exit_code == 0


def test_main_exits_2_on_missing_env(monkeypatch, capsys):
    monkeypatch.delenv("RELIQUARY_DILOCO_RUN_ID", raising=False)
    monkeypatch.delenv("RELIQUARY_DILOCO_WINDOW_ID", raising=False)
    monkeypatch.delenv("RELIQUARY_DILOCO_BACKEND_ROOT", raising=False)
    exit_code = main(["--dry-run"])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "RELIQUARY_DILOCO_RUN_ID" in err


# ---------------------------------------------------------------------------
# Shell wrapper smoke
# ---------------------------------------------------------------------------


def test_shell_wrapper_exists_and_executable():
    """The shell wrapper is present, executable, and references torch.distributed.run."""
    path = Path(__file__).resolve().parents[1] / "deploy" / "scripts" / "launch_diloco.sh"
    assert path.is_file(), f"missing shell wrapper at {path}"
    # File mode should include at least one executable bit.
    mode = path.stat().st_mode
    assert mode & 0o111, "launch_diloco.sh must be executable"
    body = path.read_text()
    assert "torch.distributed.run" in body
    assert "RELIQUARY_DILOCO_RUN_ID" in body


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _with_overrides(launch: LaunchConfig, **overrides) -> LaunchConfig:
    """Return a new :class:`LaunchConfig` with ``overrides`` applied."""
    from dataclasses import replace

    return replace(launch, **overrides)
