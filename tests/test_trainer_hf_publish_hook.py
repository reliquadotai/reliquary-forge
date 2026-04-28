"""Tests for the Phase 1.4 trainer wiring of HfHubPublisher.

Covers the env-driven enable/disable + the maybe-publish hook semantics
without exercising the heavy ``train_window`` path. The actual upload
is mocked via the publisher's dependency-injected ``api`` slot, so
these tests run on CPU + plain stdlib.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from reliquary.roles.trainer import (
    _hf_hub_publisher_for,
    _maybe_publish_to_hf_hub,
)


def _empty_cfg() -> dict[str, Any]:
    return {}


def _snapshot_dir(tmp_path: Path) -> Path:
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    return snap


# ────────  _hf_hub_publisher_for  ────────


def test_publisher_disabled_when_repo_id_unset(monkeypatch) -> None:
    monkeypatch.delenv("RELIQUARY_HF_REPO_ID", raising=False)
    assert _hf_hub_publisher_for(_empty_cfg()) is None


def test_publisher_disabled_when_repo_id_blank(monkeypatch) -> None:
    monkeypatch.setenv("RELIQUARY_HF_REPO_ID", "   ")
    assert _hf_hub_publisher_for(_empty_cfg()) is None


def test_publisher_enabled_when_repo_id_set(monkeypatch) -> None:
    monkeypatch.setenv("RELIQUARY_HF_REPO_ID", "reliquadotai/reliquary-test")
    monkeypatch.setenv("RELIQUARY_HF_TOKEN", "fake-hf-token")
    monkeypatch.setenv("RELIQUARY_HF_PUBLISH_INTERVAL_WINDOWS", "5")
    monkeypatch.setenv("RELIQUARY_TRAINER_SECRET", "trainer-secret-x")
    cfg: dict[str, Any] = {}
    pub = _hf_hub_publisher_for(cfg)
    assert pub is not None
    assert pub.repo_id == "reliquadotai/reliquary-test"
    assert pub.interval_windows == 5
    # Caches on cfg.
    assert cfg["_hf_hub_publisher"] is pub
    pub.shutdown()


def test_publisher_caches_singleton_per_cfg(monkeypatch) -> None:
    monkeypatch.setenv("RELIQUARY_HF_REPO_ID", "reliquadotai/reliquary-test")
    monkeypatch.setenv("RELIQUARY_TRAINER_SECRET", "x")
    cfg: dict[str, Any] = {}
    pub1 = _hf_hub_publisher_for(cfg)
    pub2 = _hf_hub_publisher_for(cfg)
    assert pub1 is pub2
    pub1.shutdown()


def test_publisher_default_interval_when_env_invalid(monkeypatch) -> None:
    monkeypatch.setenv("RELIQUARY_HF_REPO_ID", "reliquadotai/reliquary-test")
    monkeypatch.setenv("RELIQUARY_HF_PUBLISH_INTERVAL_WINDOWS", "not-a-number")
    monkeypatch.setenv("RELIQUARY_TRAINER_SECRET", "x")
    pub = _hf_hub_publisher_for(_empty_cfg())
    assert pub is not None
    assert pub.interval_windows == 10
    pub.shutdown()


def test_publisher_falls_back_to_miner_secret(monkeypatch) -> None:
    monkeypatch.setenv("RELIQUARY_HF_REPO_ID", "reliquadotai/reliquary-test")
    monkeypatch.delenv("RELIQUARY_HF_SIGNING_SECRET", raising=False)
    monkeypatch.delenv("RELIQUARY_TRAINER_SECRET", raising=False)
    monkeypatch.setenv("RELIQUARY_MINER_SECRET", "miner-secret-y")
    pub = _hf_hub_publisher_for(_empty_cfg())
    assert pub is not None
    pub.shutdown()


# ────────  _maybe_publish_to_hf_hub  ────────


def test_maybe_publish_returns_none_when_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("RELIQUARY_HF_REPO_ID", raising=False)
    out = _maybe_publish_to_hf_hub(
        _empty_cfg(), window_n=10, snapshot_dir=str(_snapshot_dir(tmp_path))
    )
    assert out is None


def test_maybe_publish_returns_none_off_interval(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RELIQUARY_HF_REPO_ID", "reliquadotai/reliquary-test")
    monkeypatch.setenv("RELIQUARY_HF_PUBLISH_INTERVAL_WINDOWS", "10")
    monkeypatch.setenv("RELIQUARY_TRAINER_SECRET", "x")
    cfg: dict[str, Any] = {}
    out = _maybe_publish_to_hf_hub(
        cfg, window_n=7, snapshot_dir=str(_snapshot_dir(tmp_path))
    )
    assert out is None
    cfg["_hf_hub_publisher"].shutdown()


def test_maybe_publish_returns_none_when_snapshot_missing(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RELIQUARY_HF_REPO_ID", "reliquadotai/reliquary-test")
    monkeypatch.setenv("RELIQUARY_TRAINER_SECRET", "x")
    cfg: dict[str, Any] = {}
    out = _maybe_publish_to_hf_hub(
        cfg, window_n=10, snapshot_dir=str(tmp_path / "missing")
    )
    assert out is None
    cfg["_hf_hub_publisher"].shutdown()


def test_maybe_publish_schedules_on_interval_window(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RELIQUARY_HF_REPO_ID", "reliquadotai/reliquary-test")
    monkeypatch.setenv("RELIQUARY_HF_PUBLISH_INTERVAL_WINDOWS", "10")
    monkeypatch.setenv("RELIQUARY_TRAINER_SECRET", "x")
    cfg: dict[str, Any] = {}
    snap = _snapshot_dir(tmp_path)
    # Inject a fake api into the cached publisher so we don't hit HF.
    pub = _hf_hub_publisher_for(cfg)
    fake_api = MagicMock()
    fake_api.upload_folder.return_value = MagicMock(oid="abc1234567")
    pub._api = fake_api  # noqa: SLF001 — test boundary

    out = _maybe_publish_to_hf_hub(cfg, window_n=10, snapshot_dir=str(snap))
    assert out is not None
    assert out["status"] == "scheduled"
    assert out["window_n"] == 10
    assert out["repo_id"] == "reliquadotai/reliquary-test"
    assert out["interval_windows"] == 10
    pub.shutdown()
