"""Tests for LocalArtifactRegistry + SqliteMetadataIndex + ArtifactRegistry base.

Exercises the full put/get/list cycle, window-id filters, policy
ordering, directory helpers, and export-artifacts copy behavior. All
tests use ``tmp_path`` so filesystem state is fully isolated.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from reliquary.protocol.artifacts import (
    finalize_policy_artifact,
    make_artifact,
    make_policy_artifact,
)
from reliquary.storage.local_registry import LocalArtifactRegistry
from reliquary.storage.metadata_index import SqliteMetadataIndex
from reliquary.storage.registry import ArtifactRegistry


def _rollout(window_id: int = 1, payload_seed: int = 0) -> dict:
    return make_artifact(
        artifact_type="rollout",
        producer_id="miner-1",
        policy_id="policy-1",
        window_id=window_id,
        payload={"seed": payload_seed},
        created_at=f"2026-04-18T{window_id:02d}:00:00+00:00",
    )


def _policy(window_id: int = 1) -> dict:
    draft = make_policy_artifact(
        producer_id="policy-auth",
        window_id=window_id,
        model_ref="qwen3-4b",
        env_id="sat",
        env_config={"window": window_id},
        generation_config={},
        trainer_config={},
    )
    return finalize_policy_artifact(draft)


# ---------------------------------------------------------------------------
# LocalArtifactRegistry construction
# ---------------------------------------------------------------------------


def test_local_registry_creates_subdirectories(tmp_path: Path) -> None:
    LocalArtifactRegistry(str(tmp_path))
    for name in [
        "policies",
        "rollouts",
        "distill_traces",
        "verdicts",
        "checkpoints",
        "scorecards",
        "metadata",
    ]:
        assert (tmp_path / name).is_dir()


def test_local_registry_default_metadata_index_sqlite(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    assert isinstance(registry.metadata_index, SqliteMetadataIndex)
    assert Path(registry.metadata_index.db_path).parent.is_dir()


# ---------------------------------------------------------------------------
# put_artifact + get_artifact
# ---------------------------------------------------------------------------


def test_put_artifact_writes_json_file(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    artifact = _rollout(window_id=5, payload_seed=3)

    path = registry.put_artifact(artifact)
    assert path.is_file()
    assert path.name == f"{artifact['artifact_id']}.json"
    assert path.parent.name == "rollouts"


def test_put_artifact_round_trip(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    artifact = _rollout(window_id=3, payload_seed=1)
    registry.put_artifact(artifact)

    fetched = registry.get_artifact("rollout", artifact["artifact_id"])
    assert fetched == artifact


def test_get_artifact_missing_returns_none(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    assert registry.get_artifact("rollout", "deadbeef" * 8) is None


def test_put_artifact_records_metadata_index(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    artifact = _rollout(window_id=7, payload_seed=9)
    registry.put_artifact(artifact)

    record = registry.metadata_index.get_record("rollout", artifact["artifact_id"])
    assert record is not None
    assert record["storage_kind"] == "local"
    assert record["window_id"] == 7
    assert Path(record["local_path"]).is_file()


# ---------------------------------------------------------------------------
# list_artifacts
# ---------------------------------------------------------------------------


def test_list_artifacts_returns_all_for_type(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    for i in range(3):
        registry.put_artifact(_rollout(window_id=i, payload_seed=i))

    listed = registry.list_artifacts("rollout")
    assert len(listed) == 3
    assert {artifact["window_id"] for artifact in listed} == {0, 1, 2}


def test_list_artifacts_filters_by_window_id(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    for i in range(4):
        registry.put_artifact(_rollout(window_id=i, payload_seed=i))

    listed = registry.list_artifacts("rollout", window_id=2)
    assert len(listed) == 1
    assert listed[0]["window_id"] == 2


def test_list_artifacts_empty_type(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    assert registry.list_artifacts("rollout") == []


# ---------------------------------------------------------------------------
# latest_policy
# ---------------------------------------------------------------------------


def test_latest_policy_none_when_absent(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    assert registry.latest_policy() is None


def test_latest_policy_returns_highest_window(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    p1 = _policy(window_id=1)
    p5 = _policy(window_id=5)
    p3 = _policy(window_id=3)
    registry.put_artifact(p1)
    registry.put_artifact(p5)
    registry.put_artifact(p3)

    latest = registry.latest_policy()
    assert latest is not None
    assert latest["window_id"] == 5


# ---------------------------------------------------------------------------
# ArtifactRegistry base-class helpers
# ---------------------------------------------------------------------------


def test_path_for_uses_pluralized_directory(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    path = registry.path_for("rollout", "abc123")
    assert path == tmp_path / "rollouts" / "abc123.json"


def test_artifact_path_uses_artifact_fields(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    artifact = _rollout()
    assert registry.artifact_path(artifact) == registry.path_for(
        artifact["artifact_type"], artifact["artifact_id"]
    )


def test_checkpoint_dir_creates_directory(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    path = registry.checkpoint_dir("ckpt-42")
    assert path.is_dir()
    assert path == tmp_path / "checkpoints" / "ckpt-42"


def test_training_dir_formats_window_four_digit(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    path = registry.training_dir(7)
    assert path.is_dir()
    assert path.name == "window_0007"


def test_run_dir_and_pack_dir_create(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    run_path = registry.run_dir("run-A")
    pack_path = registry.pack_dir("pack-A")
    assert run_path.is_dir()
    assert pack_path.is_dir()


def test_cache_dir_creates(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    path = registry.cache_dir()
    assert path.is_dir()
    assert path.name == "cache"


def test_export_artifacts_copies_to_destination(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path / "src"))
    artifacts = [_rollout(window_id=i, payload_seed=i) for i in range(3)]
    for artifact in artifacts:
        registry.put_artifact(artifact)

    destination = tmp_path / "exports"
    copied = registry.export_artifacts(artifacts, destination)
    assert len(copied) == 3
    for path in copied:
        assert Path(path).is_file()
        assert Path(path).suffix == ".json"
    assert (destination / "rollouts").is_dir()


def test_export_artifacts_skips_missing_sources(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path / "src"))
    present = _rollout(window_id=1, payload_seed=1)
    registry.put_artifact(present)
    missing = _rollout(window_id=2, payload_seed=2)

    destination = tmp_path / "exports"
    copied = registry.export_artifacts([present, missing], destination)
    assert len(copied) == 1


# ---------------------------------------------------------------------------
# SqliteMetadataIndex direct API
# ---------------------------------------------------------------------------


def test_sqlite_metadata_index_records_and_reads(tmp_path: Path) -> None:
    index = SqliteMetadataIndex(tmp_path / "meta.sqlite3")
    artifact = _rollout(window_id=9, payload_seed=3)
    index.record_artifact(artifact, local_path="/tmp/x.json")

    record = index.get_record("rollout", artifact["artifact_id"])
    assert record is not None
    assert record["window_id"] == 9
    assert record["local_path"] == "/tmp/x.json"


def test_sqlite_metadata_index_upserts_on_conflict(tmp_path: Path) -> None:
    index = SqliteMetadataIndex(tmp_path / "meta.sqlite3")
    artifact = _rollout()
    index.record_artifact(artifact, local_path="/tmp/v1.json")
    index.record_artifact(artifact, local_path="/tmp/v2.json", storage_kind="r2")

    record = index.get_record("rollout", artifact["artifact_id"])
    assert record is not None
    assert record["local_path"] == "/tmp/v2.json"
    assert record["storage_kind"] == "r2"


def test_sqlite_metadata_index_list_records_filters_window(tmp_path: Path) -> None:
    index = SqliteMetadataIndex(tmp_path / "meta.sqlite3")
    for i in range(3):
        artifact = _rollout(window_id=i, payload_seed=i)
        index.record_artifact(artifact, local_path=f"/tmp/{i}.json")

    windowed = index.list_records("rollout", window_id=1)
    assert len(windowed) == 1
    assert windowed[0]["window_id"] == 1


def test_sqlite_metadata_index_list_records_orders_by_window_then_time(
    tmp_path: Path,
) -> None:
    index = SqliteMetadataIndex(tmp_path / "meta.sqlite3")
    for window_id in (5, 1, 3):
        artifact = _rollout(window_id=window_id, payload_seed=window_id)
        index.record_artifact(artifact, local_path=f"/tmp/{window_id}.json")

    records = index.list_records("rollout")
    assert [r["window_id"] for r in records] == [1, 3, 5]


def test_sqlite_metadata_index_latest_policy_picks_highest_window(
    tmp_path: Path,
) -> None:
    index = SqliteMetadataIndex(tmp_path / "meta.sqlite3")
    for window_id in (1, 5, 3):
        policy = _policy(window_id=window_id)
        index.record_artifact(policy, local_path=f"/tmp/{window_id}.json")

    latest = index.latest_policy_record()
    assert latest is not None
    assert latest["window_id"] == 5


def test_sqlite_metadata_index_latest_policy_none_when_empty(tmp_path: Path) -> None:
    index = SqliteMetadataIndex(tmp_path / "meta.sqlite3")
    assert index.latest_policy_record() is None


def test_sqlite_metadata_index_creates_parent_dir(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "deep" / "meta.sqlite3"
    SqliteMetadataIndex(db_path)
    assert db_path.parent.is_dir()


def test_sqlite_metadata_index_table_exists(tmp_path: Path) -> None:
    db_path = tmp_path / "meta.sqlite3"
    SqliteMetadataIndex(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='artifacts'"
        ).fetchone()
        assert row is not None and row[0] == "artifacts"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Abstract ArtifactRegistry rejects direct instantiation
# ---------------------------------------------------------------------------


def test_abstract_registry_cannot_be_instantiated(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        ArtifactRegistry(str(tmp_path))  # type: ignore[abstract]
