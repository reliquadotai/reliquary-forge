"""Tests for the LOCAL backend path of reliquary.inference_adapter.

The S3 backend paths need boto3 + a running minio/r2 and are covered by
integration tests on the fleet. Here we pin the local-filesystem
behavior + cursor primitives so a misconfiguration under local-dev
doesn't silently desynchronize the RL importer.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from reliquary.inference_adapter import (
    InferenceRegistryAdapter,
    _read_gzip_json_bytes,
    _utc_now,
)
from reliquary.utils.json_io import write_json


# ---------------------------------------------------------------------------
# _read_gzip_json_bytes / _utc_now
# ---------------------------------------------------------------------------


def test_read_gzip_json_bytes_round_trip() -> None:
    payload = {"a": 1, "b": [1, 2, 3]}
    raw = gzip.compress(json.dumps(payload).encode("utf-8"))
    assert _read_gzip_json_bytes(raw) == payload


def test_read_gzip_json_bytes_list_payload() -> None:
    payload = [{"x": i} for i in range(3)]
    raw = gzip.compress(json.dumps(payload).encode("utf-8"))
    assert _read_gzip_json_bytes(raw) == payload


def test_utc_now_is_iso_utc() -> None:
    value = _utc_now()
    assert "+00:00" in value or value.endswith("+00:00")


# ---------------------------------------------------------------------------
# InferenceRegistryAdapter — local backend configuration
# ---------------------------------------------------------------------------


def _cfg(tmp_path: Path, **overrides: Any) -> dict:
    base = {
        "inference_registry_backend": "local",
        "inference_registry_local_root": str(tmp_path / "inference"),
    }
    base.update(overrides)
    return base


def test_adapter_creates_cache_root(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    adapter = InferenceRegistryAdapter(_cfg(tmp_path), cache_root=cache)
    assert (cache / "inference_imports").is_dir()
    assert adapter.cache_root == cache / "inference_imports"


def test_adapter_backend_defaults_to_local_when_unset(tmp_path: Path) -> None:
    adapter = InferenceRegistryAdapter({}, cache_root=tmp_path / "cache")
    assert adapter.backend == "local"


def test_adapter_region_default_is_auto_when_empty(tmp_path: Path) -> None:
    adapter = InferenceRegistryAdapter(
        _cfg(tmp_path, inference_registry_region=""),
        cache_root=tmp_path / "cache",
    )
    assert adapter.region == "auto"


# ---------------------------------------------------------------------------
# cursor_path / read_cursor / write_cursor
# ---------------------------------------------------------------------------


def test_read_cursor_missing_returns_none(tmp_path: Path) -> None:
    adapter = InferenceRegistryAdapter(_cfg(tmp_path), cache_root=tmp_path / "cache")
    assert adapter.read_cursor("reasoning_rl") is None


def test_write_then_read_cursor_round_trip(tmp_path: Path) -> None:
    adapter = InferenceRegistryAdapter(_cfg(tmp_path), cache_root=tmp_path / "cache")
    adapter.write_cursor("reasoning_rl", window_id=42)
    assert adapter.read_cursor("reasoning_rl") == 42


def test_write_cursor_overwrites_existing(tmp_path: Path) -> None:
    adapter = InferenceRegistryAdapter(_cfg(tmp_path), cache_root=tmp_path / "cache")
    adapter.write_cursor("reasoning_rl", window_id=1)
    adapter.write_cursor("reasoning_rl", window_id=10)
    assert adapter.read_cursor("reasoning_rl") == 10


def test_cursor_path_includes_name(tmp_path: Path) -> None:
    adapter = InferenceRegistryAdapter(_cfg(tmp_path), cache_root=tmp_path / "cache")
    path = adapter.cursor_path("my-cursor")
    assert path.name == "my-cursor.json"
    assert path.parent == adapter.cache_root


def test_cursor_file_payload_shape(tmp_path: Path) -> None:
    adapter = InferenceRegistryAdapter(_cfg(tmp_path), cache_root=tmp_path / "cache")
    adapter.write_cursor("reasoning_rl", window_id=7)
    payload = json.loads(adapter.cursor_path("reasoning_rl").read_text())
    assert payload["cursor_name"] == "reasoning_rl"
    assert payload["window_id"] == 7
    assert "updated_at" in payload


def test_multiple_cursors_are_independent(tmp_path: Path) -> None:
    adapter = InferenceRegistryAdapter(_cfg(tmp_path), cache_root=tmp_path / "cache")
    adapter.write_cursor("cursor-A", window_id=1)
    adapter.write_cursor("cursor-B", window_id=5)
    assert adapter.read_cursor("cursor-A") == 1
    assert adapter.read_cursor("cursor-B") == 5


# ---------------------------------------------------------------------------
# _list_manifest_refs / finalized_window_manifests (local backend)
# ---------------------------------------------------------------------------


def _manifest(
    window_id: int,
    *,
    finalized: bool = True,
    created_at: str = "2026-04-18T00:00:00Z",
) -> dict:
    return {
        "artifact_type": "window_manifest",
        "artifact_id": f"manifest-{window_id}",
        "window_id": window_id,
        "created_at": created_at,
        "payload": {
            "chain_publish_result": ({"tx_hash": "deadbeef"} if finalized else None),
            "task_batch_ref": "task_batches/tb-1.json",
            "task_batch_id": "tb-1",
            "completion_bundle_refs": [],
        },
    }


def _seed_local_manifest(root: Path, manifest: dict) -> None:
    manifest_dir = root / "window_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_dir / f"{manifest['artifact_id']}.json", manifest)


def test_list_manifest_refs_empty_when_directory_missing(tmp_path: Path) -> None:
    adapter = InferenceRegistryAdapter(_cfg(tmp_path), cache_root=tmp_path / "cache")
    assert adapter._list_manifest_refs() == []


def test_list_manifest_refs_lists_local_jsons(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    root = Path(cfg["inference_registry_local_root"])
    _seed_local_manifest(root, _manifest(window_id=1))
    _seed_local_manifest(root, _manifest(window_id=2))
    adapter = InferenceRegistryAdapter(cfg, cache_root=tmp_path / "cache")
    refs = adapter._list_manifest_refs()
    assert len(refs) == 2
    assert all(ref.endswith(".json") for ref in refs)


def test_finalized_window_manifests_skips_non_finalized(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    root = Path(cfg["inference_registry_local_root"])
    _seed_local_manifest(root, _manifest(window_id=1, finalized=False))
    _seed_local_manifest(root, _manifest(window_id=2, finalized=True))
    adapter = InferenceRegistryAdapter(cfg, cache_root=tmp_path / "cache")
    manifests = adapter.finalized_window_manifests()
    assert len(manifests) == 1
    assert manifests[0]["window_id"] == 2


def test_finalized_window_manifests_orders_by_window_ascending(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    root = Path(cfg["inference_registry_local_root"])
    for window in (3, 1, 5):
        _seed_local_manifest(root, _manifest(window_id=window))
    adapter = InferenceRegistryAdapter(cfg, cache_root=tmp_path / "cache")
    manifests = adapter.finalized_window_manifests()
    assert [m["window_id"] for m in manifests] == [1, 3, 5]


def test_finalized_window_manifests_respects_limit(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    root = Path(cfg["inference_registry_local_root"])
    for window in range(5):
        _seed_local_manifest(root, _manifest(window_id=window))
    adapter = InferenceRegistryAdapter(cfg, cache_root=tmp_path / "cache")
    latest_two = adapter.finalized_window_manifests(limit=2)
    assert [m["window_id"] for m in latest_two] == [3, 4]


def test_finalized_window_manifests_dedupes_by_window_picking_later_created_at(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    root = Path(cfg["inference_registry_local_root"])
    earlier = _manifest(window_id=1, created_at="2026-04-18T00:00:00Z")
    later = _manifest(window_id=1, created_at="2026-04-18T01:00:00Z")
    # Use distinct artifact_ids so both files live on disk.
    earlier["artifact_id"] = "manifest-early"
    later["artifact_id"] = "manifest-late"
    _seed_local_manifest(root, earlier)
    _seed_local_manifest(root, later)
    adapter = InferenceRegistryAdapter(cfg, cache_root=tmp_path / "cache")
    manifests = adapter.finalized_window_manifests()
    assert len(manifests) == 1
    assert manifests[0]["artifact_id"] == "manifest-late"


# ---------------------------------------------------------------------------
# next_finalized_window
# ---------------------------------------------------------------------------


def test_next_finalized_window_returns_first_window_when_no_cursor(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    root = Path(cfg["inference_registry_local_root"])
    _seed_local_manifest(root, _manifest(window_id=3))
    _seed_local_manifest(root, _manifest(window_id=5))
    adapter = InferenceRegistryAdapter(cfg, cache_root=tmp_path / "cache")
    assert adapter.next_finalized_window(cursor_name="reasoning_rl") == 3


def test_next_finalized_window_returns_none_when_none_past_cursor(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    root = Path(cfg["inference_registry_local_root"])
    _seed_local_manifest(root, _manifest(window_id=3))
    adapter = InferenceRegistryAdapter(cfg, cache_root=tmp_path / "cache")
    adapter.write_cursor("reasoning_rl", window_id=5)
    assert adapter.next_finalized_window(cursor_name="reasoning_rl") is None


def test_next_finalized_window_returns_window_above_cursor(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    root = Path(cfg["inference_registry_local_root"])
    for window in (1, 2, 3, 4):
        _seed_local_manifest(root, _manifest(window_id=window))
    adapter = InferenceRegistryAdapter(cfg, cache_root=tmp_path / "cache")
    adapter.write_cursor("reasoning_rl", window_id=2)
    # Iterates ascending; first window_id > 2 is 3.
    assert adapter.next_finalized_window(cursor_name="reasoning_rl") == 3


def test_next_finalized_window_none_on_empty_registry(tmp_path: Path) -> None:
    adapter = InferenceRegistryAdapter(_cfg(tmp_path), cache_root=tmp_path / "cache")
    assert adapter.next_finalized_window(cursor_name="reasoning_rl") is None


# ---------------------------------------------------------------------------
# _artifact_key
# ---------------------------------------------------------------------------


def test_artifact_key_known_types(tmp_path: Path) -> None:
    adapter = InferenceRegistryAdapter(_cfg(tmp_path), cache_root=tmp_path / "cache")
    assert adapter._artifact_key("task_batch", "tb-1") == "task_batches/tb-1.json"
    assert adapter._artifact_key("scorecard", "sc-1") == "scorecards/sc-1.json"
    assert adapter._artifact_key("window_manifest", "wm-1") == "window_manifests/wm-1.json"


def test_artifact_key_fallback_pluralizes(tmp_path: Path) -> None:
    adapter = InferenceRegistryAdapter(_cfg(tmp_path), cache_root=tmp_path / "cache")
    assert adapter._artifact_key("widget", "w-1") == "widgets/w-1.json"
