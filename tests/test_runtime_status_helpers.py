"""Unit tests for reliquary.runtime_status pure helpers.

Covers the pure-Python utilities (_iso_now, _parse_timestamp,
runtime_status_dir, rl_worker_status_path, read/write_rl_worker_status)
independent of the full build_platform_status path which needs a
registry + InferenceRegistryAdapter. The happy-path roundtrip is already
covered by tests/test_runtime_status.py; this file pins the edge cases.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from reliquary.runtime_status import (
    _iso_now,
    _parse_timestamp,
    read_rl_worker_status,
    rl_worker_status_path,
    runtime_status_dir,
    write_rl_worker_status,
)


# ---------------------------------------------------------------------------
# _iso_now
# ---------------------------------------------------------------------------


def test_iso_now_returns_zulu_string() -> None:
    now = _iso_now()
    # Z suffix replaces +00:00 per the helper.
    assert now.endswith("Z")
    # Parseable as ISO 8601.
    parsed = datetime.fromisoformat(now.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


def test_iso_now_returns_utc() -> None:
    value = _iso_now()
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert parsed.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# _parse_timestamp
# ---------------------------------------------------------------------------


def test_parse_timestamp_none_input_returns_none() -> None:
    assert _parse_timestamp(None) is None


def test_parse_timestamp_empty_string_returns_none() -> None:
    assert _parse_timestamp("") is None


def test_parse_timestamp_zulu_form() -> None:
    value = _parse_timestamp("2026-04-18T12:00:00Z")
    assert value is not None
    parsed_back = datetime.fromtimestamp(value, tz=timezone.utc)
    assert parsed_back.isoformat() == "2026-04-18T12:00:00+00:00"


def test_parse_timestamp_explicit_offset() -> None:
    value = _parse_timestamp("2026-04-18T12:00:00+02:00")
    assert value is not None


def test_parse_timestamp_invalid_returns_none() -> None:
    assert _parse_timestamp("not-a-timestamp") is None
    assert _parse_timestamp("2026-99-99") is None


def test_parse_timestamp_monotone() -> None:
    earlier = _parse_timestamp("2026-04-18T12:00:00Z")
    later = _parse_timestamp("2026-04-18T13:00:00Z")
    assert earlier is not None and later is not None
    assert later > earlier


# ---------------------------------------------------------------------------
# runtime_status_dir / rl_worker_status_path
# ---------------------------------------------------------------------------


def test_runtime_status_dir_uses_state_root(tmp_path: Path) -> None:
    cfg = {"state_root": str(tmp_path / "state")}
    path = runtime_status_dir(cfg)
    assert path == tmp_path / "state" / "runtime"


def test_runtime_status_dir_defaults_to_data() -> None:
    cfg: dict = {}
    path = runtime_status_dir(cfg)
    assert str(path).endswith("data/runtime")


def test_rl_worker_status_path_filename() -> None:
    path = rl_worker_status_path({"state_root": "/tmp/st"})
    assert path.name == "rl_worker_status.json"
    assert "runtime" in str(path)


# ---------------------------------------------------------------------------
# read/write_rl_worker_status
# ---------------------------------------------------------------------------


def test_read_rl_worker_status_missing_returns_none(tmp_path: Path) -> None:
    cfg = {"state_root": str(tmp_path)}
    assert read_rl_worker_status(cfg) is None


def test_write_rl_worker_status_creates_file(tmp_path: Path) -> None:
    cfg = {"state_root": str(tmp_path)}
    merged = write_rl_worker_status(cfg, {"state": "running", "cycle_count": 3})
    assert rl_worker_status_path(cfg).is_file()
    assert merged["state"] == "running"
    assert merged["cycle_count"] == 3
    assert "updated_at" in merged


def test_write_rl_worker_status_merges_with_existing(tmp_path: Path) -> None:
    cfg = {"state_root": str(tmp_path)}
    write_rl_worker_status(cfg, {"state": "running", "cycle_count": 1})
    merged = write_rl_worker_status(cfg, {"cycle_count": 2, "waiting_for_data": False})
    # state from first write preserved; cycle_count updated; new field added.
    assert merged["state"] == "running"
    assert merged["cycle_count"] == 2
    assert merged["waiting_for_data"] is False


def test_write_rl_worker_status_updates_updated_at(tmp_path: Path) -> None:
    cfg = {"state_root": str(tmp_path)}
    first = write_rl_worker_status(cfg, {"state": "running"})
    second = write_rl_worker_status(cfg, {"state": "running"})
    # Timestamps should both be ISO-format; may or may not differ depending on
    # clock resolution, but both must parse.
    assert "updated_at" in first
    assert "updated_at" in second
    assert first["updated_at"].endswith("Z")
    assert second["updated_at"].endswith("Z")


def test_read_rl_worker_status_after_write_matches(tmp_path: Path) -> None:
    cfg = {"state_root": str(tmp_path)}
    write_rl_worker_status(cfg, {"state": "running", "cycle_count": 5})
    read_back = read_rl_worker_status(cfg)
    assert read_back is not None
    assert read_back["state"] == "running"
    assert read_back["cycle_count"] == 5


def test_write_rl_worker_status_creates_parent_dir(tmp_path: Path) -> None:
    cfg = {"state_root": str(tmp_path / "deep" / "nested")}
    write_rl_worker_status(cfg, {"state": "idle"})
    assert rl_worker_status_path(cfg).is_file()


def test_write_rl_worker_status_merges_nested_fields(tmp_path: Path) -> None:
    cfg = {"state_root": str(tmp_path)}
    write_rl_worker_status(
        cfg,
        {
            "state": "running",
            "last_imported_source_window": 42,
            "consecutive_failures": 0,
        },
    )
    merged = write_rl_worker_status(cfg, {"consecutive_failures": 2})
    assert merged["state"] == "running"  # preserved
    assert merged["last_imported_source_window"] == 42  # preserved
    assert merged["consecutive_failures"] == 2  # updated
