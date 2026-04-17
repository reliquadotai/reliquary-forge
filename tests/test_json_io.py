"""Unit tests for reliquary.utils.json_io.

These helpers underpin every content-addressable artifact hash in the
protocol — a silent change in canonicalization rules would break
cross-host determinism, so the invariants have to be pinned by test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from reliquary.utils.json_io import (
    read_json,
    sha256_json,
    stable_json_dumps,
    write_json,
)


def test_stable_json_dumps_sorts_keys() -> None:
    out = stable_json_dumps({"z": 1, "a": 2, "m": 3})
    assert out == '{"a":2,"m":3,"z":1}'


def test_stable_json_dumps_is_deterministic_across_insertion_orders() -> None:
    payload_one = {"alpha": 1, "beta": 2}
    payload_two = {"beta": 2, "alpha": 1}
    assert stable_json_dumps(payload_one) == stable_json_dumps(payload_two)


def test_stable_json_dumps_uses_compact_separators() -> None:
    out = stable_json_dumps({"a": [1, 2, 3]})
    assert out == '{"a":[1,2,3]}'
    assert ", " not in out
    assert ": " not in out


def test_stable_json_dumps_ensure_ascii_escapes_non_ascii() -> None:
    out = stable_json_dumps({"greeting": "héllo"})
    assert "\\u00e9" in out


def test_stable_json_dumps_nested_sort() -> None:
    out = stable_json_dumps({"outer": {"z": 1, "a": 2}})
    assert out == '{"outer":{"a":2,"z":1}}'


def test_stable_json_dumps_empty_container() -> None:
    assert stable_json_dumps({}) == "{}"
    assert stable_json_dumps([]) == "[]"


def test_stable_json_dumps_numeric_and_boolean() -> None:
    assert stable_json_dumps({"flag": True, "count": 0}) == '{"count":0,"flag":true}'


def test_sha256_json_hex_length() -> None:
    digest = sha256_json({"key": "value"})
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_sha256_json_matches_manual_hash() -> None:
    payload = {"b": 2, "a": 1}
    digest = sha256_json(payload)
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert digest == expected


def test_sha256_json_deterministic_across_insertion_orders() -> None:
    payload_one = {"z": 1, "a": 2}
    payload_two = {"a": 2, "z": 1}
    assert sha256_json(payload_one) == sha256_json(payload_two)


def test_sha256_json_changes_when_payload_changes() -> None:
    a = sha256_json({"x": 1})
    b = sha256_json({"x": 2})
    assert a != b


def test_write_json_creates_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "file.json"
    write_json(target, {"value": 42})
    assert target.is_file()
    assert target.parent.is_dir()


def test_write_json_produces_pretty_output(tmp_path: Path) -> None:
    target = tmp_path / "pretty.json"
    write_json(target, {"b": 2, "a": 1})
    text = target.read_text(encoding="utf-8")
    assert "\n" in text  # indent=2 forces newlines
    # sort_keys=True sorts keys
    assert text.index('"a"') < text.index('"b"')


def test_write_json_overwrites_existing(tmp_path: Path) -> None:
    target = tmp_path / "rewrite.json"
    write_json(target, {"first": 1})
    write_json(target, {"second": 2})
    assert read_json(target) == {"second": 2}


def test_read_json_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "trip.json"
    payload = {"nested": {"list": [1, 2, 3], "flag": True}, "empty": []}
    write_json(target, payload)
    assert read_json(target) == payload


def test_read_json_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_json(tmp_path / "missing.json")


def test_stable_json_dumps_rejects_non_json_types() -> None:
    with pytest.raises(TypeError):
        stable_json_dumps({"bytes": b"raw"})


def test_write_json_raises_on_non_json_types(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        write_json(tmp_path / "out.json", {"set": {1, 2, 3}})
