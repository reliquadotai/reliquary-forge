"""Tests for the HF Hub continuous-publishing side-channel.

Phase 1.4 of the parallel-harvest plan. Coverage:

- ``should_publish(window_n)`` triggers exactly on multiples of
  ``interval_windows`` and never on window 0
- ``maybe_publish`` schedules an upload on a triggering window, returns
  the Future, and stores the result for ``last_result()``
- ``maybe_publish`` skips when a previous publish is still in flight
- ``maybe_publish`` skips on non-triggering windows
- ``sign_manifest`` + ``verify_manifest`` round-trip; signature changes
  if either checkpoint_n or revision changes
- ``_extract_revision`` handles the common HfApi return shapes
  (CommitInfo with oid, string URL, plain string)
- Publish forwards repo_id + token + commit_message to the API client
- Failure in upload_folder bubbles up via the Future without poisoning
  ``last_result``

Parallel-work credit: romain13190/reliquary@0b9795d.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from reliquary.training.hf_hub_publisher import (
    HfHubPublisher,
    HfPublishResult,
    _extract_revision,
    sign_manifest,
    verify_manifest,
)


@dataclass
class FakeCommitInfo:
    oid: str = "abc1234567"


@dataclass
class FakeHfApi:
    upload_calls: list[dict[str, Any]] = field(default_factory=list)
    return_revision: str = "abc1234567"
    raise_on_upload: Exception | None = None

    def upload_folder(self, **kwargs: Any) -> Any:
        self.upload_calls.append(kwargs)
        if self.raise_on_upload is not None:
            raise self.raise_on_upload
        return FakeCommitInfo(oid=self.return_revision)


def _make_snapshot(tmp_path: Path) -> Path:
    snap = tmp_path / "snapshot"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"\x00" * 16)
    return snap


def _publisher(
    *,
    api: FakeHfApi | None = None,
    interval_windows: int = 10,
    executor: ThreadPoolExecutor | None = None,
) -> HfHubPublisher:
    return HfHubPublisher(
        repo_id="reliquadotai/reliquary-test",
        token="hf_fake_token",
        signing_secret="test-secret-123",
        interval_windows=interval_windows,
        api=api or FakeHfApi(),
        executor=executor,
    )


# ────────  should_publish gating  ────────


def test_should_publish_triggers_on_interval_multiples() -> None:
    pub = _publisher(interval_windows=10)
    assert pub.should_publish(10)
    assert pub.should_publish(20)
    assert pub.should_publish(100)


def test_should_publish_skips_non_multiples() -> None:
    pub = _publisher(interval_windows=10)
    for w in [1, 5, 9, 11, 99]:
        assert not pub.should_publish(w)


def test_should_publish_never_triggers_on_window_zero() -> None:
    pub = _publisher(interval_windows=10)
    assert not pub.should_publish(0)


def test_should_publish_rejects_negative_window() -> None:
    pub = _publisher(interval_windows=10)
    assert not pub.should_publish(-1)


def test_publisher_rejects_zero_or_negative_interval() -> None:
    with pytest.raises(ValueError, match="interval_windows must be >= 1"):
        _publisher(interval_windows=0)


# ────────  sign / verify  ────────


def test_sign_verify_roundtrips() -> None:
    sig = sign_manifest(checkpoint_n=42, revision="abc1234", secret="s")
    assert verify_manifest(
        checkpoint_n=42, revision="abc1234", signature=sig, secret="s"
    )


def test_sign_changes_with_checkpoint_n() -> None:
    a = sign_manifest(checkpoint_n=42, revision="abc1234", secret="s")
    b = sign_manifest(checkpoint_n=43, revision="abc1234", secret="s")
    assert a != b


def test_sign_changes_with_revision() -> None:
    a = sign_manifest(checkpoint_n=42, revision="abc1234", secret="s")
    b = sign_manifest(checkpoint_n=42, revision="def5678", secret="s")
    assert a != b


def test_sign_changes_with_secret() -> None:
    a = sign_manifest(checkpoint_n=42, revision="abc1234", secret="s1")
    b = sign_manifest(checkpoint_n=42, revision="abc1234", secret="s2")
    assert a != b


def test_verify_rejects_tampered_revision() -> None:
    sig = sign_manifest(checkpoint_n=42, revision="abc1234", secret="s")
    assert not verify_manifest(
        checkpoint_n=42, revision="abc7777", signature=sig, secret="s"
    )


def test_sign_rejects_negative_checkpoint_n() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        sign_manifest(checkpoint_n=-1, revision="x", secret="s")


def test_sign_rejects_empty_revision() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        sign_manifest(checkpoint_n=1, revision="", secret="s")


# ────────  _extract_revision dispatch  ────────


def test_extract_revision_from_oid_attribute() -> None:
    assert _extract_revision(FakeCommitInfo(oid="abcdef0")) == "abcdef0"


def test_extract_revision_from_commit_url_string() -> None:
    url = "https://huggingface.co/owner/repo/commit/aabbccdd11223344"
    assert _extract_revision(url) == "aabbccdd11223344"


def test_extract_revision_from_commit_url_with_query() -> None:
    url = "https://huggingface.co/owner/repo/commit/aabbccdd?ref=main"
    assert _extract_revision(url) == "aabbccdd"


def test_extract_revision_from_plain_string_returns_string() -> None:
    assert _extract_revision("just-a-sha-12345") == "just-a-sha-12345"


def test_extract_revision_raises_on_none() -> None:
    with pytest.raises(RuntimeError, match="returned None"):
        _extract_revision(None)


# ────────  maybe_publish  ────────


def test_maybe_publish_schedules_on_interval(tmp_path: Path) -> None:
    api = FakeHfApi(return_revision="rev-1234567")
    with ThreadPoolExecutor(max_workers=1) as exe:
        pub = _publisher(api=api, executor=exe)
        snap = _make_snapshot(tmp_path)
        future = pub.maybe_publish(window_n=10, snapshot_dir=snap)
        assert future is not None
        result = future.result(timeout=2.0)
        assert isinstance(result, HfPublishResult)
        assert result.checkpoint_n == 1
        assert result.repo_id == "reliquadotai/reliquary-test"
        assert result.revision == "rev-1234567"
        assert result.binding == "1|rev-1234567"
        # Signature verifies
        assert verify_manifest(
            checkpoint_n=1,
            revision="rev-1234567",
            signature=result.signature,
            secret="test-secret-123",
        )
        # last_result reflects the publish
        assert pub.last_result() == result


def test_maybe_publish_skips_non_triggering_windows(tmp_path: Path) -> None:
    api = FakeHfApi()
    pub = _publisher(api=api)
    snap = _make_snapshot(tmp_path)
    assert pub.maybe_publish(window_n=5, snapshot_dir=snap) is None
    assert pub.maybe_publish(window_n=9, snapshot_dir=snap) is None
    assert api.upload_calls == []


def test_maybe_publish_forwards_repo_token_and_commit_message(tmp_path: Path) -> None:
    api = FakeHfApi()
    with ThreadPoolExecutor(max_workers=1) as exe:
        pub = _publisher(api=api, executor=exe)
        snap = _make_snapshot(tmp_path)
        future = pub.maybe_publish(
            window_n=10,
            snapshot_dir=snap,
            commit_message="custom msg",
        )
        assert future is not None
        future.result(timeout=2.0)
    assert len(api.upload_calls) == 1
    call = api.upload_calls[0]
    assert call["repo_id"] == "reliquadotai/reliquary-test"
    assert call["folder_path"] == str(snap)
    assert call["commit_message"] == "custom msg"
    assert call["token"] == "hf_fake_token"


def test_maybe_publish_skips_when_inflight(tmp_path: Path) -> None:
    # Use a manually-controlled future so the first publish doesn't finish
    # before we call maybe_publish a second time.
    fake_executor = MagicMock()
    pending_future: Any = MagicMock()
    pending_future.done.return_value = False
    fake_executor.submit.return_value = pending_future
    pub = HfHubPublisher(
        repo_id="reliquadotai/reliquary-test",
        token=None,
        signing_secret="x",
        interval_windows=10,
        api=FakeHfApi(),
        executor=fake_executor,
    )
    snap = _make_snapshot(tmp_path)
    first = pub.maybe_publish(window_n=10, snapshot_dir=snap)
    second = pub.maybe_publish(window_n=20, snapshot_dir=snap)
    assert first is pending_future
    assert second is None
    assert fake_executor.submit.call_count == 1


def test_publish_failure_does_not_poison_last_result(tmp_path: Path) -> None:
    api = FakeHfApi(raise_on_upload=RuntimeError("hf api down"))
    with ThreadPoolExecutor(max_workers=1) as exe:
        pub = _publisher(api=api, executor=exe)
        snap = _make_snapshot(tmp_path)
        future = pub.maybe_publish(window_n=10, snapshot_dir=snap)
        assert future is not None
        with pytest.raises(RuntimeError, match="hf api down"):
            future.result(timeout=2.0)
        assert pub.last_result() is None  # unchanged


def test_publish_validates_snapshot_dir(tmp_path: Path) -> None:
    api = FakeHfApi()
    with ThreadPoolExecutor(max_workers=1) as exe:
        pub = _publisher(api=api, executor=exe)
        bogus = tmp_path / "nope"
        future = pub.maybe_publish(window_n=10, snapshot_dir=bogus)
        assert future is not None
        with pytest.raises(ValueError, match="does not exist"):
            future.result(timeout=2.0)
