"""Continuous HF Hub checkpoint publishing — side-channel beside delta-to-R2.

Phase 1.4 of the parallel-harvest plan: every N windows the trainer
pushes the full HF-format snapshot of the trained model to a public
Hugging Face repository. Miners and downstream consumers can
``snapshot_download`` directly from there for ecosystem use; the
Reliquary protocol still uses the delta-checkpoint + signed
PolicyCommitment flow as the source of truth for in-subnet policy
installation.

Design notes:

- ``huggingface_hub`` is bundled with the ``[ml]`` extra. If absent at
  publish time, the public class still loads but raises
  ``HfHubUnavailableError`` on first ``maybe_publish``. This mirrors the
  WandbSink fail-soft pattern.
- The ``HfApi`` and ``snapshot_download`` callable Protocols let tests
  inject fakes without requiring the SDK.
- The publisher signs ``f"{checkpoint_n}|{revision}"`` with HMAC over a
  shared secret so downstream consumers can verify the manifest belongs
  to the trainer hotkey. Pulls in ``reliquary_protocol.signatures.hmac_sign``
  to match the signing primitive used by every other Reliquary artifact.
- Publishing runs in a thread executor so it doesn't block the trainer.
  The trainer's loop calls ``maybe_publish(window_n, ...)``; the actual
  upload happens off-thread, and ``last_manifest()`` returns the most
  recent successful publish (with revision sha) once it lands.

Parallel-work credit: romain13190/reliquary@0b9795d. The upstream
pattern (every-N-windows full HF snapshot, signed manifest, miner-side
``snapshot_download`` on revision change) is what we adopt. The signing
flow uses our protocol package's HMAC primitives; the dependency
injection (``api`` + ``thread_pool``) is independent.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)


class HfHubUnavailableError(RuntimeError):
    """Raised when ``huggingface_hub`` is needed but not installed."""


class HfApiClient(Protocol):
    """Subset of huggingface_hub.HfApi we depend on."""

    def upload_folder(
        self,
        *,
        repo_id: str,
        folder_path: str,
        commit_message: str,
        token: str | None = None,
    ) -> Any: ...


def _default_api() -> HfApiClient:
    try:
        from huggingface_hub import HfApi  # type: ignore[import-not-found]
    except ImportError as exc:
        raise HfHubUnavailableError(
            "huggingface_hub is not installed; install with `pip install reliquary[ml]`"
        ) from exc
    return HfApi()


@dataclass(frozen=True)
class HfPublishResult:
    """Outcome of one publish call, suitable for the ``/checkpoint`` manifest."""

    checkpoint_n: int
    repo_id: str
    revision: str
    signature: str
    published_at: float
    binding: str
    """Canonical signed string ``f'{checkpoint_n}|{revision}'`` — what the
    signature was computed over. Repeated here so verifiers don't have to
    reconstruct it."""


class HfHubPublisher:
    """Periodic full-snapshot publisher for HF Hub.

    Invariants:

    - ``maybe_publish`` is the only entry point trainers call. It returns
      a ``Future`` if a publish was scheduled, or ``None`` if the current
      window doesn't trigger one.
    - At most one publish runs at a time per publisher (a second
      ``maybe_publish`` while one is in flight returns ``None`` and logs).
    - ``last_result()`` returns the most recent successful publish; reads
      are concurrency-safe via a lock.
    - On failure (network, auth, etc.), we log and keep ``last_result``
      pointing at the previous successful publish; the trainer is not
      blocked.
    """

    def __init__(
        self,
        *,
        repo_id: str,
        token: str | None,
        signing_secret: str,
        interval_windows: int = 10,
        api: HfApiClient | None = None,
        executor: ThreadPoolExecutor | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if interval_windows < 1:
            raise ValueError(f"interval_windows must be >= 1, got {interval_windows}")
        if not repo_id or not isinstance(repo_id, str):
            raise ValueError("repo_id must be a non-empty string")
        if not signing_secret:
            raise ValueError("signing_secret is required (for manifest HMAC)")
        self._repo_id = repo_id
        self._token = token
        self._signing_secret = signing_secret
        self._interval_windows = interval_windows
        self._api = api  # resolved lazily so tests don't import the SDK
        self._owned_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="hf-publisher"
        )
        self._clock = clock
        self._lock = threading.Lock()
        self._inflight: Future[HfPublishResult] | None = None
        self._last_result: HfPublishResult | None = None
        self._publish_counter = 0

    @property
    def repo_id(self) -> str:
        return self._repo_id

    @property
    def interval_windows(self) -> int:
        return self._interval_windows

    def should_publish(self, window_n: int) -> bool:
        """Return True if this window number triggers a publish."""
        if window_n <= 0:
            return False
        return window_n % self._interval_windows == 0

    def maybe_publish(
        self,
        *,
        window_n: int,
        snapshot_dir: Path,
        commit_message: str | None = None,
    ) -> Future[HfPublishResult] | None:
        """Schedule a publish if ``window_n`` triggers one and no publish is
        currently in flight; otherwise return ``None``.

        ``snapshot_dir`` must be a directory containing a HF-format model
        snapshot (model.safetensors + config.json + tokenizer files). The
        caller is responsible for serializing the model before calling.
        """
        if not self.should_publish(window_n):
            return None
        with self._lock:
            if self._inflight is not None and not self._inflight.done():
                logger.warning(
                    "hf_hub_publisher: previous publish still in flight at "
                    "window %d; skipping",
                    window_n,
                )
                return None
            self._publish_counter += 1
            checkpoint_n = self._publish_counter
            message = commit_message or f"reliquary-forge checkpoint {checkpoint_n} (window {window_n})"
            future = self._executor.submit(
                self._do_publish,
                checkpoint_n=checkpoint_n,
                snapshot_dir=Path(snapshot_dir),
                commit_message=message,
            )
            self._inflight = future
        return future

    def last_result(self) -> HfPublishResult | None:
        """Most recent successful publish, or ``None`` if none yet."""
        with self._lock:
            return self._last_result

    def shutdown(self, *, wait: bool = True) -> None:
        """Tear down the executor if we own it."""
        if self._owned_executor:
            self._executor.shutdown(wait=wait)

    def _do_publish(
        self,
        *,
        checkpoint_n: int,
        snapshot_dir: Path,
        commit_message: str,
    ) -> HfPublishResult:
        if not snapshot_dir.exists() or not snapshot_dir.is_dir():
            raise ValueError(f"snapshot_dir does not exist or is not a directory: {snapshot_dir}")
        api = self._api or _default_api()
        commit_info = api.upload_folder(
            repo_id=self._repo_id,
            folder_path=str(snapshot_dir),
            commit_message=commit_message,
            token=self._token,
        )
        revision = _extract_revision(commit_info)
        signature = sign_manifest(
            checkpoint_n=checkpoint_n,
            revision=revision,
            secret=self._signing_secret,
        )
        result = HfPublishResult(
            checkpoint_n=checkpoint_n,
            repo_id=self._repo_id,
            revision=revision,
            signature=signature,
            published_at=self._clock(),
            binding=f"{checkpoint_n}|{revision}",
        )
        with self._lock:
            self._last_result = result
            self._inflight = None
        logger.info(
            "hf_hub_publisher: published checkpoint %d to %s @ %s",
            checkpoint_n,
            self._repo_id,
            revision,
        )
        return result


def sign_manifest(*, checkpoint_n: int, revision: str, secret: str) -> str:
    """HMAC-sign the canonical ``checkpoint_n|revision`` binding.

    Uses the same HMAC primitive every other Reliquary artifact uses
    (``reliquary_protocol.signatures.hmac_sign``) so downstream
    verification reuses one code path.
    """
    if not isinstance(checkpoint_n, int) or checkpoint_n < 0:
        raise ValueError(f"checkpoint_n must be a non-negative int, got {checkpoint_n!r}")
    if not isinstance(revision, str) or not revision:
        raise ValueError("revision must be a non-empty string")
    binding = f"{checkpoint_n}|{revision}".encode("utf-8")
    try:
        from reliquary_protocol.signatures import hmac_sign
    except ImportError:  # pragma: no cover - fallback for editable installs
        import hmac
        import hashlib
        return hmac.new(secret.encode("utf-8"), binding, hashlib.sha256).hexdigest()
    return hmac_sign(binding, secret)


def verify_manifest(
    *,
    checkpoint_n: int,
    revision: str,
    signature: str,
    secret: str,
) -> bool:
    """Verify a manifest signature. Constant-time per HMAC contract."""
    expected = sign_manifest(
        checkpoint_n=checkpoint_n, revision=revision, secret=secret
    )
    # Use hmac.compare_digest for constant-time comparison.
    import hmac as _hmac
    return _hmac.compare_digest(expected, signature)


def _extract_revision(commit_info: Any) -> str:
    """Pull the commit SHA from an HfApi.upload_folder return value.

    huggingface_hub returns a ``CommitInfo`` (newer SDK) or a string URL
    (older). We try a few common attribute names + fall back to a string
    parse so the publisher is robust across SDK versions.
    """
    if commit_info is None:
        raise RuntimeError("upload_folder returned None; cannot derive revision")
    for attr in ("oid", "commit_oid", "revision", "sha", "head_sha"):
        value = getattr(commit_info, attr, None)
        if isinstance(value, str) and value:
            return value
    if isinstance(commit_info, str):
        # Older SDK: returns the commit URL "https://huggingface.co/.../commit/<sha>"
        if "/commit/" in commit_info:
            return commit_info.rsplit("/commit/", 1)[-1].split("?", 1)[0]
        return commit_info
    raise RuntimeError(
        f"could not extract revision from upload_folder return value: {commit_info!r}"
    )


__all__ = [
    "HfApiClient",
    "HfHubPublisher",
    "HfHubUnavailableError",
    "HfPublishResult",
    "sign_manifest",
    "verify_manifest",
]
