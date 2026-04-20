"""Live closed-loop smoke test over a real R2 bucket.

Runs the full Forge → Ledger flywheel against the production R2 bucket
exercised by :mod:`reliquary_protocol.storage.r2`. Clean up is best-
effort: every written key is deleted at the end; failures during
cleanup are logged but don't mask a test failure.

Environment (required):
    RELIQUARY_R2_ACCOUNT_ID
    RELIQUARY_R2_BUCKET
    RELIQUARY_R2_CF_API_TOKEN
    RELIQUARY_R2_PUBLIC_URL  (optional; enables r2.dev fastpath)

Usage:
    cd /Users/malouk/GRAIL/reliquary
    python3 scripts/run_closed_loop_smoke.py

Exits 0 on full-loop success, non-zero on any step failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make sibling repos importable when run from anywhere.
_ROOT = Path(__file__).resolve().parents[1]
for p in (
    _ROOT,
    _ROOT.parent / "reliquary-inference",
    _ROOT.parent / "reliquary-protocol",
):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


import torch  # noqa: E402

from reliquary.training.checkpoint_storage import (  # noqa: E402
    CheckpointStorageBackend,
    publish_bundle,
)
from reliquary.training.delta_checkpoints import compute_delta  # noqa: E402
from reliquary.training.policy_attestation import (  # noqa: E402
    make_hmac_authority,
    publish_attestation,
)
from reliquary_inference.shared.policy_consumer import (  # noqa: E402
    LoadedDelta,
    PolicyConsumer,
    default_smoke_runner,
)
from reliquary_inference.validator.mesh import (  # noqa: E402
    MedianVerdict,
    MeshAggregationReport,
    ValidatorIdentity,
    VerdictArtifact,
)
from reliquary_inference.validator.rollout_bundle import (  # noqa: E402
    RolloutBundleFetcher,
    RolloutBundlePublisher,
    make_hmac_signer,
    make_hmac_verifier,
)
from reliquary_protocol.storage import R2ObjectBackend  # noqa: E402


# Isolated namespace — never collides with production keys.
RUN_ID = f"closed-loop-smoke-{int(time.time())}"
INFERENCE_NETUID = 9181  # test-only
TRAINING_NETUID = 9003  # test-only
AUTHORITY = "forge-authority-smoke"
AUTHORITY_SECRET = "smoke-authority-secret"
VALIDATOR = "ledger-validator-smoke"
VALIDATOR_SECRET = "smoke-validator-secret"


class _NamespacedBackend:
    """Wraps an R2 backend to transparently prefix every key under ``namespace``.

    Keeps the smoke test isolated from real data AND makes cleanup a
    single ``list(namespace/).delete()`` pass at teardown.
    """

    def __init__(self, inner: CheckpointStorageBackend, namespace: str) -> None:
        self.inner = inner
        self.namespace = namespace.rstrip("/") + "/"
        self._written_keys: list[str] = []

    def put(self, key: str, data: bytes) -> None:
        ns_key = self.namespace + key
        self.inner.put(ns_key, data)
        if ns_key not in self._written_keys:
            self._written_keys.append(ns_key)

    def get(self, key: str) -> bytes | None:
        return self.inner.get(self.namespace + key)

    def list(self, prefix: str) -> list[str]:
        ns_prefix = self.namespace + prefix
        raw = self.inner.list(ns_prefix)
        return [k[len(self.namespace):] for k in raw if k.startswith(self.namespace)]

    def delete(self, key: str) -> None:
        self.inner.delete(self.namespace + key)

    def cleanup(self) -> int:
        """Delete every key written under the namespace."""
        deleted = 0
        for k in list(self._written_keys):
            try:
                self.inner.delete(k)
                deleted += 1
            except Exception as exc:  # pragma: no cover - best-effort
                print(f"  cleanup warn: {k}: {exc}", file=sys.stderr)
        return deleted


def _build_delta():
    torch.manual_seed(0)
    base = {
        "encoder.weight": torch.zeros(4, 4),
        "decoder.weight": torch.zeros(2, 4),
    }
    target = {
        "encoder.weight": torch.ones(4, 4) * 0.1,
        "decoder.weight": torch.ones(2, 4) * 0.2,
    }
    return compute_delta(base, target)


def _mesh_report(window_id: int, completions, validators) -> MeshAggregationReport:
    report = MeshAggregationReport(window_id=window_id)
    for cid in completions:
        report.median_verdicts[cid] = MedianVerdict(
            completion_id=cid,
            accepted=True,
            acceptance_score=1.0,
            median_scores={"reward": 0.9, "format": 1.0},
            participating_validators=list(validators),
            outlier_validators=[],
            quorum_satisfied=True,
        )
    return report


def _verdict_artifact(completion_id: str, validator_hotkey: str) -> VerdictArtifact:
    return VerdictArtifact(
        completion_id=completion_id,
        miner_hotkey="miner_x",
        window_id=110,
        validator=ValidatorIdentity(
            hotkey=validator_hotkey, stake=100.0, signer_id=validator_hotkey
        ),
        accepted=True,
        stage_failed=None,
        reject_reason=None,
        scores={"reward": 0.9, "format": 1.0},
        signed_at=float(time.time()),
    )


def _forge_delta_loader(*, run_id, window_id, expected_merkle_root_hex, backend):
    manifest_key = f"checkpoints/{run_id}/{window_id}/manifest.json"
    manifest_bytes = backend.get(manifest_key)
    assert manifest_bytes is not None, f"missing manifest at {manifest_key}"
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    shard_digests = tuple(entry["payload_sha256"] for entry in manifest["shards"])
    return LoadedDelta(
        run_id=run_id,
        window_id=window_id,
        merkle_root_hex=manifest["merkle_root_hex"],
        raw_manifest_bytes=manifest_bytes,
        shard_digests=shard_digests,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-cleanup", action="store_true", help="Skip post-test key deletion."
    )
    args = parser.parse_args()

    print(f"==> run_id={RUN_ID}")
    print(f"==> inference_netuid={INFERENCE_NETUID} training_netuid={TRAINING_NETUID}")

    try:
        r2 = R2ObjectBackend.from_env()
    except Exception as exc:
        print(f"fatal: R2 env not configured: {exc}", file=sys.stderr)
        return 2

    backend = _NamespacedBackend(r2, f"smoke/{RUN_ID}")

    ok = True
    try:
        # ---- Phase 1: Forge publishes delta ----
        print("\n[1/4] Forge publishes delta bundle → R2")
        delta_bundle = _build_delta()
        manifest = publish_bundle(
            delta_bundle, run_id=RUN_ID, window_id=42, backend=backend
        )
        print(f"      manifest={manifest.manifest_key}")
        print(f"      merkle_root={manifest.merkle_root_hex}")

        # ---- Phase 2: Forge publishes attestation + commitment ----
        print("\n[2/4] Forge publishes attestation + policy commitment")
        publication = publish_attestation(
            backend=backend,
            signer=make_hmac_authority(AUTHORITY, AUTHORITY_SECRET),
            training_netuid=TRAINING_NETUID,
            inference_netuid=INFERENCE_NETUID,
            checkpoint_manifest=manifest,
            base_snapshot_window_id=0,
            consumed_rollout_windows=[100, 101],
            consumed_rollout_keys=[
                f"rollouts/{INFERENCE_NETUID}/100-100/{VALIDATOR}.json",
                f"rollouts/{INFERENCE_NETUID}/101-101/{VALIDATOR}.json",
            ],
            effective_at_ledger_window=102,
        )
        print(f"      attestation={publication.attestation_key}")
        print(f"      attestation_id={publication.attestation.artifact_id()}")
        print(f"      commitment={publication.commitment_key}")
        print(f"      smoke_hash={publication.smoke_hash_hex}")

        # ---- Phase 3: Ledger policy consumer ----
        print("\n[3/4] Ledger policy consumer polls + applies")
        applied = []
        consumer = PolicyConsumer(
            backend=backend,
            verifier=make_hmac_verifier({AUTHORITY: AUTHORITY_SECRET}),
            inference_netuid=INFERENCE_NETUID,
            training_netuid=TRAINING_NETUID,
            delta_loader=_forge_delta_loader,
            smoke_runner=default_smoke_runner,
            applier=lambda delta: applied.append(delta),
        )
        outcome = consumer.poll_once(ledger_window=102)
        print(f"      state={outcome.state}")
        if outcome.state != "applied":
            print(f"      reason={outcome.reason}")
            raise RuntimeError("policy consumer did not apply on effective window")
        print(f"      applied={len(applied)} delta(s)")
        print(f"      current_policy_window={consumer.current_policy_window}")

        # ---- Phase 4: Ledger publishes rollout bundle + Forge fetches ----
        print("\n[4/4] Ledger publishes rollout bundle + fetcher round-trip")
        publisher = RolloutBundlePublisher(
            backend=backend,
            signer=make_hmac_signer(VALIDATOR, VALIDATOR_SECRET),
            netuid=INFERENCE_NETUID,
        )
        rollout_outcome = publisher.publish(
            window_id=110,
            mesh_report=_mesh_report(
                window_id=110,
                completions=["c1", "c2"],
                validators=[VALIDATOR, "ledger-validator-b"],
            ),
            manifest={"task_source": "sat", "task_count": 2, "seed": 0xCAFEF00D},
            task_batch={
                "tasks": [
                    {"id": "c1", "prompt": "prompt-1"},
                    {"id": "c2", "prompt": "prompt-2"},
                ]
            },
            scorecard={"miner_x": {"score": 0.9}},
            accepted_completions=[
                {"completion_id": "c1", "miner_hotkey": "miner_x", "text": "ans1"},
                {"completion_id": "c2", "miner_hotkey": "miner_y", "text": "ans2"},
            ],
            verdict_artifacts=[
                _verdict_artifact("c1", VALIDATOR),
                _verdict_artifact("c2", VALIDATOR),
                _verdict_artifact("c1", "ledger-validator-b"),
                _verdict_artifact("c2", "ledger-validator-b"),
            ],
        )
        print(f"      publish_ok={rollout_outcome.success} key={rollout_outcome.key}")
        if not rollout_outcome.success:
            raise RuntimeError(
                f"rollout bundle publish failed: {rollout_outcome.last_error}"
            )
        fetcher = RolloutBundleFetcher(
            backend=backend,
            verifier=make_hmac_verifier({VALIDATOR: VALIDATOR_SECRET}),
            netuid=INFERENCE_NETUID,
        )
        fetched = fetcher.fetch()
        print(f"      fetched_bundles={len(fetched.bundles)}")
        assert len(fetched.bundles) == 1, (
            f"expected 1 bundle, got {len(fetched.bundles)}"
        )
        bundle = fetched.bundles[0]
        print(f"      bundle.window_id={bundle.window_id}")
        print(f"      bundle.completions={len(bundle.completions)}")
        print(f"      bundle.verdicts={len(bundle.verdicts)}")
        assert len(bundle.completions) == 2
        assert len(bundle.verdicts) == 2
        assert bundle.netuid == INFERENCE_NETUID

        print("\n==> all phases passed — closed loop round-trip verified end-to-end")
    except Exception as exc:
        print(f"\nfailed: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        ok = False
    finally:
        if args.no_cleanup:
            print(f"\n(skipping cleanup; keys live under smoke/{RUN_ID}/)")
        else:
            print(f"\n==> cleanup: removing {len(backend._written_keys)} keys")
            deleted = backend.cleanup()
            print(f"    cleaned {deleted} keys")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
