from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .protocol.artifacts import make_artifact
from .utils.json_io import read_json, write_json


def _read_gzip_json_bytes(data: bytes) -> Any:
    return json.loads(gzip.decompress(data).decode("utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ImportedWindow:
    manifest: dict[str, Any]
    task_batch: dict[str, Any]
    scorecard: dict[str, Any] | None
    completions: list[dict[str, Any]]
    verdicts: list[dict[str, Any]]


class InferenceRegistryAdapter:
    def __init__(self, cfg: dict[str, Any], cache_root: Path) -> None:
        self.cfg = cfg
        self.cache_root = cache_root / "inference_imports"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.backend = str(cfg.get("inference_registry_backend", "local"))
        self.local_root = Path(str(cfg.get("inference_registry_local_root", "")).strip())
        self.bucket = str(cfg.get("inference_registry_bucket", "")).strip()
        self.endpoint_url = str(cfg.get("inference_registry_endpoint_url", "")).strip()
        self.access_key_id = str(cfg.get("inference_registry_access_key_id", "")).strip()
        self.secret_access_key = str(cfg.get("inference_registry_secret_access_key", "")).strip()
        self.region = str(cfg.get("inference_registry_region", "auto")).strip() or "auto"
        self._client = None

    def _s3_client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url or None,
                aws_access_key_id=self.access_key_id or None,
                aws_secret_access_key=self.secret_access_key or None,
                region_name=self.region or None,
            )
        return self._client

    def _read_json_artifact(self, key_or_path: str) -> dict[str, Any]:
        if self.backend == "local":
            path = Path(key_or_path)
            if not path.is_absolute():
                path = self.local_root / path
            return read_json(path)
        response = self._s3_client().get_object(Bucket=self.bucket, Key=key_or_path)
        return json.loads(response["Body"].read().decode("utf-8"))

    def _read_gzip_bundle(self, ref: dict[str, Any]) -> list[dict[str, Any]]:
        if ref.get("backend") == "local" or (self.backend == "local" and ref.get("path")):
            return _read_gzip_json_bytes(Path(str(ref["path"])).read_bytes())
        key = str(ref.get("key", ""))
        response = self._s3_client().get_object(Bucket=self.bucket, Key=key)
        return _read_gzip_json_bytes(response["Body"].read())

    def _list_manifest_refs(self) -> list[str]:
        if self.backend == "local":
            directory = self.local_root / "window_manifests"
            if not directory.exists():
                return []
            return [str(path) for path in sorted(directory.glob("*.json"))]
        response = self._s3_client().list_objects_v2(Bucket=self.bucket, Prefix="window_manifests/")
        return sorted(str(item["Key"]) for item in response.get("Contents", []) if item["Key"].endswith(".json"))

    def _artifact_key(self, artifact_type: str, artifact_id: str) -> str:
        folder = {
            "task_batch": "task_batches",
            "scorecard": "scorecards",
            "window_manifest": "window_manifests",
        }.get(artifact_type, f"{artifact_type}s")
        return f"{folder}/{artifact_id}.json"

    def _manifest_rank(self, manifest: dict[str, Any]) -> tuple[int, str]:
        publish_result = manifest.get("payload", {}).get("chain_publish_result")
        finalized = 1 if publish_result is not None else 0
        return finalized, str(manifest.get("created_at", ""))

    def finalized_window_manifests(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        manifests = [self._read_json_artifact(ref) for ref in self._list_manifest_refs()]
        by_window: dict[int, dict[str, Any]] = {}
        for manifest in manifests:
            if manifest.get("payload", {}).get("chain_publish_result") is None:
                continue
            window_id = int(manifest["window_id"])
            current = by_window.get(window_id)
            if current is None or self._manifest_rank(manifest) > self._manifest_rank(current):
                by_window[window_id] = manifest
        ordered = sorted(by_window.values(), key=lambda item: int(item["window_id"]))
        return ordered[-limit:] if limit is not None else ordered

    def load_window(self, *, window_id: int) -> ImportedWindow:
        manifests = [manifest for manifest in self.finalized_window_manifests() if int(manifest["window_id"]) == int(window_id)]
        if not manifests:
            raise FileNotFoundError(f"No finalized inference window found for {window_id}")
        manifest = manifests[-1]
        task_batch = self._read_json_artifact(
            str(manifest["payload"].get("task_batch_ref", "")) or self._artifact_key("task_batch", manifest["payload"]["task_batch_id"])
        )
        scorecard_id = manifest["payload"].get("scorecard_id")
        scorecard = (
            self._read_json_artifact(self._artifact_key("scorecard", scorecard_id))
            if scorecard_id
            else None
        )
        completions: list[dict[str, Any]] = []
        for ref in manifest["payload"].get("completion_bundle_refs", []):
            completions.extend(self._read_gzip_bundle(ref))
        verdict_ref = manifest["payload"].get("verdict_bundle_ref")
        verdicts = self._read_gzip_bundle(verdict_ref) if verdict_ref else []
        return ImportedWindow(
            manifest=manifest,
            task_batch=task_batch,
            scorecard=scorecard,
            completions=completions,
            verdicts=verdicts,
        )

    def cursor_path(self, cursor_name: str) -> Path:
        return self.cache_root / f"{cursor_name}.json"

    def read_cursor(self, cursor_name: str) -> int | None:
        path = self.cursor_path(cursor_name)
        if not path.exists():
            return None
        payload = read_json(path)
        window_id = payload.get("window_id")
        return int(window_id) if window_id is not None else None

    def write_cursor(self, cursor_name: str, *, window_id: int) -> None:
        write_json(
            self.cursor_path(cursor_name),
            {"cursor_name": cursor_name, "window_id": int(window_id), "updated_at": _utc_now()},
        )

    def next_finalized_window(self, *, cursor_name: str) -> int | None:
        current = self.read_cursor(cursor_name)
        manifests = self.finalized_window_manifests()
        for manifest in manifests:
            window_id = int(manifest["window_id"])
            if current is None or window_id > current:
                return window_id
        return None

    def import_reasoning_window(
        self,
        *,
        registry,
        policy_artifact: dict[str, Any],
        target_window_id: int,
        source_window_id: int,
        cursor_name: str,
    ) -> dict[str, Any]:
        for existing_batch in registry.list_artifacts("training_batch", window_id=target_window_id):
            payload = existing_batch.get("payload", {})
            if int(source_window_id) in [int(value) for value in payload.get("source_window_ids", [])]:
                rollout_artifacts = [
                    artifact
                    for artifact in registry.list_artifacts("rollout", window_id=target_window_id)
                    if int(artifact.get("payload", {}).get("source_lineage", {}).get("source_registry_ref", {}).get("window_id", -1)) == int(source_window_id)
                ]
                verdict_artifacts = [
                    artifact
                    for artifact in registry.list_artifacts("verdict", window_id=target_window_id)
                    if int(artifact.get("payload", {}).get("source_window_id", -1)) == int(source_window_id)
                ]
                self.write_cursor(cursor_name, window_id=source_window_id)
                return {
                    "imported_window": self.load_window(window_id=source_window_id),
                    "rollout_artifacts": rollout_artifacts,
                    "verdict_artifacts": verdict_artifacts,
                    "training_batch_artifact": existing_batch,
                }
        imported = self.load_window(window_id=source_window_id)
        tasks_by_id = {task["task_id"]: task for task in imported.task_batch["payload"]["tasks"]}
        completions_by_id = {completion["artifact_id"]: completion for completion in imported.completions}
        rollout_artifacts: list[dict[str, Any]] = []
        verdict_artifacts: list[dict[str, Any]] = []
        source_registry_ref = {
            "backend": self.backend,
            "local_root": str(self.local_root) if self.backend == "local" else "",
            "bucket": self.bucket if self.backend != "local" else "",
            "window_id": source_window_id,
            "manifest_id": imported.manifest["artifact_id"],
        }
        reward_contract = dict(policy_artifact["payload"].get("reward_contract", {}))
        for imported_verdict in imported.verdicts:
            source_payload = imported_verdict["payload"]
            completion = completions_by_id.get(source_payload["completion_id"])
            if completion is None:
                continue
            completion_payload = completion["payload"]
            task = tasks_by_id.get(source_payload["task_id"], {})
            correctness = float(source_payload.get("correctness", 0.0))
            policy_compliance = float(source_payload.get("policy_compliance", 0.0))
            format_ok = bool(source_payload.get("format_ok", False))
            accepted = bool(source_payload.get("accepted", False))
            difficulty = float(source_payload.get("difficulty", task.get("difficulty", 0.0)))
            dense_reward = (
                0.55 * correctness
                + 0.20 * policy_compliance
                + 0.15 * (1.0 if format_ok else 0.0)
                + 0.10 * (1.0 if accepted else 0.0)
            ) * (0.75 + (0.25 * difficulty))
            challenge = {
                "challenge_id": source_payload["task_id"],
                "prompt": task.get("source_prompt", task.get("prompt", "")),
                "task": task,
                "source_window_id": source_window_id,
            }
            rollout_artifact = make_artifact(
                artifact_type="rollout",
                producer_id=str(completion["producer_id"]),
                producer_role="inference_import",
                policy_id=policy_artifact["artifact_id"],
                window_id=target_window_id,
                lane="rl",
                commodity_use="dual_use",
                parent_ids=[policy_artifact["artifact_id"], completion["artifact_id"], imported_verdict["artifact_id"]],
                payload={
                    "env_id": "reasoning",
                    "task_source": str(imported.manifest["payload"].get("task_source", "reasoning_tasks")),
                    "challenge_id": source_payload["task_id"],
                    "challenge": challenge,
                    "generation": {
                        "prompt_token_ids": completion_payload.get(
                            "prompt_token_ids",
                            completion_payload.get("tokens", [])[: int(completion_payload.get("prompt_length", 0))],
                        ),
                        "completion_token_ids": completion_payload.get("completion_token_ids", []),
                        "sequence_token_ids": completion_payload.get("tokens", []),
                        "completion_text": completion_payload.get("completion_text", ""),
                        "completion_logprobs": completion_payload.get("completion_logprobs", []),
                        "sum_logprob": float(completion_payload.get("old_sum_logprob", 0.0)),
                    },
                    "proof": {
                        "scheme": "inference_window_import",
                        "source_completion_id": completion["artifact_id"],
                        "source_verdict_id": imported_verdict["artifact_id"],
                        "source_window_manifest_id": imported.manifest["artifact_id"],
                    },
                    "reward_contract": reward_contract,
                    "reward_features": {
                        "final_answer": source_payload.get("final_answer"),
                        "correctness": correctness,
                        "policy_compliance": policy_compliance,
                        "format_ok": format_ok,
                        "format_reason": source_payload.get("format_reason"),
                        "difficulty": difficulty,
                        "contamination_overlap_tags": source_payload.get("contamination_overlap_tags", []),
                    },
                    "source_lineage": {
                        "source_repo": "reliquary-inference",
                        "source_runtime": "reliquary-inference",
                        "source_registry_ref": source_registry_ref,
                        "source_artifact_ids": {
                            "task_batch_id": imported.task_batch["artifact_id"],
                            "completion_id": completion["artifact_id"],
                            "verdict_id": imported_verdict["artifact_id"],
                            "scorecard_id": imported.scorecard["artifact_id"] if imported.scorecard else None,
                        },
                        "imported_at": _utc_now(),
                    },
                },
            )
            rollout_artifacts.append(rollout_artifact)
            training_eligible = bool(accepted and format_ok and correctness > 0.0)
            verdict_artifacts.append(
                make_artifact(
                    artifact_type="verdict",
                    producer_id=str(self.cfg.get("validator_id", self.cfg.get("trainer_id", "trainer"))),
                    producer_role="trainer_quorum_importer",
                    policy_id=policy_artifact["artifact_id"],
                    window_id=target_window_id,
                    lane="rl",
                    commodity_use="dual_use",
                    parent_ids=[policy_artifact["artifact_id"], rollout_artifact["artifact_id"], imported_verdict["artifact_id"]],
                    payload={
                        "env_id": "reasoning",
                        "task_source": imported.manifest["payload"].get("task_source", "reasoning_tasks"),
                        "challenge_id": source_payload["task_id"],
                        "task_id": source_payload["task_id"],
                        "rollout_artifact_id": rollout_artifact["artifact_id"],
                        "rollout_producer_id": completion["producer_id"],
                        "accepted": accepted,
                        "training_eligible": training_eligible,
                        "proof_valid": accepted,
                        "proof_reason": source_payload.get("hard_fail_reason") or source_payload.get("soft_fail_reason") or "ok",
                        "score": correctness,
                        "dense_reward": dense_reward,
                        "difficulty": difficulty,
                        "format_ok": format_ok,
                        "format_reason": source_payload.get("format_reason"),
                        "correctness_or_judge": correctness,
                        "policy_compliance": policy_compliance,
                        "final_answer": source_payload.get("final_answer"),
                        "challenge": challenge,
                        "generation": rollout_artifact["payload"]["generation"],
                        "reward_contract": reward_contract,
                        "source_registry_ref": source_registry_ref,
                        "source_window_id": source_window_id,
                        "source_completion_id": completion["artifact_id"],
                        "source_verdict_id": imported_verdict["artifact_id"],
                        "verdict_explanation": {
                            "summary": "imported_from_finalized_inference_window",
                            "imported": True,
                            "accepted": accepted,
                            "training_eligible": training_eligible,
                            "dense_reward": dense_reward,
                        },
                    },
                )
            )
        for artifact in rollout_artifacts + verdict_artifacts:
            registry.put_artifact(artifact)
        source_window_ids = [source_window_id]
        training_batch_artifact = make_artifact(
            artifact_type="training_batch",
            producer_id=str(self.cfg.get("trainer_id", "trainer")),
            producer_role="trainer_quorum_importer",
            policy_id=policy_artifact["artifact_id"],
            window_id=target_window_id,
            lane="rl",
            commodity_use="dual_use",
            parent_ids=[policy_artifact["artifact_id"], imported.manifest["artifact_id"], *[item["artifact_id"] for item in verdict_artifacts]],
            payload={
                "env_id": "reasoning",
                "task_source": imported.manifest["payload"].get("task_source", "reasoning_tasks"),
                "source_registry": source_registry_ref,
                "source_window_ids": source_window_ids,
                "source_artifact_ids": {
                    "window_manifest_id": imported.manifest["artifact_id"],
                    "task_batch_id": imported.task_batch["artifact_id"],
                    "scorecard_id": imported.scorecard["artifact_id"] if imported.scorecard else None,
                },
                "source_verdict_ids": [item["artifact_id"] for item in verdict_artifacts if item["payload"].get("training_eligible")],
                "challenge_ids": sorted({item["payload"]["challenge_id"] for item in verdict_artifacts}),
                "group_count": len({item["payload"]["challenge_id"] for item in verdict_artifacts if item["payload"].get("training_eligible")}),
                "accepted_verdict_count": sum(1 for item in verdict_artifacts if item["payload"].get("accepted")),
                "training_eligible_count": sum(1 for item in verdict_artifacts if item["payload"].get("training_eligible")),
                "mean_dense_reward": (
                    sum(float(item["payload"]["dense_reward"]) for item in verdict_artifacts if item["payload"].get("training_eligible"))
                    / max(sum(1 for item in verdict_artifacts if item["payload"].get("training_eligible")), 1)
                ),
                "mean_binary_score": (
                    sum(float(item["payload"]["score"]) for item in verdict_artifacts if item["payload"].get("training_eligible"))
                    / max(sum(1 for item in verdict_artifacts if item["payload"].get("training_eligible")), 1)
                ),
                "reward_contract": reward_contract,
                "import_cursor_name": cursor_name,
                "imported_at": _utc_now(),
            },
        )
        registry.put_artifact(training_batch_artifact)
        self.write_cursor(cursor_name, window_id=source_window_id)
        return {
            "imported_window": imported,
            "rollout_artifacts": rollout_artifacts,
            "verdict_artifacts": verdict_artifacts,
            "training_batch_artifact": training_batch_artifact,
        }
