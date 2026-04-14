from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..protocol.artifacts import make_artifact
from ..storage.registry import ArtifactRegistry


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def build_distill_pack(
    registry: ArtifactRegistry,
    cfg: dict[str, Any],
    policy_artifact: dict[str, Any],
    verdict_artifacts: list[dict[str, Any]],
    *,
    window_id: int,
) -> dict[str, Any]:
    accepted = [
        verdict
        for verdict in verdict_artifacts
        if verdict["payload"].get("training_eligible", False)
    ]
    pack_records = [
        {
            "task_id": verdict["payload"]["task_id"],
            "prompt": verdict["payload"]["task"]["prompt"],
            "trace_text": verdict["payload"]["trace"]["trace_text"],
            "final_answer": verdict["payload"]["trace"]["final_answer"],
            "reference_answer": verdict["payload"]["task"]["reference_answer"],
            "operation": verdict["payload"]["task"]["operation"],
            "difficulty": verdict["payload"]["difficulty"],
            "producer_id": verdict["payload"]["trace_producer_id"],
            "weight_score": verdict["payload"]["weight_score"],
            "commodity_use": verdict.get("commodity_use", "dual_use"),
        }
        for verdict in accepted
    ]
    pack_records = sorted(pack_records, key=lambda item: (item["task_id"], item["producer_id"]))
    pack_id = f"distill-pack-window-{window_id:04d}"
    pack_dir = registry.pack_dir(pack_id)
    jsonl_path = pack_dir / "pack.jsonl"
    parquet_path = pack_dir / "pack.parquet"
    _write_jsonl(jsonl_path, pack_records)
    _write_parquet(parquet_path, pack_records)
    task_mix: dict[str, int] = {}
    for record in pack_records:
        task_mix[record["operation"]] = task_mix.get(record["operation"], 0) + 1

    pack_artifact = make_artifact(
        artifact_type="distill_pack",
        producer_id=cfg["trainer_id"],
        producer_role="packer",
        policy_id=policy_artifact["artifact_id"],
        window_id=window_id,
        lane="distill",
        commodity_use="distill_only",
        parent_ids=[policy_artifact["artifact_id"], *[item["artifact_id"] for item in accepted]],
        payload={
            "pack_id": pack_id,
            "source_verdict_ids": [item["artifact_id"] for item in accepted],
            "pack_schema_version": "v1",
            "record_count": len(pack_records),
            "task_mix": task_mix,
            "quality_summary": {
                "mean_weight_score": (
                    sum(float(record["weight_score"]) for record in pack_records) / len(pack_records)
                    if pack_records
                    else 0.0
                ),
            },
            "dedup_summary": {
                "unique_record_count": len(pack_records),
                "duplicate_count": len(verdict_artifacts) - len(accepted),
            },
            "contamination_summary": {
                "contaminated_rejected_count": sum(
                    1
                    for verdict in verdict_artifacts
                    if not verdict["payload"].get("contamination_ok", True)
                ),
            },
            "export_refs": {
                "jsonl": str(jsonl_path),
                "parquet": str(parquet_path),
            },
        },
    )
    registry.put_artifact(pack_artifact)
    return pack_artifact
