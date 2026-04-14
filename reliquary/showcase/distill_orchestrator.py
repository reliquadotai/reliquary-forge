from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

from ..chain.adapters import LocalChainAdapter
from ..protocol.artifacts import make_artifact
from ..roles.generation import generate_distill_traces
from ..roles.learner import train_distill_window
from ..roles.packer import build_distill_pack
from ..roles.policy_authority import publish_distill_policy
from ..roles.tasking import generate_reasoning_task_batch_artifact
from ..roles.validator_mesh import validate_distill_window
from ..storage.registry import ArtifactRegistry
from ..utils.json_io import write_json
from .distill_dashboard import build_distill_dashboard_bundle, write_distill_dashboard
from .profiles import miner_secret_map


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _build_window_manifest(
    cfg: dict[str, Any],
    *,
    run_id: str,
    policy_artifact: dict[str, Any],
    task_batch_artifact: dict[str, Any],
    trace_artifacts: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
    scorecard_artifact: dict[str, Any],
    pack_artifact: dict[str, Any],
    training_summary: dict[str, Any],
    miner_profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    return make_artifact(
        artifact_type="window_manifest",
        producer_id=cfg["trainer_id"],
        producer_role="policy_authority",
        policy_id=policy_artifact["artifact_id"],
        window_id=policy_artifact["window_id"],
        lane="distill",
        commodity_use="dual_use",
        parent_ids=[policy_artifact["artifact_id"], task_batch_artifact["artifact_id"]],
        payload={
            "run_id": run_id,
            "window_id": policy_artifact["window_id"],
            "policy_id": policy_artifact["artifact_id"],
            "task_batch_artifact_id": task_batch_artifact["artifact_id"],
            "trace_artifact_ids": [artifact["artifact_id"] for artifact in trace_artifacts],
            "verdict_artifact_ids": [artifact["artifact_id"] for artifact in verdicts],
            "scorecard_artifact_id": scorecard_artifact["artifact_id"],
            "pack_artifact_id": pack_artifact["artifact_id"],
            "checkpoint_artifact_id": training_summary["checkpoint_artifact"]["artifact_id"],
            "benchmark_result_artifact_id": training_summary["benchmark_result_artifact"]["artifact_id"],
            "decision_artifact_id": training_summary["decision_artifact"]["artifact_id"],
            "next_policy_artifact_id": training_summary["next_policy_artifact"]["artifact_id"],
            "participating_miners": [
                {
                    "miner_id": miner["miner_id"],
                    "label": miner["label"],
                    "archetype": miner["archetype"],
                }
                for miner in miner_profiles
            ],
        },
    )


def _build_run_manifest(
    cfg: dict[str, Any],
    *,
    run_id: str,
    profile: dict[str, Any],
    run_dir,
    initial_policy: dict[str, Any],
    final_policy: dict[str, Any],
    window_manifests: list[dict[str, Any]],
    aggregate_metrics: dict[str, int],
) -> dict[str, Any]:
    return make_artifact(
        artifact_type="run_manifest",
        producer_id=cfg["trainer_id"],
        producer_role="policy_authority",
        policy_id=final_policy["artifact_id"],
        window_id=window_manifests[-1]["window_id"] if window_manifests else 0,
        lane="distill",
        commodity_use="dual_use",
        parent_ids=[initial_policy["artifact_id"], final_policy["artifact_id"]],
        payload={
            "run_id": run_id,
            "profile": profile["name"],
            "profile_label": profile["label"],
            "description": profile["description"],
            "teacher_model": cfg.get("model_teacher", ""),
            "student_model": cfg.get("model_student", ""),
            "control_plane_version": cfg.get("service_version", "0.1.0"),
            "host_mode": cfg.get("host_mode", "local"),
            "service_endpoints": {
                "control_plane": f"http://{cfg.get('control_api_host', '127.0.0.1')}:{cfg.get('control_api_port', 8000)}",
                "teacher": cfg.get("vllm_base_url", ""),
            },
            "initial_policy_id": initial_policy["artifact_id"],
            "final_policy_id": final_policy["artifact_id"],
            "window_manifest_ids": [artifact["artifact_id"] for artifact in window_manifests],
            "aggregate_metrics": aggregate_metrics,
            "output_assets": {
                "run_dir": str(run_dir),
                "chain_events": str(run_dir / "chain-events.json"),
                "dashboard_html": str(run_dir / "dashboard" / "index.html"),
            },
        },
    )


def run_distill_showcase(
    registry: ArtifactRegistry,
    cfg: dict[str, Any],
    profile: dict[str, Any],
    *,
    run_id: str | None = None,
    console: Console | None = None,
) -> dict[str, Any]:
    active_cfg = dict(cfg)
    active_cfg["miner_secrets"] = miner_secret_map(profile)
    console = console or Console()
    chain_adapter = LocalChainAdapter()
    run_id = run_id or f"{profile['name']}-{profile['seed']}-{_timestamp_slug()}"
    run_dir = Path(active_cfg.get("export_dir", str(registry.run_dir(run_id)))) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    console.rule(f"[bold]Reliquary distillation[/bold] `{run_id}`")
    console.print(profile["description"])

    initial_policy = publish_distill_policy(active_cfg, window_id=0)
    registry.put_artifact(initial_policy)
    chain_events = [chain_adapter.publish_policy_reference(initial_policy)]
    window_records: list[dict[str, Any]] = []
    current_policy = initial_policy

    for window_id in range(int(profile["windows"])):
        policy_artifact = current_policy
        task_batch_artifact = generate_reasoning_task_batch_artifact(
            active_cfg,
            policy_artifact,
            window_id=window_id,
            count=int(profile["task_count"]),
            split="train",
            seed_offset=int(profile["seed"]) * 10_000,
        )
        registry.put_artifact(task_batch_artifact)
        trace_artifacts: list[dict[str, Any]] = []
        for miner in profile["miners"]:
            miner_traces = generate_distill_traces(
                active_cfg,
                policy_artifact,
                task_batch_artifact,
                window_id=window_id,
                miner_profile=miner,
            )
            for artifact in miner_traces:
                registry.put_artifact(artifact)
                chain_events.append(chain_adapter.submit_miner_result_reference(artifact))
            trace_artifacts.extend(miner_traces)
        verdicts, scorecard_artifact = validate_distill_window(
            active_cfg,
            policy_artifact,
            trace_artifacts,
            window_id=window_id,
        )
        for artifact in verdicts:
            registry.put_artifact(artifact)
        registry.put_artifact(scorecard_artifact)
        pack_artifact = build_distill_pack(
            registry,
            active_cfg,
            policy_artifact,
            verdicts,
            window_id=window_id,
        )
        training_summary = train_distill_window(
            registry,
            active_cfg,
            policy_artifact,
            pack_artifact,
            window_id=window_id,
        )
        chain_events.append(chain_adapter.publish_scorecard(scorecard_artifact))
        chain_events.append(
            chain_adapter.publish_checkpoint_reference(
                training_summary["checkpoint_artifact"],
                training_summary["next_policy_artifact"],
            )
        )
        window_manifest = _build_window_manifest(
            active_cfg,
            run_id=run_id,
            policy_artifact=policy_artifact,
            task_batch_artifact=task_batch_artifact,
            trace_artifacts=trace_artifacts,
            verdicts=verdicts,
            scorecard_artifact=scorecard_artifact,
            pack_artifact=pack_artifact,
            training_summary=training_summary,
            miner_profiles=profile["miners"],
        )
        registry.put_artifact(window_manifest)
        current_policy = training_summary["next_policy_artifact"]
        window_records.append(
            {
                "window_id": window_id,
                "policy_artifact": policy_artifact,
                "task_batch_artifact": task_batch_artifact,
                "trace_artifacts": trace_artifacts,
                "verdicts": verdicts,
                "scorecard_artifact": scorecard_artifact,
                "pack_artifact": pack_artifact,
                "training_summary": training_summary,
                "window_manifest": window_manifest,
            }
        )

    aggregate_metrics = {
        "accepted_count": sum(
            int(record["scorecard_artifact"]["payload"]["window_metrics"]["accepted_count"])
            for record in window_records
        ),
        "training_eligible_count": sum(
            int(record["scorecard_artifact"]["payload"]["window_metrics"]["training_eligible_count"])
            for record in window_records
        ),
        "rejected_count": sum(
            int(record["scorecard_artifact"]["payload"]["window_metrics"]["rejected_count"])
            for record in window_records
        ),
        "proof_failures": sum(
            int(record["scorecard_artifact"]["payload"]["window_metrics"]["proof_failures"])
            for record in window_records
        ),
        "format_failures": sum(
            int(record["scorecard_artifact"]["payload"]["window_metrics"]["format_failures"])
            for record in window_records
        ),
        "non_unique_count": sum(
            int(record["scorecard_artifact"]["payload"]["window_metrics"]["non_unique_count"])
            for record in window_records
        ),
        "contamination_failures": sum(
            int(record["scorecard_artifact"]["payload"]["window_metrics"]["contamination_failures"])
            for record in window_records
        ),
    }
    final_policy = current_policy
    run_manifest = _build_run_manifest(
        active_cfg,
        run_id=run_id,
        profile=profile,
        run_dir=run_dir,
        initial_policy=initial_policy,
        final_policy=final_policy,
        window_manifests=[record["window_manifest"] for record in window_records],
        aggregate_metrics=aggregate_metrics,
    )
    registry.put_artifact(run_manifest)
    write_json(run_dir / "chain-events.json", chain_events)
    write_json(run_dir / "run-manifest.json", run_manifest)
    bundle = build_distill_dashboard_bundle(run_manifest, window_records, chain_events, profile)
    dashboard_paths = write_distill_dashboard(run_dir / "dashboard", bundle)
    console.print(
        "Distillation run complete: "
        f"dashboard={dashboard_paths['dashboard_html']}"
    )
    return {
        "run_id": run_id,
        "run_manifest": run_manifest,
        "dashboard": dashboard_paths,
        "window_records": window_records,
    }
