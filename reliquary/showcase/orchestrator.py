from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from ..chain.adapters import BittensorChainAdapter, LocalChainAdapter
from ..protocol.artifacts import make_artifact
from ..roles.miner import mine_window
from ..roles.trainer import generate_challenge_artifacts, publish_policy, train_window
from ..roles.validator import validate_window
from ..storage.local_registry import LocalArtifactRegistry
from ..utils.json_io import write_json
from .dashboard import build_dashboard_bundle, write_dashboard
from .profiles import miner_secret_map


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _make_chain_adapter(cfg: dict[str, Any], profile: dict[str, Any]):
    if profile.get("chain_mode") == "bittensor":
        return BittensorChainAdapter(
            network=cfg["network"],
            netuid=cfg["netuid"],
            wallet_name=cfg["wallet_name"],
            hotkey_name=cfg["hotkey_name"],
        )
    return LocalChainAdapter()


def _aggregate_metrics(window_records: list[dict[str, Any]]) -> dict[str, int]:
    totals = {
        "accepted_count": 0,
        "training_eligible_count": 0,
        "rejected_count": 0,
        "proof_failures": 0,
        "format_failures": 0,
        "non_unique_count": 0,
    }
    for record in window_records:
        metrics = record["scorecard_artifact"]["payload"]["window_metrics"]
        for key in totals:
            totals[key] += int(metrics[key])
    return totals


def _build_window_manifest(
    cfg: dict[str, Any],
    *,
    run_id: str,
    policy_artifact: dict[str, Any],
    challenge_artifacts: list[dict[str, Any]],
    rollout_artifacts: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
    scorecard_artifact: dict[str, Any],
    training_summary: dict[str, Any],
    miner_profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "run_id": run_id,
        "window_id": policy_artifact["window_id"],
        "policy_id": policy_artifact["artifact_id"],
        "challenge_ids": [artifact["payload"]["challenge"]["challenge_id"] for artifact in challenge_artifacts],
        "challenge_artifact_ids": [artifact["artifact_id"] for artifact in challenge_artifacts],
        "rollout_artifact_ids": [artifact["artifact_id"] for artifact in rollout_artifacts],
        "verdict_artifact_ids": [artifact["artifact_id"] for artifact in verdicts],
        "scorecard_artifact_id": scorecard_artifact["artifact_id"],
        "training_batch_artifact_id": training_summary["training_batch_artifact"]["artifact_id"],
        "checkpoint_artifact_id": training_summary["checkpoint_artifact"]["artifact_id"],
        "next_policy_artifact_id": training_summary["next_policy_artifact"]["artifact_id"],
        "participating_miners": [
            {
                "miner_id": miner["miner_id"],
                "label": miner["label"],
                "archetype": miner["archetype"],
            }
            for miner in miner_profiles
        ],
        "accepted_count": scorecard_artifact["payload"]["window_metrics"]["accepted_count"],
        "training_eligible_count": scorecard_artifact["payload"]["window_metrics"][
            "training_eligible_count"
        ],
        "rejected_count": scorecard_artifact["payload"]["window_metrics"]["rejected_count"],
        "weights": scorecard_artifact["payload"]["weights"],
        "eval_before": training_summary["eval_before"],
        "eval_after": training_summary["eval_after"],
    }
    return make_artifact(
        artifact_type="window_manifest",
        producer_id=cfg["trainer_id"],
        policy_id=policy_artifact["artifact_id"],
        window_id=policy_artifact["window_id"],
        payload=payload,
    )


def _build_run_manifest(
    cfg: dict[str, Any],
    *,
    run_id: str,
    profile: dict[str, Any],
    run_dir: Path,
    initial_policy: dict[str, Any],
    final_policy: dict[str, Any],
    window_manifests: list[dict[str, Any]],
    aggregate_metrics: dict[str, int],
) -> dict[str, Any]:
    payload = {
        "run_id": run_id,
        "profile": profile["name"],
        "profile_label": profile["label"],
        "description": profile["description"],
        "initial_policy_id": initial_policy["artifact_id"],
        "final_policy_id": final_policy["artifact_id"],
        "window_manifest_ids": [artifact["artifact_id"] for artifact in window_manifests],
        "aggregate_metrics": aggregate_metrics,
        "output_assets": {
            "run_dir": str(run_dir),
            "artifacts_dir": str(run_dir / "artifacts"),
            "chain_events": str(run_dir / "chain-events.json"),
            "run_manifest": str(run_dir / "run-manifest.json"),
            "dashboard_html": str(run_dir / "dashboard" / "index.html"),
            "dashboard_data": str(run_dir / "dashboard" / "dashboard-data.json"),
            "summary_markdown": str(run_dir / "SUMMARY.md"),
        },
    }
    return make_artifact(
        artifact_type="run_manifest",
        producer_id=cfg["trainer_id"],
        policy_id=final_policy["artifact_id"],
        window_id=window_manifests[-1]["window_id"] if window_manifests else 0,
        payload=payload,
    )


def _print_window_summary(console: Console, record: dict[str, Any], profile: dict[str, Any]) -> None:
    window = record["window_id"]
    console.rule(f"Window {window:04d}")
    metrics = record["scorecard_artifact"]["payload"]["window_metrics"]
    summary_table = Table(show_header=True, header_style="bold cyan")
    summary_table.add_column("Miner")
    summary_table.add_column("Archetype")
    summary_table.add_column("Accepted")
    summary_table.add_column("Training Eligible")
    summary_table.add_column("Weight")
    for miner in profile["miners"]:
        stats = record["scorecard_artifact"]["payload"]["miners"].get(
            miner["miner_id"],
            {
                "accepted_count": 0,
                "training_eligible_count": 0,
                "weight": 0.0,
            },
        )
        summary_table.add_row(
            miner["label"],
            miner["archetype"],
            str(stats["accepted_count"]),
            str(stats["training_eligible_count"]),
            f"{float(stats.get('weight', 0.0)):.3f}",
        )
    console.print(summary_table)
    console.print(
        "[bold]Window metrics:[/bold] "
        f"accepted={metrics['accepted_count']} "
        f"training_eligible={metrics['training_eligible_count']} "
        f"rejected={metrics['rejected_count']} "
        f"proof_failures={metrics['proof_failures']} "
        f"format_failures={metrics['format_failures']} "
        f"non_unique={metrics['non_unique_count']}"
    )
    console.print(
        "[bold]Training delta:[/bold] "
        f"dense {record['training_summary']['eval_before']['mean_dense_reward']:.3f} -> "
        f"{record['training_summary']['eval_after']['mean_dense_reward']:.3f} | "
        f"success {record['training_summary']['eval_before']['success_rate']:.3f} -> "
        f"{record['training_summary']['eval_after']['success_rate']:.3f}"
    )


def run_showcase(
    registry: LocalArtifactRegistry,
    cfg: dict[str, Any],
    profile: dict[str, Any],
    *,
    run_id: str | None = None,
    console: Console | None = None,
) -> dict[str, Any]:
    active_cfg = dict(cfg)
    active_cfg["rollouts_per_challenge"] = int(profile["rollouts_per_challenge"])
    active_cfg["miner_secrets"] = miner_secret_map(profile)
    seed_offset = int(profile["seed"]) * 100_000
    run_id = run_id or f"{profile['name']}-{profile['seed']}-{_timestamp_slug()}"
    run_dir = registry.run_dir(run_id)
    console = console or Console()
    chain_adapter = _make_chain_adapter(active_cfg, profile)

    console.rule(f"[bold]Reliquary showcase[/bold] `{run_id}`")
    console.print(profile["description"])

    all_artifacts: list[dict[str, Any]] = []
    window_records: list[dict[str, Any]] = []
    chain_events: list[dict[str, Any]] = []

    initial_policy = publish_policy(active_cfg, window_id=0)
    registry.put_artifact(initial_policy)
    all_artifacts.append(initial_policy)
    chain_events.append(chain_adapter.publish_policy_reference(initial_policy))

    for window in range(int(profile["windows"])):
        current_policy = registry.latest_policy()
        challenge_artifacts = generate_challenge_artifacts(
            active_cfg,
            current_policy,
            window_id=window,
            count=int(profile["challenge_count"]),
            seed_offset=seed_offset,
        )
        for artifact in challenge_artifacts:
            registry.put_artifact(artifact)
        all_artifacts.extend(challenge_artifacts)

        chain_events.append(
            chain_adapter.publish_window_metadata(
                run_id=run_id,
                window_id=window,
                policy_id=current_policy["artifact_id"],
                challenge_ids=[artifact["payload"]["challenge"]["challenge_id"] for artifact in challenge_artifacts],
                miner_ids=[miner["miner_id"] for miner in profile["miners"]],
            )
        )

        rollout_artifacts: list[dict[str, Any]] = []
        for miner in profile["miners"]:
            miner_rollouts = mine_window(
                active_cfg,
                current_policy,
                challenge_artifacts,
                window_id=window,
                miner_profile=miner,
                rollouts_per_challenge=int(profile["rollouts_per_challenge"]),
            )
            for artifact in miner_rollouts:
                registry.put_artifact(artifact)
                chain_events.append(chain_adapter.submit_miner_result_reference(artifact))
            rollout_artifacts.extend(miner_rollouts)
        all_artifacts.extend(rollout_artifacts)

        verdicts, scorecard_artifact = validate_window(
            active_cfg,
            current_policy,
            rollout_artifacts,
            window_id=window,
        )
        for artifact in verdicts:
            registry.put_artifact(artifact)
        registry.put_artifact(scorecard_artifact)
        all_artifacts.extend(verdicts)
        all_artifacts.append(scorecard_artifact)
        chain_events.append(chain_adapter.publish_scorecard(scorecard_artifact))

        training_summary = train_window(
            registry,
            active_cfg,
            current_policy,
            verdicts,
            window_id=window,
        )
        all_artifacts.extend(
            [
                training_summary["training_batch_artifact"],
                training_summary["checkpoint_artifact"],
                training_summary["next_policy_artifact"],
            ]
        )
        chain_events.append(
            chain_adapter.publish_checkpoint_reference(
                training_summary["checkpoint_artifact"],
                training_summary["next_policy_artifact"],
            )
        )

        window_manifest = _build_window_manifest(
            active_cfg,
            run_id=run_id,
            policy_artifact=current_policy,
            challenge_artifacts=challenge_artifacts,
            rollout_artifacts=rollout_artifacts,
            verdicts=verdicts,
            scorecard_artifact=scorecard_artifact,
            training_summary=training_summary,
            miner_profiles=profile["miners"],
        )
        registry.put_artifact(window_manifest)
        all_artifacts.append(window_manifest)

        record = {
            "window_id": window,
            "policy_artifact": current_policy,
            "challenge_artifacts": challenge_artifacts,
            "rollout_artifacts": rollout_artifacts,
            "verdicts": verdicts,
            "scorecard_artifact": scorecard_artifact,
            "training_summary": training_summary,
            "window_manifest": window_manifest,
        }
        window_records.append(record)
        _print_window_summary(console, record, profile)

    final_policy = window_records[-1]["training_summary"]["next_policy_artifact"]
    aggregate_metrics = _aggregate_metrics(window_records)
    window_manifests = [record["window_manifest"] for record in window_records]
    run_manifest = _build_run_manifest(
        active_cfg,
        run_id=run_id,
        profile=profile,
        run_dir=run_dir,
        initial_policy=initial_policy,
        final_policy=final_policy,
        window_manifests=window_manifests,
        aggregate_metrics=aggregate_metrics,
    )
    registry.put_artifact(run_manifest)
    all_artifacts.append(run_manifest)

    write_json(run_dir / "chain-events.json", chain_events)
    write_json(run_dir / "run-manifest.json", run_manifest)
    registry.export_artifacts(all_artifacts, run_dir / "artifacts")

    bundle = build_dashboard_bundle(run_manifest, window_records, chain_events, profile)
    dashboard_paths = write_dashboard(run_dir, bundle)

    console.rule("[bold]Showcase export ready[/bold]")
    console.print(f"Run directory: {run_dir}")
    console.print(f"Dashboard: {dashboard_paths['dashboard_html']}")
    console.print(f"Summary: {dashboard_paths['summary_markdown']}")

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "run_manifest": run_manifest,
        "dashboard": dashboard_paths,
        "chain_events": chain_events,
        "window_records": window_records,
    }
