from __future__ import annotations

from pathlib import Path
import time

import typer

from .roles.generation import generate_distill_traces
from .roles.learner import train_distill_window
from .roles.packer import build_distill_pack
from .roles.policy_authority import publish_distill_policy
from .roles.tasking import generate_reasoning_task_batch_artifact
from .roles.miner import mine_window
from .roles.trainer import (
    generate_challenge_artifacts,
    import_inference_window,
    publish_policy,
    run_reasoning_rl_cycle,
    train_window,
)
from .roles.validator import validate_distill_window, validate_window
from .shared.bootstrap import bootstrap_runtime
from .showcase.orchestrator import run_showcase
from .showcase.distill_orchestrator import run_distill_showcase
from .showcase.profiles import get_demo_profile
from .utils.json_io import write_json

app = typer.Typer(no_args_is_help=True)


@app.command("publish-policy")
def publish_policy_cmd(window: int = 0) -> None:
    cfg, registry = bootstrap_runtime()
    policy = publish_policy(cfg, window_id=window)
    registry.put_artifact(policy)
    typer.echo(f"Published policy {policy['artifact_id']} for window {window}")


@app.command("generate-challenges")
def generate_challenges_cmd(window: int = 0, count: int = 8) -> None:
    cfg, registry = bootstrap_runtime()
    policy = registry.latest_policy()
    if policy is None:
        policy = publish_policy(cfg, window_id=window)
        registry.put_artifact(policy)
    challenges = generate_challenge_artifacts(cfg, policy, window_id=window, count=count)
    for artifact in challenges:
        registry.put_artifact(artifact)
    typer.echo(f"Generated {len(challenges)} challenges for window {window}")


@app.command("mine")
def mine_cmd(window: int = 0) -> None:
    cfg, registry = bootstrap_runtime()
    policy = registry.latest_policy()
    if policy is None:
        raise typer.BadParameter("No policy found. Run `reliquary publish-policy` first.")
    challenges = registry.list_artifacts("challenge", window_id=window)
    if not challenges:
        challenges = generate_challenge_artifacts(cfg, policy, window_id=window, count=4)
        for artifact in challenges:
            registry.put_artifact(artifact)
    rollouts = mine_window(cfg, policy, challenges, window_id=window)
    for artifact in rollouts:
        registry.put_artifact(artifact)
    typer.echo(f"Mined {len(rollouts)} rollout artifacts for window {window}")


@app.command("validate")
def validate_cmd(window: int = 0) -> None:
    cfg, registry = bootstrap_runtime()
    policy = registry.latest_policy()
    if policy is None:
        raise typer.BadParameter("No policy found. Run `reliquary publish-policy` first.")
    rollouts = registry.list_artifacts("rollout", window_id=window)
    if not rollouts:
        raise typer.BadParameter("No rollout artifacts found. Run `reliquary mine` first.")
    verdicts, scorecard_artifact = validate_window(cfg, policy, rollouts, window_id=window)
    for artifact in verdicts:
        registry.put_artifact(artifact)
    registry.put_artifact(scorecard_artifact)
    write_json(
        Path(cfg["data_dir"]) / "verdicts" / f"weights-window-{window:04d}.json",
        scorecard_artifact["payload"],
    )
    typer.echo(f"Validated {len(verdicts)} rollouts for window {window}")


@app.command("train")
def train_cmd(window: int = 0) -> None:
    cfg, registry = bootstrap_runtime()
    policy = registry.latest_policy()
    if policy is None:
        raise typer.BadParameter("No policy found. Run `reliquary publish-policy` first.")
    verdicts = registry.list_artifacts("verdict", window_id=window)
    if not verdicts:
        raise typer.BadParameter("No verdict artifacts found. Run `reliquary validate` first.")
    summary = train_window(registry, cfg, policy, verdicts, window_id=window)
    typer.echo(
        "Trained window "
        f"{window}: before={summary['eval_before']['mean_dense_reward']:.3f} "
        f"after={summary['eval_after']['mean_dense_reward']:.3f}"
    )


@app.command("import-inference-window")
def import_inference_window_cmd(window: int = 0, source_window: int | None = None) -> None:
    cfg, registry = bootstrap_runtime()
    policy = registry.latest_policy()
    if policy is None:
        policy = publish_policy(
            cfg,
            window_id=window,
            env_id="reasoning",
            task_source=str(cfg.get("rl_task_source", "reasoning_tasks")),
        )
        registry.put_artifact(policy)
    summary = import_inference_window(
        registry,
        cfg,
        policy,
        source_window_id=source_window,
        target_window_id=window,
    )
    typer.echo(
        f"Imported inference window into RL window {window}: "
        f"{len(summary['verdict_artifacts'])} trajectories"
    )


@app.command("run-rl-cycle")
def run_rl_cycle_cmd(window: int = 0, source_window: int | None = None) -> None:
    cfg, registry = bootstrap_runtime()
    result = run_reasoning_rl_cycle(
        registry,
        cfg,
        source_window_id=source_window,
        window_id=window,
    )
    training = result["training_summary"]
    metric_before = training["eval_before"].get("contract_score", training["eval_before"].get("mean_dense_reward", 0.0))
    metric_after = training["eval_after"].get("contract_score", training["eval_after"].get("mean_dense_reward", 0.0))
    typer.echo(
        "RL cycle complete: "
        f"before={metric_before:.3f} "
        f"after={metric_after:.3f} "
        f"approved={training['decision_artifact']['payload']['approved']}"
    )


@app.command("publish-distill-policy")
def publish_distill_policy_cmd(window: int = 0) -> None:
    cfg, registry = bootstrap_runtime()
    policy = publish_distill_policy(cfg, window_id=window)
    registry.put_artifact(policy)
    typer.echo(f"Published distill policy {policy['artifact_id']} for window {window}")


@app.command("generate-task-batch")
def generate_task_batch_cmd(window: int = 0, count: int | None = None) -> None:
    cfg, registry = bootstrap_runtime()
    policy = registry.latest_policy()
    if policy is None or policy["payload"].get("env_id") != "reasoning":
        policy = publish_distill_policy(cfg, window_id=window)
        registry.put_artifact(policy)
    task_batch = generate_reasoning_task_batch_artifact(
        cfg,
        policy,
        window_id=window,
        count=count or cfg["distill_train_task_count"],
    )
    registry.put_artifact(task_batch)
    typer.echo(f"Generated task batch {task_batch['artifact_id']} for window {window}")


@app.command("distill-generate")
def distill_generate_cmd(window: int = 0) -> None:
    cfg, registry = bootstrap_runtime()
    policy = registry.latest_policy()
    if policy is None or policy["payload"].get("env_id") != "reasoning":
        policy = publish_distill_policy(cfg, window_id=window)
        registry.put_artifact(policy)
    task_batches = registry.list_artifacts("task_batch", window_id=window)
    if not task_batches:
        task_batch = generate_reasoning_task_batch_artifact(
            cfg,
            policy,
            window_id=window,
            count=cfg["distill_train_task_count"],
        )
        registry.put_artifact(task_batch)
    else:
        task_batch = task_batches[-1]
    profile = get_demo_profile("distill")
    count = 0
    for miner in profile["miners"]:
        artifacts = generate_distill_traces(
            cfg,
            policy,
            task_batch,
            window_id=window,
            miner_profile=miner,
        )
        count += len(artifacts)
        for artifact in artifacts:
            registry.put_artifact(artifact)
    typer.echo(f"Generated {count} distillation traces for window {window}")


@app.command("distill-validate")
def distill_validate_cmd(window: int = 0) -> None:
    cfg, registry = bootstrap_runtime()
    policy = registry.latest_policy()
    if policy is None:
        raise typer.BadParameter("No policy found. Run `reliquary publish-distill-policy` first.")
    traces = registry.list_artifacts("distill_trace", window_id=window)
    if not traces:
        raise typer.BadParameter("No distill traces found. Run `reliquary distill-generate` first.")
    verdicts, scorecard = validate_distill_window(cfg, policy, traces, window_id=window)
    for artifact in verdicts:
        registry.put_artifact(artifact)
    registry.put_artifact(scorecard)
    typer.echo(f"Validated {len(verdicts)} distillation traces for window {window}")


@app.command("pack-distill")
def pack_distill_cmd(window: int = 0) -> None:
    cfg, registry = bootstrap_runtime()
    policy = registry.latest_policy()
    if policy is None:
        raise typer.BadParameter("No policy found. Run `reliquary publish-distill-policy` first.")
    verdicts = [artifact for artifact in registry.list_artifacts("verdict", window_id=window) if artifact.get("lane") == "distill"]
    if not verdicts:
        raise typer.BadParameter("No distill verdicts found. Run `reliquary distill-validate` first.")
    pack = build_distill_pack(registry, cfg, policy, verdicts, window_id=window)
    typer.echo(f"Built distill pack {pack['artifact_id']} for window {window}")


@app.command("distill-train")
def distill_train_cmd(window: int = 0) -> None:
    cfg, registry = bootstrap_runtime()
    policy = registry.latest_policy()
    if policy is None:
        raise typer.BadParameter("No policy found. Run `reliquary publish-distill-policy` first.")
    packs = registry.list_artifacts("distill_pack", window_id=window)
    if not packs:
        raise typer.BadParameter("No distill pack found. Run `reliquary pack-distill` first.")
    summary = train_distill_window(registry, cfg, policy, packs[-1], window_id=window)
    typer.echo(
        "Distill-trained window "
        f"{window}: before={summary['eval_before']['success_rate']:.3f} "
        f"after={summary['eval_after']['success_rate']:.3f}"
    )


@app.command("distill-cycle")
def distill_cycle_cmd(
    windows: int = 2,
    tasks: int | None = None,
    profile: str | None = None,
    run_id: str | None = None,
) -> None:
    cfg, registry = bootstrap_runtime()
    demo_profile = get_demo_profile(
        profile or cfg.get("distill_profile", "distill"),
        windows=windows,
        challenge_count=tasks or cfg["distill_train_task_count"],
    )
    miner_limit = int(cfg.get("distill_generation_miner_limit", 0))
    if miner_limit > 0:
        demo_profile["miners"] = demo_profile["miners"][:miner_limit]
    result = run_distill_showcase(registry, cfg, demo_profile, run_id=run_id)
    typer.echo(
        f"Distillation cycle {result['run_id']} complete. "
        f"Dashboard: {result['dashboard']['dashboard_html']}"
    )


@app.command("run-distill-worker")
def run_distill_worker_cmd(
    once: bool = False,
    windows: int = 2,
    tasks: int | None = None,
    sleep_seconds: int | None = None,
) -> None:
    cfg, _ = bootstrap_runtime()
    interval = sleep_seconds or int(cfg["distill_cycle_interval_seconds"])
    while True:
        distill_cycle_cmd(windows=windows, tasks=tasks, profile=None, run_id=None)
        if once:
            break
        time.sleep(interval)


@app.command("serve-control-plane")
def serve_control_plane_cmd(host: str | None = None, port: int | None = None) -> None:
    import uvicorn

    from .control.api import app as control_api

    cfg, _ = bootstrap_runtime()
    uvicorn.run(
        control_api,
        host=host or cfg["control_api_host"],
        port=port or cfg["control_api_port"],
        log_level=cfg["log_level"].lower(),
    )


@app.command("demo")
def demo_cmd(
    profile: str = typer.Argument("showcase"),
    windows: int | None = None,
    challenges: int | None = None,
    rollouts_per_challenge: int | None = None,
    run_id: str | None = None,
) -> None:
    cfg, registry = bootstrap_runtime()
    demo_profile = get_demo_profile(
        profile,
        windows=windows,
        challenge_count=challenges,
        rollouts_per_challenge=rollouts_per_challenge,
    )
    mode = demo_profile.get("mode", "rl")
    if mode == "distill":
        result = run_distill_showcase(registry, cfg, demo_profile, run_id=run_id)
        typer.echo(
            f"Distillation run {result['run_id']} complete. "
            f"Dashboard: {result['dashboard']['dashboard_html']}"
        )
    else:
        result = run_showcase(registry, cfg, demo_profile, run_id=run_id)
        typer.echo(
            f"Showcase run {result['run_id']} complete. "
            f"Dashboard: {result['dashboard']['dashboard_html']}"
        )


if __name__ == "__main__":
    app()


def main() -> None:
    app()
