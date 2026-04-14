from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from ..utils.json_io import write_json


def build_dashboard_bundle(
    run_manifest: dict[str, Any],
    window_records: list[dict[str, Any]],
    chain_events: list[dict[str, Any]],
    profile: dict[str, Any],
) -> dict[str, Any]:
    miner_catalog = {miner["miner_id"]: dict(miner) for miner in profile["miners"]}
    aggregate_miners: dict[str, dict[str, Any]] = {}

    for record in window_records:
        for miner_id, stats in record["scorecard_artifact"]["payload"]["miners"].items():
            aggregate = aggregate_miners.setdefault(
                miner_id,
                {
                    "miner_id": miner_id,
                    "label": miner_catalog.get(miner_id, {}).get("label", miner_id),
                    "archetype": stats.get("archetype", miner_catalog.get(miner_id, {}).get("archetype")),
                    "rollouts_seen": 0,
                    "accepted_count": 0,
                    "training_eligible_count": 0,
                    "rejected_count": 0,
                    "proof_failures": 0,
                    "format_failures": 0,
                    "non_unique_count": 0,
                    "total_dense_reward": 0.0,
                    "window_weights": [],
                },
            )
            aggregate["rollouts_seen"] += int(stats["rollouts_seen"])
            aggregate["accepted_count"] += int(stats["accepted_count"])
            aggregate["training_eligible_count"] += int(stats["training_eligible_count"])
            aggregate["rejected_count"] += int(stats["rejected_count"])
            aggregate["proof_failures"] += int(stats["proof_failures"])
            aggregate["format_failures"] += int(stats["format_failures"])
            aggregate["non_unique_count"] += int(stats["non_unique_count"])
            aggregate["total_dense_reward"] += float(stats["total_dense_reward"])
            aggregate["window_weights"].append(float(stats.get("weight", 0.0)))

    miners = []
    for miner in aggregate_miners.values():
        count = max(miner["rollouts_seen"], 1)
        miner["mean_dense_reward"] = miner["total_dense_reward"] / count
        miner["mean_weight"] = sum(miner["window_weights"]) / max(len(miner["window_weights"]), 1)
        miners.append(miner)

    miners = sorted(miners, key=lambda item: (-item["mean_weight"], item["miner_id"]))
    windows = []
    for record in window_records:
        verdicts = record["verdicts"]
        rejections = [
            {
                "rollout_artifact_id": verdict["payload"]["rollout_artifact_id"],
                "miner_id": verdict["payload"]["rollout_producer_id"],
                "archetype": verdict["payload"].get("rollout_archetype", "unknown"),
                "summary": verdict["payload"]["verdict_explanation"]["summary"],
            }
            for verdict in verdicts
            if not verdict["payload"]["accepted"]
        ]
        windows.append(
            {
                "window_id": record["window_id"],
                "policy_id": record["policy_artifact"]["artifact_id"],
                "challenge_count": len(record["challenge_artifacts"]),
                "rollout_count": len(record["rollout_artifacts"]),
                "accepted_count": record["scorecard_artifact"]["payload"]["window_metrics"]["accepted_count"],
                "training_eligible_count": record["scorecard_artifact"]["payload"]["window_metrics"][
                    "training_eligible_count"
                ],
                "rejected_count": record["scorecard_artifact"]["payload"]["window_metrics"]["rejected_count"],
                "proof_failures": record["scorecard_artifact"]["payload"]["window_metrics"]["proof_failures"],
                "format_failures": record["scorecard_artifact"]["payload"]["window_metrics"][
                    "format_failures"
                ],
                "non_unique_count": record["scorecard_artifact"]["payload"]["window_metrics"][
                    "non_unique_count"
                ],
                "weights": record["scorecard_artifact"]["payload"]["weights"],
                "eval_before": record["training_summary"]["eval_before"],
                "eval_after": record["training_summary"]["eval_after"],
                "checkpoint_id": record["training_summary"]["checkpoint_artifact"]["artifact_id"],
                "next_policy_id": record["training_summary"]["next_policy_artifact"]["artifact_id"],
                "training_batch_id": record["training_summary"]["training_batch_artifact"]["artifact_id"],
                "rejections": rejections[:8],
            }
        )

    return {
        "run": {
            "run_id": run_manifest["payload"]["run_id"],
            "profile": run_manifest["payload"]["profile"],
            "profile_label": profile["label"],
            "description": profile["description"],
            "window_count": len(window_records),
            "miner_count": len(profile["miners"]),
            "chain_event_count": len(chain_events),
            "initial_policy_id": run_manifest["payload"]["initial_policy_id"],
            "final_policy_id": run_manifest["payload"]["final_policy_id"],
            "accepted_count": run_manifest["payload"]["aggregate_metrics"]["accepted_count"],
            "training_eligible_count": run_manifest["payload"]["aggregate_metrics"][
                "training_eligible_count"
            ],
            "rejected_count": run_manifest["payload"]["aggregate_metrics"]["rejected_count"],
            "proof_failures": run_manifest["payload"]["aggregate_metrics"]["proof_failures"],
            "format_failures": run_manifest["payload"]["aggregate_metrics"]["format_failures"],
            "non_unique_count": run_manifest["payload"]["aggregate_metrics"]["non_unique_count"],
        },
        "miners": miners,
        "windows": windows,
        "chain_events": chain_events,
    }


def _format_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _rows_to_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_dashboard_html(bundle: dict[str, Any]) -> str:
    run = bundle["run"]
    miner_rows = [
        [
            escape(miner["label"]),
            escape(str(miner["archetype"])),
            str(miner["rollouts_seen"]),
            str(miner["accepted_count"]),
            str(miner["training_eligible_count"]),
            _format_pct(float(miner["mean_dense_reward"])),
            _format_pct(float(miner["mean_weight"])),
        ]
        for miner in bundle["miners"]
    ]
    window_cards = []
    for window in bundle["windows"]:
        weight_lines = "".join(
            f"<li><strong>{escape(miner_id)}</strong> {_format_pct(float(weight))}</li>"
            for miner_id, weight in sorted(
                window["weights"].items(),
                key=lambda item: (-item[1], item[0]),
            )
        ) or "<li>No weights published</li>"
        rejection_lines = "".join(
            "<li>"
            f"<strong>{escape(item['miner_id'])}</strong> "
            f"<span>{escape(item['summary'])}</span>"
            "</li>"
            for item in window["rejections"]
        ) or "<li>No rejected rollouts</li>"
        window_cards.append(
            """
            <section class="window-card">
              <div class="window-header">
                <h3>Window {window_id:04d}</h3>
                <p>Policy <code>{policy_id}</code></p>
              </div>
              <div class="window-grid">
                <div class="metric"><span>Challenges</span><strong>{challenge_count}</strong></div>
                <div class="metric"><span>Rollouts</span><strong>{rollout_count}</strong></div>
                <div class="metric"><span>Accepted</span><strong>{accepted_count}</strong></div>
                <div class="metric"><span>Training Eligible</span><strong>{training_eligible_count}</strong></div>
                <div class="metric"><span>Rejected</span><strong>{rejected_count}</strong></div>
                <div class="metric"><span>Non-Unique</span><strong>{non_unique_count}</strong></div>
              </div>
              <div class="window-columns">
                <div>
                  <h4>Validator Signals</h4>
                  <ul>
                    <li>Proof failures: <strong>{proof_failures}</strong></li>
                    <li>Format failures: <strong>{format_failures}</strong></li>
                    <li>Checkpoint: <code>{checkpoint_id}</code></li>
                    <li>Next policy: <code>{next_policy_id}</code></li>
                  </ul>
                </div>
                <div>
                  <h4>Weights</h4>
                  <ul>{weight_lines}</ul>
                </div>
                <div>
                  <h4>Training Delta</h4>
                  <ul>
                    <li>Dense reward: <strong>{dense_before:.3f}</strong> → <strong>{dense_after:.3f}</strong></li>
                    <li>Success rate: <strong>{success_before:.3f}</strong> → <strong>{success_after:.3f}</strong></li>
                    <li>Training batch: <code>{training_batch_id}</code></li>
                  </ul>
                </div>
              </div>
              <div>
                <h4>Rejected Rollouts</h4>
                <ul>{rejection_lines}</ul>
              </div>
            </section>
            """.format(
                window_id=window["window_id"],
                policy_id=escape(window["policy_id"]),
                challenge_count=window["challenge_count"],
                rollout_count=window["rollout_count"],
                accepted_count=window["accepted_count"],
                training_eligible_count=window["training_eligible_count"],
                rejected_count=window["rejected_count"],
                non_unique_count=window["non_unique_count"],
                proof_failures=window["proof_failures"],
                format_failures=window["format_failures"],
                checkpoint_id=escape(window["checkpoint_id"]),
                next_policy_id=escape(window["next_policy_id"]),
                dense_before=float(window["eval_before"]["mean_dense_reward"]),
                dense_after=float(window["eval_after"]["mean_dense_reward"]),
                success_before=float(window["eval_before"]["success_rate"]),
                success_after=float(window["eval_after"]["success_rate"]),
                training_batch_id=escape(window["training_batch_id"]),
                weight_lines=weight_lines,
                rejection_lines=rejection_lines,
            )
        )

    chain_rows = [
        [
            str(event["event_index"]),
            escape(event["event_type"]),
            escape(str(event["payload"]).replace("{", "").replace("}", "")),
        ]
        for event in bundle["chain_events"][:20]
    ]
    chain_table = _rows_to_table(["#", "Event", "Payload"], chain_rows)
    miner_table = _rows_to_table(
        [
            "Miner",
            "Archetype",
            "Rollouts",
            "Accepted",
            "Training Eligible",
            "Mean Dense Reward",
            "Mean Weight",
        ],
        miner_rows,
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reliquary Showcase {escape(run['run_id'])}</title>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: rgba(14, 20, 37, 0.92);
      --panel-soft: rgba(23, 31, 56, 0.88);
      --ink: #eef2ff;
      --muted: #b0b9d8;
      --accent: #5eead4;
      --accent-2: #f59e0b;
      --border: rgba(176, 185, 216, 0.18);
      --shadow: 0 24px 60px rgba(0, 0, 0, 0.35);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(94, 234, 212, 0.18), transparent 34%),
        radial-gradient(circle at top right, rgba(245, 158, 11, 0.16), transparent 30%),
        linear-gradient(180deg, #0b1020, #11182d 52%, #0f1425);
    }}
    .page {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 48px 24px 72px;
    }}
    .hero, .panel, .window-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 24px;
      box-shadow: var(--shadow);
    }}
    .hero {{
      padding: 32px;
      margin-bottom: 24px;
    }}
    .hero p {{ color: var(--muted); line-height: 1.6; max-width: 860px; }}
    .stats {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      margin-top: 22px;
    }}
    .stat, .metric {{
      background: var(--panel-soft);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 16px;
    }}
    .stat span, .metric span {{
      display: block;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    .stat strong, .metric strong {{
      font-size: 28px;
      color: var(--accent);
    }}
    .layout {{
      display: grid;
      gap: 24px;
    }}
    .panel {{
      padding: 28px;
    }}
    .panel h2, .window-card h3 {{
      margin-top: 0;
      font-family: "Georgia", serif;
      letter-spacing: 0.01em;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 12px 10px;
      text-align: left;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      text-transform: uppercase;
      font-size: 12px;
      letter-spacing: 0.06em;
    }}
    code {{
      color: var(--accent-2);
      word-break: break-all;
    }}
    .window-card {{
      padding: 28px;
      margin-top: 20px;
    }}
    .window-header {{
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: baseline;
    }}
    .window-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
      margin: 18px 0 20px;
    }}
    .window-columns {{
      display: grid;
      gap: 18px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}
    ul {{
      margin: 10px 0 0;
      padding-left: 18px;
      color: var(--muted);
    }}
    h4 {{
      margin-bottom: 8px;
      color: var(--ink);
    }}
    .timeline {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 10px;
    }}
    .pill {{
      border-radius: 999px;
      border: 1px solid var(--border);
      padding: 8px 12px;
      background: rgba(17, 24, 45, 0.7);
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 720px) {{
      .page {{ padding: 28px 14px 40px; }}
      .hero, .panel, .window-card {{ border-radius: 18px; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <p class="pill">Verifiable coordination demo</p>
      <h1>{escape(run['profile_label'])}</h1>
      <p>{escape(run['description'])}</p>
      <div class="stats">
        <div class="stat"><span>Windows</span><strong>{run['window_count']}</strong></div>
        <div class="stat"><span>Miners</span><strong>{run['miner_count']}</strong></div>
        <div class="stat"><span>Accepted</span><strong>{run['accepted_count']}</strong></div>
        <div class="stat"><span>Training Eligible</span><strong>{run['training_eligible_count']}</strong></div>
        <div class="stat"><span>Rejected</span><strong>{run['rejected_count']}</strong></div>
        <div class="stat"><span>Chain Events</span><strong>{run['chain_event_count']}</strong></div>
      </div>
      <div class="timeline">
        <span class="pill">Initial policy {escape(run['initial_policy_id'][:12])}</span>
        <span class="pill">Final policy {escape(run['final_policy_id'][:12])}</span>
        <span class="pill">Proof failures {run['proof_failures']}</span>
        <span class="pill">Format failures {run['format_failures']}</span>
        <span class="pill">Non-unique {run['non_unique_count']}</span>
      </div>
    </section>
    <section class="panel">
      <h2>Miner Scoreboard</h2>
      {miner_table}
    </section>
    <section class="panel">
      <h2>Chain Event Trace</h2>
      {chain_table}
    </section>
    {''.join(window_cards)}
  </div>
</body>
</html>"""


def render_summary_markdown(bundle: dict[str, Any]) -> str:
    run = bundle["run"]
    lines = [
        f"# Reliquary Showcase Run `{run['run_id']}`",
        "",
        f"- Profile: `{run['profile']}`",
        f"- Windows: `{run['window_count']}`",
        f"- Miners: `{run['miner_count']}`",
        f"- Accepted rollouts: `{run['accepted_count']}`",
        f"- Training-eligible rollouts: `{run['training_eligible_count']}`",
        f"- Rejected rollouts: `{run['rejected_count']}`",
        f"- Proof failures: `{run['proof_failures']}`",
        f"- Format failures: `{run['format_failures']}`",
        f"- Non-unique rollouts: `{run['non_unique_count']}`",
        "",
        "## Top miners",
        "",
    ]
    for miner in bundle["miners"][:5]:
        lines.append(
            "- "
            f"`{miner['label']}` ({miner['archetype']}) "
            f"weight={miner['mean_weight']:.3f} "
            f"dense={miner['mean_dense_reward']:.3f} "
            f"accepted={miner['accepted_count']}"
        )
    return "\n".join(lines) + "\n"


def write_dashboard(run_dir: Path, bundle: dict[str, Any]) -> dict[str, str]:
    dashboard_dir = run_dir / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    data_path = dashboard_dir / "dashboard-data.json"
    html_path = dashboard_dir / "index.html"
    summary_path = run_dir / "SUMMARY.md"

    write_json(data_path, bundle)
    html_path.write_text(render_dashboard_html(bundle), encoding="utf-8")
    summary_path.write_text(render_summary_markdown(bundle), encoding="utf-8")

    return {
        "dashboard_dir": str(dashboard_dir),
        "dashboard_html": str(html_path),
        "dashboard_data": str(data_path),
        "summary_markdown": str(summary_path),
    }
