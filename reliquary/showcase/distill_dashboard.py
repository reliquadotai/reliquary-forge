from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from ..utils.json_io import write_json


def build_distill_dashboard_bundle(
    run_manifest: dict[str, Any],
    window_records: list[dict[str, Any]],
    chain_events: list[dict[str, Any]],
    profile: dict[str, Any],
) -> dict[str, Any]:
    miners = []
    for record in window_records:
        for miner_id, stats in record["scorecard_artifact"]["payload"]["miners"].items():
            miners.append(
                {
                    "miner_id": miner_id,
                    "archetype": stats.get("archetype", "unknown"),
                    "accepted_count": stats["accepted_count"],
                    "training_eligible_count": stats["training_eligible_count"],
                    "mean_weight_score": stats.get("mean_weight_score", 0.0),
                    "weight": stats.get("weight", 0.0),
                }
            )
    return {
        "run": run_manifest["payload"],
        "windows": [
            {
                "window_id": record["window_id"],
                "task_count": len(record["task_batch_artifact"]["payload"]["tasks"]),
                "trace_count": len(record["trace_artifacts"]),
                "accepted_count": record["scorecard_artifact"]["payload"]["window_metrics"]["accepted_count"],
                "training_eligible_count": record["scorecard_artifact"]["payload"]["window_metrics"][
                    "training_eligible_count"
                ],
                "contamination_failures": record["scorecard_artifact"]["payload"]["window_metrics"][
                    "contamination_failures"
                ],
                "pack_id": record["pack_artifact"]["artifact_id"],
                "checkpoint_id": record["training_summary"]["checkpoint_artifact"]["artifact_id"],
                "next_policy_id": record["training_summary"]["next_policy_artifact"]["artifact_id"],
                "eval_before": record["training_summary"]["eval_before"],
                "eval_after": record["training_summary"]["eval_after"],
                "decision": record["training_summary"]["decision_artifact"]["payload"]["rationale"],
                "rejections": [
                    verdict["payload"]["verdict_explanation"]["summary"]
                    for verdict in record["verdicts"]
                    if not verdict["payload"]["accepted"]
                ][:8],
            }
            for record in window_records
        ],
        "miners": miners,
        "chain_events": chain_events,
        "profile": profile,
    }


def render_distill_dashboard_html(bundle: dict[str, Any]) -> str:
    run = bundle["run"]
    miner_rows = "".join(
        """
        <tr>
          <td>{miner_id}</td>
          <td>{archetype}</td>
          <td>{accepted_count}</td>
          <td>{training_eligible_count}</td>
          <td>{mean_weight_score:.3f}</td>
          <td>{weight:.3f}</td>
        </tr>
        """.format(**miner)
        for miner in bundle["miners"]
    )
    window_sections = "".join(
        """
        <section class="window-card">
          <h3>Window {window_id:04d}</h3>
          <ul>
            <li>Tasks: <strong>{task_count}</strong></li>
            <li>Traces: <strong>{trace_count}</strong></li>
            <li>Accepted: <strong>{accepted_count}</strong></li>
            <li>Training eligible: <strong>{training_eligible_count}</strong></li>
            <li>Contamination failures: <strong>{contamination_failures}</strong></li>
            <li>Pack: <code>{pack_id}</code></li>
            <li>Checkpoint: <code>{checkpoint_id}</code></li>
            <li>Next policy: <code>{next_policy_id}</code></li>
            <li>Policy authority: <strong>{decision}</strong></li>
            <li>Holdout success: <strong>{before:.3f}</strong> → <strong>{after:.3f}</strong></li>
          </ul>
        </section>
        """.format(
            window_id=window["window_id"],
            task_count=window["task_count"],
            trace_count=window["trace_count"],
            accepted_count=window["accepted_count"],
            training_eligible_count=window["training_eligible_count"],
            contamination_failures=window["contamination_failures"],
            pack_id=escape(window["pack_id"]),
            checkpoint_id=escape(window["checkpoint_id"]),
            next_policy_id=escape(window["next_policy_id"]),
            decision=escape(window["decision"]),
            before=float(window["eval_before"]["success_rate"]),
            after=float(window["eval_after"]["success_rate"]),
        )
        for window in bundle["windows"]
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reliquary Distill {escape(run['run_id'])}</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 0; background: linear-gradient(160deg, #f5efe1, #e2e0d6); color: #1b1e18; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 64px; }}
    .hero {{ background: rgba(255,255,255,0.82); border-radius: 20px; padding: 24px; box-shadow: 0 20px 60px rgba(0,0,0,0.08); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin: 24px 0; }}
    .metric, .window-card {{ background: rgba(255,255,255,0.84); border-radius: 18px; padding: 18px; box-shadow: 0 12px 36px rgba(0,0,0,0.06); }}
    table {{ width: 100%; border-collapse: collapse; background: rgba(255,255,255,0.84); border-radius: 18px; overflow: hidden; }}
    th, td {{ padding: 12px 14px; border-bottom: 1px solid rgba(0,0,0,0.08); text-align: left; }}
    code {{ font-size: 0.9em; }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Reliquary Distillation Run</h1>
      <p>{escape(run['description'])}</p>
      <div class="grid">
        <div class="metric"><span>Run ID</span><br><strong>{escape(run['run_id'])}</strong></div>
        <div class="metric"><span>Profile</span><br><strong>{escape(run['profile_label'])}</strong></div>
        <div class="metric"><span>Initial policy</span><br><strong><code>{escape(run['initial_policy_id'])}</code></strong></div>
        <div class="metric"><span>Final policy</span><br><strong><code>{escape(run['final_policy_id'])}</code></strong></div>
      </div>
    </section>
    <section>
      <h2>Miner Outcomes</h2>
      <table>
        <thead>
          <tr><th>Miner</th><th>Archetype</th><th>Accepted</th><th>Training Eligible</th><th>Mean Score</th><th>Weight</th></tr>
        </thead>
        <tbody>{miner_rows}</tbody>
      </table>
    </section>
    <section>
      <h2>Window Timeline</h2>
      {window_sections}
    </section>
  </main>
</body>
</html>"""


def write_distill_dashboard(destination: Path, bundle: dict[str, Any]) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    dashboard_data = destination / "dashboard-data.json"
    dashboard_html = destination / "index.html"
    summary_markdown = destination / "SUMMARY.md"
    write_json(dashboard_data, bundle)
    dashboard_html.write_text(render_distill_dashboard_html(bundle), encoding="utf-8")
    summary_markdown.write_text(
        "# Reliquary Distillation Run\n\n"
        f"- Run ID: `{bundle['run']['run_id']}`\n"
        f"- Initial policy: `{bundle['run']['initial_policy_id']}`\n"
        f"- Final policy: `{bundle['run']['final_policy_id']}`\n",
        encoding="utf-8",
    )
    return {
        "dashboard_html": str(dashboard_html),
        "dashboard_data": str(dashboard_data),
        "summary_markdown": str(summary_markdown),
    }
