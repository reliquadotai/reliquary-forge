from pathlib import Path

from reliquary.protocol.artifacts import make_artifact
from reliquary.showcase.dashboard import build_dashboard_bundle, write_dashboard
from reliquary.showcase.profiles import get_demo_profile, miner_secret_map


def test_showcase_profile_contains_expected_archetypes() -> None:
    profile = get_demo_profile("showcase")
    archetypes = {miner["archetype"] for miner in profile["miners"]}
    assert "honest_high_quality" in archetypes
    assert "low_diversity" in archetypes
    assert "malformed_proof_failing" in archetypes
    assert miner_secret_map(profile)["atlas-honest"] == "atlas-secret"


def test_dashboard_export_writes_static_assets(tmp_path: Path) -> None:
    run_manifest = make_artifact(
        artifact_type="run_manifest",
        producer_id="trainer",
        policy_id="policy-final",
        window_id=1,
        payload={
            "run_id": "showcase-test",
            "profile": "showcase",
            "profile_label": "Reliquary Showcase",
            "description": "Test run",
            "initial_policy_id": "policy-start",
            "final_policy_id": "policy-final",
            "window_manifest_ids": ["window-0"],
            "aggregate_metrics": {
                "accepted_count": 3,
                "training_eligible_count": 2,
                "rejected_count": 1,
                "proof_failures": 1,
                "format_failures": 1,
                "non_unique_count": 1,
            },
            "output_assets": {},
        },
    )
    scorecard = make_artifact(
        artifact_type="scorecard",
        producer_id="validator",
        policy_id="policy-start",
        window_id=0,
        payload={
            "totals": {"atlas-honest": 2.0},
            "weights": {"atlas-honest": 1.0},
            "miners": {
                "atlas-honest": {
                    "rollouts_seen": 2,
                    "accepted_count": 2,
                    "training_eligible_count": 2,
                    "rejected_count": 0,
                    "proof_failures": 0,
                    "format_failures": 0,
                    "non_unique_count": 0,
                    "total_dense_reward": 2.0,
                    "total_score": 2.0,
                    "archetype": "honest_high_quality",
                    "weight": 1.0,
                }
            },
            "window_metrics": {
                "rollouts_seen": 2,
                "accepted_count": 2,
                "training_eligible_count": 2,
                "rejected_count": 0,
                "proof_failures": 0,
                "format_failures": 0,
                "non_unique_count": 0,
            },
        },
    )
    window_record = {
        "window_id": 0,
        "policy_artifact": {"artifact_id": "policy-start"},
        "challenge_artifacts": [{"artifact_id": "challenge-1"}],
        "rollout_artifacts": [{"artifact_id": "rollout-1"}, {"artifact_id": "rollout-2"}],
        "verdicts": [],
        "scorecard_artifact": scorecard,
        "training_summary": {
            "eval_before": {"mean_dense_reward": 0.4, "success_rate": 0.1},
            "eval_after": {"mean_dense_reward": 0.7, "success_rate": 0.2},
            "training_batch_artifact": {"artifact_id": "training-batch-1"},
            "checkpoint_artifact": {"artifact_id": "checkpoint-1"},
            "next_policy_artifact": {"artifact_id": "policy-final"},
        },
    }
    profile = get_demo_profile("recording")
    bundle = build_dashboard_bundle(run_manifest, [window_record], [{"event_index": 0, "event_type": "policy_reference", "payload": {"policy_id": "policy-start"}}], profile)
    paths = write_dashboard(tmp_path, bundle)
    assert Path(paths["dashboard_html"]).exists()
    assert Path(paths["dashboard_data"]).exists()
    assert Path(paths["summary_markdown"]).read_text(encoding="utf-8").startswith("# Reliquary Showcase Run")
