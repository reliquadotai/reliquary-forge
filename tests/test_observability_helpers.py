"""Unit tests for reliquary.observability helpers.

Pins the Prometheus exposition format + registry-backed aggregation
helpers independent of the full ``build_metrics_text`` path (which
requires a populated runtime-status snapshot).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reliquary.observability import (
    _decision_counts,
    _latest_artifact,
    _latest_rl_training_batch,
    _metric_lines,
    _recent_scorecard_totals,
)
from reliquary.protocol.artifacts import make_artifact
from reliquary.storage.local_registry import LocalArtifactRegistry

# ---------------------------------------------------------------------------
# _metric_lines
# ---------------------------------------------------------------------------


def test_metric_lines_no_labels() -> None:
    lines = _metric_lines("foo", 1.5, help_text="A test metric.")
    assert lines[0] == "# HELP foo A test metric."
    assert lines[1] == "# TYPE foo gauge"
    assert lines[2] == "foo 1.5"


def test_metric_lines_with_labels_sorts_by_key() -> None:
    lines = _metric_lines(
        "foo",
        2.0,
        help_text="Labeled",
        labels={"zeta": "z", "alpha": "a", "mu": "m"},
    )
    # Label key order must be alpha, mu, zeta.
    assert lines[2] == 'foo{alpha="a",mu="m",zeta="z"} 2.0'


def test_metric_lines_escapes_quotes_and_backslashes() -> None:
    lines = _metric_lines(
        "foo",
        3.0,
        help_text="Escapes",
        labels={"msg": 'needs"escaping\\'},
    )
    assert 'msg="needs\\"escaping\\\\"' in lines[2]


def test_metric_lines_empty_labels_dict_omits_braces() -> None:
    lines = _metric_lines("foo", 4.0, help_text="Empty labels", labels={})
    assert lines[2] == "foo 4.0"
    assert "{" not in lines[2]


def test_metric_lines_stringifies_non_string_values() -> None:
    lines = _metric_lines(
        "foo",
        5.0,
        help_text="Coerce",
        labels={"count": 42, "flag": True},
    )
    assert 'count="42"' in lines[2]
    assert 'flag="True"' in lines[2]


def test_metric_lines_zero_value_emitted() -> None:
    lines = _metric_lines("foo", 0.0, help_text="Zero")
    assert lines[2] == "foo 0.0"


def test_metric_lines_negative_value_emitted() -> None:
    lines = _metric_lines("foo", -1.5, help_text="Negative")
    assert lines[2] == "foo -1.5"


def test_metric_lines_returns_three_entries() -> None:
    lines = _metric_lines("foo", 1.0, help_text="Count")
    assert len(lines) == 3


# ---------------------------------------------------------------------------
# _latest_artifact (with LocalArtifactRegistry)
# ---------------------------------------------------------------------------


def _make_policy_like(window_id: int, created_at: str = "2026-04-18T00:00:00+00:00") -> dict:
    return make_artifact(
        artifact_type="policy",
        producer_id="trainer-1",
        policy_id=f"policy-window-{window_id}",
        window_id=window_id,
        payload={"model_ref": f"model-v{window_id}"},
        created_at=created_at,
    )


def test_latest_artifact_none_when_registry_empty(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    assert _latest_artifact(registry, "policy") is None


def test_latest_artifact_returns_highest_window(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    for window in (1, 5, 3):
        registry.put_artifact(_make_policy_like(window))
    latest = _latest_artifact(registry, "policy")
    assert latest is not None
    assert latest["window_id"] == 5


def test_latest_artifact_breaks_tie_by_created_at(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    registry.put_artifact(_make_policy_like(7, created_at="2026-04-18T00:00:00+00:00"))
    registry.put_artifact(_make_policy_like(7, created_at="2026-04-18T01:00:00+00:00"))
    latest = _latest_artifact(registry, "policy")
    assert latest is not None
    assert latest["created_at"] == "2026-04-18T01:00:00+00:00"


# ---------------------------------------------------------------------------
# _recent_scorecard_totals
# ---------------------------------------------------------------------------


def _scorecard(window_id: int, **totals: Any) -> dict:
    return make_artifact(
        artifact_type="scorecard",
        producer_id="validator-1",
        policy_id="policy-1",
        window_id=window_id,
        payload={
            "window_metrics": {
                "accepted_count": totals.get("accepted", 0),
                "training_eligible_count": totals.get("training_eligible", 0),
                "contamination_failures": totals.get("contamination_failures", 0),
                "format_failures": totals.get("format_failures", 0),
                "proof_failures": totals.get("proof_failures", 0),
            }
        },
    )


def test_recent_scorecard_totals_empty_registry(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    totals = _recent_scorecard_totals(registry)
    assert totals == {
        "accepted": 0.0,
        "training_eligible": 0.0,
        "contamination_failures": 0.0,
        "format_failures": 0.0,
        "proof_failures": 0.0,
    }


def test_recent_scorecard_totals_sums_latest_five(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    for window in range(10):
        registry.put_artifact(_scorecard(window, accepted=window, training_eligible=window * 2))
    totals = _recent_scorecard_totals(registry)
    # Latest 5 windows: 5,6,7,8,9 → accepted = 5+6+7+8+9 = 35
    assert totals["accepted"] == 35.0
    assert totals["training_eligible"] == 70.0


def test_recent_scorecard_totals_respects_custom_limit(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    for window in range(10):
        registry.put_artifact(_scorecard(window, accepted=1))
    totals = _recent_scorecard_totals(registry, limit=3)
    assert totals["accepted"] == 3.0


def test_recent_scorecard_totals_counts_all_failure_types(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    registry.put_artifact(
        _scorecard(
            1,
            contamination_failures=1,
            format_failures=2,
            proof_failures=3,
        )
    )
    totals = _recent_scorecard_totals(registry)
    assert totals["contamination_failures"] == 1.0
    assert totals["format_failures"] == 2.0
    assert totals["proof_failures"] == 3.0


# ---------------------------------------------------------------------------
# _latest_rl_training_batch
# ---------------------------------------------------------------------------


def _training_batch(*, window_id: int, lane: str = "rl") -> dict:
    artifact = make_artifact(
        artifact_type="training_batch",
        producer_id="trainer-1",
        policy_id="policy-1",
        window_id=window_id,
        payload={"source_pack_id": f"pack-{window_id}"},
        lane=lane,
    )
    return artifact


def test_latest_rl_training_batch_none_when_empty(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    assert _latest_rl_training_batch(registry) is None


def test_latest_rl_training_batch_filters_by_lane(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    registry.put_artifact(_training_batch(window_id=3, lane="distill"))
    registry.put_artifact(_training_batch(window_id=5, lane="distill"))
    assert _latest_rl_training_batch(registry) is None


def test_latest_rl_training_batch_picks_highest_rl_window(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    registry.put_artifact(_training_batch(window_id=2, lane="rl"))
    registry.put_artifact(_training_batch(window_id=5, lane="rl"))
    registry.put_artifact(_training_batch(window_id=3, lane="distill"))
    latest = _latest_rl_training_batch(registry)
    assert latest is not None
    assert latest["window_id"] == 5
    assert latest["lane"] == "rl"


# ---------------------------------------------------------------------------
# _decision_counts
# ---------------------------------------------------------------------------


def _decision(approved: bool, window_id: int = 1) -> dict:
    return make_artifact(
        artifact_type="policy_authority_decision",
        producer_id="trainer-1",
        policy_id="policy-1",
        window_id=window_id,
        payload={"approved": approved, "rationale": "x"},
    )


def test_decision_counts_empty(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    approvals, rejections = _decision_counts(registry)
    assert approvals == 0.0
    assert rejections == 0.0


def test_decision_counts_mix(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    for i in range(3):
        registry.put_artifact(_decision(approved=True, window_id=i))
    for i in range(2):
        registry.put_artifact(_decision(approved=False, window_id=i + 10))
    approvals, rejections = _decision_counts(registry)
    assert approvals == 3.0
    assert rejections == 2.0


def test_decision_counts_missing_payload_approved_is_false(tmp_path: Path) -> None:
    registry = LocalArtifactRegistry(str(tmp_path))
    artifact = make_artifact(
        artifact_type="policy_authority_decision",
        producer_id="trainer-1",
        policy_id="policy-1",
        window_id=1,
        payload={},  # no "approved" key
    )
    registry.put_artifact(artifact)
    approvals, rejections = _decision_counts(registry)
    assert approvals == 0.0
    assert rejections == 1.0
