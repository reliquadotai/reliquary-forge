"""Tests for the cheater-curve threshold sweep harness.

Phase 1.1 of the parallel-harvest plan. The full sweep is gated on
GPU + transformers + reliquary-forge GRPO trainer (production-only).
These tests cover the harness logic that *doesn't* require those:
the CSV writer, the per-step aggregator, the quantile helper, and the
summary table renderer. That's what catches typos + schema drift
before an operator commits to a multi-hour fleet sweep.
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

# Make scripts/ importable so we can pull in the helpers under test.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import cheater_curve_threshold as cct  # noqa: E402


# ────────  quantile  ────────


def test_quantile_empty_returns_nan() -> None:
    import math

    assert math.isnan(cct.quantile([], 0.5))


def test_quantile_single_value_returns_that_value() -> None:
    assert cct.quantile([7.0], 0.0) == 7.0
    assert cct.quantile([7.0], 0.5) == 7.0
    assert cct.quantile([7.0], 1.0) == 7.0


def test_quantile_p50_is_median() -> None:
    assert cct.quantile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0


def test_quantile_p95_p99_match_expected() -> None:
    values = [float(i) for i in range(1, 101)]  # 1..100
    # Linear interpolation: pos = q * (n-1) = 0.95 * 99 = 94.05.
    # values[94]=95.0, values[95]=96.0; result = 95.0 + 0.05 * 1.0 = 95.05.
    assert abs(cct.quantile(values, 0.95) - 95.05) < 1e-9
    # pos = 0.99 * 99 = 98.01; values[98]=99.0, values[99]=100.0; result = 99.01.
    assert abs(cct.quantile(values, 0.99) - 99.01) < 1e-9


def test_quantile_p0_is_min_p100_is_max() -> None:
    values = [3.0, 1.0, 2.0]
    assert cct.quantile(values, 0.0) == 1.0
    assert cct.quantile(values, 1.0) == 3.0


# ────────  Measurement + write_measurements_csv  ────────


def test_write_measurements_csv_round_trips(tmp_path: Path) -> None:
    measurements = [
        cct.Measurement(
            run=0,
            step=0,
            sketch_diff_max=1234.5,
            lp_dev_max=0.001,
            dist_median=0.42,
            dist_q10=0.1,
            completion_length=64,
        ),
        cct.Measurement(
            run=0,
            step=1,
            sketch_diff_max=1500.0,
            lp_dev_max=0.005,
            dist_median=0.40,
            dist_q10=0.05,
            completion_length=70,
        ),
    ]
    out = tmp_path / "x.csv"
    cct.write_measurements_csv(measurements, out)
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["sketch_diff_max"] == "1234.5"
    assert rows[0]["completion_length"] == "64"
    assert rows[1]["lp_dev_max"] == "0.005"


def test_write_measurements_csv_handles_empty(tmp_path: Path) -> None:
    out = tmp_path / "empty.csv"
    cct.write_measurements_csv([], out)
    text = out.read_text()
    # Only the header line is present.
    assert "run,step,sketch_diff_max" in text
    assert text.count("\n") == 1


# ────────  aggregate_per_step  ────────


def _measurements_for_step(step: int, sketches: list[float]) -> list[cct.Measurement]:
    return [
        cct.Measurement(
            run=i,
            step=step,
            sketch_diff_max=s,
            lp_dev_max=0.0,
            dist_median=1.0,
            dist_q10=1.0,
            completion_length=32,
        )
        for i, s in enumerate(sketches)
    ]


def test_aggregate_per_step_groups_by_step() -> None:
    measurements = (
        _measurements_for_step(0, [100.0, 200.0])
        + _measurements_for_step(1, [300.0, 400.0])
    )
    rows = cct.aggregate_per_step(measurements)
    assert [r["step"] for r in rows] == [0, 1]
    assert rows[0]["n"] == 2
    assert rows[1]["sketch_diff_max_max"] == 400.0


def test_aggregate_per_step_p95_uses_quantile() -> None:
    measurements = _measurements_for_step(
        0, [float(i) for i in range(1, 101)]  # 1..100
    )
    rows = cct.aggregate_per_step(measurements)
    assert rows[0]["sketch_diff_max_max"] == 100.0
    assert abs(rows[0]["sketch_diff_max_p95"] - 95.05) < 1e-9
    assert abs(rows[0]["sketch_diff_max_p99"] - 99.01) < 1e-9


def test_aggregate_per_step_handles_empty_input() -> None:
    assert cct.aggregate_per_step([]) == []


def test_aggregate_per_step_steps_emitted_in_sorted_order() -> None:
    measurements = (
        _measurements_for_step(5, [50.0])
        + _measurements_for_step(2, [20.0])
        + _measurements_for_step(7, [70.0])
    )
    rows = cct.aggregate_per_step(measurements)
    assert [r["step"] for r in rows] == [2, 5, 7]


# ────────  print_summary_table  ────────


def test_print_summary_table_renders_header_and_row() -> None:
    rows = cct.aggregate_per_step(_measurements_for_step(0, [1500.0, 2000.0]))
    buf = io.StringIO()
    cct.print_summary_table(rows, file=buf)
    text = buf.getvalue()
    assert "step" in text
    assert "sketch_p95" in text
    # Step 0 row has n=2.
    assert "    0    2" in text or "0    2" in text


# ────────  CLI: smoke (no --real-grpo) is a no-op that writes empty CSV  ────────


def test_main_without_real_grpo_writes_empty_csv(tmp_path: Path) -> None:
    out = tmp_path / "smoke.csv"
    rc = cct.main(
        [
            "--n-runs",
            "1",
            "--n-steps",
            "1",
            "--measurements-per-step",
            "1",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    text = out.read_text()
    # Only header.
    assert text.startswith("run,step,sketch_diff_max")
    assert text.count("\n") == 1
