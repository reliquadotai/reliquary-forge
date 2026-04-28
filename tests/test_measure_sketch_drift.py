"""Tests for the sketch-drift measurement helper.

Phase 1.1 of the parallel-harvest plan. The drift measurement itself
needs two HF revisions + GPU; tests here cover the pure-python pieces
(quantile, summary, CSV writer).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import measure_sketch_drift as msd  # noqa: E402


def test_quantile_basic() -> None:
    assert msd.quantile([1.0], 0.5) == 1.0
    assert msd.quantile([1.0, 2.0, 3.0], 0.5) == 2.0


def test_summarize_drift_empty_returns_zeroes() -> None:
    s = msd.summarize_drift([])
    assert s == {"n": 0, "max": 0.0, "p99": 0.0, "p95": 0.0, "median": 0.0}


def test_summarize_drift_uses_absolute_value() -> None:
    rows = [
        msd.DriftRow(0, 0, 100, 80, -20),
        msd.DriftRow(0, 1, 80, 100, 20),
        msd.DriftRow(0, 2, 50, 50, 0),
    ]
    s = msd.summarize_drift(rows)
    assert s["n"] == 3
    assert s["max"] == 20.0
    assert s["median"] == 20.0


def test_write_drift_csv_round_trip(tmp_path: Path) -> None:
    rows = [
        msd.DriftRow(0, 5, 1000, 800, 200),
        msd.DriftRow(1, 7, 1500, 1700, -200),
    ]
    out = tmp_path / "drift.csv"
    msd.write_drift_csv(rows, out)
    with out.open() as f:
        parsed = list(csv.DictReader(f))
    assert len(parsed) == 2
    assert parsed[0]["prompt_idx"] == "0"
    assert parsed[0]["sketch_a"] == "1000"
    assert parsed[1]["diff"] == "-200"
