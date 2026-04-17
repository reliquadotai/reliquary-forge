"""Tests for the reliquary env submission CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("RestrictedPython", reason="RestrictedPython required")

from reliquary.envs.cli import main
from reliquary.envs.registry import EnvBundle
from reliquary.envs.spec import EnvMetadata, EnvSpec, canonical_env_spec_bytes


_EVALUATOR = (
    "def evaluate(task, solution):\n"
    "    target = task['answer']\n"
    "    guess = solution['guess']\n"
    "    return 1.0 if guess == target else 0.0\n"
)
_TASK_GEN = "def generate(seed):\n    return {'answer': seed * 2}\n"
_REWARD = "def reward(solution, target):\n    return 1.0 if solution == target else 0.0\n"


def _write_bundle_files(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "task_generator": tmp_path / "task_gen.py",
        "reward_function": tmp_path / "reward.py",
        "evaluator": tmp_path / "evaluator.py",
    }
    paths["task_generator"].write_text(_TASK_GEN)
    paths["reward_function"].write_text(_REWARD)
    paths["evaluator"].write_text(_EVALUATOR)
    return paths


def _spec_for_bundle(
    bundle: EnvBundle, *, expected_reward: float = 1.0
) -> EnvSpec:
    hashes = bundle.code_hashes()
    return EnvSpec(
        name="cli_test_env",
        version="1.0.0",
        author_hotkey="A" * 48,
        author_stake_lock_tao=10.0,
        task_generator_hash=hashes["task_generator_hash"],
        reward_function_hash=hashes["reward_function_hash"],
        evaluator_hash=hashes["evaluator_hash"],
        sample_task='{"answer": 42}',
        sample_solution='{"guess": 42}',
        expected_reward=expected_reward,
        metadata=EnvMetadata(
            context_length=2048,
            timeout_ms=5_000,
            difficulty="easy",
            domain="math",
        ),
    )


def _write_spec(tmp_path: Path, spec: EnvSpec) -> Path:
    # Emit canonical JSON (round-trips through EnvSpec.from_dict on load).
    path = tmp_path / "spec.json"
    path.write_bytes(canonical_env_spec_bytes(spec))
    return path


def test_cli_submit_happy_path(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    files = _write_bundle_files(tmp_path)
    bundle = EnvBundle(
        task_generator_source=_TASK_GEN,
        reward_function_source=_REWARD,
        evaluator_source=_EVALUATOR,
    )
    spec_path = _write_spec(tmp_path, _spec_for_bundle(bundle))

    rc = main(
        [
            "submit",
            "--spec",
            str(spec_path),
            "--task-generator",
            str(files["task_generator"]),
            "--reward-function",
            str(files["reward_function"]),
            "--evaluator",
            str(files["evaluator"]),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["valid"] is True
    assert parsed["observed_reward"] == pytest.approx(1.0)


def test_cli_submit_register_emits_env_id(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    files = _write_bundle_files(tmp_path)
    bundle = EnvBundle(
        task_generator_source=_TASK_GEN,
        reward_function_source=_REWARD,
        evaluator_source=_EVALUATOR,
    )
    spec_path = _write_spec(tmp_path, _spec_for_bundle(bundle))

    rc = main(
        [
            "submit",
            "--spec",
            str(spec_path),
            "--task-generator",
            str(files["task_generator"]),
            "--reward-function",
            str(files["reward_function"]),
            "--evaluator",
            str(files["evaluator"]),
            "--register",
        ]
    )
    assert rc == 0
    out_lines = capsys.readouterr().out.splitlines()
    # First line is the env_spec_id; remaining lines are the JSON report.
    env_id = out_lines[0]
    assert len(env_id) == 64
    assert all(ch in "0123456789abcdef" for ch in env_id)


def test_cli_submit_reward_mismatch_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    files = _write_bundle_files(tmp_path)
    bundle = EnvBundle(
        task_generator_source=_TASK_GEN,
        reward_function_source=_REWARD,
        evaluator_source=_EVALUATOR,
    )
    spec_path = _write_spec(
        tmp_path, _spec_for_bundle(bundle, expected_reward=0.5)
    )

    rc = main(
        [
            "submit",
            "--spec",
            str(spec_path),
            "--task-generator",
            str(files["task_generator"]),
            "--reward-function",
            str(files["reward_function"]),
            "--evaluator",
            str(files["evaluator"]),
        ]
    )
    assert rc == 1


def test_cli_validate_only_no_register(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    files = _write_bundle_files(tmp_path)
    bundle = EnvBundle(
        task_generator_source=_TASK_GEN,
        reward_function_source=_REWARD,
        evaluator_source=_EVALUATOR,
    )
    spec_path = _write_spec(tmp_path, _spec_for_bundle(bundle))

    rc = main(
        [
            "validate",
            "--spec",
            str(spec_path),
            "--task-generator",
            str(files["task_generator"]),
            "--reward-function",
            str(files["reward_function"]),
            "--evaluator",
            str(files["evaluator"]),
        ]
    )
    assert rc == 0


def test_cli_missing_file_returns_two(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    rc = main(
        [
            "submit",
            "--spec",
            str(tmp_path / "missing.json"),
            "--task-generator",
            str(tmp_path / "missing_gen.py"),
            "--reward-function",
            str(tmp_path / "missing_reward.py"),
            "--evaluator",
            str(tmp_path / "missing_eval.py"),
        ]
    )
    assert rc == 2


def test_cli_malicious_source_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    files = _write_bundle_files(tmp_path)
    # Inject a blocked import.
    files["evaluator"].write_text("import os\n" + _EVALUATOR)
    bundle = EnvBundle(
        task_generator_source=_TASK_GEN,
        reward_function_source=_REWARD,
        evaluator_source=files["evaluator"].read_text(),
    )
    spec_path = _write_spec(tmp_path, _spec_for_bundle(bundle))

    rc = main(
        [
            "submit",
            "--spec",
            str(spec_path),
            "--task-generator",
            str(files["task_generator"]),
            "--reward-function",
            str(files["reward_function"]),
            "--evaluator",
            str(files["evaluator"]),
        ]
    )
    assert rc == 1


def test_cli_output_file_written(tmp_path: Path) -> None:
    files = _write_bundle_files(tmp_path)
    bundle = EnvBundle(
        task_generator_source=_TASK_GEN,
        reward_function_source=_REWARD,
        evaluator_source=_EVALUATOR,
    )
    spec_path = _write_spec(tmp_path, _spec_for_bundle(bundle))
    output_path = tmp_path / "report.json"

    rc = main(
        [
            "submit",
            "--spec",
            str(spec_path),
            "--task-generator",
            str(files["task_generator"]),
            "--reward-function",
            str(files["reward_function"]),
            "--evaluator",
            str(files["evaluator"]),
            "--output",
            str(output_path),
        ]
    )
    assert rc == 0
    assert output_path.is_file()
    data = json.loads(output_path.read_text())
    assert data["valid"] is True
