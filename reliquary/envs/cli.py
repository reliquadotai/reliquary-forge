"""CLI entry point for env spec submission + validation.

Usage:
    reliquary env submit --spec path/to/spec.json \
                         --task-generator path/to/gen.py \
                         --reward-function path/to/reward.py \
                         --evaluator path/to/eval.py

Exit codes:
    0 — submission validated (bundle hashes match spec, sample reward
        matches expected within tolerance, sandbox audit passes).
    1 — validation failure (hash mismatch, reward mismatch, sandbox
        rejection, or spec schema violation).
    2 — CLI misuse (missing files, bad args).

The CLI ONLY performs the validator-side pre-quorum check. It does not
submit onchain or record stake — those flows land as separate adapters
in the onchain-commit follow-up.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .registry import (
    BundleRejected,
    EnvBundle,
    EnvRegistry,
    register_submission,
    validate_bundle_against_spec,
)
from .spec import EnvSpec, EnvSpecInvalid


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_spec(path: Path) -> EnvSpec:
    payload = json.loads(_read_text(path))
    return EnvSpec.from_dict(payload)


def _load_bundle(
    *,
    task_generator_path: Path,
    reward_function_path: Path,
    evaluator_path: Path,
    task_generator_entry: str,
    reward_function_entry: str,
    evaluator_entry: str,
) -> EnvBundle:
    return EnvBundle(
        task_generator_source=_read_text(task_generator_path),
        reward_function_source=_read_text(reward_function_path),
        evaluator_source=_read_text(evaluator_path),
        task_generator_entry=task_generator_entry,
        reward_function_entry=reward_function_entry,
        evaluator_entry=evaluator_entry,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reliquary-env",
        description="Permissionless environment spec submission + validation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser(
        "submit",
        help="Validate a spec + bundle; optionally record in a local registry.",
    )
    submit.add_argument("--spec", type=Path, required=True, help="path to spec.json")
    submit.add_argument(
        "--task-generator",
        type=Path,
        required=True,
        help="path to task_generator source .py",
    )
    submit.add_argument(
        "--reward-function",
        type=Path,
        required=True,
        help="path to reward_function source .py",
    )
    submit.add_argument(
        "--evaluator",
        type=Path,
        required=True,
        help="path to evaluator source .py",
    )
    submit.add_argument("--task-generator-entry", type=str, default="generate")
    submit.add_argument("--reward-function-entry", type=str, default="reward")
    submit.add_argument("--evaluator-entry", type=str, default="evaluate")
    submit.add_argument(
        "--register",
        action="store_true",
        help="record the validated submission in an in-memory registry and "
        "emit the env_spec_id on stdout",
    )
    submit.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the structured validation report as JSON to this path",
    )

    validate = subparsers.add_parser(
        "validate",
        help="Validate a spec + bundle WITHOUT side effects.",
    )
    validate.add_argument("--spec", type=Path, required=True)
    validate.add_argument("--task-generator", type=Path, required=True)
    validate.add_argument("--reward-function", type=Path, required=True)
    validate.add_argument("--evaluator", type=Path, required=True)
    validate.add_argument("--task-generator-entry", type=str, default="generate")
    validate.add_argument("--reward-function-entry", type=str, default="reward")
    validate.add_argument("--evaluator-entry", type=str, default="evaluate")
    validate.add_argument("--output", type=Path, default=None)

    return parser


def _report_as_json(report) -> dict:  # type: ignore[no-untyped-def]
    return {
        "valid": report.valid,
        "bundle_hashes": report.bundle_hashes,
        "observed_reward": report.observed_reward,
        "expected_reward": report.expected_reward,
        "errors": list(report.errors),
        "sandbox_results": {
            label: {
                "ok": result.ok,
                "value": result.value,
                "timeout": result.timeout,
                "oom": result.oom,
                "error_type": result.error_type,
                "error_message": result.error_message,
                "duration_ms": result.duration_ms,
                "stdout_bytes": len(result.stdout),
            }
            for label, result in report.sandbox_results.items()
        },
    }


def _emit_report(report, output_path: Path | None) -> None:  # type: ignore[no-untyped-def]
    payload = _report_as_json(report)
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if output_path is not None:
        output_path.write_text(serialized, encoding="utf-8")
    else:
        print(serialized)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command in {"submit", "validate"}:
        for path_arg in (args.spec, args.task_generator, args.reward_function, args.evaluator):
            if not Path(path_arg).is_file():
                print(f"error: file not found: {path_arg}", file=sys.stderr)
                return 2

        try:
            spec = _load_spec(Path(args.spec))
        except EnvSpecInvalid as exc:
            print(f"error: spec invalid: {exc}", file=sys.stderr)
            return 1
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: spec load failed: {exc}", file=sys.stderr)
            return 2

        bundle = _load_bundle(
            task_generator_path=Path(args.task_generator),
            reward_function_path=Path(args.reward_function),
            evaluator_path=Path(args.evaluator),
            task_generator_entry=args.task_generator_entry,
            reward_function_entry=args.reward_function_entry,
            evaluator_entry=args.evaluator_entry,
        )

        if args.command == "validate":
            try:
                report = validate_bundle_against_spec(bundle, spec)
            except BundleRejected as exc:
                print(f"rejected: {exc}", file=sys.stderr)
                return 1
            _emit_report(report, args.output)
            return 0 if report.valid else 1

        if args.command == "submit":
            registry = EnvRegistry() if args.register else None
            try:
                if registry is not None:
                    env_id, report = register_submission(registry, spec, bundle)
                    print(env_id)
                else:
                    report = validate_bundle_against_spec(bundle, spec)
            except BundleRejected as exc:
                print(f"rejected: {exc}", file=sys.stderr)
                return 1
            _emit_report(report, args.output)
            return 0 if report.valid else 1

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
