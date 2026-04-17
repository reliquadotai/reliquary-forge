"""Acceptance tests for the permissionless env sandbox.

The sandbox runs untrusted source strings in a spawned subprocess with
RestrictedPython + AST audit + rlimit caps. These tests pin the defense-
in-depth surface so an accidental relaxation is caught at CI.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "RestrictedPython",
    reason="RestrictedPython required for sandbox tests",
)

from reliquary.envs.sandbox import (
    SandboxCompileError,
    SandboxExecutor,
    SandboxPolicy,
    SandboxPolicyViolation,
    SandboxResult,
    _ast_audit,
)


def _executor(timeout_ms: int = 3_000) -> SandboxExecutor:
    return SandboxExecutor(SandboxPolicy(timeout_ms=timeout_ms, memory_limit_mb=128))


# ---------------------------------------------------------------------------
# AST audit
# ---------------------------------------------------------------------------


def test_ast_audit_rejects_import_statement() -> None:
    with pytest.raises(SandboxPolicyViolation):
        _ast_audit("import os")


def test_ast_audit_rejects_from_import() -> None:
    with pytest.raises(SandboxPolicyViolation):
        _ast_audit("from socket import socket")


def test_ast_audit_rejects_dunder_class_escape() -> None:
    source = "def f():\n    return ().__class__.__bases__"
    with pytest.raises(SandboxPolicyViolation):
        _ast_audit(source)


def test_ast_audit_rejects_dunder_mro_escape() -> None:
    source = "def f():\n    return type.__mro__"
    with pytest.raises(SandboxPolicyViolation):
        _ast_audit(source)


def test_ast_audit_rejects_subclasses_escape() -> None:
    source = "def f():\n    return object.__subclasses__"
    with pytest.raises(SandboxPolicyViolation):
        _ast_audit(source)


def test_ast_audit_rejects_globals_reference() -> None:
    with pytest.raises(SandboxPolicyViolation):
        _ast_audit("def f(): return globals()")


def test_ast_audit_rejects_exec_name_reference() -> None:
    with pytest.raises(SandboxPolicyViolation):
        _ast_audit("def f(): return exec('x = 1')")


def test_ast_audit_rejects_global_statement() -> None:
    with pytest.raises(SandboxPolicyViolation):
        _ast_audit("def f():\n    global x\n    x = 1")


def test_ast_audit_accepts_benign_function() -> None:
    _ast_audit("def f(x): return x + 1")


def test_ast_audit_accepts_math_usage() -> None:
    _ast_audit(
        "def f(x):\n    return math.sqrt(x) + math.pi"
    )


def test_ast_audit_syntax_error_wrapped() -> None:
    with pytest.raises(SandboxCompileError):
        _ast_audit("def f(: pass")


# ---------------------------------------------------------------------------
# Benign execution
# ---------------------------------------------------------------------------


def test_run_simple_addition() -> None:
    executor = _executor()
    result = executor.run(
        "def compute(a, b):\n    return a + b",
        entry_point="compute",
        args=[3, 4],
    )
    assert result.ok
    assert result.value == 7
    assert result.timeout is False
    assert result.oom is False


def test_run_returns_list_preserved() -> None:
    executor = _executor()
    result = executor.run(
        "def build():\n    return [1, 2, 3]",
        entry_point="build",
    )
    assert result.value == [1, 2, 3]


def test_run_uses_safe_math_module() -> None:
    executor = _executor()
    result = executor.run(
        "def compute(x):\n    return math.sqrt(x)",
        entry_point="compute",
        args=[16],
    )
    assert result.ok
    assert result.value == pytest.approx(4.0)


def test_run_print_captures_stdout() -> None:
    executor = _executor()
    result = executor.run(
        "def greet(name):\n    print('hello', name)\n    return name",
        entry_point="greet",
        args=["world"],
    )
    assert result.ok
    assert "hello world" in result.stdout


# ---------------------------------------------------------------------------
# Runtime blocks
# ---------------------------------------------------------------------------


def test_run_rejects_import_in_body() -> None:
    executor = _executor()
    with pytest.raises(SandboxPolicyViolation):
        executor.run(
            "import os\ndef f():\n    return os.listdir('.')",
            entry_point="f",
        )


def test_run_rejects_open_attempt_at_ast_audit() -> None:
    executor = _executor()
    with pytest.raises(SandboxPolicyViolation):
        executor.run(
            "def read():\n    return open('/etc/passwd').read()",
            entry_point="read",
        )


def test_run_rejects_eval_attempt() -> None:
    executor = _executor()
    with pytest.raises(SandboxPolicyViolation):
        executor.run(
            "def f():\n    return eval('1+1')",
            entry_point="f",
        )


def test_run_rejects_attribute_escape() -> None:
    executor = _executor()
    with pytest.raises(SandboxPolicyViolation):
        executor.run(
            "def f():\n    return ().__class__.__bases__[0]",
            entry_point="f",
        )


# ---------------------------------------------------------------------------
# Timeout + error propagation
# ---------------------------------------------------------------------------


def test_run_timeout_infinite_loop() -> None:
    executor = _executor(timeout_ms=500)
    result = executor.run(
        "def loop():\n    x = 0\n    while True:\n        x += 1",
        entry_point="loop",
    )
    assert result.timeout is True
    assert result.value is None


def test_run_user_exception_captured() -> None:
    executor = _executor()
    result = executor.run(
        "def f():\n    raise ValueError('nope')",
        entry_point="f",
    )
    assert result.timeout is False
    assert result.error_type == "ValueError"
    assert "nope" in (result.error_message or "")


def test_run_missing_entry_point_reported() -> None:
    executor = _executor()
    result = executor.run(
        "def other():\n    return 1",
        entry_point="nonexistent",
    )
    assert result.error_type == "NameError"
    assert "nonexistent" in (result.error_message or "")


def test_run_non_json_return_flagged() -> None:
    executor = _executor()
    result = executor.run(
        "def f():\n    def inner(): return 1\n    return inner",
        entry_point="f",
    )
    assert result.error_type == "SerializationError"


# ---------------------------------------------------------------------------
# Isolation between runs
# ---------------------------------------------------------------------------


def test_runs_do_not_share_state() -> None:
    executor = _executor()
    result1 = executor.run(
        "counter = 1\ndef bump():\n    return counter + 100",
        entry_point="bump",
    )
    assert result1.ok
    # Fresh child, fresh globals: even the var `counter` doesn't leak.
    result2 = executor.run(
        "def use():\n    try:\n        return counter\n    except NameError:\n        return 'undefined'",
        entry_point="use",
    )
    assert result2.ok
    assert result2.value == "undefined"


# ---------------------------------------------------------------------------
# Sample-task re-run flow (dogfood of the Epic 2 validation path)
# ---------------------------------------------------------------------------


def test_sample_task_rerun_matches_expected_reward() -> None:
    """Simulate the validator's pre-quorum check: run author code on the
    sample task, compare the computed reward to the declared
    ``expected_reward``. Author says the right answer earns 1.0."""
    executor = _executor()
    source = (
        "def evaluate(task, solution):\n"
        "    target = task['answer']\n"
        "    guess = solution['guess']\n"
        "    return 1.0 if guess == target else 0.0\n"
    )
    result = executor.run(
        source,
        entry_point="evaluate",
        args=[{"answer": 42}, {"guess": 42}],
    )
    assert result.ok
    assert result.value == pytest.approx(1.0)

    # Wrong guess → 0.0.
    result2 = executor.run(
        source,
        entry_point="evaluate",
        args=[{"answer": 42}, {"guess": 41}],
    )
    assert result2.value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Executor audit-only API
# ---------------------------------------------------------------------------


def test_executor_audit_only_accepts_benign() -> None:
    executor = _executor()
    # Should not raise.
    executor.audit("def f(x): return x + 1")


def test_executor_audit_only_rejects_bad() -> None:
    executor = _executor()
    with pytest.raises(SandboxPolicyViolation):
        executor.audit("import os")
