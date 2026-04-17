"""Permissionless environment sandbox v1.

Runs untrusted author-supplied Python source strings in an isolated
subprocess with resource caps, a whitelisted builtin surface, and a
post-compile AST audit. Validators call ``SandboxExecutor.run`` both
when verifying an env submission and when scoring miner rollouts.

Defense in depth:
- RestrictedPython compile_restricted strips unsafe bytecode at compile.
- AST audit rejects import statements + dunder escape paths before
  compile ever runs.
- Builtin whitelist limits what's callable at runtime.
- subprocess isolation via multiprocessing.spawn so a native-extension
  crash or stuck loop can be SIGKILL'd cleanly.
- resource.setrlimit caps address space + CPU seconds inside the child.
- JSON-only return values across the parent/child pipe (no pickle).

Spec: private/reliquary-plan/notes/spec-env-sandbox.md.
"""

from __future__ import annotations

import ast
import json
import multiprocessing
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any


SANDBOX_VERSION = "v1"

DEFAULT_TIMEOUT_MS = 30_000
DEFAULT_MEMORY_LIMIT_MB = 256
DEFAULT_STDOUT_CAP_BYTES = 64 * 1024


BLOCKED_NAMES = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
        "help",
        "quit",
        "exit",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "dir",
        "id",
        "hash",
    }
)

BLOCKED_ATTRIBUTE_SUFFIXES = frozenset(
    {
        "__class__",
        "__subclasses__",
        "__mro__",
        "__bases__",
        "__globals__",
        "__code__",
        "__builtins__",
        "__getattribute__",
        "__setattr__",
        "__delattr__",
        "__dict__",
        "__init_subclass__",
        "__subclasshook__",
        "__reduce__",
        "__reduce_ex__",
    }
)


class SandboxError(Exception):
    """Base class for sandbox-layer errors."""


class SandboxCompileError(SandboxError):
    """Raised when RestrictedPython refuses to compile the source."""


class SandboxPolicyViolation(SandboxError):
    """Raised when the AST audit rejects the source before compilation."""


@dataclass(frozen=True)
class SandboxPolicy:
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB
    stdout_cap_bytes: int = DEFAULT_STDOUT_CAP_BYTES


@dataclass
class SandboxResult:
    value: Any = None
    stdout: str = ""
    duration_ms: int = 0
    oom: bool = False
    timeout: bool = False
    error_type: str | None = None
    error_message: str | None = None
    policy: SandboxPolicy = field(default_factory=SandboxPolicy)

    @property
    def ok(self) -> bool:
        return self.error_type is None and not self.timeout and not self.oom


def _ast_audit(source: str) -> None:
    """Reject obviously dangerous constructs before compile_restricted runs."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SandboxCompileError(f"syntax error: {exc}") from exc

    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            errors.append(
                f"import statement not allowed: {ast.dump(node)[:120]}"
            )
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            errors.append(f"blocked name reference: {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr in BLOCKED_ATTRIBUTE_SUFFIXES:
                errors.append(f"blocked attribute access: .{node.attr}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            # Allow nonlocal inside nested closures but reject global statements
            # that would attempt to reach escaping scope.
            if isinstance(node, ast.Global):
                errors.append("`global` statement not allowed in sandbox")

    if errors:
        raise SandboxPolicyViolation("; ".join(errors[:10]))


def _restricted_builtins() -> dict[str, Any]:
    """Return the whitelisted builtin surface for sandboxed code."""
    safe_names = (
        "abs",
        "all",
        "any",
        "bool",
        "bytes",
        "chr",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "frozenset",
        "hasattr",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
        "True",
        "False",
        "None",
        "Exception",
        "ArithmeticError",
        "AssertionError",
        "AttributeError",
        "IndexError",
        "KeyError",
        "LookupError",
        "NameError",
        "NotImplementedError",
        "OverflowError",
        "RuntimeError",
        "StopIteration",
        "TypeError",
        "ValueError",
        "ZeroDivisionError",
    )
    import builtins as real_builtins

    surface: dict[str, Any] = {}
    for name in safe_names:
        if hasattr(real_builtins, name):
            surface[name] = getattr(real_builtins, name)
    return surface


def _safe_math_module() -> Any:
    import math

    allowed = {
        "sqrt",
        "log",
        "log2",
        "log10",
        "exp",
        "pow",
        "floor",
        "ceil",
        "fabs",
        "pi",
        "e",
        "sin",
        "cos",
        "tan",
        "atan",
        "atan2",
        "copysign",
        "trunc",
        "gcd",
        "inf",
        "nan",
        "isnan",
        "isinf",
        "isfinite",
    }

    class _SafeMath:
        pass

    holder = _SafeMath()
    for name in allowed:
        if hasattr(math, name):
            setattr(holder, name, getattr(math, name))
    return holder


def _sandbox_child(source: str, entry_point: str, args_json: str, policy_dict: dict, conn) -> None:  # type: ignore[no-untyped-def]
    """Subprocess worker that executes ``source`` under restricted globals."""
    import io

    try:
        import resource as _resource  # Unix-only; guarded.

        mem_bytes = int(policy_dict["memory_limit_mb"]) * 1024 * 1024
        try:
            _resource.setrlimit(_resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            pass
        cpu_seconds = max(1, int(policy_dict["timeout_ms"] / 1000) + 5)
        try:
            _resource.setrlimit(_resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except (ValueError, OSError):
            pass
    except ImportError:
        pass

    captured_stdout = io.StringIO()
    sys.stdout = captured_stdout
    sys.stderr = captured_stdout

    def _sandbox_print(*args: Any, sep: str = " ", end: str = "\n", **kwargs: Any) -> None:
        captured_stdout.write(sep.join(str(a) for a in args) + end)

    result: dict[str, Any] = {
        "value": None,
        "stdout": "",
        "error_type": None,
        "error_message": None,
    }

    try:
        from RestrictedPython import compile_restricted
        from RestrictedPython.PrintCollector import PrintCollector

        bytecode = compile_restricted(source, "<sandbox>", "exec")

        class _StdoutPrintCollector:
            """PrintCollector subclass that forwards to our captured_stdout
            in addition to the default ``printed`` accumulator."""

            def __init__(self, _getattr_: Any = None) -> None:
                self.txt: list[str] = []

            def write(self, text: str) -> None:
                self.txt.append(text)
                captured_stdout.write(text)

            def __call__(self) -> str:
                return "".join(self.txt)

            def _call_print(self, *args: Any, **kwargs: Any) -> None:
                sep = kwargs.get("sep", " ")
                end = kwargs.get("end", "\n")
                captured_stdout.write(sep.join(str(a) for a in args) + end)

        builtins_surface = _restricted_builtins()
        builtins_surface["print"] = _sandbox_print
        restricted_globals: dict[str, Any] = {
            "__builtins__": builtins_surface,
            "math": _safe_math_module(),
            "_getiter_": iter,
            "_getitem_": lambda obj, key: obj[key],
            "_iter_unpack_sequence_": lambda it, count, _: iter(it),
            "_unpack_sequence_": lambda it, count, _: list(it),
            "_write_": lambda obj: obj,
            "_inplacevar_": _default_inplace,
            "_print_": _StdoutPrintCollector,
            "_getattr_": _safe_getattr,
        }
        # Exec into a single namespace so module-level assignments are
        # visible to functions defined alongside them (Python's default
        # module-exec behavior). RestrictedPython's compile already blocks
        # unsafe scope elevation.
        exec(bytecode, restricted_globals)

        entry = restricted_globals.get(entry_point)
        if entry is None:
            raise NameError(f"entry_point {entry_point!r} not defined in sandbox source")
        if not callable(entry):
            raise TypeError(f"entry_point {entry_point!r} is not callable")
        args = json.loads(args_json)
        if not isinstance(args, list):
            raise TypeError(f"args must be a JSON list, got {type(args).__name__}")
        value = entry(*args)
        try:
            # JSON round-trip forces a serializable result and prunes
            # arbitrary object graphs from crossing the pipe.
            result["value"] = json.loads(json.dumps(value))
        except (TypeError, ValueError) as exc:
            result["error_type"] = "SerializationError"
            result["error_message"] = str(exc)
    except SandboxError as exc:
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error_message"] = f"{exc}\n{traceback.format_exc(limit=3)}"

    stdout_raw = captured_stdout.getvalue()
    stdout_cap = int(policy_dict.get("stdout_cap_bytes", DEFAULT_STDOUT_CAP_BYTES))
    if len(stdout_raw) > stdout_cap:
        result["stdout"] = (
            stdout_raw[:stdout_cap] + "\n[truncated:stdout_cap_bytes_exceeded]"
        )
    else:
        result["stdout"] = stdout_raw

    try:
        conn.send(result)
    except Exception:
        pass
    finally:
        conn.close()


def _safe_getattr(obj: Any, name: str, *args: Any) -> Any:
    """Guarded getattr — refuses access to blocked dunder names."""
    if name in BLOCKED_NAMES or name in BLOCKED_ATTRIBUTE_SUFFIXES:
        raise AttributeError(f"sandbox: attribute {name!r} is blocked")
    return getattr(obj, name, *args)


def _default_inplace(op: str, x: Any, y: Any) -> Any:
    ops = {
        "+=": lambda a, b: a + b,
        "-=": lambda a, b: a - b,
        "*=": lambda a, b: a * b,
        "/=": lambda a, b: a / b,
        "//=": lambda a, b: a // b,
        "%=": lambda a, b: a % b,
        "**=": lambda a, b: a ** b,
        "&=": lambda a, b: a & b,
        "|=": lambda a, b: a | b,
        "^=": lambda a, b: a ^ b,
        "<<=": lambda a, b: a << b,
        ">>=": lambda a, b: a >> b,
    }
    if op not in ops:
        raise ValueError(f"unsupported inplace op {op!r}")
    return ops[op](x, y)


class SandboxExecutor:
    """Runs author-supplied source strings under policy-enforced isolation."""

    def __init__(self, policy: SandboxPolicy | None = None) -> None:
        self.policy = policy or SandboxPolicy()
        # The spawn context ensures no inherited file descriptors or imports.
        self._ctx = multiprocessing.get_context("spawn")

    def audit(self, source: str) -> None:
        """AST-audit + compile_restricted dry run; raises on rejection."""
        _ast_audit(source)
        try:
            from RestrictedPython import compile_restricted

            compile_restricted(source, "<sandbox>", "exec")
        except SyntaxError as exc:
            raise SandboxCompileError(str(exc)) from exc

    def run(
        self,
        source: str,
        *,
        entry_point: str,
        args: list[Any] | None = None,
        policy: SandboxPolicy | None = None,
    ) -> SandboxResult:
        """Execute ``source`` under the sandbox and call ``entry_point(*args)``."""
        effective = policy or self.policy
        self.audit(source)

        args_json = json.dumps(args or [])
        parent_conn, child_conn = self._ctx.Pipe(duplex=False)
        process = self._ctx.Process(
            target=_sandbox_child,
            args=(source, entry_point, args_json, _policy_as_dict(effective), child_conn),
        )

        start = time.perf_counter()
        process.start()
        timeout_seconds = max(0.05, effective.timeout_ms / 1000.0)
        process.join(timeout_seconds)
        duration_ms = int((time.perf_counter() - start) * 1000)

        result = SandboxResult(policy=effective, duration_ms=duration_ms)

        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
            result.timeout = True
            try:
                child_conn.close()
            except Exception:
                pass
            try:
                parent_conn.close()
            except Exception:
                pass
            return result

        try:
            if parent_conn.poll():
                payload = parent_conn.recv()
                result.value = payload.get("value")
                result.stdout = payload.get("stdout") or ""
                result.error_type = payload.get("error_type")
                result.error_message = payload.get("error_message")
            else:
                # Child exited without sending — usually a signal kill
                # (OOM via SIGKILL on RLIMIT_AS or memory pressure).
                exitcode = process.exitcode
                if exitcode is not None and exitcode < 0:
                    # Negative exit code = killed by signal.
                    result.oom = True
                    result.error_type = "SandboxKilled"
                    result.error_message = f"child exited with signal {-exitcode}"
                else:
                    result.error_type = "SandboxEmpty"
                    result.error_message = f"child exited with no result (code={exitcode})"
        finally:
            try:
                parent_conn.close()
            except Exception:
                pass

        return result


def _policy_as_dict(policy: SandboxPolicy) -> dict[str, Any]:
    return {
        "timeout_ms": policy.timeout_ms,
        "memory_limit_mb": policy.memory_limit_mb,
        "stdout_cap_bytes": policy.stdout_cap_bytes,
    }


__all__ = [
    "BLOCKED_ATTRIBUTE_SUFFIXES",
    "BLOCKED_NAMES",
    "DEFAULT_MEMORY_LIMIT_MB",
    "DEFAULT_STDOUT_CAP_BYTES",
    "DEFAULT_TIMEOUT_MS",
    "SANDBOX_VERSION",
    "SandboxCompileError",
    "SandboxError",
    "SandboxExecutor",
    "SandboxPolicy",
    "SandboxPolicyViolation",
    "SandboxResult",
]
