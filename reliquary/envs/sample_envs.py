"""First-class registry entries: SAT + arithmetic-reasoning envs.

These dogfood the permissionless env registry by expressing the two
envs that already live in ``envs/sat.py`` and ``envs/reasoning.py`` as
author-supplied source-string bundles that pass through the sandbox and
registry the same way any external submission would.

Each env has:
- ``*_source`` constants: the actual source strings a validator would
  hash + audit + execute.
- ``build_<name>_bundle()``: returns an EnvBundle wrapping the sources.
- ``build_<name>_spec()``: returns an EnvSpec whose ``*_hash`` fields
  match the bundle's sha256 and whose sample_task / sample_solution /
  expected_reward yield reward==1.0 under the evaluator.

Author-hotkey is set to a well-known testnet hotkey (``A`` * 48) since
these are dogfood entries, not permissionless submissions. Stake lock
is 0.0 — testnet policy. Production authors replace both.
"""

from __future__ import annotations

import hashlib

from .registry import EnvBundle
from .spec import EnvMetadata, EnvSpec


# ---------------------------------------------------------------------------
# SAT env source (self-contained, no reliance on reliquary.envs.sat)
# ---------------------------------------------------------------------------

SAT_TASK_GENERATOR_SOURCE = (
    "def generate(seed):\n"
    "    # Deterministic 3-SAT generator. Fixed small instance for the\n"
    "    # sample; real validators can pass larger (n, k, m, seed) via\n"
    "    # the registry's task-batch fanout (outside sandbox).\n"
    "    n = 3\n"
    "    solution = {1: True, 2: False, 3: True}\n"
    "    clauses = [\n"
    "        [1, -2, 3],\n"
    "        [1, 2, -3],\n"
    "        [-1, 2, 3],\n"
    "    ]\n"
    "    return {\n"
    "        'n': n,\n"
    "        'solution': solution,\n"
    "        'clauses': clauses,\n"
    "    }\n"
)


SAT_REWARD_FUNCTION_SOURCE = (
    "def reward(assignment, clauses):\n"
    "    # Fraction of clauses satisfied by the assignment.\n"
    "    if not clauses:\n"
    "        return 0.0\n"
    "    satisfied = 0\n"
    "    for clause in clauses:\n"
    "        clause_ok = False\n"
    "        for lit in clause:\n"
    "            var = lit if lit > 0 else -lit\n"
    "            want = lit > 0\n"
    "            got = assignment.get(str(var))\n"
    "            if got is None:\n"
    "                got = assignment.get(var)\n"
    "            if got == want:\n"
    "                clause_ok = True\n"
    "                break\n"
    "        if clause_ok:\n"
    "            satisfied += 1\n"
    "    return satisfied / len(clauses)\n"
)


SAT_EVALUATOR_SOURCE = (
    "def evaluate(task, solution):\n"
    "    clauses = task['clauses']\n"
    "    assignment = solution['assignment']\n"
    "    if not clauses:\n"
    "        return 0.0\n"
    "    satisfied = 0\n"
    "    for clause in clauses:\n"
    "        clause_ok = False\n"
    "        for lit in clause:\n"
    "            var = lit if lit > 0 else -lit\n"
    "            want = lit > 0\n"
    "            got = assignment.get(str(var))\n"
    "            if got is None:\n"
    "                got = assignment.get(var)\n"
    "            if got == want:\n"
    "                clause_ok = True\n"
    "                break\n"
    "        if clause_ok:\n"
    "            satisfied += 1\n"
    "    return 1.0 if satisfied == len(clauses) else 0.0\n"
)


SAT_SAMPLE_TASK = (
    '{"n": 3, "clauses": [[1, -2, 3], [1, 2, -3], [-1, 2, 3]], '
    '"solution": {"1": true, "2": false, "3": true}}'
)
SAT_SAMPLE_SOLUTION = '{"assignment": {"1": true, "2": false, "3": true}}'


# ---------------------------------------------------------------------------
# Arithmetic reasoning env
# ---------------------------------------------------------------------------

ARITHMETIC_TASK_GENERATOR_SOURCE = (
    "def generate(seed):\n"
    "    # Deterministic arithmetic task: addition of two small ints.\n"
    "    a = 12 + (seed % 7)\n"
    "    b = 5 + (seed % 11)\n"
    "    return {\n"
    "        'operation': 'add',\n"
    "        'inputs': {'left': a, 'right': b},\n"
    "        'prompt': 'Add ' + str(a) + ' and ' + str(b) + '.',\n"
    "        'answer': a + b,\n"
    "    }\n"
)


ARITHMETIC_REWARD_FUNCTION_SOURCE = (
    "def reward(predicted, target):\n"
    "    return 1.0 if int(predicted) == int(target) else 0.0\n"
)


ARITHMETIC_EVALUATOR_SOURCE = (
    "def evaluate(task, solution):\n"
    "    target = task['answer']\n"
    "    guess = solution.get('final_answer')\n"
    "    try:\n"
    "        return 1.0 if int(guess) == int(target) else 0.0\n"
    "    except (ValueError, TypeError):\n"
    "        return 0.0\n"
)


ARITHMETIC_SAMPLE_TASK = (
    '{"operation": "add", "inputs": {"left": 12, "right": 5}, '
    '"prompt": "Add 12 and 5.", "answer": 17}'
)
ARITHMETIC_SAMPLE_SOLUTION = '{"final_answer": 17}'


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _hashes(*sources: str) -> list[str]:
    return [hashlib.sha256(src.encode("utf-8")).hexdigest() for src in sources]


def build_sat_bundle() -> EnvBundle:
    return EnvBundle(
        task_generator_source=SAT_TASK_GENERATOR_SOURCE,
        reward_function_source=SAT_REWARD_FUNCTION_SOURCE,
        evaluator_source=SAT_EVALUATOR_SOURCE,
    )


def build_sat_spec(
    *,
    author_hotkey: str = "A" * 48,
    stake_lock_tao: float = 0.0,
) -> EnvSpec:
    bundle = build_sat_bundle()
    hashes = bundle.code_hashes()
    return EnvSpec(
        name="sat_classic",
        version="1.0.0",
        author_hotkey=author_hotkey,
        author_stake_lock_tao=stake_lock_tao,
        task_generator_hash=hashes["task_generator_hash"],
        reward_function_hash=hashes["reward_function_hash"],
        evaluator_hash=hashes["evaluator_hash"],
        sample_task=SAT_SAMPLE_TASK,
        sample_solution=SAT_SAMPLE_SOLUTION,
        expected_reward=1.0,
        metadata=EnvMetadata(
            context_length=2048,
            timeout_ms=10_000,
            difficulty="medium",
            domain="logic",
        ),
    )


def build_arithmetic_bundle() -> EnvBundle:
    return EnvBundle(
        task_generator_source=ARITHMETIC_TASK_GENERATOR_SOURCE,
        reward_function_source=ARITHMETIC_REWARD_FUNCTION_SOURCE,
        evaluator_source=ARITHMETIC_EVALUATOR_SOURCE,
    )


def build_arithmetic_spec(
    *,
    author_hotkey: str = "A" * 48,
    stake_lock_tao: float = 0.0,
) -> EnvSpec:
    bundle = build_arithmetic_bundle()
    hashes = bundle.code_hashes()
    return EnvSpec(
        name="arithmetic_basic",
        version="1.0.0",
        author_hotkey=author_hotkey,
        author_stake_lock_tao=stake_lock_tao,
        task_generator_hash=hashes["task_generator_hash"],
        reward_function_hash=hashes["reward_function_hash"],
        evaluator_hash=hashes["evaluator_hash"],
        sample_task=ARITHMETIC_SAMPLE_TASK,
        sample_solution=ARITHMETIC_SAMPLE_SOLUTION,
        expected_reward=1.0,
        metadata=EnvMetadata(
            context_length=1024,
            timeout_ms=5_000,
            difficulty="easy",
            domain="math",
        ),
    )


__all__ = [
    "ARITHMETIC_EVALUATOR_SOURCE",
    "ARITHMETIC_REWARD_FUNCTION_SOURCE",
    "ARITHMETIC_SAMPLE_SOLUTION",
    "ARITHMETIC_SAMPLE_TASK",
    "ARITHMETIC_TASK_GENERATOR_SOURCE",
    "SAT_EVALUATOR_SOURCE",
    "SAT_REWARD_FUNCTION_SOURCE",
    "SAT_SAMPLE_SOLUTION",
    "SAT_SAMPLE_TASK",
    "SAT_TASK_GENERATOR_SOURCE",
    "build_arithmetic_bundle",
    "build_arithmetic_spec",
    "build_sat_bundle",
    "build_sat_spec",
]
