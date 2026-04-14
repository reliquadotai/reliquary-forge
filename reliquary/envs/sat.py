from __future__ import annotations

import random
import re
from typing import Any


def generate_sat_challenge(
    n: int,
    k: int,
    m: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    solution = {i: rng.choice([True, False]) for i in range(1, n + 1)}
    clauses: list[list[int]] = []

    for _ in range(m):
        vars_in_clause = rng.sample(list(solution), k)
        satisfying_var = rng.choice(vars_in_clause)
        clause: list[int] = []
        for var in vars_in_clause:
            if var == satisfying_var:
                lit = var if solution[var] else -var
            else:
                lit = var if rng.choice([True, False]) else -var
            clause.append(lit)
        clauses.append(clause)

    formula = " ^ ".join(
        "(" + " v ".join(f"{'' if lit > 0 else '!'}x{abs(lit)}" for lit in clause) + ")"
        for clause in clauses
    )
    prompt = (
        f"Find a satisfying assignment for the following {k}-SAT formula over variables x1..x{n}.\n"
        f"{formula}\n"
        "Return comma-separated assignments like x1=True, x2=False. "
        "If you cannot fully solve it, still provide your best assignment guess."
    )

    return {
        "challenge_id": f"sat-{seed}",
        "env_id": "sat",
        "seed": seed,
        "prompt": prompt,
        "extra": {
            "n": n,
            "k": k,
            "m": m,
            "solution": solution,
            "clauses": clauses,
            "formula": formula,
        },
    }


def parse_assignments(text: str) -> dict[int, bool]:
    matches = re.findall(r"x(\d+)\s*=\s*(True|False|1|0)", text or "")
    parsed: dict[int, bool] = {}
    for var, raw in matches:
        parsed[int(var)] = raw.lower() in {"true", "1"}
    return parsed


def evaluate_sat_response(challenge: dict[str, Any], response_text: str) -> dict[str, Any]:
    clauses = challenge["extra"]["clauses"]
    expected = challenge["extra"]["solution"]
    got = parse_assignments(response_text)
    expected_vars = set(expected)
    provided_vars = set(got)
    out_of_range_vars = sorted(var for var in provided_vars if var not in expected_vars)
    missing_vars = sorted(var for var in expected_vars if var not in provided_vars)
    format_ok = bool(got) and not out_of_range_vars
    format_reason = "ok"
    if not got:
        format_reason = "missing_assignments"
    elif out_of_range_vars:
        format_reason = "out_of_range_variables"

    if not clauses:
        return {
            "score": 0.0,
            "dense_reward": 0.0,
            "satisfied_fraction": 0.0,
            "got": got,
            "expected": expected,
            "assignment_count": len(got),
            "expected_assignment_count": len(expected),
            "missing_variables": missing_vars,
            "out_of_range_variables": out_of_range_vars,
            "format_ok": format_ok,
            "format_reason": format_reason,
            "accepted": False,
        }

    satisfied = 0
    for clause in clauses:
        clause_ok = any((lit > 0) == got.get(abs(lit), None) for lit in clause)
        satisfied += int(clause_ok)

    satisfied_fraction = satisfied / len(clauses)
    exact = float(satisfied == len(clauses))

    return {
        "score": exact,
        "dense_reward": satisfied_fraction,
        "satisfied_fraction": satisfied_fraction,
        "got": got,
        "expected": expected,
        "assignment_count": len(got),
        "expected_assignment_count": len(expected),
        "missing_variables": missing_vars,
        "out_of_range_variables": out_of_range_vars,
        "format_ok": format_ok,
        "format_reason": format_reason,
        "accepted": format_ok,
    }
