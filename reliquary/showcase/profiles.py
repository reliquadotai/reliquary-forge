from __future__ import annotations

from copy import deepcopy
from typing import Any

SHOWCASE_PROFILES: dict[str, dict[str, Any]] = {
    "showcase": {
        "name": "showcase",
        "label": "Reliquary Showcase",
        "description": "Coordination-first local subnet walk-through with five differentiated miners.",
        "seed": 7,
        "windows": 2,
        "challenge_count": 4,
        "rollouts_per_challenge": 3,
        "eval_count": 6,
        "chain_mode": "local",
        "miners": [
            {
                "miner_id": "atlas-honest",
                "label": "Atlas",
                "archetype": "honest_high_quality",
                "strategy": "guided_exact",
                "proof_corruption": "none",
                "miner_secret": "atlas-secret",
            },
            {
                "miner_id": "nova-steady",
                "label": "Nova",
                "archetype": "honest_average",
                "strategy": "guided_mixed",
                "proof_corruption": "none",
                "miner_secret": "nova-secret",
            },
            {
                "miner_id": "echo-repeat",
                "label": "Echo",
                "archetype": "low_diversity",
                "strategy": "guided_duplicate",
                "proof_corruption": "none",
                "miner_secret": "echo-secret",
            },
            {
                "miner_id": "drift-weak",
                "label": "Drift",
                "archetype": "weak_policy",
                "strategy": "random_guess",
                "proof_corruption": "none",
                "miner_secret": "drift-secret",
            },
            {
                "miner_id": "shade-broken",
                "label": "Shade",
                "archetype": "malformed_proof_failing",
                "strategy": "malformed",
                "proof_corruption": "bad_signature",
                "malformed_text": "assignment unavailable",
                "miner_secret": "shade-secret",
            },
        ],
    },
    "recording": {
        "name": "recording",
        "label": "Reliquary Recording",
        "description": "Tighter deterministic profile for screen captures and short walkthroughs.",
        "seed": 11,
        "windows": 2,
        "challenge_count": 3,
        "rollouts_per_challenge": 2,
        "eval_count": 4,
        "chain_mode": "local",
        "miners": [
            {
                "miner_id": "atlas-honest",
                "label": "Atlas",
                "archetype": "honest_high_quality",
                "strategy": "guided_exact",
                "proof_corruption": "none",
                "miner_secret": "atlas-secret",
            },
            {
                "miner_id": "nova-steady",
                "label": "Nova",
                "archetype": "honest_average",
                "strategy": "guided_mixed",
                "proof_corruption": "none",
                "miner_secret": "nova-secret",
            },
            {
                "miner_id": "echo-repeat",
                "label": "Echo",
                "archetype": "low_diversity",
                "strategy": "guided_duplicate",
                "proof_corruption": "none",
                "miner_secret": "echo-secret",
            },
            {
                "miner_id": "shade-broken",
                "label": "Shade",
                "archetype": "malformed_proof_failing",
                "strategy": "malformed",
                "proof_corruption": "bad_signature",
                "malformed_text": "assignment unavailable",
                "miner_secret": "shade-secret",
            },
        ],
    },
    "distill": {
        "name": "distill",
        "label": "Reliquary Distillation",
        "description": "Reasoning-trace distillation walk-through with verifiable provenance, contamination checks, and policy promotion.",
        "mode": "distill",
        "seed": 13,
        "windows": 2,
        "task_count": 6,
        "chain_mode": "local",
        "miners": [
            {
                "miner_id": "sage-teacher",
                "label": "Sage",
                "archetype": "reasoned_exact",
                "strategy": "reasoned_exact",
                "miner_secret": "sage-secret",
            },
            {
                "miner_id": "nova-mixed",
                "label": "Nova",
                "archetype": "reasoned_mixed",
                "strategy": "reasoned_mixed",
                "miner_secret": "nova-secret",
            },
            {
                "miner_id": "echo-duplicate",
                "label": "Echo",
                "archetype": "reasoned_duplicate",
                "strategy": "reasoned_duplicate",
                "miner_secret": "echo-secret",
            },
            {
                "miner_id": "leak-bad",
                "label": "Leak",
                "archetype": "reasoned_contaminated",
                "strategy": "reasoned_contaminated",
                "miner_secret": "leak-secret",
            },
            {
                "miner_id": "mist-malformed",
                "label": "Mist",
                "archetype": "reasoned_malformed",
                "strategy": "reasoned_malformed",
                "miner_secret": "mist-secret",
            },
        ],
    },
}


def get_demo_profile(
    name: str,
    *,
    windows: int | None = None,
    challenge_count: int | None = None,
    rollouts_per_challenge: int | None = None,
) -> dict[str, Any]:
    if name not in SHOWCASE_PROFILES:
        known = ", ".join(sorted(SHOWCASE_PROFILES))
        raise ValueError(f"Unknown demo profile '{name}'. Known profiles: {known}")

    profile = deepcopy(SHOWCASE_PROFILES[name])
    if windows is not None:
        profile["windows"] = windows
    if challenge_count is not None:
        if profile.get("mode") == "distill":
            profile["task_count"] = challenge_count
        else:
            profile["challenge_count"] = challenge_count
    if rollouts_per_challenge is not None:
        profile["rollouts_per_challenge"] = rollouts_per_challenge
    return profile


def miner_secret_map(profile: dict[str, Any]) -> dict[str, str]:
    return {
        miner["miner_id"]: miner["miner_secret"]
        for miner in profile["miners"]
        if "miner_secret" in miner
    }
