# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Interpret Skill2Env's native component rewards for Atlas reviews."""

import math
from collections.abc import Mapping
from enum import StrEnum
from typing import Literal, TypedDict


class HarborRewardMode(StrEnum):
    SCALAR = "scalar"
    SKILL2ENV_COMPONENTS = "skill2env_components"


class Skill2EnvDiagnostics(TypedDict):
    native_rewards: dict[str, float]
    score_aggregation: Literal["component_mean"]
    pass_criterion: Literal["all_components_equal_one"]


class Skill2EnvVerification(TypedDict):
    """Serialized verdict shared with the separately configured native runtime."""

    status: Literal["verified"]
    score: float
    passed: bool
    score_min: float
    score_max: float
    reason: None
    diagnostics: Skill2EnvDiagnostics


def skill2env_verification(rewards: Mapping[str, float]) -> Skill2EnvVerification:
    """Preserve component scores and apply Skill2Env's full-pass criterion."""
    if not rewards or any(not math.isfinite(value) or not 0 <= value <= 1 for value in rewards.values()):
        raise ValueError("Skill2Env requires nonempty finite component rewards in [0, 1]")
    return {
        "status": "verified",
        "score": sum(rewards.values()) / len(rewards),
        "passed": all(math.isclose(value, 1.0, abs_tol=1e-9) for value in rewards.values()),
        "score_min": 0.0,
        "score_max": 1.0,
        "reason": None,
        "diagnostics": {
            "native_rewards": dict(rewards),
            "score_aggregation": "component_mean",
            "pass_criterion": "all_components_equal_one",
        },
    }
