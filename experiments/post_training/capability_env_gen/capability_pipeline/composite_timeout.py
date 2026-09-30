"""Whole-trial time budgets for the pinned composite verifier."""

from __future__ import annotations

import math

# The outer verifier is a wall clock budget, not a sum of every provider's
# individual retry cap. A slow provider remains an infrastructure outcome. The
# 300-second allowance is per isolated machine check, on top of its declared
# script timeout. It exceeds the measured ~74-second/check empty controls in
# pilot003 while leaving room for provisioning and confirmed cleanup.
MACHINE_ORCHESTRATION_SECONDS = 300
# The pinned TaskCompendium checklist contract uses this per-request bound.
JUDGE_REQUEST_SECONDS = 120
JUDGE_SETUP_SECONDS = 60
CALIBRATION_STARTUP_SECONDS = 120


def composite_verifier_timeout(config: dict, step_index: int) -> int:
    """Bound all sequential private checks and every permitted judge call."""
    try:
        step = config["steps"][step_index]
        if step["step_index"] != step_index:
            raise ValueError("composite timeout step index differs from configuration")
        checks = step["machine_checks"]
        judge = step["judge"]
        criteria = len(judge["criterion_weights"])
        initial_samples = (judge.get("consensus") or {}).get("initial_samples", 1)
        adjudicator_samples = 1 if judge.get("consensus") is not None else 0
        if not checks or criteria < 1 or initial_samples < 1:
            raise ValueError("composite timeout needs positive checks and judge calls")
        machine = sum(
            MACHINE_ORCHESTRATION_SECONDS
            + int(check["timeout"])
            for check in checks
        )
        judge_seconds = (
            criteria * (initial_samples + adjudicator_samples) * JUDGE_REQUEST_SECONDS
        )
        return machine + judge_seconds + JUDGE_SETUP_SECONDS
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("composite timeout requires a complete verified config") from error


def calibration_wall_timeout(
    config: dict, *, cases: int, repeats: int, concurrency: int
) -> int:
    """Allow every scheduled wave its declared whole-verifier budget."""
    if min(cases, repeats, concurrency) < 1:
        raise ValueError("calibration cases, repeats and concurrency must be positive")
    per_trial = max(
        composite_verifier_timeout(config, step_index)
        for step_index in range(len(config["steps"]))
    )
    return math.ceil(cases * repeats / concurrency) * per_trial + CALIBRATION_STARTUP_SECONDS
