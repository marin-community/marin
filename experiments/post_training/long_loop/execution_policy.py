# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controlled execution envelope for comparable long-loop experiments."""

from datetime import timedelta
from enum import StrEnum
from types import MappingProxyType

DEFAULT_TASK_TIMEOUT_SECONDS = 3 * 24 * 60 * 60
SFT_CHECKPOINT_INTERVAL = timedelta(hours=2)
SFT_TRAINING_DURATION = timedelta(hours=20)
SFT_FINALIZATION_RESERVE = timedelta(minutes=40)
SFT_UPDATE_RESERVE = timedelta(minutes=10)
RL_TRAINING_BUDGET = MappingProxyType({"seconds": 43200, "drain_margin": 1800})


class RlStopping(StrEnum):
    WALL_TIME = "wall_time"
    MAX_STEPS = "max_steps"


def rl_training_callbacks(
    callbacks: list[dict], *, groups_per_step: int, stopping: RlStopping = RlStopping.WALL_TIME
) -> list[dict]:
    """Bind the shared RL clock, or remove it for an explicit step-bounded campaign."""
    if stopping is RlStopping.MAX_STEPS:
        return [callback for callback in callbacks if callback["type"] != "wall_time_budget"]
    for callback in callbacks:
        if callback["type"] == "wall_time_budget":
            for field, expected in RL_TRAINING_BUDGET.items():
                if callback[field] != expected:
                    raise ValueError(
                        f"Long-loop RL {field}={callback[field]} differs from the canonical budget {expected}"
                    )
    return [
        *(callback for callback in callbacks if callback["type"] != "wall_time_budget"),
        {"type": "wall_time_budget", **RL_TRAINING_BUDGET, "groups_per_step": groups_per_step},
    ]


# Final evaluation is deliberately outside the 36-hour training/finalization envelope.
STAGE_TIMEOUT_SECONDS = {
    "inputs": 1800,
    "sft": int(SFT_TRAINING_DURATION.total_seconds()),
    "export": 3600,
    "rl": 12 * 3600,
    "eval_inputs": 1800,
    "accounting": 1800,
    "archive": 5400,
    "evaluation": DEFAULT_TASK_TIMEOUT_SECONDS,
}
STAGE_GPU_COUNTS = {
    "inputs": 0,
    "sft": 256,
    "export": 0,
    "rl": 256,
    "eval_inputs": 0,
    "archive": 0,
    "evaluation": 136,
}
COORDINATOR_TIMEOUT_SECONDS = sum(STAGE_TIMEOUT_SECONDS.values()) + 3600


def bind_stage_timeouts(resolved: dict) -> None:
    """Fill omitted stage deadlines from the canonical experiment controls."""
    execution = resolved["execution"]
    timeouts = execution.setdefault("task_timeouts", {})
    unknown = timeouts.keys() - STAGE_TIMEOUT_SECONDS.keys()
    if unknown:
        raise ValueError(f"Unknown stage timeouts: {sorted(unknown)}")
    explicit_dispatch = {
        "sft": resolved["sft"].get("task_timeout_seconds"),
        **{stage: resolved[stage]["execution"].get("job_timeout_seconds") for stage in ("rl", "evaluation")},
    }
    for stage, seconds in explicit_dispatch.items():
        if seconds is not None:
            if stage in timeouts and timeouts[stage] != seconds:
                raise ValueError(f"Conflicting task deadlines for {stage}")
            timeouts.setdefault(stage, seconds)
    for stage, seconds in STAGE_TIMEOUT_SECONDS.items():
        timeouts.setdefault(stage, seconds)
    # Keep a single source for the native dispatch timeout.
    resolved["sft"].setdefault("task_timeout_seconds", timeouts["sft"])
    coordinator = execution["coordinator"]
    coordinator.setdefault("job_timeout_seconds", COORDINATOR_TIMEOUT_SECONDS)
    coordinator.setdefault("coordinator_timeout_hours", COORDINATOR_TIMEOUT_SECONDS // 3600)
    for stage in ("rl", "evaluation"):
        resolved[stage]["execution"].setdefault("job_timeout_seconds", timeouts[stage])


def experimental_controls(resolved: dict, *, acknowledge: bool, reason: str | None) -> dict:
    """Reject allocation/deadline drift unless the caller explicitly records an exception."""
    actual = {
        **{f"timeout/{stage}": seconds for stage, seconds in resolved["execution"]["task_timeouts"].items()},
        **{f"gpus/{stage}": count for stage, count in resolved["gpu_peaks"].items()},
        "dispatch/sft": resolved["sft"]["task_timeout_seconds"],
        "dispatch/rl": resolved["rl"]["execution"]["job_timeout_seconds"],
        "dispatch/evaluation": resolved["evaluation"]["execution"]["job_timeout_seconds"],
        "coordinator/job": resolved["execution"]["coordinator"]["job_timeout_seconds"],
        "coordinator/lifetime": resolved["execution"]["coordinator"]["coordinator_timeout_hours"] * 3600,
    }
    expected = {
        **{f"timeout/{stage}": seconds for stage, seconds in STAGE_TIMEOUT_SECONDS.items()},
        **{f"gpus/{stage}": count for stage, count in STAGE_GPU_COUNTS.items()},
        "dispatch/sft": STAGE_TIMEOUT_SECONDS["sft"],
        "dispatch/rl": STAGE_TIMEOUT_SECONDS["rl"],
        "dispatch/evaluation": STAGE_TIMEOUT_SECONDS["evaluation"],
        "coordinator/job": COORDINATOR_TIMEOUT_SECONDS,
        "coordinator/lifetime": COORDINATOR_TIMEOUT_SECONDS,
    }
    changes = [
        {"control": name, "default": value, "actual": actual[name]}
        for name, value in expected.items()
        if actual[name] != value
    ]
    if changes and not acknowledge:
        raise ValueError(
            "Experimental controls changed; require --i-know-i-am-breaking-experimental-controls: " + str(changes)
        )
    if changes and not reason:
        raise ValueError("Supply --experimental-controls-reason to archive the override decision")
    return {
        "acknowledged": acknowledge,
        "reason": reason,
        "changes": changes,
        "defaults": expected,
        "effective": actual,
        "non_eval_timeout_seconds": sum(
            value for stage, value in STAGE_TIMEOUT_SECONDS.items() if stage != "evaluation"
        ),
        "final_evaluation_outside_envelope": True,
    }
