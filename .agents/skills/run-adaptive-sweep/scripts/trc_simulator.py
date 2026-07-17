#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Black-box simulator of experiment results and TPU execution."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import runpy
import secrets
import sys
from collections import Counter
from pathlib import Path
from typing import Any

TOOLS = runpy.run_path(Path(__file__).with_name("sweep_tools.py"))
check_convergence = TOOLS["check_convergence"]
predict_objectives = TOOLS["predict_objectives"]
rank_targets = TOOLS["rank_targets"]

AXES: list[dict[str, Any]] = [
    {
        "name": "learning_rate",
        "values": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        "scale": "log10",
        "preferred_max_gap": 0.55,
        "domain": {"min": 1e-5, "max": 1e-2},
    },
    {
        "name": "weight_decay",
        "values": [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
        "scale": "log10",
        "special_values": [0.0],
        "preferred_max_gap": 0.55,
        "domain": {"min": 0.0, "max": 0.3},
    },
]
RESOURCE_LEVELS = [1, 4, 16, 64]
RESOURCE_RATIOS = [1, 4, 16, 64]
TARGETS = [
    {"target": "us-east5/v5p-32", "region": "us-east5", "tpu_slice": "v5p-32", "chips": 16},
    {"target": "us-east5/v5p-64", "region": "us-east5", "tpu_slice": "v5p-64", "chips": 32},
    {"target": "us-east5/v5p-128", "region": "us-east5", "tpu_slice": "v5p-128", "chips": 64},
    {"target": "us-central1/v5p-32", "region": "us-central1", "tpu_slice": "v5p-32", "chips": 16},
    {"target": "europe-west4/v6e-32", "region": "europe-west4", "tpu_slice": "v6e-32", "chips": 32},
]
TARGET_BY_ID = {target["target"]: target for target in TARGETS}
MAX_INFLIGHT_CHIPS = 64
OBSERVATION_INTERVAL = 0.25
WALL_TIME_LIMIT = 3 * 7 * 24
DEFAULT_FULL_EXPLOITATION_LEVEL = 64
RECOVERY = {
    "startup_relocation_timeout": 1.0,
    "same_target_restart_timeout": 2.0,
    "same_region_relocation_timeout": 4.0,
    "cross_region_restart_timeout": 48.0,
}
TRC_TARGET_PROFILES = {
    "us-east5/v5p-32": {
        "start_rate": 0.85,
        "end_rate": 0.85,
        "capacity_chips": 32,
        "outage_windows": [],
        "retry_probability": 0.08,
    },
    "us-east5/v5p-64": {
        "start_rate": 0.90,
        "end_rate": 1.80,
        "capacity_chips": 64,
        "outage_windows": [],
        "retry_probability": 0.13,
    },
    "us-east5/v5p-128": {
        "start_rate": 3.00,
        "end_rate": 1.60,
        "capacity_chips": 64,
        "outage_windows": [[0.00, 0.15], [0.75, 0.85]],
        "retry_probability": 0.22,
    },
    "us-central1/v5p-32": {
        "start_rate": 1.05,
        "end_rate": 1.05,
        "capacity_chips": 32,
        "outage_windows": [],
        "retry_probability": 0.10,
    },
    "europe-west4/v6e-32": {
        "start_rate": 0.65,
        "end_rate": 1.45,
        "capacity_chips": 32,
        "outage_windows": [],
        "retry_probability": 0.15,
    },
}
TRC_REGION_OUTAGES = {
    "us-east5": [[0.45, 0.50]],
    "us-central1": [[0.70, 0.75]],
    "europe-west4": [[0.20, 0.25]],
}
RETRY_REASONS = ["worker_failed", "preempted", "rpc_timeout", "tpu_environment_error"]
SCENARIOS = ("stable-interior", "predictable-off-grid")


def load_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def emit(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def stable_fraction(*values: Any) -> float:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def initial_axes(scenario: str) -> list[dict[str, Any]]:
    axes = copy.deepcopy(AXES)
    if scenario == "predictable-off-grid":
        axes[0]["values"] = axes[0]["values"][:5]
    return axes


def experiment_truth(seed: int, scenario: str) -> dict[str, Any]:
    if scenario == "predictable-off-grid":
        centers = [[1, 3], [3, 3], [5, 3], [6, 3]]
    else:
        center0 = [
            1 + int(stable_fraction(seed, scenario, "x") * 3),
            1 + int(stable_fraction(seed, scenario, "y") * 3),
        ]
        center1 = center0.copy()
        shifted_axis = int(stable_fraction(seed, scenario, "axis") * 2)
        center1[shifted_axis] = min(center1[shifted_axis] + 1, 5)
        centers = [center0, center1, center1.copy(), center1.copy()]
    return {"centers": centers}


def trc_truth(time_horizon: float) -> dict[str, Any]:
    return {
        "time_horizon": time_horizon,
        "targets": copy.deepcopy(TRC_TARGET_PROFILES),
        "region_outages": copy.deepcopy(TRC_REGION_OUTAGES),
    }


def hidden_truth(seed: int, scenario: str, time_horizon: float) -> dict[str, Any]:
    return {
        "seed": seed,
        "experiment": experiment_truth(seed, scenario),
        "trc": trc_truth(time_horizon),
    }


def _within_window(fraction: float, windows: list[list[float]]) -> bool:
    return any(start <= fraction < end for start, end in windows)


def target_conditions(state: dict[str, Any], target: str, time: float) -> dict[str, float]:
    truth = state["truth"]["trc"]
    profile = truth["targets"][target]
    fraction = min(max(time / truth["time_horizon"], 0.0), 1.0)
    normalized_rate = profile["start_rate"] + fraction * (profile["end_rate"] - profile["start_rate"])
    region = TARGET_BY_ID[target]["region"]
    unavailable = _within_window(fraction, truth["region_outages"].get(region, [])) or _within_window(
        fraction, profile["outage_windows"]
    )
    return {
        "capacity_chips": 0 if unavailable else profile["capacity_chips"],
        "normalized_rate": normalized_rate,
    }


def objective(state: dict[str, Any], rung: int, point: list[int]) -> float:
    center = state["truth"]["experiment"]["centers"][rung]
    declared_point = [
        AXES[axis_index]["values"].index(state["axes"][axis_index]["values"][coordinate])
        for axis_index, coordinate in enumerate(point)
    ]
    dx = declared_point[0] - center[0]
    dy = declared_point[1] - center[1]
    interaction = 0.025 * dx * dy
    roughness = 0.018 * (stable_fraction(state["truth"]["seed"], rung, point) - 0.5)
    rung_offset = 0.12 / (rung + 1)
    return round(0.31 * dx * dx + 0.39 * dy * dy + interaction + roughness + rung_offset, 6)


def point_values(state: dict[str, Any], point: list[int]) -> list[float]:
    return [float(axis["values"][coordinate]) for axis, coordinate in zip(state["axes"], point, strict=True)]


def point_indices(state: dict[str, Any], point: list[float]) -> list[int]:
    return [axis["values"].index(value) for axis, value in zip(state["axes"], point, strict=True)]


def convergence_input(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "axes": state["axes"],
        "resource_levels": RESOURCE_LEVELS,
        "objective": {"direction": "minimize"},
        "trials": [
            {"rung": trial["rung"], "point": point_values(state, trial["point"]), "objective": trial["objective"]}
            for trial in state["completed"]
        ],
    }


def convergence_result(state: dict[str, Any]) -> dict[str, Any]:
    result = check_convergence(convergence_input(state))
    for snapshot in result["snapshots"]:
        for dominant in snapshot["dominant_points"]:
            dominant["point"] = point_indices(state, dominant["point"])
    return result


def active_keys(state: dict[str, Any]) -> set[tuple[int, tuple[int, ...]]]:
    return {(trial["rung"], tuple(trial["point"])) for trial in state["active"]}


def completed_keys(state: dict[str, Any]) -> set[tuple[int, tuple[int, ...]]]:
    return {(trial["rung"], tuple(trial["point"])) for trial in state["completed"]}


def candidates(state: dict[str, Any]) -> list[dict[str, Any]]:
    occupied = active_keys(state) | completed_keys(state)
    shape = [len(axis["values"]) for axis in state["axes"]]
    return [
        {"candidate_id": f"r{rung}-x{x}-y{y}", "rung": rung, "point": [x, y]}
        for rung in range(len(RESOURCE_LEVELS))
        for x in range(shape[0])
        for y in range(shape[1])
        if (rung, (x, y)) not in occupied
    ]


def prediction_input(state: dict[str, Any]) -> dict[str, Any]:
    tool_candidates = [
        {**candidate, "point": point_values(state, candidate["point"])} for candidate in candidates(state)
    ]
    return convergence_input(state) | {"candidates": tool_candidates}


def prediction_result(state: dict[str, Any]) -> dict[str, Any]:
    result = predict_objectives(prediction_input(state))
    for prediction in result["predictions"]:
        prediction["point"] = point_indices(state, prediction["point"])
    return result


def target_input(state: dict[str, Any], rung: int) -> dict[str, Any]:
    return {
        "now": state["now"],
        "resource_levels": RESOURCE_LEVELS,
        "resource_ratios": RESOURCE_RATIOS,
        "wall_time_limit": state["time_horizon"],
        "max_inflight_chips": MAX_INFLIGHT_CHIPS,
        "current_rung": rung,
        "full_exploitation_rung": state["full_exploitation_rung"],
        "recovery": RECOVERY,
        "targets": TARGETS,
        "observations": state["observations"],
    }


def current_chips(state: dict[str, Any]) -> int:
    return sum(TARGET_BY_ID[trial["target"]]["chips"] for trial in state["active"])


def save_regional_state(trial: dict[str, Any]) -> None:
    trial["regional_runs"][trial["region"]] = {
        "regional_run_id": trial["regional_run_id"],
        "progress": trial["progress"],
        "wandb_registered": trial["wandb_registered"],
    }


def iris_job_id(trial: dict[str, Any]) -> str:
    return f"{trial['dispatch_id']}-attempt-{trial['submission_attempt']}"


def arm_retryable_failure(state: dict[str, Any], trial: dict[str, Any], *, force: bool = False) -> None:
    profile = TRC_TARGET_PROFILES[trial["target"]]
    roll = stable_fraction(
        state["truth"]["seed"],
        trial["trial_id"],
        trial["dispatch_id"],
        trial["submission_attempt"],
        trial["target"],
    )
    force = force or trial["must_fail_after_progress"]
    if trial["retry_count"] >= 2 or (not force and roll >= profile["retry_probability"]):
        trial["failure_at_progress"] = None
        return
    remaining = 1.0 - trial["progress"]
    trial["failure_at_progress"] = trial["progress"] + remaining * (0.15 + 0.25 * roll)


def append_observation(state: dict[str, Any], trial: dict[str, Any], state_name: str) -> None:
    state["observations"].append(
        {
            "trial_id": trial["trial_id"],
            "regional_run_id": trial["regional_run_id"],
            "dispatch_id": trial["dispatch_id"],
            "iris_job_id": iris_job_id(trial),
            "submission_attempt": trial["submission_attempt"],
            "rung": trial["rung"],
            "target": trial["target"],
            "state": state_name,
            "submitted_at": trial["dispatch_submitted_at"],
            "observed_at": state["now"],
            "wandb_run_id": trial["regional_run_id"] if trial["wandb_registered"] else None,
            "run_progress": trial["progress"],
        }
    )


def command_init(args: argparse.Namespace) -> None:
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    if args.time_horizon <= 0:
        raise ValueError("time_horizon must be positive")
    full_exploitation_rung = RESOURCE_LEVELS.index(args.full_exploitation_level)
    axes = initial_axes(args.scenario)
    state = {
        "scenario_id": hashlib.sha256(f"{args.scenario}:{seed}".encode()).hexdigest()[:12],
        "scenario": args.scenario,
        "axes": axes,
        "initial_axes": copy.deepcopy(axes),
        "truth": hidden_truth(seed, args.scenario, args.time_horizon),
        "time_horizon": args.time_horizon,
        "full_exploitation_rung": full_exploitation_rung,
        "now": 0.0,
        "next_trial_id": 1,
        "next_dispatch_id": 1,
        "next_regional_run_id": 1,
        "completed": [],
        "active": [],
        "observations": [],
        "decisions": [],
        "prediction_history": [],
        "target_history": [],
        "convergence_history": [],
        "retry_events": [],
    }
    save_state(args.state, state)
    emit(
        {
            "scenario_id": state["scenario_id"],
            "scenario": state["scenario"],
            "axes": state["axes"],
            "resource_levels": RESOURCE_LEVELS,
            "resource_ratios": RESOURCE_RATIOS,
            "targets": TARGETS,
            "observation_interval": OBSERVATION_INTERVAL,
            "wall_time_limit": state["time_horizon"],
            "target_time_horizon": state["time_horizon"],
            "full_exploitation_level": args.full_exploitation_level,
            "recovery": RECOVERY,
            "max_inflight_chips": MAX_INFLIGHT_CHIPS,
        }
    )


def command_status(args: argparse.Namespace) -> None:
    state = load_state(args.state)
    emit(
        {
            "time": state["now"],
            "completed_by_rung": dict(Counter(str(trial["rung"]) for trial in state["completed"])),
            "inflight_chips": current_chips(state),
            "available_chips": MAX_INFLIGHT_CHIPS - current_chips(state),
            "retry_events": len(state["retry_events"]),
            "axes": state["axes"],
            "active": [
                {
                    "trial_id": trial["trial_id"],
                    "rung": trial["rung"],
                    "point": trial["point"],
                    "target": trial["target"],
                    "progress": trial["progress"],
                    "state": trial["state"],
                    "submission_attempt": trial["submission_attempt"],
                    "iris_job_id": iris_job_id(trial),
                }
                for trial in state["active"]
            ],
        }
    )


def command_snapshot(args: argparse.Namespace) -> None:
    state = load_state(args.state)
    result = convergence_result(state)
    state["convergence_history"].append({"time": state["now"], "result": result})
    save_state(args.state, state)
    emit({"time": state["now"], **result})


def command_predict(args: argparse.Namespace) -> None:
    state = load_state(args.state)
    result = prediction_result(state)
    state["prediction_history"].append({"time": state["now"], "result": result})
    save_state(args.state, state)
    top_by_rung = {
        str(rung): sorted(
            (prediction for prediction in result["predictions"] if prediction["rung"] == rung),
            key=lambda prediction: prediction["rank_within_rung"] or 10**9,
        )[:10]
        for rung in range(len(RESOURCE_LEVELS))
    }
    emit(
        {
            "time": state["now"],
            "model_revision": result["model_revision"],
            "training_count": result["training_count"],
            "status": result["status"],
            "top_candidates_by_rung": top_by_rung,
        }
    )


def command_targets(args: argparse.Namespace) -> None:
    state = load_state(args.state)
    result = rank_targets(target_input(state, args.rung))
    state["target_history"].append({"time": state["now"], "rung": args.rung, "result": result})
    save_state(args.state, state)
    emit({"time": state["now"], "rung": args.rung, **result})


def command_launch(args: argparse.Namespace) -> None:
    state = load_state(args.state)
    actions = json.loads(args.actions)
    occupied = active_keys(state) | completed_keys(state)
    requested_chips = sum(TARGET_BY_ID[action["target"]]["chips"] for action in actions)
    if current_chips(state) + requested_chips > MAX_INFLIGHT_CHIPS:
        raise ValueError("launch exceeds max_inflight_chips")

    launched = []
    shape = [len(axis["values"]) for axis in state["axes"]]
    for action in actions:
        rung = int(action["rung"])
        point = [int(value) for value in action["point"]]
        target = action["target"]
        key = (rung, tuple(point))
        if (
            not 0 <= rung < len(RESOURCE_LEVELS)
            or len(point) != len(shape)
            or any(not 0 <= value < size for value, size in zip(point, shape, strict=True))
        ):
            raise ValueError("launch is outside the grid or resource levels")
        if target not in TARGET_BY_ID or key in occupied:
            raise ValueError("launch has an unknown target or duplicate logical trial")
        target_spec = TARGET_BY_ID[target]
        trial = {
            "trial_id": f"trial-{state['next_trial_id']:04d}",
            "regional_run_id": f"regional-{state['next_regional_run_id']:04d}",
            "dispatch_id": f"dispatch-{state['next_dispatch_id']:04d}",
            "rung": rung,
            "point": point,
            "target": target,
            "region": target_spec["region"],
            "progress": 0.0,
            "wandb_registered": False,
            "started_at": state["now"],
            "dispatch_submitted_at": state["now"],
            "submission_attempt": 1,
            "retry_count": 0,
            "retry_ready_at": state["now"],
            "must_fail_after_progress": state["next_trial_id"] == 1,
            "state": "submitted",
            "reason": action.get("reason", ""),
            "predicted_objective": action.get("predicted_objective"),
        }
        trial["regional_runs"] = {}
        arm_retryable_failure(state, trial, force=state["next_trial_id"] == 1)
        save_regional_state(trial)
        state["next_trial_id"] += 1
        state["next_regional_run_id"] += 1
        state["next_dispatch_id"] += 1
        state["active"].append(trial)
        state["decisions"].append({"time": state["now"], "action": "launch", **trial})
        append_observation(state, trial, "submitted")
        occupied.add(key)
        launched.append({key: trial[key] for key in ("trial_id", "rung", "point", "target", "dispatch_id")})
    save_state(args.state, state)
    emit(
        {
            "launched": launched,
            "inflight_chips": current_chips(state),
            "available_chips": MAX_INFLIGHT_CHIPS - current_chips(state),
        }
    )


def command_expand_grid(args: argparse.Namespace) -> None:
    state = load_state(args.state)
    action = json.loads(args.action)
    axis_name = str(action["axis"])
    axis_index = next(index for index, axis in enumerate(state["axes"]) if axis["name"] == axis_name)
    current_values = state["axes"][axis_index]["values"]
    declared_values = AXES[axis_index]["values"]
    requested_value = float(action["value"])
    available_values = [value for value in declared_values if value > current_values[-1] and value not in current_values]
    if requested_value not in available_values:
        raise ValueError("grid expansion must add a declared value above the current edge")
    preferred_value = available_values[0]
    deviation_reason = action.get("reason")
    if requested_value != preferred_value and (not isinstance(deviation_reason, str) or not deviation_reason.strip()):
        raise ValueError(f"skipping preferred next value {preferred_value} requires a reason")
    current_values.append(requested_value)
    decision = {
        "time": state["now"],
        "action": "expand_grid",
        "axis": axis_name,
        "value": requested_value,
        "deviation_reason": deviation_reason if requested_value != preferred_value else None,
    }
    state["decisions"].append(decision)
    save_state(args.state, state)
    emit({**decision, "axes": state["axes"]})


def command_relocate(args: argparse.Namespace) -> None:
    state = load_state(args.state)
    action = json.loads(args.action)
    trial = next(trial for trial in state["active"] if trial["trial_id"] == action["trial_id"])
    target = action["target"]
    if target not in TARGET_BY_ID or target == trial["target"]:
        raise ValueError("relocation requires a different known target")
    target_spec = TARGET_BY_ID[target]
    old_chips = int(TARGET_BY_ID[trial["target"]]["chips"])
    if current_chips(state) - old_chips + int(target_spec["chips"]) > MAX_INFLIGHT_CHIPS:
        raise ValueError("relocation exceeds max_inflight_chips")

    old_target = trial["target"]
    old_region = trial["region"]
    append_observation(state, trial, "stopped")
    if target_spec["region"] != old_region:
        save_regional_state(trial)
        prior_run = trial["regional_runs"].get(target_spec["region"])
        if prior_run is None:
            trial["regional_run_id"] = f"regional-{state['next_regional_run_id']:04d}"
            state["next_regional_run_id"] += 1
            trial["progress"] = 0.0
            trial["wandb_registered"] = False
        else:
            trial["regional_run_id"] = prior_run["regional_run_id"]
            trial["progress"] = prior_run["progress"]
            trial["wandb_registered"] = prior_run["wandb_registered"]
    trial["dispatch_id"] = f"dispatch-{state['next_dispatch_id']:04d}"
    state["next_dispatch_id"] += 1
    trial["target"] = target
    trial["region"] = target_spec["region"]
    trial["dispatch_submitted_at"] = state["now"]
    trial["submission_attempt"] = 1
    trial["retry_ready_at"] = state["now"]
    trial["state"] = "submitted"
    arm_retryable_failure(state, trial)
    save_regional_state(trial)
    state["decisions"].append(
        {
            "time": state["now"],
            "action": "relocate",
            "trial_id": trial["trial_id"],
            "from_target": old_target,
            "to_target": target,
            "regional_run_id": trial["regional_run_id"],
            "dispatch_id": trial["dispatch_id"],
            "progress": trial["progress"],
        }
    )
    append_observation(state, trial, "submitted")
    save_state(args.state, state)
    emit(
        {
            "trial_id": trial["trial_id"],
            "from_target": old_target,
            "to_target": target,
            "regional_run_id": trial["regional_run_id"],
            "dispatch_id": trial["dispatch_id"],
            "progress": trial["progress"],
            "inflight_chips": current_chips(state),
        }
    )


def command_advance(args: argparse.Namespace) -> None:
    state = load_state(args.state)
    completed_now = []
    retry_events_now = []
    for _ in range(args.steps):
        state["now"] = round(state["now"] + OBSERVATION_INTERVAL, 10)
        by_target: dict[str, list[dict[str, Any]]] = {}
        for trial in state["active"]:
            by_target.setdefault(trial["target"], []).append(trial)

        progressing = set()
        for target, trials in by_target.items():
            conditions = target_conditions(state, target, state["now"])
            used_chips = 0
            for trial in sorted(trials, key=lambda trial: trial["dispatch_submitted_at"]):
                if trial["retry_ready_at"] > state["now"]:
                    continue
                chips = int(TARGET_BY_ID[target]["chips"])
                if used_chips + chips <= conditions["capacity_chips"]:
                    progressing.add(trial["trial_id"])
                    used_chips += chips

        for trial in list(state["active"]):
            if trial["trial_id"] in progressing:
                trial["state"] = "running"
                trial["wandb_registered"] = True
                conditions = target_conditions(state, trial["target"], state["now"])
                jitter = 0.94 + 0.12 * stable_fraction(trial["target"], trial["rung"], trial["point"])
                normalized_work = conditions["normalized_rate"] * jitter * OBSERVATION_INTERVAL
                trial["progress"] = min(
                    trial["progress"] + normalized_work / RESOURCE_RATIOS[trial["rung"]],
                    1.0,
                )
                save_regional_state(trial)
                failure_at_progress = trial["failure_at_progress"]
                if failure_at_progress is not None and trial["progress"] >= failure_at_progress:
                    append_observation(state, trial, "failed")
                    failed_job_id = iris_job_id(trial)
                    reason_index = int(stable_fraction(state["truth"]["seed"], failed_job_id) * len(RETRY_REASONS))
                    event = {
                        "time": state["now"],
                        "trial_id": trial["trial_id"],
                        "dispatch_id": trial["dispatch_id"],
                        "iris_job_id": failed_job_id,
                        "target": trial["target"],
                        "progress": trial["progress"],
                        "reason": RETRY_REASONS[reason_index],
                    }
                    state["retry_events"].append(event)
                    retry_events_now.append(event)
                    trial["retry_count"] += 1
                    trial["must_fail_after_progress"] = False
                    trial["submission_attempt"] += 1
                    trial["retry_ready_at"] = state["now"] + OBSERVATION_INTERVAL
                    trial["state"] = "retrying"
                    arm_retryable_failure(state, trial)
                    append_observation(state, trial, "retrying")
                    continue
            if trial["progress"] >= 1:
                trial["state"] = "succeeded"
                append_observation(state, trial, "succeeded")
                state["active"].remove(trial)
                trial["objective"] = objective(state, trial["rung"], trial["point"])
                trial["finished_at"] = state["now"]
                state["completed"].append(trial)
                completed_now.append(
                    {
                        "trial_id": trial["trial_id"],
                        "rung": trial["rung"],
                        "point": trial["point"],
                        "target": trial["target"],
                        "objective": trial["objective"],
                        "retry_count": trial["retry_count"],
                    }
                )
            else:
                if trial["retry_ready_at"] <= state["now"] and trial["trial_id"] not in progressing:
                    trial["state"] = "submitted"
                append_observation(state, trial, trial["state"])
    save_state(args.state, state)
    emit(
        {
            "time": state["now"],
            "completed": completed_now,
            "retry_events": retry_events_now,
            "inflight_chips": current_chips(state),
            "available_chips": MAX_INFLIGHT_CHIPS - current_chips(state),
            "active": [
                {
                    "trial_id": trial["trial_id"],
                    "rung": trial["rung"],
                    "point": trial["point"],
                    "target": trial["target"],
                    "progress": round(trial["progress"], 6),
                    "state": trial["state"],
                    "submission_attempt": trial["submission_attempt"],
                }
                for trial in state["active"]
            ],
        }
    )


def command_report(args: argparse.Namespace) -> None:
    state = load_state(args.state)
    convergence = convergence_result(state)
    if not convergence["all_rungs_converged"]:
        raise ValueError("hidden report is available only after strict all-rung convergence")
    emit(
        {
            "scenario_id": state["scenario_id"],
            "scenario": state["scenario"],
            "time": state["now"],
            "actual_centers": state["truth"]["experiment"]["centers"],
            "trc_truth": state["truth"]["trc"],
            "initial_axes": state["initial_axes"],
            "final_axes": state["axes"],
            "grid_expansions": [decision for decision in state["decisions"] if decision["action"] == "expand_grid"],
            "convergence": convergence,
            "completed_trials": len(state["completed"]),
            "completed_by_rung": dict(Counter(str(trial["rung"]) for trial in state["completed"])),
            "decision_count": len(state["decisions"]),
            "retry_events": state["retry_events"],
            "full_exploitation_level": RESOURCE_LEVELS[state["full_exploitation_rung"]],
            "target_time_horizon": state["time_horizon"],
        }
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init")
    init.add_argument("state", type=Path)
    init.add_argument("--seed", type=int)
    init.add_argument("--scenario", choices=SCENARIOS, default="stable-interior")
    init.add_argument("--time-horizon", type=float, default=WALL_TIME_LIMIT)
    init.add_argument(
        "--full-exploitation-level",
        type=int,
        choices=RESOURCE_LEVELS,
        default=DEFAULT_FULL_EXPLOITATION_LEVEL,
    )
    init.set_defaults(handler=command_init)

    for name, handler in (
        ("status", command_status),
        ("snapshot", command_snapshot),
        ("predict", command_predict),
        ("report", command_report),
    ):
        command = commands.add_parser(name)
        command.add_argument("state", type=Path)
        command.set_defaults(handler=handler)

    targets = commands.add_parser("targets")
    targets.add_argument("state", type=Path)
    targets.add_argument("rung", type=int)
    targets.set_defaults(handler=command_targets)

    launch = commands.add_parser("launch")
    launch.add_argument("state", type=Path)
    launch.add_argument("actions")
    launch.set_defaults(handler=command_launch)

    expand_grid = commands.add_parser("expand-grid")
    expand_grid.add_argument("state", type=Path)
    expand_grid.add_argument("action")
    expand_grid.set_defaults(handler=command_expand_grid)

    relocate = commands.add_parser("relocate")
    relocate.add_argument("state", type=Path)
    relocate.add_argument("action")
    relocate.set_defaults(handler=command_relocate)

    advance = commands.add_parser("advance")
    advance.add_argument("state", type=Path)
    advance.add_argument("--steps", type=int, default=1)
    advance.set_defaults(handler=command_advance)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        args.handler(args)
    except (KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as error:
        json.dump({"error": str(error)}, sys.stderr, sort_keys=True)
        sys.stderr.write("\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
