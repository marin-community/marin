#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic evidence calculations for adaptive sweeps."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

Point = tuple[int, ...]
ACTIVE_STATES = {"submitted", "running", "retrying"}
DEFAULT_MAX_INFLIGHT_CHIPS = 64


def _stable_id(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _axes(request: Mapping[str, Any]) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
    axes = [dict(axis) for axis in request["axes"]]
    if not axes or any(not axis.get("values") for axis in axes):
        raise ValueError("axes must contain at least one non-empty values list")
    names = [axis.get("name") for axis in axes]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("axis names must be non-empty and unique")

    for axis in axes:
        name = axis["name"]
        values = [float(value) for value in axis["values"]]
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"axis {name} values must be finite")
        if any(left >= right for left, right in itertools.pairwise(values)):
            raise ValueError(f"axis {name} values must be strictly increasing")

        domain = axis.get("domain")
        if not isinstance(domain, Mapping) or "min" not in domain or "max" not in domain:
            raise ValueError(f"axis {name} requires domain.min and domain.max")
        domain_min = float(domain["min"])
        domain_max = float(domain["max"])
        if not math.isfinite(domain_min) or not math.isfinite(domain_max) or domain_min > domain_max:
            raise ValueError(f"axis {name} has an invalid domain")
        if values[0] < domain_min or values[-1] > domain_max:
            raise ValueError(f"axis {name} values must lie within its domain")

        axis["values"] = values
        axis["domain"] = {"min": domain_min, "max": domain_max}
    return axes, tuple(len(axis["values"]) for axis in axes)


def _resource_levels(request: Mapping[str, Any]) -> list[float]:
    levels = [float(value) for value in request["resource_levels"]]
    if not levels or any(not math.isfinite(value) or value <= 0 for value in levels):
        raise ValueError("resource_levels must be non-empty, finite, and positive")
    if any(left >= right for left, right in itertools.pairwise(levels)):
        raise ValueError("resource_levels must be strictly increasing")
    return levels


def _resource_ratios(request: Mapping[str, Any], expected_count: int) -> list[float]:
    ratios = [float(value) for value in request["resource_ratios"]]
    if len(ratios) != expected_count or any(not math.isfinite(value) or value <= 0 for value in ratios):
        raise ValueError("resource_ratios must match resource_levels and be finite and positive")
    if any(left >= right for left, right in itertools.pairwise(ratios)):
        raise ValueError("resource_ratios must be strictly increasing")
    return ratios


def _point(value: Sequence[Any], axes: Sequence[Mapping[str, Any]]) -> Point:
    coordinates = tuple(float(coordinate) for coordinate in value)
    if len(coordinates) != len(axes):
        raise ValueError(f"point {coordinates} has {len(coordinates)} dimensions, expected {len(axes)}")
    if any(not math.isfinite(coordinate) for coordinate in coordinates):
        raise ValueError(f"point {coordinates} must contain finite values")

    indices = []
    for coordinate, axis in zip(coordinates, axes, strict=True):
        try:
            indices.append(axis["values"].index(coordinate))
        except ValueError as error:
            raise ValueError(f"point value {coordinate} is not in axis {axis['name']}") from error
    return tuple(indices)


def _point_values(point: Point, axes: Sequence[Mapping[str, Any]]) -> list[float]:
    return [float(axis["values"][index]) for index, axis in zip(point, axes, strict=True)]


def _completed_trials(
    request: Mapping[str, Any], axes: Sequence[Mapping[str, Any]], resource_count: int
) -> dict[tuple[int, Point], float]:
    completed: dict[tuple[int, Point], float] = {}
    for trial in request.get("trials", []):
        rung = int(trial["rung"])
        if rung < 0 or rung >= resource_count:
            raise ValueError(f"invalid rung {rung}")
        point = _point(trial["point"], axes)
        key = (rung, point)
        if key in completed:
            raise ValueError(f"duplicate completed trial at rung {rung}, point {point}")
        objective = float(trial["objective"])
        if not math.isfinite(objective):
            raise ValueError("completed trials require a finite objective")
        completed[key] = objective
    return completed


def _hard_boundary(axis: Mapping[str, Any], coordinate: int, step: int) -> bool:
    values = axis["values"]
    if step < 0:
        return coordinate == 0 and values[0] == axis["domain"]["min"]
    return coordinate == len(values) - 1 and values[-1] == axis["domain"]["max"]


def check_convergence(request: Mapping[str, Any]) -> dict[str, Any]:
    """Check strict one-step neighbor dominance on every resource rung."""
    axes, shape = _axes(request)
    resources = _resource_levels(request)
    direction = request["objective"]["direction"]
    if direction not in {"minimize", "maximize"}:
        raise ValueError("objective.direction must be minimize or maximize")
    completed = _completed_trials(request, axes, len(resources))

    snapshots = []
    for rung, resource in enumerate(resources):
        rung_trials = {point: objective for (trial_rung, point), objective in completed.items() if trial_rung == rung}
        dominant_points = []
        for point, objective in sorted(rung_trials.items()):
            dominates = True
            for axis_index, axis in enumerate(axes):
                for step in (-1, 1):
                    neighbor_index = point[axis_index] + step
                    if not 0 <= neighbor_index < shape[axis_index]:
                        if not _hard_boundary(axis, point[axis_index], step):
                            dominates = False
                        continue
                    neighbor = list(point)
                    neighbor[axis_index] = neighbor_index
                    neighbor_objective = rung_trials.get(tuple(neighbor))
                    if neighbor_objective is None:
                        dominates = False
                    elif direction == "minimize" and objective > neighbor_objective:
                        dominates = False
                    elif direction == "maximize" and objective < neighbor_objective:
                        dominates = False
            if dominates:
                dominant_points.append({"point": _point_values(point, axes), "objective": objective})
        snapshots.append(
            {
                "rung": rung,
                "resource": resource,
                "converged": bool(dominant_points),
                "dominant_points": dominant_points,
            }
        )

    return {
        "revision": _stable_id(
            {
                "axes": axes,
                "resources": resources,
                "direction": direction,
                "trials": sorted((rung, point, objective) for (rung, point), objective in completed.items()),
            }
        ),
        "snapshots": snapshots,
        "unconverged_rungs": [snapshot["rung"] for snapshot in snapshots if not snapshot["converged"]],
        "all_rungs_converged": all(snapshot["converged"] for snapshot in snapshots),
    }


def _features(point: Point, rung: int, shape: Sequence[int], resource_count: int) -> list[float]:
    coordinates = [index / (size - 1) if size > 1 else 0.0 for index, size in zip(point, shape, strict=True)]
    coordinates.append(rung / (resource_count - 1) if resource_count > 1 else 0.0)
    return coordinates


def predict_objectives(request: Mapping[str, Any]) -> dict[str, Any]:
    """Fit one gradient-boosted regressor and rank unobserved candidates per rung."""
    axes, shape = _axes(request)
    resources = _resource_levels(request)
    direction = request["objective"]["direction"]
    if direction not in {"minimize", "maximize"}:
        raise ValueError("objective.direction must be minimize or maximize")
    completed = _completed_trials(request, axes, len(resources))

    candidates = []
    candidate_ids = set()
    for candidate in request.get("candidates", []):
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate candidate_id {candidate_id}")
        candidate_ids.add(candidate_id)
        rung = int(candidate["rung"])
        if rung < 0 or rung >= len(resources):
            raise ValueError(f"invalid candidate rung {rung}")
        point = _point(candidate["point"], axes)
        if (rung, point) in completed:
            raise ValueError(f"candidate {candidate_id} is already completed")
        candidates.append({"candidate_id": candidate_id, "rung": rung, "point": point})

    if not completed:
        return {
            "model_revision": None,
            "training_count": 0,
            "status": "unavailable",
            "predictions": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "rung": candidate["rung"],
                    "point": _point_values(candidate["point"], axes),
                    "predicted_objective": None,
                    "rank_within_rung": None,
                }
                for candidate in candidates
            ],
        }

    ensemble = importlib.import_module("sklearn.ensemble")
    model = ensemble.GradientBoostingRegressor(random_state=0)
    ordered_trials = sorted(completed.items())
    train_features = [_features(point, rung, shape, len(resources)) for (rung, point), _ in ordered_trials]
    train_objectives = [objective for _, objective in ordered_trials]
    model.fit(train_features, train_objectives)

    predicted_values = (
        model.predict(
            [_features(candidate["point"], candidate["rung"], shape, len(resources)) for candidate in candidates]
        )
        if candidates
        else []
    )
    predictions = [
        {
            "candidate_id": candidate["candidate_id"],
            "rung": candidate["rung"],
            "point": _point_values(candidate["point"], axes),
            "predicted_objective": float(prediction),
        }
        for candidate, prediction in zip(candidates, predicted_values, strict=True)
    ]
    for rung in range(len(resources)):
        rung_predictions = [prediction for prediction in predictions if prediction["rung"] == rung]
        rung_predictions.sort(
            key=lambda prediction: (
                prediction["predicted_objective"] if direction == "minimize" else -prediction["predicted_objective"],
                prediction["candidate_id"],
            )
        )
        rank = 0
        previous_objective = None
        for prediction in rung_predictions:
            if previous_objective is None or prediction["predicted_objective"] != previous_objective:
                rank += 1
                previous_objective = prediction["predicted_objective"]
            prediction["rank_within_rung"] = rank

    return {
        "model_revision": _stable_id({"axes": request["axes"], "resources": resources, "trials": ordered_trials}),
        "training_count": len(completed),
        "status": "fit",
        "predictions": sorted(predictions, key=lambda prediction: prediction["candidate_id"]),
    }


def _targets(request: Mapping[str, Any], max_chips: int) -> list[dict[str, Any]]:
    targets = []
    seen = set()
    for raw_target in request["targets"]:
        target = dict(raw_target)
        target_id = str(target["target"])
        if target_id in seen:
            raise ValueError(f"duplicate target {target_id}")
        seen.add(target_id)
        chips = int(target["chips"])
        if chips != float(target["chips"]) or chips <= 0 or chips > max_chips:
            raise ValueError(f"target {target_id} has invalid chips")
        target["target"] = target_id
        target["chips"] = chips
        targets.append(target)
    if not targets:
        raise ValueError("targets must be non-empty")
    return targets


def _observation_groups(request: Mapping[str, Any], target_ids: set[str], now: float) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for raw_observation in request.get("observations", []):
        observation = dict(raw_observation)
        target = str(observation["target"])
        if target not in target_ids:
            raise ValueError(f"observation uses unknown target {target}")
        raw_progress = observation.get("run_progress")
        progress = None if raw_progress is None else float(raw_progress)
        if progress is not None and (not math.isfinite(progress) or not 0 <= progress <= 1):
            raise ValueError("run_progress must be null or a finite value between zero and one")
        submitted = float(observation["submitted_at"])
        observed = float(observation["observed_at"])
        if submitted > observed or observed > now:
            raise ValueError("observation timestamps must satisfy submitted_at <= observed_at <= now")
        observation["target"] = target
        observation["run_progress"] = progress
        observation["submitted_at"] = submitted
        observation["observed_at"] = observed
        wandb_run_id = observation.get("wandb_run_id")
        if wandb_run_id is not None and (not isinstance(wandb_run_id, str) or not wandb_run_id):
            raise ValueError("wandb_run_id must be a non-empty string or null")
        observation["wandb_run_id"] = wandb_run_id
        groups.setdefault(str(observation["dispatch_id"]), []).append(observation)
    for observations in groups.values():
        observations.sort(key=lambda observation: (observation["observed_at"], observation["submitted_at"]))
        if len({observation["target"] for observation in observations}) != 1:
            raise ValueError("a dispatch must remain on one target")
    return groups


def _cosine_weight_integral(age: float, wall_time_limit: float) -> float:
    age = min(max(age, 0.0), wall_time_limit)
    return age / 2 + wall_time_limit * math.sin(math.pi * age / wall_time_limit) / (2 * math.pi)


def _age_weighted_rate(
    observations: Sequence[Mapping[str, Any]],
    now: float,
    wall_time_limit: float,
    resource_ratios: Sequence[float],
) -> tuple[float, float] | None:
    weighted_work = 0.0
    weighted_elapsed = 0.0
    for start, end in itertools.pairwise(observations):
        start_time = float(start["observed_at"])
        end_time = float(end["observed_at"])
        elapsed = end_time - start_time
        if elapsed <= 0:
            continue
        if start["run_progress"] is None or end["run_progress"] is None:
            continue
        age_start = now - start_time
        age_end = now - end_time
        interval_weight = _cosine_weight_integral(age_start, wall_time_limit) - _cosine_weight_integral(
            age_end, wall_time_limit
        )
        if interval_weight <= 0:
            continue
        rung = int(end["rung"])
        if rung < 0 or rung >= len(resource_ratios):
            raise ValueError(f"invalid observation rung {rung}")
        gained_progress = max(float(end["run_progress"]) - float(start["run_progress"]), 0.0)
        normalized_rate = gained_progress * float(resource_ratios[rung]) / elapsed
        weighted_work += normalized_rate * interval_weight
        weighted_elapsed += interval_weight
    if weighted_elapsed <= 0:
        return None
    return weighted_work, weighted_elapsed


def _stagnation(
    request: Mapping[str, Any],
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    now: float,
    ranked_targets: Sequence[Mapping[str, Any]],
    available_chips: int,
) -> list[dict[str, Any]]:
    thresholds = request["stagnation"]
    initial_wandb_timeout = float(thresholds["initial_wandb_timeout"])
    progress_stall_timeout = float(thresholds["progress_stall_timeout"])
    cross_region_restart_timeout = float(thresholds["cross_region_restart_timeout"])
    if initial_wandb_timeout <= 0 or progress_stall_timeout <= 0:
        raise ValueError("initial_wandb_timeout and progress_stall_timeout must be positive")
    if cross_region_restart_timeout <= progress_stall_timeout:
        raise ValueError("cross_region_restart_timeout must exceed progress_stall_timeout")

    by_regional_run: dict[str, list[Mapping[str, Any]]] = {}
    for observations in groups.values():
        for observation in observations:
            by_regional_run.setdefault(str(observation["regional_run_id"]), []).append(observation)

    results = []
    targets_by_id = {str(target["target"]): target for target in ranked_targets}
    for regional_run_id, observations in by_regional_run.items():
        observations.sort(key=lambda observation: observation["observed_at"])
        latest = observations[-1]
        if latest["state"] not in ACTIVE_STATES:
            continue
        dispatch_starts = {
            dispatch_id: min(
                float(observation["submitted_at"])
                for observation in observations
                if observation["dispatch_id"] == dispatch_id
            )
            for dispatch_id in {str(observation["dispatch_id"]) for observation in observations}
        }
        current_target = targets_by_id[str(latest["target"])]
        replacement_budget = available_chips + int(current_target["chips"])
        same_region_targets = [
            str(target["target"])
            for target in ranked_targets
            if target["target"] != current_target["target"]
            and target["region"] == current_target["region"]
            and int(target["chips"]) <= replacement_budget
        ]
        cross_region_targets = [
            str(target["target"])
            for target in ranked_targets
            if target["region"] != current_target["region"] and int(target["chips"]) <= replacement_budget
        ]
        eligible_targets = []
        blocked_reason = None
        registered = [observation for observation in observations if observation["wandb_run_id"] is not None]
        current_dispatch_started = dispatch_starts[str(latest["dispatch_id"])]
        if not registered:
            condition = "awaiting_wandb"
            condition_since = min(dispatch_starts.values())
            same_region_moves_since_condition = len(dispatch_starts) - 1
            current_dispatch_inactive_for = max(now - current_dispatch_started, 0.0)
            if (
                now - condition_since >= cross_region_restart_timeout
                and same_region_moves_since_condition > 0
                and current_dispatch_inactive_for >= initial_wandb_timeout
            ):
                if cross_region_targets:
                    action = "cross_region_restart"
                    eligible_targets = cross_region_targets
                else:
                    action = "observe"
                    blocked_reason = "no chip-feasible cross-region target"
            elif current_dispatch_inactive_for >= initial_wandb_timeout:
                if same_region_targets:
                    action = "initial_same_region_move"
                    eligible_targets = same_region_targets
                else:
                    action = "observe"
                    blocked_reason = "no chip-feasible alternate target in region"
            else:
                action = "observe"
        else:
            condition = "wandb_registered"
            condition_since = float(registered[0]["observed_at"])
            best_progress = None
            for observation in registered:
                progress = observation["run_progress"]
                if progress is not None and (best_progress is None or progress > best_progress):
                    best_progress = progress
                    condition_since = float(observation["observed_at"])
            same_region_moves_since_condition = sum(start > condition_since for start in dispatch_starts.values())
            inactive_for = max(now - condition_since, 0.0)
            current_dispatch_inactive_for = max(now - max(condition_since, current_dispatch_started), 0.0)
            if (
                inactive_for >= cross_region_restart_timeout
                and same_region_moves_since_condition > 0
                and current_dispatch_inactive_for >= progress_stall_timeout
            ):
                if cross_region_targets:
                    action = "cross_region_restart"
                    eligible_targets = cross_region_targets
                else:
                    action = "observe"
                    blocked_reason = "no chip-feasible cross-region target"
            elif current_dispatch_inactive_for >= progress_stall_timeout:
                if same_region_targets:
                    action = "stalled_same_region_move"
                    eligible_targets = same_region_targets
                else:
                    action = "observe"
                    blocked_reason = "no chip-feasible alternate target in region"
            else:
                action = "observe"
        results.append(
            {
                "regional_run_id": regional_run_id,
                "trial_id": latest["trial_id"],
                "current_dispatch_id": latest["dispatch_id"],
                "current_target": latest["target"],
                "condition": condition,
                "inactive_for": max(now - condition_since, 0.0),
                "same_region_moves_since_condition": same_region_moves_since_condition,
                "eligible_action": action,
                "eligible_targets": eligible_targets,
                "blocked_reason": blocked_reason,
            }
        )
    return sorted(results, key=lambda result: result["regional_run_id"])


def rank_targets(request: Mapping[str, Any]) -> dict[str, Any]:
    """Estimate normalized target throughput and expose an exploration-ranked placement pool."""
    resources = _resource_levels(request)
    resource_ratios = _resource_ratios(request, len(resources))
    now = float(request["now"])
    wall_time_limit = float(request["wall_time_limit"])
    if wall_time_limit <= 0:
        raise ValueError("wall_time_limit must be positive")
    raw_max_chips = request.get("max_inflight_chips", DEFAULT_MAX_INFLIGHT_CHIPS)
    max_chips = int(raw_max_chips)
    if max_chips != float(raw_max_chips) or max_chips <= 0:
        raise ValueError("max_inflight_chips must be a positive integer")
    targets = _targets(request, max_chips)
    target_ids = {target["target"] for target in targets}
    groups = _observation_groups(request, target_ids, now)

    latest_by_dispatch = {dispatch_id: observations[-1] for dispatch_id, observations in groups.items()}
    inflight_chips = sum(
        next(target["chips"] for target in targets if target["target"] == observation["target"])
        for observation in latest_by_dispatch.values()
        if observation["state"] in ACTIVE_STATES
    )
    if inflight_chips > max_chips:
        raise ValueError("active dispatches exceed max_inflight_chips")
    available_chips = max_chips - inflight_chips

    rates_by_target: dict[str, list[tuple[float, float]]] = {target["target"]: [] for target in targets}
    active_rates_by_target: dict[str, list[float]] = {target["target"]: [] for target in targets}
    for observations in groups.values():
        sample = _age_weighted_rate(observations, now, wall_time_limit, resource_ratios)
        target_id = observations[-1]["target"]
        if sample is not None:
            rates_by_target[target_id].append(sample)
        if observations[-1]["state"] in ACTIVE_STATES:
            active_rates_by_target[target_id].append(0.0 if sample is None else sample[0] / sample[1])

    ranked = []
    for policy_order, target in enumerate(targets):
        samples = rates_by_target[target["target"]]
        normalized_work = sum(sample[0] for sample in samples)
        elapsed = sum(sample[1] for sample in samples)
        mean_throughput = normalized_work / elapsed if elapsed > 0 else None
        active_rates = active_rates_by_target[target["target"]]
        throughput = min(active_rates) if active_rates else mean_throughput
        ranked.append(
            {
                **target,
                "normalized_progress_throughput": throughput,
                "mean_normalized_progress_throughput": mean_throughput,
                "evidence_count": len(samples),
                "status": "observed" if len(samples) >= 3 else "provisional" if samples else "unknown",
                "feasible_for_available_chips": target["chips"] <= available_chips,
                "_policy_order": policy_order,
            }
        )
    ranked.sort(
        key=lambda target: (
            target["normalized_progress_throughput"] is None,
            -(target["normalized_progress_throughput"] or 0.0),
            target["_policy_order"],
        )
    )
    for rank, target in enumerate(ranked, start=1):
        target["throughput_rank"] = rank
        del target["_policy_order"]

    current_rung = int(request["current_rung"])
    full_exploitation_rung = int(request["full_exploitation_rung"])
    if not 0 <= current_rung < len(resources) or not 0 <= full_exploitation_rung < len(resources):
        raise ValueError("current_rung and full_exploitation_rung must index resource_levels")
    if full_exploitation_rung == 0 or current_rung >= full_exploitation_rung:
        exploration_fraction = 0.0
    else:
        exploration_fraction = (full_exploitation_rung - current_rung) / full_exploitation_rung
    feasible = [target for target in ranked if target["feasible_for_available_chips"]]
    exploration_depth = 0 if not feasible else 1 + math.floor(exploration_fraction * (len(feasible) - 1))

    return {
        "max_inflight_chips": max_chips,
        "current_inflight_chips": inflight_chips,
        "available_chips": available_chips,
        "exploration_fraction": exploration_fraction,
        "exploration_depth": exploration_depth,
        "selection_pool": [target["target"] for target in feasible[:exploration_depth]],
        "targets": ranked,
        "stagnation": _stagnation(request, groups, now, ranked, available_chips),
    }


OPERATIONS = {
    "check-convergence": check_convergence,
    "predict-objectives": predict_objectives,
    "rank-targets": rank_targets,
}


def run_request(operation: str, request: Mapping[str, Any]) -> dict[str, Any]:
    return OPERATIONS[operation](request)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=OPERATIONS)
    parser.add_argument("request", help="JSON request path, or - for stdin")
    args = parser.parse_args()
    try:
        if args.request == "-":
            request = json.load(sys.stdin)
        else:
            with Path(args.request).open(encoding="utf-8") as source:
                request = json.load(source)
        result = run_request(args.operation, request)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        json.dump({"error": str(error)}, sys.stderr, sort_keys=True)
        sys.stderr.write("\n")
        return 2
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
