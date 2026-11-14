# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate calibration artifacts and price the approved primary trajectories."""

import argparse
import heapq
import json
import math
import statistics
from dataclasses import asdict

from iris.cli.connect import open_iris_client
from iris.cluster.types import JobName
from iris.resources.state import JobState
from rigging.filesystem.buckets import filesystem_for
from rigging.timing import Timestamp

from experiments.post_training.snowball_isoflop.budget import allocated_seconds
from experiments.post_training.snowball_isoflop.config import (
    PREFIX,
    RUNTIME_FIELDS,
    VERSION,
    build_pod,
    calibration_cells,
    primary_cells,
)
from experiments.post_training.snowball_isoflop.measurements import read_json, training_history, warm_measurement


def trajectory_seconds(observation: dict, steps: int) -> float:
    return (
        observation["startup_and_finalization_seconds"]
        + steps * observation["steady_inclusive_seconds"]
        + math.floor(steps / 15) * observation["periodic_eval_seconds"]
    )


def scheduled_hours(durations: list[float], initial_nodes: int, full_release_hours: float = 0) -> float:
    """Simulate longest-first placement with the remaining nodes released later."""
    available = [0.0] * initial_nodes + [full_release_hours * 3600] * (8 - initial_nodes)
    heapq.heapify(available)
    last_finish = 0.0
    for duration in sorted(durations, reverse=True):
        earliest = heapq.heappop(available)
        finished = earliest + duration
        heapq.heappush(available, finished)
        last_finish = max(last_finish, finished)
    return last_finish / 3600


def calibration_report(prior_h100_hours: float) -> dict:
    observations = {}
    with open_iris_client(cluster_name="marin", workspace=None) as client:
        for cell in calibration_cells():
            root = f"{PREFIX}/{VERSION}/{cell.name}"
            dispatch = read_json(f"{root}/dispatch.json")
            job_name = JobName.from_wire(dispatch["job_id"])
            if not job_name.to_wire().startswith("/benfeuer/snowball-isoflop-cal-"):
                raise ValueError("Calibration dispatch points outside this experiment")
            status = client.job(job_name).status()
            if status.state != JobState.SUCCEEDED:
                raise ValueError(f"Calibration has not succeeded: {cell.name}: {status.state}")
            tasks = client.list_tasks(job_name)
            elapsed = sum(allocated_seconds(task, Timestamp.now(), status.started_at) for task in tasks)
            gate = read_json(f"{root}/initial-eval-match.json")
            consumed = read_json(f"{root}/consumed-config.json")
            runtime = {key: consumed["dispatch_environment"].get(key) for key in RUNTIME_FIELDS}
            expected_environment = build_pod(cell).env_vars or {}
            if runtime != {key: expected_environment.get(key) for key in RUNTIME_FIELDS}:
                raise ValueError(f"Calibration runtime differs from the approved recipe: {cell.name}")
            if not gate["passed"] or not gate["raw_match"]["passed"]:
                raise ValueError(f"Dual initialization gate did not pass for {cell.name}")
            checkpoint = read_json(f"{root}/checkpoints/step-{cell.stop_step - 1}/metadata.json")
            if checkpoint["step"] != cell.stop_step - 1 or checkpoint["is_temporary"]:
                raise ValueError(f"Missing permanent endpoint for {cell.name}")
            state, rows = training_history(cell.name, cell.stop_step)
            training = [row for row in rows if "train/loss" in row]
            if state != "finished" or {int(row["global_step"]) for row in training} != set(range(cell.stop_step)):
                raise ValueError(f"Incomplete training history for {cell.name}")
            measurement = warm_measurement(rows, cell.size, 8)
            if (
                measurement.minimum_memory_headroom_fraction is None
                or measurement.minimum_memory_headroom_fraction < 0.1
            ):
                raise ValueError(f"Calibration lacks safe measured memory headroom: {cell.name}")
            warm = [row for row in training if int(row["global_step"]) >= 3]
            evaluations = [float(row["eval/total_time"]) for row in rows if "eval/total_time" in row]
            if not evaluations:
                raise ValueError(f"Calibration did not measure SFT validation: {cell.name}")
            eval_seconds = max(evaluations)
            steady = statistics.median(
                sum(
                    float(row[key])
                    for key in (
                        "throughput/duration",
                        "throughput/loading_time",
                        "throughput/hook_time",
                        "throughput/gc_time",
                    )
                )
                - (float(row.get("eval/total_time", 0)) if (int(row["global_step"]) + 1) % 15 == 0 else 0)
                for row in warm
            )
            overhead = max(0, elapsed - cell.stop_step * steady - math.floor(cell.stop_step / 15) * eval_seconds)
            observations[f"{cell.size}-{cell.mask}"] = {
                "cell": cell.name,
                "output": root,
                "warm": asdict(measurement),
                "steady_inclusive_seconds": steady,
                "periodic_eval_seconds": eval_seconds,
                "startup_and_finalization_seconds": overhead,
                "recorded_h100_hours": elapsed * 8 / 3600,
                "initial_eval": gate,
                "runtime_environment": runtime,
                "checkpoint": checkpoint,
                "dropless_basis": "EP1 Sonic local expert execution has no bounded EP transport queues",
            }
    runs = [
        {"cell": asdict(cell), "seconds": trajectory_seconds(observations[f"{cell.size}-{cell.mask}"], cell.stop_step)}
        for cell in primary_cells()
    ]
    training_hours = sum(run["seconds"] for run in runs) * 8 / 3600
    calibration_hours = sum(row["recorded_h100_hours"] for row in observations.values())
    evaluation_reserve = 48.0
    contingency = training_hours * 0.15
    projected = prior_h100_hours + calibration_hours + training_hours + contingency + evaluation_reserve
    durations = [run["seconds"] for run in runs]
    wall = scheduled_hours(durations, 5, full_release_hours=22)
    return {
        "version": VERSION,
        "all_six_cells_valid": True,
        "observations": observations,
        "runtime_environment_by_cell": {key: row["runtime_environment"] for key, row in observations.items()},
        "runs": runs,
        "prior_h100_hours": prior_h100_hours,
        "calibration_h100_hours": calibration_hours,
        "primary_training_h100_hours": training_hours,
        "timing_and_checkpoint_contingency_h100_hours": contingency,
        "endpoint_evaluation_reserve_h100_hours": evaluation_reserve,
        "projected_primary_gpu_hours": projected,
        "eight_node_primary_hours": scheduled_hours(durations, 8),
        "five_then_eight_node_primary_hours": wall,
        "shared_release_assumption": "three B/C nodes release to A by22hours; update from B/C measurements",
        "expansion_gate_passed": projected <= 2200 and wall <= 40,
        "inclusive_step_seconds": {
            str(size): max(
                row["steady_inclusive_seconds"] + row["periodic_eval_seconds"] / 15
                for key, row in observations.items()
                if key.startswith(f"{size}-")
            )
            for size in (512, 768, 1024)
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-h100-hours", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = calibration_report(args.prior_h100_hours)
    fs, path = filesystem_for(args.output)
    with fs.open(path, "w") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key not in ("observations", "runs")}, indent=2))


if __name__ == "__main__":
    main()
