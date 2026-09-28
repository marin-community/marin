# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Polish final response heads and check independent coefficient starts offline.

Uses only the comparison's frozen input snapshot and training outcomes to fit.
The original fits are untouched. Evaluation predictions and restart diagnostics
are written under stability/ so numerical sensitivity stays separate from the
first comparison. No prediction or measured bank value selects a solver start.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import compare_joint_floor_20260908 as harness
import numpy as np
import pandas as pd

GAMMA_STARTS = (1.2, 2.0, 4.5, 5.8)
PERTURBATION_SD = 0.1
RANDOM_SEED = 20260908


def check_task(job: tuple[str, str, int, int, float]) -> dict[str, Any]:
    output_name, objective, index, max_nfev, tolerance = job
    output = Path(output_name)
    source = output / "tasks" / objective / f"r0_f-1_c{index}.json"
    destination = output / "stability/tasks" / objective / f"c{index}.json"
    signature = {
        "input_task_sha256": harness.sha256(source),
        "solver_sha256": harness.sha256(output / "inputs/joint_floor_solver.py"),
        "script_sha256": harness.sha256(Path(__file__)),
        "max_nfev": max_nfev,
        "tolerance": tolerance,
        "gamma_starts": list(GAMMA_STARTS),
        "coefficient_perturbation_sd": PERTURBATION_SD,
        "seed": RANDOM_SEED + index + (1000 if objective == "table9" else 0),
    }
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved["signature"] != signature:
            raise ValueError(f"cached stability output has different inputs or settings: {destination}")
        return {"cached": True, "path": str(destination)}
    started = time.monotonic()
    baseline, solver, swarm, _ = harness.inputs(output_name)
    original = json.loads(source.read_text())
    component = original["component"]
    matrix = baseline.design_matrix(swarm.exposures, original["shape"])
    response = swarm.outcomes[component].to_numpy(float)
    train = np.asarray(original["train"], dtype=int)
    incumbent_gamma = original["variants"]["frozen"]["gamma"]
    common = {
        "matrix": matrix[train],
        "response": response[train],
        "ridge": original["ridge"],
        "anchor": original["anchor"]["proportional"],
        "noise_sd": original["anchor"]["repeat_sd"],
        "incumbent_gamma": incumbent_gamma,
        "conditional_solve": baseline.nonnegative_solve,
        "gamma_starts": (),
        "max_nfev": max_nfev,
        "tolerance": tolerance,
    }
    fixed_before = original["variants"]["response_fixed"]
    fixed_start = solver.ResponseStart(
        fixed_before["intercept"], np.asarray(fixed_before["coefficients"]), fixed_before["gamma"]
    )
    fixed = solver.fit_response_head(**common, mode="fixed", extra_starts=(fixed_start,))
    joint_before = original["variants"]["response_joint"]
    original_coefficients = np.asarray(joint_before["coefficients"])
    starts = [
        solver.ResponseStart(joint_before["intercept"], original_coefficients, joint_before["gamma"]),
        solver.ResponseStart(fixed.intercept, fixed.coefficients, fixed.gamma),
    ]
    rng = np.random.default_rng(signature["seed"])
    feature_mean = matrix[train].mean(axis=0)
    for gamma in GAMMA_STARTS:
        coefficients = np.maximum(
            original_coefficients + rng.normal(0.0, PERTURBATION_SD, len(original_coefficients)), 0.0
        )
        # Keep the initial mean linear predictor near the solution while opening
        # inactive coefficients and perturbing directions beyond the NNLS starts.
        intercept = (
            joint_before["intercept"]
            + float(feature_mean @ (original_coefficients - coefficients))
            + float(rng.normal(0.0, PERTURBATION_SD))
        )
        starts.append(solver.ResponseStart(intercept, coefficients, gamma))
    joint = solver.fit_response_head(**common, mode="joint", extra_starts=starts)
    if joint.cost > fixed.cost + 1e-8 * max(1.0, fixed.cost):
        raise ValueError(f"joint restart is worse than its nested fixed control: {component}")
    bank = pd.read_csv(output / "inputs" / f"heldout_bank_{objective}.csv")
    bank_matrix = baseline.design_matrix(bank[list(swarm.buckets)].to_numpy(float) * swarm.inventory, original["shape"])
    records = {}
    for variant, head in (("response_fixed", fixed), ("response_joint", joint)):
        before = original["variants"][variant]
        old_head = baseline.Head(before["intercept"], np.asarray(before["coefficients"]), before["floor"])
        old_cost = before["solver"]["cost"]
        if head.cost > old_cost + 1e-8 * max(1.0, old_cost):
            raise ValueError(f"polishing increased objective: {component}/{variant}")
        record = harness.head_payload(head, head.gamma)
        record["solver"] = {
            key: value
            for key, value in dataclasses.asdict(head).items()
            if key not in {"coefficients", "intercept", "floor", "gamma"}
        }
        record["comparison"] = {
            "original_cost": old_cost,
            "cost_reduction": old_cost - head.cost,
            "relative_cost_reduction": (old_cost - head.cost) / max(abs(old_cost), 1e-12),
            "gamma_change": head.gamma - before["gamma"],
            "floor_change": head.floor - before["floor"],
            "max_abs_swarm_prediction_change": float(np.max(np.abs(head.predict(matrix) - old_head.predict(matrix)))),
            "max_abs_bank_prediction_change": float(
                np.max(np.abs(head.predict(bank_matrix) - old_head.predict(bank_matrix)))
            ),
            "gamma_at_lower": bool(head.gamma <= 1.0 + 1e-6),
            "gamma_at_upper": bool(head.gamma >= 6.0 - 1e-6),
            "original_success": before["solver"]["success"],
        }
        records[variant] = record
    payload = {
        "signature": signature,
        "objective": objective,
        "component": component,
        "component_index": index,
        "variants": records,
        "elapsed_seconds": time.monotonic() - started,
    }
    harness.write_json(destination, payload)
    return {
        "cached": False,
        "path": str(destination),
        "joint_success": joint.success,
        "joint_bank_max_change": records["response_joint"]["comparison"]["max_abs_bank_prediction_change"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }


def summarize(output: Path) -> None:
    baseline, _, swarm, objectives = harness.inputs(str(output))
    diagnostics, aggregate_changes = [], []
    for objective in objectives.values():
        variants = {"response_fixed": [], "response_joint": []}
        for index, component in enumerate(objective.components):
            path = output / "stability/tasks" / objective.name / f"c{index}.json"
            row = json.loads(path.read_text())
            source = json.loads((output / "tasks" / objective.name / f"r0_f-1_c{index}.json").read_text())
            for variant, tasks in variants.items():
                head = row["variants"][variant]
                tasks.append(
                    baseline.TaskFit(
                        component,
                        source["shape"],
                        source["ridge"],
                        head["gamma"],
                        False,
                        source["frozen_diagnostics"]["inner_cv_rmse"],
                        baseline.Head(head["intercept"], np.asarray(head["coefficients"]), head["floor"]),
                    )
                )
                diagnostics.append(
                    {
                        "objective": objective.name,
                        "component": component,
                        "variant": variant,
                        "gamma": head["gamma"],
                        "cost": head["solver"]["cost"],
                        "success": head["solver"]["success"],
                        "optimality": head["solver"]["optimality"],
                        "total_nfev": sum(start["nfev"] for start in head["solver"]["starts"]),
                        **head["comparison"],
                    }
                )
        bank = pd.read_csv(output / "inputs" / f"heldout_bank_{objective.name}.csv")
        bank_weights = bank[list(swarm.buckets)].to_numpy(float)
        bank_predictions = bank[["coordinate_id", "measured_mean_bpb"]].copy()
        swarm_predictions = pd.DataFrame({"run": swarm.runs})
        for variant, tasks in variants.items():
            fit = baseline.ObjectiveFit(objective.name, swarm.buckets, swarm.inventory, objective.weights, tuple(tasks))
            original_fit = harness.fitted_objective(output, objective.name, variant)
            harness.write_json(output / "stability/fits" / f"{objective.name}_{variant}.json", fit.to_json())
            for population, weights, table in (
                ("bank", bank_weights, bank_predictions),
                ("swarm", swarm.weights, swarm_predictions),
            ):
                prediction, original_prediction = fit.predict(weights), original_fit.predict(weights)
                table[f"original_{variant}"] = original_prediction
                table[f"polished_{variant}"] = prediction
                aggregate_changes.append(
                    {
                        "objective": objective.name,
                        "variant": variant,
                        "population": population,
                        "max_abs_prediction_change": float(np.max(np.abs(prediction - original_prediction))),
                        "rms_prediction_change": float(np.sqrt(np.mean((prediction - original_prediction) ** 2))),
                        "same_minimum_index": bool(np.argmin(prediction) == np.argmin(original_prediction)),
                    }
                )
        bank_predictions.to_csv(output / "stability" / f"bank_predictions_{objective.name}.csv", index=False)
        swarm_predictions.to_csv(output / "stability" / f"swarm_predictions_{objective.name}.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "stability/diagnostics.csv", index=False)
    pd.DataFrame(aggregate_changes).to_csv(output / "stability/aggregate_changes.csv", index=False)
    print(pd.DataFrame(aggregate_changes).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=harness.DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=5000)
    parser.add_argument("--tolerance", type=float, default=1e-11)
    args = parser.parse_args()
    _, _, _, objectives = harness.inputs(str(args.output))
    jobs = [
        (str(args.output), objective.name, index, args.max_nfev, args.tolerance)
        for objective in objectives.values()
        for index in range(len(objective.components))
    ]
    # Fail before launching any restart if the first-stage comparison is partial.
    for _, objective, index, _, _ in jobs:
        for fold in [-1, 0, 1, 2, 3, 4]:
            path = args.output / "tasks" / objective / f"r0_f{fold}_c{index}.json"
            if not path.exists():
                raise FileNotFoundError(f"complete the first-stage fits before stability checks: {path}")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(check_task, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            print(json.dumps({"done": count, "total": len(jobs), **future.result()}), flush=True)
    summarize(args.output)


if __name__ == "__main__":
    main()
