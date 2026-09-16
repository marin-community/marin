# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Post-hoc ridge-zero sensitivity of the joint-floor heads, without proposals.

This diagnostic reuses frozen shapes and training-only floor inputs. It does not
retune shapes, estimate out-of-fold performance, or nominate a replacement.
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


def fit_task(job: tuple[str, str, int]) -> dict[str, Any]:
    output_name, objective, index = job
    output = Path(output_name)
    path = output / "ridge0_diagnostic/tasks" / objective / f"c{index}.json"
    source = output / "tasks" / objective / f"r0_f-1_c{index}.json"
    signature = {
        "source_task_sha256": harness.sha256(source),
        "script_sha256": harness.sha256(Path(__file__)),
        "solver_sha256": harness.sha256(output / "inputs/joint_floor_solver.py"),
        "ridge": 0,
        "max_nfev": 2000,
        "tolerance": 1e-9,
    }
    if path.exists():
        if json.loads(path.read_text())["signature"] != signature:
            raise ValueError(f"cached ridge-zero task inputs differ: {path}")
        return {"cached": True, "path": str(path)}
    started = time.monotonic()
    baseline, solver, swarm, _ = harness.inputs(output_name)
    original = json.loads(source.read_text())
    component = original["component"]
    train = np.asarray(original["train"], int)
    matrix = baseline.design_matrix(swarm.exposures, original["shape"])
    response = swarm.outcomes[component].to_numpy(float)
    previous = original["variants"]["response_joint"]
    head = solver.fit_response_head(
        matrix=matrix[train],
        response=response[train],
        ridge=0.0,
        anchor=original["anchor"]["proportional"],
        noise_sd=original["anchor"]["repeat_sd"],
        incumbent_gamma=original["variants"]["frozen"]["gamma"],
        conditional_solve=baseline.nonnegative_solve,
        mode="joint",
        extra_starts=(
            solver.ResponseStart(previous["intercept"], np.asarray(previous["coefficients"]), previous["gamma"]),
        ),
        max_nfev=2000,
        tolerance=1e-9,
    )
    payload = {
        "signature": signature,
        "objective": objective,
        "component": component,
        "component_index": index,
        "head": harness.head_payload(head, head.gamma),
        "solver": {
            key: value
            for key, value in dataclasses.asdict(head).items()
            if key not in {"coefficients", "intercept", "floor", "gamma"}
        },
        "original_ridge": original["ridge"],
        "original_joint_gamma": previous["gamma"],
        "train_rmse": float(np.sqrt(np.mean((head.predict(matrix[train]) - response[train]) ** 2))),
        "elapsed_seconds": time.monotonic() - started,
    }
    harness.write_json(path, payload)
    return {
        "cached": False,
        "path": str(path),
        "success": head.success,
        "gamma": head.gamma,
        "elapsed_seconds": payload["elapsed_seconds"],
    }


def evaluate(output: Path) -> None:
    baseline, _, swarm, objectives = harness.inputs(str(output))
    root = output / "ridge0_diagnostic"
    components, metrics, fresh_metrics = [], [], []
    for objective in objectives.values():
        tasks = []
        for index, component in enumerate(objective.components):
            row = json.loads((root / "tasks" / objective.name / f"c{index}.json").read_text())
            source = json.loads((output / "tasks" / objective.name / f"r0_f-1_c{index}.json").read_text())
            head = row["head"]
            # TaskFit requires this field, but the diagnostic does not rescore
            # inner CV. Only the prediction interface is used below.
            tasks.append(
                baseline.TaskFit(
                    component,
                    source["shape"],
                    0.0,
                    head["gamma"],
                    False,
                    source["frozen_diagnostics"]["inner_cv_rmse"],
                    baseline.Head(head["intercept"], np.asarray(head["coefficients"]), head["floor"]),
                )
            )
            components.append(
                {
                    "objective": objective.name,
                    "component": component,
                    "gamma": head["gamma"],
                    "floor": head["floor"],
                    "original_ridge": row["original_ridge"],
                    "original_joint_gamma": row["original_joint_gamma"],
                    "success": row["solver"]["success"],
                    "optimality": row["solver"]["optimality"],
                    "cost": row["solver"]["cost"],
                    "train_rmse": row["train_rmse"],
                    "gamma_at_lower": head["gamma"] <= 1 + 1e-6,
                    "gamma_at_upper": head["gamma"] >= 6 - 1e-6,
                }
            )
        fit = baseline.ObjectiveFit(objective.name, swarm.buckets, swarm.inventory, objective.weights, tuple(tasks))
        bank = pd.read_csv(output / "inputs" / f"heldout_bank_{objective.name}.csv")
        bank["prediction_joint_ridge0"] = fit.predict(bank[list(swarm.buckets)].to_numpy(float))
        max_epochs = (bank[list(swarm.buckets)].to_numpy(float) * swarm.inventory).max(axis=1)
        masks = {
            "all": np.ones(len(bank), bool),
            "optima": (
                ~bank.sources.str.split(";").apply(lambda values: bool(set(values) & harness.INTERVENTIONS)).to_numpy()
            ),
            "cap6": max_epochs <= 6 + 1e-8,
            "cap8": max_epochs <= 8 + 1e-8,
        }
        for population, mask in masks.items():
            prediction = bank.loc[mask, "prediction_joint_ridge0"].to_numpy(float)
            measured = bank.loc[mask, "measured_mean_bpb"].to_numpy(float)
            selected = bank.loc[mask].iloc[int(np.argmin(prediction))].coordinate_id
            metrics.append(
                {
                    "objective": objective.name,
                    "population": population,
                    "variant": "posthoc_response_joint_ridge0",
                    "selected_id": selected,
                    **harness.metric_values(prediction, measured),
                }
            )
        bank.to_csv(root / f"bank_predictions_{objective.name}.csv", index=False)
        fresh = pd.read_csv(output / "inputs/fresh_runs.csv")
        fresh["prediction_joint_ridge0"] = fit.predict(fresh[list(swarm.buckets)].to_numpy(float))
        for population, group in (("all", fresh), ("targeted", fresh[fresh.target.eq(objective.name)])):
            fresh_metrics.append(
                {
                    "objective": objective.name,
                    "population": population,
                    "variant": "posthoc_response_joint_ridge0",
                    **harness.metric_values(
                        group.prediction_joint_ridge0.to_numpy(float),
                        group[f"measured_{objective.name}"].to_numpy(float),
                    ),
                }
            )
        fresh.to_csv(root / f"fresh_predictions_{objective.name}.csv", index=False)
    pd.DataFrame(components).to_csv(root / "component_fits.csv", index=False)
    pd.DataFrame(metrics).to_csv(root / "bank_metrics.csv", index=False)
    pd.DataFrame(fresh_metrics).to_csv(root / "fresh_metrics.csv", index=False)
    print(pd.DataFrame(metrics).query("population == 'optima'").to_string(index=False), flush=True)
    print(pd.DataFrame(fresh_metrics).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=harness.DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    _, _, _, objectives = harness.inputs(str(args.output))
    root = args.output / "ridge0_diagnostic"
    harness.write_json(
        root / "manifest.json",
        {
            "role": "Post-hoc ridge-geometry sensitivity; not a retuned or prospectively validated candidate.",
            "change": "Set coefficient ridge to zero; keep frozen shapes, gamma bounds and noise rule.",
            "normalization": "Same fixed mean deficit at incumbent floor as the initial comparison.",
            "selection": "Lowest training objective across conditional NNLS starts and original joint head.",
            "no_oof": True,
            "no_proposals": True,
            "no_lm_training": True,
            "script_sha256": harness.sha256(Path(__file__)),
        },
    )
    jobs = [
        (str(args.output), objective.name, index)
        for objective in objectives.values()
        for index in range(len(objective.components))
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fit_task, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            print(json.dumps({"done": count, "total": len(jobs), **future.result()}), flush=True)
    evaluate(args.output)


if __name__ == "__main__":
    main()
