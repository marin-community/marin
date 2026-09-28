# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Check failed raw SLSQP searches in unconstrained softmax coordinates.

No response parameters or policy constraints change. Finite logits describe the
simplex interior and approach its boundary; they do not certify an attained
boundary optimum. A 1e-12 starting-weight floor only initializes logits from
zero-containing starts. Model values remain exactly those of the raw audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import optimize_two_phase_link_transfer_20260907 as raw
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

START_WEIGHT_FLOOR = 1e-12
OPTIONS = {"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-9, "maxls": 50}


def optimize_start(model: raw.ObjectiveModel, initial: np.ndarray, tied: bool, name: str) -> dict[str, Any]:
    phases = 1 if tied else 2
    weights = np.maximum(initial[:phases], START_WEIGHT_FLOOR)
    weights /= weights.sum(axis=1, keepdims=True)
    log_weights = np.log(weights)
    logits = (log_weights[:, :-1] - log_weights[:, -1:]).ravel()

    def policy(parameters: np.ndarray) -> np.ndarray:
        augmented = np.column_stack([parameters.reshape(phases, model.buckets - 1), np.zeros(phases)])
        value = softmax(augmented, axis=1)
        return np.repeat(value, 2, axis=0) if tied else value

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        w = policy(parameters)
        value, gradient, _ = model.value_gradient(w)
        if tied:
            gradient = gradient.sum(axis=0, keepdims=True)
            w = w[:1]
        logit_gradient = w * (gradient - np.sum(w * gradient, axis=1, keepdims=True))
        return value, logit_gradient[:, :-1].ravel()

    solution = minimize(objective, logits, jac=True, method="L-BFGS-B", options=OPTIONS)
    endpoint = policy(solution.x)
    value, _, clips = model.value_gradient(endpoint)
    return {
        "name": name,
        "initial_weights": initial.tolist(),
        "initial_logit_projection_tv": raw.policy_tv(policy(logits), initial, float(model.phase_fraction[0])),
        "predicted_bpb": value,
        "success": bool(solution.success),
        "message": str(solution.message),
        "iterations": int(solution.nit),
        "function_evaluations": int(solution.nfev),
        "max_abs_logit": float(np.max(np.abs(solution.x))),
        "min_weight": float(endpoint.min()),
        "weights": endpoint.tolist(),
        "clips": clips,
    }


def run_cell(output: Path, source: Path) -> dict[str, Any]:
    original = json.loads(source.read_text())
    changed = [path for path, digest in original["input_hashes"].items() if raw.driver.file_hash(Path(path)) != digest]
    if changed:
        raise ValueError(f"Frozen raw-audit inputs changed: {changed}")
    context, objective, arm = (original[k] for k in ("context", "objective", "arm"))
    destination = output / "optima/softmax" / source.name
    identity = {str(path): raw.driver.file_hash(path) for path in (Path(__file__), Path(raw.__file__), source)}
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved["input_hashes"] != identity:
            raise ValueError(f"Softmax sensitivity source changed: {destination}")
        return saved
    model = raw.load_model(output, context, objective, arm)
    starts = [(r["name"], np.asarray(r["initial_weights"])) for r in original["starts"]]
    starts.append(("best_slsqp_endpoint", np.asarray(original["weights"])))
    results = [optimize_start(model, start, arm == "aggregate", name) for name, start in starts]
    successful = [r for r in results if r["success"]]
    best = min(successful or results, key=lambda r: r["predicted_bpb"])
    policy = np.asarray(best["weights"])
    _, panel, splits = raw.driver.inputs(str(output))
    alpha = float(panel["alpha"])
    train = raw.driver.training_rows(panel, splits, context)
    support = raw.support_audit(policy, panel["weights"], alpha)
    train_support = raw.support_audit(policy, panel["weights"][train], alpha)
    stability = [
        raw.policy_tv(np.asarray(left["weights"]), np.asarray(right["weights"]), alpha)
        for i, left in enumerate(successful)
        for right in successful[i + 1 :]
    ]
    baseline_path = output / "optima" / f"{context}_{objective}_aggregate.json"
    baseline = json.loads(baseline_path.read_text())
    floor_bound = float(model.floor @ model.aggregation)
    record = {
        "context": context,
        "objective": objective,
        "arm": arm,
        "input_hashes": identity,
        "predicted_bpb": best["predicted_bpb"],
        "slsqp_predicted_bpb": original["predicted_bpb"],
        "softmax_minus_slsqp_bpb": best["predicted_bpb"] - original["predicted_bpb"],
        "weighted_fitted_floor_lower_bound": floor_bound,
        "floor_bound_status": "Response lower bound; joint attainability is not established.",
        "gap_above_floor_bound": best["predicted_bpb"] - floor_bound,
        "predicted_gain_over_fitted_tied_optimum": baseline["predicted_bpb"] - best["predicted_bpb"],
        "predicted_improvement_over_best_observed": original["best_observed_bpb"] - best["predicted_bpb"],
        "selected_start": best["name"],
        "successful_starts": len(successful),
        "selected_solver_success": best["success"],
        "phase_tv": float(np.abs(policy[0] - policy[1]).sum() / 2),
        "max_phase_bucket_weight": float(policy.max()),
        "max_total_epochs": float(np.max(panel["c0"] * policy[0] + panel["c1"] * policy[1])),
        "min_weight": best["min_weight"],
        "max_abs_logit": best["max_abs_logit"],
        "weights": policy.tolist(),
        "full_design_support": support,
        "training_support": train_support,
        "successful_start_policy_tv_max": max(stability, default=0.0),
        "successful_start_policy_tv_median": float(np.median(stability)) if stability else 0.0,
        "successful_start_bpb_spread": float(np.ptp([r["predicted_bpb"] for r in successful])) if successful else None,
        "clips": best["clips"],
        "starts": results,
        "limitations": [
            "Finite-logit search approaches boundaries; it does not certify an attained boundary optimum.",
            "Nonconvex local searches do not certify a global optimum.",
            "Exposure derivatives retain the raw audit's numerical floor at zero; values do not change.",
        ],
    }
    raw.driver.write_json(destination, record)
    print(
        json.dumps({k: record[k] for k in ("context", "objective", "arm", "predicted_bpb", "successful_starts")}),
        flush=True,
    )
    return record


def summaries(output: Path, records: list[dict[str, Any]]) -> None:
    columns = (
        "context",
        "objective",
        "arm",
        "predicted_bpb",
        "slsqp_predicted_bpb",
        "softmax_minus_slsqp_bpb",
        "predicted_gain_over_fitted_tied_optimum",
        "predicted_improvement_over_best_observed",
        "weighted_fitted_floor_lower_bound",
        "gap_above_floor_bound",
        "successful_starts",
        "selected_solver_success",
        "phase_tv",
        "max_phase_bucket_weight",
        "max_total_epochs",
        "min_weight",
        "max_abs_logit",
        "successful_start_policy_tv_max",
        "successful_start_policy_tv_median",
        "successful_start_bpb_spread",
    )
    rows = []
    for record in records:
        rows.append(
            {key: record[key] for key in columns}
            | {f"full_{key}": value for key, value in record["full_design_support"].items()}
            | record["clips"]
        )
    pd.DataFrame(rows).to_csv(output / "optima/softmax/summary.csv", index=False)
    floors = []
    _, panel, _ = raw.driver.inputs(str(output))
    consolidated = []
    for context in raw.driver.CONTEXTS:
        for objective in raw.driver.OBJECTIVES:
            for arm in raw.driver.ARMS:
                path = output / "optima" / f"{context}_{objective}_{arm}.json"
                source = json.loads(path.read_text())
                alternate = [r for r in records if (r["context"], r["objective"], r["arm"]) == (context, objective, arm)]
                if alternate and alternate[0]["selected_solver_success"]:
                    source = alternate[0]
                model = raw.load_model(output, context, objective, arm)
                floor = float(model.aggregation @ model.floor)
                floors.append(
                    {
                        "context": context,
                        "objective": objective,
                        "arm": arm,
                        "predicted_bpb": source["predicted_bpb"],
                        "weighted_fitted_floor_lower_bound": floor,
                        "gap_above_floor_bound": source["predicted_bpb"] - floor,
                    }
                )
                consolidated.append(source)
    pd.DataFrame(floors).to_csv(output / "optima/floor_bound_audit.csv", index=False)
    distances = []
    for objective in raw.driver.OBJECTIVES:
        for arm in raw.driver.ARMS:
            cells = [r for r in consolidated if r["objective"] == objective and r["arm"] == arm]
            for i, left in enumerate(cells):
                for right in cells[i + 1 :]:
                    l, r = np.asarray(left["weights"]), np.asarray(right["weights"])
                    la = (panel["c0"] * l[0] + panel["c1"] * l[1]) / panel["inventory"]
                    ra = (panel["c0"] * r[0] + panel["c1"] * r[1]) / panel["inventory"]
                    distances.append(
                        {
                            "objective": objective,
                            "arm": arm,
                            "left_context": left["context"],
                            "right_context": right["context"],
                            "policy_tv": raw.policy_tv(l, r, float(panel["alpha"])),
                            "aggregate_tv": float(np.abs(la - ra).sum() / 2),
                        }
                    )
    pd.DataFrame(distances).to_csv(output / "optima/softmax/context_stability.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=raw.driver.DEFAULT_OUTPUT)
    args = parser.parse_args()
    destination = args.output / "optima/softmax"
    destination.mkdir(exist_ok=True)
    records = []
    for context in raw.driver.CONTEXTS:
        for objective in raw.driver.OBJECTIVES:
            for arm in raw.driver.ARMS:
                path = args.output / "optima" / f"{context}_{objective}_{arm}.json"
                record = json.loads(path.read_text())
                if record["successful_starts"] == len(record["starts"]):
                    continue
                records.append(run_cell(args.output, path))
    summaries(args.output, records)


if __name__ == "__main__":
    main()
