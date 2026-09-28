# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Adaptive recency-range and separate-acquisition WSPU followups."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fit_two_phase_creative_joint_20260907 import (
    CONTEXTS,
    INPUT_OUTPUT,
    OBJECTIVES,
    RIDGE_MULTIPLIERS,
    file_hash,
    floor_spec,
    inputs,
    load_spine,
    source_paths,
    training_rows,
    write_json,
)
from fit_two_phase_creative_joint_20260907 import design as initial_design

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "reference_outputs/two_phase_creative_sweep_20260907/joint_followup"
STATE_GRIDS = {"CRE2-013": (1.0, 0.0, 0.01, 0.03, 0.1, 0.25, 0.5, 2.0, 4.0), "CRE2-014": (0.0,)}


def design(
    module: Any,
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    shape: dict[str, float],
    model: str,
    state: float,
) -> np.ndarray:
    """Return physical-coefficient features for both registered extensions."""
    if model == "CRE2-013" and state > 0:
        return initial_design(module, weights, c0, c1, shape, "CRE2-002", state)
    inventory = c0 + c1
    early, late = c0 * weights[:, 0], c1 * weights[:, 1]
    total = np.where(weights[:, 0] == weights[:, 1], inventory * weights[:, 0], early + late)
    rate, power, threshold = shape["rate"], shape["power"], shape["threshold"]
    harm = module.harm(total, threshold) - module.harm(np.zeros_like(total), threshold)
    if model == "CRE2-013":
        effective = np.where(c1 > 0, inventory * weights[:, 1], total)
        return np.hstack([-module.benefit(effective, rate, power), harm])
    if model == "CRE2-014":
        return np.hstack(
            [
                -module.benefit(inventory * weights[:, 0], rate, power),
                -module.benefit(inventory * weights[:, 1], rate, power),
                harm,
            ]
        )
    raise ValueError(f"unknown model {model}")


def penalty_scale(model: str) -> np.ndarray:
    if model == "CRE2-014":
        return np.concatenate([np.full(78, np.sqrt(2)), np.ones(39)])
    return np.ones(78)


def identities(context: str, objective: str, component: int, output: Path) -> dict[str, str]:
    paths = [Path(__file__), *source_paths(context, objective, component, output)]
    return {str(path): file_hash(path) for path in paths}


def fit_component(argument: tuple[str, str, int, str]) -> dict[str, Any]:
    objective, context, component, output_string = argument
    output = Path(output_string)
    destination = output / "fits" / context / f"{objective}_c{component}.json"
    hashes = identities(context, objective, component, output)
    if destination.exists():
        if json.loads(destination.read_text())["input_hashes"] != hashes:
            raise ValueError(f"cached followup changed: {destination}")
        return {"context": context, "objective": objective, "component": component, "cached": True}
    module, panel, splits = inputs(str(INPUT_OUTPUT))
    response = panel[f"{objective}_outcomes"][:, component]
    sweeps = {}
    for model, states in STATE_GRIDS.items():
        scale = penalty_scale(model)
        rows = []
        for state in states:
            errors = {multiplier: [0.0, 0] for multiplier in RIDGE_MULTIPLIERS}
            for inner in range(3):
                prefix = f"{context}_inner{inner}"
                train, test = splits[f"{prefix}_train"], splits[f"{prefix}_test"]
                spine = load_spine(INPUT_OUTPUT, prefix, objective, component)
                matrix = design(module, panel["weights"], panel["c0"], panel["c1"], spine.shape, model, state) / scale
                spec = floor_spec(module, panel, spine, objective, component)
                for multiplier in RIDGE_MULTIPLIERS:
                    head = module.fit_head(matrix[train], response[train], multiplier * spine.ridge, spec)
                    prediction = head.predict(matrix[test])
                    errors[multiplier][0] += float(np.sum((prediction - response[test]) ** 2))
                    errors[multiplier][1] += len(test)
            for multiplier in RIDGE_MULTIPLIERS:
                error, count = errors[multiplier]
                rows.append({"state": state, "ridge_multiplier": multiplier, "mse": error / count, "n": int(count)})
        sweeps[model] = rows
    train = training_rows(panel, splits, context)
    spine = load_spine(INPUT_OUTPUT, context, objective, component)
    fitted = {}
    for model in STATE_GRIDS:
        winner = min(sweeps[model], key=lambda row: row["mse"])
        physical = design(module, panel["weights"], panel["c0"], panel["c1"], spine.shape, model, winner["state"])
        scale = penalty_scale(model)
        matrix = physical / scale
        spec = floor_spec(module, panel, spine, objective, component)
        ridge = winner["ridge_multiplier"] * spine.ridge
        head = module.fit_head(matrix[train], response[train], ridge, spec)
        coefficients = head.coefficients / scale
        eta = head.intercept + physical @ coefficients
        assert np.isfinite(head.predict(matrix)).all()
        fitted[model] = winner | {
            "shape": spine.shape,
            "kappa": spine.kappa,
            "ridge": ridge,
            "floor": head.floor,
            "intercept": head.intercept,
            "coefficients": coefficients.tolist(),
            "coefficient_penalty_weights": (scale**2).tolist(),
            "training_below_floor_count": int(np.sum(response[train] <= head.floor)),
            "training_deficit_clamped_count": int(np.sum(response[train] - head.floor < module.DEFICIT_FLOOR)),
            "prediction_clipped_count": int(np.sum(np.abs(eta) > 30)),
        }
    write_json(
        destination,
        {
            "objective": objective,
            "context": context,
            "component": component,
            "component_name": str(panel[f"{objective}_components"][component]),
            "train_rows": train.tolist(),
            "input_hashes": hashes,
            "fits": fitted,
            "sweeps": sweeps,
        },
    )
    return {"context": context, "objective": objective, "component": component, "cached": False}


@dataclass(frozen=True)
class FollowupPredictor:
    module: Any
    c0: np.ndarray
    c1: np.ndarray
    model: str
    records: tuple[dict[str, Any], ...]
    aggregation_weights: np.ndarray

    def __call__(self, weights: np.ndarray) -> np.ndarray:
        if weights.ndim != 3 or weights.shape[1:] != (2, 39):
            raise ValueError("weights must have shape (n,2,39)")
        result = np.zeros(len(weights))
        for record, coefficient in zip(self.records, self.aggregation_weights, strict=True):
            matrix = design(self.module, weights, self.c0, self.c1, record["shape"], self.model, record["state"])
            eta = record["intercept"] + matrix @ np.asarray(record["coefficients"])
            result += coefficient * (record["floor"] + np.exp(np.clip(eta, -30, 30)))
        return result


def load_predictor(model: str, objective: str, context: str, output: Path = DEFAULT_OUTPUT) -> FollowupPredictor:
    """Load the component ensemble for arbitrary nonnegative simplex policies."""
    module, panel, _ = inputs(str(INPUT_OUTPUT))
    records = tuple(
        json.loads((output / "fits" / context / f"{objective}_c{component}.json").read_text())["fits"][model]
        for component in range(panel[f"{objective}_outcomes"].shape[1])
    )
    return FollowupPredictor(module, panel["c0"], panel["c1"], model, records, panel[f"{objective}_aggregation_weights"])


def export_predictions(output: Path) -> None:
    _, panel, _ = inputs(str(INPUT_OUTPUT))
    predictions = []
    selections = []
    for context in CONTEXTS:
        mask = ~panel["calibration_mask"]
        if context != "final":
            mask = mask & (panel["outer_fold"] == int(context[-1]))
        for objective in OBJECTIVES:
            for model in STATE_GRIDS:
                predictor = load_predictor(model, objective, context, output)
                predictions.append(
                    pd.DataFrame(
                        {
                            "objective": objective,
                            "model": model,
                            "context": context,
                            "row": np.arange(520),
                            "run": panel["runs"],
                            "group": panel["groups"],
                            "fold": panel["outer_fold"],
                            "tied": panel["physical_tied"],
                            "scored": mask,
                            "measured": panel[f"{objective}_aggregate"],
                            "predicted": predictor(panel["weights"]),
                        }
                    )
                )
                for component, record in enumerate(predictor.records):
                    selections.append(
                        {"objective": objective, "context": context, "model": model, "component": component}
                        | {key: record[key] for key in ("state", "ridge_multiplier", "mse", "ridge", "floor", "kappa")}
                    )
    pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(output / "selections.csv", index=False)


def structural_checks(output: Path) -> None:
    module, panel, splits = inputs(str(INPUT_OUTPUT))
    weights = panel["weights"]
    tied = np.stack([panel["aggregate"], panel["aggregate"]], axis=1)
    corners = np.stack([np.eye(39), np.roll(np.eye(39), 1, axis=0)], axis=1)
    train, test = splits["outer0_train"], splits["outer0_test"]
    random = np.random.default_rng(2026090713)
    checks = []
    for objective in OBJECTIVES:
        shape = load_spine(INPUT_OUTPUT, "final", objective, 0).shape
        reference = initial_design(module, tied, panel["c0"], panel["c1"], shape, "CRE2-001", 0.0)
        for model, states in STATE_GRIDS.items():
            for state in states:
                matrix = design(module, weights, panel["c0"], panel["c1"], shape, model, state)
                tied_matrix = design(module, tied, panel["c0"], panel["c1"], shape, model, state)
                corner_matrix = design(module, corners, panel["c0"], panel["c1"], shape, model, state)
                assert np.isfinite(corner_matrix).all()
                unused = (corners[:, 0] == 0) & (corners[:, 1] == 0)
                for block in range(matrix.shape[1] // 39):
                    assert np.max(np.abs(corner_matrix[:, 39 * block : 39 * (block + 1)][unused]), initial=0) == 0
                if model == "CRE2-013":
                    assert np.max(np.abs(tied_matrix - reference)) < 1e-12
                    for c0, c1 in ((np.zeros(39), panel["inventory"]), (panel["inventory"], np.zeros(39))):
                        collapsed = design(module, weights, c0, c1, shape, model, state)
                        null = initial_design(module, weights, c0, c1, shape, "CRE2-001", 0.0)
                        assert np.max(np.abs(collapsed - null)) < 1e-12
                else:
                    assert np.max(np.abs(tied_matrix[:, :39] - reference[:, :39])) == 0
                    assert np.max(np.abs(tied_matrix[:, 39:] - reference)) == 0
                    values = random.uniform(0, 1, 39)
                    assert np.max(np.abs(2 * (values / 2) ** 2 + 2 * (values / 2) ** 2 - values**2)) < 1e-14
                rms = np.sqrt(np.mean(matrix**2, axis=0))
                truth = random.uniform(0.5, 1.5, matrix.shape[1]) * 0.001 / np.maximum(rms, 1e-9)
                target = -3 + matrix @ truth
                scale = penalty_scale(model)
                intercept, coefficients = module.nonnegative_solve(matrix[train] / scale, target[train], 0.0)
                error = float(np.max(np.abs(intercept + matrix[test] @ (coefficients / scale) - target[test])))
                assert error < 1e-7, (objective, model, state, error)
                rank = int(np.linalg.matrix_rank(matrix[train] - matrix[train].mean(axis=0)))
                checks.append(
                    {"objective": objective, "model": model, "state": state, "rank": rank, "max_eta_error": error}
                )
    write_json(output / "structural_checks.json", {"source_hash": file_hash(Path(__file__)), "checks": checks})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("check", "fit", "export", "all"), default="all")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        raise ValueError("use one or two workers")
    if args.stage in ("check", "all"):
        structural_checks(args.output)
    if args.stage in ("fit", "all"):
        checks = json.loads((args.output / "structural_checks.json").read_text())
        if checks["source_hash"] != file_hash(Path(__file__)):
            raise ValueError("rerun checks after source changes")
        _, panel, _ = inputs(str(INPUT_OUTPUT))
        jobs = [
            (objective, context, component, str(args.output))
            for context in CONTEXTS
            for objective in OBJECTIVES
            for component in range(panel[f"{objective}_outcomes"].shape[1])
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fit_component, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), 1):
                record = future.result()
                if index % 8 == 0 or index == len(jobs):
                    print(json.dumps({"completed": index, "total": len(jobs)} | record), flush=True)
    if args.stage in ("export", "all"):
        export_predictions(args.output)


if __name__ == "__main__":
    main()
