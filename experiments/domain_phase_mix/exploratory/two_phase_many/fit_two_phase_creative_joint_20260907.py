# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Fit joint component WSPU phase constructions on frozen grouped splits."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fit_two_phase_link_transfer_20260907 import (
    CONTEXTS,
    OBJECTIVES,
    file_hash,
    inputs,
    load_spine,
    training_rows,
    write_json,
)
from fit_two_phase_link_transfer_20260907 import (
    DEFAULT_OUTPUT as INPUT_OUTPUT,
)

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "reference_outputs/two_phase_creative_sweep_20260907/joint_wspu"
STATE_GRIDS = {
    "CRE2-001": (0.0,),
    "CRE2-002": (1.0, 0.25, 0.5, 2.0, 4.0),
    "CRE2-003": (0.0, -1.0, -0.5, 0.5, 1.0),
    "CRE2-004": (0.0, 0.5, 1.0),
}
RIDGE_MULTIPLIERS = (10.0, 1.0, 0.1)


def design(
    module: Any,
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    shape: dict[str, float],
    model: str,
    state: float,
) -> np.ndarray:
    """Return joint benefit and harm features in physical epoch coordinates."""
    rho = c0 / (c0 + c1)
    early, late = c0 * weights[:, 0], c1 * weights[:, 1]
    tied = weights[:, 0] == weights[:, 1]
    total = np.where(tied, (c0 + c1) * weights[:, 0], early + late)
    rate, power, threshold = shape["rate"], shape["power"], shape["threshold"]
    benefit = module.benefit(total, rate, power)
    harm_zero = module.harm(np.zeros_like(total), threshold)
    harm = module.harm(total, threshold) - harm_zero
    if model == "CRE2-002":
        effective = (state * early + late) / (state * rho + 1 - rho)
        effective = np.where(tied, total, effective)
        benefit = module.benefit(effective, rate, power)
    elif model == "CRE2-003":
        delta_b = module.benefit(early, rate, power) - module.benefit(rho * total, rate, power)
        delta_h = module.harm(early, threshold) - module.harm(rho * total, threshold)
        benefit = benefit - state * np.where(tied, 0.0, delta_b)
        harm = harm + state * np.where(tied, 0.0, delta_h)
    elif model == "CRE2-004":
        delta_b = (
            module.benefit(early, rate, power)
            + module.benefit(late, rate, power)
            - module.benefit(rho * total, rate, power)
            - module.benefit((1 - rho) * total, rate, power)
        )
        delta_h = (
            module.harm(early, threshold)
            + module.harm(late, threshold)
            - module.harm(rho * total, threshold)
            - module.harm((1 - rho) * total, threshold)
        )
        benefit = benefit + state * np.where(tied, 0.0, delta_b)
        harm = harm + state * np.where(tied, 0.0, delta_h)
    elif model != "CRE2-001":
        raise ValueError(f"unknown model {model}")
    return np.hstack([-benefit, harm])


def source_paths(context: str, objective: str, component: int, output: Path) -> list[Path]:
    prefixes = [context, *[f"{context}_inner{inner}" for inner in range(3)]]
    return [
        Path(__file__),
        HERE / "fit_two_phase_link_transfer_20260907.py",
        INPUT_OUTPUT / "inputs/single_phase.py",
        INPUT_OUTPUT / "inputs/panel.npz",
        INPUT_OUTPUT / "inputs/splits.npz",
        output / "PROTOCOL.md",
        *[INPUT_OUTPUT / "spines" / prefix / f"{objective}_c{component}.json" for prefix in prefixes],
    ]


def floor_spec(module: Any, panel: dict[str, np.ndarray], spine: Any, objective: str, component: int) -> Any:
    anchor_index = component if objective == "uncheatable" else component + 7
    return module.FloorSpec(
        anchor=float(panel["anchor_proportional_bpb"][anchor_index]),
        noise_sd=float(panel["anchor_repeat_sd"][anchor_index]),
        kappa=spine.kappa,
    )


def fit_component(argument: tuple[str, str, int, str]) -> dict[str, Any]:
    objective, context, component, output_string = argument
    output = Path(output_string)
    destination = output / "fits" / context / f"{objective}_c{component}.json"
    identities = {str(path): file_hash(path) for path in source_paths(context, objective, component, output)}
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous["input_hashes"] != identities:
            raise ValueError(f"cached fit inputs changed: {destination}")
        return {"context": context, "objective": objective, "component": component, "cached": True}
    module, panel, splits = inputs(str(INPUT_OUTPUT))
    response = panel[f"{objective}_outcomes"][:, component]
    sweeps = {}
    for model, state_grid in STATE_GRIDS.items():
        rows = []
        for state in state_grid:
            errors = {multiplier: [0.0, 0] for multiplier in RIDGE_MULTIPLIERS}
            for inner in range(3):
                prefix = f"{context}_inner{inner}"
                train, test = splits[f"{prefix}_train"], splits[f"{prefix}_test"]
                spine = load_spine(INPUT_OUTPUT, prefix, objective, component)
                matrix = design(module, panel["weights"], panel["c0"], panel["c1"], spine.shape, model, state)
                spec = floor_spec(module, panel, spine, objective, component)
                for multiplier in RIDGE_MULTIPLIERS:
                    fitted = module.fit_head(matrix[train], response[train], multiplier * spine.ridge, spec)
                    prediction = fitted.predict(matrix[test])
                    errors[multiplier][0] += float(np.sum((response[test] - prediction) ** 2))
                    errors[multiplier][1] += len(test)
            for multiplier in RIDGE_MULTIPLIERS:
                error, count = errors[multiplier]
                rows.append({"state": state, "ridge_multiplier": multiplier, "mse": error / count, "n": int(count)})
        sweeps[model] = rows
    spine = load_spine(INPUT_OUTPUT, context, objective, component)
    train = training_rows(panel, splits, context)
    fitted_models = {}
    for model in STATE_GRIDS:
        winner = min(sweeps[model], key=lambda row: row["mse"])
        matrix = design(module, panel["weights"], panel["c0"], panel["c1"], spine.shape, model, winner["state"])
        spec = floor_spec(module, panel, spine, objective, component)
        ridge = winner["ridge_multiplier"] * spine.ridge
        head = module.fit_head(matrix[train], response[train], ridge, spec)
        if not np.isfinite(head.predict(matrix)).all():
            raise ValueError("nonfinite joint prediction")
        fitted_models[model] = winner | {
            "shape": spine.shape,
            "kappa": spine.kappa,
            "ridge": ridge,
            "floor": head.floor,
            "intercept": head.intercept,
            "coefficients": head.coefficients.tolist(),
            "inherited_tied_floor": spine.head.floor,
            "training_minimum": float(np.min(response[train])),
            "training_below_floor_count": int(np.sum(response[train] <= head.floor)),
            "training_deficit_clamped_count": int(np.sum(response[train] - head.floor < module.DEFICIT_FLOOR)),
            "prediction_clipped_count": int(np.sum(np.abs(head.intercept + matrix @ head.coefficients) > 30.0)),
        }
    record = {
        "context": context,
        "objective": objective,
        "component": component,
        "component_name": str(panel[f"{objective}_components"][component]),
        "train_rows": train.tolist(),
        "input_hashes": identities,
        "fits": fitted_models,
        "sweeps": sweeps,
    }
    write_json(destination, record)
    return {"context": context, "objective": objective, "component": component, "cached": False}


@dataclass(frozen=True)
class JointPredictor:
    module: Any
    c0: np.ndarray
    c1: np.ndarray
    model: str
    records: tuple[dict[str, Any], ...]
    aggregation_weights: np.ndarray

    def __call__(self, weights: np.ndarray) -> np.ndarray:
        if weights.ndim != 3 or weights.shape[1:] != (2, 39):
            raise ValueError("weights must have shape (n, 2, 39)")
        result = np.zeros(len(weights))
        for record, aggregation_weight in zip(self.records, self.aggregation_weights, strict=True):
            matrix = design(self.module, weights, self.c0, self.c1, record["shape"], self.model, record["state"])
            eta = record["intercept"] + matrix @ np.asarray(record["coefficients"])
            result += aggregation_weight * (record["floor"] + np.exp(np.clip(eta, -30.0, 30.0)))
        return result


def load_predictor(model: str, objective: str, context: str, output: Path = DEFAULT_OUTPUT) -> JointPredictor:
    """Load a saved component ensemble accepting policies shaped (n, 2, 39)."""
    module, panel, _ = inputs(str(INPUT_OUTPUT))
    records = tuple(
        json.loads((output / "fits" / context / f"{objective}_c{component}.json").read_text())["fits"][model]
        for component in range(panel[f"{objective}_outcomes"].shape[1])
    )
    return JointPredictor(module, panel["c0"], panel["c1"], model, records, panel[f"{objective}_aggregation_weights"])


def export_predictions(output: Path) -> None:
    _, panel, splits = inputs(str(INPUT_OUTPUT))
    tables = []
    selection = []
    for objective in OBJECTIVES:
        for context in CONTEXTS:
            scored = np.zeros(len(panel["runs"]), bool)
            if context == "final":
                scored[:] = True
            else:
                scored[splits[f"{context}_test"]] = True
            scored &= ~panel["calibration_mask"]
            for model in STATE_GRIDS:
                predictor = load_predictor(model, objective, context, output)
                tables.append(
                    pd.DataFrame(
                        {
                            "objective": objective,
                            "model": model,
                            "context": context,
                            "row": np.arange(len(panel["runs"])),
                            "run": panel["runs"],
                            "group": panel["groups"],
                            "fold": panel["outer_fold"],
                            "tied": panel["physical_tied"],
                            "scored": scored,
                            "measured": panel[f"{objective}_aggregate"],
                            "predicted": predictor(panel["weights"]),
                        }
                    )
                )
                for component, record in enumerate(predictor.records):
                    selection.append(
                        {"objective": objective, "context": context, "model": model, "component": component}
                        | {key: record[key] for key in ("state", "ridge_multiplier", "mse", "ridge", "floor", "kappa")}
                    )
    pd.concat(tables, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(selection).to_csv(output / "selections.csv", index=False)


def structural_checks(output: Path) -> None:
    module, panel, splits = inputs(str(INPUT_OUTPUT))
    weights = panel["weights"]
    tied_weights = np.stack([panel["aggregate"], panel["aggregate"]], axis=1)
    corners = np.stack([np.eye(39), np.roll(np.eye(39), 1, axis=0)], axis=1)
    records = []
    random = np.random.default_rng(20260907)
    for objective in OBJECTIVES:
        shape = load_spine(INPUT_OUTPUT, "final", objective, 0).shape
        reference = design(module, tied_weights, panel["c0"], panel["c1"], shape, "CRE2-001", 0.0)
        for model, states in STATE_GRIDS.items():
            for state in states:
                matrix = design(module, weights, panel["c0"], panel["c1"], shape, model, state)
                tied_matrix = design(module, tied_weights, panel["c0"], panel["c1"], shape, model, state)
                corner_matrix = design(module, corners, panel["c0"], panel["c1"], shape, model, state)
                assert np.max(np.abs(tied_matrix - reference), initial=0) < 1e-12
                unused = (corners[:, 0] == 0) & (corners[:, 1] == 0)
                assert np.max(np.abs(corner_matrix[:, :39][unused]), initial=0) == 0.0
                assert np.max(np.abs(corner_matrix[:, 39:][unused]), initial=0) == 0.0
                assert np.isfinite(corner_matrix).all()
                for c0, c1 in ((np.zeros(39), panel["inventory"]), (panel["inventory"], np.zeros(39))):
                    collapsed = design(module, weights, c0, c1, shape, model, state)
                    null = design(module, weights, c0, c1, shape, "CRE2-001", 0.0)
                    assert np.max(np.abs(collapsed - null), initial=0) < 1e-12
                scale = np.sqrt(np.mean(matrix**2, axis=0))
                coefficients = random.uniform(0.5, 1.5, 78) * 0.001 / np.maximum(scale, 1e-9)
                target = -3.0 + matrix @ coefficients
                train, test = splits["outer0_train"], splits["outer0_test"]
                intercept, fitted = module.nonnegative_solve(matrix[train], target[train], 0.0)
                error = float(np.max(np.abs(intercept + matrix[test] @ fitted - target[test])))
                assert error < 1e-7, (objective, model, state, error)
                rank = int(np.linalg.matrix_rank(matrix[train] - matrix[train].mean(axis=0)))
                records.append(
                    {"objective": objective, "model": model, "state": state, "rank": rank, "max_eta_error": error}
                )
    write_json(output / "structural_checks.json", {"source_hash": file_hash(Path(__file__)), "checks": records})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--stage", choices=("check", "fit", "export", "all"), default="all")
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        raise ValueError("use one or two local workers")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage in ("check", "all"):
        structural_checks(args.output)
    if args.stage in ("fit", "all"):
        checks = json.loads((args.output / "structural_checks.json").read_text())
        if checks["source_hash"] != file_hash(Path(__file__)):
            raise ValueError("rerun structural checks after changing source")
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
