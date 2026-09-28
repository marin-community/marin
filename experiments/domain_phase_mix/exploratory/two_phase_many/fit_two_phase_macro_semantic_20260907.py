# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Fit the registered macro additive versus positive-deficit semantic heads."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fit_two_phase_creative_geometry_20260907 as geometry
import fit_two_phase_creative_semantic_20260907 as semantic
import fit_two_phase_link_transfer_20260907 as previous
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
SOURCE = geometry.SOURCE
OUTPUT = HERE / "reference_outputs/two_phase_refinement_20260907/macro_semantic"
MODELS = ("CRE2-019", "CRE2-020")
RIDGES = (10.0, 1.0, 0.1, 0.01, 0.001)


def policy_inputs(weights: np.ndarray, metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    alpha = metadata["alpha"]
    aggregate = alpha * weights[:, 0] + (1 - alpha) * weights[:, 1]
    components, _ = geometry.spine_values(aggregate, metadata["spines"], np.asarray(metadata["c_total"]))
    component_weights = np.asarray(metadata["aggregation_weights"])
    floors = np.array([record["floor"] for record in metadata["spines"]])
    floor = float(floors @ component_weights)
    baseline = components @ component_weights
    q = (components - floors) @ component_weights
    assert np.all(q > 0)
    matrix = semantic.semantic_design(weights, "CRE2-016", metadata)
    return matrix, baseline, q, floor


def phase_prediction(design: np.ndarray, coefficients: np.ndarray, q: np.ndarray, model: str) -> np.ndarray:
    linear = design @ coefficients
    if model == "CRE2-019":
        return linear
    with np.errstate(over="raise", invalid="raise"):
        return q * np.expm1(linear)


def residual_jacobian(
    coefficients: np.ndarray, design: np.ndarray, q: np.ndarray, response: np.ndarray, ridge: float
) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(over="raise", invalid="raise"):
        ratio = np.exp(design @ coefficients)
        residual = q * (ratio - 1) - response
        jacobian = (q * ratio)[:, None] * design
    penalty = np.sqrt(len(response) * ridge)
    return np.r_[residual, penalty * coefficients], np.vstack([jacobian, penalty * np.eye(design.shape[1])])


@dataclass(frozen=True)
class Predictor:
    record: dict[str, Any]

    def predict(self, weights: np.ndarray) -> np.ndarray:
        """Predict macro BPB for arbitrary policies with shape (n,2,39)."""
        matrix, baseline, q, _ = policy_inputs(np.asarray(weights), self.record)
        if self.record["config"]["zero"]:
            return baseline
        design = matrix / np.asarray(self.record["scale"])
        return baseline + phase_prediction(design, np.asarray(self.record["coefficients"]), q, self.record["model"])


def load_predictor(output: Path, objective: str, context: str, model: str) -> Predictor:
    """Load a frozen macro fit; no component phase predictions are implied."""
    return Predictor(json.loads((output / "fits" / model / objective / context / "fit.json").read_text()))


def fit_head(matrix: np.ndarray, q: np.ndarray, response: np.ndarray, model: str, ridge: float) -> dict[str, Any]:
    local_design = matrix if model == "CRE2-019" else q[:, None] * matrix
    raw_scale = np.sqrt(np.mean(local_design**2, axis=0))
    scale = np.where(raw_scale > 1e-10, raw_scale, 1.0)
    design = matrix / scale
    initial = geometry.constrained_coefficients(local_design / scale, response[:, None], ridge)[:, 0]
    initial[raw_scale <= 1e-10] = 0
    if model == "CRE2-019":
        residual = design @ initial - response
        cost = float(0.5 * (residual @ residual + len(response) * ridge * (initial @ initial)))
        starts = [{"name": "convex_constrained_ridge", "success": True, "cost": cost, "nfev": 1}]
        selected, coefficients = 0, initial
    else:
        width = matrix.shape[1] // 2
        lower = np.r_[np.full(width, -np.inf), np.zeros(width)]
        starts, solutions = [], []
        for name, start in (
            ("zero", np.zeros_like(initial)),
            ("linearized", initial),
            ("half_linearized", 0.5 * initial),
        ):
            try:
                result = least_squares(
                    lambda value: residual_jacobian(value, design, q, response, ridge)[0],
                    start,
                    jac=lambda value: residual_jacobian(value, design, q, response, ridge)[1],
                    bounds=(lower, np.full_like(lower, np.inf)),
                    max_nfev=1000,
                    ftol=1e-9,
                    xtol=1e-9,
                    gtol=1e-9,
                )
                success = bool(result.success and np.isfinite(result.cost) and np.isfinite(result.x).all())
                starts.append(
                    {
                        "name": name,
                        "success": success,
                        "cost": float(result.cost),
                        "nfev": int(result.nfev),
                        "optimality": float(result.optimality),
                        "message": str(result.message),
                    }
                )
                solutions.append(np.asarray(result.x))
            except FloatingPointError as error:
                starts.append({"name": name, "success": False, "cost": None, "nfev": None, "message": str(error)})
                solutions.append(None)
        eligible = [index for index, start in enumerate(starts) if start["success"]]
        if not eligible:
            return {"success": False, "starts": starts, "scale": scale.tolist()}
        selected = min(eligible, key=lambda index: starts[index]["cost"])
        coefficients = solutions[selected]
        assert coefficients is not None
        coefficients[raw_scale <= 1e-10] = 0
    return {
        "success": True,
        "coefficients": coefficients.tolist(),
        "scale": scale.tolist(),
        "inactive_columns": np.flatnonzero(raw_scale <= 1e-10).tolist(),
        "starts": starts,
        "selected_start": selected,
    }


def identity(output: Path) -> dict[str, str]:
    paths = [
        Path(__file__),
        Path(geometry.__file__),
        Path(semantic.__file__),
        Path(previous.__file__),
        output / "PROTOCOL.md",
        SOURCE / "inputs/panel.npz",
        SOURCE / "inputs/splits.npz",
        SOURCE / "inputs/single_phase.py",
    ]
    return {str(path): previous.file_hash(path) for path in paths}


def structural_checks(output: Path) -> None:
    _, panel, splits = previous.inputs(str(SOURCE))
    train, _ = previous.selected_pairs(panel, splits["outer0_train"])
    test, _ = previous.selected_pairs(panel, splits["outer0_test"])
    rng = np.random.default_rng(20260907)
    records = []
    for objective in previous.OBJECTIVES:
        metadata = geometry.context_metadata(objective, "outer0")
        matrix, baseline, q, floor = policy_inputs(panel["weights"], metadata)
        tied = np.repeat(panel["aggregate"][:, None], 2, axis=1)
        tied_design, _, _, _ = policy_inputs(tied, metadata)
        assert np.max(np.abs(tied_design)) < 1e-12
        width = matrix.shape[1] // 2
        for model in MODELS:
            local = matrix if model == "CRE2-019" else q[:, None] * matrix
            scale = np.sqrt(np.mean(local[train] ** 2, axis=0))
            design = matrix / scale
            truth_coef = np.zeros(matrix.shape[1])
            truth_coef[:3] = rng.normal(size=3) * 0.002
            truth_coef[width : width + 3] = np.abs(rng.normal(size=3)) * 0.001
            truth = phase_prediction(design, truth_coef, q, model)
            head = fit_head(matrix[train], q[train], truth[train], model, 1e-10)
            assert head["success"], head
            prediction = phase_prediction(matrix / np.asarray(head["scale"]), np.asarray(head["coefficients"]), q, model)
            relative = float(np.linalg.norm(prediction[test] - truth[test]) / np.linalg.norm(truth[test]))
            assert relative < 0.001, (objective, model, relative)
            assert np.min(np.asarray(head["coefficients"])[width:]) >= 0
            row = {
                "objective": objective,
                "model": model,
                "columns": matrix.shape[1],
                "weighted_rank": int(np.linalg.matrix_rank(local[train] / scale)),
                "noiseless_held_relative_error": relative,
            }
            if model == "CRE2-020":
                direction = rng.normal(size=matrix.shape[1])
                direction /= np.linalg.norm(direction)
                epsilon = 1e-6
                analytic = residual_jacobian(truth_coef, design[train], q[train], truth[train], 0.1)[1] @ direction
                numerical = (
                    residual_jacobian(truth_coef + epsilon * direction, design[train], q[train], truth[train], 0.1)[0]
                    - residual_jacobian(truth_coef - epsilon * direction, design[train], q[train], truth[train], 0.1)[0]
                ) / (2 * epsilon)
                error = float(np.linalg.norm(analytic - numerical) / np.linalg.norm(analytic))
                assert error < 1e-6
                assert np.all(baseline + truth >= floor - 1e-12)
                row["jacobian_relative_error"] = error
            records.append(row)
    previous.write_json(output / "structural_checks.json", {"identity": identity(output), "checks": records})
    print(json.dumps(records, indent=2), flush=True)


def fit_cell(output: Path, objective: str, context: str, model: str) -> None:
    destination = output / "fits" / model / objective / context
    marker = destination / "complete.json"
    if marker.exists():
        complete = json.loads(marker.read_text())
        assert complete["identity"] == identity(output)
        for name, digest in complete["sha256"].items():
            assert previous.file_hash(destination / name) == digest
        print(f"cached {model}/{objective}/{context}", flush=True)
        return
    _, panel, splits = previous.inputs(str(SOURCE))
    response = panel[f"{objective}_aggregate"]
    rows = previous.training_rows(panel, splits, context)
    ta, tt = previous.selected_pairs(panel, rows)
    assert panel["calibration_mask"][rows].sum() == 2
    configs = [{"zero": True, "ridge": 0.0}] + [{"zero": False, "ridge": ridge} for ridge in RIDGES]
    errors, counts, valid = np.zeros(len(configs)), np.zeros(len(configs), dtype=int), np.ones(len(configs), dtype=bool)
    traces, source_paths, inner_rows = [], [], []
    for inner in range(3):
        prefix = f"{context}_inner{inner}"
        train_rows, test_rows = splits[f"{prefix}_train"], splits[f"{prefix}_test"]
        assert panel["calibration_mask"][train_rows].sum() == 2 and not panel["calibration_mask"][test_rows].any()
        ia, it = previous.selected_pairs(panel, train_rows)
        va, vt = previous.selected_pairs(panel, test_rows)
        assert not set(panel["groups"][ia]).intersection(panel["groups"][va])
        inner_rows.append({"prefix": prefix, "train_pairs": ia.tolist(), "test_pairs": va.tolist()})
        if not len(va):
            continue
        metadata = geometry.context_metadata(objective, prefix)
        matrix, _, q, _ = policy_inputs(panel["weights"], metadata)
        delta, held_delta = response[ia] - response[it], response[va] - response[vt]
        assert np.all(q[ia] + delta > 0) and np.all(q[va] + held_delta > 0)
        for index, config in enumerate(configs):
            head = (
                {"success": True, "starts": []}
                if config["zero"]
                else fit_head(matrix[ia], q[ia], delta, model, config["ridge"])
            )
            prediction = np.zeros(len(va))
            success = head["success"]
            if success and not config["zero"]:
                try:
                    prediction = phase_prediction(
                        matrix[va] / np.asarray(head["scale"]), np.asarray(head["coefficients"]), q[va], model
                    )
                except FloatingPointError:
                    success = False
            success = success and bool(np.isfinite(prediction).all())
            traces.append({"prefix": prefix, "config": config, "success": success, "head": head})
            valid[index] = valid[index] and success
            errors[index] += float(np.sum((prediction - held_delta) ** 2)) if success else 0.0
            counts[index] += len(va)
        source_paths += [SOURCE / "spines" / prefix / f"{objective}_c{i}.json" for i in range(len(metadata["spines"]))]
    assert counts.min() > 0
    best = min(np.flatnonzero(valid), key=lambda index: (errors[index] / counts[index], index))
    config = configs[best]
    metadata = geometry.context_metadata(objective, context)
    matrix, baseline, q, floor = policy_inputs(panel["weights"], metadata)
    delta = response[ta] - response[tt]
    assert np.all(q[ta] + delta > 0)
    head = (
        {"success": True, "starts": []} if config["zero"] else fit_head(matrix[ta], q[ta], delta, model, config["ridge"])
    )
    if not head["success"]:
        previous.write_json(destination / "failed_final_fit.json", head)
        raise RuntimeError(f"Selected final refit failed: {model}/{objective}/{context}")
    record = (
        metadata
        | head
        | {
            "objective": objective,
            "context": context,
            "model": model,
            "config": config,
            "train_rows": rows.tolist(),
            "train_asymmetric_rows": ta.tolist(),
            "train_tied_rows": tt.tolist(),
            "inner_rows": inner_rows,
            "selected_inner_mse": float(errors[best] / counts[best]),
            "floor_objective": floor,
        }
    )
    prediction = Predictor(record).predict(panel["weights"])
    assert np.isfinite(prediction).all()
    assert np.max(np.abs(prediction[panel["physical_tied"]] - baseline[panel["physical_tied"]])) < 1e-12
    if model == "CRE2-020":
        assert np.all(prediction >= floor - 1e-12)
    record |= {
        "minimum_prediction": float(prediction.min()),
        "below_objective_floor": int(np.sum(prediction < floor - 1e-12)),
        "negative_predictions": int(np.sum(prediction < 0)),
    }
    scored = ~panel["calibration_mask"] if context == "final" else panel["outer_fold"] == int(context[-1])
    destination.mkdir(parents=True, exist_ok=True)
    previous.write_json(destination / "fit.json", record)
    previous.write_json(destination / "inner_fit_traces.json", traces)
    previous.write_json(
        destination / "sweep.json",
        [
            config | {"valid": bool(ok), "mse": float(error / count) if ok else None, "n": int(count)}
            for config, ok, error, count in zip(configs, valid, errors, counts, strict=True)
        ],
    )
    pd.DataFrame(
        {
            "objective": objective,
            "model": model,
            "context": context,
            "row": np.arange(len(prediction)),
            "run": panel["runs"],
            "group": panel["groups"],
            "fold": panel["outer_fold"],
            "tied": panel["physical_tied"],
            "scored": scored,
            "measured": response,
            "predicted": prediction,
        }
    ).to_csv(destination / "predictions.csv", index=False)
    source_paths += [SOURCE / "spines" / context / f"{objective}_c{i}.json" for i in range(len(metadata["spines"]))]
    previous.write_json(
        marker,
        {
            "identity": identity(output),
            "spine_sha256": {str(path): previous.file_hash(path) for path in source_paths},
            "sha256": {
                name: previous.file_hash(destination / name)
                for name in ("fit.json", "inner_fit_traces.json", "sweep.json", "predictions.csv")
            },
        },
    )
    print(
        f"fit {model}/{objective}/{context}: {config}, CV RMSE={np.sqrt(record['selected_inner_mse']):.6f}", flush=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("checks", "fit", "collect"), required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.mode == "checks":
        structural_checks(args.output)
        return
    if args.mode == "fit":
        assert json.loads((args.output / "structural_checks.json").read_text())["identity"] == identity(args.output)
        for model in MODELS:
            for objective in previous.OBJECTIVES:
                for context in previous.CONTEXTS:
                    fit_cell(args.output, objective, context, model)
    frames = [
        pd.read_csv(args.output / "fits" / model / objective / context / "predictions.csv")
        for model in MODELS
        for objective in previous.OBJECTIVES
        for context in previous.CONTEXTS
    ]
    pd.concat(frames, ignore_index=True).to_csv(args.output / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
