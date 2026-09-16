# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Fit registered component-supervised phase geometry diagnostics offline.

Use ``--mode checks`` before ``--mode fit``. The public ``load_predictor``
returns a predictor for arbitrary weights of shape (n, 2, 39). Its tied
restriction is the immutable finalized WSPU aggregate component model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import fit_two_phase_link_transfer_20260907 as previous
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "reference_outputs/two_phase_link_transfer_20260907"
OUTPUT = HERE / "reference_outputs/two_phase_creative_sweep_20260907/phase_geometry"
MODELS = ("CRE2-009", "CRE2-010", "CRE2-011", "CRE2-012")
RIDGES = (10.0, 1.0, 0.1, 0.01, 0.001)


@dataclass(frozen=True)
class Predictor:
    """Saved component prediction plus its exact training-fold aggregate spine."""

    record: dict[str, Any]

    def predict_components(self, weights: np.ndarray) -> np.ndarray:
        record = self.record
        alpha = float(record["alpha"])
        aggregate = alpha * weights[:, 0] + (1 - alpha) * weights[:, 1]
        base, _ = spine_values(aggregate, record["spines"], np.asarray(record["c_total"]))
        config = record["config"]
        if config["zero"]:
            return base
        matrix, tied = coordinates(weights, record["model"], record)
        scale = np.asarray(record["scale"])
        if record["model"] in MODELS[:2]:
            correction = (matrix / scale) @ np.asarray(record["coefficients"])
        else:
            correction = difference_kernel(
                matrix / scale,
                tied / scale,
                np.asarray(record["centers"]),
                np.asarray(record["tied_centers"]),
                config["bandwidth"],
            ) @ np.asarray(record["coefficients"])
        return base + correction

    def predict(self, weights: np.ndarray) -> np.ndarray:
        """Predict objective BPB for arbitrary two-phase policies."""
        values = self.predict_components(np.asarray(weights, dtype=float))
        return values @ np.asarray(self.record["aggregation_weights"])


def load_predictor(output: Path, objective: str, context: str, model: str) -> Predictor:
    """Load one completed fit without refitting or consulting evaluation labels."""
    path = output / "fits" / model / objective / context / "fit.json"
    return Predictor(json.loads(path.read_text()))


@cache
def context_metadata(objective: str, context: str) -> dict[str, Any]:
    module, panel, _ = previous.inputs(str(SOURCE))
    spines = [
        previous.load_spine(SOURCE, context, objective, index).to_json()
        for index in range(len(panel[f"{objective}_aggregation_weights"]))
    ]
    del module
    q = panel["aggregate"][np.flatnonzero(panel["calibration_mask"])[0]]
    assert np.all(q > 0)
    return {
        "alpha": float(panel["alpha"]),
        "q": q.tolist(),
        "c_total": (panel["c0"] + panel["c1"]).tolist(),
        "spines": spines,
        "aggregation_weights": panel[f"{objective}_aggregation_weights"].tolist(),
    }


def spine_values(
    weights: np.ndarray, spines: list[dict[str, Any]], c_total: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    module, _, _ = previous.inputs(str(SOURCE))
    eta, levels = [], []
    for record in spines:
        spine = module.TaskFit.from_json(record)
        design = module.design_matrix(weights * c_total, spine.shape)
        linear = spine.head.intercept + design @ spine.head.coefficients
        eta.append(linear)
        levels.append(spine.head.floor + np.exp(np.clip(linear, -30, 30)))
    return np.column_stack(levels), np.column_stack(eta)


def coordinates(weights: np.ndarray, model: str, metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    alpha = float(metadata["alpha"])
    a = alpha * weights[:, 0] + (1 - alpha) * weights[:, 1]
    delta = weights[:, 1] - weights[:, 0]
    if model in MODELS[:2]:
        matrix = np.hstack([delta, delta**2 / (a + np.asarray(metadata["q"]))])
        return matrix, np.zeros_like(matrix)
    if model == "CRE2-011":
        c_total = np.asarray(metadata["c_total"])
        _, a = spine_values(a, metadata["spines"], c_total)
        _, eta0 = spine_values(weights[:, 0], metadata["spines"], c_total)
        _, eta1 = spine_values(weights[:, 1], metadata["spines"], c_total)
        delta = eta1 - eta0
    return np.hstack([a, delta]), np.hstack([a, np.zeros_like(delta)])


def feature_scale(matrix: np.ndarray, model: str) -> np.ndarray:
    scale = np.sqrt(np.mean(matrix**2, axis=0))
    if model in MODELS[2:]:
        width = matrix.shape[1] // 2
        scale[:width] = np.std(matrix[:, :width], axis=0)
    return np.where(scale > 1e-10, scale, 1.0)


def difference_kernel(
    matrix: np.ndarray, tied: np.ndarray, centers: np.ndarray, tied_centers: np.ndarray, bandwidth: float
) -> np.ndarray:
    """Evaluate a PSD kernel on differences of policy and tied feature maps."""
    denominator = 2 * matrix.shape[1] * bandwidth**2
    return (
        np.exp(-cdist(matrix, centers, metric="sqeuclidean") / denominator)
        - np.exp(-cdist(tied, centers, metric="sqeuclidean") / denominator)
        - np.exp(-cdist(matrix, tied_centers, metric="sqeuclidean") / denominator)
        + np.exp(-cdist(tied, tied_centers, metric="sqeuclidean") / denominator)
    )


def config_grid(model: str) -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = [{"zero": True, "ridge": 0.0, "rank": 0, "bandwidth": 1.0}]
    ranks = (1, 2, 4, 0) if model == "CRE2-010" else (0,)
    bandwidths = (2.0, 1.0, 0.5) if model in MODELS[2:] else (1.0,)
    for ridge in RIDGES:
        for rank in ranks:
            for bandwidth in bandwidths:
                grid.append({"zero": False, "ridge": ridge, "rank": rank, "bandwidth": bandwidth})
    return grid


def constrained_coefficients(design: np.ndarray, response: np.ndarray, ridge: float) -> np.ndarray:
    """Eliminate the signed block exactly before nonnegative even-cost solves."""
    width = design.shape[1] // 2
    augmented = np.vstack([design, np.sqrt(len(design) * ridge) * np.eye(design.shape[1])])
    target = np.vstack([response, np.zeros((design.shape[1], response.shape[1]))])
    odd, even = augmented[:, :width], augmented[:, width:]
    gram = odd.T @ odd
    odd_even = np.linalg.solve(gram, odd.T @ even)
    odd_target = np.linalg.solve(gram, odd.T @ target)
    residual_even = even - odd @ odd_even
    residual_target = target - odd @ odd_target
    positive = np.column_stack([nnls(residual_even, y, maxiter=1000)[0] for y in residual_target.T])
    return np.vstack([odd_target - odd_even @ positive, positive])


def fit_response(
    matrix: np.ndarray, tied: np.ndarray, response: np.ndarray, model: str, config: dict[str, Any]
) -> dict[str, Any]:
    if config["zero"]:
        return {}
    scale = feature_scale(matrix, model)
    design = matrix / scale
    response_scale = np.maximum(np.sqrt(np.mean(response**2, axis=0)), 1e-10)
    normalized = response / response_scale
    if model == "CRE2-009":
        coefficients = constrained_coefficients(design, normalized, config["ridge"])
    elif model == "CRE2-010":
        gram = design.T @ design + len(matrix) * config["ridge"] * np.eye(design.shape[1])
        coefficients = np.linalg.solve(gram, design.T @ normalized)
        if config["rank"]:
            _, _, right = np.linalg.svd(design @ coefficients, full_matrices=False)
            directions = right[: config["rank"]].T
            coefficients = coefficients @ directions @ directions.T
    else:
        kernel = difference_kernel(design, tied / scale, design, tied / scale, config["bandwidth"])
        coefficients = np.linalg.solve(kernel + len(matrix) * config["ridge"] * np.eye(len(matrix)), normalized)
    fitted = {"scale": scale.tolist(), "coefficients": (coefficients * response_scale).tolist()}
    if model in MODELS[2:]:
        fitted |= {"centers": design.tolist(), "tied_centers": (tied / scale).tolist()}
    return fitted


def response_prediction(
    matrix: np.ndarray, tied: np.ndarray, model: str, config: dict[str, Any], fitted: dict[str, Any], components: int
) -> np.ndarray:
    if config["zero"]:
        return np.zeros((len(matrix), components))
    scale = np.asarray(fitted["scale"])
    if model in MODELS[:2]:
        return (matrix / scale) @ np.asarray(fitted["coefficients"])
    return difference_kernel(
        matrix / scale,
        tied / scale,
        np.asarray(fitted["centers"]),
        np.asarray(fitted["tied_centers"]),
        config["bandwidth"],
    ) @ np.asarray(fitted["coefficients"])


def source_identity(output: Path) -> dict[str, str]:
    paths = [Path(__file__), output / "PROTOCOL.md", SOURCE / "inputs/panel.npz", SOURCE / "inputs/splits.npz"]
    paths += [SOURCE / "inputs/single_phase.py", HERE / "fit_two_phase_link_transfer_20260907.py"]
    return {str(path): previous.file_hash(path) for path in paths}


def structural_checks(output: Path) -> None:
    """Check geometry and noiseless recoverability before observed-response fits."""
    _, panel, splits = previous.inputs(str(SOURCE))
    metadata = context_metadata("uncheatable", "outer0")
    weights = panel["weights"]
    alpha = metadata["alpha"]
    a = alpha * weights[:, 0] + (1 - alpha) * weights[:, 1]
    assert np.max(np.abs(a - panel["aggregate"])) < 1e-12
    tied_weights = np.repeat(a[:, None, :], 2, axis=1)
    delta = weights[:, 1] - weights[:, 0]
    opposite = np.stack([a + (1 - alpha) * delta, a - alpha * delta], axis=1)
    design, _ = coordinates(weights, "CRE2-009", metadata)
    reversed_design, _ = coordinates(opposite, "CRE2-009", metadata)
    assert np.max(np.abs(design[:, :39] + reversed_design[:, :39])) < 1e-12
    assert np.max(np.abs(design[:, 39:] - reversed_design[:, 39:])) < 1e-12
    train_a, _ = previous.selected_pairs(panel, splits["outer0_train"])
    test_a, _ = previous.selected_pairs(panel, splits["outer0_test"])
    scale = feature_scale(design[train_a], "CRE2-009")
    rank = int(np.linalg.matrix_rank(design[train_a] / scale))
    rng = np.random.default_rng(20260907)
    checks: dict[str, Any] = {"linear_training_columns": 78, "linear_training_rank": rank, "train_pairs": len(train_a)}
    for model in MODELS:
        matrix, zero = coordinates(weights, model, metadata)
        tied_matrix, tied_zero = coordinates(tied_weights, model, metadata)
        if model in MODELS[:2]:
            assert np.max(np.abs(tied_matrix)) < 1e-12
            coefficients = rng.normal(size=(78, 2))
            coefficients[39:] = np.abs(coefficients[39:])
            synthetic = matrix / feature_scale(matrix[train_a], model) @ coefficients
            config = {"zero": False, "ridge": 1e-10, "rank": 2, "bandwidth": 1.0}
        else:
            scale = feature_scale(matrix[train_a], model)
            kernel = difference_kernel(matrix / scale, zero / scale, matrix[train_a] / scale, zero[train_a] / scale, 1.0)
            eigenvalues = np.linalg.eigvalsh(kernel[train_a])
            assert eigenvalues.min() > -1e-9
            zero_kernel = difference_kernel(
                tied_matrix / scale, tied_zero / scale, matrix[train_a] / scale, zero[train_a] / scale, 1.0
            )
            assert np.max(np.abs(zero_kernel)) < 1e-12
            synthetic = kernel @ rng.normal(size=(len(train_a), 2))
            config = {"zero": False, "ridge": 1e-10, "rank": 0, "bandwidth": 1.0}
        fitted = fit_response(matrix[train_a], zero[train_a], synthetic[train_a], model, config)
        predicted = response_prediction(matrix, zero, model, config, fitted, 2)
        relative = float(np.linalg.norm(predicted[test_a] - synthetic[test_a]) / np.linalg.norm(synthetic[test_a]))
        assert relative < 1e-4, (model, relative)
        if model == "CRE2-009":
            positive = np.asarray(fitted["coefficients"])[39:]
            assert np.min(positive) >= 0
        checks[model] = {"noiseless_held_pair_relative_error": relative, "tied_zero": True}
    checks["source_identity"] = source_identity(output)
    previous.write_json(output / "structural_checks.json", checks)
    print(json.dumps(checks, indent=2), flush=True)


def fit_cell(output: Path, objective: str, context: str, model: str) -> None:
    destination = output / "fits" / model / objective / context
    complete_path = destination / "complete.json"
    identity = source_identity(output)
    if complete_path.exists():
        complete = json.loads(complete_path.read_text())
        assert complete["identity"] == identity
        for name, digest in complete["sha256"].items():
            assert previous.file_hash(destination / name) == digest
        print(f"cached {model}/{objective}/{context}", flush=True)
        return
    _, panel, splits = previous.inputs(str(SOURCE))
    rows = previous.training_rows(panel, splits, context)
    train_a, train_t = previous.selected_pairs(panel, rows)
    assert panel["calibration_mask"][rows].sum() == 2
    response = panel[f"{objective}_outcomes"]
    weights = panel[f"{objective}_aggregation_weights"]
    grid = config_grid(model)
    errors, counts = np.zeros(len(grid)), np.zeros(len(grid), dtype=int)
    source_paths = []
    inner_rows = []
    for inner in range(3):
        prefix = f"{context}_inner{inner}"
        inner_train = splits[f"{prefix}_train"]
        inner_test = splits[f"{prefix}_test"]
        assert panel["calibration_mask"][inner_train].sum() == 2
        assert not panel["calibration_mask"][inner_test].any()
        ia, it = previous.selected_pairs(panel, inner_train)
        va, vt = previous.selected_pairs(panel, inner_test)
        inner_rows.append({"prefix": prefix, "train_pairs": ia.tolist(), "test_pairs": va.tolist()})
        if not len(va):
            continue
        metadata = context_metadata(objective, prefix)
        matrix, zero = coordinates(panel["weights"], model, metadata)
        y, vy = response[ia] - response[it], response[va] - response[vt]
        for index, config in enumerate(grid):
            fitted = fit_response(matrix[ia], zero[ia], y, model, config)
            predicted = response_prediction(matrix[va], zero[va], model, config, fitted, response.shape[1])
            errors[index] += np.sum(((predicted - vy) @ weights) ** 2)
            counts[index] += len(va)
        source_paths += [SOURCE / "spines" / prefix / f"{objective}_c{i}.json" for i in range(response.shape[1])]
    assert counts.min() > 0
    best = int(np.argmin(errors / counts))
    config = grid[best]
    metadata = context_metadata(objective, context)
    matrix, zero = coordinates(panel["weights"], model, metadata)
    fitted = fit_response(matrix[train_a], zero[train_a], response[train_a] - response[train_t], model, config)
    record = (
        metadata
        | fitted
        | {
            "model": model,
            "objective": objective,
            "context": context,
            "config": config,
            "train_rows": rows.tolist(),
            "train_asymmetric_rows": train_a.tolist(),
            "train_tied_rows": train_t.tolist(),
            "inner_rows": inner_rows,
            "selected_inner_mse": float(errors[best] / counts[best]),
        }
    )
    predictor = Predictor(record)
    components = predictor.predict_components(panel["weights"])
    prediction = components @ weights
    tied_weights = np.repeat(panel["aggregate"][:, None, :], 2, axis=1)
    tied_components = predictor.predict_components(tied_weights)
    assert np.max(np.abs(components[panel["physical_tied"]] - tied_components[panel["physical_tied"]])) < 1e-12
    assert np.isfinite(components).all()
    record["minimum_prediction"] = float(np.min(prediction))
    record["minimum_component_prediction"] = float(np.min(components))
    floors = np.array([item["floor"] for item in metadata["spines"]])
    record["below_floor_component_cells"] = int(np.sum(components < floors))
    record["negative_component_cells"] = int(np.sum(components < 0))
    scored = ~panel["calibration_mask"] if context == "final" else panel["outer_fold"] == int(context[-1])
    destination.mkdir(parents=True, exist_ok=True)
    previous.write_json(destination / "fit.json", record)
    previous.write_json(
        destination / "sweep.json",
        [
            config | {"mse": float(error / count), "n": int(count)}
            for config, error, count in zip(grid, errors, counts, strict=True)
        ],
    )
    np.savez_compressed(destination / "components.npz", predicted=components, tied_predicted=tied_components)
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
            "measured": panel[f"{objective}_aggregate"],
            "predicted": prediction,
        }
    ).to_csv(destination / "predictions.csv", index=False)
    source_paths += [SOURCE / "spines" / context / f"{objective}_c{i}.json" for i in range(response.shape[1])]
    previous.write_json(
        complete_path,
        {
            "identity": identity,
            "spine_sha256": {str(path): previous.file_hash(path) for path in source_paths},
            "sha256": {
                name: previous.file_hash(destination / name)
                for name in ("fit.json", "sweep.json", "components.npz", "predictions.csv")
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
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    if args.mode == "checks":
        structural_checks(args.output)
        return
    if args.mode == "fit":
        checks = json.loads((args.output / "structural_checks.json").read_text())
        assert checks["source_identity"] == source_identity(args.output)
        for model in args.models:
            for objective in previous.OBJECTIVES:
                for context in previous.CONTEXTS:
                    fit_cell(args.output, objective, context, model)
    paths = [
        args.output / "fits" / model / objective / context / "predictions.csv"
        for model in args.models
        for objective in previous.OBJECTIVES
        for context in previous.CONTEXTS
    ]
    pd.concat([pd.read_csv(path) for path in paths], ignore_index=True).to_csv(
        args.output / "predictions.csv", index=False
    )


if __name__ == "__main__":
    main()
