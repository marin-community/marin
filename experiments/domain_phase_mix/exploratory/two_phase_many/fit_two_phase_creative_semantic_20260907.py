# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Fit adaptive parametric WSPU semantic phase comparisons without LM jobs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fit_two_phase_creative_geometry_20260907 as geometry
import fit_two_phase_link_transfer_20260907 as previous
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = geometry.SOURCE
OUTPUT = HERE / "reference_outputs/two_phase_creative_sweep_20260907/semantic_followup"
MODELS = ("CRE2-015", "CRE2-016")
ESTIMATORS = {"CRE2-015": "CRE2-010", "CRE2-016": "CRE2-009"}


@dataclass(frozen=True)
class Predictor:
    """Predict component losses and their fixed weighted objective."""

    record: dict[str, Any]

    def predict_components(self, weights: np.ndarray) -> np.ndarray:
        record = self.record
        alpha = record["alpha"]
        aggregate = alpha * weights[:, 0] + (1 - alpha) * weights[:, 1]
        levels, _ = geometry.spine_values(aggregate, record["spines"], np.asarray(record["c_total"]))
        if record["config"]["zero"]:
            return levels
        design = semantic_design(weights, record["model"], record)
        return levels + (design / np.asarray(record["scale"])) @ np.asarray(record["coefficients"])

    def predict(self, weights: np.ndarray) -> np.ndarray:
        """Return objective BPB for weights with shape (n, 2, 39)."""
        return self.predict_components(np.asarray(weights)) @ np.asarray(self.record["aggregation_weights"])


def load_predictor(output: Path, objective: str, context: str, model: str) -> Predictor:
    """Load a completed semantic fit for arbitrary-policy prediction."""
    return Predictor(json.loads((output / "fits" / model / objective / context / "fit.json").read_text()))


def semantic_design(weights: np.ndarray, model: str, metadata: dict[str, Any]) -> np.ndarray:
    c_total = np.asarray(metadata["c_total"])
    _, eta0 = geometry.spine_values(weights[:, 0], metadata["spines"], c_total)
    _, eta1 = geometry.spine_values(weights[:, 1], metadata["spines"], c_total)
    delta = eta1 - eta0
    return delta if model == "CRE2-015" else np.hstack([delta, delta**2])


def identity(output: Path) -> dict[str, str]:
    paths = [
        Path(__file__),
        Path(geometry.__file__),
        Path(previous.__file__),
        output / "PROTOCOL.md",
        SOURCE / "inputs/panel.npz",
        SOURCE / "inputs/splits.npz",
        SOURCE / "inputs/single_phase.py",
    ]
    return {str(path): previous.file_hash(path) for path in paths}


def structural_checks(output: Path) -> None:
    """Verify tied restriction and semantic sign parity before actual fits."""
    _, panel, splits = previous.inputs(str(SOURCE))
    train, _ = previous.selected_pairs(panel, splits["outer0_train"])
    test, _ = previous.selected_pairs(panel, splits["outer0_test"])
    weights = panel["weights"]
    tied_weights = np.repeat(panel["aggregate"][:, None, :], 2, axis=1)
    rng = np.random.default_rng(20260907)
    records = []
    for objective in previous.OBJECTIVES:
        metadata = geometry.context_metadata(objective, "outer0")
        for model in MODELS:
            estimator = ESTIMATORS[model]
            design = semantic_design(weights, model, metadata)
            tied_design = semantic_design(tied_weights, model, metadata)
            reverse_design = semantic_design(weights[:, ::-1], model, metadata)
            width = design.shape[1] if model == "CRE2-015" else design.shape[1] // 2
            assert np.max(np.abs(tied_design)) < 1e-12
            assert np.max(np.abs(design[:, :width] + reverse_design[:, :width])) < 1e-12
            if model == "CRE2-016":
                assert np.max(np.abs(design[:, width:] - reverse_design[:, width:])) < 1e-12
            scale = geometry.feature_scale(design[train], estimator)
            normalized = design / scale
            coefficients = np.zeros((design.shape[1], 2))
            coefficients[:4] = rng.normal(size=(4, 2))
            if model == "CRE2-016":
                coefficients[width : width + 4] = np.abs(rng.normal(size=(4, 2)))
            synthetic = normalized @ coefficients
            config = {"zero": False, "ridge": 1e-10, "rank": 2 if model == "CRE2-015" else 0, "bandwidth": 1.0}
            fitted = geometry.fit_response(
                design[train], np.zeros_like(design[train]), synthetic[train], estimator, config
            )
            prediction = geometry.response_prediction(design, np.zeros_like(design), estimator, config, fitted, 2)
            relative = float(np.linalg.norm(prediction[test] - synthetic[test]) / np.linalg.norm(synthetic[test]))
            assert relative < 0.01, (model, objective, relative)
            if model == "CRE2-016":
                assert np.min(np.asarray(fitted["coefficients"])[width:]) >= 0
            singular = np.linalg.svd(normalized[train], compute_uv=False)
            nonzero = singular[singular > singular[0] * max(normalized[train].shape) * np.finfo(float).eps]
            records.append(
                {
                    "model": model,
                    "objective": objective,
                    "columns": design.shape[1],
                    "rank": int(np.linalg.matrix_rank(normalized[train])),
                    "condition_nonzero": float(nonzero[0] / nonzero[-1]),
                    "noiseless_held_pair_relative_error": relative,
                    "tied_zero": True,
                    "semantic_parity": True,
                }
            )
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
    rows = previous.training_rows(panel, splits, context)
    train_a, train_t = previous.selected_pairs(panel, rows)
    assert panel["calibration_mask"][rows].sum() == 2
    response = panel[f"{objective}_outcomes"]
    weights = panel[f"{objective}_aggregation_weights"]
    estimator = ESTIMATORS[model]
    configs = geometry.config_grid(estimator)
    errors, counts = np.zeros(len(configs)), np.zeros(len(configs), dtype=int)
    inner_rows, source_paths = [], []
    for inner in range(3):
        prefix = f"{context}_inner{inner}"
        inner_train = splits[f"{prefix}_train"]
        inner_test = splits[f"{prefix}_test"]
        assert panel["calibration_mask"][inner_train].sum() == 2
        assert not panel["calibration_mask"][inner_test].any()
        ia, it = previous.selected_pairs(panel, inner_train)
        va, vt = previous.selected_pairs(panel, inner_test)
        assert not set(panel["groups"][ia]).intersection(panel["groups"][va])
        inner_rows.append({"prefix": prefix, "train_pairs": ia.tolist(), "test_pairs": va.tolist()})
        if not len(va):
            continue
        metadata = geometry.context_metadata(objective, prefix)
        matrix = semantic_design(panel["weights"], model, metadata)
        y, vy = response[ia] - response[it], response[va] - response[vt]
        for index, config in enumerate(configs):
            fitted = geometry.fit_response(matrix[ia], np.zeros_like(matrix[ia]), y, estimator, config)
            prediction = geometry.response_prediction(
                matrix[va], np.zeros_like(matrix[va]), estimator, config, fitted, response.shape[1]
            )
            errors[index] += np.sum(((prediction - vy) @ weights) ** 2)
            counts[index] += len(va)
        source_paths += [SOURCE / "spines" / prefix / f"{objective}_c{i}.json" for i in range(response.shape[1])]
    assert counts.min() > 0
    best = int(np.argmin(errors / counts))
    config = configs[best]
    metadata = geometry.context_metadata(objective, context)
    matrix = semantic_design(panel["weights"], model, metadata)
    fitted = geometry.fit_response(
        matrix[train_a], np.zeros_like(matrix[train_a]), response[train_a] - response[train_t], estimator, config
    )
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
    raw_scale = np.sqrt(np.mean(matrix[train_a] ** 2, axis=0))
    scale = np.where(raw_scale > 1e-10, raw_scale, 1.0)
    singular = np.linalg.svd(matrix[train_a] / scale, compute_uv=False)
    nonzero = singular[singular > singular[0] * max(matrix[train_a].shape) * np.finfo(float).eps]
    record |= {
        "coordinate_count": matrix.shape[1],
        "coordinate_rank": int(np.linalg.matrix_rank(matrix[train_a] / scale)),
        "coordinate_nonzero_condition": float(nonzero[0] / nonzero[-1]),
        "zero_scale_coordinates": int(np.sum(raw_scale <= 1e-10)),
        "minimum_prediction": float(prediction.min()),
        "minimum_component_prediction": float(components.min()),
        "below_floor_component_cells": int(np.sum(components < np.array([s["floor"] for s in metadata["spines"]]))),
        "negative_component_cells": int(np.sum(components < 0)),
    }
    if model == "CRE2-016" and not config["zero"]:
        coefficients = np.asarray(fitted["coefficients"])
        assert np.min(coefficients[matrix.shape[1] // 2 :]) >= 0
    scored = ~panel["calibration_mask"] if context == "final" else panel["outer_fold"] == int(context[-1])
    destination.mkdir(parents=True, exist_ok=True)
    previous.write_json(destination / "fit.json", record)
    previous.write_json(
        destination / "sweep.json",
        [c | {"mse": float(e / n), "n": int(n)} for c, e, n in zip(configs, errors, counts, strict=True)],
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
        marker,
        {
            "identity": identity(output),
            "spine_sha256": {str(p): previous.file_hash(p) for p in source_paths},
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
    args = parser.parse_args()
    if args.mode == "checks":
        structural_checks(args.output)
        return
    if args.mode == "fit":
        checks = json.loads((args.output / "structural_checks.json").read_text())
        assert checks["identity"] == identity(args.output)
        for model in MODELS:
            for objective in previous.OBJECTIVES:
                for context in previous.CONTEXTS:
                    fit_cell(args.output, objective, context, model)
    paths = [
        args.output / "fits" / model / objective / context / "predictions.csv"
        for model in MODELS
        for objective in previous.OBJECTIVES
        for context in previous.CONTEXTS
    ]
    pd.concat([pd.read_csv(path) for path in paths], ignore_index=True).to_csv(
        args.output / "predictions.csv", index=False
    )


if __name__ == "__main__":
    main()
