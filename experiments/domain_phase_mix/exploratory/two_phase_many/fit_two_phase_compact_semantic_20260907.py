# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Fit registered task-local benefit/harm contrast ablations offline."""

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
OUTPUT = HERE / "reference_outputs/two_phase_refinement_20260907/compact_semantic"
MODELS = ("CRE2-021", "CRE2-022")
ESTIMATOR = "CRE2-009"


def compact_design(weights: np.ndarray, model: str, metadata: dict[str, Any]) -> np.ndarray:
    """Return per-component signed contrasts followed by their squared costs."""
    module, _, _ = previous.inputs(str(geometry.SOURCE))
    components = []
    inventory = np.asarray(metadata["c_total"])
    width = weights.shape[-1]
    for record in metadata["spines"]:
        spine = module.TaskFit.from_json(record)
        difference = module.design_matrix(weights[:, 1] * inventory, spine.shape) - module.design_matrix(
            weights[:, 0] * inventory, spine.shape
        )
        benefit = -difference[:, :width] @ spine.head.coefficients[:width]
        harm = difference[:, width:] @ spine.head.coefficients[width:]
        signed = (-benefit + harm)[:, None] if model == "CRE2-021" else np.column_stack([benefit, harm])
        components.append(np.column_stack([signed, signed**2]))
    return np.stack(components, axis=1)


def fit_components(design: np.ndarray, response: np.ndarray, config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        geometry.fit_response(design[:, t], np.zeros_like(design[:, t]), response[:, [t]], ESTIMATOR, config)
        for t in range(response.shape[1])
    ]


def phase_prediction(design: np.ndarray, config: dict[str, Any], heads: list[dict[str, Any]]) -> np.ndarray:
    return np.column_stack(
        [
            geometry.response_prediction(design[:, t], np.zeros_like(design[:, t]), ESTIMATOR, config, head, 1)[:, 0]
            for t, head in enumerate(heads)
        ]
    )


@dataclass(frozen=True)
class Predictor:
    record: dict[str, Any]

    def predict_components(self, weights: np.ndarray) -> np.ndarray:
        metadata = self.record
        aggregate = metadata["alpha"] * weights[:, 0] + (1 - metadata["alpha"]) * weights[:, 1]
        base, _ = geometry.spine_values(aggregate, metadata["spines"], np.asarray(metadata["c_total"]))
        design = compact_design(weights, metadata["model"], metadata)
        return base + phase_prediction(design, metadata["config"], metadata["heads"])

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return self.predict_components(weights) @ np.asarray(self.record["aggregation_weights"])


def load_predictor(output: Path, model: str, objective: str, context: str) -> Predictor:
    return Predictor(json.loads((output / "fits" / model / objective / context / "fit.json").read_text()))


def identity(output: Path) -> dict[str, str]:
    paths = [
        Path(__file__),
        Path(geometry.__file__),
        Path(previous.__file__),
        output / "PROTOCOL.md",
        geometry.SOURCE / "inputs/panel.npz",
        geometry.SOURCE / "inputs/splits.npz",
        geometry.SOURCE / "inputs/single_phase.py",
    ]
    return {str(p): previous.file_hash(p) for p in paths}


def structural_checks(output: Path) -> None:
    _, panel, splits = previous.inputs(str(geometry.SOURCE))
    train, _ = previous.selected_pairs(panel, splits["outer0_train"])
    test, _ = previous.selected_pairs(panel, splits["outer0_test"])
    rng = np.random.default_rng(20260907)
    checks = []
    config = {"zero": False, "ridge": 1e-10, "rank": 0, "bandwidth": 1.0}
    for objective in previous.OBJECTIVES:
        metadata = geometry.context_metadata(objective, "outer0")
        _, eta0 = geometry.spine_values(panel["weights"][:, 0], metadata["spines"], np.asarray(metadata["c_total"]))
        _, eta1 = geometry.spine_values(panel["weights"][:, 1], metadata["spines"], np.asarray(metadata["c_total"]))
        for model in MODELS:
            design = compact_design(panel["weights"], model, metadata)
            tied = compact_design(np.repeat(panel["aggregate"][:, None, :], 2, axis=1), model, metadata)
            assert np.max(np.abs(tied)) == 0
            net = design[:, :, 0] if model == "CRE2-021" else -design[:, :, 0] + design[:, :, 1]
            assert np.max(np.abs(net - (eta1 - eta0))) < 1e-12
            for t in range(design.shape[1]):
                x = design[:, t]
                scale = geometry.feature_scale(x[train], ESTIMATOR)
                normalized = x / scale
                coefficient = rng.normal(size=x.shape[1])
                coefficient[x.shape[1] // 2 :] = np.abs(coefficient[x.shape[1] // 2 :])
                response = (normalized @ coefficient)[:, None]
                head = geometry.fit_response(x[train], np.zeros_like(x[train]), response[train], ESTIMATOR, config)
                pred = geometry.response_prediction(x[test], np.zeros_like(x[test]), ESTIMATOR, config, head, 1)
                error = float(np.linalg.norm(pred - response[test]) / max(np.linalg.norm(response[test]), 1e-12))
                assert error < 1e-5, (model, objective, t, error)
                singular = np.linalg.svd(normalized[train], compute_uv=False)
                rank = int(np.linalg.matrix_rank(normalized[train]))
                checks.append(
                    {
                        "model": model,
                        "objective": objective,
                        "component": t,
                        "rank": rank,
                        "columns": x.shape[1],
                        "synthetic_relative_prediction_error": error,
                        "singular_values": singular.tolist(),
                    }
                )
    previous.write_json(output / "structural_checks.json", {"identity": identity(output), "checks": checks})
    print(f"Passed {len(checks)} structural and conditional recovery checks.", flush=True)


def fit_cell(output: Path, model: str, objective: str, context: str) -> None:
    destination = output / "fits" / model / objective / context
    marker = destination / "complete.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        assert record["identity"] == identity(output)
        for path, digest in record["spine_sha256"].items():
            assert previous.file_hash(Path(path)) == digest
        for name, digest in record["sha256"].items():
            assert previous.file_hash(destination / name) == digest
        print(f"cached {model}/{objective}/{context}", flush=True)
        return
    _, panel, splits = previous.inputs(str(geometry.SOURCE))
    rows = previous.training_rows(panel, splits, context)
    train_a, train_t = previous.selected_pairs(panel, rows)
    response = panel[f"{objective}_outcomes"]
    aggregate_weights = panel[f"{objective}_aggregation_weights"]
    configs = geometry.config_grid(ESTIMATOR)
    errors, counts = np.zeros(len(configs)), np.zeros(len(configs), dtype=int)
    source_paths, fold_records = [], []
    for inner in range(3):
        prefix = f"{context}_inner{inner}"
        training, testing = splits[f"{prefix}_train"], splits[f"{prefix}_test"]
        assert panel["calibration_mask"][training].sum() == 2
        assert not panel["calibration_mask"][testing].any()
        ia, it = previous.selected_pairs(panel, training)
        va, vt = previous.selected_pairs(panel, testing)
        assert not set(panel["groups"][ia]).intersection(panel["groups"][va])
        fold_records.append({"context": prefix, "train_pairs": ia.tolist(), "test_pairs": va.tolist()})
        if not len(va):
            continue
        metadata = geometry.context_metadata(objective, prefix)
        design = compact_design(panel["weights"], model, metadata)
        y, vy = response[ia] - response[it], response[va] - response[vt]
        for index, config in enumerate(configs):
            heads = fit_components(design[ia], y, config)
            pred = phase_prediction(design[va], config, heads)
            errors[index] += np.sum(((pred - vy) @ aggregate_weights) ** 2)
            counts[index] += len(va)
        source_paths += [
            geometry.SOURCE / "spines" / prefix / f"{objective}_c{i}.json" for i in range(response.shape[1])
        ]
    assert counts.min() > 0
    best = int(np.argmin(errors / counts))
    config = configs[best]
    metadata = geometry.context_metadata(objective, context)
    design = compact_design(panel["weights"], model, metadata)
    heads = fit_components(design[train_a], response[train_a] - response[train_t], config)
    fitted = metadata | {
        "model": model,
        "objective": objective,
        "context": context,
        "config": config,
        "heads": heads,
        "train_rows": rows.tolist(),
        "inner_folds": fold_records,
        "selected_inner_mse": float(errors[best] / counts[best]),
    }
    prediction = Predictor(fitted).predict_components(panel["weights"])
    assert np.isfinite(prediction).all()
    floors = np.array([s["floor"] for s in metadata["spines"]])
    fitted["observed_component_floor_crossings"] = int((prediction < floors).sum())
    source_paths += [geometry.SOURCE / "spines" / context / f"{objective}_c{i}.json" for i in range(response.shape[1])]
    scored = ~panel["calibration_mask"]
    if context != "final":
        scored &= panel["outer_fold"] == int(context[-1])
    destination.mkdir(parents=True, exist_ok=True)
    previous.write_json(destination / "fit.json", fitted)
    previous.write_json(
        destination / "sweep.json",
        [c | {"mse": float(e / n), "n": int(n)} for c, e, n in zip(configs, errors, counts, strict=True)],
    )
    np.savez_compressed(destination / "components.npz", prediction=prediction)
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
            "predicted": prediction @ aggregate_weights,
        }
    ).to_csv(destination / "predictions.csv", index=False)
    previous.write_json(
        marker,
        {
            "identity": identity(output),
            "spine_sha256": {str(p): previous.file_hash(p) for p in source_paths},
            "sha256": {
                name: previous.file_hash(destination / name)
                for name in ["fit.json", "sweep.json", "components.npz", "predictions.csv"]
            },
        },
    )
    print(f"fit {model}/{objective}/{context}: {config}, CV RMSE={np.sqrt(errors[best]/counts[best]):.6f}", flush=True)


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
                    fit_cell(args.output, model, objective, context)
    paths = [
        args.output / "fits" / m / o / c / "predictions.csv"
        for m in MODELS
        for o in previous.OBJECTIVES
        for c in previous.CONTEXTS
    ]
    pd.concat([pd.read_csv(p) for p in paths], ignore_index=True).to_csv(args.output / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
