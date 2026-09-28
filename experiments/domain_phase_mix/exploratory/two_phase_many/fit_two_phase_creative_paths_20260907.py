# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Screen four registered WSPU path states with nested, pinned calibration folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import fit_two_phase_link_transfer_20260907 as previous
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE / "reference_outputs/two_phase_link_transfer_20260907"
OUTPUT = HERE / "reference_outputs/two_phase_creative_sweep_20260907/path_states"
MODELS = ("CRE2-005", "CRE2-006", "CRE2-007", "CRE2-008")
RATES = (1.0, 4.0, 16.0)
PENALTIES = (10.0, 1.0, 0.1, 0.01, 0.0)
LOG_LIMIT = 30.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@cache
def spine_pack(context: str, objective: str) -> dict[str, np.ndarray]:
    """Load only the independently fitted spines belonging to this context."""
    _, panel, _ = previous.inputs(str(PREVIOUS))
    width = len(panel["buckets"])
    spines = [previous.load_spine(PREVIOUS, context, objective, i) for i in range(len(panel[f"{objective}_components"]))]
    return {
        "rate": np.array([s.shape["rate"] for s in spines])[None, :, None],
        "power": np.array([s.shape["power"] for s in spines])[None, :, None],
        "threshold": np.array([s.shape["threshold"] for s in spines])[None, :, None],
        "alpha": np.array([s.head.coefficients[:width] for s in spines])[None, :, :],
        "beta": np.array([s.head.coefficients[width:] for s in spines])[None, :, :],
        "intercept": np.array([s.head.intercept for s in spines]),
        "floor": np.array([s.head.floor for s in spines]),
        "weights": panel[f"{objective}_aggregation_weights"],
        "c0": panel["c0"],
        "c1": panel["c1"],
    }


def benefit(exposure: np.ndarray, pack: dict[str, np.ndarray]) -> np.ndarray:
    return -np.expm1(-np.maximum(exposure * pack["rate"], 0) ** pack["power"])


def harm(exposure: np.ndarray, pack: dict[str, np.ndarray]) -> np.ndarray:
    return np.logaddexp(np.log1p(exposure) - pack["threshold"], 0) ** 2


def path_statistics(weights: np.ndarray, pack: dict[str, np.ndarray], bins: int = 16) -> dict[str, np.ndarray]:
    """Return each fixed-rate path statistic, with axes rate, row, component."""
    w0, w1 = weights[:, 0, None, :], weights[:, 1, None, :]
    e0, e1 = w0 * pack["c0"], w1 * pack["c1"]
    total = e0 + e1
    rho = float(np.ravel(pack["c0"] / (pack["c0"] + pack["c1"]))[0])
    absence = (1 - rho) * (1 - w1)
    b0, b1, bt = benefit(e0, pack), benefit(e1, pack), benefit(total, pack)
    h0, ht = harm(e0, pack) - harm(np.zeros_like(e0), pack), harm(total, pack)
    competition = np.sum(w1 * b1, axis=-1, keepdims=True) - w1 * b1
    outputs: dict[str, list[np.ndarray]] = {model: [] for model in MODELS}
    for rate in RATES:
        retained = b0 * np.exp(-rate * absence / (1 + e1)) + bt - b0
        repaired = ht - (-np.expm1(-rate * absence)) * h0
        competitive = b0 / (1 + rate * competition / (1 + e1)) + bt - b0
        outputs[MODELS[0]].append(-np.sum(pack["alpha"] * retained, axis=-1))
        outputs[MODELS[1]].append(np.sum(pack["beta"] * repaired, axis=-1))
        outputs[MODELS[2]].append(-np.sum(pack["alpha"] * competitive, axis=-1))
    aged = np.zeros((len(RATES), len(weights), len(pack["weights"])))
    last = benefit(np.zeros_like(e0), pack)
    for phase, fraction in ((0, rho), (1, 1 - rho)):
        for step in range(bins):
            exposure = e0 * (step + 1) / bins if phase == 0 else e0 + e1 * (step + 1) / bins
            current = benefit(exposure, pack)
            increment = np.sum(pack["alpha"] * (current - last), axis=-1)
            midpoint = fraction * (step + 0.5) / bins + (rho if phase else 0)
            for index, rate in enumerate(RATES):
                aged[index] -= np.exp(-rate * (1 - midpoint)) * increment
            last = current
    return {model: np.stack(values) for model, values in outputs.items() if values} | {MODELS[3]: aged}


def features(weights: np.ndarray, pack: dict[str, np.ndarray], bins: int = 16) -> dict[str, Any]:
    """Build tied-centered path contrasts for arbitrary (rows, 2, 39) policies."""
    total = weights[:, 0, :] * pack["c0"] + weights[:, 1, :] * pack["c1"]
    aggregate = total / (pack["c0"] + pack["c1"])
    tied = np.repeat(aggregate[:, None, :], 2, axis=1)
    eta = pack["intercept"] + np.sum(
        -pack["alpha"] * benefit(total[:, None, :], pack) + pack["beta"] * harm(total[:, None, :], pack), axis=-1
    )
    q = np.exp(np.clip(eta, -LOG_LIMIT, LOG_LIMIT))
    observed, reference = path_statistics(weights, pack, bins), path_statistics(tied, pack, bins)
    return {
        "q": q,
        "aggregate": (pack["floor"] + q) @ pack["weights"],
        "z": {m: observed[m] - reference[m] for m in MODELS},
    }


def delta_prediction(q: np.ndarray, z: np.ndarray, aggregation: np.ndarray, theta: float) -> np.ndarray:
    return (q * np.expm1(np.clip(theta * z, -LOG_LIMIT, LOG_LIMIT))) @ aggregation


def fit_amplitude(
    q: np.ndarray, z: np.ndarray, aggregation: np.ndarray, target: np.ndarray, penalty: float
) -> dict[str, Any]:
    """Fit one nonnegative coefficient with contribution-space ridge."""
    linear = (q * z) @ aggregation
    norm = float(np.linalg.norm(linear))
    if norm < 1e-14 or not len(target):
        return {"theta": 0.0, "cost": float(target @ target / 2), "optimality": 0.0, "success": True, "clipped": 0}
    regularization = np.sqrt(penalty) * norm

    def residual(theta: np.ndarray) -> np.ndarray:
        return np.r_[delta_prediction(q, z, aggregation, float(theta[0])) - target, regularization * theta[0]]

    def jacobian(theta: np.ndarray) -> np.ndarray:
        argument = theta[0] * z
        derivative = (
            q * z * np.exp(np.clip(argument, -LOG_LIMIT, LOG_LIMIT)) * (np.abs(argument) < LOG_LIMIT)
        ) @ aggregation
        return np.r_[derivative, regularization][:, None]

    fits = [
        least_squares(
            residual,
            np.array([start]),
            jac=jacobian,
            bounds=(0, np.inf),
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
            max_nfev=150,
        )
        for start in (1e-5, 1.0)
    ]
    best = min(fits, key=lambda fitted: fitted.cost)
    null_cost = float(target @ target / 2)
    theta = float(best.x[0]) if best.cost < null_cost else 0.0
    return {
        "theta": theta,
        "cost": min(float(best.cost), null_cost),
        "optimality": float(best.optimality),
        "success": bool(best.success),
        "clipped": int(np.sum(np.abs(theta * z) > LOG_LIMIT)),
    }


def predict_weights(weights: np.ndarray, objective: str, context: str, model: str, output: Path = OUTPUT) -> np.ndarray:
    """Predict arbitrary simplex policies using one saved fitted candidate."""
    fitted = json.loads((output / "cells" / objective / context / f"{model}.json").read_text())
    pack = spine_pack(context, objective)
    design = features(np.asarray(weights), pack)
    index = RATES.index(float(fitted["rate"]))
    return design["aggregate"] + delta_prediction(
        design["q"], design["z"][model][index], pack["weights"], fitted["theta"]
    )


def structural_checks(output: Path) -> None:
    """Verify the path geometry and conditional synthetic recovery before fitting."""
    _, panel, splits = previous.inputs(str(PREVIOUS))
    checks = []
    rng = np.random.default_rng(20260907)
    for objective in previous.OBJECTIVES:
        pack = spine_pack("outer0", objective)
        design = features(panel["weights"], pack)
        zero = features(np.zeros((1, 2, 39)), pack)
        perturbation = np.zeros((1, 2, 39))
        perturbation[0, 0, :2] = (1e-6, -1e-6)
        interior = np.full((1, 2, 39), 1 / 39)
        nearby = features(interior + perturbation, pack)
        interior_features = features(interior, pack)
        corners = np.zeros((2, 2, 39))
        corners[0, 0, 0], corners[0, 1, 1], corners[1, :, 0] = 1, 1, 1
        boundary = features(corners, pack)
        boundary_nearby = features(corners * (1 - 1e-6) + 1e-6 / 39, pack)
        converged = features(panel["weights"][:20], pack, bins=32)
        train_a, _ = previous.selected_pairs(panel, splits["outer0_train"])
        test_a, _ = previous.selected_pairs(panel, splits["outer0_test"])
        for model in MODELS:
            z = design["z"][model][1]
            signal = delta_prediction(design["q"], z, pack["weights"], 0.5)
            fitted = fit_amplitude(design["q"][train_a], z[train_a], pack["weights"], signal[train_a], 0.0)
            prediction = delta_prediction(design["q"][test_a], z[test_a], pack["weights"], fitted["theta"])
            noise = rng.normal(0, np.sqrt(2) * previous.NOISE_REFERENCE[objective], len(train_a))
            null = fit_amplitude(design["q"][train_a], z[train_a], pack["weights"], noise, 1.0)
            exact_null = fit_amplitude(design["q"][train_a], z[train_a], pack["weights"], np.zeros(len(train_a)), 0.0)
            linear_columns = np.stack(
                [(design["q"] * column) @ pack["weights"] for column in design["z"][model]], axis=1
            )
            normalized = linear_columns[train_a] / np.maximum(np.linalg.norm(linear_columns[train_a], axis=0), 1e-30)
            singular_values = np.linalg.svd(normalized, compute_uv=False)
            row = {
                "objective": objective,
                "model": model,
                "tied_error": float(np.max(np.abs(design["z"][model][:, panel["physical_tied"]]))),
                "zero_error": float(np.max(np.abs(zero["z"][model]))),
                "max_interior_derivative": float(
                    np.max(np.abs(nearby["z"][model] - interior_features["z"][model])) / 1e-6
                ),
                "boundary_finite": bool(np.isfinite(boundary["z"][model]).all()),
                "boundary_finite_difference_max": float(
                    np.max(np.abs(boundary_nearby["z"][model] - boundary["z"][model])) / 1e-6
                ),
                "phase_parameters": 1,
                "rate_grid_linearized_rank": int(np.linalg.matrix_rank(normalized)),
                "rate_grid_singular_values": singular_values.tolist(),
                "zero_signal_theta": exact_null["theta"],
                "synthetic_theta": fitted["theta"],
                "synthetic_holdout_rmse": float(np.sqrt(np.mean((prediction - signal[test_a]) ** 2))),
                "synthetic_signal_rms": float(np.sqrt(np.mean(signal[test_a] ** 2))),
                "null_fitted_theta": null["theta"],
                "null_predicted_gain_rms": float(
                    np.sqrt(
                        np.mean(
                            np.minimum(
                                delta_prediction(design["q"][test_a], z[test_a], pack["weights"], null["theta"]), 0
                            )
                            ** 2
                        )
                    )
                ),
                "quadrature_16_to_32_max_z": float(np.max(np.abs(design["z"][model][:, :20] - converged["z"][model]))),
            }
            assert row["tied_error"] < 1e-12 and row["zero_error"] < 1e-12
            assert row["boundary_finite"] and row["synthetic_holdout_rmse"] < 1e-7
            assert row["zero_signal_theta"] == 0
            checks.append(row)
    previous.write_json(output / "structural_checks.json", checks)


def fit_all(output: Path) -> None:
    """Select each candidate using independently nested component features."""
    _, panel, splits = previous.inputs(str(PREVIOUS))
    sources = {str(Path(__file__)): digest(Path(__file__)), str(output / "PROTOCOL.md"): digest(output / "PROTOCOL.md")}
    sources |= {
        str(PREVIOUS / "inputs" / name): digest(PREVIOUS / "inputs" / name)
        for name in ("panel.npz", "splits.npz", "single_phase.py")
    }
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for objective in previous.OBJECTIVES:
        response = panel[f"{objective}_aggregate"]
        aggregation = panel[f"{objective}_aggregation_weights"]
        for context in previous.CONTEXTS:
            cell = output / "cells" / objective / context
            cell.mkdir(parents=True, exist_ok=True)
            contexts = (context, *(f"{context}_inner{i}" for i in range(3)))
            for prefix in contexts:
                for index in range(len(aggregation)):
                    path = PREVIOUS / "spines" / prefix / f"{objective}_c{index}.json"
                    sources[str(path)] = digest(path)
            identity = {
                p: h
                for p, h in sources.items()
                if "/spines/" not in p or any(f"/spines/{prefix}/" in p for prefix in contexts)
            }
            cached = all((cell / f"{model}.json").exists() for model in MODELS)
            if cached:
                records = {model: json.loads((cell / f"{model}.json").read_text()) for model in MODELS}
                assert all(record["input_hashes"] == identity for record in records.values()), "Changed cached inputs"
                design = features(panel["weights"], spine_pack(context, objective))
            else:
                designs = {prefix: features(panel["weights"], spine_pack(prefix, objective)) for prefix in contexts}
                design = designs[context]
                records = {}
                for model in MODELS:
                    sweep: list[dict[str, Any]] = [{"rate": 1.0, "penalty": "null", "sse": 0.0, "n": 0}]
                    for rate in RATES:
                        for penalty in PENALTIES:
                            sweep.append({"rate": rate, "penalty": penalty, "sse": 0.0, "n": 0})
                    for inner in range(3):
                        prefix = f"{context}_inner{inner}"
                        train_a, train_t = previous.selected_pairs(panel, splits[f"{prefix}_train"])
                        test_a, test_t = previous.selected_pairs(panel, splits[f"{prefix}_test"])
                        if not len(test_a):
                            continue
                        held_target = response[test_a] - response[test_t]
                        for setting in sweep:
                            index = RATES.index(setting["rate"])
                            q, z = designs[prefix]["q"], designs[prefix]["z"][model][index]
                            fitted = (
                                {"theta": 0.0, "success": True}
                                if setting["penalty"] == "null"
                                else fit_amplitude(
                                    q[train_a],
                                    z[train_a],
                                    aggregation,
                                    response[train_a] - response[train_t],
                                    setting["penalty"],
                                )
                            )
                            assert fitted["success"]
                            prediction = delta_prediction(q[test_a], z[test_a], aggregation, fitted["theta"])
                            setting["sse"] += float(np.sum((prediction - held_target) ** 2))
                            setting["n"] += len(test_a)
                    selected = min(sweep, key=lambda setting: setting["sse"] / setting["n"])
                    train_a, train_t = previous.selected_pairs(panel, previous.training_rows(panel, splits, context))
                    index = RATES.index(selected["rate"])
                    fitted = (
                        {"theta": 0.0, "cost": 0.0, "optimality": 0.0, "success": True, "clipped": 0}
                        if selected["penalty"] == "null"
                        else fit_amplitude(
                            design["q"][train_a],
                            design["z"][model][index][train_a],
                            aggregation,
                            response[train_a] - response[train_t],
                            selected["penalty"],
                        )
                    )
                    assert fitted["success"]
                    records[model] = fitted | {
                        "objective": objective,
                        "context": context,
                        "model": model,
                        "rate": selected["rate"],
                        "penalty": selected["penalty"],
                        "inner_mse": selected["sse"] / selected["n"],
                        "sweep": sweep,
                        "input_hashes": identity,
                    }
                    previous.write_json(cell / f"{model}.json", records[model])
            scored = (
                ~panel["calibration_mask"]
                if context == "final"
                else (panel["outer_fold"] == int(context[-1])) & ~panel["calibration_mask"]
            )
            for model, record in records.items():
                index = RATES.index(record["rate"])
                prediction = design["aggregate"] + delta_prediction(
                    design["q"], design["z"][model][index], aggregation, record["theta"]
                )
                assert np.isfinite(prediction).all()
                assert (
                    np.max(np.abs(prediction[panel["physical_tied"]] - design["aggregate"][panel["physical_tied"]]))
                    < 1e-12
                )
                assert np.min(prediction - np.sum(spine_pack(context, objective)["floor"] * aggregation)) > 0
                summary.append(
                    {key: value for key, value in record.items() if key not in ("sweep", "input_hashes")}
                    | {"prediction_clips": int(np.sum(np.abs(record["theta"] * design["z"][model][index]) > LOG_LIMIT))}
                )
                for row, value in enumerate(prediction):
                    rows.append(
                        {
                            "objective": objective,
                            "model": model,
                            "context": context,
                            "row": row,
                            "run": panel["runs"][row],
                            "group": panel["groups"][row],
                            "fold": int(panel["outer_fold"][row]),
                            "tied": bool(panel["physical_tied"][row]),
                            "scored": bool(scored[row]),
                            "measured": float(response[row]),
                            "predicted": float(value),
                        }
                    )
            print(f"Completed {objective}/{context}", flush=True)
    pd.DataFrame(rows).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "fit_summary.csv", index=False)
    previous.write_json(output / "input_manifest.json", sources)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    structural_checks(args.output)
    if not args.checks_only:
        fit_all(args.output)


if __name__ == "__main__":
    main()
