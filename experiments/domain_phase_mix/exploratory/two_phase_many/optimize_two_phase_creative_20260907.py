# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Uncapped raw-policy diagnostics for three frozen creative-screen models."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import creative_joint_gradient_20260907 as joint_gradient
import fit_two_phase_creative_geometry_20260907 as geometry
import fit_two_phase_creative_joint_20260907 as joint
import fit_two_phase_creative_joint_followup_20260907 as joint_followup
import fit_two_phase_creative_paths_20260907 as paths
import fit_two_phase_creative_semantic_20260907 as semantic
import fit_two_phase_link_transfer_20260907 as previous
import numpy as np
import optimize_two_phase_link_transfer_20260907 as prior_raw
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import expit, softmax

HERE = Path(__file__).resolve().parent
ROOT = HERE / "reference_outputs/two_phase_creative_sweep_20260907"
OUTPUT = ROOT / "optima"
MODELS = ("CRE2-002", "CRE2-007", "CRE2-011", "CRE2-016", "CRE2-013")
STEP = 1e-4


@dataclass(frozen=True)
class Surface:
    predict: Callable[[np.ndarray], np.ndarray]
    diagnostics: Callable[[np.ndarray], dict[str, Any]]
    floor: float
    files: tuple[Path, ...]
    parity: float
    weight_gradient: Callable[[np.ndarray], np.ndarray] | None


def load_surface(model: str, objective: str, context: str) -> Surface:
    """Vectorize frozen equations and verify their public predictors before use."""
    _, panel, _ = previous.inputs(str(paths.PREVIOUS))
    weight_gradient: Callable[[np.ndarray], np.ndarray] | None = None
    files = [
        Path(__file__),
        Path(paths.__file__),
        Path(previous.__file__),
        Path(prior_raw.__file__),
        OUTPUT / "PROTOCOL.md",
    ]
    files += [paths.PREVIOUS / "inputs" / name for name in ("panel.npz", "splits.npz", "single_phase.py")]
    if model in ("CRE2-002", "CRE2-013"):
        folder = "joint_wspu" if model == "CRE2-002" else "joint_followup"
        original = (
            joint.load_predictor(model, objective, context, ROOT / folder)
            if model == "CRE2-002"
            else joint_followup.load_predictor(model, objective, context, ROOT / folder)
        )
        records = original.records
        module = original.module
        width = len(panel["buckets"])
        coeff = np.asarray([r["coefficients"] for r in records])[None]
        rates = np.asarray([r["shape"]["rate"] for r in records])[None, :, None]
        powers = np.asarray([r["shape"]["power"] for r in records])[None, :, None]
        thresholds = np.asarray([r["shape"]["threshold"] for r in records])[None, :, None]
        states = np.asarray([r["state"] for r in records])[None, :, None]
        intercepts = np.asarray([r["intercept"] for r in records])
        floors = np.asarray([r["floor"] for r in records])
        aggregation = original.aggregation_weights
        c0, c1 = original.c0, original.c1

        def joint_components(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            early, late = c0 * weights[:, 0, None, :], c1 * weights[:, 1, None, :]
            tied = weights[:, 0, None, :] == weights[:, 1, None, :]
            total = np.where(tied, (c0 + c1) * weights[:, 0, None, :], early + late)
            fraction = c0 / (c0 + c1)
            effective = np.where(tied, total, (states * early + late) / (states * fraction + 1 - fraction))
            b = module.benefit(effective, rates, powers)
            h = module.harm(total, thresholds) - module.harm(np.zeros_like(total), thresholds)
            eta = intercepts + np.sum(-coeff[:, :, :width] * b + coeff[:, :, width:] * h, axis=-1)
            return floors + np.exp(np.clip(eta, -30, 30)), eta

        def predict(weights: np.ndarray) -> np.ndarray:
            return joint_components(weights)[0] @ aggregation

        def diagnostics(weights: np.ndarray) -> dict[str, Any]:
            values, eta = joint_components(weights)
            return {
                "aggregate_exponent_clips": int(np.sum(np.abs(eta) > 30)),
                "correction_exponent_clips": 0,
                "component_floor_crossings": int(np.sum(values < floors)),
                "minimum_component_margin": float(np.min(values - floors)),
            }

        files.append(Path(joint.__file__))
        if model == "CRE2-013":
            files.append(Path(joint_followup.__file__))
        files += [ROOT / folder / "fits" / context / f"{objective}_c{i}.json" for i in range(len(records))]
        original_values = original(panel["weights"])
        analytic = joint_gradient.load_gradient(model, objective, context)

        def joint_weight_gradient(weights: np.ndarray) -> np.ndarray:
            return analytic.value_and_gradient(weights)[1]

        weight_gradient = joint_weight_gradient
        files.append(Path(joint_gradient.__file__))
    else:
        pack = paths.spine_pack(context, objective)
        aggregation, floors = pack["weights"], pack["floor"]

        def base(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            total = pack["c0"] * weights[:, 0] + pack["c1"] * weights[:, 1]
            eta = pack["intercept"] + np.sum(
                -pack["alpha"] * paths.benefit(total[:, None], pack) + pack["beta"] * paths.harm(total[:, None], pack),
                axis=-1,
            )
            return np.exp(np.clip(eta, -30, 30)), eta, total / (pack["c0"] + pack["c1"])

        def eta_gradient(single: np.ndarray) -> np.ndarray:
            exposure = single[:, None] * (pack["c0"] + pack["c1"])
            powered = (pack["rate"] * exposure) ** pack["power"]
            db = pack["power"] * powered * np.exp(-powered) / exposure
            argument = np.log1p(exposure) - pack["threshold"]
            dh = 2 * np.logaddexp(0, argument) * expit(argument) / (1 + exposure)
            return (-pack["alpha"] * db + pack["beta"] * dh) * (pack["c0"] + pack["c1"])

        if model == MODELS[1]:
            folder = "path_states"
            fit_path = ROOT / folder / "cells" / objective / context / f"{model}.json"
            record = json.loads(fit_path.read_text())
            rate, theta = float(record["rate"]), float(record["theta"])

            def path_statistic(weights: np.ndarray) -> np.ndarray:
                w0, w1 = weights[:, 0, None], weights[:, 1, None]
                e0, e1 = w0 * pack["c0"], w1 * pack["c1"]
                b0, b1, bt = paths.benefit(e0, pack), paths.benefit(e1, pack), paths.benefit(e0 + e1, pack)
                competition = np.sum(w1 * b1, axis=-1, keepdims=True) - w1 * b1
                retained = b0 / (1 + rate * competition / (1 + e1)) + bt - b0
                return -np.sum(pack["alpha"] * retained, axis=-1)

            def path_components(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                q, eta, aggregate = base(weights)
                tied = np.repeat(aggregate[:, None], 2, axis=1)
                correction = theta * (path_statistic(weights) - path_statistic(tied))
                return floors + q * np.exp(np.clip(correction, -30, 30)), eta, correction

            def predict(weights: np.ndarray) -> np.ndarray:
                return path_components(weights)[0] @ aggregation

            def diagnostics(weights: np.ndarray) -> dict[str, Any]:
                values, eta, correction = path_components(weights)
                return {
                    "aggregate_exponent_clips": int(np.sum(np.abs(eta) > 30)),
                    "correction_exponent_clips": int(np.sum(np.abs(correction) > 30)),
                    "component_floor_crossings": int(np.sum(values < floors)),
                    "minimum_component_margin": float(np.min(values - floors)),
                }

            original_values = paths.predict_weights(panel["weights"], objective, context, model, ROOT / folder)
        else:
            folder = "phase_geometry" if model == "CRE2-011" else "semantic_followup"
            original_geometry = (
                geometry.load_predictor(ROOT / folder, objective, context, model)
                if model == "CRE2-011"
                else semantic.load_predictor(ROOT / folder, objective, context, model)
            )
            record = original_geometry.record
            fit_path = ROOT / folder / "fits" / model / objective / context / "fit.json"
            scale, coefficients = np.asarray(record["scale"]), np.asarray(record["coefficients"])

            def geometry_components(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                q, eta, _ = base(weights)
                if record["config"]["zero"]:
                    return floors + q, eta
                if model == "CRE2-016":
                    early = np.repeat(weights[:, :1], 2, axis=1)
                    late = np.repeat(weights[:, 1:], 2, axis=1)
                    delta = base(late)[1] - base(early)[1]
                    correction = (np.hstack([delta, delta**2]) / scale) @ coefficients
                    return floors + q + correction, eta
                early = np.repeat(weights[:, :1], 2, axis=1)
                late = np.repeat(weights[:, 1:], 2, axis=1)
                delta = base(late)[1] - base(early)[1]
                matrix, tied = np.hstack([eta, delta]), np.hstack([eta, np.zeros_like(delta)])
                correction = (
                    geometry.difference_kernel(
                        matrix / scale,
                        tied / scale,
                        np.asarray(record["centers"]),
                        np.asarray(record["tied_centers"]),
                        record["config"]["bandwidth"],
                    )
                    @ coefficients
                )
                return floors + q + correction, eta

            def predict(weights: np.ndarray) -> np.ndarray:
                return geometry_components(weights)[0] @ aggregation

            def diagnostics(weights: np.ndarray) -> dict[str, Any]:
                values, eta = geometry_components(weights)
                return {
                    "aggregate_exponent_clips": int(np.sum(np.abs(eta) > 30)),
                    "correction_exponent_clips": 0,
                    "component_floor_crossings": int(np.sum(values < floors)),
                    "minimum_component_margin": float(np.min(values - floors)),
                }

            original_values = original_geometry.predict(panel["weights"])
            files.append(Path(geometry.__file__))
            if model == "CRE2-016":
                files.append(Path(semantic.__file__))

            def semantic_gradient(weights: np.ndarray) -> np.ndarray:
                q, eta, aggregate = base(weights)
                ja = eta_gradient(aggregate)
                a = float(record["alpha"])
                coefficient_a = q * aggregation * (np.abs(eta) < 30)
                coefficient_delta = np.zeros_like(coefficient_a)
                if not record["config"]["zero"]:
                    early = np.repeat(weights[:, :1], 2, axis=1)
                    late = np.repeat(weights[:, 1:], 2, axis=1)
                    delta = base(late)[1] - base(early)[1]
                    linear = coefficients @ aggregation
                    width = delta.shape[1]
                    if model == "CRE2-016":
                        coefficient_delta = linear[:width] / scale[:width] + 2 * delta * linear[width:] / scale[width:]
                    else:
                        matrix = np.hstack([eta, delta]) / scale
                        tied = np.hstack([eta, np.zeros_like(delta)]) / scale
                        centers, tied_centers = np.asarray(record["centers"]), np.asarray(record["tied_centers"])
                        denominator = 2 * matrix.shape[1] * record["config"]["bandwidth"] ** 2

                        def kernel_derivative(points: np.ndarray, locations: np.ndarray) -> np.ndarray:
                            kernel = np.exp(-cdist(points, locations, metric="sqeuclidean") / denominator) * linear
                            return (-2 / denominator) * (points * kernel.sum(axis=1, keepdims=True) - kernel @ locations)

                        dm = kernel_derivative(matrix, centers) - kernel_derivative(matrix, tied_centers)
                        dt = -kernel_derivative(tied, centers) + kernel_derivative(tied, tied_centers)
                        coefficient_a += (dm[:, :width] + dt[:, :width]) / scale[:width]
                        coefficient_delta = dm[:, width:] / scale[width:]
                aggregate_gradient = np.sum(coefficient_a[:, :, None] * ja, axis=1)
                j0, j1 = eta_gradient(weights[:, 0]), eta_gradient(weights[:, 1])
                return np.stack(
                    [
                        a * aggregate_gradient - np.sum(coefficient_delta[:, :, None] * j0, axis=1),
                        (1 - a) * aggregate_gradient + np.sum(coefficient_delta[:, :, None] * j1, axis=1),
                    ],
                    axis=1,
                )

            weight_gradient = semantic_gradient
        files.append(fit_path)
        files += [paths.PREVIOUS / "spines" / context / f"{objective}_c{i}.json" for i in range(len(aggregation))]
    saved = pd.read_csv(ROOT / folder / "predictions.csv")
    saved = saved[(saved.objective == objective) & (saved.context == context) & (saved.model == model)].sort_values(
        "row"
    )
    values = predict(panel["weights"])
    parity = max(
        float(np.max(np.abs(values - original_values))), float(np.max(np.abs(values - saved.predicted.to_numpy())))
    )
    assert parity < 1e-10, (model, objective, context, parity)
    return Surface(predict, diagnostics, float(floors @ aggregation), tuple(files), parity, weight_gradient)


def policies(logits: np.ndarray, tied: bool) -> np.ndarray:
    shape = (-1, 1 if tied else 2, 39)
    values = softmax(np.asarray(logits).reshape(shape), axis=-1)
    return np.repeat(values, 2, axis=1) if tied else values


def value_gradient(surface: Surface, logits: np.ndarray, tied: bool) -> tuple[float, np.ndarray]:
    if surface.weight_gradient is not None:
        weights = policies(logits, tied)
        gradient = surface.weight_gradient(weights)[0]
        probabilities = weights[0]
        if tied:
            gradient, probabilities = gradient.sum(axis=0, keepdims=True), probabilities[:1]
        logit_gradient = probabilities * (gradient - np.sum(probabilities * gradient, axis=1, keepdims=True))
        return float(surface.predict(weights)[0]), logit_gradient.ravel()
    batch = np.repeat(logits[None], len(logits) + 1, axis=0)
    batch[1:] += STEP * np.eye(len(logits))
    values = surface.predict(policies(batch, tied))
    return float(values[0]), (values[1:] - values[0]) / STEP


def solve_start(surface: Surface, start: np.ndarray, tied: bool, name: str) -> dict[str, Any]:
    initialized = 0.999 * start + 0.001 / 39
    logits = np.log(initialized[:1] if tied else initialized).ravel()
    initial_value = float(surface.predict(policies(logits, tied))[0])
    begin = time.monotonic()
    result = minimize(
        lambda x: value_gradient(surface, x, tied),
        logits,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 600, "ftol": 1e-11, "gtol": 1e-7, "maxls": 30},
    )
    endpoint = policies(result.x, tied)[0]
    endpoint_value = float(surface.predict(endpoint[None])[0])
    keep_endpoint = np.isfinite(endpoint_value) and endpoint_value <= initial_value
    retained = endpoint if keep_endpoint else policies(logits, tied)[0]
    return {
        "name": name,
        "original_weights": start.tolist(),
        "initialized_weights": policies(logits, tied)[0].tolist(),
        "initial_bpb": initial_value,
        "endpoint_bpb": endpoint_value,
        "retained_bpb": min(initial_value, endpoint_value),
        "weights": retained.tolist(),
        "success": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "max_logit_gradient": float(np.max(np.abs(result.jac))),
        "elapsed_seconds": time.monotonic() - begin,
        "retained_initial": not keep_endpoint,
        "diagnostics": surface.diagnostics(retained[None]),
    }


def run_cell(model: str, objective: str, context: str, tied: bool, tied_result: dict[str, Any] | None) -> dict[str, Any]:
    _, panel, splits = previous.inputs(str(paths.PREVIOUS))
    kind = "tied" if tied else "two_phase"
    destination = OUTPUT / "cells" / model / objective / context / kind
    destination.mkdir(parents=True, exist_ok=True)
    surface = load_surface(model, objective, context)
    aggregate_path = paths.PREVIOUS / "optima" / f"{context}_{objective}_aggregate.json"
    source_paths = (*surface.files, aggregate_path)
    identity = {str(path): paths.digest(path) for path in source_paths}
    if (destination / "complete.json").exists():
        cached = json.loads((destination / "complete.json").read_text())
        assert cached["input_hashes"] == identity, "Raw audit inputs changed"
        return cached
    train = previous.training_rows(panel, splits, context)
    alpha = float(panel["alpha"])
    observed = panel["weights"][train]
    truth = panel[f"{objective}_aggregate"]
    best_train = int(train[np.argmin(truth[train])])
    proportional = panel["weights"][np.flatnonzero(panel["calibration_mask"])[0]]
    saved_aggregate = np.asarray(json.loads(aggregate_path.read_text())["weights"])
    rng = np.random.default_rng(20260907)
    starts = [
        ("proportional", proportional),
        ("saved_aggregate", saved_aggregate),
        ("best_training", panel["weights"][best_train]),
        *[(f"dirichlet_{i}", rng.dirichlet(np.ones(39), size=2)) for i in range(2)],
    ]
    if tied:
        starts = [(name, np.repeat((alpha * w[0] + (1 - alpha) * w[1])[None], 2, axis=0)) for name, w in starts]
    elif tied_result is not None:
        starts.append(("own_tied_minimum", np.asarray(tied_result["weights"])))
    logits = np.log(np.full(39 if tied else 78, 1 / 39))
    _, gradient = value_gradient(surface, logits, tied)
    direction = rng.normal(size=len(logits))
    direction /= np.linalg.norm(direction)
    symmetric = float(
        (
            surface.predict(policies(logits + STEP * direction, tied))[0]
            - surface.predict(policies(logits - STEP * direction, tied))[0]
        )
        / (2 * STEP)
    )
    derivative_error = abs(symmetric - float(gradient @ direction))
    assert derivative_error < 3e-6, (model, objective, derivative_error)
    results = []
    for name, start in starts:
        path = destination / f"{name}.json"
        if path.exists():
            result = json.loads(path.read_text())
            assert result["input_hashes"] == identity
        else:
            result = solve_start(surface, start, tied, name) | {"input_hashes": identity}
            previous.write_json(path, result)
        results.append(result)
        print(
            f"{model}/{objective}/{context}/{kind}/{name}: {result['retained_bpb']:.8f} success={result['success']}",
            flush=True,
        )
    best = min(results, key=lambda result: result["retained_bpb"])
    policy = np.asarray(best["weights"])
    spread = [
        prior_raw.policy_tv(np.asarray(a["weights"]), np.asarray(b["weights"]), alpha)
        for i, a in enumerate(results)
        for b in results[i + 1 :]
    ]
    epochs = panel["c0"] * policy[0] + panel["c1"] * policy[1]
    record = {
        "model": model,
        "objective": objective,
        "context": context,
        "kind": kind,
        "input_hashes": identity,
        "prediction_parity": surface.parity,
        "directional_derivative_error": derivative_error,
        "gradient_method": "analytic" if surface.weight_gradient is not None else "batched_forward_logits",
        "predicted_bpb": best["retained_bpb"],
        "weighted_floor": surface.floor,
        "weighted_floor_margin": best["retained_bpb"] - surface.floor,
        "predicted_gain_over_own_tied": (
            0.0 if tied_result is None else tied_result["predicted_bpb"] - best["retained_bpb"]
        ),
        "selected_start": best["name"],
        "selected_solver_success": best["success"],
        "successful_starts": sum(r["success"] for r in results),
        "n_starts": len(results),
        "start_bpb_spread": float(np.ptp([r["retained_bpb"] for r in results])),
        "start_policy_tv_max": max(spread, default=0.0),
        "weights": policy.tolist(),
        "epochs": epochs.tolist(),
        "max_epochs": float(epochs.max()),
        "max_phase_weight": float(policy.max()),
        "phase_tv": float(np.abs(policy[0] - policy[1]).sum() / 2),
        "training_support": prior_raw.support_audit(policy, observed, alpha),
        "full_support": prior_raw.support_audit(policy, panel["weights"], alpha),
        "diagnostics": best["diagnostics"],
        "starts": results,
    }
    previous.write_json(destination / "complete.json", record)
    return record


def summarize() -> None:
    records = [json.loads(p.read_text()) for p in sorted((OUTPUT / "cells").glob("*/*/*/*/complete.json"))]
    rows = []
    for record in records:
        row = {
            k: v
            for k, v in record.items()
            if k
            not in ("input_hashes", "weights", "epochs", "starts", "training_support", "full_support", "diagnostics")
        }
        row |= {f"training_{k}": v for k, v in record["training_support"].items()}
        row |= record["diagnostics"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT / "summary.csv", index=False)
    _, panel, _ = previous.inputs(str(paths.PREVIOUS))
    stability = []
    for record in records:
        if record["context"] == "final":
            continue
        final = next(
            (
                r
                for r in records
                if r["model"] == record["model"]
                and r["objective"] == record["objective"]
                and r["kind"] == record["kind"]
                and r["context"] == "final"
            ),
            None,
        )
        if final is not None:
            stability.append(
                {
                    "model": record["model"],
                    "objective": record["objective"],
                    "kind": record["kind"],
                    "context": record["context"],
                    "final_policy_tv": prior_raw.policy_tv(
                        np.asarray(final["weights"]), np.asarray(record["weights"]), float(panel["alpha"])
                    ),
                }
            )
    pd.DataFrame(stability).to_csv(OUTPUT / "context_stability.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["CRE2-013", "CRE2-011", "CRE2-016", "CRE2-007"])
    parser.add_argument("--contexts", nargs="+", choices=previous.CONTEXTS, default=["final"])
    parser.add_argument("--objectives", nargs="+", choices=previous.OBJECTIVES, default=list(previous.OBJECTIVES))
    args = parser.parse_args()
    for context in args.contexts:
        for objective in args.objectives:
            for model in args.models:
                tied = run_cell(model, objective, context, True, None)
                run_cell(model, objective, context, False, tied)
                summarize()


if __name__ == "__main__":
    main()
