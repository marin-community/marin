# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Fixed-hyperparameter BPB and pair-weighted joint WSPU refits."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fit_two_phase_creative_joint_followup_20260907 import (
    CONTEXTS,
    INPUT_OUTPUT,
    OBJECTIVES,
    FollowupPredictor,
    design,
    file_hash,
    inputs,
    write_json,
)
from fit_two_phase_creative_joint_followup_20260907 import DEFAULT_OUTPUT as BASELINE_OUTPUT
from joint_floor_solver_20260908 import floor_parameters, response_residual_jacobian
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "reference_outputs/two_phase_refinement_20260907/joint_estimators"
CONTRAST_WEIGHTS = {"CRE2-017": 1.0, "CRE2-018": 4.0}
MAX_EVALUATIONS = 300
TOLERANCE = 1e-9


@dataclass(frozen=True)
class PairChannels:
    asymmetric: np.ndarray
    tied: np.ndarray
    unpaired: np.ndarray
    count: int

    def transform(self, values: np.ndarray, contrast_weight: float) -> np.ndarray:
        """Rotate and weight data rows, preserving appended penalty rows."""
        pairs = len(self.asymmetric)
        factor = np.sqrt(self.count / (len(self.unpaired) + (1 + contrast_weight) * pairs))
        data = np.concatenate(
            [
                values[self.unpaired],
                (values[self.asymmetric] + values[self.tied]) / np.sqrt(2),
                np.sqrt(contrast_weight / 2) * (values[self.asymmetric] - values[self.tied]),
            ],
            axis=0,
        )
        return np.concatenate([factor * data, values[self.count :]], axis=0)


def pair_channels(panel: dict[str, np.ndarray], train: np.ndarray) -> PairChannels:
    """Locate complete physical pairs in the current training-row coordinates."""
    lookup = np.full(len(panel["runs"]), -1, dtype=int)
    lookup[train] = np.arange(len(train))
    asymmetric = lookup[panel["pair_asymmetric_rows"]]
    tied = lookup[panel["pair_tied_rows"]]
    assert np.array_equal(asymmetric >= 0, tied >= 0), "pair crossed the fitting split"
    asymmetric, tied = asymmetric[asymmetric >= 0], tied[tied >= 0]
    unpaired = np.setdiff1d(np.arange(len(train)), np.concatenate([asymmetric, tied]))
    assert len(unpaired) + 2 * len(asymmetric) == len(train)
    return PairChannels(asymmetric, tied, unpaired, len(train))


@dataclass(frozen=True)
class ResponseSystem:
    matrix: np.ndarray
    response: np.ndarray
    ridge: float
    anchor: float
    gap: float
    noise_sd: float
    gamma: float
    scale: float
    channels: PairChannels

    def residual_jacobian(self, parameters: np.ndarray, contrast_weight: float) -> tuple[np.ndarray, np.ndarray]:
        residual, jacobian = response_residual_jacobian(
            parameters,
            self.matrix,
            self.response,
            self.ridge,
            self.anchor,
            self.gap,
            self.noise_sd,
            self.scale,
            self.gamma,
        )
        return self.channels.transform(residual, contrast_weight), self.channels.transform(jacobian, contrast_weight)


def fit_system(system: ResponseSystem, saved_start: np.ndarray, contrast_weight: float) -> dict[str, Any]:
    """Fit the two preregistered starts, retaining every numerical endpoint."""
    width = system.matrix.shape[1]
    starts = {
        "saved_log_nnls": saved_start,
        "intercept_only": np.concatenate([[np.log(system.scale)], np.zeros(width)]),
    }
    lower = np.concatenate([[-np.inf], np.zeros(width)])
    upper = np.full(width + 1, np.inf)
    records: list[dict[str, Any]] = []
    for name, start in starts.items():
        result = least_squares(
            lambda parameters: system.residual_jacobian(parameters, contrast_weight)[0],
            start,
            jac=lambda parameters: system.residual_jacobian(parameters, contrast_weight)[1],
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            max_nfev=MAX_EVALUATIONS,
            ftol=TOLERANCE,
            xtol=TOLERANCE,
            gtol=TOLERANCE,
        )
        if not np.isfinite(result.cost):
            raise ValueError("nonfinite response-fit objective")
        records.append(
            {
                "name": name,
                "parameters": result.x.tolist(),
                "cost": float(result.cost),
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "optimality": float(result.optimality),
                "nfev": int(result.nfev),
                "njev": int(result.njev or 0),
                "active_coefficient_indices": np.flatnonzero(result.active_mask[1:] == -1).tolist(),
            }
        )
    winner = min(records, key=lambda record: record["cost"])
    return {
        "intercept": winner["parameters"][0],
        "coefficients": winner["parameters"][1:],
        "selected_start": winner["name"],
        "cost": winner["cost"],
        "success": winner["success"],
        "optimality": winner["optimality"],
        "starts": records,
    }


def inherited_record(context: str, objective: str, component: int) -> tuple[Path, dict[str, Any]]:
    path = BASELINE_OUTPUT / "fits" / context / f"{objective}_c{component}.json"
    return path, json.loads(path.read_text())


def fit_component(argument: tuple[str, str, int, str]) -> dict[str, Any]:
    context, objective, component, output_string = argument
    output = Path(output_string)
    baseline_path, baseline = inherited_record(context, objective, component)
    for path, expected in baseline["input_hashes"].items():
        assert file_hash(Path(path)) == expected, f"inherited fit source changed: {path}"
    paths = [
        Path(__file__),
        HERE / "joint_floor_solver_20260908.py",
        HERE / "fit_two_phase_creative_joint_followup_20260907.py",
        output / "PROTOCOL.md",
        output.parent / "JOINT_ESTIMATOR_PROPOSAL.md",
        baseline_path,
        INPUT_OUTPUT / "inputs/panel.npz",
        INPUT_OUTPUT / "inputs/splits.npz",
    ]
    hashes = {str(path): file_hash(path) for path in paths}
    destination = output / "fits" / context / f"{objective}_c{component}.json"
    if destination.exists():
        if json.loads(destination.read_text())["input_hashes"] != hashes:
            raise ValueError(f"cached estimator fit changed: {destination}")
        return {"context": context, "objective": objective, "component": component, "cached": True}
    module, panel, _ = inputs(str(INPUT_OUTPUT))
    inherited = baseline["fits"]["CRE2-013"]
    train = np.asarray(baseline["train_rows"], dtype=int)
    assert set(np.flatnonzero(panel["calibration_mask"])) <= set(train)
    response = panel[f"{objective}_outcomes"][:, component]
    matrix = design(
        module, panel["weights"], panel["c0"], panel["c1"], inherited["shape"], "CRE2-013", inherited["state"]
    )
    anchor_index = component if objective == "uncheatable" else component + 7
    noise_sd = float(panel["anchor_repeat_sd"][anchor_index])
    anchor, gap = floor_parameters(response[train], float(panel["anchor_proportional_bpb"][anchor_index]))
    floor = anchor - max(inherited["kappa"] * gap, 3 * noise_sd)
    assert abs(floor - inherited["floor"]) < 1e-12, "frozen floor was not reproduced"
    scale = max(float(np.mean(response[train] - floor)), 1e-9)
    system = ResponseSystem(
        matrix[train],
        response[train],
        inherited["ridge"],
        anchor,
        gap,
        noise_sd,
        inherited["kappa"],
        scale,
        pair_channels(panel, train),
    )
    start = np.concatenate([[inherited["intercept"]], inherited["coefficients"]])
    fits = {}
    for model, contrast_weight in CONTRAST_WEIGHTS.items():
        fitted = fit_system(system, start, contrast_weight)
        eta = fitted["intercept"] + matrix @ np.asarray(fitted["coefficients"])
        fits[model] = (
            {key: inherited[key] for key in ("shape", "state", "ridge", "kappa", "floor")}
            | fitted
            | {
                "contrast_weight": contrast_weight,
                "deficit_scale": scale,
                "training_below_floor_count": int(np.sum(response[train] <= floor)),
                "training_deficit_clamped_count": int(np.sum(response[train] - floor < 1e-9)),
                "prediction_clipped_count": int(np.sum(np.abs(eta) >= 30)),
                "train_pairs": len(system.channels.asymmetric),
                "train_unpaired": len(system.channels.unpaired),
            }
        )
    write_json(
        destination,
        {
            "context": context,
            "objective": objective,
            "component": component,
            "component_name": baseline["component_name"],
            "train_rows": train.tolist(),
            "input_hashes": hashes,
            "fits": fits,
        },
    )
    return {"context": context, "objective": objective, "component": component, "cached": False}


def load_predictor(model: str, objective: str, context: str, output: Path = DEFAULT_OUTPUT) -> FollowupPredictor:
    """Load an exact inherited-geometry component ensemble without refitting."""
    module, panel, _ = inputs(str(INPUT_OUTPUT))
    records = tuple(
        json.loads((output / "fits" / context / f"{objective}_c{component}.json").read_text())["fits"][model]
        for component in range(panel[f"{objective}_outcomes"].shape[1])
    )
    return FollowupPredictor(
        module, panel["c0"], panel["c1"], "CRE2-013", records, panel[f"{objective}_aggregation_weights"]
    )


def export_predictions(output: Path) -> None:
    _, panel, _ = inputs(str(INPUT_OUTPUT))
    frames, selections = [], []
    for context in CONTEXTS:
        scored = ~panel["calibration_mask"]
        if context != "final":
            scored = scored & (panel["outer_fold"] == int(context[-1]))
        for objective in OBJECTIVES:
            for model in CONTRAST_WEIGHTS:
                predictor = load_predictor(model, objective, context, output)
                frames.append(
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
                    selections.append(
                        {"model": model, "objective": objective, "context": context, "component": component}
                        | {
                            key: record[key]
                            for key in ("state", "ridge", "floor", "contrast_weight", "cost", "success", "optimality")
                        }
                    )
    pd.concat(frames, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(output / "fit_summary.csv", index=False)


def structural_checks(output: Path) -> None:
    module, panel, splits = inputs(str(INPUT_OUTPUT))
    train, test = splits["outer0_train"], splits["outer0_test"]
    channels = pair_channels(panel, train)
    random = np.random.default_rng(2026090717)
    checks = []
    noise_checks = []
    for objective in OBJECTIVES:
        _, baseline = inherited_record("final", objective, 0)
        shape = baseline["fits"]["CRE2-013"]["shape"]
        for state in (1.0, 0.03):
            matrix = design(module, panel["weights"], panel["c0"], panel["c1"], shape, "CRE2-013", state)
            tied_weights = np.stack([panel["aggregate"], panel["aggregate"]], axis=1)
            tied_matrix = design(module, tied_weights, panel["c0"], panel["c1"], shape, "CRE2-013", state)
            tied_reference = design(module, tied_weights, panel["c0"], panel["c1"], shape, "CRE2-013", 1.0)
            assert np.max(np.abs(tied_matrix - tied_reference)) < 1e-12
            assert np.isfinite(matrix).all()
            coefficients = random.uniform(0.5, 1.5, 78) * 0.002 / np.maximum(np.sqrt(np.mean(matrix**2, axis=0)), 1e-9)
            truth = np.concatenate([[-2.0], coefficients])
            response = np.exp(truth[0] + matrix @ truth[1:])
            assert np.isfinite(response).all()
            a, t = panel["pair_asymmetric_rows"], panel["pair_tied_rows"]
            assert np.max(np.abs(((response + 0.4)[a] - (response + 0.4)[t]) - (response[a] - response[t]))) < 1e-12
            system = ResponseSystem(
                matrix[train], response[train], 0, 1.0, 0.4, 0, 2.5, response[train].mean(), channels
            )
            start = np.concatenate(
                [
                    [-1.95],
                    coefficients * 0.9,
                ]
            )
            raw, jacobian = response_residual_jacobian(
                start, matrix[train], response[train], 0, 1, 0.4, 0, system.scale, 2.5
            )
            rotated, rotated_jacobian = system.residual_jacobian(start, 1.0)
            cost_error = float(abs(raw @ raw - rotated @ rotated))
            gradient_error = float(np.max(np.abs(jacobian.T @ raw - rotated_jacobian.T @ rotated)))
            hessian_error = float(np.max(np.abs(jacobian.T @ jacobian - rotated_jacobian.T @ rotated_jacobian)))
            assert max(cost_error, gradient_error, hessian_error) < 1e-8
            for contrast_weight in CONTRAST_WEIGHTS.values():
                transformation = channels.transform(np.eye(len(train)), contrast_weight)
                assert abs(float(np.sum(transformation**2)) - len(train)) < 1e-10
                _, analytic = system.residual_jacobian(start, contrast_weight)
                direction = random.normal(size=len(start))
                direction /= np.linalg.norm(direction)
                step = 1e-6
                plus = system.residual_jacobian(start + step * direction, contrast_weight)[0]
                minus = system.residual_jacobian(start - step * direction, contrast_weight)[0]
                finite_error = float(np.max(np.abs((plus - minus) / (2 * step) - analytic @ direction)))
                assert finite_error < 1e-7
                fitted = fit_system(system, start, contrast_weight)
                predicted = np.exp(fitted["intercept"] + matrix[test] @ np.asarray(fitted["coefficients"]))
                recovery_error = float(np.max(np.abs(predicted - response[test])))
                assert recovery_error < 1e-6, (objective, state, contrast_weight, recovery_error)
                checks.append(
                    {
                        "objective": objective,
                        "state": state,
                        "contrast_weight": contrast_weight,
                        "cost_identity_error": cost_error,
                        "gradient_identity_error": gradient_error,
                        "gauss_newton_identity_error": hessian_error,
                        "finite_difference_error": finite_error,
                        "heldout_recovery_error": recovery_error,
                        "selected_success": fitted["success"],
                    }
                )
            if state != 0.03:
                continue
            null_coefficients = coefficients.copy()
            null_coefficients[:39] = 0
            null_response = np.exp(-2 + matrix @ null_coefficients)
            anchor_index = 0 if objective == "uncheatable" else 7
            noise_sd = float(panel["anchor_repeat_sd"][anchor_index])
            for draw in range(4):
                noisy = null_response + random.normal(0, noise_sd, len(matrix))
                assert noisy.min() > 0
                intercept, fitted_coefficients = module.nonnegative_solve(matrix[train], np.log(noisy[train]), 0.01)
                initial = np.concatenate([[intercept], fitted_coefficients])
                noise_system = ResponseSystem(
                    matrix[train], noisy[train], 0.01, 1, 0.4, 0, 2.5, noisy[train].mean(), channels
                )
                a, t = panel["pair_asymmetric_rows"], panel["pair_tied_rows"]
                selected = np.isin(a, test)
                for contrast_weight in CONTRAST_WEIGHTS.values():
                    fitted = fit_system(noise_system, initial, contrast_weight)
                    prediction = np.exp(fitted["intercept"] + matrix @ np.asarray(fitted["coefficients"]))
                    gain = prediction[t[selected]] - prediction[a[selected]]
                    noise_checks.append(
                        {
                            "objective": objective,
                            "draw": draw,
                            "contrast_weight": contrast_weight,
                            "noise_sd": noise_sd,
                            "heldout_false_gain_rms": float(np.sqrt(np.mean(np.maximum(gain, 0) ** 2))),
                            "heldout_false_gain_max": float(np.max(gain)),
                            "selected_success": fitted["success"],
                        }
                    )
    write_json(
        output / "structural_checks.json",
        {
            "source_hash": file_hash(Path(__file__)),
            "checks": checks,
            "null_noise_checks": noise_checks,
            "note": "Null noise is a diagnostic at fixed r=.03, not a full-pipeline state-selection test.",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("check", "fit", "export", "all"), default="all")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        raise ValueError("use one or two local workers")
    if args.stage in ("check", "all"):
        structural_checks(args.output)
    if args.stage in ("fit", "all"):
        checks = json.loads((args.output / "structural_checks.json").read_text())
        if checks["source_hash"] != file_hash(Path(__file__)):
            raise ValueError("structural checks do not match this source")
        _, panel, _ = inputs(str(INPUT_OUTPUT))
        jobs = [
            (context, objective, component, str(args.output))
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
