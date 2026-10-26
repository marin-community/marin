# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Preflight and fixed-floor/shape oracle for the registered rate-harm reopening.

This module does not implement the prospective full J2 combined estimator.
It never reads actual outcome arrays. Synthetic labels and oracle fits require
an execution receipt naming the final registration and frozen source identity.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_two_phase_20260921 as benchmark
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_joint_estimation_20261014 as joint
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_mechanism_models_20261014 as historical
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_tangent_models_20261015 as odds

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reference_outputs/tpp40_rate_reopening_20261015"
DATA = ROOT / "reference_outputs/tpp40_temporal_round_20261014/data"
ROUND3 = ROOT / "reference_outputs/tpp40_joint_estimation_round3_20261014"
ROUND5 = ROOT / "reference_outputs/tpp40_tangent_round5_20261015"
TRUTH_SOURCE = ROUND3 / "contexts/aggregate_null_noise0_seed20261015/truth.json"
FAMILIES = ("aggregate_mariner", "bounded_benefit_odds", "rate_damage", "bounded_benefit_rate")
RATE_GRID = historical.RATE_GRID
mariner = joint.mariner


def parameter_grid(family: str) -> tuple[dict[str, float], ...]:
    """Return the existing strength/rate grids, with their exact null first."""
    if family not in FAMILIES:
        raise ValueError(f"Unsupported family: {family}")
    kappas = odds.STRENGTHS if family in ("bounded_benefit_odds", "bounded_benefit_rate") else (0.0,)
    nus = RATE_GRID if family in ("rate_damage", "bounded_benefit_rate") else (0.0,)
    return tuple({"kappa": kappa, "nu": nu} for kappa, nu in itertools.product(kappas, nus))


def design(
    early: np.ndarray, late: np.ndarray, alpha: float, shape: dict[str, float], parameters: dict[str, float]
) -> np.ndarray:
    """Compose frozen stable odds benefit with the original native rate-harm law."""
    matrix, _ = odds.design_and_offset(early, late, alpha, shape, "bounded_benefit_odds", parameters)
    if parameters["nu"]:
        harm_matrix, _ = historical.design_and_offset(early, late, alpha, shape, "rate_damage", parameters)
        width = early.shape[1]
        matrix[:, width:] = harm_matrix[:, width:]
    return matrix


def prediction(
    early: np.ndarray,
    late: np.ndarray,
    alpha: float,
    shape: dict[str, float],
    parameters: dict[str, float],
    head: mariner.Head,
) -> np.ndarray:
    return head.predict(design(early, late, alpha, shape, parameters))


def oracle_fit(
    early: np.ndarray,
    late: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
    query_early: np.ndarray,
    query_late: np.ndarray,
    alpha: float,
    shape: dict[str, float],
    floor: float,
) -> dict[str, Any]:
    """Select kinetics by inner BPB CV at known shape/floor and zero ridge.

    Every candidate refits native nonnegative amplitudes in each inner split.
    Saved full heads and query predictions expose functionally different
    candidates with indistinguishable training scores. No query labels enter.
    """
    profile = []
    with threadpool_limits(limits=1):
        for index, parameters in enumerate(parameter_grid("bounded_benefit_rate")):
            matrix = design(early, late, alpha, shape, parameters)
            inner = joint.candidate_oof(matrix, np.zeros(len(early)), observed, labels, 0.0, None, known_floor=floor)
            head = joint.fit_fixed_head(matrix, np.zeros(len(early)), observed, 0.0, None, floor)
            profile.append(
                {
                    "parameter_index": index,
                    "parameters": parameters,
                    "inner_rmse": float(np.sqrt(np.mean((inner - observed) ** 2))),
                    "inner_predictions": inner.tolist(),
                    "query_predictions": prediction(query_early, query_late, alpha, shape, parameters, head).tolist(),
                    "head": {
                        "intercept": head.intercept,
                        "coefficients": head.coefficients.tolist(),
                        "floor": head.floor,
                    },
                }
            )
    selected = min(profile, key=lambda item: (item["inner_rmse"], item["parameter_index"]))
    return {"selected_parameter_index": selected["parameter_index"], "profile": profile}


def floor_report(observed: np.ndarray, floor: float) -> dict[str, Any]:
    """Describe whether the fixed synthetic floor is in this training slice's domain."""
    median, minimum = float(np.median(observed)), float(np.min(observed))
    gap = median - minimum
    gamma = (median - floor) / gap if gap > mariner.DEFICIT_FLOOR else None
    return {
        "median": median,
        "minimum": minimum,
        "true_floor": floor,
        "gamma_at_true_floor": gamma,
        "floor_domain_at_gamma_1_6": [median - 6 * gap, minimum],
        "feasible": gamma is not None and 1 - 1e-10 <= gamma <= 6 + 1e-10,
    }


def local_sensitivity(
    early: np.ndarray,
    late: np.ndarray,
    alpha: float,
    shape: dict[str, float],
    parameters: dict[str, float],
    head: mariner.Head,
    step: float,
    svd_relative_tolerance: float,
) -> dict[str, Any]:
    """Residualize kinetic BPB derivatives against a local static nuisance span.

    The span contains amplitude/intercept, absolute-floor and shared-shape
    derivatives. Unconstrained amplitude directions are conservative even at
    nonnegative coefficient boundaries. This is not a global identification test.
    """
    base_matrix = design(early, late, alpha, shape, parameters)
    base = head.predict(base_matrix)
    deficit = base - head.floor
    nuisance = [np.ones(len(base)), deficit, *(deficit[:, None] * base_matrix).T]
    for key in ("rate", "power", "threshold"):
        delta = step * max(1.0, abs(shape[key]))
        left, right = dict(shape), dict(shape)
        left[key] -= delta
        right[key] += delta
        nuisance.append(
            (
                prediction(early, late, alpha, right, parameters, head)
                - prediction(early, late, alpha, left, parameters, head)
            )
            / (2 * delta)
        )
    derivatives = []
    for key in ("kappa", "nu"):
        delta = step * max(1.0, abs(parameters[key]))
        left, right = dict(parameters), dict(parameters)
        left[key] = max(0.0, parameters[key] - delta)
        right[key] += delta
        derivatives.append(
            (prediction(early, late, alpha, shape, right, head) - prediction(early, late, alpha, shape, left, head))
            / (right[key] - left[key])
        )
    nuisance_matrix = np.column_stack(nuisance)
    norms = np.linalg.norm(nuisance_matrix, axis=0)
    nuisance_matrix = nuisance_matrix[:, norms > 0] / norms[norms > 0]
    left, singular, _ = np.linalg.svd(nuisance_matrix, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * svd_relative_tolerance))
    basis = left[:, :rank]
    gradient = np.column_stack(derivatives)
    residual = gradient - basis @ (basis.T @ gradient)
    full_norm, residual_norm = np.linalg.norm(gradient, axis=0), np.linalg.norm(residual, axis=0)
    scaled = np.divide(residual, full_norm, out=np.zeros_like(residual), where=full_norm > 0)
    pair_singular = np.linalg.svd(scaled, compute_uv=False)
    return {
        "rows": len(early),
        "nuisance_rank": rank,
        "kinetic_derivative_rms_bpb": (full_norm / np.sqrt(len(early))).tolist(),
        "residual_derivative_rms_bpb": (residual_norm / np.sqrt(len(early))).tolist(),
        "retained_norm_fraction": (
            np.divide(residual_norm, full_norm, out=np.zeros_like(residual_norm), where=full_norm > 0).tolist()
        ),
        "scaled_residual_singular_values": pair_singular.tolist(),
        "scaled_residual_singular_ratio": float(pair_singular[-1] / pair_singular[0]) if pair_singular[0] else 0.0,
        "derivatives": gradient.tolist(),
        "residual_derivatives": residual.tolist(),
    }


def source_identity() -> dict[str, Any]:
    paths = {
        Path(__file__).resolve(),
        Path(benchmark.__file__),
        Path(joint.__file__),
        Path(historical.__file__),
        Path(odds.__file__),
        Path(joint.temporal.__file__),
        Path(mariner.__file__),
        ROOT / "two_phase_benchmark_data_20260921.py",
        ROOT / "two_phase_benchmark_scoring_20260921.py",
        OUTPUT / "registration.json",
        ROUND5 / "execution_identity.json",
        ROUND3 / "result_manifest.json",
        TRUTH_SOURCE,
        TRUTH_SOURCE.parent / "complete.json",
        DATA / "arrays.npz",
        DATA / "folds.csv",
        DATA / "manifest.json",
    }
    return {
        "sources": {str(path.resolve()): benchmark.sha256(path) for path in sorted(paths)},
        "environment": benchmark.environment_identity(),
    }


def registration() -> dict[str, Any]:
    value = json.loads((OUTPUT / "registration.json").read_text())
    if value["candidate_id"] != "TP40-026" or value["status"] != "registered":
        raise ValueError("Root must register the reviewed TP40-026 protocol before execution")
    if tuple(value["families"]) != FAMILIES or tuple(value["kappa_grid"]) != odds.STRENGTHS:
        raise ValueError("Registered model/grid differs")
    if tuple(value["nu_grid"]) != RATE_GRID:
        raise ValueError("Registered harm grid differs")
    if value["environment"] != benchmark.environment_identity():
        raise ValueError("Registered numerical environment differs")
    benchmark.verify_hashes(value["frozen_inputs"])
    return value


def freeze() -> None:
    registration()
    reference = json.loads((ROUND5 / "execution_identity.json").read_text())
    for module in (joint, historical, odds, joint.temporal, mariner):
        path = Path(module.__file__).resolve()
        if benchmark.sha256(path) != reference[str(path)]:
            raise ValueError(f"Frozen scientific dependency changed: {path}")
    current = source_identity()
    destination = OUTPUT / "execution_identity.json"
    if destination.exists() and json.loads(destination.read_text()) != current:
        raise ValueError("Preserve previous execution identity; register a separate amended execution")
    benchmark.write_json(destination, current)


def execution_identity() -> dict[str, Any]:
    current = source_identity()
    if json.loads((OUTPUT / "execution_identity.json").read_text()) != current:
        raise ValueError("Frozen source/input/runtime changed")
    approval = json.loads((OUTPUT / "execution_approval.json").read_text())
    expected = {
        "scope": "rate_reopening_preflight_and_oracle",
        "registration_sha256": benchmark.sha256(OUTPUT / "registration.json"),
        "execution_identity_sha256": benchmark.sha256(OUTPUT / "execution_identity.json"),
    }
    if any(approval.get(key) != value for key, value in expected.items()):
        raise ValueError("Execution approval does not name this frozen registration")
    return current


def inputs() -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any], mariner.Head]:
    with np.load(DATA / "arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ("phase0_epochs", "phase1_epochs", "alpha", "tied")}
    folds = pd.read_csv(DATA / "folds.csv")
    truth = json.loads(TRUTH_SOURCE.read_text())
    raw = truth["head"]
    return (
        arrays,
        folds[folds.outer_fold.eq(0)],
        truth,
        mariner.Head(raw["intercept"], np.asarray(raw["coefficients"]), raw["floor"]),
    )


def world_inputs(
    arrays: dict[str, np.ndarray],
    folds: pd.DataFrame,
    truth: dict[str, Any],
    head: mariner.Head,
    parameters: dict[str, float],
) -> dict[str, Any]:
    early, late, alpha = arrays["phase0_epochs"], arrays["phase1_epochs"], float(arrays["alpha"])
    train = folds.loc[folds.role.eq("train"), "row_index"].to_numpy(int)
    held = folds.loc[folds.role.eq("test"), "row_index"].to_numpy(int)
    tied = np.flatnonzero(arrays["tied"])
    if (len(train), len(held), len(tied)) != (172, 66, 42) or arrays["tied"][train].any():
        raise ValueError("Frozen asymmetric-only geometry changed")
    labels = folds.loc[folds.role.eq("train"), "inner_fold"].to_numpy(int)
    total = early + late
    matrix = design(early, late, alpha, truth["shape"], parameters)
    linear = head.intercept + matrix @ head.coefficients
    tied_linear = head.intercept + mariner.design_matrix(total, truth["shape"]) @ head.coefficients
    clean = head.floor + np.exp(linear)
    clean_tied = head.floor + np.exp(tied_linear)
    return {
        "early": early,
        "late": late,
        "alpha": alpha,
        "train": train,
        "held": held,
        "tied": tied,
        "labels": labels,
        "total": total,
        "linear": linear,
        "tied_linear": tied_linear,
        "clean": clean,
        "clean_tied": clean_tied,
    }


def preflight() -> None:
    frozen, config = execution_identity(), registration()
    arrays, folds, truth, head = inputs()
    criteria = config["preflight"]
    records: list[dict[str, Any]] = []
    for world in config["worlds"]:
        parameters = world["parameters"]
        data = world_inputs(arrays, folds, truth, head, parameters)
        train = data["train"]
        partitions = [("outer_train", train)] + [
            (f"inner_train_{label}", train[data["labels"] != label]) for label in sorted(set(data["labels"]))
        ]
        slices: list[dict[str, Any]] = []
        for name, indices in partitions:
            floor = floor_report(data["clean"][indices], head.floor)
            reports = [
                local_sensitivity(
                    data["early"][indices],
                    data["late"][indices],
                    data["alpha"],
                    truth["shape"],
                    parameters,
                    head,
                    step,
                    criteria["nuisance_svd_relative_tolerance"],
                )
                for step in criteria["finite_difference_steps"]
            ]
            gradient, refined = (np.asarray(item["derivatives"]) for item in reports)
            denominator = np.maximum(np.linalg.norm(refined, axis=0), 1e-15)
            relative_error = np.linalg.norm(gradient - refined, axis=0) / denominator
            stable = bool(np.all(relative_error <= criteria["derivative_step_relative_error_max"]))
            resolved = all(
                min(item["retained_norm_fraction"]) >= criteria["minimum_retained_norm_fraction"]
                and item["scaled_residual_singular_ratio"] >= criteria["minimum_residual_singular_ratio"]
                for item in reports
            )
            slices.append(
                {
                    "partition": name,
                    "row_indices": indices.tolist(),
                    "floor": floor,
                    "sensitivity": reports,
                    "derivative_step_relative_error": relative_error.tolist(),
                    "local_sensitivity_resolved": resolved,
                    "derivative_stable": stable,
                }
            )
        native_tied = head.predict(mariner.design_matrix(data["total"], truth["shape"]))
        law_tied = prediction(
            data["alpha"] * data["total"],
            (1 - data["alpha"]) * data["total"],
            data["alpha"],
            truth["shape"],
            parameters,
            head,
        )
        tied_error = float(np.max(abs(law_tied - native_tied)))
        unclipped = all(bool(np.all((data[key] > -30) & (data[key] < 30))) for key in ("linear", "tied_linear"))
        minimum_deficit = min(float(np.min(data[key] - head.floor)) for key in ("clean", "clean_tied"))
        deficit_unclamped = minimum_deficit >= mariner.DEFICIT_FLOOR
        oracle_execution_valid = (
            unclipped and deficit_unclamped and tied_error <= criteria["structural_tied_max_abs_bpb"]
        )
        passed = oracle_execution_valid and all(
            item["floor"]["feasible"] and item["local_sensitivity_resolved"] and item["derivative_stable"]
            for item in slices
        )
        records.append(
            {
                "world": world,
                "partitions": slices,
                "unclipped_truth": unclipped,
                "minimum_truth_deficit": minimum_deficit,
                "native_deficit_clamp_inactive": deficit_unclamped,
                "tied_max_abs_error": tied_error,
                "oracle_execution_valid": oracle_execution_valid,
                "passed": passed,
            }
        )
    benchmark.write_json(
        OUTPUT / "preflight.json",
        {
            "identity": frozen,
            "worlds": records,
            "passed": all(item["passed"] for item in records),
            "oracle_execution_valid": all(item["oracle_execution_valid"] for item in records),
            "interpretation": "Local geometry/floor screen only; positive rank does not prove global identification.",
        },
    )
    benchmark.write_json(
        OUTPUT / "preflight_receipt.json",
        {"identity": execution_identity(), "outputs": {"preflight.json": benchmark.sha256(OUTPUT / "preflight.json")}},
    )


def oracle() -> None:
    frozen, config = execution_identity(), registration()
    preflight_receipt = json.loads((OUTPUT / "preflight_receipt.json").read_text())
    if preflight_receipt["identity"] != frozen:
        raise ValueError("Preflight receipt identity differs")
    benchmark.verify_hashes(preflight_receipt["outputs"], OUTPUT)
    preflight_record = json.loads((OUTPUT / "preflight.json").read_text())
    if preflight_record["identity"] != frozen or not preflight_record["oracle_execution_valid"]:
        raise ValueError("Synthetic truth is invalid; preserve the diagnostic and do not fit the oracle")
    arrays, folds, truth, head = inputs()
    tolerance, records = config["oracle"]["maximum_absolute_prediction_error_bpb"], []
    for world in config["worlds"]:
        folder = OUTPUT / "oracle" / world["name"]
        complete = folder / "complete.json"
        if complete.exists():
            receipt = json.loads(complete.read_text())
            if receipt["identity"] != frozen or receipt["world"] != world:
                raise ValueError("Oracle receipt source identity changed")
            if receipt["preflight_receipt_sha256"] != benchmark.sha256(OUTPUT / "preflight_receipt.json"):
                raise ValueError("Oracle preflight receipt changed")
            benchmark.verify_hashes(receipt["outputs"], folder)
            records.append(json.loads((folder / "diagnostic.json").read_text()))
            continue
        data = world_inputs(arrays, folds, truth, head, world["parameters"])
        train, held, tied = data["train"], data["held"], data["tied"]
        asymmetric_count, actual_tied_end = len(held), len(held) + len(tied)
        query_early = np.concatenate([data["early"][held], data["early"][tied], data["alpha"] * data["total"][held]])
        query_late = np.concatenate([data["late"][held], data["late"][tied], (1 - data["alpha"]) * data["total"][held]])
        clean = np.r_[data["clean"][held], data["clean"][tied], data["clean_tied"][held]]
        fitted = oracle_fit(
            data["early"][train],
            data["late"][train],
            data["clean"][train],
            data["labels"],
            query_early,
            query_late,
            data["alpha"],
            truth["shape"],
            head.floor,
        )
        selected = fitted["profile"][fitted["selected_parameter_index"]]
        predicted = np.asarray(selected["query_predictions"])
        eligible = [item for item in fitted["profile"] if item["inner_rmse"] <= tolerance]
        alternative_error = max(
            (float(np.max(abs(np.asarray(item["query_predictions"]) - clean))) for item in eligible),
            default=None,
        )
        error = predicted - clean
        gain_error = (predicted[actual_tied_end:] - predicted[:asymmetric_count]) - (
            clean[actual_tied_end:] - clean[:asymmetric_count]
        )
        diagnostic = {
            "world": world,
            "selected_parameters": selected["parameters"],
            "selected_inner_rmse": selected["inner_rmse"],
            "query_max_abs_error": float(np.max(abs(error))),
            "gain_rmse": float(np.sqrt(np.mean(gain_error**2))),
            "asymmetric_rmse": float(np.sqrt(np.mean(error[:asymmetric_count] ** 2))),
            "actual_tied_rmse": float(np.sqrt(np.mean(error[asymmetric_count:actual_tied_end] ** 2))),
            "counterfactual_tied_rmse": float(np.sqrt(np.mean(error[actual_tied_end:] ** 2))),
            "near_exact_cv_candidate_parameters": [item["parameters"] for item in eligible],
            "near_exact_cv_max_query_error": alternative_error,
            "kinetics_unique_at_oracle_tolerance": len(eligible) == 1,
            "passed": bool(
                selected["inner_rmse"] <= tolerance
                and np.max(abs(error)) <= tolerance
                and alternative_error is not None
                and alternative_error <= tolerance
                and np.max(abs(gain_error)) <= tolerance
            ),
        }
        benchmark.write_json(folder / "fit.json", fitted)
        benchmark.write_json(folder / "diagnostic.json", diagnostic)
        benchmark.write_json(
            complete,
            {
                "identity": execution_identity(),
                "world": world,
                "preflight_receipt_sha256": benchmark.sha256(OUTPUT / "preflight_receipt.json"),
                "outputs": {name: benchmark.sha256(folder / name) for name in ("fit.json", "diagnostic.json")},
            },
        )
        records.append(diagnostic)
    benchmark.write_json(
        OUTPUT / "oracle_summary.json",
        {
            "identity": frozen,
            "worlds": records,
            "passed": all(item["passed"] for item in records),
            "preflight_advancement_passed": preflight_record["passed"],
            "stage2_prerequisites_satisfied": preflight_record["passed"] and all(item["passed"] for item in records),
            "interpretation": "Known-shape/absolute-floor representability check; not qualification of learned J2.",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "preflight", "oracle"))
    args = parser.parse_args()
    {"freeze": freeze, "preflight": preflight, "oracle": oracle}[args.stage]()


if __name__ == "__main__":
    main()
