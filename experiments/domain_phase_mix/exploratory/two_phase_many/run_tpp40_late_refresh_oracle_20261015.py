# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Frozen two-world, known-shape/floor late-refresh oracle; no learned estimator.

Every data-reading stage requires a registered protocol and matching execution
approval. Only physical covariates and split indices are read from the archived
panel. Actual outcome arrays are never indexed. The draft cannot execute.
"""

from __future__ import annotations

import argparse
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
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_late_refresh_20261015 as law
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_rate_reopening_20261015 as prior

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reference_outputs/tpp40_late_refresh_20261015"
DATA = prior.DATA
TRUTH_SOURCE = prior.TRUTH_SOURCE
PRIOR_ROOT = prior.OUTPUT
WORLDS: tuple[dict[str, Any], ...] = ({"name": "null", "lambda": 0.0}, {"name": "positive", "lambda": 0.5})
SHAPE = {"rate": 0.25, "power": 0.7, "threshold": 3.0}
FLOOR = 0.6
TOLERANCE = 1e-8
SENSITIVITY_STEPS = (1e-4, 5e-5)
SVD_TOLERANCE = 1e-10
EXECUTION_CONTRACT = {
    "outer_fold": 0,
    "training_rows": 172,
    "held_rows": 66,
    "actual_tied_rows": 42,
    "buckets": 39,
    "inner_folds": 3,
    "shape": SHAPE,
    "floor": FLOOR,
    "ridge": 0.0,
    "noise_sd": 0.0,
    "lambda_grid": list(law.STRENGTHS),
    "worlds": list(WORLDS),
    "tolerance_bpb": TOLERANCE,
    "finite_difference_steps": list(SENSITIVITY_STEPS),
    "nuisance_svd_relative_tolerance": SVD_TOLERANCE,
}
mariner = joint.mariner


def design(early: np.ndarray, late: np.ndarray, alpha: float, shape: dict[str, float], strength: float) -> np.ndarray:
    return law.design_and_offset(early, late, alpha, shape, "late_refresh_hazard", {"lambda": strength})[0]


def head_record(head: mariner.Head) -> dict[str, Any]:
    return {"intercept": head.intercept, "coefficients": head.coefficients.tolist(), "floor": head.floor}


def clipping_count(matrix: np.ndarray, head: mariner.Head) -> int:
    linear = head.intercept + matrix @ head.coefficients
    return int(np.count_nonzero((linear <= -30) | (linear >= 30)))


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
    """Score all seven strengths with training-only native NNLS at known shape/floor."""
    if not np.isfinite(observed).all() or np.any(observed - floor < mariner.DEFICIT_FLOOR):
        raise ValueError("Oracle labels must be finite with the native deficit clamp inactive")
    profile = []
    splits = joint.temporal.inner_splits(labels)
    with threadpool_limits(limits=1):
        for index, strength in enumerate(law.STRENGTHS):
            matrix = design(early, late, alpha, shape, strength)
            query_matrix = design(query_early, query_late, alpha, shape, strength)
            inner = np.empty(len(observed))
            inner_heads = []
            for train, held in splits:
                fitted = joint.fit_fixed_head(matrix[train], np.zeros(len(train)), observed[train], 0.0, None, floor)
                inner[held] = fitted.predict(matrix[held])
                inner_heads.append(
                    {
                        "train_indices": train.tolist(),
                        "held_indices": held.tolist(),
                        "head": head_record(fitted),
                        "training_clipped_count": clipping_count(matrix[train], fitted),
                        "held_clipped_count": clipping_count(matrix[held], fitted),
                    }
                )
            head = joint.fit_fixed_head(matrix, np.zeros(len(early)), observed, 0.0, None, floor)
            query = head.predict(query_matrix)
            if not np.isfinite(inner).all() or not np.isfinite(query).all():
                raise FloatingPointError("Nonfinite oracle prediction")
            profile.append(
                {
                    "parameter_index": index,
                    "parameters": {"lambda": strength},
                    "inner_rmse": float(np.sqrt(np.mean((inner - observed) ** 2))),
                    "inner_predictions": inner.tolist(),
                    "inner_heads": inner_heads,
                    "query_predictions": query.tolist(),
                    "head": head_record(head),
                    "training_clipped_count": clipping_count(matrix, head),
                    "query_clipped_count": clipping_count(query_matrix, head),
                }
            )
    selected = min(profile, key=lambda item: (item["inner_rmse"], item["parameter_index"]))
    return {"selected_parameter_index": selected["parameter_index"], "profile": profile}


def oracle_diagnostic(
    fitted: dict[str, Any], clean: np.ndarray, asymmetric_count: int, actual_tied_count: int
) -> dict[str, Any]:
    """Check finite-menu functional recovery, including every near-exact CV candidate."""
    selected = fitted["profile"][fitted["selected_parameter_index"]]
    predicted = np.asarray(selected["query_predictions"])
    tied_end = asymmetric_count + actual_tied_count
    error = predicted - clean
    gain_error = (predicted[tied_end:] - predicted[:asymmetric_count]) - (clean[tied_end:] - clean[:asymmetric_count])
    eligible = [item for item in fitted["profile"] if item["inner_rmse"] <= TOLERANCE]
    alternatives = [float(np.max(abs(np.asarray(item["query_predictions"]) - clean))) for item in eligible]
    alternative_error = max(alternatives) if alternatives else None
    return {
        "selected_parameters": selected["parameters"],
        "selected_inner_rmse": selected["inner_rmse"],
        "query_max_abs_error": float(np.max(abs(error))),
        "gain_max_abs_error": float(np.max(abs(gain_error))),
        "gain_rmse": float(np.sqrt(np.mean(gain_error**2))),
        "asymmetric_rmse": float(np.sqrt(np.mean(error[:asymmetric_count] ** 2))),
        "actual_tied_rmse": float(np.sqrt(np.mean(error[asymmetric_count:tied_end] ** 2))),
        "counterfactual_tied_rmse": float(np.sqrt(np.mean(error[tied_end:] ** 2))),
        "near_exact_cv_candidate_parameters": [item["parameters"] for item in eligible],
        "near_exact_cv_max_query_error": alternative_error,
        "kinetics_unique_at_oracle_tolerance": len(eligible) == 1,
        "passed": bool(
            selected["inner_rmse"] <= TOLERANCE
            and np.max(abs(error)) <= TOLERANCE
            and np.max(abs(gain_error)) <= TOLERANCE
            and alternative_error is not None
            and alternative_error <= TOLERANCE
        ),
    }


def local_sensitivity(
    early: np.ndarray, late: np.ndarray, alpha: float, strength: float, head: mariner.Head, step: float
) -> dict[str, Any]:
    """Describe local refresh sensitivity after unconstrained static-nuisance projection."""
    matrix = design(early, late, alpha, SHAPE, strength)
    base = head.predict(matrix)
    deficit = base - head.floor
    nuisance = [np.ones(len(base)), deficit, *(deficit[:, None] * matrix).T]
    for key in SHAPE:
        delta = step * max(1.0, abs(SHAPE[key]))
        left, right = dict(SHAPE), dict(SHAPE)
        left[key] -= delta
        right[key] += delta
        nuisance.append(
            (
                head.predict(design(early, late, alpha, right, strength))
                - head.predict(design(early, late, alpha, left, strength))
            )
            / (2 * delta)
        )
    lower, upper = max(0.0, strength - step), min(1.0, strength + step)
    gradient = (
        head.predict(design(early, late, alpha, SHAPE, upper)) - head.predict(design(early, late, alpha, SHAPE, lower))
    ) / (upper - lower)
    nuisance_matrix = np.column_stack(nuisance)
    norms = np.linalg.norm(nuisance_matrix, axis=0)
    nuisance_matrix = nuisance_matrix[:, norms > 0] / norms[norms > 0]
    basis, singular, _ = np.linalg.svd(nuisance_matrix, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * SVD_TOLERANCE))
    residual = gradient - basis[:, :rank] @ (basis[:, :rank].T @ gradient)
    full_norm, residual_norm = np.linalg.norm(gradient), np.linalg.norm(residual)
    return {
        "step": step,
        "rows": len(early),
        "nuisance_rank": rank,
        "derivative_rms_bpb": float(full_norm / np.sqrt(len(early))),
        "residual_derivative_rms_bpb": float(residual_norm / np.sqrt(len(early))),
        "retained_norm_fraction": float(residual_norm / full_norm) if full_norm else 0.0,
        "derivatives": gradient.tolist(),
        "residual_derivatives": residual.tolist(),
        "interpretation": "Descriptive local nuisance projection, not a gate or global identification test.",
    }


def adoption_paths() -> set[Path]:
    return {
        Path(benchmark.__file__),
        Path(joint.__file__),
        Path(joint.temporal.__file__),
        Path(joint.mechanisms.__file__),
        Path(prior.__file__),
        Path(prior.odds.__file__),
        Path(mariner.__file__),
        ROOT / "two_phase_benchmark_data_20260921.py",
        ROOT / "two_phase_benchmark_scoring_20260921.py",
        DATA / "arrays.npz",
        DATA / "folds.csv",
        DATA / "manifest.json",
        TRUTH_SOURCE,
        TRUTH_SOURCE.parent / "complete.json",
        prior.ROUND3 / "result_manifest.json",
        prior.ROUND5 / "execution_identity.json",
    }


def source_paths() -> set[Path]:
    return adoption_paths() | {
        Path(__file__).resolve(),
        Path(law.__file__),
        ROOT / "test_tpp40_late_refresh_20261015.py",
        ROOT / "test_tpp40_late_refresh_oracle_20261015.py",
        PRIOR_ROOT / "execution_identity.json",
        PRIOR_ROOT / "result_manifest.json",
        OUTPUT / "registration.json",
    }


def verify_adoption() -> None:
    """Require exact relevant dependency/data parity with sealed TP40-026 stage1."""
    previous = json.loads((PRIOR_ROOT / "execution_identity.json").read_text())["sources"]
    for path in adoption_paths():
        if benchmark.sha256(path) != previous[str(path.resolve())]:
            raise ValueError(f"Adopted source differs from sealed stage1: {path}")
    seal = json.loads((PRIOR_ROOT / "result_manifest.json").read_text())
    if benchmark.sha256(PRIOR_ROOT / "execution_identity.json") != seal["files"]["execution_identity.json"]["sha256"]:
        raise ValueError("Stage1 adoption identity differs from its seal")
    receipt = json.loads((TRUTH_SOURCE.parent / "complete.json").read_text())
    if benchmark.sha256(TRUTH_SOURCE) != receipt["outputs"]["truth.json"]:
        raise ValueError("Frozen synthetic head differs from its completion receipt")


def registration() -> dict[str, Any]:
    config = json.loads((OUTPUT / "registration.json").read_text())
    if config["candidate_id"] != "TP40-028" or config["status"] != "registered":
        raise ValueError("Root must register the reviewed TP40-028 draft before execution")
    proposal = config["oracle_proposal"]
    if not proposal["implemented"] or config["execution_contract"] != EXECUTION_CONTRACT:
        raise ValueError("Registered oracle implementation contract differs")
    if tuple(config["lambda_grid"]) != law.STRENGTHS or tuple(proposal["grid"]) != law.STRENGTHS:
        raise ValueError("Registered strength grid differs")
    if tuple(proposal["worlds"]) != WORLDS or proposal["noise_sd"] != 0 or proposal["tolerance_bpb"] != TOLERANCE:
        raise ValueError("Registered worlds, noise or oracle tolerance differ")
    if config["environment"] != benchmark.environment_identity():
        raise ValueError("Registered numerical runtime differs")
    expected_paths = {str(path.resolve()) for path in source_paths() - {OUTPUT / "registration.json"}}
    if set(config["frozen_reference_inputs"]) != expected_paths:
        raise ValueError("Registration must pin the complete reviewed source/input set")
    benchmark.verify_hashes(config["frozen_reference_inputs"])
    verify_adoption()
    return config


def source_identity() -> dict[str, Any]:
    return {
        "sources": {str(path.resolve()): benchmark.sha256(path) for path in sorted(source_paths())},
        "environment": benchmark.environment_identity(),
    }


def freeze() -> None:
    registration()
    current = source_identity()
    destination = OUTPUT / "execution_identity.json"
    if destination.exists() and json.loads(destination.read_text()) != current:
        raise ValueError("Preserve prior execution identity; register a separate amended execution")
    benchmark.write_json(destination, current)


def execution_identity() -> dict[str, Any]:
    registration()
    current = source_identity()
    if json.loads((OUTPUT / "execution_identity.json").read_text()) != current:
        raise ValueError("Frozen source/input/runtime changed")
    approval = json.loads((OUTPUT / "execution_approval.json").read_text())
    expected = {
        "scope": "late_refresh_known_shape_floor_oracle",
        "registration_sha256": benchmark.sha256(OUTPUT / "registration.json"),
        "execution_identity_sha256": benchmark.sha256(OUTPUT / "execution_identity.json"),
    }
    if any(approval.get(key) != value for key, value in expected.items()):
        raise ValueError("Execution approval does not name this frozen registration")
    return current


def inputs() -> tuple[dict[str, np.ndarray], pd.DataFrame, mariner.Head]:
    with np.load(DATA / "arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ("phase0_epochs", "phase1_epochs", "alpha", "tied")}
    folds = pd.read_csv(DATA / "folds.csv")
    truth = json.loads(TRUTH_SOURCE.read_text())
    raw = truth["head"]
    if truth["shape"] != SHAPE or raw["floor"] != FLOOR or len(raw["coefficients"]) != 78:
        raise ValueError("Frozen nonlinear truth head changed")
    return (
        arrays,
        folds[folds.outer_fold.eq(0)],
        mariner.Head(raw["intercept"], np.asarray(raw["coefficients"]), raw["floor"]),
    )


def world_inputs(
    arrays: dict[str, np.ndarray], folds: pd.DataFrame, head: mariner.Head, strength: float
) -> dict[str, Any]:
    early, late, alpha = arrays["phase0_epochs"], arrays["phase1_epochs"], float(arrays["alpha"])
    train = folds.loc[folds.role.eq("train"), "row_index"].to_numpy(int)
    held = folds.loc[folds.role.eq("test"), "row_index"].to_numpy(int)
    tied = np.flatnonzero(arrays["tied"])
    labels = folds.loc[folds.role.eq("train"), "inner_fold"].to_numpy(int)
    asymmetric = np.flatnonzero(~arrays["tied"])
    if (len(train), len(held), len(tied)) != (172, 66, 42) or early.shape != (280, 39) or late.shape != early.shape:
        raise ValueError("Frozen panel geometry changed")
    if not np.array_equal(np.sort(np.r_[train, held]), asymmetric) or set(labels) != {0, 1, 2}:
        raise ValueError("Training/test partitions must cover asymmetric rows exactly with three inner folds")
    joint.temporal.inner_splits(labels)
    total = early + late
    matrix = design(early, late, alpha, SHAPE, strength)
    native = mariner.design_matrix(total, SHAPE)
    linear, tied_linear = head.intercept + matrix @ head.coefficients, head.intercept + native @ head.coefficients
    with np.errstate(over="ignore", invalid="ignore"):
        clean, clean_tied = head.floor + np.exp(linear), head.floor + np.exp(tied_linear)
    return {
        "early": early,
        "late": late,
        "alpha": alpha,
        "total": total,
        "train": train,
        "held": held,
        "tied": tied,
        "labels": labels,
        "linear": linear,
        "tied_linear": tied_linear,
        "clean": clean,
        "clean_tied": clean_tied,
    }


def verified_record(receipt_path: Path, frozen: dict[str, Any], output_names: tuple[str, ...]) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text())
    if receipt["identity"] != frozen or set(receipt["outputs"]) != set(output_names):
        raise ValueError("Completed artifact identity or output set changed")
    benchmark.verify_hashes(receipt["outputs"], receipt_path.parent)
    return receipt


def preflight() -> None:
    frozen = execution_identity()
    receipt_path = OUTPUT / "preflight_receipt.json"
    if receipt_path.exists():
        verified_record(receipt_path, frozen, ("preflight.json",))
        return
    arrays, folds, head = inputs()
    records = []
    for world in WORLDS:
        data = world_inputs(arrays, folds, head, world["lambda"])
        finite = all(np.isfinite(data[key]).all() for key in ("linear", "tied_linear", "clean", "clean_tied"))
        unclipped = finite and all(
            bool(np.all((data[key] > -30) & (data[key] < 30))) for key in ("linear", "tied_linear")
        )
        minimum = min(float(np.min(data[key] - head.floor)) for key in ("clean", "clean_tied")) if finite else None
        unclamped = minimum is not None and minimum >= mariner.DEFICIT_FLOOR
        native = mariner.design_matrix(data["total"], SHAPE)
        tied = design(
            data["alpha"] * data["total"], (1 - data["alpha"]) * data["total"], data["alpha"], SHAPE, world["lambda"]
        )
        native_recombined = mariner.design_matrix(
            data["alpha"] * data["total"] + (1 - data["alpha"]) * data["total"], SHAPE
        )
        exact_tied_design = bool(np.array_equal(tied, native_recombined))
        tied_error = float(np.max(abs(head.predict(tied) - head.predict(native))))
        valid = bool(finite and unclipped and unclamped and exact_tied_design and tied_error <= TOLERANCE)
        slices: list[dict[str, Any]] = []
        if valid:
            partitions = [("outer_train", data["train"])] + [
                (f"inner_train_{label}", data["train"][data["labels"] != label]) for label in sorted(set(data["labels"]))
            ]
            for name, indices in partitions:
                sensitivity = [
                    local_sensitivity(
                        data["early"][indices], data["late"][indices], data["alpha"], world["lambda"], head, step
                    )
                    for step in SENSITIVITY_STEPS
                ]
                gradient, refined = (np.asarray(item["derivatives"]) for item in sensitivity)
                slices.append(
                    {
                        "partition": name,
                        "row_indices": indices.tolist(),
                        "floor": prior.floor_report(data["clean"][indices], head.floor),
                        "sensitivity": sensitivity,
                        "derivative_step_relative_error": float(
                            np.linalg.norm(gradient - refined) / max(np.linalg.norm(refined), 1e-15)
                        ),
                    }
                )
        records.append(
            {
                "world": world,
                "truth_head": head_record(head),
                "shape": SHAPE,
                "finite_truth": bool(finite),
                "unclipped_truth": bool(unclipped),
                "minimum_truth_deficit": minimum,
                "native_deficit_clamp_inactive": bool(unclamped),
                "exact_tied_design": exact_tied_design,
                "tied_max_abs_error": tied_error,
                "partitions": slices,
                "oracle_execution_valid": valid,
                "learned_floor_domain_feasible": valid and all(item["floor"]["feasible"] for item in slices),
                "train_gain_rms": (
                    float(np.sqrt(np.mean((data["clean_tied"][data["train"]] - data["clean"][data["train"]]) ** 2)))
                    if finite
                    else None
                ),
                "held_gain_rms": (
                    float(np.sqrt(np.mean((data["clean_tied"][data["held"]] - data["clean"][data["held"]]) ** 2)))
                    if finite
                    else None
                ),
            }
        )
    record = {
        "identity": frozen,
        "worlds": records,
        "oracle_execution_valid": all(item["oracle_execution_valid"] for item in records),
        "learned_floor_domain_feasible": all(item["learned_floor_domain_feasible"] for item in records),
        "interpretation": (
            "Known-floor oracle validity is separate from learned-floor feasibility. Derivative conditioning is "
            "descriptive; slice-wise feasible gamma does not imply a common gamma."
        ),
    }
    benchmark.write_json(OUTPUT / "preflight.json", record)
    benchmark.write_json(
        receipt_path,
        {"identity": execution_identity(), "outputs": {"preflight.json": benchmark.sha256(OUTPUT / "preflight.json")}},
    )


def oracle() -> None:
    frozen = execution_identity()
    verified_record(OUTPUT / "preflight_receipt.json", frozen, ("preflight.json",))
    preflight_record = json.loads((OUTPUT / "preflight.json").read_text())
    if preflight_record["identity"] != frozen or not preflight_record["oracle_execution_valid"]:
        raise ValueError("Truth validity failed; preserve the diagnostic without fitting")
    arrays, folds, head = inputs()
    records = []
    for world in WORLDS:
        folder = OUTPUT / "oracle" / world["name"]
        complete = folder / "complete.json"
        if complete.exists():
            receipt = verified_record(complete, frozen, ("fit.json", "diagnostic.json", "evaluation.json"))
            if receipt["world"] != world or receipt["preflight_receipt_sha256"] != benchmark.sha256(
                OUTPUT / "preflight_receipt.json"
            ):
                raise ValueError("Oracle world or preflight receipt changed")
            records.append(json.loads((folder / "diagnostic.json").read_text()))
            continue
        data = world_inputs(arrays, folds, head, world["lambda"])
        train, held, tied = data["train"], data["held"], data["tied"]
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
            SHAPE,
            head.floor,
        )
        diagnostic = {"world": world, **oracle_diagnostic(fitted, clean, len(held), len(tied))}
        evaluation = {
            "world": world,
            "truth_head": head_record(head),
            "shape": SHAPE,
            "alpha": data["alpha"],
            "train_row_indices": train.tolist(),
            "train_inner_labels": data["labels"].tolist(),
            "train_observed": data["clean"][train].tolist(),
            "held_row_indices": held.tolist(),
            "tied_row_indices": tied.tolist(),
            "query_roles": (
                ["asymmetric"] * len(held) + ["actual_tied"] * len(tied) + ["counterfactual_tied"] * len(held)
            ),
            "query_row_indices": np.r_[held, tied, held].tolist(),
            "query_early": query_early.tolist(),
            "query_late": query_late.tolist(),
            "query_truth": clean.tolist(),
        }
        for name, value in (("fit.json", fitted), ("diagnostic.json", diagnostic), ("evaluation.json", evaluation)):
            benchmark.write_json(folder / name, value)
        benchmark.write_json(
            complete,
            {
                "identity": execution_identity(),
                "world": world,
                "preflight_receipt_sha256": benchmark.sha256(OUTPUT / "preflight_receipt.json"),
                "outputs": {
                    name: benchmark.sha256(folder / name) for name in ("fit.json", "diagnostic.json", "evaluation.json")
                },
            },
        )
        records.append(diagnostic)
    summary = {
        "identity": frozen,
        "worlds": records,
        "passed": all(item["passed"] for item in records),
        "learned_floor_domain_feasible": preflight_record["learned_floor_domain_feasible"],
        "eligible_for_separate_learned_procedure_review": (
            preflight_record["learned_floor_domain_feasible"] and all(item["passed"] for item in records)
        ),
        "interpretation": (
            "Known-shape/floor finite-grid functional check only; not learned-estimator or optimum qualification, "
            "and no authorization for another stage."
        ),
    }
    benchmark.write_json(OUTPUT / "oracle_summary.json", summary)
    benchmark.write_json(
        OUTPUT / "oracle_summary_receipt.json",
        {
            "identity": execution_identity(),
            "outputs": {"oracle_summary.json": benchmark.sha256(OUTPUT / "oracle_summary.json")},
            "world_receipts": {
                world["name"]: benchmark.sha256(OUTPUT / "oracle" / world["name"] / "complete.json") for world in WORLDS
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "preflight", "oracle"))
    args = parser.parse_args()
    {"freeze": freeze, "preflight": preflight, "oracle": oracle}[args.stage]()


if __name__ == "__main__":
    main()
