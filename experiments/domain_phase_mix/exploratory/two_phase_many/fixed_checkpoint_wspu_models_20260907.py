# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Conditional WSPU response fits for continuations of one fixed checkpoint.

Only training outcomes enter this module. Supplied inner folds index those
training rows; calibration rows remain in every inner training partition.
Response functions, constrained solver, and fitting constants are imported
from an explicitly supplied snapshot of the authoritative single-phase source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from fit_two_phase_link_spines_20260907 import load_module, write_json_atomic
from scipy.optimize import minimize_scalar

Folds = tuple[tuple[np.ndarray, np.ndarray], ...]


class Variant(StrEnum):
    CUMULATIVE_LOG = "cumulative_log"
    CUMULATIVE_BPB = "cumulative_bpb"
    CONTINUATION_LOG = "continuation_log"


@dataclass(frozen=True)
class ConditionalHead:
    intercept: float
    coefficients: np.ndarray
    floor: float | None
    clamped_deficits: int


@lru_cache(maxsize=8)
def load_response_source(path: str, digest: str) -> ModuleType:
    """Load a content-identified authoritative response implementation."""
    source = Path(path)
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError("response source does not match its recorded hash")
    return load_module(source, f"checkpoint_wspu_{digest[:16]}")


def response_source(path: Path) -> tuple[ModuleType, str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return load_response_source(str(path.resolve()), digest), digest


def branch_design(
    prefix_epochs: np.ndarray,
    continuation_epochs: np.ndarray,
    shape: dict[str, float],
    variant: Variant,
    source: ModuleType,
) -> np.ndarray:
    """Build benefit and harm columns conditional on a fixed prefix exposure."""
    prefix = np.asarray(prefix_epochs, float)
    continuation = np.asarray(continuation_epochs, float)
    if prefix.ndim != 1 or continuation.ndim != 2 or continuation.shape[1] != len(prefix):
        raise ValueError("expected one fixed prefix vector and an N by J continuation matrix")
    if not np.isfinite(prefix).all() or not np.isfinite(continuation).all():
        raise ValueError("exposures must be finite")
    if np.any(prefix < 0) or np.any(continuation < 0):
        raise ValueError("exposures must be nonnegative")
    if variant == Variant.CONTINUATION_LOG:
        return source.design_matrix(continuation, shape) - source.design_matrix(np.zeros_like(prefix)[None], shape)
    return source.design_matrix(prefix[None] + continuation, shape) - source.design_matrix(prefix[None], shape)


def validate_partitions(count: int, calibration: np.ndarray, folds: Folds) -> None:
    """Reject overlap, missing rows, and scored calibration anchors."""
    if calibration.shape != (count,) or calibration.dtype != bool:
        raise ValueError("calibration_mask must be a Boolean training-row vector")
    if not calibration.any():
        raise ValueError("at least one matched calibration row is required")
    expected = np.arange(count)
    validation_count = np.zeros(count, int)
    for train, validation in folds:
        if len(train) == 0 or len(validation) == 0:
            raise ValueError("inner partitions must be nonempty")
        if not np.array_equal(np.sort(np.concatenate([train, validation])), expected):
            raise ValueError("each inner fold must partition the supplied training cohort exactly")
        if np.any(calibration[validation]):
            raise ValueError("calibration rows cannot enter inner validation")
        validation_count[validation] += 1
    if not np.all(validation_count[~calibration] == 1) or not np.all(validation_count[calibration] == 0):
        raise ValueError("each noncalibration training row must be scored exactly once")


def fit_conditional_head(
    matrix: np.ndarray,
    response: np.ndarray,
    ridge: float,
    variant: Variant,
    anchor: float,
    noise_sd: float,
    kappa: float,
    source: ModuleType,
) -> ConditionalHead:
    if variant == Variant.CUMULATIVE_BPB:
        intercept, coefficients = source.nonnegative_solve(matrix, response, ridge)
        return ConditionalHead(float(intercept), coefficients, None, 0)
    spec = source.FloorSpec(anchor, noise_sd, kappa)
    head = source.fit_head(matrix, response, ridge, spec)
    clamped = int(np.sum(response - head.floor < source.DEFICIT_FLOOR))
    return ConditionalHead(float(head.intercept), head.coefficients, float(head.floor), clamped)


def head_prediction(matrix: np.ndarray, head: ConditionalHead, source: ModuleType) -> np.ndarray:
    linear = head.intercept + matrix @ head.coefficients
    if head.floor is None:
        return linear
    return head.floor + np.exp(np.clip(linear, -source.LOG_CLIP, source.LOG_CLIP))


def score_configuration(
    matrix: np.ndarray,
    response: np.ndarray,
    ridge: float,
    variant: Variant,
    anchor: float,
    noise_sd: float,
    kappa: float,
    folds: Folds,
    source: ModuleType,
) -> float:
    error = 0.0
    count = 0
    for train, validation in folds:
        head = fit_conditional_head(matrix[train], response[train], ridge, variant, anchor, noise_sd, kappa, source)
        prediction = head_prediction(matrix[validation], head, source)
        if not np.isfinite(prediction).all():
            return float("inf")
        error += float(np.sum((prediction - response[validation]) ** 2))
        count += len(validation)
    return math.sqrt(error / count)


def fit_model(
    prefix_epochs: np.ndarray,
    continuation_epochs: np.ndarray,
    response: np.ndarray,
    calibration_mask: np.ndarray,
    inner_folds: Folds,
    *,
    variant: Variant,
    source_path: Path,
    noise_sd: float,
) -> dict[str, Any]:
    """Select shape and ridge, then the conditional response floor, using inner CV."""
    source, source_hash = response_source(source_path)
    response = np.asarray(response, float)
    continuation = np.asarray(continuation_epochs, float)
    if response.ndim != 1 or continuation.shape[0] != len(response) or not np.isfinite(response).all():
        raise ValueError("finite response must have one value per continuation")
    if not math.isfinite(noise_sd) or noise_sd < 0:
        raise ValueError("noise_sd must be an explicit nonnegative matched repeat estimate, or zero")
    validate_partitions(len(response), calibration_mask, inner_folds)
    anchor = float(np.mean(response[calibration_mask]))
    best: tuple[float, int, int] | None = None
    grid_records = []
    for shape_index, shape in enumerate(source.SHAPES):
        matrix = branch_design(prefix_epochs, continuation, shape, variant, source)
        for ridge_index, ridge in enumerate(source.RIDGE_GRID):
            score = score_configuration(
                matrix, response, ridge, variant, anchor, noise_sd, source.KAPPA_PROVISIONAL, inner_folds, source
            )
            grid_records.append([shape_index, ridge_index, float(score)])
            proposal = (score, shape_index, ridge_index)
            if best is None or proposal < best:
                best = proposal
    if best is None or not math.isfinite(best[0]):
        raise ValueError("no finite inner-CV model")
    _, shape_index, ridge_index = best
    shape = dict(source.SHAPES[shape_index])
    ridge = float(source.RIDGE_GRID[ridge_index])
    matrix = branch_design(prefix_epochs, continuation, shape, variant, source)
    kappa = source.KAPPA_PROVISIONAL
    flat_profile = False
    search_records = []
    if variant != Variant.CUMULATIVE_BPB:

        def objective(log_kappa: float) -> float:
            value = score_configuration(
                matrix, response, ridge, variant, anchor, noise_sd, math.exp(log_kappa), inner_folds, source
            )
            search_records.append([float(math.exp(log_kappa)), float(value)])
            return value if math.isfinite(value) else source.INFINITE_CV_PENALTY

        low, high = source.KAPPA_BOUNDS
        search = minimize_scalar(
            objective,
            bounds=(math.log(low), math.log(high)),
            method="bounded",
            options={"maxiter": source.KAPPA_SEARCH_EVALUATIONS, "xatol": source.KAPPA_SEARCH_XATOL},
        )
        kappa = float(math.exp(search.x))
        flat_profile = math.log(kappa) >= (1 - source.FLAT_PROFILE_FRACTION) * math.log(high)
        if flat_profile:
            kappa = source.FLAT_PROFILE_KAPPA
    score = score_configuration(matrix, response, ridge, variant, anchor, noise_sd, kappa, inner_folds, source)
    head = fit_conditional_head(matrix, response, ridge, variant, anchor, noise_sd, kappa, source)
    fold_audit = []
    for train, validation in inner_folds:
        inner_head = fit_conditional_head(
            matrix[train], response[train], ridge, variant, anchor, noise_sd, kappa, source
        )
        fold_audit.append(
            {
                "train_rows": train.tolist(),
                "validation_rows": validation.tolist(),
                "floor": inner_head.floor,
                "training_minimum": float(response[train].min()),
                "clamped_training_deficits": inner_head.clamped_deficits,
            }
        )
    linear = head.intercept + matrix @ head.coefficients
    return {
        "variant": str(variant),
        "source_path": str(source_path.resolve()),
        "source_sha256": source_hash,
        "prefix_epochs": np.asarray(prefix_epochs, float).tolist(),
        "shape": shape,
        "ridge": ridge,
        "kappa": None if variant == Variant.CUMULATIVE_BPB else kappa,
        "flat_profile": flat_profile,
        "anchor": anchor,
        "noise_sd": noise_sd,
        "intercept": head.intercept,
        "coefficients": head.coefficients.tolist(),
        "floor": head.floor,
        "inner_cv_rmse": score,
        "training_rows": len(response),
        "calibration_rows": np.flatnonzero(calibration_mask).tolist(),
        "clamped_training_deficits": head.clamped_deficits,
        "training_log_clip_count": int(np.sum(np.abs(linear) > source.LOG_CLIP)) if head.floor is not None else 0,
        "shape_grid_cv": grid_records,
        "kappa_search_cv": search_records,
        "fold_audit": fold_audit,
    }


def predict_model(fit: dict[str, Any], continuation_epochs: np.ndarray, *, source_path: Path) -> np.ndarray:
    """Predict branch endpoints from one fitted, fixed-checkpoint response model."""
    source, digest = response_source(source_path)
    if digest != fit["source_sha256"]:
        raise ValueError("prediction source differs from the source used for fitting")
    matrix = branch_design(
        np.asarray(fit["prefix_epochs"]), continuation_epochs, fit["shape"], Variant(fit["variant"]), source
    )
    head = ConditionalHead(fit["intercept"], np.asarray(fit["coefficients"]), fit["floor"], 0)
    return head_prediction(matrix, head, source)


def prediction_audit(fit: dict[str, Any], continuation_epochs: np.ndarray, *, source_path: Path) -> dict[str, Any]:
    """Audit numerical guards and JSON replay without accessing any outcomes."""
    source, digest = response_source(source_path)
    if digest != fit["source_sha256"]:
        raise ValueError("audit source differs from the source used for fitting")
    matrix = branch_design(
        np.asarray(fit["prefix_epochs"]), continuation_epochs, fit["shape"], Variant(fit["variant"]), source
    )
    coefficients = np.asarray(fit["coefficients"])
    linear = fit["intercept"] + matrix @ coefficients
    prediction = predict_model(fit, continuation_epochs, source_path=source_path)
    replay = predict_model(json.loads(json.dumps(fit, allow_nan=False)), continuation_epochs, source_path=source_path)
    finite_linear = bool(np.isfinite(linear).all())
    finite_prediction = bool(np.isfinite(prediction).all())
    log_link = fit["floor"] is not None
    return {
        "rows": len(matrix),
        "nonfinite_latent_count": int(np.sum(~np.isfinite(linear))),
        "nonfinite_prediction_count": int(np.sum(~np.isfinite(prediction))),
        "latent_minimum": float(np.min(linear)) if finite_linear and len(linear) else None,
        "latent_maximum": float(np.max(linear)) if finite_linear and len(linear) else None,
        "lower_exponent_guard_count": int(np.sum(linear < -source.LOG_CLIP)) if log_link else None,
        "upper_exponent_guard_count": int(np.sum(linear > source.LOG_CLIP)) if log_link else None,
        "prediction_floor_contact_count": int(np.sum(prediction <= fit["floor"])) if log_link else None,
        "json_replay_max_error": (
            float(np.max(np.abs(prediction - replay))) if finite_prediction and len(prediction) else None
        ),
        "minimum_coefficient": float(coefficients.min()),
        "active_coefficients": int(np.sum(coefficients > source.ACTIVE_TOLERANCE)),
        "numerical_guard": "authoritative_log_clip" if log_link else "none_for_additive_bpb",
    }


def structural_checks(source_path: Path, output: Path) -> None:
    """Check coordinate identities and out-of-sample recovery without branch outcomes."""
    source, digest = response_source(source_path)
    rng = np.random.default_rng(20260907)
    prefix = np.asarray([0.4, 1.1, 2.0])
    train = rng.uniform(0.01, 3.0, (160, 3))
    heldout = rng.uniform(0.01, 3.0, (80, 3))
    shape = {"rate": 0.5, "power": 0.7, "threshold": 2.0}
    coefficients = np.asarray([0.3, 0.1, 0.2, 0.02, 0.01, 0.03])
    reports = []
    for variant in Variant:
        matrix = branch_design(prefix, train, shape, variant, source)
        test_matrix = branch_design(prefix, heldout, shape, variant, source)
        latent_train = 0.1 + matrix @ coefficients
        latent_test = 0.1 + test_matrix @ coefficients
        if variant == Variant.CUMULATIVE_BPB:
            train_y, test_y = latent_train, latent_test
            fit = fit_conditional_head(matrix, train_y, 0.0, variant, 1.0, 0.0, 2.5, source)
        else:
            floor = 0.6
            train_y, test_y = floor + np.exp(latent_train), floor + np.exp(latent_test)
            intercept, fitted_coefficients = source.nonnegative_solve(matrix, np.log(train_y - floor), 0.0)
            fit = ConditionalHead(intercept, fitted_coefficients, floor, 0)
        error = float(np.max(np.abs(head_prediction(test_matrix, fit, source) - test_y)))
        zero = branch_design(prefix, np.zeros((2, 3)), shape, variant, source)
        assert error < 1e-10, (variant, error)
        assert np.max(np.abs(zero)) == 0
        reports.append(
            {
                "variant": str(variant),
                "noiseless_heldout_max_error": error,
                "design_rank": int(np.linalg.matrix_rank(matrix)),
            }
        )
    cumulative = branch_design(np.zeros(3), train, shape, Variant.CUMULATIVE_LOG, source)
    continuation = branch_design(prefix, train, shape, Variant.CONTINUATION_LOG, source)
    assert np.array_equal(cumulative, continuation)
    original = source.design_matrix(prefix[None] + train, shape)
    increment = branch_design(prefix, train, shape, Variant.CUMULATIVE_LOG, source)
    centered_parity = float(np.max(np.abs((original - original.mean(axis=0)) - (increment - increment.mean(axis=0)))))
    assert centered_parity < 1e-14
    write_json_atomic(
        output / "structural_checks.json",
        {
            "source_sha256": digest,
            "zero_prefix_coordinate_identity": True,
            "fixed_prefix_centered_design_parity": centered_parity,
            "synthetic_response_recovery": reports,
            "scope": (
                "Fixed known shape and floor recovery; does not establish nonlinear identification from branch data."
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checks-output", type=Path, required=True)
    args = parser.parse_args()
    structural_checks(args.source, args.checks_output)
    print(json.dumps({"structural_checks": "passed", "output": str(args.checks_output)}))


if __name__ == "__main__":
    main()
