# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0", "fsspec"]
# ///
"""Historical Hellinger fits and a free-intercept control for fixed checkpoints.

BRW-000 uses the archived tangent feature map and constrained solver exactly.
Its observed tied continuation supplies the only intercept. BRW-001 preserves
the feature map and coefficient penalty, while fitting an unpenalized intercept
on training actions and the declared calibration rows. Neither procedure uses
the held panel's anchor at prediction time.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from fit_two_phase_link_spines_20260907 import load_module
from fixed_checkpoint_wspu_models_20260907 import Folds, validate_partitions
from scipy import optimize


class Baseline(StrEnum):
    HISTORICAL = "BRW-000"
    FREE_INTERCEPT = "BRW-001"


@lru_cache(maxsize=8)
def load_baseline_source(path: str, digest: str) -> ModuleType:
    """Import the archive only after verifying its content identity."""
    source = Path(path)
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError("baseline source does not match its recorded hash")
    return load_module(source, f"checkpoint_hellinger_{digest[:16]}")


def baseline_source(source_path: Path) -> tuple[ModuleType, str]:
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return load_baseline_source(str(source_path.resolve()), digest), digest


def archive_frame(weights: np.ndarray, center: np.ndarray) -> pd.DataFrame:
    """Construct the ordered action columns required by the archived fitter."""
    weights = np.asarray(weights, dtype=float)
    center = np.asarray(center, dtype=float)
    if weights.ndim != 2 or weights.shape[1] != 39 or center.shape != (39,):
        raise ValueError("expected N by 39 weights and one fixed 39-bucket center")
    if not np.isfinite(weights).all() or not np.isfinite(center).all():
        raise ValueError("weights must be finite")
    if np.any(weights < 0.0) or np.any(center < 0.0):
        raise ValueError("weights must be nonnegative")
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("each action must sum to one")
    if not np.isclose(center.sum(), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("the fixed center must sum to one")
    return pd.DataFrame(
        {
            **{f"w::{bucket:02d}": weights[:, bucket] for bucket in range(39)},
            **{f"c::{bucket:02d}": np.full(len(weights), center[bucket]) for bucket in range(39)},
        }
    )


def fit_baseline_head(
    frame: pd.DataFrame,
    target: np.ndarray,
    calibration: np.ndarray,
    alpha: float,
    variant: Baseline,
    source: ModuleType,
) -> dict[str, Any]:
    spec = source.ModelSpec("sqrt_h2", "sqrt_h2", False)
    anchor = float(np.mean(target[calibration]))
    if variant == Baseline.HISTORICAL:
        actions = frame.loc[~calibration].copy()
        actions["effect"] = target[~calibration] - anchor
        fitted = source.fit_model(actions, spec, alpha)
        return {
            "intercept": anchor,
            "feature_scale": fitted.scale.tolist(),
            "coefficients": fitted.coefficients.tolist(),
            "free_intercept": False,
        }
    matrix, lower = source.feature_matrix(frame, spec)
    scale = np.sqrt(np.mean(matrix**2, axis=0))
    scale[scale < 1e-12] = 1.0
    normalized = matrix / scale
    feature_mean = normalized.mean(axis=0)
    target_mean = float(np.mean(target))
    augmented = np.vstack([normalized - feature_mean, np.sqrt(alpha) * np.eye(matrix.shape[1])])
    augmented_target = np.concatenate([target - target_mean, np.zeros(matrix.shape[1])])
    fitted = optimize.lsq_linear(
        augmented,
        augmented_target,
        bounds=(lower * scale, np.full(len(lower), np.inf)),
        lsmr_tol="auto",
    )
    if not fitted.success:
        raise ValueError(fitted.message)
    return {
        "intercept": float(target_mean - feature_mean @ fitted.x),
        "feature_scale": scale.tolist(),
        "coefficients": fitted.x.tolist(),
        "free_intercept": True,
        "solver_status": int(fitted.status),
        "solver_optimality": float(fitted.optimality),
    }


def baseline_head_prediction(fit: dict[str, Any], frame: pd.DataFrame, source: ModuleType) -> np.ndarray:
    spec = source.ModelSpec("sqrt_h2", "sqrt_h2", False)
    matrix, _ = source.feature_matrix(frame, spec)
    return float(fit["intercept"]) + (matrix / np.asarray(fit["feature_scale"])) @ np.asarray(fit["coefficients"])


def fit_baseline(
    weights: np.ndarray,
    center: np.ndarray,
    y: np.ndarray,
    calibration_mask: np.ndarray,
    folds: Folds,
    variant: str,
    source_path: Path,
) -> dict[str, Any]:
    """Select the archived ridge grid using only the supplied training cohort.

    Fold indices address all supplied rows. Calibration rows must be pinned in
    every inner training partition and are never scored. The historical model
    uses them only to calculate its fixed anchor; the free-intercept control
    also includes them as fit observations.
    """
    kind = Baseline(variant)
    frame = archive_frame(weights, center)
    target = np.asarray(y, dtype=float)
    calibration = np.asarray(calibration_mask)
    if target.shape != (len(frame),) or not np.isfinite(target).all():
        raise ValueError("target must be one finite value per training row")
    validate_partitions(len(frame), calibration, folds)
    if np.all(calibration):
        raise ValueError("the training cohort needs continuation actions")
    if not np.allclose(np.asarray(weights)[calibration], center, atol=1e-12, rtol=0.0):
        raise ValueError("calibration rows must be tied continuations of the supplied fixed center")
    source, digest = baseline_source(source_path)
    cv_rows = []
    for alpha in source.RIDGE_ALPHAS:
        prediction = np.full(len(frame), np.nan)
        for train, validation in folds:
            head = fit_baseline_head(frame.iloc[train], target[train], calibration[train], float(alpha), kind, source)
            prediction[validation] = baseline_head_prediction(head, frame.iloc[validation], source)
        rmse = float(np.sqrt(np.mean((prediction[~calibration] - target[~calibration]) ** 2)))
        cv_rows.append({"ridge": float(alpha), "rmse_bpb": rmse})
    selected = min(cv_rows, key=lambda row: (row["rmse_bpb"], row["ridge"]))
    result = fit_baseline_head(frame, target, calibration, selected["ridge"], kind, source)
    result.update(
        {
            "variant": kind.value,
            "ridge": selected["ridge"],
            "cv_rmse_bpb": selected["rmse_bpb"],
            "cv_rows": cv_rows,
            "training_anchor": float(np.mean(target[calibration])),
            "training_actions": int(np.sum(~calibration)),
            "calibration_rows": int(np.sum(calibration)),
            "center": np.asarray(center, dtype=float).tolist(),
            "archive_source_path": str(source_path.resolve()),
            "archive_source_sha256": digest,
            "wrapper_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    return result


def predict_baseline(fit: dict[str, Any], weights: np.ndarray, center: np.ndarray, source_path: Path) -> np.ndarray:
    """Predict endpoints with the training anchor or fitted intercept intact."""
    source, digest = baseline_source(source_path)
    if digest != fit["archive_source_sha256"]:
        raise ValueError("prediction requires the archive source used for fitting")
    if not np.array_equal(np.asarray(center), np.asarray(fit["center"])):
        raise ValueError("prediction must use the fitted checkpoint's continuation center")
    return baseline_head_prediction(fit, archive_frame(weights, center), source)
