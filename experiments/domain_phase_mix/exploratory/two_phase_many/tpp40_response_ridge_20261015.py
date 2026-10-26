# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "scikit-learn"]
# ///
"""TP048: canonical amplitude ridge added to the frozen TP040 response solver."""

from __future__ import annotations

import importlib.util
import math
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    tpp40_response_profile_trust_region_20261015 as frozen,
)

prior = frozen.prior


def evaluate(
    matrix: np.ndarray, observed: np.ndarray, amplitudes: np.ndarray, floor: float | None, *, ridge: float
) -> dict[str, Any]:
    """Keep exact linear profiling and separately record data and penalty terms."""
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("Ridge must be finite and nonnegative")
    value = prior.evaluate(matrix, observed, amplitudes, floor)
    data_gradient = np.asarray(value["gradient"])
    gradient = data_gradient + 2 * ridge * amplitudes
    projected = np.where((amplitudes == 0) & (gradient > 0), 0, gradient)
    penalty = 0.0 if ridge == 0 else float(ridge * (amplitudes @ amplitudes))
    total = value["scaled_objective"] + penalty
    if not math.isfinite(penalty) or not math.isfinite(total) or not np.isfinite(gradient).all():
        raise ArithmeticError("Nonfinite penalized objective or gradient")
    return value | {
        "ridge": ridge,
        "data_scaled_objective": value["scaled_objective"],
        "penalty": penalty,
        "scaled_objective": total,
        "data_gradient": value["gradient"],
        "data_projected_gradient_inf": value["projected_gradient_inf"],
        "data_first_order_tolerance_met": value["first_order_tolerance_met"],
        "gradient": gradient.tolist(),
        "projected_gradient_inf": float(np.max(abs(projected))),
        "first_order_tolerance_met": bool(
            not value["nonregular_learned_floor"] and np.max(abs(projected)) <= prior.SOLVER_OPTIONS["gtol"]
        ),
    }


def residual_jacobian(
    matrix: np.ndarray, observed: np.ndarray, record: dict[str, Any], floor: float | None
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Append ridge residuals after the unchanged active-face projection."""
    native = record | {"gradient": record["data_gradient"]}
    try:
        residual, jacobian, diagnostics = frozen.residual_jacobian(matrix, observed, native, floor)
    except (ArithmeticError, np.linalg.LinAlgError) as error:
        logits = matrix @ np.asarray(record["amplitudes"])
        residual = (record["floor"] + record["scale"] * np.exp(logits - record["normalization"]) - observed) / (
            prior.NOISE_SCALE * math.sqrt(len(observed))
        )
        jacobian = None
        diagnostics = {
            "jacobian_regular": False,
            "jacobian_limitations": ["nonfinite_jacobian"],
            "jacobian_exception": str(error),
            "zero_scale_closure": record["scale"] == 0,
        }
    weight = math.sqrt(record["ridge"])
    residual = np.concatenate((residual, weight * np.asarray(record["amplitudes"])))
    if jacobian is not None:
        jacobian = np.vstack((jacobian, weight * np.eye(matrix.shape[1])))
        diagnostics["penalized_envelope_gradient_max_discrepancy"] = float(
            np.max(abs(2 * jacobian.T @ residual - np.asarray(record["gradient"])))
        )
    return residual, jacobian, diagnostics


def fit_response(
    matrix: np.ndarray,
    observed: np.ndarray,
    floor: float | None,
    starts: list[dict[str, Any]],
    *,
    ridge: float,
    evaluation_cap: int,
) -> dict[str, Any]:
    """Run an isolated frozen solver instance without modifying its source/globals."""
    spec = importlib.util.spec_from_file_location("_tp048_private_solver", Path(frozen.__file__))
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the frozen response solver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    namespace = SimpleNamespace(**vars(prior))
    namespace.evaluate = partial(evaluate, ridge=ridge)
    module.prior = namespace
    module.residual_jacobian = residual_jacobian
    fitted = module.fit_response(matrix, observed, floor, starts, evaluation_cap=evaluation_cap)
    return fitted | {
        "ridge": ridge,
        "objective": "mean_squared_bpb_divided_by_0.003_squared_plus_ridge_times_amplitude_norm_squared",
        "canonical_basis": "native minus-benefit then physical-harm columns; no scaling",
        "linear_profile_unchanged": True,
    }
