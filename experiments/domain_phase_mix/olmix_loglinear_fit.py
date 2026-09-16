# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light OLMix log-linear surrogate fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

DEFAULT_HUBER_DELTA = 0.02
FIT_START_SEED = 0
FIT_N_STARTS = 48
MAX_LOG_MAGNITUDE = 50.0


@dataclass(frozen=True)
class OlmixLoglinearFit:
    """Fitted OLMix log-linear surrogate in flattened policy-weight space."""

    log_c: float
    coefficients: tuple[float, ...]
    huber_loss: float

    def predict(self, weights: np.ndarray) -> np.ndarray:
        matrix = np.asarray(weights, dtype=float).reshape(len(weights), -1)
        logits = np.clip(
            matrix @ np.asarray(self.coefficients, dtype=float),
            -MAX_LOG_MAGNITUDE,
            MAX_LOG_MAGNITUDE,
        )
        return np.exp(np.clip(self.log_c, -MAX_LOG_MAGNITUDE, MAX_LOG_MAGNITUDE)) + np.exp(logits)


def _huber_sum(residuals: np.ndarray, *, delta: float) -> float:
    abs_residuals = np.abs(residuals)
    quadratic = 0.5 * residuals * residuals
    linear = delta * (abs_residuals - 0.5 * delta)
    return float(np.where(abs_residuals <= delta, quadratic, linear).sum())


def fit_olmix_loglinear_model(
    weights: np.ndarray,
    targets: np.ndarray,
    *,
    delta: float = DEFAULT_HUBER_DELTA,
    seed: int = FIT_START_SEED,
    n_starts: int = FIT_N_STARTS,
) -> OlmixLoglinearFit:
    """Fit the OLMix positive log-linear law with multistart Huber minimization."""
    x = np.asarray(weights, dtype=float).reshape(len(weights), -1)
    y = np.asarray(targets, dtype=float)
    if np.any(y <= 0.0):
        raise ValueError("OLMix log-linear fitting requires positive targets")
    rng = np.random.default_rng(seed)

    def objective(params: np.ndarray) -> float:
        log_c = float(params[0])
        coefficients = params[1:]
        logits = np.clip(x @ coefficients, -MAX_LOG_MAGNITUDE, MAX_LOG_MAGNITUDE)
        predictions = np.exp(log_c) + np.exp(logits)
        return _huber_sum(predictions - y, delta=delta)

    log_c_candidates = np.linspace(np.log(max(np.min(y) * 0.25, 1e-3)), np.log(max(np.median(y), 1e-3)), 6)
    starts: list[np.ndarray] = []
    for log_c in log_c_candidates:
        starts.append(np.concatenate([[log_c], np.zeros(x.shape[1], dtype=float)]))
        for _ in range(max(n_starts // len(log_c_candidates) - 1, 0)):
            starts.append(np.concatenate([[log_c], rng.normal(0.0, 1.0, size=x.shape[1])]))

    best_params = None
    best_loss = float("inf")
    bounds = [(-MAX_LOG_MAGNITUDE, MAX_LOG_MAGNITUDE), *[(None, None)] * x.shape[1]]
    for start in starts:
        result = minimize(objective, start, method="L-BFGS-B", bounds=bounds)
        if not result.success and best_params is not None:
            continue
        if float(result.fun) < best_loss:
            best_loss = float(result.fun)
            best_params = np.asarray(result.x, dtype=float)
    if best_params is None:
        raise RuntimeError("OLMix log-linear fit failed")
    return OlmixLoglinearFit(
        log_c=float(best_params[0]),
        coefficients=tuple(float(value) for value in best_params[1:]),
        huber_loss=best_loss,
    )
