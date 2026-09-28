# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Distance-to-swarm regularization over a product of phase simplices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.optimize import minimize


class LossAndGradient(Protocol):
    """A smooth surrogate loss expressed in full phase-simplex logits."""

    def __call__(self, logits: np.ndarray) -> tuple[float, np.ndarray]: ...


@dataclass(frozen=True)
class SupportGeometry:
    """Frozen Hellinger geometry induced by a fit swarm and content basis."""

    basis: np.ndarray
    phase_fractions: np.ndarray
    train_histograms: np.ndarray
    train_sqrt_features: np.ndarray
    loo_radius_q95: float


@dataclass(frozen=True)
class PathPoint:
    """One optimum on a distance-regularization path."""

    regularization: float
    surrogate_loss: float
    support_distance: float
    normalized_support_distance: float
    objective: float
    nearest_fit_index: int
    optimizer_success: bool
    optimizer_message: str
    weights: np.ndarray


def logits_to_weights(logits: np.ndarray, bucket_count: int) -> np.ndarray:
    """Map unconstrained full logits to two simplex-valued phase policies."""
    values = np.asarray(logits, dtype=float).reshape(2, bucket_count)
    values -= values.max(axis=1, keepdims=True)
    exponential = np.exp(values)
    return exponential / exponential.sum(axis=1, keepdims=True)


def weights_to_logits(weights: np.ndarray) -> np.ndarray:
    """Map positive phase policies to full logits."""
    values = np.asarray(weights, dtype=float)
    assert values.ndim == 2 and values.shape[0] == 2
    return np.log(np.clip(values, 1e-12, None)).reshape(-1)


def _histograms(weights: np.ndarray, basis: np.ndarray) -> np.ndarray:
    histograms = np.asarray(weights, dtype=float) @ np.asarray(basis, dtype=float)
    return np.clip(histograms, 1e-15, None)


def _sqrt_features(histograms: np.ndarray, phase_fractions: np.ndarray) -> np.ndarray:
    weighted = histograms * np.asarray(phase_fractions, dtype=float)[:, None]
    return np.sqrt(np.clip(weighted, 0.0, None)).reshape(-1)


def build_support_geometry(
    train_weights: np.ndarray,
    basis: np.ndarray,
    phase_fractions: np.ndarray,
) -> SupportGeometry:
    """Build a nearest-swarm Hellinger geometry and its empirical radius scale."""
    train_weights = np.asarray(train_weights, dtype=float)
    basis = np.asarray(basis, dtype=float)
    phase_fractions = np.asarray(phase_fractions, dtype=float)
    assert train_weights.ndim == 3 and train_weights.shape[1] == 2
    assert basis.shape[0] == train_weights.shape[2]
    assert phase_fractions.shape == (2,)
    assert np.isclose(phase_fractions.sum(), 1.0)

    train_histograms = np.stack([train_weights[:, phase] @ basis for phase in range(2)], axis=1)
    weighted = train_histograms * phase_fractions[None, :, None]
    train_sqrt_features = np.sqrt(np.clip(weighted, 0.0, None)).reshape(len(train_weights), -1)
    norms = np.sum(train_sqrt_features**2, axis=1)
    assert np.max(np.abs(norms - 1.0)) < 1e-8

    pairwise = np.clip(1.0 - train_sqrt_features @ train_sqrt_features.T, 0.0, 1.0)
    np.fill_diagonal(pairwise, np.inf)
    nearest = pairwise.min(axis=1)
    return SupportGeometry(
        basis=basis,
        phase_fractions=phase_fractions,
        train_histograms=train_histograms,
        train_sqrt_features=train_sqrt_features,
        loo_radius_q95=float(np.quantile(nearest, 0.95)),
    )


def support_distance(weights: np.ndarray, geometry: SupportGeometry) -> tuple[float, int]:
    """Return exact nearest-swarm squared Hellinger distance."""
    histograms = _histograms(weights, geometry.basis)
    feature = _sqrt_features(histograms, geometry.phase_fractions)
    distance = np.clip(1.0 - geometry.train_sqrt_features @ feature, 0.0, 1.0)
    nearest = int(np.argmin(distance))
    return float(distance[nearest]), nearest


def support_distance_batch(weights: np.ndarray, geometry: SupportGeometry, chunk_size: int = 1024) -> np.ndarray:
    """Return nearest-swarm distance for a batch of phase policies."""
    weights = np.asarray(weights, dtype=float)
    out = np.empty(len(weights), dtype=float)
    for start in range(0, len(weights), chunk_size):
        stop = min(start + chunk_size, len(weights))
        histograms = np.stack(
            [weights[start:stop, phase] @ geometry.basis for phase in range(2)],
            axis=1,
        )
        weighted = histograms * geometry.phase_fractions[None, :, None]
        features = np.sqrt(np.clip(weighted, 0.0, None)).reshape(stop - start, -1)
        distance = np.clip(1.0 - features @ geometry.train_sqrt_features.T, 0.0, 1.0)
        out[start:stop] = distance.min(axis=1)
    return out


def support_distance_and_gradient(
    logits: np.ndarray,
    geometry: SupportGeometry,
) -> tuple[float, np.ndarray, int]:
    """Nearest-swarm distance and its piecewise analytic logit gradient."""
    bucket_count = geometry.basis.shape[0]
    weights = logits_to_weights(logits, bucket_count)
    histograms = _histograms(weights, geometry.basis)
    feature = _sqrt_features(histograms, geometry.phase_fractions)
    distances = np.clip(1.0 - geometry.train_sqrt_features @ feature, 0.0, 1.0)
    nearest = int(np.argmin(distances))

    gradient_weights = np.empty_like(weights)
    reference = np.sqrt(np.clip(geometry.train_histograms[nearest], 0.0, None))
    query = np.sqrt(histograms)
    for phase in range(2):
        histogram_gradient = -0.5 * geometry.phase_fractions[phase] * reference[phase] / query[phase]
        gradient_weights[phase] = geometry.basis @ histogram_gradient
    gradient_logits = weights * (gradient_weights - np.sum(gradient_weights * weights, axis=1, keepdims=True))
    return float(distances[nearest]), gradient_logits.reshape(-1), nearest


def optimize_regularization_path(
    loss_and_gradient: LossAndGradient,
    geometry: SupportGeometry,
    regularization_values: tuple[float, ...],
    start_weights: np.ndarray,
    *,
    maxiter: int = 700,
) -> list[PathPoint]:
    """Optimize a warm-started path for loss plus normalized support distance."""
    if geometry.loo_radius_q95 <= 0.0:
        raise ValueError("The support radius must be positive")
    starts = [weights_to_logits(weights) for weights in np.asarray(start_weights, dtype=float)]
    points: list[PathPoint] = []

    for regularization in regularization_values:
        best_result = None
        best_weights = None
        candidate_starts = list(starts)
        candidate_starts.extend(weights_to_logits(point.weights) for point in points[-2:])

        def objective(
            value: np.ndarray,
            regularization: float = regularization,
        ) -> tuple[float, np.ndarray]:
            loss, loss_gradient = loss_and_gradient(value)
            distance, distance_gradient, _ = support_distance_and_gradient(value, geometry)
            scale = regularization / geometry.loo_radius_q95
            return loss + scale * distance, loss_gradient + scale * distance_gradient

        for start in candidate_starts:
            result = minimize(
                objective,
                start,
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
            )
            if best_result is None or float(result.fun) < float(best_result.fun):
                best_result = result
                best_weights = logits_to_weights(result.x, geometry.basis.shape[0])

        assert best_result is not None and best_weights is not None
        loss, _ = loss_and_gradient(np.asarray(best_result.x, dtype=float))
        distance, nearest = support_distance(best_weights, geometry)
        normalized = distance / geometry.loo_radius_q95
        points.append(
            PathPoint(
                regularization=float(regularization),
                surrogate_loss=float(loss),
                support_distance=distance,
                normalized_support_distance=normalized,
                objective=float(loss + regularization * normalized),
                nearest_fit_index=nearest,
                optimizer_success=bool(best_result.success),
                optimizer_message=str(best_result.message),
                weights=best_weights,
            )
        )
        starts = [np.asarray(best_result.x, dtype=float), *candidate_starts[:4]]
    return points
