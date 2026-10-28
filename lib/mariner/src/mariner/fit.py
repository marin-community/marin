# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-task fit of the MARINER surrogate on a proxy swarm.

The floor is phi_t = anchor_t - max(nu_t (anchor_t - min_r y_rt), 3 sigma_t), with the proportional mixture's mean
and repeat SD as anchor and sigma. The head is fitted by nonnegative least squares on the log-deficit log(y - phi_t)
with centered, unscaled columns and ridge weight gamma_t. The shape (rho_t, kappa_t, tau_t) and gamma_t come from a
fixed grid by mixture-blocked inner cross-validation of the predictions in BPB at a provisional nu of 2.5; nu_t comes
from a bounded scalar search on log nu in [1, 6] of the same inner-CV error, with a boundary rule: when the search
ends within 10% of the log upper bound, nu_t is set to 1.5 (a conservative default chosen on development data, not a
curvature test). The inner folds are k-means clusters of the runs' square-root weights, with the calibration runs
pinned to training.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize_scalar, nnls
from sklearn.cluster import KMeans

from mariner.model import Anchor, Head, Objective, ObjectiveFit, Shape, Swarm, TaskFit, design_matrix

# ---------------------------------------------------------------------------------------------------------------
# Constants of the frozen procedure (every value is the one the paper's runs used)
# ---------------------------------------------------------------------------------------------------------------

CALIBRATION_LABEL = -1

RHO_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
KAPPA_GRID = (0.3, 0.5, 0.7, 1.0)
TAU_GRID = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
GAMMA_GRID = (0.0, 1e-3, 1e-2, 0.1, 1.0)
SHAPES = tuple(Shape(rho, kappa, tau) for rho in RHO_GRID for kappa in KAPPA_GRID for tau in TAU_GRID)

NU_PROVISIONAL = 2.5  # the multiplier at which the shape and ridge are scored
NU_BOUNDS = (1.0, 6.0)
NU_SEARCH_EVALUATIONS = 24
NU_SEARCH_XATOL = 0.02
FLAT_PROFILE_FRACTION = 0.1
NU_DEFAULT = 1.5
NOISE_MARGIN_SDS = 3.0
INFINITE_CV_PENALTY = 1e3

INNER_FOLDS = 3
FOLD_SEED = 20_260_902
FINAL_FIT_FOLD_SEED = FOLD_SEED + 90_000
KMEANS_RESTARTS = 50

DEFICIT_FLOOR = 1e-9
NNLS_MAXITER_FACTOR = 200


# ---------------------------------------------------------------------------------------------------------------
# Folds: mixture-blocked k-means on sqrt(weights); calibration rows pinned to training
# ---------------------------------------------------------------------------------------------------------------


Folds = tuple[tuple[np.ndarray, np.ndarray], ...]  # (train rows, validation rows) per fold


def block_labels(weights: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """Cluster runs into mixture blocks (k-means on sqrt weights)."""
    labels = KMeans(n_clusters=folds, n_init=KMEANS_RESTARTS, random_state=seed).fit_predict(np.sqrt(weights))
    if len(np.unique(labels)) != folds:
        raise ValueError("mixture-block construction produced an empty fold")
    return labels


def pin_calibration(labels: np.ndarray, positions: np.ndarray) -> np.ndarray:
    pinned = np.array(labels, copy=True)
    pinned[positions] = CALIBRATION_LABEL
    return pinned


def folds_from_labels(rows: np.ndarray, labels: np.ndarray, folds: int) -> Folds:
    return tuple((rows[labels != index], rows[labels == index]) for index in range(folds))


def labels_from_folds(folds: Folds, runs: int) -> np.ndarray:
    """The fold that validates each of the ``runs`` rows; CALIBRATION_LABEL for rows that no fold validates."""
    labels = np.full(runs, CALIBRATION_LABEL)
    for index, (_train, validation) in enumerate(folds):
        labels[validation] = index
    recovered = folds_from_labels(np.arange(runs), labels, len(folds))
    for (train, validation), (expected_train, expected_validation) in zip(folds, recovered, strict=True):
        if not (
            np.array_equal(np.sort(train), expected_train) and np.array_equal(np.sort(validation), expected_validation)
        ):
            raise ValueError("folds must validate each run at most once and train on every run they do not validate")
    return labels


def final_inner_fold_labels(swarm: Swarm) -> np.ndarray:
    """The final fit's inner fold of every run, recomputed from the swarm's weights; calibration runs get -1."""
    labels = pin_calibration(
        block_labels(swarm.weights, INNER_FOLDS, FINAL_FIT_FOLD_SEED), np.flatnonzero(swarm.calibration)
    )
    if any(not np.any(labels == index) for index in range(INNER_FOLDS)):
        raise ValueError("an inner fold validates no run: its mixture block holds only calibration runs")
    return labels


def final_inner_folds(swarm: Swarm) -> Folds:
    """Inner folds of the final fit on every run (``final_inner_fold_labels``)."""
    return folds_from_labels(np.arange(len(swarm.runs)), final_inner_fold_labels(swarm), INNER_FOLDS)


# ---------------------------------------------------------------------------------------------------------------
# Floored log-deficit head
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FloorSpec:
    anchor: float
    noise_sd: float
    nu: float


def floor_value(response: np.ndarray, spec: FloorSpec) -> float:
    gap = spec.anchor - float(np.min(response))
    if gap <= 0.0:
        gap = max(float(np.std(response)), 1e-3 * abs(spec.anchor), 1e-6)
    return spec.anchor - max(spec.nu * gap, NOISE_MARGIN_SDS * spec.noise_sd)


def nonnegative_solve(matrix: np.ndarray, target: np.ndarray, gamma: float) -> tuple[float, np.ndarray]:
    """Centered nonnegative least squares with a free intercept and ridge rows, on the QR-reduced system."""
    width = matrix.shape[1]
    design_mean = matrix.mean(axis=0)
    target_mean = float(target.mean())
    rows = matrix - design_mean[None, :]
    rhs = target - target_mean
    if gamma > 0.0:
        rows = np.vstack([rows, np.diag(np.full(width, math.sqrt(gamma)))])
        rhs = np.concatenate([rhs, np.zeros(width)])
    if rows.shape[0] > 2 * width:
        orthogonal, triangular = np.linalg.qr(rows, mode="reduced")
        rows, rhs = triangular, orthogonal.T @ rhs
    coefficients, _residual = nnls(rows, rhs, maxiter=NNLS_MAXITER_FACTOR * width)
    intercept = target_mean - float(design_mean @ coefficients)
    return intercept, coefficients


def fit_head(matrix: np.ndarray, response: np.ndarray, gamma: float, spec: FloorSpec) -> Head:
    phi = floor_value(response, spec)
    target = np.log(np.maximum(response - phi, DEFICIT_FLOOR))
    intercept, coefficients = nonnegative_solve(matrix, target, gamma)
    return Head(float(intercept), np.asarray(coefficients, dtype=float), phi)


def cv_rmse(matrix: np.ndarray, response: np.ndarray, gamma: float, spec: FloorSpec, folds: Folds) -> float:
    error = 0.0
    count = 0
    for train, validation in folds:
        head = fit_head(matrix[train], response[train], gamma, spec)
        prediction = head.predict(matrix[validation])
        if not np.isfinite(prediction).all():
            return float("inf")
        error += float(np.sum((prediction - response[validation]) ** 2))
        count += len(validation)
    return math.sqrt(error / count)


# ---------------------------------------------------------------------------------------------------------------
# Per-task fit: grid, nu search, final head
# ---------------------------------------------------------------------------------------------------------------


class ShapeAndRidge(NamedTuple):
    """A task's shape and ridge weight from the grids, and their inner-CV RMSE (``select_shape_and_ridge``)."""

    shape: Shape
    gamma: float
    score: float


class FloorMultiplier(NamedTuple):
    """A task's floor multiplier, whether its profile was flat, and the inner-CV RMSE at it (``search_nu``)."""

    nu: float
    flat: bool
    score: float


def select_shape_and_ridge(exposures: np.ndarray, response: np.ndarray, spec: FloorSpec, folds: Folds) -> ShapeAndRidge:
    """Exhaustive inner-CV search over the shape and ridge grids; ties go to the earlier grid entry."""
    best: tuple[float, int, int] | None = None
    for shape_index, shape in enumerate(SHAPES):
        matrix = design_matrix(exposures, shape)
        for gamma_index, gamma in enumerate(GAMMA_GRID):
            candidate = (cv_rmse(matrix, response, gamma, spec, folds), shape_index, gamma_index)
            if best is None or candidate < best:
                best = candidate
    if best is None or not math.isfinite(best[0]):
        raise ValueError("no finite inner-CV candidate")
    score, shape_index, gamma_index = best
    return ShapeAndRidge(SHAPES[shape_index], GAMMA_GRID[gamma_index], score)


def search_nu(matrix: np.ndarray, response: np.ndarray, gamma: float, anchor: Anchor, folds: Folds) -> FloorMultiplier:
    """Bounded scalar search on log nu; flat profiles (argmin near the upper bound) get the default."""
    low, high = NU_BOUNDS

    def objective(log_nu: float) -> float:
        spec = FloorSpec(anchor.proportional, anchor.repeat_sd, math.exp(log_nu))
        value = cv_rmse(matrix, response, gamma, spec, folds)
        return value if math.isfinite(value) else INFINITE_CV_PENALTY

    search = minimize_scalar(
        objective,
        bounds=(math.log(low), math.log(high)),
        method="bounded",
        options={"maxiter": NU_SEARCH_EVALUATIONS, "xatol": NU_SEARCH_XATOL},
    )
    nu, score = float(math.exp(search.x)), float(search.fun)
    flat = math.log(nu) >= (1 - FLAT_PROFILE_FRACTION) * math.log(high)
    if flat:
        nu = NU_DEFAULT
        score = objective(math.log(nu))
    return FloorMultiplier(nu, flat, score)


def fit_task(swarm: Swarm, task: str, anchor: Anchor, folds: Folds) -> TaskFit:
    """Fit one task on every run of the swarm; ``folds`` index the swarm's rows."""
    exposures = swarm.exposures
    response = swarm.outcomes[task].to_numpy(float)
    provisional = FloorSpec(anchor.proportional, anchor.repeat_sd, NU_PROVISIONAL)
    selected = select_shape_and_ridge(exposures, response, provisional, folds)
    matrix = design_matrix(exposures, selected.shape)
    multiplier = search_nu(matrix, response, selected.gamma, anchor, folds)
    head = fit_head(matrix, response, selected.gamma, FloorSpec(anchor.proportional, anchor.repeat_sd, multiplier.nu))
    return TaskFit(task, selected.shape, selected.gamma, multiplier.nu, multiplier.flat, multiplier.score, head)


# ---------------------------------------------------------------------------------------------------------------
# Objective fit: every task
# ---------------------------------------------------------------------------------------------------------------


def fit_objective(
    swarm: Swarm,
    objective: Objective,
    anchors: dict[str, Anchor],
    folds: Folds | None = None,
    log: Callable[[str], None] | None = None,
) -> ObjectiveFit:
    """Fit every task of ``objective`` on every run of ``swarm``.

    ``folds`` index the swarm's rows; ``None``, the procedure, recomputes the final fit's inner folds. The fit
    records the folds it used as each run's label, keyed by observation id.
    """
    if folds is None:
        folds = final_inner_folds(swarm)
    inner_fold_labels = dict(zip(swarm.runs, labels_from_folds(folds, len(swarm.runs)).tolist(), strict=True))
    if len(inner_fold_labels) != len(swarm.runs):
        raise ValueError("the swarm's observation ids must be unique")
    tasks = []
    for task in objective.tasks:
        if task not in anchors:
            raise ValueError(f"no anchor for {task}")
        fit = fit_task(swarm, task, anchors[task], folds)
        if log is not None:
            log(
                f"{task}: shape {fit.shape} gamma {fit.gamma} nu {fit.nu:.3f}"
                f"{' (flat)' if fit.flat_profile else ''} cv {fit.inner_cv_rmse:.5f}"
            )
        tasks.append(fit)
    return ObjectiveFit(
        objective.name, swarm.buckets, swarm.epochs_per_unit_weight, objective.weights, tuple(tasks), inner_fold_labels
    )
