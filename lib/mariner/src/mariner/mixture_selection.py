# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Epoch-aware pretraining-mixture selection from a proxy swarm, in one file.

This is the authoritative implementation of the procedure of the one-phase data-mixing paper: fit one
epoch-aware surrogate per evaluation task on a swarm of proxy runs, aggregate the tasks with the evaluation's own
weights, and minimize the aggregate over the target-feasible mixtures. It was extracted from the Marin research
code path and reproduces that code's fits on the calibration-pinned protocol to 1e-8 and its materialized policies
exactly (``tests/test_parity.py``).

Procedure (paper Section 4; symbols in the paper: gamma_t is ``kappa`` here, kappa in the paper is the Weibull
``power``):

1. Materialized epochs: E_i(w) = w_i * (D_tgt / P_i), the number of passes over bucket i at weight w_i.
2. Per task t, a floored log-deficit surrogate on the shared response functions
   f_t(w) = phi_t + exp(c_t - sum_i alpha_ti b(E_i) + sum_i beta_ti h(E_i)), alpha, beta >= 0,
   b(E) = 1 - exp(-(rate E)^power), h(E) = softplus(log(1 + E) - threshold)^2.
3. The floor phi_t = anchor_t - max(kappa_t (anchor_t - min_r y_rt), 3 sigma_t), with the proportional mixture's
   mean and repeat SD as anchor and sigma. The head is fitted by nonnegative least squares on the log-deficit
   log(y - phi_t) with centered, unscaled columns and ridge rows; the shape (rate, power, threshold) and ridge come
   from a fixed grid by mixture-blocked inner cross-validation of the predictions in BPB at a provisional kappa of
   2.5; kappa_t by a bounded scalar search on log kappa in [1, 6] of the same inner-CV error, with a boundary rule:
   when the search ends within 10% of the log upper bound, kappa_t is set to 1.5 (a conservative default chosen
   on development data, not a curvature test).
4. Aggregate sum_t a_t f_t(w), minimized by SLSQP from several feasible starts over the simplex (an epoch cap is
   optional and not part of the procedure: the reported optima are unconstrained), polished, then rounded to the
   sampler grid of 2,048 blocks with an exchange search. SLSQP finds a local minimizer; the multistart is the only
   guard against local optima.

Also included: the Olmix baseline as an independent implementation: the positive log-linear law per task fitted
with a Huber loss by scipy multistart (not upstream's torch fitter), and the exact KL-regularized proposer, which
solves the same convex program as allenai/olmix's exact proposer.

Inputs are CSV files (see ``data/`` and the README): swarm weights, swarm outcomes, objectives (component
weights), anchors, optional fold assignments, bucket pool sizes, and held-out banks for evaluation.

usage:
  python mixture_selection.py fit --objective table9 --out fit_table9.json
  python mixture_selection.py predict --fit fit_table9.json --mixtures mixtures.csv --out predictions.csv
  python mixture_selection.py optimize --fit fit_table9.json --cap 6 --out policy.csv
  python mixture_selection.py evaluate --fit fit_table9.json --bank data/heldout_bank_table9.csv
  python mixture_selection.py olmix --objective table9 --kl 0.005 --cap 4 --out olmix_table9.csv
  python mixture_selection.py self-test
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar, nnls
from scipy.special import xlogy
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------------------------------------------
# Constants of the frozen procedure (every value is the one the paper's runs used)
# ---------------------------------------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

EXPOSURE_REFERENCE_TOKENS = 6_325_183_647_689  # D_tgt: the production budget whose repetition the proxies reproduce
CALIBRATION_MARKER = "baseline_proportional"  # swarm run(s) that anchor the floors; never scored out of fold
CALIBRATION_LABEL = -1

SHAPE_RATES = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
SHAPE_POWERS = (0.3, 0.5, 0.7, 1.0)
SHAPE_THRESHOLDS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
RIDGE_GRID = (0.0, 1e-3, 1e-2, 0.1, 1.0)
SHAPES = tuple(
    {"rate": rate, "power": power, "threshold": threshold}
    for rate in SHAPE_RATES
    for power in SHAPE_POWERS
    for threshold in SHAPE_THRESHOLDS
)

KAPPA_PROVISIONAL = 2.5  # the multiplier at which the shape and ridge are scored
KAPPA_BOUNDS = (1.0, 6.0)
KAPPA_SEARCH_EVALUATIONS = 24
KAPPA_SEARCH_XATOL = 0.02
FLAT_PROFILE_FRACTION = 0.1
FLAT_PROFILE_KAPPA = 1.5
NOISE_MARGIN_SDS = 3.0
INFINITE_CV_PENALTY = 1e3

INNER_FOLDS = 3
OUTER_FOLDS = 5
FOLD_SEED = 20_260_902
HELDOUT_INNER_SEED = FOLD_SEED + 90_000
KMEANS_RESTARTS = 50

LOG_CLIP = 30.0  # numerical overflow guard on the linear predictor; not a modeling constant
DEFICIT_FLOOR = 1e-9
NNLS_MAXITER_FACTOR = 200
ACTIVE_TOLERANCE = 1e-10

START_SEED = 20_260_906
PANEL_STARTS = 4
OPTIMIZER_OPTIONS = {"maxiter": 500, "ftol": 1e-11, "eps": 1e-8}
POLISH_OPTIONS = {"maxiter": 2000, "ftol": 1e-10, "eps": 1e-6}
PROJECTION_OPTIONS = {"maxiter": 200, "ftol": 1e-13}
FEASIBILITY_TOLERANCE = 1e-7
ENDPOINT_FEASIBILITY_TOLERANCE = 10 * FEASIBILITY_TOLERANCE
MIXTURE_BLOCK_SIZE = 2_048
MAX_EXCHANGE_STEPS = 4 * MIXTURE_BLOCK_SIZE
REFINE_TOLERANCE = 1e-13

OLMIX_HUBER_DELTA = 0.01
OLMIX_FIT_STARTS = 48
OLMIX_FIT_SEED = 0
OLMIX_MAX_LOG = 50.0
OLMIX_SOLVERS = ("CLARABEL", "ECOS", "SCS")

PARITY_TOLERANCE = 1e-8


# ---------------------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Swarm:
    """A proxy swarm: runs x buckets weights, the exposure scale per bucket, and per-task outcomes."""

    runs: tuple[str, ...]
    buckets: tuple[str, ...]
    weights: np.ndarray  # (runs, buckets), rows sum to one
    inventory: np.ndarray  # (buckets,) epochs per unit weight = D_tgt / P_i
    outcomes: pd.DataFrame  # (runs, tasks) with task names as columns
    calibration: np.ndarray  # (runs,) bool: rows that are calibration data

    @property
    def exposures(self) -> np.ndarray:
        return self.weights * self.inventory[None, :]


@dataclasses.dataclass(frozen=True)
class Objective:
    name: str
    components: tuple[str, ...]
    weights: np.ndarray  # evaluation weights a_t (byte shares or 1/n)


@dataclasses.dataclass(frozen=True)
class Anchor:
    proportional: float  # the proportional mixture's mean over its repeats
    repeat_sd: float  # its repeat SD (0 disables the noise margin)


def read_swarm(weights_path: Path, outcomes_path: Path, buckets_path: Path) -> Swarm:
    buckets_table = pd.read_csv(buckets_path)
    buckets = tuple(str(b) for b in buckets_table["bucket"])
    if "epochs_per_unit_weight" in buckets_table:
        inventory = buckets_table["epochs_per_unit_weight"].to_numpy(float)
    else:
        inventory = EXPOSURE_REFERENCE_TOKENS / buckets_table["pool_tokens"].to_numpy(float)
    weights_table = pd.read_csv(weights_path)
    outcomes_table = pd.read_csv(outcomes_path)
    if list(weights_table["run"]) != list(outcomes_table["run"]):
        raise ValueError("swarm weights and outcomes must list the same runs in the same order")
    weights = weights_table[list(buckets)].to_numpy(float)
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("swarm weights must sum to one per run")
    runs = tuple(str(r) for r in weights_table["run"])
    if "calibration" in weights_table:
        calibration = weights_table["calibration"].astype(bool).to_numpy()
    else:
        calibration = np.asarray([CALIBRATION_MARKER in r for r in runs])
    outcomes = outcomes_table.drop(columns=["run"]).astype(float)
    return Swarm(runs, buckets, weights, inventory, outcomes, calibration)


def read_objectives(path: Path) -> dict[str, Objective]:
    table = pd.read_csv(path)
    result = {}
    for name, group in table.groupby("objective", sort=False):
        result[str(name)] = Objective(str(name), tuple(group["component"].astype(str)), group["weight"].to_numpy(float))
    return result


def read_anchors(path: Path, objective: str) -> dict[str, Anchor]:
    table = pd.read_csv(path)
    table = table[table["objective"].eq(objective)]
    return {str(r.component): Anchor(float(r.proportional_bpb), float(r.repeat_sd)) for r in table.itertuples()}


# ---------------------------------------------------------------------------------------------------------------
# Folds: mixture-blocked k-means on sqrt(weights); calibration rows pinned to training
# ---------------------------------------------------------------------------------------------------------------

Folds = tuple[tuple[np.ndarray, np.ndarray], ...]  # (train rows, validation rows) per fold


def block_labels(weights: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """Cluster runs into mixture blocks (k-means on sqrt weights); needs scikit-learn, pinned in pyproject."""
    from sklearn.cluster import KMeans  # noqa: PLC0415  (optional dependency: only when folds are not supplied)

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


def final_inner_folds(swarm: Swarm, splits: pd.DataFrame | None) -> Folds:
    """Inner folds of the final fit on every run: from a splits table (fold -1 rows) or recomputed."""
    rows = np.arange(len(swarm.runs))
    if splits is not None:
        table = splits[splits["fold"].eq(-1) & splits["repeat"].eq(0)].sort_values("row")
        if list(table["row"]) != list(rows):
            raise ValueError("splits table must list every run once for fold -1")
        labels = table["inner_fold"].to_numpy(int)
        if not set(np.unique(labels)) <= set(range(INNER_FOLDS)) | {CALIBRATION_LABEL}:
            raise ValueError("inner fold labels must be 0..INNER_FOLDS-1 or the calibration label")
        if np.any(labels[swarm.calibration] != CALIBRATION_LABEL):
            raise ValueError("calibration rows must carry the calibration label (never validated)")
        if np.any(labels[~swarm.calibration] == CALIBRATION_LABEL):
            raise ValueError("only calibration rows may carry the calibration label")
        if any(np.sum(labels == index) == 0 for index in range(INNER_FOLDS)):
            raise ValueError("every inner fold must validate at least one run")
    else:
        labels = pin_calibration(
            block_labels(swarm.weights, INNER_FOLDS, HELDOUT_INNER_SEED), np.flatnonzero(swarm.calibration)
        )
    return folds_from_labels(rows, labels, INNER_FOLDS)


# ---------------------------------------------------------------------------------------------------------------
# Response functions and design
# ---------------------------------------------------------------------------------------------------------------


def benefit(exposure: np.ndarray, rate: float | np.ndarray, power: float | np.ndarray) -> np.ndarray:
    return -np.expm1(-((np.maximum(rate * exposure, 0.0)) ** power))


def harm(exposure: np.ndarray, threshold: float | np.ndarray) -> np.ndarray:
    return np.logaddexp(np.log1p(np.maximum(exposure, 0.0)) - threshold, 0.0) ** 2


def design_matrix(exposures: np.ndarray, shape: dict[str, float]) -> np.ndarray:
    """Columns: minus the benefit of every bucket, then the harm of every bucket (ridge multiplier one each)."""
    return np.hstack(
        [-benefit(exposures, float(shape["rate"]), float(shape["power"])), harm(exposures, float(shape["threshold"]))]
    )


# ---------------------------------------------------------------------------------------------------------------
# Floored log-deficit head
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FloorSpec:
    anchor: float  # NaN: the training median
    noise_sd: float
    kappa: float
    noise_margin_sds: float = NOISE_MARGIN_SDS


def floor_value(response: np.ndarray, spec: FloorSpec) -> float:
    anchor = float(np.median(response)) if math.isnan(spec.anchor) else spec.anchor
    gap = anchor - float(np.min(response))
    if gap <= 0.0:
        gap = max(float(np.std(response)), 1e-3 * abs(anchor), 1e-6)
    return anchor - max(spec.kappa * gap, spec.noise_margin_sds * spec.noise_sd)


def nonnegative_solve(matrix: np.ndarray, target: np.ndarray, ridge: float) -> tuple[float, np.ndarray]:
    """Centered nonnegative least squares with a free intercept and ridge rows, on the QR-reduced system."""
    width = matrix.shape[1]
    design_mean = matrix.mean(axis=0)
    target_mean = float(target.mean())
    rows = matrix - design_mean[None, :]
    rhs = target - target_mean
    if ridge > 0.0:
        rows = np.vstack([rows, np.diag(np.full(width, math.sqrt(ridge)))])
        rhs = np.concatenate([rhs, np.zeros(width)])
    if rows.shape[0] > 2 * width:
        orthogonal, triangular = np.linalg.qr(rows, mode="reduced")
        rows, rhs = triangular, orthogonal.T @ rhs
    coefficients, _residual = nnls(rows, rhs, maxiter=NNLS_MAXITER_FACTOR * width)
    intercept = target_mean - float(design_mean @ coefficients)
    return intercept, coefficients


@dataclasses.dataclass(frozen=True)
class Head:
    intercept: float
    coefficients: np.ndarray
    floor: float

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        linear = self.intercept + matrix @ self.coefficients
        return self.floor + np.exp(np.clip(linear, -LOG_CLIP, LOG_CLIP))


def fit_head(matrix: np.ndarray, response: np.ndarray, ridge: float, spec: FloorSpec) -> Head:
    floor = floor_value(response, spec)
    target = np.log(np.maximum(response - floor, DEFICIT_FLOOR))
    intercept, coefficients = nonnegative_solve(matrix, target, ridge)
    return Head(float(intercept), np.asarray(coefficients, dtype=float), floor)


def cv_rmse(matrix: np.ndarray, response: np.ndarray, ridge: float, spec: FloorSpec, folds: Folds) -> float:
    error = 0.0
    count = 0
    for train, validation in folds:
        head = fit_head(matrix[train], response[train], ridge, spec)
        prediction = head.predict(matrix[validation])
        if not np.isfinite(prediction).all():
            return float("inf")
        error += float(np.sum((prediction - response[validation]) ** 2))
        count += len(validation)
    return math.sqrt(error / count)


# ---------------------------------------------------------------------------------------------------------------
# Per-task fit: grid, kappa search, final head
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TaskFit:
    component: str
    shape: dict[str, float]
    ridge: float
    kappa: float
    flat_profile: bool
    inner_cv_rmse: float
    head: Head

    def to_json(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "shape": self.shape,
            "ridge": self.ridge,
            "kappa": self.kappa,
            "flat_profile": self.flat_profile,
            "inner_cv_rmse": self.inner_cv_rmse,
            "intercept": self.head.intercept,
            "coefficients": self.head.coefficients.tolist(),
            "floor": self.head.floor,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> TaskFit:
        head = Head(float(payload["intercept"]), np.asarray(payload["coefficients"], float), float(payload["floor"]))
        return cls(
            str(payload["component"]),
            {k: float(v) for k, v in payload["shape"].items()},
            float(payload["ridge"]),
            float(payload["kappa"]),
            bool(payload["flat_profile"]),
            float(payload["inner_cv_rmse"]),
            head,
        )


def select_shape_and_ridge(
    exposures: np.ndarray, response: np.ndarray, spec: FloorSpec, folds: Folds
) -> tuple[dict[str, float], float, float]:
    """Exhaustive inner-CV search over the shape and ridge grids; ties go to the earlier grid entry."""
    best: tuple[float, int, int] | None = None
    for shape_index, shape in enumerate(SHAPES):
        matrix = design_matrix(exposures, shape)
        for ridge_index, ridge in enumerate(RIDGE_GRID):
            candidate = (cv_rmse(matrix, response, ridge, spec, folds), shape_index, ridge_index)
            if best is None or candidate < best:
                best = candidate
    if best is None or not math.isfinite(best[0]):
        raise ValueError("no finite inner-CV candidate")
    score, shape_index, ridge_index = best
    return dict(SHAPES[shape_index]), RIDGE_GRID[ridge_index], score


def search_kappa(
    matrix: np.ndarray, response: np.ndarray, ridge: float, anchor: Anchor, folds: Folds
) -> tuple[float, bool, float]:
    """Bounded scalar search on log kappa; flat profiles (argmin near the upper bound) get the default."""
    low, high = KAPPA_BOUNDS

    def objective(log_kappa: float) -> float:
        spec = FloorSpec(anchor.proportional, anchor.repeat_sd, math.exp(log_kappa))
        value = cv_rmse(matrix, response, ridge, spec, folds)
        return value if math.isfinite(value) else INFINITE_CV_PENALTY

    search = minimize_scalar(
        objective,
        bounds=(math.log(low), math.log(high)),
        method="bounded",
        options={"maxiter": KAPPA_SEARCH_EVALUATIONS, "xatol": KAPPA_SEARCH_XATOL},
    )
    kappa, score = float(math.exp(search.x)), float(search.fun)
    flat = math.log(kappa) >= (1 - FLAT_PROFILE_FRACTION) * math.log(high)
    if flat:
        kappa = FLAT_PROFILE_KAPPA
        score = objective(math.log(kappa))
    return kappa, flat, score


def fit_task(swarm: Swarm, component: str, anchor: Anchor, folds: Folds) -> TaskFit:
    """Fit one task on every run of the swarm; ``folds`` index the swarm's rows."""
    exposures = swarm.exposures
    response = swarm.outcomes[component].to_numpy(float)
    provisional = FloorSpec(anchor.proportional, anchor.repeat_sd, KAPPA_PROVISIONAL)
    shape, ridge, _ = select_shape_and_ridge(exposures, response, provisional, folds)
    matrix = design_matrix(exposures, shape)
    kappa, flat, score = search_kappa(matrix, response, ridge, anchor, folds)
    head = fit_head(matrix, response, ridge, FloorSpec(anchor.proportional, anchor.repeat_sd, kappa))
    return TaskFit(component, shape, ridge, kappa, flat, score, head)


# ---------------------------------------------------------------------------------------------------------------
# Objective fit: every task, vectorized prediction
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ObjectiveFit:
    objective: str
    buckets: tuple[str, ...]
    inventory: np.ndarray
    weights: np.ndarray  # evaluation weights per task
    tasks: tuple[TaskFit, ...]

    def _arrays(self) -> tuple[np.ndarray, ...]:
        rate = np.asarray([t.shape["rate"] for t in self.tasks])
        power = np.asarray([t.shape["power"] for t in self.tasks])
        threshold = np.asarray([t.shape["threshold"] for t in self.tasks])
        intercept = np.asarray([t.head.intercept for t in self.tasks])
        coefficients = np.stack([t.head.coefficients for t in self.tasks])
        floor = np.asarray([t.head.floor for t in self.tasks])
        return rate, power, threshold, intercept, coefficients, floor

    def predict_tasks(self, mixtures: np.ndarray) -> np.ndarray:
        """Per-task predictions for every mixture row, shape (mixtures, tasks)."""
        rate, power, threshold, intercept, coefficients, floor = self._arrays()
        exposure = np.atleast_2d(mixtures)[:, None, :] * self.inventory[None, None, :]
        columns = np.concatenate(
            [-benefit(exposure, rate[None, :, None], power[None, :, None]), harm(exposure, threshold[None, :, None])],
            axis=2,
        )
        linear = intercept[None, :] + np.einsum("nck,ck->nc", columns, coefficients)
        return floor[None, :] + np.exp(np.clip(linear, -LOG_CLIP, LOG_CLIP))

    def predict(self, mixtures: np.ndarray) -> np.ndarray:
        return self.predict_tasks(mixtures) @ self.weights

    def to_json(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "buckets": list(self.buckets),
            "epochs_per_unit_weight": self.inventory.tolist(),
            "task_weights": self.weights.tolist(),
            "tasks": [t.to_json() for t in self.tasks],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ObjectiveFit:
        return cls(
            str(payload["objective"]),
            tuple(payload["buckets"]),
            np.asarray(payload["epochs_per_unit_weight"], float),
            np.asarray(payload["task_weights"], float),
            tuple(TaskFit.from_json(t) for t in payload["tasks"]),
        )


def fit_objective(
    swarm: Swarm,
    objective: Objective,
    anchors: dict[str, Anchor],
    folds: Folds,
    log: Callable[[str], None] | None = None,
) -> ObjectiveFit:
    tasks = []
    for component in objective.components:
        if component not in anchors:
            raise ValueError(f"no anchor for {component}")
        fit = fit_task(swarm, component, anchors[component], folds)
        if log is not None:
            log(
                f"{component}: shape {fit.shape} ridge {fit.ridge} kappa {fit.kappa:.3f}"
                f"{' (flat)' if fit.flat_profile else ''} cv {fit.inner_cv_rmse:.5f}"
            )
        tasks.append(fit)
    return ObjectiveFit(objective.name, swarm.buckets, swarm.inventory, objective.weights, tuple(tasks))


# ---------------------------------------------------------------------------------------------------------------
# Mixture optimization: SLSQP from feasible starts, then rounding to the sampler grid
# ---------------------------------------------------------------------------------------------------------------


def project_start(weights: np.ndarray, upper: np.ndarray, feasible: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the capped simplex by a convex quadratic solve."""
    if np.max(weights - upper) <= 1e-12 and weights.min() >= 0 and abs(weights.sum() - 1) <= 1e-12:
        return weights.copy()
    result = minimize(
        lambda candidate: 0.5 * float(np.square(candidate - weights).sum()),
        feasible,
        jac=lambda candidate: candidate - weights,
        bounds=list(zip(np.zeros(len(upper)), upper, strict=True)),
        constraints={"type": "eq", "fun": lambda c: c.sum() - 1, "jac": lambda c: np.ones_like(c)},
        method="SLSQP",
        options=PROJECTION_OPTIONS,
    )
    if not result.success or max(abs(result.x.sum() - 1), -result.x.min(), np.max(result.x - upper)) > 1e-10:
        raise ValueError(f"start projection failed: {result.message}")
    return result.x


def categorical_kl(weights: np.ndarray, natural: np.ndarray) -> float:
    return float(xlogy(weights, weights / natural).sum())


def optimize_start(
    predict: Callable[[np.ndarray], np.ndarray],
    start: np.ndarray,
    natural: np.ndarray,
    upper: np.ndarray,
    kl: float,
    options: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    def objective(weights: np.ndarray) -> float:
        return float(predict(weights[None])[0]) + kl * categorical_kl(weights, natural)

    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=list(zip(np.zeros(len(upper)), upper, strict=True)),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)},
        options=options,
    )
    violation = float(max(abs(result.x.sum() - 1), -result.x.min(), np.max(result.x - upper)))
    weights = project_start(result.x, upper, natural)
    value = objective(weights)
    if not np.isfinite(value):
        raise ValueError("continuous optimizer produced a nonfinite objective")
    return weights, {
        "objective": value,
        "raw_feasibility_violation": violation,
        "success": bool(result.success),
        "iterations": int(result.nit),
    }


def bucket_upper_bounds(fit: ObjectiveFit, cap: float | None) -> np.ndarray:
    """Per-bucket weight bounds: the simplex alone (``cap`` None, the procedure) or an optional epoch cap."""
    if cap is None:
        return np.ones(len(fit.inventory))
    return np.minimum(1.0, cap / fit.inventory)


def continuous_optimum(
    fit: ObjectiveFit, swarm_weights: np.ndarray, cap: float | None = None, kl: float = 0.0
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """The lowest feasible SLSQP endpoint over the proportional start and four seeded swarm rows, polished."""
    natural = 1.0 / fit.inventory
    natural = natural / natural.sum()
    upper = bucket_upper_bounds(fit, cap)
    indices = np.random.default_rng(START_SEED).choice(len(swarm_weights), PANEL_STARTS, replace=False)
    starts = [project_start(row, upper, natural) for row in (natural, *swarm_weights[indices])]
    records, solutions = [], []
    for index, start in enumerate(starts):
        weights, diagnostics = optimize_start(fit.predict, start, natural, upper, kl, OPTIMIZER_OPTIONS)
        records.append({"start": index, **diagnostics})
        solutions.append(weights)
    for index, start in enumerate(list(solutions)):
        weights, diagnostics = optimize_start(fit.predict, start, natural, upper, kl, POLISH_OPTIONS)
        if diagnostics["objective"] <= records[index]["objective"]:
            solutions[index] = weights
            records[index] = {"start": index, **diagnostics, "polished": True}
    feasible = [i for i, row in enumerate(records) if row["raw_feasibility_violation"] <= ENDPOINT_FEASIBILITY_TOLERANCE]
    if not feasible:
        raise ValueError("no restart ended feasibly")
    best = min(feasible, key=lambda i: records[i]["objective"])
    for row in records:
        row["selected"] = row["start"] == best
    return solutions[best], records


def constrained_counts(weights: np.ndarray, maximum_counts: np.ndarray) -> np.ndarray:
    target = np.asarray(weights, dtype=float) * MIXTURE_BLOCK_SIZE
    counts = np.minimum(np.floor(target).astype(np.int64), maximum_counts)
    remaining = MIXTURE_BLOCK_SIZE - int(counts.sum())
    while remaining:
        available = np.flatnonzero(counts < maximum_counts)
        if len(available) == 0:
            raise ValueError("the epoch cap cannot support a complete mixture block")
        deficit = target[available] - counts[available]
        counts[int(available[int(np.argmax(deficit))])] += 1
        remaining -= 1
    return counts


def refine_counts(
    predict: Callable[[np.ndarray], np.ndarray], initial: np.ndarray, maximum: np.ndarray
) -> tuple[np.ndarray, int]:
    """Improving one-block exchanges on the sampler grid until none lowers the objective."""
    current = np.asarray(initial, dtype=np.int64).copy()
    current_value = float(predict((current / MIXTURE_BLOCK_SIZE)[None, :])[0])
    for step in range(MAX_EXCHANGE_STEPS):
        donors = np.flatnonzero(current > 0)
        receivers = np.flatnonzero(current < maximum)
        proposals, moves = [], []
        for donor in donors:
            for receiver in receivers:
                if donor == receiver:
                    continue
                proposal = current.copy()
                proposal[donor] -= 1
                proposal[receiver] += 1
                proposals.append(proposal)
                moves.append((donor, receiver))
        if not proposals:
            return current, step
        values = predict(np.asarray(proposals, dtype=float) / MIXTURE_BLOCK_SIZE)
        choice = int(np.argmin(values))
        if float(values[choice]) >= current_value - REFINE_TOLERANCE:
            return current, step
        donor, receiver = moves[choice]
        current[donor] -= 1
        current[receiver] += 1
        current_value = float(values[choice])
    raise RuntimeError(f"exchange refinement exceeded {MAX_EXCHANGE_STEPS} steps")


def runtime_policy(
    fit: ObjectiveFit, weights: np.ndarray, cap: float | None = None, kl: float = 0.0
) -> tuple[np.ndarray, dict[str, float]]:
    """Round a continuous optimum to block counts (under the optional cap) and refine; returns counts and predictions."""
    maximum = np.floor(bucket_upper_bounds(fit, cap) * MIXTURE_BLOCK_SIZE + 1e-12).astype(np.int64)
    natural = 1.0 / fit.inventory
    natural = natural / natural.sum()

    def predict(rows: np.ndarray) -> np.ndarray:
        rows = np.atleast_2d(rows)
        value = fit.predict(rows)
        if kl:
            safe = np.maximum(rows, 1e-300)
            value = value + kl * np.sum(np.where(rows > 0, rows * np.log(safe / natural[None, :]), 0.0), axis=1)
        return value

    initial = constrained_counts(weights, maximum)
    counts, steps = refine_counts(predict, initial, maximum)
    runtime = counts / MIXTURE_BLOCK_SIZE
    if int(counts.sum()) != MIXTURE_BLOCK_SIZE or counts.min() < 0 or np.any(counts > maximum):
        raise ValueError("invalid refined runtime allocation")
    return counts, {
        "continuous_prediction": float(predict(weights)[0]),
        "runtime_prediction": float(predict(runtime)[0]),
        "exchange_steps": float(steps),
        "max_materialized_epoch": float((runtime * fit.inventory).max()),
    }


# ---------------------------------------------------------------------------------------------------------------
# Olmix baseline: positive log-linear law per task, exact KL-regularized proposer
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LogLinearLaw:
    log_c: float
    coefficients: np.ndarray
    huber_loss: float

    def predict(self, weights: np.ndarray) -> np.ndarray:
        logits = np.clip(np.atleast_2d(weights) @ self.coefficients, -OLMIX_MAX_LOG, OLMIX_MAX_LOG)
        return np.exp(np.clip(self.log_c, -OLMIX_MAX_LOG, OLMIX_MAX_LOG)) + np.exp(logits)


def huber_sum(residuals: np.ndarray, delta: float) -> float:
    magnitude = np.abs(residuals)
    quadratic = 0.5 * residuals * residuals
    linear = delta * (magnitude - 0.5 * delta)
    return float(np.where(magnitude <= delta, quadratic, linear).sum())


def fit_log_linear(
    weights: np.ndarray,
    targets: np.ndarray,
    delta: float = OLMIX_HUBER_DELTA,
    seed: int = OLMIX_FIT_SEED,
    starts: int = OLMIX_FIT_STARTS,
) -> LogLinearLaw:
    x = np.asarray(weights, dtype=float)
    y = np.asarray(targets, dtype=float)
    if np.any(y <= 0.0):
        raise ValueError("the log-linear law needs positive targets")
    rng = np.random.default_rng(seed)

    def objective(params: np.ndarray) -> float:
        logits = np.clip(x @ params[1:], -OLMIX_MAX_LOG, OLMIX_MAX_LOG)
        return huber_sum(np.exp(params[0]) + np.exp(logits) - y, delta)

    log_c_candidates = np.linspace(np.log(max(np.min(y) * 0.25, 1e-3)), np.log(max(np.median(y), 1e-3)), 6)
    initial: list[np.ndarray] = []
    for log_c in log_c_candidates:
        initial.append(np.concatenate([[log_c], np.zeros(x.shape[1])]))
        for _ in range(max(starts // len(log_c_candidates) - 1, 0)):
            initial.append(np.concatenate([[log_c], rng.normal(0.0, 1.0, size=x.shape[1])]))
    best_params, best_loss = None, float("inf")
    bounds = [(-OLMIX_MAX_LOG, OLMIX_MAX_LOG), *[(None, None)] * x.shape[1]]
    for start in initial:
        result = minimize(objective, start, method="L-BFGS-B", bounds=bounds)
        if not result.success and best_params is not None:
            continue
        if float(result.fun) < best_loss:
            best_loss, best_params = float(result.fun), np.asarray(result.x, dtype=float)
    if best_params is None:
        raise RuntimeError("log-linear fit failed")
    return LogLinearLaw(float(best_params[0]), best_params[1:], best_loss)


def olmix_exact_proposal(
    laws: Sequence[LogLinearLaw], task_weights: np.ndarray, natural: np.ndarray, caps: np.ndarray, kl_reg: float
) -> tuple[np.ndarray, float, str]:
    """Olmix's exact proposer: minimize the weighted mean prediction plus kl_reg * KL(w || natural) on the
    capped simplex."""
    import cvxpy as cp  # noqa: PLC0415  (optional dependency: only for the Olmix baseline)

    weights = cp.Variable(len(natural))
    predicted = sum(
        float(task_weights[i]) * (float(np.exp(law.log_c)) + cp.exp(cp.sum(cp.multiply(law.coefficients, weights))))
        for i, law in enumerate(laws)
    )
    problem = cp.Problem(
        cp.Minimize(predicted + kl_reg * cp.sum(cp.rel_entr(weights, natural))),
        [weights >= 0, cp.sum(weights) == 1, weights <= caps],
    )
    for solver in OLMIX_SOLVERS:
        try:
            problem.solve(solver=solver, warm_start=True, verbose=False)
        except cp.error.SolverError:
            continue
        if problem.status in ("optimal", "optimal_inaccurate"):
            solution = np.clip(np.asarray(weights.value, float), 0.0, None)
            return solution / solution.sum(), float(predicted.value), f"{solver}:{problem.status}"
    raise RuntimeError("no solver converged")


# ---------------------------------------------------------------------------------------------------------------
# Evaluation on a held-out bank
# ---------------------------------------------------------------------------------------------------------------


def bank_metrics(predicted: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    pick = int(np.argmin(predicted))
    return {
        "rows": float(len(measured)),
        "regret_at_1": float(measured[pick] - measured.min()),
        "selected_rank": float(np.sum(measured <= measured[pick])),
        "optimism_at_pick": float(measured[pick] - predicted[pick]),
        "rmse": float(np.sqrt(np.mean((predicted - measured) ** 2))),
        "spearman": float(spearmanr(predicted, measured).correlation),
    }


# ---------------------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------------------


def _default_inputs(args: argparse.Namespace) -> tuple[Swarm, dict[str, Objective], Path, pd.DataFrame | None, Path]:
    swarm = read_swarm(args.swarm_weights, args.swarm_outcomes, args.buckets)
    objectives = read_objectives(args.objectives)
    splits = pd.read_csv(args.splits) if args.splits is not None and Path(args.splits).exists() else None
    return swarm, objectives, args.anchors, splits, args.buckets


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--swarm-weights", type=Path, default=DATA / "swarm_weights.csv")
    parser.add_argument("--swarm-outcomes", type=Path, default=DATA / "swarm_outcomes.csv")
    parser.add_argument("--buckets", type=Path, default=DATA / "buckets.csv")
    parser.add_argument("--objectives", type=Path, default=DATA / "objectives.csv")
    parser.add_argument("--anchors", type=Path, default=DATA / "anchors.csv")
    parser.add_argument(
        "--splits", type=Path, default=DATA / "splits.csv", help="fold table; recomputed with k-means if absent"
    )


def command_fit(args: argparse.Namespace) -> None:
    swarm, objectives, anchors_path, splits, _ = _default_inputs(args)
    objective = objectives[args.objective]
    fit = fit_objective(
        swarm, objective, read_anchors(anchors_path, args.objective), final_inner_folds(swarm, splits), print
    )
    Path(args.out).write_text(json.dumps(fit.to_json(), indent=1))
    print(f"wrote {args.out}: {len(fit.tasks)} tasks")


def command_predict(args: argparse.Namespace) -> None:
    fit = ObjectiveFit.from_json(json.loads(Path(args.fit).read_text()))
    table = pd.read_csv(args.mixtures)
    mixtures = table[list(fit.buckets)].to_numpy(float)
    out = table[[c for c in table.columns if c not in fit.buckets]].copy()
    out[f"predicted_{fit.objective}"] = fit.predict(mixtures)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))


def command_optimize(args: argparse.Namespace) -> None:
    swarm = read_swarm(args.swarm_weights, args.swarm_outcomes, args.buckets)
    fit = ObjectiveFit.from_json(json.loads(Path(args.fit).read_text()))
    weights, records = continuous_optimum(fit, swarm.weights, args.cap, args.kl)
    counts, summary = runtime_policy(fit, weights, args.cap, args.kl)
    table = pd.DataFrame(
        {
            "bucket": fit.buckets,
            "continuous_weight": weights,
            "runtime_count": counts,
            "runtime_weight": counts / MIXTURE_BLOCK_SIZE,
            "materialized_epochs": counts / MIXTURE_BLOCK_SIZE * fit.inventory,
        }
    )
    table.to_csv(args.out, index=False)
    diagnostics = {
        "fit": str(args.fit),
        "cap": args.cap,
        "kl": args.kl,
        "block_size": MIXTURE_BLOCK_SIZE,
        "start_seed": START_SEED,
        "panel_starts": PANEL_STARTS,
        **summary,
        "starts": records,
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(diagnostics, indent=1, default=float))
    print(json.dumps(diagnostics, indent=1, default=float))


def command_evaluate(args: argparse.Namespace) -> None:
    fit = ObjectiveFit.from_json(json.loads(Path(args.fit).read_text()))
    bank = pd.read_csv(args.bank)
    predicted = fit.predict(bank[list(fit.buckets)].to_numpy(float))
    print(json.dumps(bank_metrics(predicted, bank["measured_mean_bpb"].to_numpy(float)), indent=1))


def command_olmix(args: argparse.Namespace) -> None:
    swarm, objectives, _, _, _ = _default_inputs(args)
    objective = objectives[args.objective]
    laws = [fit_log_linear(swarm.weights, swarm.outcomes[c].to_numpy(float), args.delta) for c in objective.components]
    natural = 1.0 / swarm.inventory
    natural = natural / natural.sum()
    caps = np.minimum(1.0, args.cap / swarm.inventory)
    solution, predicted, status = olmix_exact_proposal(laws, objective.weights, natural, caps, args.kl)
    pd.DataFrame(
        {"bucket": swarm.buckets, "weight": solution, "materialized_epochs": solution * swarm.inventory}
    ).to_csv(args.out, index=False)
    laws_json = [
        {"component": c, "log_c": law.log_c, "coefficients": law.coefficients.tolist(), "huber_loss": law.huber_loss}
        for c, law in zip(objective.components, laws, strict=True)
    ]
    record = {
        "objective": args.objective,
        "kl_reg": args.kl,
        "cap": args.cap,
        "huber_delta": args.delta,
        "starts": OLMIX_FIT_STARTS,
        "seed": OLMIX_FIT_SEED,
        "solver": status,
        "predicted": predicted,
        "max_epochs": float((solution * swarm.inventory).max()),
        "laws": laws_json,
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k != "laws"}, indent=1))


def self_test() -> None:
    """Refit both objectives on the bundled swarm and check parity with the bundled reference fits and policies."""
    swarm = read_swarm(DATA / "swarm_weights.csv", DATA / "swarm_outcomes.csv", DATA / "buckets.csv")
    objectives = read_objectives(DATA / "objectives.csv")
    splits = pd.read_csv(DATA / "splits.csv")
    folds = final_inner_folds(swarm, splits)
    policies = pd.read_csv(DATA / "reference_policies.csv")
    # The unconstrained optimum is the procedure's proposal; the capped cases are the sensitivity policies.
    for name, cases in (
        ("uncheatable", ((None, "lwspu_u_snc_cap06"), (6, "lwspu_u_snc_cap06"))),
        ("table9", ((None, "lwspu_t9_snc_cap08"), (6, "lwspu_t9_snc_cap06"), (8, "lwspu_t9_snc_cap08"))),
    ):
        fit = fit_objective(swarm, objectives[name], read_anchors(DATA / "anchors.csv", name), folds)
        reference = pd.read_csv(DATA / f"reference_swarm_predictions_{name}.csv")
        error = float(
            np.max(
                np.abs(fit.predict_tasks(swarm.weights) - reference[list(objectives[name].components)].to_numpy(float))
            )
        )
        bank = pd.read_csv(DATA / f"heldout_bank_{name}.csv")
        reference_bank = pd.read_csv(DATA / f"reference_bank_predictions_{name}.csv")
        bank_error = float(
            np.max(
                np.abs(
                    fit.predict_tasks(bank[list(swarm.buckets)].to_numpy(float))
                    - reference_bank[list(objectives[name].components)].to_numpy(float)
                )
            )
        )
        print(f"{name}: swarm prediction parity {error:.1e}, bank {bank_error:.1e}")
        if max(error, bank_error) > PARITY_TOLERANCE:
            raise AssertionError(f"{name}: prediction parity failed")
        for cap, candidate in cases:
            weights, _ = continuous_optimum(fit, swarm.weights, cap)
            counts, summary = runtime_policy(fit, weights, cap)
            expected = (
                policies[policies.candidate_id.eq(candidate)]
                .set_index("domain")
                .runtime_count.reindex(swarm.buckets)
                .fillna(0)
                .to_numpy(int)
            )
            same = bool(np.array_equal(counts, expected))
            label = "unconstrained" if cap is None else f"cap {cap}"
            print(
                f"{name} {label}: predicted {summary['runtime_prediction']:.4f}, "
                f"realized policy identical to reference: {same}"
            )
            if not same:
                raise AssertionError(f"{name} {label}: realized policy differs from the reference")
    print("self-test passed")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="fit one objective's surrogate on a swarm")
    _add_data_arguments(fit)
    fit.add_argument("--objective", required=True)
    fit.add_argument("--out", required=True)
    fit.set_defaults(func=command_fit)
    predict = commands.add_parser(
        "predict", help="predict the objective for mixtures in a CSV with one column per bucket"
    )
    predict.add_argument("--fit", required=True)
    predict.add_argument("--mixtures", required=True)
    predict.add_argument("--out", required=True)
    predict.set_defaults(func=command_predict)
    optimize = commands.add_parser(
        "optimize", help="minimize the fitted aggregate over the simplex and round to the sampler grid"
    )
    _add_data_arguments(optimize)
    optimize.add_argument("--fit", required=True)
    optimize.add_argument(
        "--cap",
        type=float,
        default=None,
        help="optional per-bucket epoch cap (not part of the procedure; the reported optima are unconstrained)",
    )
    optimize.add_argument(
        "--kl",
        type=float,
        default=0.0,
        help="optional KL penalty toward the proportional mixture (not part of the procedure)",
    )
    optimize.add_argument("--out", required=True)
    optimize.set_defaults(func=command_optimize)
    evaluate = commands.add_parser("evaluate", help="selection metrics on a held-out bank")
    evaluate.add_argument("--fit", required=True)
    evaluate.add_argument("--bank", required=True)
    evaluate.set_defaults(func=command_evaluate)
    olmix = commands.add_parser("olmix", help="the Olmix baseline: log-linear laws and the exact KL proposer")
    _add_data_arguments(olmix)
    olmix.add_argument("--objective", required=True)
    olmix.add_argument("--kl", type=float, required=True)
    olmix.add_argument("--cap", type=float, default=4.0)
    olmix.add_argument("--delta", type=float, default=OLMIX_HUBER_DELTA)
    olmix.add_argument("--out", required=True)
    olmix.set_defaults(func=command_olmix)
    test = commands.add_parser("self-test", help="parity with the bundled reference fits and policies")
    test.set_defaults(func=lambda args: self_test())
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
