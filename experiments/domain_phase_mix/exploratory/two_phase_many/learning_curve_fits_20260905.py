# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Learning-curve fits for the one-phase paper: WSPU and OLMix on random subsets of the Delphi 3e18 panel.

For every subset size ``k`` and draw, ``k`` runs are drawn without replacement from the 280-run
``delphi_3e18_39bucket`` fit panel and both surrogates are fitted under the Observatory benchmark
protocol: five mixture-blocked outer folds with a three-fold inner selection, plus one fit on the whole
subset that predicts the complement rows and the frozen held-out bank. Draw ``d`` at ``k = 280``
reproduces benchmark repeat ``d`` for the out-of-fold protocol (same fold seeds, same OLMix start seeds),
and the whole-subset fit uses the benchmark's held-out inner folds and seeds for every draw, so at
``k = 280`` the out-of-fold curve ends at the benchmark repeats and the held-out curve ends at the
benchmark's held-out fit (with no draw-to-draw spread, since no subset sampling remains).

Two protocol-equivalent fast paths replace the benchmark's per-component fits:

* ``fit_wspu_components`` is ``GridModel.fit`` evaluated fold-major. The response-independent centring
  and QR reduction of every (shape, ridge, inner fold) candidate is computed once and shared by all
  components of the target; each component runs the same NNLS on the same reduced system, so the
  inner-CV tables, selections, heads and predictions are bit-identical (``verify_wspu``).
* ``fit_olmix_loglinear_batched`` is the repository OLMix solver with the same start bank, bounds and
  L-BFGS-B options. The 2-point finite-difference gradient is reproduced step for step (scipy's absolute
  step 1e-8 with bound adjustment) from one matrix-vector product per coordinate and a batched Huber
  evaluation, and ``maxfun`` is rescaled so the repository's 15 000-evaluation stop fires on the same
  iteration (``verify_olmix``).

Every (target, model, k, draw, component chunk) job writes one compressed record with the subset rows,
fold labels, per-fold parameters, out-of-fold, complement and held-out predictions; completed records
are skipped on rerun, so the study can be interrupted, resumed, or extended with more draws.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import importlib
import inspect
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from joblib import Parallel, delayed
from scipy.optimize import minimize, nnls

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix import olmix_loglinear_fit as olmix_loglinear  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_registry_20260902 as registry,
)

LOGGER = logging.getLogger("learning_curve")
MODULE_NAME = "experiments.domain_phase_mix.exploratory.two_phase_many.learning_curve_fits_20260905"

PANEL = "delphi_3e18_39bucket"
TARGETS = ("uncheatable", "table9")
MODEL_IDS = {"wspu": "weibull_softplus_unscaled", "olmix": "olmix_loglinear_taskwise"}
MODEL_KEYS = tuple(MODEL_IDS)
SUBSET_SIZES = tuple(range(20, 281, 10))
SUBSET_SEED_BASE = 20_260_905
OUTER_FOLDS = benchmark.OUTER_FOLDS
INNER_FOLDS = benchmark.INNER_FOLDS
# Fit index of the whole-subset fit that predicts the complement rows and the held-out bank.
FULL_FIT = OUTER_FOLDS
FIT_COUNT = OUTER_FOLDS + 1
COMPONENT_CHUNK = 9
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "learning_curve_delphi_3e18_20260905"
RECORD_DIR = OUTPUT_DIR / "records"
RECORD_VERSION = 1
# scipy L-BFGS-B defaults reproduced by the batched OLMix solver.
LBFGSB_MAXFUN = 15_000
FD_ABS_STEP = 1e-8
FD_REL_STEP = float(np.finfo(np.float64).eps) ** 0.5


# --------------------------------------------------------------------------------------------------
# Subsets and folds
# --------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SubsetDesign:
    """One random subset of panel rows with its outer, inner and whole-subset fold labels."""

    k: int
    draw: int
    seed: int
    rows: np.ndarray
    outer_labels: np.ndarray
    inner_labels: tuple[np.ndarray, ...]
    full_inner_labels: np.ndarray

    def fit_rows(self, fit: int) -> tuple[np.ndarray, models.InnerFolds, np.ndarray]:
        """Training rows, inner folds and test rows (absolute panel indices) of one fit."""
        if fit == FULL_FIT:
            train = self.rows
            labels = self.full_inner_labels
            test = np.zeros(0, dtype=int)
        else:
            train = self.rows[self.outer_labels != fit]
            labels = self.inner_labels[fit]
            test = self.rows[self.outer_labels == fit]
        inner = tuple((train[labels != index], train[labels == index]) for index in range(INNER_FOLDS))
        return train, inner, test


def subset_seed(k: int, draw: int) -> int:
    return SUBSET_SEED_BASE + 1000 * k + draw


def subset_design(panel: benchmark.BenchPanel, k: int, draw: int) -> SubsetDesign:
    """Draw ``k`` rows without replacement and block them exactly as ``benchmark.panel_splits`` does.

    The outer seed ``FOLD_SEED + 100 * draw`` and inner seeds ``FOLD_SEED + 10_000 * draw + 100 * fold`` are
    the benchmark's repeat-``draw`` seeds; the whole-subset inner seed is the benchmark's held-out inner
    seed with no draw term, so ``k = 280`` reproduces the benchmark's outer splits for every draw and its
    held-out inner folds for every draw.
    """
    if not 0 < k <= panel.rows:
        raise ValueError(f"subset size {k} outside 1..{panel.rows}")
    seed = subset_seed(k, draw)
    rows = np.sort(np.random.default_rng(seed).choice(panel.rows, size=k, replace=False))
    weights = panel.features.weights[rows]
    outer = benchmark.olmix_benchmark.block_labels(weights, OUTER_FOLDS, benchmark.FOLD_SEED + 100 * draw)
    inner = tuple(
        benchmark.olmix_benchmark.block_labels(
            weights[outer != fold], INNER_FOLDS, benchmark.FOLD_SEED + 10_000 * draw + 100 * fold
        )
        for fold in range(OUTER_FOLDS)
    )
    full_inner = benchmark.olmix_benchmark.block_labels(weights, INNER_FOLDS, benchmark.HELDOUT_INNER_SEED)
    return SubsetDesign(k, draw, seed, rows, outer, inner, full_inner)


@functools.cache
def cached_subset_design(k: int, draw: int) -> SubsetDesign:
    return subset_design(benchmark.load_panel(PANEL), k, draw)


def fit_seed(target: str, component_index: int, draw: int, fit: int) -> int:
    """The benchmark's shard seed: repeat ``draw`` and outer fold ``fit`` for the outer folds; the
    whole-subset fit uses the benchmark's held-out seed (repeat 0, fold 0) for every draw."""
    repeat, fold = (0, 0) if fit == FULL_FIT else (draw, fit)
    identity = f"{PANEL}|{target}|{component_index}|{repeat}|{fold}"
    return int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)


# --------------------------------------------------------------------------------------------------
# WSPU: GridModel.fit evaluated fold-major with the response-independent factorisation shared
# --------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PreparedSystem:
    """The response-independent part of ``models._nonnegative_solve`` for one (design rows, ridge)."""

    rows: np.ndarray
    orthogonal: np.ndarray | None
    design_mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    root: np.ndarray
    total: float
    width: int
    ridge: float


def prepare_nonnegative(
    design: np.ndarray, multipliers: np.ndarray, ridge: float, spec: models.HeadSpec
) -> PreparedSystem:
    width = design.shape[1]
    scale = np.ones(width)
    if spec.scale_columns and spec.scale_rule == "max_abs":
        scale = np.maximum(np.abs(design).max(axis=0), models.RPL_COLUMN_SCALE_FLOOR)
    elif spec.scale_columns:
        scale = np.maximum(np.sqrt(np.mean(design**2, axis=0)), 1e-8)
    scaled = design / scale[None, :]
    weights = np.ones(len(design))
    total = float(weights.sum())
    design_mean = (weights[:, None] * scaled).sum(axis=0) / total
    root = np.sqrt(weights)
    rows = root[:, None] * (scaled - design_mean[None, :])
    if ridge > 0.0:
        rows = np.vstack([rows, np.diag(np.sqrt(ridge * multipliers))])
    if spec.tie_pairs:
        rows = np.vstack([rows, models._tie_rows(width, spec.tie_pairs)])
    orthogonal = None
    if spec.reduced_nnls and rows.shape[0] > 2 * width:
        orthogonal, rows = np.linalg.qr(rows, mode="reduced")
    return PreparedSystem(rows, orthogonal, design_mean, scale, weights, root, total, width, ridge)


def solve_prepared(system: PreparedSystem, response: np.ndarray, spec: models.HeadSpec) -> models.FittedHead:
    """The response-dependent part of ``models.fit_head`` for an NNLS head on a prepared system."""
    floor = models.link_floor(response, spec)
    target = models.link_forward(response, floor, spec)
    target_mean = float((system.weights * target).sum() / system.total)
    rhs = system.root * (target - target_mean)
    if system.ridge > 0.0:
        rhs = np.concatenate([rhs, np.zeros(system.width)])
    if spec.tie_pairs:
        rhs = np.concatenate([rhs, np.zeros(len(spec.tie_pairs))])
    if system.orthogonal is not None:
        rhs = system.orthogonal.T @ rhs
    coefficients, _residual = nnls(system.rows, rhs, maxiter=models.NNLS_MAXITER_FACTOR * system.width)
    coefficients = coefficients / system.scale
    intercept = target_mean - float(system.design_mean @ (coefficients * system.scale))
    cap = float(np.max(target)) + spec.cap_margin if spec.link is models.LinkKind.LOG_DEFICIT_BOUNDED else float("inf")
    return models.FittedHead(
        intercept=float(intercept), coefficients=np.asarray(coefficients, dtype=float), floor=floor, active=0, cap=cap
    )


def assert_batchable(model: models.GridModel, features: models.Features) -> tuple[models.Shape, ...]:
    candidates = model.candidate_shapes(features)
    if model.head.kind is not models.HeadKind.NNLS:
        raise ValueError(f"{model.model_id}: the batched path covers NNLS heads only")
    if model.refine or model.link_candidates or len(candidates) > model.screen_top:
        raise ValueError(f"{model.model_id}: the batched path covers the exhaustive single-link grid only")
    return candidates


def fit_wspu_components(
    model: models.GridModel,
    features: models.Features,
    responses: np.ndarray,
    train: np.ndarray,
    inner: models.InnerFolds,
) -> list[models.Fitted]:
    """``GridModel.fit`` for every column of ``responses`` at once; bit-identical to the per-component fit."""
    candidates = assert_batchable(model, features)
    count = responses.shape[1]
    ridge_grid = model.ridge_grid
    table = np.full((count, len(candidates), len(ridge_grid)), np.inf)
    validation_total = sum(len(validation) for _train, validation in inner)
    for shape_index, shape in enumerate(candidates):
        design = model.design(features, shape)
        spec = model.head_for(shape)
        fold_views = [
            (design.values[fold_train], design.values[validation], fold_train, validation)
            for fold_train, validation in inner
        ]
        for ridge_index, ridge in enumerate(ridge_grid):
            errors = [0.0] * count
            alive = [True] * count
            for train_values, validation_values, fold_train, validation in fold_views:
                system = prepare_nonnegative(train_values, design.ridge, ridge, spec)
                for column in range(count):
                    if not alive[column]:
                        continue
                    head = solve_prepared(system, responses[fold_train, column], spec)
                    prediction = models.predict_head(head, validation_values, spec)
                    if not np.isfinite(prediction).all():
                        alive[column] = False
                        continue
                    errors[column] += float(np.sum((prediction - responses[validation, column]) ** 2))
            for column in range(count):
                table[column, shape_index, ridge_index] = (
                    math.sqrt(errors[column] / validation_total) if alive[column] else float("inf")
                )
    fitted: list[models.Fitted] = []
    ranks: dict[int, int] = {}
    for column in range(count):
        flat = int(np.argmin(table[column].ravel()))
        shape_index, ridge_index = divmod(flat, len(ridge_grid))
        score = float(table[column, shape_index, ridge_index])
        if not math.isfinite(score):
            raise ValueError(f"{model.model_id}: no finite inner-CV candidate")
        shape = dict(candidates[shape_index])
        ridge = ridge_grid[ridge_index]
        design = model.design(features, shape)
        spec = model.head_for(shape)
        head = models.fit_head(
            models.Design(design.values[train], design.ridge, design.names), responses[train, column], ridge, spec
        )
        if shape_index not in ranks:
            ranks[shape_index] = models.effective_rank(design.values[train])
        fitted.append(
            models.Fitted(
                shape=shape,
                ridge=float(ridge),
                head=head,
                diagnostics={
                    "inner_cv_rmse": score,
                    "candidates": len(candidates) * len(ridge_grid),
                    "converged": True,
                    "boundary_hits": (
                        models._grid_edges(shape, candidates)
                        + int(ridge in (ridge_grid[0], ridge_grid[-1]) and len(ridge_grid) > 1)
                    ),
                    "effective_rank": ranks[shape_index],
                    "columns": design.values.shape[1],
                    "fitted_dof": head.active + 1 + model.shape_dof,
                    "nonlinear_dof": model.shape_dof,
                    "refine_evaluations": 0,
                    "link": str(spec.link),
                },
                cv_table=table[column],
            )
        )
    return fitted


# --------------------------------------------------------------------------------------------------
# OLMix: the repository solver with scipy's finite differences reproduced in batch
# --------------------------------------------------------------------------------------------------


def finite_difference_steps(x0: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """scipy ``approx_derivative`` 2-point steps for ``abs_step=1e-8`` followed by its bound adjustment."""
    sign_x0 = (x0 >= 0).astype(float) * 2 - 1
    h = np.full(len(x0), FD_ABS_STEP)
    dx = (x0 + h) - x0
    h = np.where(dx == 0, FD_REL_STEP * sign_x0 * np.maximum(1.0, np.abs(x0)), h)
    if np.all((lower == -np.inf) & (upper == np.inf)):
        return h
    adjusted = h.copy()
    lower_dist = x0 - lower
    upper_dist = upper - x0
    x = x0 + h
    violated = (x < lower) | (x > upper)
    fitting = np.abs(h) <= np.maximum(lower_dist, upper_dist)
    adjusted[violated & fitting] *= -1
    forward = (upper_dist >= lower_dist) & ~fitting
    adjusted[forward] = upper_dist[forward]
    backward = (upper_dist < lower_dist) & ~fitting
    adjusted[backward] = -lower_dist[backward]
    return adjusted


def fit_olmix_loglinear_batched(
    weights: np.ndarray, targets: np.ndarray, *, delta: float, seed: int, n_starts: int
) -> olmix_loglinear.OlmixLoglinearFit:
    """``olmix_loglinear_fit.fit_olmix_loglinear_model`` with its numerical gradient evaluated in batch.

    The objective, start bank, bounds and every L-BFGS-B option are the repository's. scipy differentiates
    the repository objective with one extra evaluation per coordinate (absolute step 1e-8); here the same
    perturbed evaluations are produced by one matrix-vector product per coordinate and one batched Huber
    pass, so ``f`` and the gradient agree bit for bit. scipy counts those 1 + n evaluations against
    ``maxfun=15000`` at every accepted iteration; with the gradient supplied it counts one, so ``maxfun`` is
    rescaled to ``15000 // (n + 1)`` to stop on the same iteration.
    """
    x = np.asarray(weights, dtype=float).reshape(len(weights), -1)
    y = np.asarray(targets, dtype=float)
    if np.any(y <= 0.0):
        raise ValueError("OLMix log-linear fitting requires positive targets")
    rng = np.random.default_rng(seed)
    limit = olmix_loglinear.MAX_LOG_MAGNITUDE
    n_params = 1 + x.shape[1]
    lower = np.full(n_params, -np.inf)
    upper = np.full(n_params, np.inf)
    lower[0], upper[0] = -limit, limit

    def objective(params: np.ndarray) -> float:
        log_c = float(params[0])
        coefficients = params[1:]
        logits = np.clip(x @ coefficients, -limit, limit)
        predictions = np.exp(log_c) + np.exp(logits)
        return olmix_loglinear._huber_sum(predictions - y, delta=delta)

    def objective_and_gradient(params: np.ndarray) -> tuple[float, np.ndarray]:
        value = objective(params)
        h = finite_difference_steps(params, lower, upper)
        perturbed = np.empty(n_params)
        perturbed[0] = objective(np.concatenate([[params[0] + h[0]], params[1:]]))
        coefficients = params[1:]
        logits = np.empty((n_params - 1, len(y)))
        for index in range(n_params - 1):
            shifted = coefficients.copy()
            shifted[index] = coefficients[index] + h[index + 1]
            logits[index] = x @ shifted
        predictions = np.exp(float(params[0])) + np.exp(np.clip(logits, -limit, limit))
        residuals = predictions - y[None, :]
        magnitude = np.abs(residuals)
        quadratic = 0.5 * residuals * residuals
        linear = delta * (magnitude - 0.5 * delta)
        perturbed[1:] = np.where(magnitude <= delta, quadratic, linear).sum(axis=1)
        dx = (params + h) - params
        return value, (perturbed - value) / dx

    log_c_candidates = np.linspace(np.log(max(np.min(y) * 0.25, 1e-3)), np.log(max(np.median(y), 1e-3)), 6)
    starts: list[np.ndarray] = []
    for log_c in log_c_candidates:
        starts.append(np.concatenate([[log_c], np.zeros(x.shape[1], dtype=float)]))
        for _ in range(max(n_starts // len(log_c_candidates) - 1, 0)):
            starts.append(np.concatenate([[log_c], rng.normal(0.0, 1.0, size=x.shape[1])]))
    best_params = None
    best_loss = float("inf")
    bounds = [(-limit, limit), *[(None, None)] * x.shape[1]]
    options = {"maxfun": LBFGSB_MAXFUN // (n_params + 1)}
    for start in starts:
        result = minimize(objective_and_gradient, start, method="L-BFGS-B", jac=True, bounds=bounds, options=options)
        if not result.success and best_params is not None:
            continue
        if float(result.fun) < best_loss:
            best_loss = float(result.fun)
            best_params = np.asarray(result.x, dtype=float)
    if best_params is None:
        raise RuntimeError("OLMix log-linear fit failed")
    return olmix_loglinear.OlmixLoglinearFit(
        log_c=float(best_params[0]),
        coefficients=tuple(float(value) for value in best_params[1:]),
        huber_loss=best_loss,
    )


def fit_olmix_components(
    model: models.OlmixTaskwiseModel,
    features: models.Features,
    responses: np.ndarray,
    train: np.ndarray,
    seeds: tuple[int, ...],
) -> list[models.Fitted]:
    """``OlmixTaskwiseModel.fit`` for every column of ``responses`` through the batched solver."""
    fitted = []
    for column, seed in enumerate(seeds):
        fit = fit_olmix_loglinear_batched(
            features.weights[train],
            responses[train, column],
            delta=olmix_loglinear.DEFAULT_HUBER_DELTA,
            seed=seed,
            n_starts=model.n_starts,
        )
        fitted.append(
            models.Fitted(
                {"log_c": fit.log_c},
                0.0,
                fit,
                {
                    "converged": True,
                    "boundary_hits": int(abs(fit.log_c) >= olmix_loglinear.MAX_LOG_MAGNITUDE - 1e-9),
                    "effective_rank": models.effective_rank(features.weights[train]),
                    "columns": features.buckets,
                    "fitted_dof": features.buckets + 1,
                    "nonlinear_dof": features.buckets + 1,
                    "inner_cv_rmse": float("nan"),
                    "candidates": model.n_starts,
                    "huber_loss": fit.huber_loss,
                },
            )
        )
    return fitted


# --------------------------------------------------------------------------------------------------
# Jobs and records
# --------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Job:
    target: str
    model: str
    k: int
    draw: int
    chunk: int
    components: tuple[str, ...]
    component_indices: tuple[int, ...]

    @property
    def label(self) -> str:
        return f"{self.target}/{self.model}/k{self.k:03d}/draw{self.draw:03d}/chunk{self.chunk:02d}"


@functools.cache
def protocol_hash() -> str:
    """Hash of everything a record depends on: the fit and split path sources, the models, registry, loader
    and OLMix solver modules, the panel and held-out inputs, chunking, solver constants and library versions."""
    fit_path = (
        subset_seed,
        subset_design,
        fit_seed,
        SubsetDesign.fit_rows,
        prepare_nonnegative,
        solve_prepared,
        assert_batchable,
        fit_wspu_components,
        finite_difference_steps,
        fit_olmix_loglinear_batched,
        fit_olmix_components,
        build_model,
        fit_components,
        benchmark.heldout_features,
        benchmark.olmix_benchmark.block_labels,
    )
    panel = benchmark.load_panel(PANEL)
    _coordinates, _components, heldout_hashes = benchmark.heldout_registry()
    payload = {
        "record_version": RECORD_VERSION,
        "panel": PANEL,
        "panel_inputs": panel.input_hashes,
        "heldout_inputs": heldout_hashes,
        "models": MODEL_IDS,
        "module_sources": benchmark.source_hashes(),
        "olmix_solver_source": benchmark.file_sha256(Path(olmix_loglinear.__file__).resolve()),
        "fit_path_source": (
            hashlib.sha256("".join(inspect.getsource(function) for function in fit_path).encode()).hexdigest()
        ),
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "fold_seed": benchmark.FOLD_SEED,
        "heldout_inner_seed": benchmark.HELDOUT_INNER_SEED,
        "subset_seed_base": SUBSET_SEED_BASE,
        "component_chunk": COMPONENT_CHUNK,
        "olmix_starts": olmix_loglinear.FIT_N_STARTS,
        "olmix_delta": olmix_loglinear.DEFAULT_HUBER_DELTA,
        "lbfgsb_maxfun": LBFGSB_MAXFUN,
        "fd_abs_step": FD_ABS_STEP,
        "fd_rel_step": FD_REL_STEP,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def job_path(job: Job, record_dir: Path = RECORD_DIR) -> Path:
    return record_dir / job.target / job.model / f"k{job.k:03d}" / f"draw{job.draw:03d}_chunk{job.chunk:02d}.npz"


def component_chunks(components: tuple[str, ...], chunk_size: int = COMPONENT_CHUNK) -> list[tuple[int, ...]]:
    indices = list(range(len(components)))
    return [tuple(indices[start : start + chunk_size]) for start in range(0, len(indices), chunk_size)]


def enumerate_jobs(
    panel: benchmark.BenchPanel,
    draws: int,
    sizes: tuple[int, ...] = SUBSET_SIZES,
    targets: tuple[str, ...] = TARGETS,
    model_keys: tuple[str, ...] = MODEL_KEYS,
) -> list[Job]:
    """Draw-major job order so every completed draw is a whole learning curve."""
    jobs: list[Job] = []
    for draw in range(draws):
        for k in sorted(sizes, reverse=True):
            for target in targets:
                components = panel.group(target).components
                for model in model_keys:
                    for chunk, indices in enumerate(component_chunks(components)):
                        jobs.append(Job(target, model, k, draw, chunk, tuple(components[i] for i in indices), indices))
    return jobs


def record_valid(path: Path, job: Job) -> bool:
    payload = benchmark.load_shard(path)
    if payload is None:
        return False
    identity = (str(payload["target"].item()), str(payload["model"].item()), int(payload["k"]), int(payload["draw"]))
    return (
        str(payload["protocol_hash"].item()) == protocol_hash()
        and str(payload["status"].item()) == "ok"
        and tuple(payload["components"].tolist()) == job.components
        and identity == (job.target, job.model, job.k, job.draw)
    )


def build_model(job: Job, panel: benchmark.BenchPanel) -> tuple[Any, models.Features, models.Features, int]:
    """Model, panel features (component pinned to the target's first component), bank features and bank size."""
    entry = registry.ENTRY_BY_ID[MODEL_IDS[job.model]]
    group = panel.group(job.target)
    features = dataclasses.replace(registry.apply_transform(panel.features, entry), component=group.components[0])
    bank, bank_features = benchmark.heldout_features(panel, job.target)
    query = dataclasses.replace(registry.apply_transform(bank_features, entry), component=group.components[0])
    return entry.build(features), features, query, len(bank)


def fit_components(
    job: Job,
    model: Any,
    features: models.Features,
    responses: np.ndarray,
    train: np.ndarray,
    inner: models.InnerFolds,
    fit: int,
) -> list[models.Fitted]:
    if job.model == "wspu":
        return fit_wspu_components(model, features, responses, train, inner)
    seeds = tuple(fit_seed(job.target, index, job.draw, fit) for index in job.component_indices)
    return fit_olmix_components(model, features, responses, train, seeds)


def run_job(job: Job, record_dir: Path = RECORD_DIR) -> str:
    """Fit one job and persist its record; returns cached | fitted | failed."""
    path = job_path(job, record_dir)
    if record_valid(path, job):
        return "cached"
    panel = benchmark.load_panel(PANEL)
    group = panel.group(job.target)
    design = cached_subset_design(job.k, job.draw)
    model, features, query, bank_size = build_model(job, panel)
    responses = group.outcomes[:, list(job.component_indices)].copy()
    count = len(job.components)
    complement = np.setdiff1d(np.arange(panel.rows), design.rows)
    position = np.full(panel.rows, -1)
    position[design.rows] = np.arange(design.k)
    started = time.monotonic()
    payload: dict[str, Any] = {
        "record_version": RECORD_VERSION,
        "protocol_hash": protocol_hash(),
        "target": job.target,
        "model": job.model,
        "model_id": MODEL_IDS[job.model],
        "k": job.k,
        "draw": job.draw,
        "chunk": job.chunk,
        "subset_seed": design.seed,
        "components": np.asarray(job.components),
        "component_indices": np.asarray(job.component_indices),
        "rows": design.rows,
        "outer_labels": design.outer_labels,
        "inner_labels": np.stack(
            [np.where(design.outer_labels == fold, -1, np.zeros(design.k, dtype=int)) for fold in range(OUTER_FOLDS)]
        ),
        "full_inner_labels": design.full_inner_labels,
        "complement_rows": complement,
        "observed": responses[design.rows],
    }
    for fold in range(OUTER_FOLDS):
        payload["inner_labels"][fold, design.outer_labels != fold] = design.inner_labels[fold]
    oof = np.full((design.k, count), np.nan)
    fit_prediction = np.full((design.k, count), np.nan)
    complement_prediction = np.full((len(complement), count), np.nan)
    heldout_prediction = np.full((bank_size, count), np.nan)
    shape_json: list[list[str]] = []
    ridge = np.full((FIT_COUNT, count), np.nan)
    inner_cv_rmse = np.full((FIT_COUNT, count), np.nan)
    intercept = np.full((FIT_COUNT, count), np.nan)
    coefficients = np.full((FIT_COUNT, count, features.buckets if job.model == "olmix" else 0), np.nan)
    diagnostics_json: list[list[str]] = []
    cv_tables: list[np.ndarray] = []
    try:
        for fit in range(FIT_COUNT):
            train, inner, test = design.fit_rows(fit)
            fitted = fit_components(job, model, features, responses, train, inner, fit)
            shape_json.append([json.dumps(item.shape, sort_keys=True) for item in fitted])
            diagnostics_json.append(
                [
                    json.dumps(
                        {
                            key: value if not isinstance(value, np.generic) else value.item()
                            for key, value in item.diagnostics.items()
                        },
                        sort_keys=True,
                        default=float,
                    )
                    for item in fitted
                ]
            )
            for column, item in enumerate(fitted):
                ridge[fit, column] = item.ridge
                inner_cv_rmse[fit, column] = float(item.diagnostics["inner_cv_rmse"])
                if job.model == "wspu":
                    intercept[fit, column] = item.head.intercept
                    if coefficients.shape[2] == 0:
                        coefficients = np.full((FIT_COUNT, count, len(item.head.coefficients)), np.nan)
                    coefficients[fit, column] = item.head.coefficients
                else:
                    intercept[fit, column] = item.head.log_c
                    coefficients[fit, column] = np.asarray(item.head.coefficients)
                if fit == FULL_FIT:
                    fit_prediction[:, column] = model.predict(item, features, design.rows)
                    if len(complement):
                        complement_prediction[:, column] = model.predict(item, features, complement)
                    heldout_prediction[:, column] = model.predict(item, query, np.arange(bank_size))
                else:
                    oof[position[test], column] = model.predict(item, features, test)
            if job.model == "wspu":
                cv_tables.append(np.stack([item.cv_table for item in fitted]))
        if (
            not np.isfinite(oof).all()
            or not np.isfinite(heldout_prediction).all()
            or not np.isfinite(fit_prediction).all()
        ):
            raise ValueError("non-finite prediction")
        if len(complement) and not np.isfinite(complement_prediction).all():
            raise ValueError("non-finite complement prediction")
        payload["status"] = "ok"
        payload["error"] = ""
    except Exception as error:  # the failure is recorded in the shard, not swallowed
        payload["status"] = "failed"
        payload["error"] = f"{type(error).__name__}: {error}"
        LOGGER.exception("job %s failed", job.label)
    payload.update(
        {
            "oof_prediction": oof,
            "fit_prediction": fit_prediction,
            "complement_prediction": complement_prediction,
            "heldout_prediction": heldout_prediction,
            "shape_json": np.asarray(shape_json) if shape_json else np.zeros((0, count), dtype=str),
            "diagnostics_json": np.asarray(diagnostics_json) if diagnostics_json else np.zeros((0, count), dtype=str),
            "ridge": ridge,
            "inner_cv_rmse": inner_cv_rmse,
            "intercept": intercept,
            "coefficients": coefficients,
            "cv_tables": np.stack(cv_tables) if cv_tables else np.zeros((0, count, 0, 0)),
            "elapsed": time.monotonic() - started,
        }
    )
    benchmark.atomic_save(path, payload)
    return "fitted" if payload["status"] == "ok" else "failed"


# --------------------------------------------------------------------------------------------------
# Verification against the reference implementations
# --------------------------------------------------------------------------------------------------


def verify_wspu(panel: benchmark.BenchPanel, target: str, component_indices: tuple[int, ...], k: int, draw: int) -> None:
    """Assert the batched WSPU fit equals ``GridModel.fit`` per component on an outer fold and the whole-subset fit."""
    entry = registry.ENTRY_BY_ID[MODEL_IDS["wspu"]]
    group = panel.group(target)
    design = cached_subset_design(k, draw)
    base = registry.apply_transform(panel.features, entry)
    reference_designs: list[tuple[np.ndarray, np.ndarray]] | None = None
    for component in group.components:
        features = dataclasses.replace(base, component=component)
        model = entry.build(features)
        designs = [
            (item.values, item.ridge)
            for item in (model.design(features, shape) for shape in model.candidate_shapes(features))
        ]
        if reference_designs is None:
            reference_designs = designs
            continue
        for (values, ridge), (reference_values, reference_ridge) in zip(designs, reference_designs, strict=True):
            if not (np.array_equal(values, reference_values) and np.array_equal(ridge, reference_ridge)):
                raise AssertionError(f"{target}: WSPU design depends on component {component}")
    features = dataclasses.replace(base, component=group.components[0])
    model = entry.build(features)
    responses = group.outcomes[:, list(component_indices)].copy()
    for fit in (0, FULL_FIT):
        train, inner, test = design.fit_rows(fit)
        rows = design.rows if fit == FULL_FIT else test
        batched = fit_wspu_components(model, features, responses, train, inner)
        for column, index in enumerate(component_indices):
            reference = model.fit(features, responses[:, column], train, inner, 0)
            item = batched[column]
            if not np.array_equal(reference.cv_table, item.cv_table):
                raise AssertionError(f"{target}[{index}] fit {fit}: inner-CV table differs")
            if reference.shape != item.shape or reference.ridge != item.ridge:
                raise AssertionError(f"{target}[{index}] fit {fit}: selection differs")
            if reference.head.intercept != item.head.intercept or not np.array_equal(
                reference.head.coefficients, item.head.coefficients
            ):
                raise AssertionError(f"{target}[{index}] fit {fit}: head differs")
            if reference.diagnostics != item.diagnostics:
                raise AssertionError(f"{target}[{index}] fit {fit}: diagnostics differ")
            if not np.array_equal(model.predict(reference, features, rows), model.predict(item, features, rows)):
                raise AssertionError(f"{target}[{index}] fit {fit}: predictions differ")


def verify_olmix(
    panel: benchmark.BenchPanel, target: str, component_indices: tuple[int, ...], k: int, draw: int
) -> None:
    """Assert the batched OLMix solver equals the repository solver per component on one subset fold."""
    group = panel.group(target)
    design = cached_subset_design(k, draw)
    for fit in (0, FULL_FIT):
        train, _inner, _test = design.fit_rows(fit)
        _verify_olmix_fit(panel, group, train, target, component_indices, draw, fit)


def _verify_olmix_fit(
    panel: benchmark.BenchPanel,
    group: benchmark.canonical.TargetGroup,
    train: np.ndarray,
    target: str,
    component_indices: tuple[int, ...],
    draw: int,
    fit: int,
) -> None:
    for index in component_indices:
        seed = fit_seed(target, index, draw, fit)
        response = group.outcomes[:, index]
        reference = olmix_loglinear.fit_olmix_loglinear_model(
            panel.features.weights[train], response[train], delta=olmix_loglinear.DEFAULT_HUBER_DELTA, seed=seed
        )
        batched = fit_olmix_loglinear_batched(
            panel.features.weights[train],
            response[train],
            delta=olmix_loglinear.DEFAULT_HUBER_DELTA,
            seed=seed,
            n_starts=olmix_loglinear.FIT_N_STARTS,
        )
        if reference != batched:
            raise AssertionError(f"{target}[{index}] fit {fit}: OLMix fits differ")


def verify_splits(panel: benchmark.BenchPanel, draws: int) -> None:
    """Assert the full-panel subset design reproduces the benchmark's repeat splits and held-out inner folds."""
    splits = benchmark.panel_splits(panel, draws)
    for draw in range(draws):
        design = cached_subset_design(panel.rows, draw)
        if not np.array_equal(design.rows, np.arange(panel.rows)):
            raise AssertionError("full-panel subset is not the identity")
        for fold in range(OUTER_FOLDS):
            split = next(item for item in splits if item.repeat == draw and item.fold == fold)
            train, inner, test = design.fit_rows(fold)
            if not (np.array_equal(train, split.train) and np.array_equal(test, split.test)):
                raise AssertionError(f"draw {draw} fold {fold}: outer split differs from the benchmark")
            for (mine_train, mine_validation), (ref_train, ref_validation) in zip(inner, split.inner, strict=True):
                if not (np.array_equal(mine_train, ref_train) and np.array_equal(mine_validation, ref_validation)):
                    raise AssertionError(f"draw {draw} fold {fold}: inner split differs from the benchmark")
    reference_inner = benchmark.heldout_inner_folds(panel)
    for draw in range(draws):
        _train, inner, _test = cached_subset_design(panel.rows, draw).fit_rows(FULL_FIT)
        for (mine_train, mine_validation), (ref_train, ref_validation) in zip(inner, reference_inner, strict=True):
            if not (np.array_equal(mine_train, ref_train) and np.array_equal(mine_validation, ref_validation)):
                raise AssertionError(f"draw {draw}: whole-subset inner folds differ from the benchmark's held-out folds")


# --------------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------------


def run_jobs(jobs: list[Job], workers: int, record_dir: Path) -> dict[str, int]:
    counts = {"cached": 0, "fitted": 0, "failed": 0}
    pending = [job for job in jobs if not record_valid(job_path(job, record_dir), job)]
    counts["cached"] = len(jobs) - len(pending)
    LOGGER.info("%d jobs, %d pending, %d workers", len(jobs), len(pending), workers)
    started = time.monotonic()
    parallel = Parallel(n_jobs=workers, backend="loky", batch_size=1, return_as="generator_unordered")
    for done, status in enumerate(parallel(delayed(run_job)(job, record_dir) for job in pending), start=1):
        counts[status] += 1
        elapsed = time.monotonic() - started
        if done % 10 == 0 or done == len(pending):
            LOGGER.info(
                "%d/%d pending jobs done in %.1f min; projected total %.1f min; failed %d",
                done,
                len(pending),
                elapsed / 60,
                elapsed / 60 * len(pending) / done,
                counts["failed"],
            )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="check the fast paths against the reference implementations")
    verify.add_argument("--k", type=int, default=100)
    verify.add_argument("--draw", type=int, default=0)
    verify.add_argument("--components", type=int, default=4)
    run = subparsers.add_parser("run", help="fit every pending job")
    run.add_argument("--draws", type=int, required=True)
    run.add_argument("--workers", type=int, default=os.cpu_count())
    run.add_argument("--sizes", type=int, nargs="*", default=list(SUBSET_SIZES))
    run.add_argument("--targets", nargs="*", default=list(TARGETS))
    run.add_argument("--models", nargs="*", default=list(MODEL_KEYS))
    run.add_argument("--record-dir", type=Path, default=RECORD_DIR)
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Dispatch through the importable module so loky workers unpickle jobs and the cached subset designs
    # by reference instead of by value from ``__main__``.
    importlib.import_module(MODULE_NAME).execute(args)


def execute(args: argparse.Namespace) -> None:
    panel = benchmark.load_panel(PANEL)
    if args.command == "verify":
        verify_splits(panel, 2)
        LOGGER.info("splits reproduce the benchmark")
        for target in TARGETS:
            indices = tuple(range(min(args.components, len(panel.group(target).components))))
            verify_wspu(panel, target, indices, args.k, args.draw)
            LOGGER.info("%s: batched WSPU fit is bit-identical on %d components", target, len(indices))
            verify_olmix(panel, target, indices, args.k, args.draw)
            LOGGER.info("%s: batched OLMix fit is bit-identical on %d components", target, len(indices))
        return
    jobs = enumerate_jobs(panel, args.draws, tuple(args.sizes), tuple(args.targets), tuple(args.models))
    if args.dry_run:
        pending = [job for job in jobs if not record_valid(job_path(job, args.record_dir), job)]
        LOGGER.info("%d jobs, %d pending", len(jobs), len(pending))
        return
    counts = run_jobs(jobs, args.workers, args.record_dir)
    LOGGER.info("done: %s", counts)


if __name__ == "__main__":
    main()
