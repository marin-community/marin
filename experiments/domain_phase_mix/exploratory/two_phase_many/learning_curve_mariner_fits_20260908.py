# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Learning curves of the frozen procedure (MARINER) under the calibration-pinned protocol.

The study of ``learning_curve_fits_20260905`` compared the additive predecessor with Olmix on random subsets of
the Qwen3 360M/1.6B swarm. This module repeats it for the frozen procedure, its per-bucket-shape ablation and
Olmix on identical subsets under the frozen protocol: the proportional run is calibration data, so it is in
every training set of every fit and is never scored. A subset of size ``k`` therefore holds ``k`` random runs
(drawn from the 279 other runs) plus the pinned run; the out-of-fold, complement and held-out predictions and
the record layout are those of the earlier study, so its metrics and plotting scripts read these records with
``--study mariner``.

Models: ``mariner`` (``weibull_softplus_unscaled@kappa_floor_link_flat15_nocap``), ``mariner_per_bucket`` (its
shape-sharing ablation, one (rate, power, threshold) per bucket), ``olmix`` (the taskwise log-linear law,
fitted with the earlier study's batched solver) and, added 2026-09-09 for the complexity ladder, the additive
``quadratic`` and natural ``spline`` in log-epochs under MARINER's floor and log link. The floored models are
fitted per component with their own floor anchors, so each component builds its own model. Adding models changes
the protocol hash, so records fitted before 2026-09-09 read as ``stale`` in the metrics; their subsets, folds and
predictions are unchanged, and the metrics accept them with ``--allow-stale``.

usage:
  uv run learning_curve_mariner_fits_20260908.py verify [--k 20] [--components 2]
  uv run learning_curve_mariner_fits_20260908.py run --draws B --workers W [--models ...] [--sizes ...] [--dry-run]
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
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from joblib import Parallel, delayed

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix import olmix_loglinear_fit as olmix_loglinear  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    learning_curve_fits_20260905 as legacy,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_registry_20260902 as registry,
)

LOGGER = logging.getLogger("learning_curve_mariner")
MODULE_NAME = "experiments.domain_phase_mix.exploratory.two_phase_many.learning_curve_mariner_fits_20260908"

PANEL = legacy.PANEL
TARGETS = legacy.TARGETS
MODEL_IDS = {
    "mariner": "weibull_softplus_unscaled@kappa_floor_link_flat15_nocap",
    "mariner_per_bucket": "weibull_softplus_unscaled@kappa_floor_link_flat15_nocap_per_bucket_shape",
    "olmix": "olmix_loglinear_taskwise",
    "quadratic": registry.QUADRATIC_LINKED_ID,
    "spline": registry.SPLINE_LINKED_ID,
}
MODEL_KEYS = tuple(MODEL_IDS)
FLOORED_MODELS = ("mariner", "mariner_per_bucket", "quadratic", "spline")
# 279 is every run but the pinned one, so the largest subset is the whole swarm.
SUBSET_SIZES = (20, 30, 40, 50, 60, 80, 100, 120, 160, 200, 240, 279)
SUBSET_SEED_BASE = 20_260_908
OUTER_FOLDS = legacy.OUTER_FOLDS
INNER_FOLDS = legacy.INNER_FOLDS
FULL_FIT = legacy.FULL_FIT
FIT_COUNT = legacy.FIT_COUNT
COMPONENT_CHUNK = legacy.COMPONENT_CHUNK
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "learning_curve_mariner_delphi_3e18_20260908"
RECORD_DIR = OUTPUT_DIR / "records"
RECORD_VERSION = 1

Job = legacy.Job
component_chunks = legacy.component_chunks
fit_seed = legacy.fit_seed


def job_path(job: Job, record_dir: Path = RECORD_DIR) -> Path:
    return legacy.job_path(job, record_dir)


# --------------------------------------------------------------------------------------------------
# Subsets with the pinned calibration run
# --------------------------------------------------------------------------------------------------


def calibration_row(panel: benchmark.BenchPanel) -> int:
    rows = benchmark.calibration_rows(panel)
    if len(rows) != 1:
        raise ValueError(f"{panel.name}: expected one calibration run, found {len(rows)}")
    return int(rows[0])


@dataclasses.dataclass(frozen=True)
class PinnedSubsetDesign:
    """``k`` random scored rows with their outer, inner and whole-subset fold labels, plus the pinned row.

    ``rows`` are the scored rows (never the calibration run). Every training set of every fit, outer and inner,
    also contains the calibration run, which is never validated.
    """

    k: int
    draw: int
    seed: int
    calibration: int
    rows: np.ndarray
    outer_labels: np.ndarray
    inner_labels: tuple[np.ndarray, ...]
    full_inner_labels: np.ndarray

    def fit_rows(self, fit: int) -> tuple[np.ndarray, models.InnerFolds, np.ndarray]:
        """Training rows (with the pinned run last), inner folds and test rows, as absolute panel indices."""
        if fit == FULL_FIT:
            scored = self.rows
            labels = self.full_inner_labels
            test = np.zeros(0, dtype=int)
        else:
            scored = self.rows[self.outer_labels != fit]
            labels = self.inner_labels[fit]
            test = self.rows[self.outer_labels == fit]
        pinned = np.asarray([self.calibration], dtype=int)
        train = np.concatenate([scored, pinned])
        inner = tuple(
            (np.concatenate([scored[labels != index], pinned]), scored[labels == index]) for index in range(INNER_FOLDS)
        )
        return train, inner, test


def subset_seed(k: int, draw: int) -> int:
    return SUBSET_SEED_BASE + 1000 * k + draw


def subset_design(panel: benchmark.BenchPanel, k: int, draw: int) -> PinnedSubsetDesign:
    """Draw ``k`` scored rows from the runs other than the calibration run and block them as the benchmark does."""
    calibration = calibration_row(panel)
    candidates = np.asarray([row for row in range(panel.rows) if row != calibration], dtype=int)
    if not 0 < k <= len(candidates):
        raise ValueError(f"subset size {k} outside 1..{len(candidates)}")
    seed = subset_seed(k, draw)
    rows = np.sort(np.random.default_rng(seed).choice(candidates, size=k, replace=False))
    weights = panel.features.weights[rows]
    outer = benchmark.olmix_benchmark.block_labels(weights, OUTER_FOLDS, benchmark.FOLD_SEED + 100 * draw)
    inner = tuple(
        benchmark.olmix_benchmark.block_labels(
            weights[outer != fold], INNER_FOLDS, benchmark.FOLD_SEED + 10_000 * draw + 100 * fold
        )
        for fold in range(OUTER_FOLDS)
    )
    full_inner = benchmark.olmix_benchmark.block_labels(weights, INNER_FOLDS, benchmark.HELDOUT_INNER_SEED)
    return PinnedSubsetDesign(k, draw, seed, calibration, rows, outer, inner, full_inner)


@functools.cache
def cached_subset_design(k: int, draw: int) -> PinnedSubsetDesign:
    return subset_design(benchmark.load_panel(PANEL), k, draw)


# --------------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BuiltModels:
    """The models of one job: one per component for the floored models, one shared model for Olmix."""

    per_component: tuple[Any, ...]
    features: tuple[models.Features, ...]
    query: tuple[models.Features, ...]
    bank_size: int

    def model(self, column: int) -> Any:
        return self.per_component[column if len(self.per_component) > 1 else 0]

    def feature_set(self, column: int) -> models.Features:
        return self.features[column if len(self.features) > 1 else 0]

    def query_set(self, column: int) -> models.Features:
        return self.query[column if len(self.query) > 1 else 0]


def build_models(job: Job, panel: benchmark.BenchPanel) -> BuiltModels:
    entry = registry.ENTRY_BY_ID[MODEL_IDS[job.model]]
    group = panel.group(job.target)
    base = registry.apply_transform(panel.features, entry)
    bank, bank_features = benchmark.heldout_features(panel, job.target)
    bank_base = registry.apply_transform(bank_features, entry)
    if job.model in FLOORED_MODELS:
        features = tuple(dataclasses.replace(base, component=name) for name in job.components)
        query = tuple(dataclasses.replace(bank_base, component=name) for name in job.components)
        return BuiltModels(tuple(entry.build(item) for item in features), features, query, len(bank))
    features = (dataclasses.replace(base, component=group.components[0]),)
    query = (dataclasses.replace(bank_base, component=group.components[0]),)
    return BuiltModels((entry.build(features[0]),), features, query, len(bank))


def fit_components(
    job: Job,
    built: BuiltModels,
    responses: np.ndarray,
    train: np.ndarray,
    inner: models.InnerFolds,
    fit: int,
) -> list[models.Fitted]:
    seeds = tuple(fit_seed(job.target, index, job.draw, fit) for index in job.component_indices)
    if job.model in FLOORED_MODELS:
        return [
            built.model(column).fit(built.feature_set(column), responses[:, column], train, inner, seed)
            for column, seed in enumerate(seeds)
        ]
    return legacy.fit_olmix_components(built.model(0), built.feature_set(0), responses, train, seeds)


# --------------------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------------------


@functools.cache
def protocol_hash() -> str:
    """Hash of everything a record depends on; see ``learning_curve_fits_20260905.protocol_hash``."""
    fit_path = (
        subset_seed,
        subset_design,
        PinnedSubsetDesign.fit_rows,
        build_models,
        fit_components,
        legacy.fit_seed,
        legacy.fit_olmix_loglinear_batched,
        legacy.fit_olmix_components,
        legacy.finite_difference_steps,
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
        "calibration_marker": benchmark.CALIBRATION_RUN_MARKER,
        "olmix_starts": olmix_loglinear.FIT_N_STARTS,
        "olmix_delta": olmix_loglinear.DEFAULT_HUBER_DELTA,
        "lbfgsb_maxfun": legacy.LBFGSB_MAXFUN,
        "fd_abs_step": legacy.FD_ABS_STEP,
        "fd_rel_step": legacy.FD_REL_STEP,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


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


def run_job(job: Job, record_dir: Path = RECORD_DIR) -> str:
    """Fit one job and persist its record in the earlier study's layout; returns cached | fitted | failed."""
    path = job_path(job, record_dir)
    if record_valid(path, job):
        return "cached"
    panel = benchmark.load_panel(PANEL)
    group = panel.group(job.target)
    design = cached_subset_design(job.k, job.draw)
    built = build_models(job, panel)
    responses = group.outcomes[:, list(job.component_indices)].copy()
    count = len(job.components)
    complement = np.setdiff1d(np.arange(panel.rows), np.append(design.rows, design.calibration))
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
        "calibration_row": design.calibration,
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
    heldout_prediction = np.full((built.bank_size, count), np.nan)
    shape_json: list[list[str]] = []
    ridge = np.full((FIT_COUNT, count), np.nan)
    inner_cv_rmse = np.full((FIT_COUNT, count), np.nan)
    intercept = np.full((FIT_COUNT, count), np.nan)
    coefficients = np.full((FIT_COUNT, count, 0), np.nan)
    diagnostics_json: list[list[str]] = []
    cv_tables: list[np.ndarray] = []
    try:
        for fit in range(FIT_COUNT):
            train, inner, test = design.fit_rows(fit)
            fitted = fit_components(job, built, responses, train, inner, fit)
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
                model = built.model(column)
                features = built.feature_set(column)
                ridge[fit, column] = item.ridge
                inner_cv_rmse[fit, column] = float(item.diagnostics["inner_cv_rmse"])
                if job.model in FLOORED_MODELS:
                    intercept[fit, column] = item.head.intercept
                    vector = np.asarray(item.head.coefficients, dtype=float)
                else:
                    intercept[fit, column] = item.head.log_c
                    vector = np.asarray(item.head.coefficients, dtype=float)
                if coefficients.shape[2] != len(vector):
                    grown = np.full((FIT_COUNT, count, max(coefficients.shape[2], len(vector))), np.nan)
                    grown[:, :, : coefficients.shape[2]] = coefficients
                    coefficients = grown
                coefficients[fit, column, : len(vector)] = vector
                if fit == FULL_FIT:
                    fit_prediction[:, column] = model.predict(item, features, design.rows)
                    if len(complement):
                        complement_prediction[:, column] = model.predict(item, features, complement)
                    heldout_prediction[:, column] = model.predict(
                        item, built.query_set(column), np.arange(built.bank_size)
                    )
                else:
                    oof[position[test], column] = model.predict(item, features, test)
            if job.model in FLOORED_MODELS and all(item.cv_table is not None for item in fitted):
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
# Verification and command line
# --------------------------------------------------------------------------------------------------


def verify(k: int, components: int, record_dir: Path) -> None:
    """Fit one small job per model into a scratch record directory and check the pinning invariants."""
    panel = benchmark.load_panel(PANEL)
    calibration = calibration_row(panel)
    for draw in (0, 1):
        design = cached_subset_design(k, draw)
        if calibration in design.rows or len(design.rows) != k:
            raise AssertionError("the calibration run must never be a scored row")
        for fit in range(FIT_COUNT):
            train, inner, test = design.fit_rows(fit)
            if calibration not in train or calibration in test:
                raise AssertionError(f"draw {draw} fit {fit}: the calibration run is not pinned to training")
            for inner_train, inner_validation in inner:
                if calibration not in inner_train or calibration in inner_validation:
                    raise AssertionError(f"draw {draw} fit {fit}: the calibration run is not pinned in an inner fold")
    for target in TARGETS:
        names = panel.group(target).components[:components]
        for model in MODEL_KEYS:
            job = Job(target, model, k, 0, 0, tuple(names), tuple(range(len(names))))
            started = time.monotonic()
            status = run_job(job, record_dir)
            payload = benchmark.load_shard(job_path(job, record_dir))
            assert payload is not None
            print(
                f"{job.label}: {status} in {time.monotonic() - started:.1f}s, status {payload['status']}, "
                f"oof finite {np.isfinite(payload['oof_prediction']).all()}, error {payload['error']!s}"
            )
            if str(payload["status"]) != "ok":
                raise AssertionError(f"{job.label}: {payload['error']}")


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
    check = subparsers.add_parser("verify", help="fit one small job per model and check the pinning")
    check.add_argument("--k", type=int, default=20)
    check.add_argument("--components", type=int, default=2)
    check.add_argument("--record-dir", type=Path, default=OUTPUT_DIR / "verify_records")
    run = subparsers.add_parser("run", help="fit every pending job")
    run.add_argument("--draws", type=int, required=True)
    run.add_argument("--workers", type=int, default=os.cpu_count())
    run.add_argument("--sizes", type=int, nargs="*", default=list(SUBSET_SIZES))
    run.add_argument("--targets", nargs="*", default=list(TARGETS))
    run.add_argument("--models", nargs="*", default=list(MODEL_KEYS))
    run.add_argument("--record-dir", type=Path, default=RECORD_DIR)
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.command == "verify":
        verify(args.k, args.components, args.record_dir)
        print("verify passed")
        return
    unknown = [model for model in args.models if model not in MODEL_KEYS]
    if unknown:
        raise ValueError(f"unknown models {unknown}")
    panel = benchmark.load_panel(PANEL)
    jobs = enumerate_jobs(panel, args.draws, tuple(args.sizes), tuple(args.targets), tuple(args.models))
    if args.dry_run:
        pending = sum(not record_valid(job_path(job, args.record_dir), job) for job in jobs)
        print(f"{len(jobs)} jobs, {pending} pending; protocol {protocol_hash()[:12]}")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "protocol_hash.txt").write_text(protocol_hash() + "\n")
    counts = run_jobs(jobs, args.workers, args.record_dir)
    print(f"completed: {counts}")


if __name__ == "__main__":
    # Dispatch through the importable module so loky workers unpickle ``run_job`` by reference.
    importable = importlib.import_module(MODULE_NAME)
    importable.main()
