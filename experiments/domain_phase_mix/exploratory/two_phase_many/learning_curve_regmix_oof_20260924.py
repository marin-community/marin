# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Out-of-fold, complement and held-out predictions of both RegMix variants on the frozen learning-curve subsets.

``learning_curve_mariner_fits_20260908`` recorded MARINER, Olmix, the quadratic and the spline on twelve subset
sizes, ten draws and five mixture-blocked outer folds. RegMix entered the paper's learning curves only through
bank regret (``learning_curve_regmix_bank_20260912``, ``learning_curve_expanded_bank_20260913``) and the
full-swarm out-of-fold fit (``regmix_official_oof_20260913``). This module fits both RegMix variants on the same
subsets, folds and calibration pin and writes records in the study's layout under the study's record directory,
so ``learning_curve_metrics_20260905`` scores them beside the four stored models (study ``mariner_regmix``) and
``plot_learning_curve_20260905`` draws them in the appendix learning-curve figures.

Models: ``regmix`` is the harness's inner-fold-tuned trees (registry entry ``lightgbm_regmix``: trees in
{100, 300, 1000} and leaves in {4, 8, 31} chosen by the three inner folds of each training set, standardized
inputs and responses, refit on all training rows), the tuned RegMix of Table 1 and the bank pass.
``regmix_official`` is the released notebook cell of ``regmix_official_oof_20260913`` (raw weights and responses,
up to 1,000 rounds, early stopping after three rounds on a third of the fold's non-anchor training rows chosen
by ``RandomState(42)``, the anchor always in training), the released RegMix of Figures 10 and 11.

Every completed job is hash-checked and skipped on rerun. Progress (jobs and work done, elapsed, projected
total and a Pacific ETA) is appended to ``progress.log`` in the output directory after every job.

usage:
  OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 uv run --offline --no-sync --with lightgbm==4.7.0 python -m \\
    experiments.domain_phase_mix.exploratory.two_phase_many.learning_curve_regmix_oof_20260924 time
  ... run --draws 10 --workers 12
"""

from __future__ import annotations

import argparse
import datetime
import functools
import hashlib
import importlib
import inspect
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import lightgbm
import numpy as np
from joblib import Parallel, delayed

# LightGBM 4.7 deprecates eval_set in favour of eval_X/eval_y; the released recipe is reproduced as written.
warnings.filterwarnings("ignore", message="The argument 'eval_set' is deprecated")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    learning_curve_mariner_fits_20260908 as fits,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    regmix_official_oof_20260913 as official,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_registry_20260902 as registry,
)

LOGGER = logging.getLogger("learning_curve_regmix_oof")
MODULE_NAME = "experiments.domain_phase_mix.exploratory.two_phase_many.learning_curve_regmix_oof_20260924"

# The study interface that learning_curve_metrics_20260905 and plot_learning_curve_20260905 read.
PANEL = fits.PANEL
TARGETS = fits.TARGETS
SUBSET_SIZES = fits.SUBSET_SIZES
OUTER_FOLDS = fits.OUTER_FOLDS
FULL_FIT = fits.FULL_FIT
FIT_COUNT = fits.FIT_COUNT
Job = fits.Job
job_path = fits.job_path
component_chunks = fits.component_chunks
RECORD_DIR = fits.RECORD_DIR
OUTPUT_DIR = fits.OUTPUT_DIR / "regmix_oof_20260924"
RECORD_VERSION = fits.RECORD_VERSION
TUNED = "regmix"
RELEASED = "regmix_official"
REGMIX_MODELS = (TUNED, RELEASED)
TUNED_ENTRY = "lightgbm_regmix"
MODEL_IDS = {**fits.MODEL_IDS, TUNED: TUNED_ENTRY, RELEASED: "lightgbm_regmix_official"}
MODEL_KEYS = tuple(MODEL_IDS)
# LightGBM fits per component and fit index: the tuned grid (nine configurations over three inner folds) plus
# the refit, against the released cell's single early-stopped fit. Used only to project the remaining time.
WORK_PER_COMPONENT = {TUNED: FIT_COUNT * (9 * 3 + 1), RELEASED: FIT_COUNT}
PACIFIC = ZoneInfo("America/Los_Angeles")
PROGRESS_LOG = "progress.log"


@functools.cache
def protocol_hash() -> str:
    """Hash of the study's subsets and folds, both fitting procedures and the LightGBM version."""
    digest = hashlib.sha256()
    digest.update(fits.protocol_hash().encode())
    for item in (models.LightGBMModel, models._fit_estimator_head, models._inner_cv_rmse_estimator):
        digest.update(inspect.getsource(item).encode())
    for item in (official.early_stopping_split, fit_released, run_job):
        digest.update(inspect.getsource(item).encode())
    digest.update(json.dumps(official.HYPER_PARAMS, sort_keys=True).encode())
    digest.update(
        f"{official.EARLY_STOPPING_ROUNDS}|{official.SPLIT_SEED}|{official.VALIDATION_FRACTION}|"
        f"{lightgbm.__version__}|{RECORD_VERSION}".encode()
    )
    return digest.hexdigest()[:16]


def now_pacific() -> str:
    return datetime.datetime.now(PACIFIC).strftime("%H:%M:%S %Z")


def fit_released(x: np.ndarray, y: np.ndarray, train: np.ndarray, calibration: int) -> Any:
    """The released RegMix cell on one training set: anchor in training, a third of the rest early-stops."""
    training, validation = official.early_stopping_split(train, calibration)
    regressor = lightgbm.LGBMRegressor(**official.HYPER_PARAMS, num_threads=1)
    regressor.fit(
        x[training],
        y[training],
        eval_set=[(x[validation], y[validation])],
        eval_metric="l2",
        callbacks=[lightgbm.early_stopping(stopping_rounds=official.EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return regressor


def record_valid(path: Path) -> bool:
    payload = benchmark.load_shard(path)
    return payload is not None and str(payload["protocol_hash"]) == protocol_hash() and str(payload["status"]) == "ok"


def enumerate_jobs(panel: benchmark.BenchPanel, draws: int, sizes: tuple[int, ...] = SUBSET_SIZES) -> list[Job]:
    """Draw-major order, so every completed draw is a whole learning curve for both variants."""
    return fits.enumerate_jobs(panel, draws, sizes, TARGETS, REGMIX_MODELS)


def job_work(job: Job) -> int:
    return WORK_PER_COMPONENT[job.model] * len(job.components)


def run_job(job: Job, record_dir: Path = RECORD_DIR) -> str:
    """Fit one job of one RegMix variant and persist its record in the study's layout; cached | fitted | failed."""
    path = job_path(job, record_dir)
    if record_valid(path):
        return "cached"
    panel = benchmark.load_panel(PANEL)
    group = panel.group(job.target)
    design = fits.cached_subset_design(job.k, job.draw)
    bank, query = benchmark.heldout_features(panel, job.target)
    responses = group.outcomes[:, list(job.component_indices)].copy()
    count = len(job.components)
    complement = np.setdiff1d(np.arange(panel.rows), np.append(design.rows, design.calibration))
    position = np.full(panel.rows, -1)
    position[design.rows] = np.arange(design.k)
    x = panel.features.weights
    tuned = registry.ENTRY_BY_ID[TUNED_ENTRY].build(panel.features) if job.model == TUNED else None
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
    heldout_prediction = np.full((len(bank), count), np.nan)
    shape_json: list[list[str]] = []
    diagnostics_json: list[list[str]] = []
    ridge = np.full((FIT_COUNT, count), np.nan)
    inner_cv_rmse = np.full((FIT_COUNT, count), np.nan)
    intercept = np.full((FIT_COUNT, count), np.nan)
    coefficients = np.full((FIT_COUNT, count, 0), np.nan)
    try:
        for fit in range(FIT_COUNT):
            train, inner, test = design.fit_rows(fit)
            shapes: list[str] = []
            diagnostics: list[str] = []
            for column, component in enumerate(job.component_indices):
                y = group.outcomes[:, component]
                if tuned is not None:
                    fitted = tuned.fit(
                        panel.features, y, train, inner, fits.fit_seed(job.target, component, job.draw, fit)
                    )
                    if fit == FULL_FIT:
                        fit_prediction[:, column] = tuned.predict(fitted, panel.features, design.rows)
                        if len(complement):
                            complement_prediction[:, column] = tuned.predict(fitted, panel.features, complement)
                        heldout_prediction[:, column] = tuned.predict(fitted, query, np.arange(len(bank)))
                    else:
                        oof[position[test], column] = tuned.predict(fitted, panel.features, test)
                    shapes.append(json.dumps(fitted.shape, sort_keys=True))
                    diagnostics.append(
                        json.dumps(
                            {
                                key: value.item() if isinstance(value, np.generic) else value
                                for key, value in fitted.diagnostics.items()
                            },
                            sort_keys=True,
                            default=float,
                        )
                    )
                    inner_cv_rmse[fit, column] = float(fitted.diagnostics["inner_cv_rmse"])
                else:
                    regressor = fit_released(x, y, train, design.calibration)
                    if fit == FULL_FIT:
                        fit_prediction[:, column] = regressor.predict(x[design.rows])
                        if len(complement):
                            complement_prediction[:, column] = regressor.predict(x[complement])
                        heldout_prediction[:, column] = regressor.predict(query.weights)
                    else:
                        oof[position[test], column] = regressor.predict(x[test])
                    best = int(regressor.best_iteration_)
                    shapes.append(json.dumps({"best_iteration": float(best)}, sort_keys=True))
                    diagnostics.append(json.dumps({"best_iteration": best, "inner_cv_rmse": None}, sort_keys=True))
            shape_json.append(shapes)
            diagnostics_json.append(diagnostics)
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
            "cv_tables": np.zeros((0, count, 0, 0)),
            "elapsed": time.monotonic() - started,
        }
    )
    benchmark.atomic_save(path, payload)
    return "fitted" if payload["status"] == "ok" else "failed"


def _run_labelled(job: Job, record_dir: Path) -> tuple[Job, str, float]:
    started = time.monotonic()
    status = run_job(job, record_dir)
    return job, status, time.monotonic() - started


def run_jobs(jobs: list[Job], workers: int, record_dir: Path, output_dir: Path) -> dict[str, int]:
    """Run the jobs in parallel and append one progress line per completed job to progress.log."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / PROGRESS_LOG
    total_work = sum(job_work(job) for job in jobs)
    counts = {"cached": 0, "fitted": 0, "failed": 0}
    fitted_work = 0
    fitted_seconds = 0.0
    done_work = 0
    started = time.monotonic()
    with log_path.open("a") as log:
        log.write(f"{now_pacific()} start jobs={len(jobs)} work={total_work} workers={workers} hash={protocol_hash()}\n")
        log.flush()
        # Dispatch the module's own function so workers import it by name instead of unpickling __main__.
        worker = importlib.import_module(MODULE_NAME)._run_labelled
        parallel = Parallel(n_jobs=workers, backend="loky", batch_size=1, return_as="generator_unordered")
        for index, (job, status, seconds) in enumerate(
            parallel(delayed(worker)(job, record_dir) for job in jobs), start=1
        ):
            counts[status] += 1
            work = job_work(job)
            done_work += work
            if status == "fitted":
                fitted_work += work
                fitted_seconds += seconds
            elapsed = time.monotonic() - started
            remaining_work = total_work - done_work
            if fitted_work:
                # Wall-clock rate of fitted work: fitted work per elapsed second, with all workers busy.
                rate = fitted_work / elapsed
                remaining_seconds = remaining_work / rate if rate > 0 else float("nan")
                eta = datetime.datetime.now(PACIFIC) + datetime.timedelta(seconds=remaining_seconds)
                eta_text = f"ETA {eta.strftime('%H:%M %Z')} ({datetime.timedelta(seconds=int(remaining_seconds))} left)"
            else:
                eta_text = "ETA pending"
            log.write(
                f"{now_pacific()} | jobs {index}/{len(jobs)} (fitted {counts['fitted']}, cached {counts['cached']}, "
                f"failed {counts['failed']}) | work {100 * done_work / total_work:.1f}% | "
                f"elapsed {datetime.timedelta(seconds=int(elapsed))} | {eta_text} | {job.label} {status} "
                f"{seconds:.0f}s\n"
            )
            log.flush()
        log.write(f"{now_pacific()} done {counts}\n")
    return counts


def time_jobs(k: int, draw: int, record_dir: Path, output_dir: Path) -> None:
    """Fit one OlmoBaseEval Easy chunk of nine components per variant at size ``k`` and project the full run."""
    panel = benchmark.load_panel(PANEL)
    jobs = [job for job in enumerate_jobs(panel, draw + 1, (k,)) if job.draw == draw and job.chunk == 0]
    jobs = [job for job in jobs if job.target == "table9"]
    full = enumerate_jobs(panel, 10)
    seconds = {}
    for job in jobs:
        started = time.monotonic()
        status = run_job(job, record_dir)
        seconds[job.model] = time.monotonic() - started
        print(f"{job.label}: {status} in {seconds[job.model]:.1f}s ({len(job.components)} components)")
    per_work = {
        model: seconds[model] / (WORK_PER_COMPONENT[model] * 9) for model in seconds
    }  # seconds per unit of work, measured at the largest subset (an upper bound for smaller ones)
    total_seconds = sum(job_work(job) * per_work[job.model] for job in full)
    print(f"projected single-core total {datetime.timedelta(seconds=int(total_seconds))} for {len(full)} jobs")
    for workers in (8, 12, 16):
        print(f"  {workers} workers: about {datetime.timedelta(seconds=int(total_seconds / workers))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    timing = sub.add_parser("time", help="fit one nine-component chunk per variant and project the full run")
    timing.add_argument("--k", type=int, default=279)
    timing.add_argument("--draw", type=int, default=0)
    run = sub.add_parser("run", help="fit every job, skipping valid records")
    run.add_argument("--draws", type=int, default=10)
    run.add_argument("--workers", type=int, default=12)
    run.add_argument("--sizes", type=int, nargs="*", default=None)
    for item in (timing, run):
        item.add_argument("--record-dir", type=Path, default=RECORD_DIR)
        item.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "time":
        time_jobs(args.k, args.draw, args.record_dir, args.output_dir)
        return
    panel = benchmark.load_panel(PANEL)
    sizes = tuple(args.sizes) if args.sizes else SUBSET_SIZES
    jobs = enumerate_jobs(panel, args.draws, sizes)
    print(f"{len(jobs)} jobs; progress in {args.output_dir / PROGRESS_LOG}")
    counts = run_jobs(jobs, args.workers, args.record_dir, args.output_dir)
    print(counts)


if __name__ == "__main__":
    main()
