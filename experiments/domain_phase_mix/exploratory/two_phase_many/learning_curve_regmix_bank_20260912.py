# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["lightgbm==4.7.0", "joblib", "numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Add RegMix bank regret using the frozen MARINER learning-curve subsets and inner folds.

Only the full-subset fits needed for bank regret are computed. No outer-fold predictions are claimed.
Run from the repository with LOKY_MAX_CPU_COUNT=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 and
``uv run --with lightgbm python -m experiments.domain_phase_mix.exploratory.two_phase_many.
learning_curve_regmix_bank_20260912 --workers 6``. Completed component chunks are hash-checked and reused.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import inspect
import json
import logging
import shutil
import time
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed

from experiments.domain_phase_mix.exploratory.two_phase_many import learning_curve_mariner_fits_20260908 as fits
from experiments.domain_phase_mix.exploratory.two_phase_many import learning_curve_metrics_20260905 as metrics

LOGGER = logging.getLogger("regmix_bank_learning_curve")
MODEL = "regmix"
MODEL_ID = "lightgbm_regmix"
OUTPUT_DIR = fits.OUTPUT_DIR / "regmix_bank_20260912"


@functools.cache
def protocol_hash() -> str:
    panel = fits.benchmark.load_panel(fits.PANEL)
    payload = {
        "stage": "full-subset bank prediction only",
        "study": fits.protocol_hash(),
        "source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": MODEL_ID,
        "model_source": hashlib.sha256(inspect.getsource(fits.models.LightGBMModel).encode()).hexdigest(),
        "lightgbm": lightgbm.__version__,
        "inputs": panel.input_hashes,
        "bank_inputs": fits.benchmark.heldout_registry()[2],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def record_path(job: fits.Job) -> Path:
    return fits.job_path(job, OUTPUT_DIR / "records")


def record_valid(job: fits.Job) -> bool:
    payload = fits.benchmark.load_shard(record_path(job))
    return payload is not None and str(payload["protocol_hash"]) == protocol_hash()


def run_job(job: fits.Job) -> str:
    if record_valid(job):
        return "cached"
    if cpu_count(only_physical_cores=True) != 1:
        raise ValueError("Set LOKY_MAX_CPU_COUNT=1 to keep each LightGBM fit on one CPU thread.")
    panel = fits.benchmark.load_panel(fits.PANEL)
    group = panel.group(job.target)
    bank, query = fits.benchmark.heldout_features(panel, job.target)
    design = fits.subset_design(panel, job.k, job.draw)
    train, inner, _test = design.fit_rows(fits.FULL_FIT)
    model = fits.registry.ENTRY_BY_ID[MODEL_ID].build(panel.features)
    predictions = np.zeros((len(bank), len(job.components)))
    shapes, scores = [], []
    started = time.monotonic()
    for column, component in enumerate(job.component_indices):
        fitted = model.fit(
            panel.features,
            group.outcomes[:, component],
            train,
            inner,
            fits.fit_seed(job.target, component, job.draw, fits.FULL_FIT),
        )
        predictions[:, column] = model.predict(fitted, query, np.arange(len(bank)))
        shapes.append(json.dumps(fitted.shape, sort_keys=True))
        scores.append(fitted.diagnostics["inner_cv_rmse"])
    if not np.isfinite(predictions).all():
        raise ValueError(f"{job.label}: nonfinite bank prediction")
    fits.benchmark.atomic_save(
        record_path(job),
        {
            "protocol_hash": protocol_hash(),
            "target": job.target,
            "model": MODEL,
            "k": job.k,
            "draw": job.draw,
            "components": np.asarray(job.components),
            "component_indices": np.asarray(job.component_indices),
            "rows": design.rows,
            "calibration_row": design.calibration,
            "full_inner_labels": design.full_inner_labels,
            "coordinate_ids": bank["coordinate_id"].to_numpy(str),
            "heldout_prediction": predictions,
            "shape_json": np.asarray(shapes),
            "inner_cv_rmse": np.asarray(scores),
            "elapsed": time.monotonic() - started,
        },
    )
    return "fitted"


def verify_archived_metrics() -> int:
    """Check every old plotted draw against the current bank ordering and frozen subset design."""
    metrics.select_study("mariner")
    panel = fits.benchmark.load_panel(fits.PANEL)
    original = pd.read_csv(fits.OUTPUT_DIR / metrics.METRICS_LONG)
    checked = 0
    for target in fits.TARGETS:
        bank, _query = fits.benchmark.heldout_features(panel, target)
        group = panel.group(target)
        for model in ("mariner", "olmix"):
            for k in fits.SUBSET_SIZES:
                for draw in range(10):
                    record, _status = metrics.load_record_set(
                        target, model, k, draw, len(group.components), fits.RECORD_DIR, strict=False
                    )
                    assert record is not None
                    design = fits.subset_design(panel, k, draw)
                    np.testing.assert_array_equal(record.rows, design.rows)
                    common = {"target": target, "model": model, "k": k, "draw": draw}
                    rescored = metrics.heldout_rows(
                        panel, target, record.heldout_prediction @ group.aggregation_weights, bank, common
                    )[0]
                    old = original[
                        original.target.eq(target)
                        & original.model.eq(model)
                        & original.k.eq(k)
                        & original.draw.eq(draw)
                        & original.evaluation.eq("heldout")
                        & original.stratum.eq("pooled")
                    ].iloc[0]
                    for key in ("regret_at_1", "selected_measured_bpb", "best_measured_bpb", "spearman"):
                        np.testing.assert_allclose(rescored[key], old[key], atol=1e-12, rtol=0)
                    assert rescored["selected_coordinate_id"] == old["selected_coordinate_id"]
                    checked += 1
    return checked


def collect(jobs: list[fits.Job]) -> None:
    panel = fits.benchmark.load_panel(fits.PANEL)
    metric_rows = []
    for target in fits.TARGETS:
        bank, _query = fits.benchmark.heldout_features(panel, target)
        group = panel.group(target)
        for k in fits.SUBSET_SIZES:
            for draw in range(10):
                subset_jobs = [job for job in jobs if (job.target, job.k, job.draw) == (target, k, draw)]
                payloads = [fits.benchmark.load_shard(record_path(job)) for job in subset_jobs]
                assert payloads and all(payload is not None for payload in payloads)
                predictions = np.concatenate([item["heldout_prediction"] for item in payloads], axis=1)
                indices = np.concatenate([item["component_indices"] for item in payloads])
                np.testing.assert_array_equal(indices, np.arange(len(group.components)))
                for item in payloads:
                    np.testing.assert_array_equal(item["coordinate_ids"], bank.coordinate_id.to_numpy(str))
                    np.testing.assert_array_equal(item["rows"], fits.subset_design(panel, k, draw).rows)
                common = {"target": target, "model": MODEL, "k": k, "draw": draw}
                metric_rows.extend(
                    metrics.heldout_rows(panel, target, predictions @ group.aggregation_weights, bank, common)
                )
    added_metrics = pd.DataFrame(metric_rows)
    added_summary, _paired = metrics.summarize(added_metrics)
    added_metrics.to_csv(OUTPUT_DIR / "regmix_metrics_long.csv", index=False)
    added_summary.to_csv(OUTPUT_DIR / "regmix_summary.csv", index=False)
    for name, added in ((metrics.METRICS_LONG, added_metrics), (metrics.SUMMARY, added_summary)):
        original = pd.read_csv(fits.OUTPUT_DIR / name)
        assert not original.model.eq(MODEL).any()
        pd.concat([original, added], ignore_index=True).to_csv(OUTPUT_DIR / name, index=False)
    shutil.copyfile(fits.OUTPUT_DIR / metrics.EFFICIENCY, OUTPUT_DIR / metrics.EFFICIENCY)
    (OUTPUT_DIR / "receipt.json").write_text(
        json.dumps(
            {
                "protocol_hash": protocol_hash(),
                "scope": "full-subset bank fits, no RegMix outer-fold predictions",
                "chunks": len(jobs),
                "task_fits": 10 * len(fits.SUBSET_SIZES) * sum(len(panel.group(t).components) for t in fits.TARGETS),
                "draws": 10,
                "nonanchor_counts": list(fits.SUBSET_SIZES),
                "bank_sizes": {t: len(fits.benchmark.heldout_features(panel, t)[0]) for t in fits.TARGETS},
                "model": MODEL_ID,
                "lightgbm_version": lightgbm.__version__,
                "source_summary_sha256": fits.benchmark.file_sha256(fits.OUTPUT_DIR / metrics.SUMMARY),
                "source_metrics_sha256": fits.benchmark.file_sha256(fits.OUTPUT_DIR / metrics.METRICS_LONG),
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Verified %d archived plotted draws against current bank", verify_archived_metrics())
    if args.verify_only:
        return
    panel = fits.benchmark.load_panel(fits.PANEL)
    jobs = fits.enumerate_jobs(panel, 10, model_keys=(MODEL,))
    pending = [job for job in jobs if not record_valid(job)]
    LOGGER.info("%d chunks, %d pending, %d workers", len(jobs), len(pending), args.workers)
    started = time.monotonic()
    parallel = Parallel(n_jobs=args.workers, backend="loky", batch_size=1, return_as="generator_unordered")
    for done, _status in enumerate(parallel(delayed(run_job)(job) for job in pending), start=1):
        if done % 20 == 0 or done == len(pending):
            elapsed = (time.monotonic() - started) / 60
            LOGGER.info(
                "%d/%d chunks in %.1f min; projected %.1f min",
                done,
                len(pending),
                elapsed,
                elapsed * len(pending) / done,
            )
    collect(jobs)
    LOGGER.info("Complete: %s", OUTPUT_DIR)


if __name__ == "__main__":
    importable = importlib.import_module(
        "experiments.domain_phase_mix.exploratory.two_phase_many.learning_curve_regmix_bank_20260912"
    )
    importable.main()
