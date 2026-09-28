# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Out-of-fold predictions of the official RegMix regression on the frozen learning-curve folds.

Fits every component of both objectives with the released RegMix fitting cell (notebook commit dd9d1c3b, cell 13:
LightGBM, learning rate 0.01, seed 42, up to 1,000 iterations, default leaves, early stopping after three rounds on
the L1/L2 metrics of a held-out split, early-stopped model retained, raw weights and raw responses) under the same
five mixture-blocked outer folds and ten fold assignments as ``learning_curve_mariner_fits_20260908`` at the full
swarm (k = 279), so its per-task out-of-fold R^2 can sit beside MARINER's and Olmix's in the R^2-versus-SNR figure.
The early-stopping split follows the official rerun's adaptation (``regmix_official_rerun_20260913``): the
proportional anchor always trains, and a third of each fold's remaining training rows, chosen by legacy NumPy
RandomState(42), is held out for early stopping.

Writes ``<target>_draw<d>.npz`` (rows, outer_labels, oof, best_iterations) and ``manifest.json``.

usage: uv run --offline --no-sync --with lightgbm python -m experiments.domain_phase_mix.exploratory.two_phase_many.\\
    regmix_official_oof_20260913 [--workers 6]
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import lightgbm
import numpy as np
from joblib import Parallel, delayed

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    learning_curve_mariner_fits_20260908 as fits,
)

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "regmix_official_oof_20260913"
NOTEBOOK_COMMIT = "dd9d1c3b2d7c1756b1a90f0ad7603068e9856cc6"
TARGETS = ("uncheatable", "table9")
DRAWS = 10
FULL_K = 279
OUTER_FOLDS = 5
SPLIT_SEED = 42
VALIDATION_FRACTION = 1 / 3  # 93 of 279 in the official rerun
# Notebook cell 13, verbatim.
HYPER_PARAMS = {
    "task": "train",
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": ["l1", "l2"],
    "num_iterations": 1000,
    "seed": 42,
    "learning_rate": 1e-2,
    "verbosity": -1,
}
EARLY_STOPPING_ROUNDS = 3


def early_stopping_split(train: np.ndarray, calibration: int) -> tuple[np.ndarray, np.ndarray]:
    """Official-rerun split of one fold's training rows: the anchor trains, a third of the rest validates."""
    scored = np.asarray([row for row in train if row != calibration], dtype=int)
    permutation = np.random.RandomState(SPLIT_SEED).permutation(len(scored))
    validation_count = round(len(scored) * VALIDATION_FRACTION)
    validation = np.sort(scored[permutation[:validation_count]])
    training = np.sort(np.r_[calibration, scored[permutation[validation_count:]]])
    return training, validation


def fit_fold(target: str, draw: int, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-fold predictions of every component of ``target`` for one outer fold: (test rows, oof, best iterations)."""
    panel = fits.benchmark.load_panel(fits.PANEL)
    design = fits.cached_subset_design(FULL_K, draw)
    train, _inner, test = design.fit_rows(fold)
    training, validation = early_stopping_split(train, design.calibration)
    x = panel.features.weights
    outcomes = panel.group(target).outcomes
    predictions = np.empty((len(test), outcomes.shape[1]))
    best = np.empty(outcomes.shape[1], dtype=int)
    for column in range(outcomes.shape[1]):
        y = outcomes[:, column]
        regressor = lightgbm.LGBMRegressor(**HYPER_PARAMS, num_threads=1)
        regressor.fit(
            x[training],
            y[training],
            eval_set=[(x[validation], y[validation])],
            eval_metric="l2",
            callbacks=[lightgbm.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        predictions[:, column] = regressor.predict(x[test])
        best[column] = int(regressor.best_iteration_)
    return test, predictions, best


def assemble(target: str, draw: int, folds: list[tuple[np.ndarray, np.ndarray, np.ndarray]], output_dir: Path) -> None:
    design = fits.cached_subset_design(FULL_K, draw)
    oof = np.full((len(design.rows), folds[0][1].shape[1]), np.nan)
    best = np.stack([item[2] for item in folds])
    position = {int(row): index for index, row in enumerate(design.rows)}
    for test, predictions, _ in folds:
        oof[[position[int(row)] for row in test]] = predictions
    if not np.isfinite(oof).all():
        raise ValueError(f"{target} draw {draw}: incomplete out-of-fold matrix")
    np.savez(
        output_dir / f"{target}_draw{draw}.npz",
        rows=design.rows,
        outer_labels=design.outer_labels,
        oof=oof,
        best_iterations=best,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    jobs = [(target, draw, fold) for target in TARGETS for draw in range(DRAWS) for fold in range(OUTER_FOLDS)]
    results = Parallel(n_jobs=args.workers, backend="loky", verbose=5)(
        delayed(fit_fold)(target, draw, fold) for target, draw, fold in jobs
    )
    by_key: dict[tuple[str, int], list] = {}
    for (target, draw, _fold), result in zip(jobs, results, strict=True):
        by_key.setdefault((target, draw), []).append(result)
    for (target, draw), folds in by_key.items():
        assemble(target, draw, folds, args.output_dir)
    panel = fits.benchmark.load_panel(fits.PANEL)
    manifest = {
        "notebook_commit": NOTEBOOK_COMMIT,
        "hyper_params": HYPER_PARAMS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "validation_fraction": VALIDATION_FRACTION,
        "split_seed": SPLIT_SEED,
        "learning_curve_protocol_hash": fits.protocol_hash(),
        "panel_input_hashes": panel.input_hashes,
        "components": {target: list(panel.group(target).components) for target in TARGETS},
        "lightgbm": importlib.metadata.version("lightgbm"),
        "numpy": np.__version__,
        "fits": len(jobs),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(by_key)} record sets to {args.output_dir} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
