# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Benchmark CEQ size-tied variants on the two-phase many-domain dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from experiments.domain_phase_mix.exploratory.general_scaling_models import GENERAL_MODELS
from experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_dsre_ceq import (
    _kfold_indices,
    _load_spec,
)

DEFAULT_MODELS = (
    "DS-RE-CEQ",
    "DS-RE-CEQ-ST",
    "DS-RE-CEQ-ST(lite)",
    "CR-CEQ-ST",
)
DEFAULT_OUTPUT = Path(__file__).with_name("ceq_st_variant_benchmark.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    return parser.parse_args()


def _benchmark_model(model_name: str, folds: list[np.ndarray]) -> dict[str, float | int | str]:
    _, spec, _ = _load_spec()
    model_map = {model.name: model for model in GENERAL_MODELS}
    model = model_map[model_name]
    all_predictions = np.full(spec.R, np.nan, dtype=float)
    regrets: list[float] = []
    n_params: int | None = None

    for fold_index, test_idx in enumerate(folds):
        train_idx = np.concatenate([indices for current_fold, indices in enumerate(folds) if current_fold != fold_index])
        train_spec = spec.subset(train_idx)
        predict_fn, info = model.fit_fn(train_spec, seed=0, n_restarts=8, maxiter=500)
        fold_predictions = np.asarray(predict_fn(spec.weights[test_idx]), dtype=float)
        y_test = spec.y[test_idx]
        chosen_idx = int(np.argmin(fold_predictions))
        regrets.append(float(y_test[chosen_idx] - np.min(y_test)))
        all_predictions[test_idx] = fold_predictions
        n_params = int(info["n_params"])
        print(f"finished {model_name} fold {fold_index + 1}/{len(folds)} regret={regrets[-1]:.6f}", flush=True)

    residuals = spec.y - all_predictions
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((spec.y - np.mean(spec.y)) ** 2))
    return {
        "model": model_name,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "spearman": float(spearmanr(spec.y, all_predictions)[0]),
        "regret_at_1": float(np.mean(regrets)),
        "n_params": int(n_params or 0),
        "n_runs": int(spec.R),
    }


def main() -> None:
    args = _parse_args()
    _, spec, _ = _load_spec()
    folds = _kfold_indices(spec.R)
    results = []
    for model_name in args.models:
        print(f"=== {model_name} ===", flush=True)
        results.append(_benchmark_model(model_name, folds))
        print(json.dumps(results[-1], indent=2), flush=True)

    args.output.write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
