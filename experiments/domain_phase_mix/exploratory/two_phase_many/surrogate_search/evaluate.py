# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Evaluate candidate surrogate models with 5-fold CV on many-domain and StarCoder data."""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Allow running as standalone script
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.general_scaling_models import DatasetSpec
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.models import MODEL_REGISTRY

SCRIPT_DIR = Path(__file__).resolve().parent
MANY_CSV = SCRIPT_DIR.parent / "two_phase_many.csv"
STARCODER_CSV = SCRIPT_DIR.parents[1] / "two_phase_starcoder_combined.csv"
RESULTS_JSON = SCRIPT_DIR / "results.json"

N_FOLDS = 5
CV_SEED = 42


@dataclass(frozen=True)
class CvResult:
    model_name: str
    dataset_name: str
    r2: float
    rmse: float
    spearman: float
    regret_at_1: float
    n_params: int
    duration_s: float
    error: str | None = None


def _kfold_indices(n_rows: int, *, n_folds: int = N_FOLDS, seed: int = CV_SEED) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(n_rows)
    return [np.asarray(idx, dtype=int) for idx in np.array_split(permuted, n_folds)]


def cross_validate(spec: DatasetSpec, model_name: str, fit_fn, *, n_folds: int = N_FOLDS) -> CvResult:
    """Run pooled k-fold CV and return metrics."""
    start = perf_counter()
    folds = _kfold_indices(spec.R, n_folds=n_folds)
    all_preds = np.full(spec.R, np.nan)
    fold_regrets: list[float] = []
    n_params = 0

    for fold_idx in range(n_folds):
        test_idx = folds[fold_idx]
        train_idx = np.concatenate([f for i, f in enumerate(folds) if i != fold_idx])
        train_spec = spec.subset(train_idx)

        try:
            predict_fn, info = fit_fn(train_spec, seed=fold_idx)
        except Exception as e:
            return CvResult(
                model_name=model_name,
                dataset_name=spec.name,
                r2=float("nan"),
                rmse=float("nan"),
                spearman=float("nan"),
                regret_at_1=float("nan"),
                n_params=0,
                duration_s=perf_counter() - start,
                error=f"Fold {fold_idx}: {e}",
            )

        n_params = int(info.get("n_params", 0))
        preds = np.asarray(predict_fn(spec.weights[test_idx]), dtype=float)
        all_preds[test_idx] = preds
        y_test = spec.y[test_idx]
        chosen = int(np.argmin(preds))
        fold_regrets.append(float(y_test[chosen] - np.min(y_test)))

    residuals = spec.y - all_preds
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((spec.y - np.mean(spec.y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residuals**2)))
    sp = float(spearmanr(spec.y, all_preds)[0])
    regret = float(np.mean(fold_regrets))

    return CvResult(
        model_name=model_name,
        dataset_name=spec.name,
        r2=r2,
        rmse=rmse,
        spearman=sp,
        regret_at_1=regret,
        n_params=n_params,
        duration_s=perf_counter() - start,
    )


def load_many_domain_spec() -> DatasetSpec:
    """Load the two_phase_many dataset as a DatasetSpec."""
    from experiments.domain_phase_mix.nextgen.contracts import LoopConfig
    from experiments.domain_phase_mix.static_batch_selection import build_dataset_spec_from_frame
    from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
        create_two_phase_dolma3_dolmino_top_level_experiment,
    )

    df = pd.read_csv(MANY_CSV)
    if "status" in df.columns:
        df = df[df["status"] == "completed"].reset_index(drop=True)

    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(name="analysis")
    phase_fractions = tuple(p.end_fraction - p.start_fraction for p in experiment.phase_schedule.phases)
    domain_token_counts = {d.name: int(d.total_weight) for d in experiment.domains}
    loop = LoopConfig(
        name="analysis",
        objective_metric="lm_eval/mmlu_5shot/bpb",
        model_names=(),
        domain_token_counts=domain_token_counts,
        phase_fractions=phase_fractions,
        target_budget=experiment.target_budget,
    )
    return build_dataset_spec_from_frame(
        df,
        objective_metric="lm_eval/mmlu_5shot/bpb",
        name="two_phase_many",
        loop=loop,
    )


def load_starcoder_spec() -> DatasetSpec:
    """Load the StarCoder two-phase dataset as a DatasetSpec."""
    from experiments.domain_phase_mix.static_batch_selection import build_dataset_spec_from_frame

    if not STARCODER_CSV.exists():
        raise FileNotFoundError(f"StarCoder CSV not found: {STARCODER_CSV}")

    df = pd.read_csv(STARCODER_CSV)
    if "status" in df.columns:
        df = df[df["status"] == "completed"].reset_index(drop=True)

    return build_dataset_spec_from_frame(
        df,
        objective_metric="eval/paloma/dolma_100_programing_languages/bpb",
        name="starcoder_2phase",
    )


def run_all(model_names: list[str] | None = None, dataset_names: list[str] | None = None) -> list[CvResult]:
    """Run all specified models on all specified datasets."""
    model_names = model_names or list(MODEL_REGISTRY.keys())
    dataset_names = dataset_names or ["two_phase_many", "starcoder_2phase"]

    specs = {}
    if "two_phase_many" in dataset_names:
        specs["two_phase_many"] = load_many_domain_spec()
    if "starcoder_2phase" in dataset_names:
        try:
            specs["starcoder_2phase"] = load_starcoder_spec()
        except FileNotFoundError as e:
            print(f"Warning: {e}", flush=True)

    results = []
    for dataset_name, spec in specs.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Dataset: {dataset_name}  (R={spec.R}, N={spec.N}, M={spec.M})", flush=True)
        print(f"y range: [{spec.y.min():.4f}, {spec.y.max():.4f}], std={spec.y.std():.4f}", flush=True)
        print(f"{'='*60}", flush=True)

        for mname in model_names:
            if mname not in MODEL_REGISTRY:
                print(f"  SKIP {mname}: not in registry", flush=True)
                continue
            fit_fn = MODEL_REGISTRY[mname]
            print(f"\n  Evaluating {mname}...", flush=True)
            try:
                result = cross_validate(spec, mname, fit_fn)
                results.append(result)
                status = "OK" if result.error is None else f"ERROR: {result.error}"
                print(
                    f"    {status}  R2={result.r2:.4f}  RMSE={result.rmse:.4f}  "
                    f"Spearman={result.spearman:.4f}  Regret@1={result.regret_at_1:.4f}  "
                    f"P={result.n_params}  ({result.duration_s:.1f}s)",
                    flush=True,
                )
            except Exception:
                traceback.print_exc()
                results.append(
                    CvResult(
                        model_name=mname,
                        dataset_name=dataset_name,
                        r2=float("nan"),
                        rmse=float("nan"),
                        spearman=float("nan"),
                        regret_at_1=float("nan"),
                        n_params=0,
                        duration_s=0.0,
                        error=traceback.format_exc(),
                    )
                )

    return results


def save_results(results: list[CvResult]) -> None:
    """Save results to JSON."""
    payload = [
        {
            "model": r.model_name,
            "dataset": r.dataset_name,
            "r2": r.r2,
            "rmse": r.rmse,
            "spearman": r.spearman,
            "regret_at_1": r.regret_at_1,
            "n_params": r.n_params,
            "duration_s": round(r.duration_s, 1),
            "error": r.error,
        }
        for r in results
    ]
    RESULTS_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved to {RESULTS_JSON}", flush=True)


def print_summary(results: list[CvResult]) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 100)
    print(
        f"{'Model':<25s} {'Dataset':<20s} {'R2':>8s} {'RMSE':>8s} {'Spearman':>10s} "
        f"{'Regret@1':>10s} {'P':>5s} {'Time':>8s}"
    )
    print("-" * 100)
    for r in sorted(results, key=lambda x: (x.dataset_name, -x.r2 if np.isfinite(x.r2) else -999)):
        print(
            f"{r.model_name:<25s} {r.dataset_name:<20s} {r.r2:>8.4f} {r.rmse:>8.4f} "
            f"{r.spearman:>10.4f} {r.regret_at_1:>10.4f} {r.n_params:>5d} {r.duration_s:>7.1f}s"
        )
    print("=" * 100)


if __name__ == "__main__":
    results = run_all()
    save_results(results)
    print_summary(results)
