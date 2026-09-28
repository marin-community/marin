# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Benchmark CLR-Ridge against DS-RE-CEQ on both datasets.

Matches the exact evaluation protocol from benchmark_dsre_ceq.py:
    - 5-fold CV, seed=42
    - Pooled OOF R², RMSE, Spearman
    - Regret@1 = mean(y[argmin(yhat_test)] - min(y_test)) across folds
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.general_scaling_models import DatasetSpec
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.clr_ridge import (
    fit_clr_ridge,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MANY_CSV = SCRIPT_DIR.parent / "two_phase_many.csv"
STARCODER_CSV = SCRIPT_DIR.parents[1] / "two_phase_starcoder_combined.csv"
RESULTS_JSON = SCRIPT_DIR / "benchmark_results.json"

N_FOLDS = 5
CV_SEED = 42

# DS-RE-CEQ baseline from the existing benchmark
DSRE_BASELINE = {
    "two_phase_many": {"r2": 0.0624, "rmse": 0.0884, "spearman": 0.2569, "regret_at_1": 0.1146, "n_params": 162},
}


def _log(msg: str, start: float = 0) -> None:
    print(f"[{perf_counter() - start:7.1f}s] {msg}", flush=True)


def _kfold_indices(n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(CV_SEED)
    return [np.asarray(idx, dtype=int) for idx in np.array_split(rng.permutation(n), N_FOLDS)]


def _cross_validate(spec: DatasetSpec, fit_fn, *, label: str, start: float) -> dict:
    folds = _kfold_indices(spec.R)
    preds = np.full(spec.R, np.nan)
    regrets: list[float] = []
    n_params = 0

    for fi in range(N_FOLDS):
        test_idx = folds[fi]
        train_idx = np.concatenate([f for i, f in enumerate(folds) if i != fi])
        train_spec = spec.subset(train_idx)
        t0 = perf_counter()
        predict_fn, info = fit_fn(train_spec, seed=fi)
        n_params = int(info.get("n_params", 0))
        fp = np.asarray(predict_fn(spec.weights[test_idx]), dtype=float)
        preds[test_idx] = fp
        y_test = spec.y[test_idx]
        regrets.append(float(y_test[np.argmin(fp)] - np.min(y_test)))
        _log(f"  {label} fold {fi}: regret={regrets[-1]:.4f} ({perf_counter()-t0:.1f}s)", start)

    residuals = spec.y - preds
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((spec.y - np.mean(spec.y)) ** 2))
    return {
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "spearman": float(spearmanr(spec.y, preds)[0]),
        "regret_at_1": float(np.mean(regrets)),
        "n_params": n_params,
    }


def _load_many_spec() -> DatasetSpec:
    from experiments.domain_phase_mix.nextgen.contracts import LoopConfig
    from experiments.domain_phase_mix.static_batch_selection import build_dataset_spec_from_frame
    from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
        create_two_phase_dolma3_dolmino_top_level_experiment,
    )

    df = pd.read_csv(MANY_CSV)
    if "status" in df.columns:
        df = df[df["status"] == "completed"].reset_index(drop=True)
    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(name="bench")
    phase_fracs = tuple(p.end_fraction - p.start_fraction for p in experiment.phase_schedule.phases)
    dtc = {d.name: int(d.total_weight) for d in experiment.domains}
    loop = LoopConfig(
        name="bench",
        objective_metric="lm_eval/mmlu_5shot/bpb",
        model_names=(),
        domain_token_counts=dtc,
        phase_fractions=phase_fracs,
        target_budget=experiment.target_budget,
    )
    return build_dataset_spec_from_frame(
        df,
        objective_metric="lm_eval/mmlu_5shot/bpb",
        name="two_phase_many",
        loop=loop,
    )


def _load_starcoder_spec() -> DatasetSpec | None:
    from experiments.domain_phase_mix.static_batch_selection import build_dataset_spec_from_frame

    if not STARCODER_CSV.exists():
        return None
    df = pd.read_csv(STARCODER_CSV)
    if "status" in df.columns:
        df = df[df["status"] == "completed"].reset_index(drop=True)
    return build_dataset_spec_from_frame(
        df,
        objective_metric="eval/paloma/dolma_100_programing_languages/bpb",
        name="starcoder_2phase",
    )


def main() -> None:
    start = perf_counter()
    results: dict[str, dict] = {}

    # ===== Many-domain dataset =====
    _log("Loading many-domain dataset", start)
    spec = _load_many_spec()
    _log(f"  R={spec.R}, N={spec.N}, M={spec.M}, y=[{spec.y.min():.4f}, {spec.y.max():.4f}]", start)

    # DS-RE-CEQ baseline (from existing benchmark)
    _log("DS-RE-CEQ baseline (from benchmark_dsre_ceq.py):", start)
    dsre = DSRE_BASELINE["two_phase_many"]
    _log(
        f"  R2={dsre['r2']:.4f}  Sp={dsre['spearman']:.4f}  RMSE={dsre['rmse']:.4f}  "
        f"Reg@1={dsre['regret_at_1']:.4f}  P={dsre['n_params']}",
        start,
    )
    results["many_dsre_ceq"] = dsre

    # CLR-Ridge variants
    variants = [
        ("CLR-Ridge(a=0.06)", 0.06),
        ("CLR-Ridge(a=0.10)", 0.10),
        ("CLR-Ridge(a=0.14)", 0.14),
        ("CLR-Ridge(a=0.20)", 0.20),
    ]
    for name, alpha in variants:
        _log(f"Evaluating {name} on many-domain", start)
        cv = _cross_validate(
            spec,
            lambda s, seed=0, a=alpha: fit_clr_ridge(s, alpha=a, seed=seed),
            label=name,
            start=start,
        )
        results[f"many_{name}"] = cv
        _log(
            f"  {name}: R2={cv['r2']:.4f}  Sp={cv['spearman']:.4f}  RMSE={cv['rmse']:.4f}  "
            f"Reg@1={cv['regret_at_1']:.4f}  P={cv['n_params']}",
            start,
        )

    # ===== StarCoder dataset =====
    sc_spec = _load_starcoder_spec()
    if sc_spec is not None:
        _log("\nLoading StarCoder dataset", start)
        _log(f"  R={sc_spec.R}, N={sc_spec.N}, M={sc_spec.M}, y=[{sc_spec.y.min():.4f}, {sc_spec.y.max():.4f}]", start)

        for name, alpha in [("CLR-Ridge(a=0.50)", 0.50), ("CLR-Ridge(a=0.80)", 0.80), ("CLR-Ridge(a=0.95)", 0.95)]:
            _log(f"Evaluating {name} on StarCoder", start)
            cv = _cross_validate(
                sc_spec,
                lambda s, seed=0, a=alpha: fit_clr_ridge(s, alpha=a, seed=seed, n_components=1),
                label=name,
                start=start,
            )
            results[f"sc_{name}"] = cv
            _log(
                f"  {name}: R2={cv['r2']:.4f}  Sp={cv['spearman']:.4f}  RMSE={cv['rmse']:.4f}  "
                f"Reg@1={cv['regret_at_1']:.4f}  P={cv['n_params']}",
                start,
            )

    # ===== Summary =====
    print("\n" + "=" * 110)
    print(f"{'Model':<30s} {'Dataset':<18s} {'R²':>8s} {'RMSE':>8s} {'Spearman':>10s} " f"{'Regret@1':>10s} {'P':>5s}")
    print("-" * 110)
    for key, cv in sorted(results.items()):
        parts = key.split("_", 1)
        ds = parts[0]
        model = parts[1] if len(parts) > 1 else key
        print(
            f"{model:<30s} {ds:<18s} {cv['r2']:>8.4f} {cv['rmse']:>8.4f} "
            f"{cv['spearman']:>10.4f} {cv['regret_at_1']:>10.4f} {cv['n_params']:>5d}"
        )
    print("=" * 110)

    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    _log(f"\nResults saved to {RESULTS_JSON}", start)


if __name__ == "__main__":
    main()
