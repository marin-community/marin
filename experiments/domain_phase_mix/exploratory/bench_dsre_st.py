# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "scipy", "scikit-learn", "cloudpickle"]
# ///
"""Benchmark DS-RE-CEQ-ST variants on single-phase dense data.

Runs 5-fold CV with regret@1, regret@5, Spearman, RMSE for each model.
Focuses on dense CSVs (epoch_dense, no_epoch_dense) and their combined
counterparts.

Usage:
  uv run experiments/domain_phase_mix/exploratory/bench_dsre_st.py
"""

import os
import sys
import warnings
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
warnings.filterwarnings("ignore")

from general_scaling_models import GENERAL_MODELS, DatasetSpec  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent

COMMON_CRAWL_TOPICS = [
    "adult_content",
    "art_and_design",
    "crime_and_law",
    "education_and_jobs",
    "electronics_and_hardware",
    "entertainment",
    "fashion_and_beauty",
    "finance_and_business",
    "food_and_dining",
    "games",
    "health",
    "history_and_geography",
    "home_and_hobbies",
    "industrial",
    "literature",
    "politics",
    "religion",
    "science_math_and_technology",
]
DOMAIN_NAMES = list(COMMON_CRAWL_TOPICS)
M = len(DOMAIN_NAMES)
EXPERIMENT_BUDGET = 1_000_000_000


def _domain_sizes_b() -> np.ndarray:
    """Measured token count per domain in billions (from GCS .stats.json)."""
    from experiments.pretraining_datasets.dolma3_pool import (
        DOLMA3_POOL_TOKEN_COUNTS_B,
        get_common_crawl_partitions_by_topic,
    )

    return np.array(
        [
            sum(DOLMA3_POOL_TOKEN_COUNTS_B[p] for p in get_common_crawl_partitions_by_topic(topic))
            for topic in COMMON_CRAWL_TOPICS
        ]
    )


def load_spec(csv_path: Path, target: str, epoch: bool) -> DatasetSpec:
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "completed"].reset_index(drop=True)
    weight_cols = [f"phase_0_{d}" for d in DOMAIN_NAMES]
    W = df[weight_cols].values[:, np.newaxis, :]
    y = df[target].values.astype(float)
    valid = ~np.isnan(y)
    if not valid.all():
        print(f"  Dropping {(~valid).sum()} rows with NaN target")
        W, y = W[valid], y[valid]

    domain_tokens = _domain_sizes_b() * 1e9
    OLD_EPOCH_TARGET_BUDGET = 290 * 28.1e9
    target_budget = OLD_EPOCH_TARGET_BUDGET if epoch else float(EXPERIMENT_BUDGET)
    C = (target_budget / domain_tokens).reshape(1, M)

    label = "epoch" if epoch else "no_epoch"
    return DatasetSpec(
        weights=W,
        y=y,
        epoch_multipliers=C,
        domain_names=DOMAIN_NAMES,
        phase_names=["phase_0"],
        small_domains=None,
        name=f"single_phase_{label}",
    )


def cross_validate_with_regret(
    spec: DatasetSpec,
    model,
    k: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """k-fold CV returning R², RMSE, Spearman, regret@1, regret@5."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(spec.R)
    folds = np.array_split(idx, k)

    nan_result = {
        "R²": float("nan"),
        "RMSE": float("nan"),
        "Spearman": float("nan"),
        "regret@1": float("nan"),
        "regret@5": float("nan"),
    }

    all_preds = np.full(spec.R, np.nan)
    fold_regrets_1: list[float] = []
    fold_regrets_5: list[float] = []

    for fold_i, test_idx in enumerate(folds):
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != fold_i])
        train_spec = spec.subset(train_idx)
        try:
            predict_fn, _info = model.fit_fn(train_spec)
            preds = predict_fn(spec.weights[test_idx])
            all_preds[test_idx] = preds

            y_test = spec.y[test_idx]
            best_true = float(np.min(y_test))

            # regret@1
            chosen_1 = int(np.argmin(preds))
            fold_regrets_1.append(float(y_test[chosen_1]) - best_true)

            # regret@5
            k5 = min(5, len(y_test))
            top5_idx = np.argsort(preds)[:k5]
            fold_regrets_5.append(float(np.min(y_test[top5_idx])) - best_true)
        except Exception as e:
            print(f"    fold {fold_i} failed: {e}")
            return nan_result

    y_true = spec.y
    residuals = y_true - all_preds
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residuals**2)))
    sp_corr, _ = spearmanr(y_true, all_preds)

    return {
        "R²": r2,
        "RMSE": rmse,
        "Spearman": float(sp_corr),
        "regret@1": float(np.mean(fold_regrets_1)),
        "regret@5": float(np.mean(fold_regrets_5)),
    }


def main():
    target = os.environ.get("TARGET", "eval/paloma/c4_en/bpb")
    model_map = {m.name: m for m in GENERAL_MODELS}

    model_names = [
        "Linear",
        "LogLinear",
        "CES",
        "CES-Overfit",
        "DS-RE-CEQ(ni)",
        "DS-RE-CEQ-ST",
        "DS-RE-CEQ-ST(lite)",
        "DS-RE-CEQ-ST(lng)",
        "CR-CEQ-ST",
    ]

    # Load CSVs
    csv_configs = {
        "epoch_dense": ("single_phase_epoch_dense.csv", True),
        "no_epoch_dense": ("single_phase_no_epoch_dense.csv", False),
    }
    specs: dict[str, DatasetSpec] = {}
    for label, (fname, is_epoch) in csv_configs.items():
        path = SCRIPT_DIR / fname
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {label}")
            continue
        specs[label] = load_spec(path, target, epoch=is_epoch)

    bench_labels = [l for l in ["epoch_dense", "no_epoch_dense"] if l in specs]

    print(f"Target: {target}")
    print(f"Models: {model_names}")
    print(f"Datasets: {bench_labels}")
    for dl in bench_labels:
        print(f"  {dl}: R={specs[dl].R}")
    print()

    # Run CV
    results: dict[tuple[str, str], dict[str, float]] = {}
    for mname in model_names:
        model = model_map.get(mname)
        if model is None:
            print(f"WARNING: {mname} not found, skipping")
            continue
        for dl in bench_labels:
            spec = specs[dl]
            if not model.applicable(spec):
                results[(mname, dl)] = {}
                print(f"  {mname} x {dl}: N/A")
                continue
            print(f"  {mname} x {dl} (R={spec.R}) ...", end=" ")
            res = cross_validate_with_regret(spec, model, k=5, seed=42)
            results[(mname, dl)] = res
            print(
                f"reg@1={res['regret@1']:.5f}  "
                f"spr={res['Spearman']:.4f}  "
                f"R²={res['R²']:.4f}  "
                f"RMSE={res['RMSE']:.4f}"
            )

    # Fit on full data and save to disk
    fits_dir = SCRIPT_DIR / "single_phase_fits"
    fits_dir.mkdir(parents=True, exist_ok=True)
    for mname in model_names:
        model = model_map.get(mname)
        if model is None:
            continue
        for dl in bench_labels:
            spec = specs[dl]
            if not model.applicable(spec):
                continue
            try:
                predict_fn, info = model.fit_fn(spec)
                out_path = fits_dir / f"{mname}_{dl}.pkl"
                with open(out_path, "wb") as f:
                    cloudpickle.dump({"predict_fn": predict_fn, "info": info, "model": mname, "dataset": dl}, f)
                print(f"  Saved {out_path.name}")
            except Exception as e:
                print(f"  Failed to save {mname} x {dl}: {e}")

    # Print summary tables
    for metric in ("regret@1", "regret@5", "Spearman", "R²", "RMSE"):
        print(f"\n  {metric}:")
        header = f"    {'Model':22s}"
        for dl in bench_labels:
            header += f" | {dl:>18s}"
        print(header)
        print("    " + "-" * (len(header) - 4))
        for mname in model_names:
            row = f"    {mname:22s}"
            for dl in bench_labels:
                res = results.get((mname, dl), {})
                val = res.get(metric, float("nan"))
                if res:
                    row += f" | {val:>18.5f}"
                else:
                    row += f" | {'N/A':>18s}"
            print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
