# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Cross-dataset evaluation of generalized scaling models.

Fits each model from general_scaling_models.py on three datasets
using k-fold cross-validation and reports R², RMSE, Spearman, and Huber loss.

Datasets:
  1. single_phase_no_epoch.csv  — 1 phase, 18 CC domains, no epoching
  2. single_phase_epoch.csv     — 1 phase, 18 CC domains, with epoching
  3. 3_partitions_3_phases_6.csv — 3 phases, 3 domains (nem/dolmino/ot)

Usage:
    uv run python experiments/domain_phase_mix/exploratory/cross_dataset_evaluation.py
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.domain_phase_mix.exploratory.general_scaling_models import (
    GENERAL_MODELS,
    DatasetSpec,
    GeneralModelSpec,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Epoch multiplier configurations
# ---------------------------------------------------------------------------

# Per-topic CC token counts in billions (from dolma3_pool.py quality tiers).
# epoch_mult = target_budget / domain_size_tokens.
# These are computed from experiments/pretraining_datasets/dolma3_pool.py COMMON_CRAWL_TOPICS.
CC_TOPIC_SIZES_B = {
    "adult_content": 365.30,
    "art_and_design": 421.50,
    "crime_and_law": 505.80,
    "education_and_jobs": 477.70,
    "electronics_and_hardware": 421.50,
    "entertainment": 421.50,
    "fashion_and_beauty": 393.40,
    "finance_and_business": 505.80,
    "food_and_dining": 449.60,
    "games": 449.60,
    "health": 505.80,
    "history_and_geography": 449.60,
    "home_and_hobbies": 421.50,
    "industrial": 477.70,
    "literature": 449.60,
    "politics": 505.80,
    "religion": 477.70,
    "science_math_and_technology": 449.60,
}
TOTAL_CC_B = sum(CC_TOPIC_SIZES_B.values())  # ~8149B

# 3-phase domain sizes in raw tokens
DOMAIN_TOKENS_3P = {
    "nemotron_full": 5_729_908_864_777,
    "dolmino": 826_715_423_646,
    "openthoughts_sft": 17_449_811_417,
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def _extract_phase_domain_columns(columns: list[str]) -> tuple[list[str], list[str]]:
    """Identify phase_K_domain columns and return (phase_names, domain_names)."""
    pat = re.compile(r"^phase_(\d+)_(.+)$")
    phases: set[str] = set()
    domains: set[str] = set()
    for c in columns:
        m = pat.match(c)
        if m:
            phases.add(f"phase_{m.group(1)}")
            domains.add(m.group(2))
    phase_names = sorted(phases)
    # Preserve column order for domains (use first phase)
    if phase_names:
        first_phase = phase_names[0]
        domain_names = [c.replace(f"{first_phase}_", "") for c in columns if c.startswith(f"{first_phase}_")]
    else:
        domain_names = sorted(domains)
    return phase_names, domain_names


def load_dataset_spec(
    csv_path: str | Path,
    target_col: str,
    epoch_multipliers: np.ndarray,
    small_domains: list[int] | None = None,
    name: str = "",
) -> DatasetSpec:
    """Load CSV and construct a DatasetSpec."""
    df = pd.read_csv(csv_path)

    # Filter to completed runs
    if "status" in df.columns:
        df = df[df["status"] == "completed"].reset_index(drop=True)

    phase_names, domain_names = _extract_phase_domain_columns(list(df.columns))
    N = len(phase_names)
    M = len(domain_names)
    R = len(df)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {csv_path}. Available: {list(df.columns)[:20]}...")

    # Build (R, N, M) weight array
    weights = np.zeros((R, N, M))
    for ki, ph in enumerate(phase_names):
        for di, dom in enumerate(domain_names):
            col = f"{ph}_{dom}"
            if col in df.columns:
                weights[:, ki, di] = df[col].values

    y = df[target_col].values.astype(float)

    # Drop rows with NaN target
    valid = ~np.isnan(y)
    if not valid.all():
        log.warning(f"Dropping {(~valid).sum()} rows with NaN target in {name}")
        weights = weights[valid]
        y = y[valid]

    log.info(f"Loaded {name}: R={len(y)}, N={N}, M={M}, target={target_col}")
    return DatasetSpec(
        weights=weights,
        y=y,
        epoch_multipliers=epoch_multipliers,
        domain_names=domain_names,
        phase_names=phase_names,
        small_domains=small_domains,
        name=name,
    )


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def _topk_metrics(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> dict[str, float]:
    """Compute top-K recall and mean-rank metrics (lower y = better).

    Args:
        y_true: True target values (lower is better, e.g. BPB).
        y_pred: Predicted target values.
        K: Number of top runs to consider.

    Returns:
        Dictionary with:
        - recall@K: fraction of predicted top-K that are in true top-K
        - mean_true_rank_of_pred_topK: avg true rank of runs predicted to be top-K
        - mean_pred_rank_of_true_topK: avg predicted rank of runs that are truly top-K
    """
    R = len(y_true)
    if K > R:
        K = R

    # Ranks: 0 = best (lowest value)
    true_ranks = np.argsort(np.argsort(y_true))
    pred_ranks = np.argsort(np.argsort(y_pred))

    true_topk = set(np.where(true_ranks < K)[0])
    pred_topk = set(np.where(pred_ranks < K)[0])

    recall = len(true_topk & pred_topk) / K

    # Mean true rank (0-indexed) of runs predicted to be top-K
    mean_true_rank = float(np.mean([true_ranks[i] for i in pred_topk]))
    # Mean predicted rank of runs that are truly top-K
    mean_pred_rank = float(np.mean([pred_ranks[i] for i in true_topk]))

    return {
        f"recall@{K}": recall,
        f"mean_true_rank_of_pred_top{K}": mean_true_rank,
        f"mean_pred_rank_of_true_top{K}": mean_pred_rank,
    }


def cross_validate(
    spec: DatasetSpec,
    model: GeneralModelSpec,
    k: int = 5,
    seed: int = 42,
    topk_values: list[int] | None = None,
) -> dict[str, float]:
    """k-fold CV returning {R², RMSE, MAE, Spearman, Huber, top-K metrics}."""
    if topk_values is None:
        topk_values = []

    rng = np.random.default_rng(seed)
    idx = rng.permutation(spec.R)
    folds = np.array_split(idx, k)

    nan_result: dict[str, float] = {
        "R²": float("nan"),
        "RMSE": float("nan"),
        "MAE": float("nan"),
        "Spearman": float("nan"),
        "Huber": float("nan"),
    }
    for K in topk_values:
        nan_result[f"recall@{K}"] = float("nan")
        nan_result[f"mean_true_rank_of_pred_top{K}"] = float("nan")
        nan_result[f"mean_pred_rank_of_true_top{K}"] = float("nan")

    all_preds = np.full(spec.R, np.nan)
    for fold_i, test_idx in enumerate(folds):
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != fold_i])
        train_spec = spec.subset(train_idx)
        try:
            predict_fn, _info = model.fit_fn(train_spec)
            preds = predict_fn(spec.weights[test_idx])
            all_preds[test_idx] = preds
        except Exception as e:
            log.warning(f"  {model.name} fold {fold_i} failed: {e}")
            return nan_result

    y_true = spec.y
    residuals = y_true - all_preds

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))

    sp_corr, _ = spearmanr(y_true, all_preds)

    # Huber loss (delta=0.01)
    delta = 0.01
    abs_r = np.abs(residuals)
    huber = np.where(abs_r <= delta, 0.5 * residuals**2, delta * abs_r - 0.5 * delta**2)
    huber_mean = float(np.mean(huber))

    result = {"R²": r2, "RMSE": rmse, "MAE": mae, "Spearman": float(sp_corr), "Huber": huber_mean}

    # Top-K metrics on full CV predictions
    for K in topk_values:
        result.update(_topk_metrics(y_true, all_preds, K))

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_datasets() -> list[DatasetSpec]:
    """Construct DatasetSpecs for the three evaluation datasets."""
    # --- single_phase_no_epoch ---
    cc_topics = list(CC_TOPIC_SIZES_B.keys())
    no_epoch_mults = np.array([1.0 / CC_TOPIC_SIZES_B[t] for t in cc_topics])  # ~0.002 each
    sp_no_epoch = load_dataset_spec(
        SCRIPT_DIR / "single_phase_no_epoch.csv",
        target_col="eval/paloma/c4_en/bpb",
        epoch_multipliers=no_epoch_mults,
        small_domains=None,  # no domain is epoched
        name="single_phase_no_epoch",
    )

    # --- single_phase_epoch ---
    epoch_mults = np.array([TOTAL_CC_B / CC_TOPIC_SIZES_B[t] for t in cc_topics])  # 16-22x
    sp_epoch = load_dataset_spec(
        SCRIPT_DIR / "single_phase_epoch.csv",
        target_col="eval/paloma/c4_en/bpb",
        epoch_multipliers=epoch_mults,
        small_domains=list(range(len(cc_topics))),  # all domains are epoched
        name="single_phase_epoch",
    )

    # --- 3_partitions_3_phases ---
    domain_order_3p = ["nemotron_full", "dolmino", "openthoughts_sft"]
    target_3p = 5_700_000_000_000
    # Epoch multipliers per phase: phase_fraction * target / domain_tokens
    # Phase fractions: [0.33, 0.34, 0.33] for boundaries at [0.33, 0.67]
    phase_fracs_3p = np.array([0.33, 0.34, 0.33])
    epoch_mults_3p = np.zeros((3, 3))
    for ki in range(3):
        for di, dom in enumerate(domain_order_3p):
            epoch_mults_3p[ki, di] = phase_fracs_3p[ki] * target_3p / DOMAIN_TOKENS_3P[dom]

    three_phase = load_dataset_spec(
        SCRIPT_DIR / "3_partitions_3_phases_6.csv",
        target_col="lm_eval/arc_challenge/bpb",
        epoch_multipliers=epoch_mults_3p,
        small_domains=[1, 2],  # dolmino (2.3x) + openthoughts_sft (108x per phase)
        name="3_partitions_3_phases",
    )

    return [sp_no_epoch, sp_epoch, three_phase]


def main():
    datasets = build_datasets()

    results: list[dict[str, str | float]] = []

    for spec in datasets:
        log.info(f"\n{'='*70}")
        log.info(f"Dataset: {spec.name} (R={spec.R}, N={spec.N}, M={spec.M})")
        log.info(f"{'='*70}")

        for model in GENERAL_MODELS:
            if not model.applicable(spec):
                log.info(f"  {model.name:25s} — SKIPPED (underdetermined)")
                results.append(
                    {
                        "dataset": spec.name,
                        "model": model.name,
                        "R²": float("nan"),
                        "RMSE": float("nan"),
                        "MAE": float("nan"),
                        "Spearman": float("nan"),
                        "Huber": float("nan"),
                        "status": "skipped",
                    }
                )
                continue

            log.info(
                f"  {model.name:25s} — fitting...",
            )
            metrics = cross_validate(spec, model)
            log.info(
                f"  {model.name:25s}   R²={metrics['R²']:+.4f}  "
                f"RMSE={metrics['RMSE']:.6f}  "
                f"Spearman={metrics['Spearman']:.4f}  "
                f"Huber={metrics['Huber']:.6f}"
            )
            results.append(
                {
                    "dataset": spec.name,
                    "model": model.name,
                    **metrics,
                    "status": "ok" if not np.isnan(metrics["R²"]) else "failed",
                }
            )

    # Build results DataFrame
    df = pd.DataFrame(results)
    csv_path = SCRIPT_DIR / "cross_dataset_results.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"\nResults saved to {csv_path}")

    # Print summary table
    log.info(f"\n{'='*90}")
    log.info("SUMMARY: R² by model × dataset (higher is better)")
    log.info(f"{'='*90}")

    ok_df = df[df["status"] == "ok"]
    dataset_names = [s.name for s in datasets]
    model_names = [m.name for m in GENERAL_MODELS]

    header = f"{'Model':25s}"
    for dn in dataset_names:
        short = dn.replace("single_phase_", "sp_").replace("3_partitions_3_phases", "3p_3d")
        header += f"  {short:>16s}"
    log.info(header)
    log.info("-" * len(header))

    for mn in model_names:
        row = f"{mn:25s}"
        for dn in dataset_names:
            match = ok_df[(ok_df["model"] == mn) & (ok_df["dataset"] == dn)]
            if len(match) == 1:
                r2 = match.iloc[0]["R²"]
                row += f"  {r2:>16.4f}"
            else:
                row += f"  {'—':>16s}"
        log.info(row)

    # Also print RMSE table
    log.info(f"\n{'='*90}")
    log.info("SUMMARY: RMSE by model × dataset (lower is better)")
    log.info(f"{'='*90}")
    log.info(header)
    log.info("-" * len(header))

    for mn in model_names:
        row = f"{mn:25s}"
        for dn in dataset_names:
            match = ok_df[(ok_df["model"] == mn) & (ok_df["dataset"] == dn)]
            if len(match) == 1:
                rmse = match.iloc[0]["RMSE"]
                row += f"  {rmse:>16.6f}"
            else:
                row += f"  {'—':>16s}"
        log.info(row)

    # Print Spearman table
    log.info(f"\n{'='*90}")
    log.info("SUMMARY: Spearman by model × dataset (higher is better)")
    log.info(f"{'='*90}")
    log.info(header)
    log.info("-" * len(header))

    for mn in model_names:
        row = f"{mn:25s}"
        for dn in dataset_names:
            match = ok_df[(ok_df["model"] == mn) & (ok_df["dataset"] == dn)]
            if len(match) == 1:
                sp = match.iloc[0]["Spearman"]
                row += f"  {sp:>16.4f}"
            else:
                row += f"  {'—':>16s}"
        log.info(row)


if __name__ == "__main__":
    main()
