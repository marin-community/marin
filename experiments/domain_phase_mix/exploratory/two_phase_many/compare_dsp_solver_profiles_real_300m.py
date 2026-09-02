# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Compare DSP solver profiles on real 300M swarm observations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_dsp_synthetic_recovery_167p import (
    SOLVER_PROFILES,
    fit_variant_with_profile,
    variant_by_name,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dsp_canonical_variants_300m import (
    SCALE,
    _load_packet,
    _pack_params,
    _predict,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import PacketData

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "dsp_solver_real_cv_300m_20260608"
DEFAULT_VARIANT = "dsp_effective_exposure_penalty_nnls"
DEFAULT_PROFILES = "current,coarse_only_nnls20,coarse_only_signed_ridge_alpha1"


def progress(message: str) -> None:
    """Print a timestamped progress line."""

    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_profiles(value: str) -> tuple[str, ...]:
    """Parse profile schedule."""

    profiles = tuple(part.strip() for part in value.split(",") if part.strip())
    if not profiles:
        raise ValueError("At least one profile is required")
    unknown = sorted(set(profiles) - set(SOLVER_PROFILES))
    if unknown:
        raise ValueError(f"Unknown profiles: {unknown}")
    return profiles


def subset_packet(packet: PacketData, indices: np.ndarray) -> PacketData:
    """Return a subset packet for fitting."""

    return PacketData(
        frame=packet.frame.iloc[indices].reset_index(drop=True).copy(),
        name_col=packet.name_col,
        y=np.asarray(packet.y[indices], dtype=float),
        w=np.asarray(packet.w[indices], dtype=float),
        m=packet.m,
        c0=packet.c0,
        c1=packet.c1,
        domain_names=packet.domain_names,
    )


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Spearman correlation with NaN mapped to zero."""

    value = float(spearmanr(y_true, y_pred).statistic)
    if np.isnan(value):
        return 0.0
    return value


def fold_metrics(
    packet: PacketData,
    profile_name: str,
    variant_name: str,
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit one profile on a fold and return held-out metrics."""

    profile = SOLVER_PROFILES[profile_name]
    variant = variant_by_name(variant_name)
    train_packet = subset_packet(packet, train_idx)
    start_time = time.time()
    model, trace = fit_variant_with_profile(train_packet, variant, profile, seed=fold)
    elapsed = time.time() - start_time
    pred_test = _predict(model, packet.w[test_idx], packet)
    y_test = packet.y[test_idx]
    pred_order = np.argsort(pred_test)
    true_order = np.argsort(y_test)
    chosen_local = int(pred_order[0])
    chosen_global = int(test_idx[chosen_local])
    best_local = int(true_order[0])
    best_global = int(test_idx[best_local])
    row = {
        "profile": profile_name,
        "variant": variant_name,
        "fold": int(fold),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "heldout_spearman": safe_spearman(y_test, pred_test),
        "heldout_rmse": float(np.sqrt(np.mean((pred_test - y_test) ** 2))),
        "heldout_regret_at_1": float(y_test[chosen_local] - y_test[best_local]),
        "heldout_chosen_run": str(packet.frame.iloc[chosen_global][packet.name_col]),
        "heldout_best_run": str(packet.frame.iloc[best_global][packet.name_col]),
        "heldout_chosen_rank": int(np.where(true_order == chosen_local)[0][0] + 1),
        "elapsed_sec": float(elapsed),
        "linear_reg": float(model.params.get("_linear_reg", 1e-6)),
        "linear_head_param_count": int(model.params.get("_linear_head_param_count", 2 * packet.m)),
        "benefit_zero_count": int(np.sum(np.isclose(model.benefit_coef, 0.0, atol=1e-10))),
        "penalty_zero_count": int(np.sum(np.isclose(model.penalty_coef, 0.0, atol=1e-10))),
        "nonlinear_param_count": int(len(_pack_params(model.params, model.variant))),
    }
    trace = trace.assign(profile=profile_name, variant=variant_name, fold=int(fold))
    return row, trace


def full_fit_boundary_metrics(packet: PacketData, profile_name: str, variant_name: str) -> dict[str, Any]:
    """Fit on all rows and report coefficient boundary diagnostics."""

    profile = SOLVER_PROFILES[profile_name]
    variant = variant_by_name(variant_name)
    model, _trace = fit_variant_with_profile(packet, variant, profile, seed=0)
    return {
        "profile": profile_name,
        "variant": variant_name,
        "fit_size": int(len(packet.y)),
        "benefit_zero_count": int(np.sum(np.isclose(model.benefit_coef, 0.0, atol=1e-10))),
        "penalty_zero_count": int(np.sum(np.isclose(model.penalty_coef, 0.0, atol=1e-10))),
        "benefit_negative_count": int(np.sum(model.benefit_coef < 0.0)),
        "penalty_negative_count": int(np.sum(model.penalty_coef < 0.0)),
        "linear_reg": float(model.params.get("_linear_reg", 1e-6)),
        "linear_head_param_count": int(model.params.get("_linear_head_param_count", 2 * packet.m)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--profiles", default=DEFAULT_PROFILES)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    """Run real-data cross-validation."""

    args = build_arg_parser().parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = parse_profiles(args.profiles)
    packet = _load_packet()
    progress(f"Loaded {SCALE} packet rows={len(packet.y)} domains={packet.m} target=eval/uncheatable_eval/bpb")
    rows: list[dict[str, Any]] = []
    traces: list[pd.DataFrame] = []
    kf = KFold(n_splits=args.splits, shuffle=True, random_state=args.seed)
    for profile_name in profiles:
        for fold, (train_idx, test_idx) in enumerate(kf.split(packet.w)):
            progress(f"Fitting profile={profile_name} fold={fold} train={len(train_idx)} test={len(test_idx)}")
            row, trace = fold_metrics(packet, profile_name, args.variant, fold, train_idx, test_idx)
            rows.append(row)
            traces.append(trace)
            pd.DataFrame.from_records(rows).to_csv(output_dir / "real_cv_fold_metrics.csv", index=False)
            pd.concat(traces, ignore_index=True).to_csv(output_dir / "real_cv_solver_trace.csv", index=False)

    boundary_rows = [full_fit_boundary_metrics(packet, profile_name, args.variant) for profile_name in profiles]
    fold_frame = pd.DataFrame.from_records(rows)
    boundary_frame = pd.DataFrame.from_records(boundary_rows)
    fold_frame.to_csv(output_dir / "real_cv_fold_metrics.csv", index=False)
    boundary_frame.to_csv(output_dir / "real_cv_boundary_metrics.csv", index=False)
    summary = (
        fold_frame.groupby(["profile", "variant"])
        .agg(
            folds=("fold", "size"),
            heldout_spearman_mean=("heldout_spearman", "mean"),
            heldout_spearman_min=("heldout_spearman", "min"),
            heldout_regret_mean=("heldout_regret_at_1", "mean"),
            heldout_regret_max=("heldout_regret_at_1", "max"),
            heldout_rmse_mean=("heldout_rmse", "mean"),
            elapsed_mean=("elapsed_sec", "mean"),
            fold_linear_head_param_count_median=("linear_head_param_count", "median"),
        )
        .reset_index()
    )
    summary = summary.merge(boundary_frame, on=["profile", "variant"], how="left")
    summary.to_csv(output_dir / "real_cv_summary.csv", index=False)
    (output_dir / "run_spec.json").write_text(
        json.dumps(
            {
                "scale": SCALE,
                "target": "eval/uncheatable_eval/bpb",
                "variant": args.variant,
                "profiles": list(profiles),
                "splits": int(args.splits),
                "seed": int(args.seed),
                "rows": int(len(packet.y)),
                "domains": int(packet.m),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    progress(f"Wrote {output_dir / 'real_cv_summary.csv'}")


if __name__ == "__main__":
    main()
