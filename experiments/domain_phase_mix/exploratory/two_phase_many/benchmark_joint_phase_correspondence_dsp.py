# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Benchmark DSP with matched one-phase/two-phase rows fitted jointly.

Corresponding one-phase and two-phase mixtures are assigned to the same CV
fold. Proportional repeats are also grouped together. This prevents leakage and
directly identifies aggregate-mixture response separately from phase ordering.
The remaining two-phase interventions stay external to model fitting.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_nested_coverage_dsp as coverage,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)

PACKET = (
    pooled.REFERENCE_OUTPUTS / "two_phase_solver_gap_collaborator_packet_20260701/data/all_300m_checkpoint_metrics.csv"
)
ONE_PHASE_SOURCE = (
    pooled.REFERENCE_OUTPUTS / "one_phase_swarm_scores_export_300m_20260630/"
    "one_phase_augmented_fit_panel_uncheatable_table9_scores_300m.csv"
)
DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "joint_phase_correspondence_dsp_20260709"
TARGET_COLUMNS = {
    "uncheatable": "eval_uncheatable_eval_bpb",
    "table9": "table9_macro_bpb",
}


def attach_single_phase_weights(
    frame: pd.DataFrame,
    path: Path,
    domains: list[str],
) -> pd.DataFrame:
    source = pd.read_csv(path).copy()
    source["packet_run_name"] = source["run_name"].astype(str)
    deletion = source["panel_source"].eq("domain_deletion")
    source.loc[deletion, "packet_run_name"] = "singleavg_" + source.loc[deletion, "packet_run_name"]
    if source["packet_run_name"].duplicated().any():
        raise ValueError("One-phase source has duplicate packet run names")
    source = source.set_index("packet_run_name")
    out = frame.copy()
    single = out["policy_family"].eq("single_phase")
    for domain in domains:
        values = out.loc[single, "run_name"].map(source[f"weight_{domain}"])
        out.loc[single, f"phase_0_{domain}"] = values.to_numpy()
        out.loc[single, f"phase_1_{domain}"] = values.to_numpy()
    phase_columns = [f"phase_{phase}_{domain}" for phase in (0, 1) for domain in domains]
    if out.loc[single, phase_columns].isna().any(axis=None):
        missing = out.loc[single & out[phase_columns].isna().any(axis=1), "run_name"].tolist()
        raise ValueError(f"Missing reconstructed single-phase weights for {missing[:5]}")
    return out


def dataset_from_frame(name: str, frame: pd.DataFrame, target: str) -> pooled.Dataset:
    reference = pooled.load_300m_dataset(name)
    frame = frame.loc[frame[target].notna()].reset_index(drop=True)
    w0 = frame[[f"phase_0_{domain}" for domain in reference.domain_names]].to_numpy(dtype=float, copy=True)
    w1 = frame[[f"phase_1_{domain}" for domain in reference.domain_names]].to_numpy(dtype=float, copy=True)
    w0 /= w0.sum(axis=1, keepdims=True)
    w1 /= w1.sum(axis=1, keepdims=True)
    return pooled.Dataset(
        name=f"300m_{name}",
        frame=frame,
        y=frame[target].to_numpy(dtype=float),
        weights=np.stack([w0, w1], axis=1),
        c0=reference.c0,
        c1=reference.c1,
        domain_names=reference.domain_names,
    )


def grouped_folds(frame: pd.DataFrame, seed: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = frame["phase_correspondence_key"].astype(str).to_numpy()
    unique = np.unique(groups)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_groups, test_groups in splitter.split(unique):
        train_set = set(unique[train_groups])
        test_set = set(unique[test_groups])
        train = np.flatnonzero(np.fromiter((group in train_set for group in groups), dtype=bool))
        test = np.flatnonzero(np.fromiter((group in test_set for group in groups), dtype=bool))
        folds.append((train, test))
    return folds


def external_metrics(name: str, target: np.ndarray, prediction: np.ndarray) -> dict[str, float | str]:
    residual = prediction - target
    selected = int(np.argmin(prediction))
    tail_count = max(5, int(np.ceil(0.15 * len(target))))
    tail = np.argsort(prediction)[:tail_count]
    return {
        "model": name,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "spearman": float(spearmanr(target, prediction).statistic),
        "regret_at_1": float(target[selected] - np.min(target)),
        "selected_observed": float(target[selected]),
        "selected_predicted": float(prediction[selected]),
        "lower_tail_optimism": float(np.mean(np.maximum(-residual[tail], 0.0))),
        "low_tail_rmse": float(np.sqrt(np.mean(residual[tail] ** 2))),
    }


def benchmark_panel(
    dataset: pooled.Dataset,
    panel_name: str,
    seeds: list[int],
    n_splits: int,
    maxiter: int,
    coarse_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = (
        coverage.FitConfig("effective_exposure", False),
        coverage.FitConfig("effective_exposure_coverage", True),
    )
    alpha0, alpha1 = coverage.phase_fractions(dataset)
    rows = []
    parameters = []
    for seed in seeds:
        folds = grouped_folds(dataset.frame, seed, n_splits)
        predictions = {config.name: np.zeros(dataset.n, dtype=float) for config in configs}
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            print(
                f"{dataset.name}/{panel_name}: seed={seed} fold={fold_id + 1}/{n_splits}",
                flush=True,
            )
            for config in configs:
                model = coverage.fit_model(
                    dataset,
                    train_idx,
                    config,
                    linear_reg=coverage.dataset_linear_reg(dataset),
                    maxiter=maxiter,
                    coarse_top_k=coarse_top_k,
                )
                predictions[config.name][test_idx] = coverage.predict(model, dataset.weights[test_idx], alpha0, alpha1)
                parameters.append(
                    {
                        "dataset": dataset.name,
                        "fit_panel": panel_name,
                        "model": config.name,
                        "seed": seed,
                        "fold": fold_id,
                        "gamma": float(model.base.params["gamma"]),
                        "theta_tv": float(model.coverage_coef[0]) if len(model.coverage_coef) else 0.0,
                        "theta_hhi_aggregate": float(model.coverage_coef[1]) if len(model.coverage_coef) else 0.0,
                        "theta_hhi_phase1": float(model.coverage_coef[2]) if len(model.coverage_coef) else 0.0,
                    }
                )
        for config in configs:
            row = asdict(pooled.metrics(dataset, config.name, seed, predictions[config.name], folds))
            row["fit_panel"] = panel_name
            row["n_groups"] = int(dataset.frame["phase_correspondence_key"].nunique())
            row["nominal_param_count"] = 4 * dataset.m + 2 + 3 * int(config.use_coverage)
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(parameters)


def evaluate_external(
    fit_dataset: pooled.Dataset,
    external_dataset: pooled.Dataset,
    panel_name: str,
    maxiter: int,
    coarse_top_k: int,
) -> pd.DataFrame:
    alpha0, alpha1 = coverage.phase_fractions(fit_dataset)
    all_indices = np.arange(fit_dataset.n)
    rows = []
    for config in (
        coverage.FitConfig("effective_exposure", False),
        coverage.FitConfig("effective_exposure_coverage", True),
    ):
        model = coverage.fit_model(
            fit_dataset,
            all_indices,
            config,
            linear_reg=coverage.dataset_linear_reg(fit_dataset),
            maxiter=maxiter,
            coarse_top_k=coarse_top_k,
        )
        prediction = coverage.predict(model, external_dataset.weights, alpha0, alpha1)
        row = external_metrics(config.name, external_dataset.y, prediction)
        row["dataset"] = fit_dataset.name
        row["fit_panel"] = panel_name
        row["external_rows"] = external_dataset.n
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=PACKET)
    parser.add_argument("--one-phase-source", type=Path, default=ONE_PHASE_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--objectives", default="uncheatable,table9")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=16)
    parser.add_argument("--coarse-top-k", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.packet)
    domains = pooled.load_300m_dataset("table9").domain_names
    frame = attach_single_phase_weights(frame, args.one_phase_source, domains)
    seeds = pooled.parse_int_list(args.seeds)
    objectives = [part.strip() for part in args.objectives.split(",") if part.strip()]
    unknown = sorted(set(objectives).difference(TARGET_COLUMNS))
    if unknown:
        raise ValueError(f"Unknown objectives: {unknown}")
    cv_frames = []
    parameter_frames = []
    external_frames = []
    for objective in objectives:
        target = TARGET_COLUMNS[objective]
        original_frame = frame.loc[frame["split"].eq("train")].copy()
        joint_frame = frame.loc[frame["split"].eq("train") | frame["policy_family"].eq("single_phase")].copy()
        external_frame = frame.loc[frame["split"].eq("heldout") & frame["policy_family"].eq("two_phase")].copy()
        original = dataset_from_frame(objective, original_frame, target)
        joint = dataset_from_frame(objective, joint_frame, target)
        external = dataset_from_frame(objective, external_frame, target)
        for panel_name, dataset in (("original_290", original), ("joint_matched", joint)):
            metrics, parameters = benchmark_panel(
                dataset,
                panel_name,
                seeds,
                args.n_splits,
                args.maxiter,
                args.coarse_top_k,
            )
            cv_frames.append(metrics)
            parameter_frames.append(parameters)
            external_frames.append(evaluate_external(dataset, external, panel_name, args.maxiter, args.coarse_top_k))
    raw = pd.concat(cv_frames, ignore_index=True)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    external = pd.concat(external_frames, ignore_index=True)
    summary = raw.groupby(["dataset", "fit_panel", "model"], as_index=False).agg(
        n_rows=("n_rows", "first"),
        n_groups=("n_groups", "first"),
        n_cv_seeds=("seed", "nunique"),
        oof_rmse_mean=("oof_rmse", "mean"),
        oof_rmse_std=("oof_rmse", "std"),
        oof_spearman_mean=("oof_spearman", "mean"),
        oof_spearman_std=("oof_spearman", "std"),
        fold_mean_regret_at_1_mean=("fold_mean_regret_at_1", "mean"),
        lower_tail_optimism_mean=("lower_tail_optimism", "mean"),
        low_tail_rmse_mean=("low_tail_rmse", "mean"),
    )
    raw.to_csv(args.output_dir / "cv_metrics_by_seed.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    parameters.to_csv(args.output_dir / "fold_parameters.csv", index=False)
    external.to_csv(args.output_dir / "external_two_phase_heldout_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(external.to_string(index=False))
    print(f"Wrote joint phase-correspondence benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
