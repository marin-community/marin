# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Benchmark partially pooled two-phase exposure bowls across data-mixing swarms.

The separate-heads model fits an independent preferred-exposure curve to each
phase. This script tests a hierarchical version that shrinks phase-specific
response coefficients toward a shared response:

    L(w) = b + D_0(w) beta_0 + D_1(w) beta_1,

where ``D_p`` contains the lower and upper halves of a two-sided quadratic bowl
in ``log(1 + phase_p_exposure)``. The linear head minimizes

    ||y - L(w)||^2 + lambda_base (||beta_0||^2 + ||beta_1||^2)
                         + lambda_phase ||beta_0 - beta_1||^2,

subject to nonnegative coefficients. ``lambda_phase=0`` recovers standardized
separate heads; increasing it spends fewer phase-specific degrees of freedom.

The benchmark refits exposure centers and heads inside every fold. It covers:

* 300M Uncheatable BPB;
* 300M OLMoBaseEval Table-9 macro BPB;
* the 840-row production Grug-MoE swarm's Uncheatable BPB.

This is a local model-selection diagnostic. It does not submit training jobs.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_olmo_base_easy_per_component_dsp_decision_300m as component_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_two_phase_canonical_bowl_candidates_300m as bowl,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "partially_pooled_phase_bowls_20260709"
PRODUCTION_DATA = REFERENCE_OUTPUTS / "grug_moe_production_swarm_results_20260704/production_swarm_840_wide.csv"
PRODUCTION_MODEL = REFERENCE_OUTPUTS / "grug_moe_production_swarm_effective_exposure_dsp_uncheatable_20260705/model.json"
PRODUCTION_TARGET = "eval/uncheatable_eval/bpb"
LOWER_TAIL_FRAC = 0.15
MU_SHIFTS = np.linspace(-2.0, 2.0, 9)
SCALE_FLOOR = 1e-8
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class Dataset:
    name: str
    frame: pd.DataFrame
    y: np.ndarray
    weights: np.ndarray
    c0: np.ndarray
    c1: np.ndarray
    domain_names: list[str]

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def m(self) -> int:
        return len(self.domain_names)


@dataclass(frozen=True)
class HeadConfig:
    name: str
    base_l2: float
    phase_l2: float
    standardize: bool


@dataclass(frozen=True)
class FittedHead:
    config: HeadConfig
    mu0: np.ndarray
    mu1: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    intercept: float
    coef0: np.ndarray
    coef1: np.ndarray


@dataclass(frozen=True)
class MetricRow:
    dataset: str
    model: str
    seed: int
    n_rows: int
    n_domains: int
    nominal_param_count: int
    oof_rmse: float
    oof_spearman: float
    fold_mean_regret_at_1: float
    global_regret_at_1: float
    lower_tail_optimism: float
    low_tail_rmse: float


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def load_300m_dataset(objective: str) -> Dataset:
    packet, _domains, _natural, _token_counts, _target_budget, _folds = bowl.load_objective(objective)
    return Dataset(
        name=f"300m_{objective}",
        frame=packet.frame.copy(),
        y=np.asarray(packet.y, dtype=float),
        weights=np.asarray(packet.w, dtype=float),
        c0=np.asarray(packet.c0, dtype=float),
        c1=np.asarray(packet.c1, dtype=float),
        domain_names=list(packet.domain_names),
    )


def load_production_dataset() -> Dataset:
    frame = pd.read_csv(PRODUCTION_DATA)
    model = json.loads(PRODUCTION_MODEL.read_text())
    domains = list(model["domain_names"])
    w0 = frame[[f"phase_0/{domain}" for domain in domains]].to_numpy(dtype=float)
    w1 = frame[[f"phase_1/{domain}" for domain in domains]].to_numpy(dtype=float)
    w0 /= w0.sum(axis=1, keepdims=True)
    w1 /= w1.sum(axis=1, keepdims=True)
    return Dataset(
        name="production_uncheatable",
        frame=frame,
        y=frame[PRODUCTION_TARGET].to_numpy(dtype=float),
        weights=np.stack([w0, w1], axis=1),
        c0=np.asarray(model["c0"], dtype=float),
        c1=np.asarray(model["c1"], dtype=float),
        domain_names=domains,
    )


def dataset_folds(dataset: Dataset, seed: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if dataset.name.startswith("300m_"):
        return component_dsp.panel_stratified_folds(dataset.frame, n_splits=n_splits, seed=seed)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(train, test) for train, test in splitter.split(np.arange(dataset.n))]


def phase_exposures(dataset: Dataset, row_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = dataset.weights[row_indices]
    return weights[:, 0, :] * dataset.c0[None, :], weights[:, 1, :] * dataset.c1[None, :]


def base_mu(exposure: np.ndarray) -> np.ndarray:
    logged = np.log1p(np.where(exposure > 1e-8, exposure, np.nan))
    median = np.nanmedian(logged, axis=0)
    return np.clip(np.where(np.isfinite(median), median, 0.0), -2.0, 8.0)


def bowl_design(exposure: np.ndarray, mu: np.ndarray) -> np.ndarray:
    delta = np.log1p(exposure) - mu[None, :]
    return np.hstack([np.minimum(delta, 0.0) ** 2, np.maximum(delta, 0.0) ** 2])


def fit_raw_nnls(design: np.ndarray, y: np.ndarray, l2: float) -> tuple[float, np.ndarray]:
    design_mean = design.mean(axis=0, keepdims=True)
    y_mean = float(y.mean())
    centered_design = design - design_mean
    centered_target = y - y_mean
    if l2 > 0.0:
        centered_design = np.vstack([centered_design, np.sqrt(l2) * np.eye(design.shape[1])])
        centered_target = np.concatenate([centered_target, np.zeros(design.shape[1])])
    coef, _residual = nnls(centered_design, centered_target, maxiter=20 * design.shape[1])
    intercept = y_mean - float((design_mean @ coef).item())
    return intercept, coef


def selected_mu(exposure: np.ndarray, y: np.ndarray) -> np.ndarray:
    median = base_mu(exposure)
    best_rmse = np.inf
    best = median
    for shift in MU_SHIFTS:
        mu = np.clip(median + shift, -2.0, 8.0)
        design = bowl_design(exposure, mu)
        intercept, coef = fit_raw_nnls(design, y, l2=0.1)
        prediction = intercept + design @ coef
        rmse = float(np.sqrt(np.mean((prediction - y) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best = mu
    return best


def scaled_phase_designs(
    design0: np.ndarray,
    design1: np.ndarray,
    *,
    standardize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not standardize:
        return design0, design1, np.ones(design0.shape[1], dtype=float)
    pair_scale = np.sqrt(np.mean(np.vstack([design0, design1]) ** 2, axis=0))
    pair_scale = np.maximum(pair_scale, SCALE_FLOOR)
    return design0 / pair_scale[None, :], design1 / pair_scale[None, :], pair_scale


def fit_head(
    exposure0: np.ndarray,
    exposure1: np.ndarray,
    y: np.ndarray,
    config: HeadConfig,
) -> FittedHead:
    mu0 = selected_mu(exposure0, y)
    mu1 = selected_mu(exposure1, y)
    design0, design1, scale = scaled_phase_designs(
        bowl_design(exposure0, mu0),
        bowl_design(exposure1, mu1),
        standardize=config.standardize,
    )
    design = np.hstack([design0, design1])
    center = design.mean(axis=0)
    target_mean = float(y.mean())
    augmented_design = design - center[None, :]
    augmented_target = y - target_mean
    num_phase_features = design0.shape[1]
    num_features = design.shape[1]
    if config.base_l2 > 0.0:
        augmented_design = np.vstack([augmented_design, np.sqrt(config.base_l2) * np.eye(num_features)])
        augmented_target = np.concatenate([augmented_target, np.zeros(num_features)])
    if config.phase_l2 > 0.0:
        difference = np.hstack([np.eye(num_phase_features), -np.eye(num_phase_features)])
        augmented_design = np.vstack([augmented_design, np.sqrt(config.phase_l2) * difference])
        augmented_target = np.concatenate([augmented_target, np.zeros(num_phase_features)])
    coef, _residual = nnls(augmented_design, augmented_target, maxiter=20 * num_features)
    intercept = target_mean - float(center @ coef)
    return FittedHead(
        config=config,
        mu0=mu0,
        mu1=mu1,
        center=center,
        scale=scale,
        intercept=intercept,
        coef0=coef[:num_phase_features],
        coef1=coef[num_phase_features:],
    )


def predict(model: FittedHead, exposure0: np.ndarray, exposure1: np.ndarray) -> np.ndarray:
    design0 = bowl_design(exposure0, model.mu0) / model.scale[None, :]
    design1 = bowl_design(exposure1, model.mu1) / model.scale[None, :]
    return model.intercept + design0 @ model.coef0 + design1 @ model.coef1


def metrics(
    dataset: Dataset,
    model_name: str,
    seed: int,
    prediction: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> MetricRow:
    residual = prediction - dataset.y
    rmse = float(np.sqrt(np.mean(residual**2)))
    spearman = float(spearmanr(dataset.y, prediction).statistic)
    fold_regrets = []
    for _train_idx, test_idx in folds:
        selected = test_idx[int(np.argmin(prediction[test_idx]))]
        fold_regrets.append(float(dataset.y[selected] - np.min(dataset.y[test_idx])))
    tail_count = max(5, int(np.ceil(LOWER_TAIL_FRAC * dataset.n)))
    tail_idx = np.argsort(prediction)[:tail_count]
    tail_residual = prediction[tail_idx] - dataset.y[tail_idx]
    return MetricRow(
        dataset=dataset.name,
        model=model_name,
        seed=seed,
        n_rows=dataset.n,
        n_domains=dataset.m,
        nominal_param_count=4 * dataset.m + 3,
        oof_rmse=rmse,
        oof_spearman=spearman,
        fold_mean_regret_at_1=float(np.mean(fold_regrets)),
        global_regret_at_1=float(dataset.y[int(np.argmin(prediction))] - np.min(dataset.y)),
        lower_tail_optimism=float(np.mean(np.maximum(-tail_residual, 0.0))),
        low_tail_rmse=float(np.sqrt(np.mean(tail_residual**2))),
    )


def benchmark_dataset(
    dataset: Dataset,
    configs: list[HeadConfig],
    seeds: list[int],
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for seed in seeds:
        folds = dataset_folds(dataset, seed, n_splits)
        oof = {config.name: np.zeros(dataset.n, dtype=float) for config in configs}
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            print(f"{dataset.name}: seed={seed} fold={fold_id + 1}/{n_splits}", flush=True)
            train_e0, train_e1 = phase_exposures(dataset, train_idx)
            test_e0, test_e1 = phase_exposures(dataset, test_idx)
            for config in configs:
                model = fit_head(train_e0, train_e1, dataset.y[train_idx], config)
                oof[config.name][test_idx] = predict(model, test_e0, test_e1)
        full_e0, full_e1 = phase_exposures(dataset, np.arange(dataset.n))
        for config in configs:
            row = metrics(dataset, config.name, seed, oof[config.name], folds)
            metric_rows.append(asdict(row))
            full_model = fit_head(full_e0, full_e1, dataset.y, config)
            common_norm = float(np.linalg.norm(0.5 * (full_model.coef0 + full_model.coef1)))
            contrast_norm = float(np.linalg.norm(full_model.coef1 - full_model.coef0))
            coefficient_rows.append(
                {
                    "dataset": dataset.name,
                    "model": config.name,
                    "seed": seed,
                    "common_coef_norm": common_norm,
                    "phase_contrast_norm": contrast_norm,
                    "contrast_to_common_ratio": contrast_norm / max(common_norm, 1e-12),
                    "nonzero_phase0": int(np.sum(full_model.coef0 > 1e-10)),
                    "nonzero_phase1": int(np.sum(full_model.coef1 > 1e-10)),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(coefficient_rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metrics_to_aggregate = [
        "oof_rmse",
        "oof_spearman",
        "fold_mean_regret_at_1",
        "global_regret_at_1",
        "lower_tail_optimism",
        "low_tail_rmse",
    ]
    grouped = raw.groupby(["dataset", "model"], sort=True)
    rows = []
    for (dataset, model), frame in grouped:
        row: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "n_rows": int(frame["n_rows"].iloc[0]),
            "n_domains": int(frame["n_domains"].iloc[0]),
            "nominal_param_count": int(frame["nominal_param_count"].iloc[0]),
            "n_cv_seeds": len(frame),
        }
        for metric in metrics_to_aggregate:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_std"] = float(frame[metric].std(ddof=1)) if len(frame) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def write_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    for metric, title in (
        ("oof_spearman_mean", "OOF Spearman across phase-head pooling strengths"),
        ("oof_rmse_mean", "OOF RMSE across phase-head pooling strengths"),
        ("fold_mean_regret_at_1_mean", "Fold mean Regret@1 across phase-head pooling strengths"),
        ("lower_tail_optimism_mean", "Lower-tail optimism across phase-head pooling strengths"),
    ):
        figure = px.bar(
            summary,
            x="model",
            y=metric,
            color="dataset",
            barmode="group",
            color_discrete_sequence=["#1a9850", "#fee08b", "#d73027"],
            title=title,
        )
        figure.update_layout(xaxis_title="Model", yaxis_title=metric.removesuffix("_mean"), legend_title="Dataset")
        figure.write_html(
            output_dir / f"{metric.removesuffix('_mean')}.html",
            include_plotlyjs="cdn",
            config=PLOT_CONFIG,
        )


def configs(base_l2: float, phase_l2_values: list[float]) -> list[HeadConfig]:
    values = [
        HeadConfig(name="legacy_separate_l2_0p1", base_l2=0.1, phase_l2=0.0, standardize=False),
        HeadConfig(name=f"standardized_separate_l2_{base_l2:g}", base_l2=base_l2, phase_l2=0.0, standardize=True),
    ]
    values.extend(
        HeadConfig(
            name=f"partially_pooled_l2_{base_l2:g}_phase_{phase_l2:g}",
            base_l2=base_l2,
            phase_l2=phase_l2,
            standardize=True,
        )
        for phase_l2 in phase_l2_values
        if phase_l2 > 0.0
    )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", default="300m_uncheatable,300m_table9,production_uncheatable")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--base-l2", type=float, default=0.1)
    parser.add_argument("--phase-l2-values", default="0.01,0.1,1,10,100")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    available = {
        "300m_uncheatable": lambda: load_300m_dataset("uncheatable"),
        "300m_table9": lambda: load_300m_dataset("table9"),
        "production_uncheatable": load_production_dataset,
    }
    selected_names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    unknown = sorted(set(selected_names).difference(available))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    model_configs = configs(args.base_l2, parse_float_list(args.phase_l2_values))
    seeds = parse_int_list(args.seeds)

    raw_frames = []
    coefficient_frames = []
    for name in selected_names:
        dataset = available[name]()
        raw, coefficients = benchmark_dataset(dataset, model_configs, seeds, args.n_splits)
        raw_frames.append(raw)
        coefficient_frames.append(coefficients)

    raw = pd.concat(raw_frames, ignore_index=True)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    summary = summarize(raw)
    raw.to_csv(args.output_dir / "cv_metrics_by_seed.csv", index=False)
    coefficients.to_csv(args.output_dir / "coefficient_diagnostics.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    write_plots(summary, args.output_dir)
    print(summary.to_string(index=False))
    print(f"Wrote benchmark artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
