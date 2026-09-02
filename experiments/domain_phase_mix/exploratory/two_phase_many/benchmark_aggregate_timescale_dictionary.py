# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Test one versus two shared saturation timescales on constant mixtures only."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_joint_phase_correspondence_dsp as joint,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_nested_coverage_dsp as geometry,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_staged_mechanistic_phase_zoo as staged,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import (  # noqa: E402
    dsp_exact as dsp,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "aggregate_timescale_dictionary_20260710"


@dataclass(frozen=True)
class TwoTimescaleModel:
    rho_slow: float
    rho_fast: float
    tau: np.ndarray
    intercept: float
    slow_coef: np.ndarray
    fast_coef: np.ndarray
    penalty_coef: np.ndarray
    hhi_coef: float


def features(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    rho_slow: float,
    rho_fast: float,
    tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    e0, e1 = staged.phase_exposures(dataset, indices)
    exposure = e0 + e1
    slow = 1.0 - np.exp(-rho_slow * exposure)
    fast = 1.0 - np.exp(-rho_fast * exposure)
    penalty = dsp.softplus(np.log1p(exposure) - tau[None, :]) ** 2
    hhi = np.sum(staged.aggregate_weights(dataset, indices) ** 2, axis=1)
    return slow, fast, penalty, hhi


def fit_head(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    rho_slow: float,
    rho_fast: float,
    tau: np.ndarray,
) -> TwoTimescaleModel:
    slow, fast, penalty, hhi = features(dataset, indices, rho_slow, rho_fast, tau)
    design = np.column_stack([-slow, -fast, penalty, hhi])
    intercept, coef = staged.fit_nonnegative_head(
        design,
        dataset.y[indices],
        geometry.dataset_linear_reg(dataset),
    )
    m = dataset.m
    return TwoTimescaleModel(
        rho_slow=rho_slow,
        rho_fast=rho_fast,
        tau=np.asarray(tau, dtype=float),
        intercept=intercept,
        slow_coef=coef[:m],
        fast_coef=coef[m : 2 * m],
        penalty_coef=coef[2 * m : 3 * m],
        hhi_coef=float(coef[-1]),
    )


def predict(model: TwoTimescaleModel, dataset: pooled.Dataset, indices: np.ndarray) -> np.ndarray:
    slow, fast, penalty, hhi = features(
        dataset,
        indices,
        model.rho_slow,
        model.rho_fast,
        model.tau,
    )
    return np.asarray(
        model.intercept
        - slow @ model.slow_coef
        - fast @ model.fast_coef
        + penalty @ model.penalty_coef
        + model.hhi_coef * hhi,
        dtype=float,
    )


def profile_objective(theta: np.ndarray, dataset: pooled.Dataset, indices: np.ndarray) -> float:
    rates = np.sort(np.exp(np.clip(theta[:2], np.log(staged.RHO_MIN), np.log(staged.RHO_MAX))))
    tau = np.clip(theta[2:], staged.TAU_MIN, staged.TAU_MAX)
    model = fit_head(dataset, indices, float(rates[0]), float(rates[1]), tau)
    prediction = predict(model, dataset, indices)
    residual = prediction - dataset.y[indices]
    tail_count = max(5, int(np.ceil(staged.LOWER_TAIL_FRAC * len(indices))))
    tail = np.argsort(prediction)[:tail_count]
    optimism = float(np.mean(np.maximum(-residual[tail], 0.0)))
    return float(np.sqrt(np.mean(residual**2))) + 0.5 * optimism


def fit_two_timescale(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    one_timescale: staged.AggregateModel,
    maxiter: int,
) -> TwoTimescaleModel:
    starts = []
    for slow_ratio, fast_ratio in ((0.1, 1.0), (0.25, 2.0), (0.5, 4.0), (1.0, 8.0)):
        rates = np.clip(
            [one_timescale.rho * slow_ratio, one_timescale.rho * fast_ratio],
            staged.RHO_MIN,
            staged.RHO_MAX,
        )
        starts.append(np.concatenate([np.log(rates), one_timescale.tau]))
    scored = sorted((profile_objective(start, dataset, indices), start) for start in starts)
    best_value, best_theta = scored[0]
    bounds = [
        (np.log(staged.RHO_MIN), np.log(staged.RHO_MAX)),
        (np.log(staged.RHO_MIN), np.log(staged.RHO_MAX)),
        *[(staged.TAU_MIN, staged.TAU_MAX)] * dataset.m,
    ]
    if maxiter > 0:
        result = minimize(
            lambda theta: profile_objective(np.asarray(theta, dtype=float), dataset, indices),
            best_theta,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-7, "maxls": 20},
        )
        if float(result.fun) < best_value:
            best_theta = np.asarray(result.x, dtype=float)
    rates = np.sort(np.exp(best_theta[:2]))
    return fit_head(
        dataset,
        indices,
        float(rates[0]),
        float(rates[1]),
        np.asarray(best_theta[2:], dtype=float),
    )


def regression_row(
    dataset: pooled.Dataset,
    model: str,
    seed: int,
    target: np.ndarray,
    prediction: np.ndarray,
    parameter_count: int,
) -> dict[str, float | int | str]:
    residual = prediction - target
    tail_count = max(5, int(np.ceil(staged.LOWER_TAIL_FRAC * len(target))))
    tail = np.argsort(prediction)[:tail_count]
    return {
        "dataset": dataset.name,
        "model": model,
        "seed": seed,
        "n_rows": len(target),
        "nominal_param_count": parameter_count,
        "oof_rmse": float(np.sqrt(np.mean(residual**2))),
        "oof_spearman": float(spearmanr(target, prediction).statistic),
        "lower_tail_optimism": float(np.mean(np.maximum(-residual[tail], 0.0))),
        "low_tail_rmse": float(np.sqrt(np.mean(residual[tail] ** 2))),
    }


def benchmark_dataset(
    dataset: pooled.Dataset,
    seeds: list[int],
    n_splits: int,
    maxiter: int,
    coarse_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    parameter_rows = []
    for seed in seeds:
        folds = joint.grouped_folds(dataset.frame, seed, n_splits)
        single = dataset.frame["policy_family"].eq("single_phase").to_numpy()
        oof_one = np.full(dataset.n, np.nan, dtype=float)
        oof_two = np.full(dataset.n, np.nan, dtype=float)
        for fold_id, (train_indices, test_indices) in enumerate(folds):
            train_single = train_indices[single[train_indices]]
            test_single = test_indices[single[test_indices]]
            one = staged.fit_aggregate_model(dataset, train_single, maxiter, coarse_top_k)
            two = fit_two_timescale(dataset, train_single, one, maxiter)
            oof_one[test_single] = staged.aggregate_prediction(one, dataset, test_single)
            oof_two[test_single] = predict(two, dataset, test_single)
            parameter_rows.append(
                {
                    "dataset": dataset.name,
                    "seed": seed,
                    "fold": fold_id,
                    "one_rho": one.rho,
                    "two_rho_slow": two.rho_slow,
                    "two_rho_fast": two.rho_fast,
                    "active_slow_coef": int(np.sum(two.slow_coef > 1e-10)),
                    "active_fast_coef": int(np.sum(two.fast_coef > 1e-10)),
                }
            )
        selected = np.flatnonzero(single)
        metric_rows.extend(
            [
                regression_row(
                    dataset,
                    "one_timescale_shared_rho",
                    seed,
                    dataset.y[selected],
                    oof_one[selected],
                    3 * dataset.m + 3,
                ),
                regression_row(
                    dataset,
                    "two_timescale_dictionary",
                    seed,
                    dataset.y[selected],
                    oof_two[selected],
                    4 * dataset.m + 4,
                ),
            ]
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(parameter_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=joint.PACKET)
    parser.add_argument("--one-phase-source", type=Path, default=joint.ONE_PHASE_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", default="uncheatable,table9")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=12)
    parser.add_argument("--coarse-top-k", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.packet)
    domains = pooled.load_300m_dataset("table9").domain_names
    frame = joint.attach_single_phase_weights(frame, args.one_phase_source, domains)
    selected_datasets = [part.strip() for part in args.datasets.split(",") if part.strip()]
    unknown = sorted(set(selected_datasets).difference(joint.TARGET_COLUMNS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    metric_frames = []
    parameter_frames = []
    for objective in selected_datasets:
        dataset = joint.dataset_from_frame(
            objective,
            frame.loc[frame["split"].eq("train") | frame["policy_family"].eq("single_phase")].copy(),
            joint.TARGET_COLUMNS[objective],
        )
        metrics, parameters = benchmark_dataset(
            dataset,
            pooled.parse_int_list(args.seeds),
            args.n_splits,
            args.maxiter,
            args.coarse_top_k,
        )
        metric_frames.append(metrics)
        parameter_frames.append(parameters)
    raw = pd.concat(metric_frames, ignore_index=True)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    summary = staged.summarize(raw, ["dataset", "model"])
    parameter_summary = staged.summarize(parameters, ["dataset"])
    raw.to_csv(args.output_dir / "cv_metrics_by_seed.csv", index=False)
    parameters.to_csv(args.output_dir / "fold_parameters.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    parameter_summary.to_csv(args.output_dir / "parameter_summary.csv", index=False)
    report = [
        "# Aggregate saturation-timescale control",
        "",
        "This control uses only actual constant-mixture checkpoints and is not eligible for phase-order selection.",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "Two shared timescales add one nonlinear rate and one nonnegative benefit amplitude per bucket.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(summary.to_string(index=False))
    print(parameter_summary.to_string(index=False))
    print(f"Wrote aggregate timescale control to {args.output_dir}")


if __name__ == "__main__":
    main()
