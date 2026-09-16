# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Compare KL paths for nested coverage DSP and effective-exposure DSP.

This consumes the model family benchmarked by ``benchmark_nested_coverage_dsp``
and evaluates whether its regularized optima stay near observed support. No
training jobs are submitted and no candidate is materialized for launch.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_nested_coverage_dsp as coverage,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_two_phase_canonical_bowl_candidates_300m as bowl,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    plot_lf_sepheads_kl_sweep_300m as separate,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "nested_coverage_dsp_optima_20260709"
DEFAULT_KL_VALUES = "0.05,0.1,0.2,0.3,0.5"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def natural_weights(dataset: pooled.Dataset) -> np.ndarray:
    if dataset.name.startswith("300m_"):
        objective = dataset.name.removeprefix("300m_")
        _packet, _domains, natural, _token_counts, _target_budget, _folds = bowl.load_objective(objective)
        return np.asarray(natural, dtype=float)
    from_c0 = 1.0 / dataset.c0
    from_c1 = 1.0 / dataset.c1
    natural = 0.5 * (from_c0 / from_c0.sum() + from_c1 / from_c1.sum())
    return natural / natural.sum()


def weights_from_logits(logits: np.ndarray, m: int) -> np.ndarray:
    output = np.zeros((2, m), dtype=float)
    for phase in range(2):
        values = logits[phase * m : (phase + 1) * m]
        exponent = np.exp(values - np.max(values))
        output[phase] = exponent / exponent.sum()
    return output


def weighted_kl(weights: np.ndarray, natural: np.ndarray, alpha0: float, alpha1: float) -> float:
    reference = np.clip(natural, 1e-12, 1.0)
    terms = []
    for phase in range(2):
        current = np.clip(weights[phase], 1e-12, 1.0)
        terms.append(float(np.sum(current * np.log(current / reference))))
    return alpha0 * terms[0] + alpha1 * terms[1]


def mean_phase_tv(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a - b).sum(axis=1).mean())


def nearest_observed(dataset: pooled.Dataset, weights: np.ndarray) -> tuple[int, float]:
    distances = 0.5 * np.abs(dataset.weights - weights[None, :, :]).sum(axis=2).mean(axis=1)
    index = int(np.argmin(distances))
    return index, float(distances[index])


def optimize(
    dataset: pooled.Dataset,
    model: coverage.CoverageModel,
    kl_reg: float,
    natural: np.ndarray,
    alpha0: float,
    alpha1: float,
) -> np.ndarray:
    m = dataset.m

    def objective(logits: np.ndarray) -> float:
        weights = weights_from_logits(logits, m)
        prediction = float(coverage.predict(model, weights[None, :, :], alpha0, alpha1)[0])
        return prediction + kl_reg * weighted_kl(weights, natural, alpha0, alpha1)

    starts = [np.log(np.clip(np.stack([natural, natural]), 1e-12, 1.0)).reshape(-1)]
    starts.extend(np.log(np.clip(dataset.weights[index], 1e-12, 1.0)).reshape(-1) for index in np.argsort(dataset.y)[:8])
    best_value = np.inf
    best_weights = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            options={"maxiter": 400, "ftol": 1e-9, "maxls": 30},
        )
        if float(result.fun) < best_value:
            best_value = float(result.fun)
            best_weights = weights_from_logits(np.asarray(result.x, dtype=float), m)
    if best_weights is None:
        raise RuntimeError("No optimizer result")
    return best_weights


def separate_head_rank_percentile(
    predictor: Callable[[np.ndarray], float] | None,
    observed_predictions: np.ndarray | None,
    candidate: np.ndarray,
) -> tuple[float, float]:
    if predictor is None or observed_predictions is None:
        return float("nan"), float("nan")
    candidate_prediction = float(predictor(candidate))
    percentile = float(np.mean(observed_predictions <= candidate_prediction))
    return candidate_prediction, percentile


def analyze_dataset(
    dataset: pooled.Dataset,
    kl_values: list[float],
    maxiter_300m: int,
    coarse_top_k: int,
) -> pd.DataFrame:
    all_indices = np.arange(dataset.n)
    natural = natural_weights(dataset)
    alpha0, alpha1 = coverage.phase_fractions(dataset)
    configs = [
        coverage.FitConfig("effective_exposure", False),
        coverage.FitConfig("effective_exposure_coverage", True),
    ]
    models = {
        config.name: coverage.fit_model(
            dataset,
            all_indices,
            config,
            linear_reg=coverage.dataset_linear_reg(dataset),
            maxiter=coverage.dataset_maxiter(dataset, maxiter_300m),
            coarse_top_k=coarse_top_k,
        )
        for config in configs
    }
    separate_predictor = None
    separate_observed = None
    if dataset.name.startswith("300m_"):
        objective = dataset.name.removeprefix("300m_")
        separate_packet, _domains, _natural, _tokens, _budget, _folds = bowl.load_objective(objective)
        separate_predictor = separate.build_predictors(separate_packet)["separate_heads"]
        separate_observed = np.array([separate_predictor(weights) for weights in dataset.weights], dtype=float)
    observed_max_weight = float(dataset.weights.max())
    observed_max_phase_epoch = float(
        max(
            np.max(dataset.weights[:, 0, :] * dataset.c0[None, :]),
            np.max(dataset.weights[:, 1, :] * dataset.c1[None, :]),
        )
    )
    rows = []
    for model_name, model in models.items():
        proportional = np.stack([natural, natural])
        proportional_prediction = float(coverage.predict(model, proportional[None, :, :], alpha0, alpha1)[0])
        for kl_reg in kl_values:
            candidate = optimize(dataset, model, kl_reg, natural, alpha0, alpha1)
            predicted = float(coverage.predict(model, candidate[None, :, :], alpha0, alpha1)[0])
            nearest_index, nearest_tv = nearest_observed(dataset, candidate)
            sep_prediction, sep_percentile = separate_head_rank_percentile(
                separate_predictor, separate_observed, candidate
            )
            phase0_epoch = candidate[0] * dataset.c0
            phase1_epoch = candidate[1] * dataset.c1
            rows.append(
                {
                    "dataset": dataset.name,
                    "model": model_name,
                    "kl_reg": kl_reg,
                    "predicted_target": predicted,
                    "regularized_objective": predicted + kl_reg * weighted_kl(candidate, natural, alpha0, alpha1),
                    "predicted_gain_vs_proportional": proportional_prediction - predicted,
                    "panel_best_observed": float(np.min(dataset.y)),
                    "optimism_below_panel_best": float(np.min(dataset.y) - predicted),
                    "tv_to_proportional": mean_phase_tv(candidate, proportional),
                    "phase_tv": float(0.5 * np.abs(candidate[0] - candidate[1]).sum()),
                    "nearest_observed_tv": nearest_tv,
                    "nearest_observed_target": float(dataset.y[nearest_index]),
                    "max_weight": float(np.max(candidate)),
                    "observed_max_weight": observed_max_weight,
                    "max_phase_epoch": float(max(np.max(phase0_epoch), np.max(phase1_epoch))),
                    "observed_max_phase_epoch": observed_max_phase_epoch,
                    "max_total_epoch": float(np.max(phase0_epoch + phase1_epoch)),
                    "effective_aggregate_buckets": float(
                        1.0 / np.sum((alpha0 * candidate[0] + alpha1 * candidate[1]) ** 2)
                    ),
                    "separate_head_prediction": sep_prediction,
                    "separate_head_percentile": sep_percentile,
                }
            )
    return pd.DataFrame(rows)


def write_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    for metric in (
        "predicted_target",
        "nearest_observed_tv",
        "max_phase_epoch",
        "effective_aggregate_buckets",
        "separate_head_percentile",
    ):
        figure = px.line(
            frame,
            x="kl_reg",
            y=metric,
            color="model",
            facet_col="dataset",
            markers=True,
            color_discrete_sequence=["#d73027", "#1a9850"],
            title=f"Coverage DSP KL path: {metric}",
        )
        figure.write_html(
            output_dir / f"kl_path_{metric}.html",
            include_plotlyjs="cdn",
            config=PLOT_CONFIG,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", default="300m_uncheatable,300m_table9,production_uncheatable")
    parser.add_argument("--kl-values", default=DEFAULT_KL_VALUES)
    parser.add_argument("--maxiter-300m", type=int, default=16)
    parser.add_argument("--coarse-top-k", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaders = {
        "300m_uncheatable": lambda: pooled.load_300m_dataset("uncheatable"),
        "300m_table9": lambda: pooled.load_300m_dataset("table9"),
        "production_uncheatable": pooled.load_production_dataset,
    }
    selected = [part.strip() for part in args.datasets.split(",") if part.strip()]
    unknown = sorted(set(selected).difference(loaders))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    kl_values = pooled.parse_float_list(args.kl_values)
    frames = [analyze_dataset(loaders[name](), kl_values, args.maxiter_300m, args.coarse_top_k) for name in selected]
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(args.output_dir / "kl_path_diagnostics.csv", index=False)
    write_plots(result, args.output_dir)
    print(result.to_string(index=False))
    print(f"Wrote optimum diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
