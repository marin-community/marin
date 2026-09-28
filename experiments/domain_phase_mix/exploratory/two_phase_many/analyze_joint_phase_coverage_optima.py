# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Audit KL-path optima from matched-panel mechanistic phase models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_nested_coverage_dsp_optima as optimum,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_coverage_augmented_separate_heads as separate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_joint_phase_correspondence_dsp as joint,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_nested_coverage_dsp as coverage,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "mechanistic_phase_optima_20260710"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
NOISE_SD_3E18 = {
    "uncheatable": 0.00091299968961728,
    "table9": 0.003771768091801164,
}


def optimize_path(
    dataset: pooled.Dataset,
    model: coverage.CoverageModel,
    natural: np.ndarray,
    alpha0: float,
    alpha1: float,
    kl_values: list[float],
) -> dict[float, np.ndarray]:
    m = dataset.m

    def objective(logits: np.ndarray, kl_reg: float) -> float:
        weights = optimum.weights_from_logits(logits, m)
        prediction = float(coverage.predict(model, weights[None, :, :], alpha0, alpha1)[0])
        return prediction + kl_reg * optimum.weighted_kl(weights, natural, alpha0, alpha1)

    natural_start = np.log(np.clip(np.stack([natural, natural]), 1e-12, 1.0)).reshape(-1)
    observed_starts = [
        np.log(np.clip(dataset.weights[index], 1e-12, 1.0)).reshape(-1) for index in np.argsort(dataset.y)[:4]
    ]
    previous = natural_start
    result_by_kl = {}
    for kl_reg in sorted(kl_values, reverse=True):
        best_value = np.inf
        best_logits = None
        for start in (previous, natural_start, *observed_starts):
            result = minimize(
                lambda logits, current_kl=kl_reg: objective(np.asarray(logits, dtype=float), current_kl),
                start,
                method="L-BFGS-B",
                options={"maxiter": 400, "ftol": 1e-9, "maxls": 30},
            )
            if float(result.fun) < best_value:
                best_value = float(result.fun)
                best_logits = np.asarray(result.x, dtype=float)
        if best_logits is None:
            raise RuntimeError(f"No optimizer result for KL={kl_reg}")
        previous = best_logits
        result_by_kl[kl_reg] = optimum.weights_from_logits(best_logits, m)
    return result_by_kl


def nearest(dataset: pooled.Dataset, candidate: np.ndarray) -> tuple[float, float]:
    distances = 0.5 * np.abs(dataset.weights - candidate[None, :, :]).sum(axis=2).mean(axis=1)
    index = int(np.argmin(distances))
    return float(distances[index]), float(dataset.y[index])


def nearest_indices(dataset: pooled.Dataset, candidate: np.ndarray, count: int) -> np.ndarray:
    distances = 0.5 * np.abs(dataset.weights - candidate[None, :, :]).sum(axis=2).mean(axis=1)
    return np.argsort(distances)[:count]


def separate_prediction(
    model: separate.SeparateCoverageModel,
    dataset: pooled.Dataset,
    weights: np.ndarray,
) -> np.ndarray:
    exposure0 = weights[:, 0, :] * dataset.c0[None, :]
    exposure1 = weights[:, 1, :] * dataset.c1[None, :]
    return (
        model.intercept
        + pooled.bowl_design(exposure0, model.mu0) @ model.coef0
        + pooled.bowl_design(exposure1, model.mu1) @ model.coef1
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=joint.PACKET)
    parser.add_argument("--one-phase-source", type=Path, default=joint.ONE_PHASE_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--objectives", default="uncheatable,table9")
    parser.add_argument("--kl-values", default="0.1,0.2,0.3,0.5,1.0,2.0")
    parser.add_argument("--maxiter", type=int, default=16)
    parser.add_argument("--coarse-top-k", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.packet)
    domains = pooled.load_300m_dataset("table9").domain_names
    frame = joint.attach_single_phase_weights(frame, args.one_phase_source, domains)
    objectives = [part.strip() for part in args.objectives.split(",") if part.strip()]
    kl_values = pooled.parse_float_list(args.kl_values)
    rows = []
    weight_rows = []
    for objective_name in objectives:
        target = joint.TARGET_COLUMNS[objective_name]
        fit_frame = frame.loc[frame["split"].eq("train") | frame["policy_family"].eq("single_phase")].copy()
        external_frame = frame.loc[frame["split"].eq("heldout") & frame["policy_family"].eq("two_phase")].copy()
        dataset = joint.dataset_from_frame(objective_name, fit_frame, target)
        external = joint.dataset_from_frame(objective_name, external_frame, target)
        alpha0, alpha1 = coverage.phase_fractions(dataset)
        natural = optimum.natural_weights(dataset)
        folds = joint.grouped_folds(dataset.frame, seed=0, n_splits=5)
        configs = (
            coverage.FitConfig("effective_exposure_geometry", True, "effective_exposure", (0, 1)),
            coverage.FitConfig(
                "split_saturation_penalty_geometry",
                True,
                "split_saturation_penalty",
                (0, 1),
            ),
        )
        full_models = {}
        fold_models = {}
        oof_rmse = {}
        two_phase_oof_rmse = {}
        oof_predictions = {}
        for config in configs:
            full_models[config.name] = coverage.fit_model(
                dataset,
                np.arange(dataset.n),
                config,
                linear_reg=coverage.dataset_linear_reg(dataset),
                maxiter=args.maxiter,
                coarse_top_k=args.coarse_top_k,
            )
            predictions = np.zeros(dataset.n, dtype=float)
            fold_models[config.name] = []
            for train_idx, test_idx in folds:
                model = coverage.fit_model(
                    dataset,
                    train_idx,
                    config,
                    linear_reg=coverage.dataset_linear_reg(dataset),
                    maxiter=args.maxiter,
                    coarse_top_k=args.coarse_top_k,
                )
                fold_models[config.name].append(model)
                predictions[test_idx] = coverage.predict(model, dataset.weights[test_idx], alpha0, alpha1)
            oof_rmse[config.name] = float(np.sqrt(np.mean((predictions - dataset.y) ** 2)))
            two_phase_indices = np.flatnonzero(
                dataset.frame["split"].eq("train") & dataset.frame["packet_panel"].eq("augmented_fit_panel")
            )
            two_phase_oof_rmse[config.name] = float(
                np.sqrt(np.mean((predictions[two_phase_indices] - dataset.y[two_phase_indices]) ** 2))
            )
            oof_predictions[config.name] = predictions

        separate_model = separate.fit_head(
            dataset,
            np.arange(dataset.n),
            use_coverage=False,
            alpha0=alpha0,
            alpha1=alpha1,
        )
        separate_fit_predictions = separate_prediction(separate_model, dataset, dataset.weights)
        separate_external_predictions = separate_prediction(separate_model, dataset, external.weights)

        proportional = np.stack([natural, natural])
        sampled_total_epochs = np.max(
            dataset.weights[:, 0, :] * dataset.c0[None, :] + dataset.weights[:, 1, :] * dataset.c1[None, :],
            axis=1,
        )
        sampled_total_epoch_p95 = float(np.quantile(sampled_total_epochs, 0.95))
        for config in configs:
            model = full_models[config.name]
            proportional_prediction = float(coverage.predict(model, proportional[None, :, :], alpha0, alpha1)[0])
            candidates = optimize_path(dataset, model, natural, alpha0, alpha1, kl_values)
            for kl_reg, candidate in candidates.items():
                full_prediction = float(coverage.predict(model, candidate[None, :, :], alpha0, alpha1)[0])
                refit_predictions = np.asarray(
                    [
                        coverage.predict(fold_model, candidate[None, :, :], alpha0, alpha1)[0]
                        for fold_model in fold_models[config.name]
                    ],
                    dtype=float,
                )
                aggregate = alpha0 * candidate[0] + alpha1 * candidate[1]
                tied_candidate = np.stack([aggregate, aggregate])
                tied_prediction = float(coverage.predict(model, tied_candidate[None, :, :], alpha0, alpha1)[0])
                fold_tied_predictions = np.asarray(
                    [
                        coverage.predict(fold_model, tied_candidate[None, :, :], alpha0, alpha1)[0]
                        for fold_model in fold_models[config.name]
                    ],
                    dtype=float,
                )
                fold_ordering_margins = fold_tied_predictions - refit_predictions
                other_name = (
                    "split_saturation_penalty_geometry"
                    if config.name == "effective_exposure_geometry"
                    else "effective_exposure_geometry"
                )
                other_prediction = float(
                    coverage.predict(full_models[other_name], candidate[None, :, :], alpha0, alpha1)[0]
                )
                fit_tv, fit_target = nearest(dataset, candidate)
                external_tv, external_target = nearest(external, candidate)
                local_indices = nearest_indices(dataset, candidate, count=3)
                local_predictions = coverage.predict(model, dataset.weights[local_indices], alpha0, alpha1)
                local_residual_max = float(np.max(np.abs(local_predictions - dataset.y[local_indices])))
                local_observed_min = float(np.min(dataset.y[local_indices]))
                separate_candidate_prediction = float(
                    separate_prediction(separate_model, dataset, candidate[None, :, :])[0]
                )
                phase0_epoch = candidate[0] * dataset.c0
                phase1_epoch = candidate[1] * dataset.c1
                optimism = float(np.min(dataset.y) - full_prediction)
                ordering_margin = tied_prediction - full_prediction
                aggregate_margin = proportional_prediction - tied_prediction
                pair_noise_sd_3e18 = np.sqrt(2.0) * NOISE_SD_3E18[objective_name]
                model_disagreement = abs(other_prediction - full_prediction)
                candidate_max_total_epoch = float(np.max(phase0_epoch + phase1_epoch))
                passes_optimism = optimism <= 2.0 * two_phase_oof_rmse[config.name]
                passes_refit = float(np.std(refit_predictions, ddof=1)) <= two_phase_oof_rmse[config.name]
                passes_support = min(fit_tv, external_tv) <= 0.2
                passes_local_residual = local_residual_max <= 2.0 * two_phase_oof_rmse[config.name]
                passes_local_floor = full_prediction >= local_observed_min - 2.0 * two_phase_oof_rmse[config.name]
                passes_cross_variant = model_disagreement <= two_phase_oof_rmse[config.name]
                passes_power = ordering_margin >= 2.0 * pair_noise_sd_3e18
                passes_epoch = candidate_max_total_epoch <= sampled_total_epoch_p95
                rows.append(
                    {
                        "dataset": dataset.name,
                        "model": config.name,
                        "kl_reg": kl_reg,
                        "predicted_target": full_prediction,
                        "other_model_prediction": other_prediction,
                        "fold_prediction_mean": float(np.mean(refit_predictions)),
                        "fold_prediction_sd": float(np.std(refit_predictions, ddof=1)),
                        "fold_prediction_min": float(np.min(refit_predictions)),
                        "fold_prediction_max": float(np.max(refit_predictions)),
                        "oof_rmse": oof_rmse[config.name],
                        "two_phase_oof_rmse": two_phase_oof_rmse[config.name],
                        "optimism_below_panel_best": optimism,
                        "optimism_in_oof_rmse": optimism / oof_rmse[config.name],
                        "tv_to_proportional": optimum.mean_phase_tv(candidate, proportional),
                        "phase_tv": float(0.5 * np.abs(candidate[0] - candidate[1]).sum()),
                        "nearest_fit_tv": fit_tv,
                        "nearest_fit_target": fit_target,
                        "nearest_external_tv": external_tv,
                        "nearest_external_target": external_target,
                        "local_neighbor_observed_min": local_observed_min,
                        "local_neighbor_residual_max": local_residual_max,
                        "proportional_prediction": proportional_prediction,
                        "tied_candidate_prediction": tied_prediction,
                        "aggregate_margin_vs_proportional": aggregate_margin,
                        "ordering_margin_vs_tied": ordering_margin,
                        "fold_ordering_margin_mean": float(np.mean(fold_ordering_margins)),
                        "fold_ordering_margin_sd": float(np.std(fold_ordering_margins, ddof=1)),
                        "ordering_margin_in_3e18_diff_sd": ordering_margin / pair_noise_sd_3e18,
                        "cross_variant_disagreement": model_disagreement,
                        "separate_head_prediction": separate_candidate_prediction,
                        "separate_head_fit_percentile": float(
                            np.mean(separate_fit_predictions <= separate_candidate_prediction)
                        ),
                        "separate_head_external_percentile": float(
                            np.mean(separate_external_predictions <= separate_candidate_prediction)
                        ),
                        "max_weight": float(np.max(candidate)),
                        "max_phase_epoch": float(max(np.max(phase0_epoch), np.max(phase1_epoch))),
                        "max_total_epoch": candidate_max_total_epoch,
                        "sampled_total_epoch_p95": sampled_total_epoch_p95,
                        "passes_optimism_gate": passes_optimism,
                        "passes_refit_gate": passes_refit,
                        "passes_support_gate": passes_support,
                        "passes_local_residual_gate": passes_local_residual,
                        "passes_local_floor_gate": passes_local_floor,
                        "passes_cross_variant_gate": passes_cross_variant,
                        "passes_power_gate": passes_power,
                        "passes_epoch_gate": passes_epoch,
                        "passes_separate_head_gate": (
                            float(np.mean(separate_external_predictions <= separate_candidate_prediction)) <= 0.1
                        ),
                        "passes_all_primary_gates": all(
                            (
                                passes_optimism,
                                passes_refit,
                                passes_support,
                                passes_local_residual,
                                passes_local_floor,
                                passes_cross_variant,
                                passes_power,
                                passes_epoch,
                            )
                        ),
                    }
                )
                for phase in range(2):
                    for domain, value in zip(dataset.domain_names, candidate[phase], strict=True):
                        weight_rows.append(
                            {
                                "dataset": dataset.name,
                                "model": config.name,
                                "kl_reg": kl_reg,
                                "phase": phase,
                                "domain": domain,
                                "weight": float(value),
                            }
                        )
    diagnostics = pd.DataFrame(rows)
    weights = pd.DataFrame(weight_rows)
    diagnostics.to_csv(args.output_dir / "kl_path_diagnostics.csv", index=False)
    weights.to_csv(args.output_dir / "kl_path_weights_long.csv", index=False)
    for metric in (
        "predicted_target",
        "optimism_in_oof_rmse",
        "fold_prediction_sd",
        "nearest_external_tv",
        "max_phase_epoch",
    ):
        figure = px.line(
            diagnostics,
            x="kl_reg",
            y=metric,
            color="model",
            facet_col="dataset",
            markers=True,
            color_discrete_sequence=["#d73027", "#1a9850"],
            title=f"Mechanistic phase-model KL path: {metric}",
        )
        figure.write_html(
            args.output_dir / f"kl_path_{metric}.html",
            include_plotlyjs="cdn",
            config=PLOT_CONFIG,
        )
    print(diagnostics.to_string(index=False))
    print(f"Wrote matched-phase optimum audit to {args.output_dir}")


if __name__ == "__main__":
    main()
