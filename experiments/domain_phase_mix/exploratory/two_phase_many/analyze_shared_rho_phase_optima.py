# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Audit supported KL-path optima from shared-rho effective-exposure DSP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_joint_phase_coverage_optima as audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_nested_coverage_dsp_optima as optimum,
)
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
    benchmark_pooled_nonlinear_phase_dsp as pooled_nonlinear,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "shared_rho_phase_optima_20260710"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=joint.PACKET)
    parser.add_argument("--one-phase-source", type=Path, default=joint.ONE_PHASE_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--objectives", default="uncheatable,table9")
    parser.add_argument("--kl-values", default="2,3,5,10,20")
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
    shared_config = pooled_nonlinear.PooledConfig("shared_rho", True, False)
    comparator_config = geometry.FitConfig(
        "effective_exposure_geometry",
        True,
        "effective_exposure",
        (0, 1),
    )

    for objective_name in objectives:
        target = joint.TARGET_COLUMNS[objective_name]
        fit_frame = frame.loc[frame["split"].eq("train") | frame["policy_family"].eq("single_phase")].copy()
        external_frame = frame.loc[frame["split"].eq("heldout") & frame["policy_family"].eq("two_phase")].copy()
        dataset = joint.dataset_from_frame(objective_name, fit_frame, target)
        external = joint.dataset_from_frame(objective_name, external_frame, target)
        alpha0, alpha1 = geometry.phase_fractions(dataset)
        natural = optimum.natural_weights(dataset)
        folds = joint.grouped_folds(dataset.frame, seed=0, n_splits=5)

        full = pooled_nonlinear.fit_model(
            dataset,
            np.arange(dataset.n),
            shared_config,
            args.maxiter,
            args.coarse_top_k,
        ).fitted
        comparator = geometry.fit_model(
            dataset,
            np.arange(dataset.n),
            comparator_config,
            linear_reg=geometry.dataset_linear_reg(dataset),
            maxiter=args.maxiter,
            coarse_top_k=args.coarse_top_k,
        )
        fold_models = []
        oof_prediction = np.zeros(dataset.n, dtype=float)
        for train_idx, test_idx in folds:
            model = pooled_nonlinear.fit_model(
                dataset,
                train_idx,
                shared_config,
                args.maxiter,
                args.coarse_top_k,
            ).fitted
            fold_models.append(model)
            oof_prediction[test_idx] = geometry.predict(
                model,
                dataset.weights[test_idx],
                alpha0,
                alpha1,
            )
        oof_rmse = float(np.sqrt(np.mean((oof_prediction - dataset.y) ** 2)))
        two_phase_indices = np.flatnonzero(
            dataset.frame["split"].eq("train") & dataset.frame["packet_panel"].eq("augmented_fit_panel")
        )
        two_phase_oof_rmse = float(
            np.sqrt(np.mean((oof_prediction[two_phase_indices] - dataset.y[two_phase_indices]) ** 2))
        )

        proportional = np.stack([natural, natural])
        proportional_prediction = float(geometry.predict(full, proportional[None, :, :], alpha0, alpha1)[0])
        sampled_total_epochs = np.max(
            dataset.weights[:, 0, :] * dataset.c0[None, :] + dataset.weights[:, 1, :] * dataset.c1[None, :],
            axis=1,
        )
        sampled_total_epoch_p95 = float(np.quantile(sampled_total_epochs, 0.95))
        candidates = audit.optimize_path(
            dataset,
            full,
            natural,
            alpha0,
            alpha1,
            kl_values,
        )
        for kl_reg, candidate in candidates.items():
            prediction = float(geometry.predict(full, candidate[None, :, :], alpha0, alpha1)[0])
            refit_predictions = np.asarray(
                [geometry.predict(model, candidate[None, :, :], alpha0, alpha1)[0] for model in fold_models],
                dtype=float,
            )
            aggregate = alpha0 * candidate[0] + alpha1 * candidate[1]
            tied = np.stack([aggregate, aggregate])
            tied_prediction = float(geometry.predict(full, tied[None, :, :], alpha0, alpha1)[0])
            fold_tied_predictions = np.asarray(
                [geometry.predict(model, tied[None, :, :], alpha0, alpha1)[0] for model in fold_models],
                dtype=float,
            )
            ordering_margin = tied_prediction - prediction
            fold_ordering_margin = fold_tied_predictions - refit_predictions
            comparator_prediction = float(geometry.predict(comparator, candidate[None, :, :], alpha0, alpha1)[0])
            fit_tv, fit_target = audit.nearest(dataset, candidate)
            external_tv, external_target = audit.nearest(external, candidate)
            local_indices = audit.nearest_indices(dataset, candidate, count=3)
            local_predictions = geometry.predict(
                full,
                dataset.weights[local_indices],
                alpha0,
                alpha1,
            )
            local_residual_max = float(np.max(np.abs(local_predictions - dataset.y[local_indices])))
            local_observed_min = float(np.min(dataset.y[local_indices]))
            phase0_epoch = candidate[0] * dataset.c0
            phase1_epoch = candidate[1] * dataset.c1
            candidate_max_total_epoch = float(np.max(phase0_epoch + phase1_epoch))
            optimism = float(np.min(dataset.y) - prediction)
            pair_noise_sd = np.sqrt(2.0) * audit.NOISE_SD_3E18[objective_name]
            model_disagreement = abs(comparator_prediction - prediction)

            gates = {
                "passes_optimism_gate": optimism <= 2.0 * two_phase_oof_rmse,
                "passes_refit_gate": float(np.std(refit_predictions, ddof=1)) <= two_phase_oof_rmse,
                "passes_support_gate": min(fit_tv, external_tv) <= 0.2,
                "passes_local_residual_gate": local_residual_max <= 2.0 * two_phase_oof_rmse,
                "passes_local_floor_gate": prediction >= local_observed_min - 2.0 * two_phase_oof_rmse,
                "passes_cross_variant_gate": model_disagreement <= two_phase_oof_rmse,
                "passes_power_gate": ordering_margin >= 2.0 * pair_noise_sd,
                "passes_epoch_gate": candidate_max_total_epoch <= sampled_total_epoch_p95,
            }
            rows.append(
                {
                    "dataset": dataset.name,
                    "model": shared_config.name,
                    "kl_reg": kl_reg,
                    "predicted_target": prediction,
                    "comparator_prediction": comparator_prediction,
                    "fold_prediction_mean": float(np.mean(refit_predictions)),
                    "fold_prediction_sd": float(np.std(refit_predictions, ddof=1)),
                    "oof_rmse": oof_rmse,
                    "two_phase_oof_rmse": two_phase_oof_rmse,
                    "optimism_below_panel_best": optimism,
                    "optimism_in_oof_rmse": optimism / oof_rmse,
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
                    "aggregate_margin_vs_proportional": proportional_prediction - tied_prediction,
                    "ordering_margin_vs_tied": ordering_margin,
                    "fold_ordering_margin_mean": float(np.mean(fold_ordering_margin)),
                    "fold_ordering_margin_sd": float(np.std(fold_ordering_margin, ddof=1)),
                    "ordering_margin_in_3e18_diff_sd": ordering_margin / pair_noise_sd,
                    "cross_variant_disagreement": model_disagreement,
                    "max_weight": float(np.max(candidate)),
                    "max_phase_epoch": float(max(np.max(phase0_epoch), np.max(phase1_epoch))),
                    "max_total_epoch": candidate_max_total_epoch,
                    "sampled_total_epoch_p95": sampled_total_epoch_p95,
                    **gates,
                    "passes_all_primary_gates": all(gates.values()),
                }
            )
            for phase in range(2):
                for domain, weight in zip(dataset.domain_names, candidate[phase], strict=True):
                    weight_rows.append(
                        {
                            "dataset": dataset.name,
                            "model": shared_config.name,
                            "kl_reg": kl_reg,
                            "phase": phase,
                            "domain": domain,
                            "weight": float(weight),
                        }
                    )

    diagnostics = pd.DataFrame(rows)
    weights = pd.DataFrame(weight_rows)
    diagnostics.to_csv(args.output_dir / "kl_path_diagnostics.csv", index=False)
    weights.to_csv(args.output_dir / "kl_path_weights_long.csv", index=False)
    print(diagnostics.to_string(index=False))
    print(f"Wrote shared-rho optimum audit to {args.output_dir}")


if __name__ == "__main__":
    main()
