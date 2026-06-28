# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas"]
# ///
"""Compare observed-only GRP deployment regularizers on top of the same retuned fit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.dataset_metadata import (
    load_two_phase_many_candidate_summary_spec,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GenericFamilyRetainedTotalSurrogate,
    load_generic_family_packet,
    optimize_generic_family_convex_hull,
)
from experiments.domain_phase_mix.static_batch_selection import pairwise_distance_matrix, retrospective_generic_selection
from experiments.domain_phase_mix.two_phase_many_ccglobalpremium_baselines import (
    ccglobalpremium_retainedtotal_summary,
)
from experiments.domain_phase_mix.two_phase_many_ccpairtotal_baseline import (
    ccpairtotal_retainedtotal_summary,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_retuned_subset_optima import (
    CSV_PATH,
    GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES,
    OBJECTIVE_METRIC,
    VALIDATED_GLOBAL_BPB,
    VALIDATED_PAIR_BPB,
    _mean_phase_tv_distance,
    _subset_packet,
    _summary_weights,
    tune_genericfamily_subset_params,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DETAIL_CSV = SCRIPT_DIR / "two_phase_many_grp_deployment_variant_curve_points.csv"
SUMMARY_JSON = SCRIPT_DIR / "two_phase_many_grp_deployment_variant_summary.json"

VARIANTS: tuple[dict[str, object], ...] = (
    {"name": "top1_observed", "kind": "predicted_hull", "topk": 1},
    {"name": "top4_hull", "kind": "predicted_hull", "topk": 4},
    {"name": "top8_hull", "kind": "predicted_hull", "topk": 8},
    {"name": "top16_hull", "kind": "predicted_hull", "topk": 16},
    {"name": "all_observed_hull", "kind": "predicted_hull", "topk": None},
    {"name": "top4_actual_hull", "kind": "actual_hull", "topk": 4},
    {"name": "top8_actual_hull", "kind": "actual_hull", "topk": 8},
    {"name": "top16_actual_hull", "kind": "actual_hull", "topk": 16},
    {"name": "all_hull_disp0.01", "kind": "dispersion_penalty", "scale": 0.01},
    {"name": "all_hull_disp0.02", "kind": "dispersion_penalty", "scale": 0.02},
    {"name": "all_hull_to_bestactual0.02", "kind": "bestactual_penalty", "scale": 0.02},
    {"name": "all_hull_to_bestactual0.05", "kind": "bestactual_penalty", "scale": 0.05},
)
HULL_START_COUNT = 8


def main() -> None:
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    _, spec, _ = load_two_phase_many_candidate_summary_spec(
        CSV_PATH,
        objective_metric=OBJECTIVE_METRIC,
        name="two_phase_many_grp_deployment_variants",
    )
    valid_weights = np.stack(
        [
            _summary_weights(ccglobalpremium_retainedtotal_summary(), packet.base.domain_names),
            _summary_weights(ccpairtotal_retainedtotal_summary(), packet.base.domain_names),
        ],
        axis=0,
    )
    valid_y = np.asarray([VALIDATED_GLOBAL_BPB, VALIDATED_PAIR_BPB], dtype=float)
    best_observed_bpb = float(packet.base.y.min())

    rows: list[dict[str, object]] = []
    previous_deployments: dict[str, np.ndarray | None] = {variant["name"]: None for variant in VARIANTS}

    for subset_size in GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES:
        selection = retrospective_generic_selection(spec, method="feature_bayes_linear", k=subset_size, seed=0)
        subset_indices = np.asarray(selection.selected_indices, dtype=int)
        train_packet = _subset_packet(packet, subset_indices)
        tuning_metrics, _ = tune_genericfamily_subset_params(train_packet, valid_weights, valid_y)
        tuned_params = {key: float(tuning_metrics[key]) for key in ("alpha", "eta", "lam", "tau", "reg", "beta")}
        model = GenericFamilyRetainedTotalSurrogate(train_packet, params=tuned_params).fit(
            train_packet.base.w,
            train_packet.base.y,
        )
        subset_predictions = model.predict(train_packet.base.w)
        actual_order = np.argsort(train_packet.base.y)
        order = np.argsort(subset_predictions)
        fullswarm_predictions = model.predict(packet.base.w)
        chosen_idx = int(np.argmin(fullswarm_predictions))
        best_actual_idx = int(actual_order[0])
        pairwise_distances = pairwise_distance_matrix(train_packet.base.w)

        for variant in VARIANTS:
            variant_name = str(variant["name"])
            topk = variant.get("topk")
            kind = str(variant["kind"])

            if kind == "predicted_hull" and topk == 1:
                deployment = train_packet.base.w[order[0]]
                predicted_value = float(subset_predictions[order[0]])
                coeff_count = 1
                top_hull_run_names = [str(train_packet.base.frame.iloc[order[0]][train_packet.base.name_col])]
                top_hull_coeffs = [1.0]
            else:
                if kind == "predicted_hull" and topk is None:
                    anchors = train_packet.base.w
                    selected = order[: min(HULL_START_COUNT, len(order))]
                    start_indices = selected
                    predicted_value, coeffs, deployment = optimize_generic_family_convex_hull(
                        model,
                        anchors,
                        start_indices=start_indices,
                    )
                    coeff_count = int(np.sum(np.asarray(coeffs) > 1e-6))
                    top_indices = np.argsort(coeffs)[::-1][: min(5, len(coeffs))]
                    name_indices = top_indices
                elif kind == "predicted_hull":
                    selected = order[: min(topk, len(order))]
                    anchors = train_packet.base.w[selected]
                    start_indices = np.arange(len(selected), dtype=int)
                    predicted_value, coeffs, deployment = optimize_generic_family_convex_hull(
                        model,
                        anchors,
                        start_indices=start_indices,
                    )
                    coeff_count = int(np.sum(np.asarray(coeffs) > 1e-6))
                    top_indices = np.argsort(coeffs)[::-1][: min(5, len(coeffs))]
                    name_indices = [selected[idx] for idx in top_indices]
                elif kind == "actual_hull":
                    selected = actual_order[: min(topk, len(actual_order))]
                    anchors = train_packet.base.w[selected]
                    start_indices = np.arange(min(HULL_START_COUNT, len(selected)), dtype=int)
                    predicted_value, coeffs, deployment = optimize_generic_family_convex_hull(
                        model,
                        anchors,
                        start_indices=start_indices,
                    )
                    coeff_count = int(np.sum(np.asarray(coeffs) > 1e-6))
                    top_indices = np.argsort(coeffs)[::-1][: min(5, len(coeffs))]
                    name_indices = [selected[idx] for idx in top_indices]
                elif kind == "dispersion_penalty":
                    predicted_value, coeffs, deployment = optimize_generic_family_convex_hull(
                        model,
                        train_packet.base.w,
                        start_indices=order[: min(HULL_START_COUNT, len(order))],
                        pairwise_penalty=float(variant["scale"]) * pairwise_distances,
                    )
                    coeff_count = int(np.sum(np.asarray(coeffs) > 1e-6))
                    top_indices = np.argsort(coeffs)[::-1][: min(5, len(coeffs))]
                    name_indices = top_indices
                elif kind == "bestactual_penalty":
                    predicted_value, coeffs, deployment = optimize_generic_family_convex_hull(
                        model,
                        train_packet.base.w,
                        start_indices=order[: min(HULL_START_COUNT, len(order))],
                        linear_penalty=float(variant["scale"]) * pairwise_distances[best_actual_idx],
                    )
                    coeff_count = int(np.sum(np.asarray(coeffs) > 1e-6))
                    top_indices = np.argsort(coeffs)[::-1][: min(5, len(coeffs))]
                    name_indices = top_indices
                else:
                    raise ValueError(f"Unsupported deployment variant kind: {kind}")
                top_hull_run_names = [
                    str(train_packet.base.frame.iloc[idx][train_packet.base.name_col]) for idx in name_indices
                ]
                top_hull_coeffs = [float(coeffs[idx]) for idx in top_indices]

            movement = (
                None
                if previous_deployments[variant_name] is None
                else _mean_phase_tv_distance(deployment, previous_deployments[variant_name])
            )
            previous_deployments[variant_name] = deployment
            rows.append(
                {
                    "variant": variant_name,
                    "variant_kind": kind,
                    "subset_size": subset_size,
                    "predicted_optimum_value": float(predicted_value),
                    "pred_minus_best_observed": float(predicted_value - best_observed_bpb),
                    "pred_minus_chosen_actual": float(predicted_value - float(packet.base.y[chosen_idx])),
                    "fullswarm_regret_at_1": float(packet.base.y[chosen_idx] - best_observed_bpb),
                    "tuning_cv_foldmean_regret_at_1": float(tuning_metrics["cv_foldmean_regret_at_1"]),
                    "move_mean_phase_tv_vs_prev": movement,
                    "chosen_run_name": str(packet.base.frame.iloc[chosen_idx][packet.base.name_col]),
                    "nearest_observed_tv_distance": float(
                        np.min(0.5 * np.abs(train_packet.base.w - deployment[None, :, :]).sum(axis=2).mean(axis=1))
                    ),
                    "hull_nonzero_coeff_count": coeff_count,
                    "top_hull_run_names": top_hull_run_names,
                    "top_hull_coeffs": top_hull_coeffs,
                }
            )

    detail_frame = pd.DataFrame(rows)
    detail_frame.to_csv(DETAIL_CSV, index=False)

    summary: dict[str, dict[str, float | int]] = {}
    for variant in VARIANTS:
        variant_name = str(variant["name"])
        variant_rows = detail_frame[detail_frame["variant"] == variant_name]
        later_rows = variant_rows[variant_rows["subset_size"] >= 80]
        summary[variant_name] = {
            "mean_regret_all": float(variant_rows["fullswarm_regret_at_1"].mean()),
            "mean_regret_after80": float(later_rows["fullswarm_regret_at_1"].mean()),
            "num_zero_regret_after80": int((later_rows["fullswarm_regret_at_1"] == 0.0).sum()),
            "mean_move_after80": float(later_rows["move_mean_phase_tv_vs_prev"].dropna().mean()),
            "mean_predicted_value_after80": float(later_rows["predicted_optimum_value"].mean()),
            "mean_pred_minus_best_observed_after80": float(later_rows["pred_minus_best_observed"].mean()),
            "mean_pred_minus_chosen_actual_after80": float(later_rows["pred_minus_chosen_actual"].mean()),
            "mean_nearest_observed_tv_after80": float(later_rows["nearest_observed_tv_distance"].mean()),
            "mean_support_after80": float(later_rows["hull_nonzero_coeff_count"].mean()),
        }

    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "objective_metric": OBJECTIVE_METRIC,
                "detail_csv": str(DETAIL_CSV),
                "variants": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
