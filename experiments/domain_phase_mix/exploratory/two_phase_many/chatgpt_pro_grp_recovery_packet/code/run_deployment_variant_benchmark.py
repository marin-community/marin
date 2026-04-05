# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Benchmark observed-only GRP deployment regularizers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grp_packet import (
    GenericFamilyRetainedTotalSurrogate,
    average_phase_tv_distance,
    load_packet,
    load_reference_state,
    load_subset_indices,
    mean_phase_tv_distance,
    optimize_generic_family_convex_hull,
    subset_packet,
    tune_genericfamily_params,
)

plt.switch_backend("Agg")
matplotlib.rcParams["text.usetex"] = False

SUBSET_SIZES = tuple(range(20, 240, 20))


def _variant_deployment(
    variant: str, train_packet, model: GenericFamilyRetainedTotalSurrogate
) -> tuple[float, np.ndarray]:
    actual_order = np.argsort(train_packet.base.y)
    anchors = train_packet.base.w

    if variant == "top1_actual":
        deployment = anchors[actual_order[0]]
        return float(model.predict(deployment[None, :, :])[0]), deployment

    if variant.startswith("top") and variant.endswith("_actual_hull"):
        count = int(variant[3 : variant.index("_")])
        hull_anchor_indices = actual_order[: min(count, len(actual_order))]
        hull_anchor_weights = anchors[hull_anchor_indices]
        start_indices = np.arange(min(8, len(hull_anchor_indices)), dtype=int)
        return optimize_generic_family_convex_hull(model, hull_anchor_weights, start_indices=start_indices)[::2]

    if variant == "all_observed_hull":
        predicted_value, _coeffs, deployment = optimize_generic_family_convex_hull(
            model,
            anchors,
            start_indices=np.arange(min(8, anchors.shape[0]), dtype=int),
        )
        return predicted_value, deployment

    if variant == "all_hull_disp0.01":
        pairwise_penalty = np.zeros((anchors.shape[0], anchors.shape[0]), dtype=float)
        for i in range(anchors.shape[0]):
            for j in range(anchors.shape[0]):
                pairwise_penalty[i, j] = 0.01 * mean_phase_tv_distance(anchors[i], anchors[j])
        predicted_value, _coeffs, deployment = optimize_generic_family_convex_hull(
            model,
            anchors,
            start_indices=np.arange(min(8, anchors.shape[0]), dtype=int),
            pairwise_penalty=pairwise_penalty,
        )
        return predicted_value, deployment

    if variant == "all_hull_to_bestactual0.02":
        best_anchor = anchors[actual_order[0]]
        linear_penalty = 0.02 * average_phase_tv_distance(anchors, best_anchor[None, :, :])
        predicted_value, _coeffs, deployment = optimize_generic_family_convex_hull(
            model,
            anchors,
            start_indices=np.arange(min(8, anchors.shape[0]), dtype=int),
            linear_penalty=linear_penalty,
        )
        return predicted_value, deployment

    raise ValueError(f"Unsupported deployment variant: {variant}")


def _plot(frame: pd.DataFrame, plot_path: Path) -> None:
    cmap = plt.colormaps["RdYlGn_r"]
    variants = list(frame["variant"].drop_duplicates())
    color_positions = np.linspace(0.12, 0.88, num=len(variants))
    color_map = {variant: cmap(pos) for variant, pos in zip(variants, color_positions, strict=True)}

    fig, axes = plt.subplots(4, 1, figsize=(10.8, 10.5), dpi=180, sharex=True, constrained_layout=True)
    panels = [
        ("predicted_optimum_value", "Predicted BPB"),
        ("fullswarm_regret_at_1", "Regret@1"),
        ("tuning_cv_foldmean_regret_at_1", "CV Mean Regret@1"),
        ("optimum_move_mean_phase_tv_vs_prev", "Mean phase TV"),
    ]
    for ax, (column, ylabel) in zip(axes, panels, strict=True):
        for variant in variants:
            sub = frame[frame["variant"] == variant].sort_values("subset_size")
            ax.plot(
                sub["subset_size"],
                sub[column],
                marker="o",
                linewidth=1.8,
                color=color_map[variant],
                label=variant,
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].set_title("GRP deployment variants on retuned subset fits")
    axes[-1].set_xlabel("Observed runs used for fitting")
    axes[-1].set_xticks(list(SUBSET_SIZES))
    axes[-1].set_xlim(min(SUBSET_SIZES), max(SUBSET_SIZES))
    axes[0].legend(loc="best", frameon=True, ncol=2)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    packet = load_packet()
    anchors = load_reference_state()
    subset_indices = load_subset_indices()
    valid_weights = np.stack([anchors.validated_global_weights, anchors.validated_pair_weights], axis=0)
    valid_y = np.asarray([anchors.validated_global_bpb, anchors.validated_pair_bpb], dtype=float)

    variants = (
        "top1_actual",
        "top4_actual_hull",
        "top8_actual_hull",
        "top16_actual_hull",
        "all_observed_hull",
        "all_hull_disp0.01",
        "all_hull_to_bestactual0.02",
    )

    best_observed_bpb = float(np.min(packet.base.y))
    rows: list[dict[str, object]] = []
    previous_by_variant: dict[str, np.ndarray] = {}
    for subset_size in SUBSET_SIZES:
        train_packet = subset_packet(packet, np.asarray(subset_indices[subset_size], dtype=int))
        tuning_metrics, _ = tune_genericfamily_params(
            train_packet,
            valid_weights,
            valid_y,
            method="L-BFGS-B",
            objective_name="single_foldmean",
            start_params=anchors.current_tuned_params,
            seed=0,
        )
        tuned_params = {key: float(tuning_metrics[key]) for key in ("alpha", "eta", "lam", "tau", "reg", "beta")}
        model = GenericFamilyRetainedTotalSurrogate(train_packet, params=tuned_params).fit(
            train_packet.base.w,
            train_packet.base.y,
        )
        fullswarm_predictions = model.predict(packet.base.w)
        chosen_idx = int(np.argmin(fullswarm_predictions))

        for variant in variants:
            predicted_value, deployment = _variant_deployment(variant, train_packet, model)
            rows.append(
                {
                    "subset_size": subset_size,
                    "variant": variant,
                    "predicted_optimum_value": float(predicted_value),
                    "fullswarm_chosen_run_name": packet.base.run_names[chosen_idx],
                    "fullswarm_chosen_value": float(packet.base.y[chosen_idx]),
                    "fullswarm_regret_at_1": float(packet.base.y[chosen_idx] - best_observed_bpb),
                    "optimum_move_mean_phase_tv_vs_prev": (
                        None
                        if variant not in previous_by_variant
                        else mean_phase_tv_distance(deployment, previous_by_variant[variant])
                    ),
                    "tuning_cv_foldmean_regret_at_1": float(tuning_metrics["cv_foldmean_regret_at_1"]),
                    "tuning_cv_rmse": float(tuning_metrics["cv_rmse"]),
                }
            )
            previous_by_variant[variant] = deployment

    frame = pd.DataFrame(rows)
    curve_csv = Path(__file__).resolve().parents[1] / "reference_outputs" / "deployment_variant_curve_points.csv"
    summary_json = Path(__file__).resolve().parents[1] / "reference_outputs" / "deployment_variant_summary.json"
    plot_path = Path(__file__).resolve().parents[1] / "reference_outputs" / "deployment_variant_comparison.png"
    curve_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(curve_csv, index=False)

    summary_rows = []
    for variant, group in frame.groupby("variant"):
        after80 = group[group["subset_size"] >= 80]
        summary_rows.append(
            {
                "variant": variant,
                "mean_predicted_bpb_after80": float(after80["predicted_optimum_value"].mean()),
                "mean_move_after80": float(after80["optimum_move_mean_phase_tv_vs_prev"].dropna().mean()),
                "mean_regret_after80": float(after80["fullswarm_regret_at_1"].mean()),
                "mean_cv_foldmean_regret_after80": float(after80["tuning_cv_foldmean_regret_at_1"].mean()),
            }
        )
    summary_json.write_text(
        json.dumps(
            {
                "curve_points_csv": str(curve_csv),
                "plot": str(plot_path),
                "summary_rows": summary_rows,
            },
            indent=2,
        )
    )
    _plot(frame, plot_path)


if __name__ == "__main__":
    main()
