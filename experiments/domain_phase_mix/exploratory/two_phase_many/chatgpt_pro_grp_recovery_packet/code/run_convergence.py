# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Run GRP convergence analyses for several deployment rules."""

from __future__ import annotations

import argparse
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
    optimize_generic_family_model,
    subset_packet,
    tune_genericfamily_params,
)

plt.switch_backend("Agg")
matplotlib.rcParams["text.usetex"] = False

SUBSET_SIZES = tuple(range(20, 240, 20))


def _deployment(
    variant: str,
    train_packet,
    model: GenericFamilyRetainedTotalSurrogate,
) -> tuple[float, np.ndarray]:
    if variant == "raw_retuned":
        result, phase0, phase1 = optimize_generic_family_model(train_packet, model, seed=0)
        return float(result.fun), np.stack([phase0, phase1], axis=0)

    if variant == "top8actual_hull":
        hull_anchor_indices = np.argsort(train_packet.base.y)[: min(8, train_packet.base.n)]
        hull_anchor_weights = train_packet.base.w[hull_anchor_indices]
        start_indices = np.arange(min(8, len(hull_anchor_indices)), dtype=int)
        predicted_value, _coeffs, deployment = optimize_generic_family_convex_hull(
            model,
            hull_anchor_weights,
            start_indices=start_indices,
        )
        return predicted_value, deployment

    if variant == "all_observed_hull":
        anchors = train_packet.base.w
        predicted_value, _coeffs, deployment = optimize_generic_family_convex_hull(
            model,
            anchors,
            start_indices=np.arange(min(8, anchors.shape[0]), dtype=int),
        )
        return predicted_value, deployment

    raise ValueError(f"Unsupported variant: {variant}")


def _curve_points(variant: str) -> pd.DataFrame:
    packet = load_packet()
    anchors = load_reference_state()
    subset_indices = load_subset_indices()
    valid_weights = np.stack([anchors.validated_global_weights, anchors.validated_pair_weights], axis=0)
    valid_y = np.asarray([anchors.validated_global_bpb, anchors.validated_pair_bpb], dtype=float)

    best_full_idx = int(np.argmin(packet.base.y))
    best_observed_bpb = float(packet.base.y[best_full_idx])
    previous_deployment: np.ndarray | None = None
    rows: list[dict[str, object]] = []

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
        predicted_optimum_value, deployment = _deployment(variant, train_packet, model)
        fullswarm_predictions = model.predict(packet.base.w)
        chosen_idx = int(np.argmin(fullswarm_predictions))
        distances = average_phase_tv_distance(packet.base.w, deployment[None, :, :])
        nearest_idx = int(np.argmin(distances))

        rows.append(
            {
                "subset_size": subset_size,
                "variant": variant,
                "predicted_optimum_value": float(predicted_optimum_value),
                "fullswarm_chosen_run_name": packet.base.run_names[chosen_idx],
                "fullswarm_chosen_value": float(packet.base.y[chosen_idx]),
                "fullswarm_regret_at_1": float(packet.base.y[chosen_idx] - best_observed_bpb),
                "nearest_observed_run_name": packet.base.run_names[nearest_idx],
                "nearest_observed_value": float(packet.base.y[nearest_idx]),
                "nearest_observed_tv_distance": float(distances[nearest_idx]),
                "optimum_move_mean_phase_tv_vs_prev": (
                    None if previous_deployment is None else mean_phase_tv_distance(deployment, previous_deployment)
                ),
                "tuning_cv_rmse": float(tuning_metrics["cv_rmse"]),
                "tuning_cv_r2": float(tuning_metrics["cv_r2"]),
                "tuning_cv_regret_at_1": float(tuning_metrics["cv_regret_at_1"]),
                "tuning_cv_foldmean_regret_at_1": float(tuning_metrics["cv_foldmean_regret_at_1"]),
                "alpha": tuned_params["alpha"],
                "eta": tuned_params["eta"],
                "lam": tuned_params["lam"],
                "tau": tuned_params["tau"],
                "reg": tuned_params["reg"],
                "beta": tuned_params["beta"],
            }
        )
        previous_deployment = deployment

    return pd.DataFrame(rows)


def _plot(frame: pd.DataFrame, plot_path: Path, *, best_observed_bpb: float) -> None:
    cmap = plt.colormaps["RdYlGn_r"]
    fig, (ax_bpb, ax_regret, ax_cvregret, ax_move) = plt.subplots(
        4,
        1,
        figsize=(10.2, 10.0),
        dpi=180,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.3, 1.0, 1.0, 1.0], "hspace": 0.08},
    )

    ax_bpb.plot(
        frame["subset_size"],
        frame["predicted_optimum_value"],
        color=cmap(0.18),
        marker="o",
        linewidth=2.2,
        label="Predicted deployment BPB",
    )
    ax_bpb.axhline(
        best_observed_bpb,
        color=cmap(0.55),
        linewidth=1.8,
        linestyle=":",
        label=f"Best observed actual BPB ({best_observed_bpb:.4f})",
    )
    ax_regret.plot(
        frame["subset_size"],
        frame["fullswarm_regret_at_1"],
        color=cmap(0.82),
        marker="s",
        linewidth=2.2,
        label="Retrospective Regret@1",
    )
    ax_cvregret.plot(
        frame["subset_size"],
        frame["tuning_cv_foldmean_regret_at_1"],
        color=cmap(0.68),
        marker="^",
        linewidth=2.2,
        label="CV Fold-Mean Regret@1",
    )
    ax_move.plot(
        frame["subset_size"],
        frame["optimum_move_mean_phase_tv_vs_prev"],
        color=cmap(0.36),
        marker="D",
        linewidth=2.2,
        label="Deployment movement (mean phase TV)",
    )
    ax_move.axhline(0.0, color="0.55", linewidth=1.0, linestyle=":")

    ax_bpb.set_title(f"GRP convergence ({frame['variant'].iloc[0]})")
    ax_bpb.set_ylabel("Predicted BPB")
    ax_regret.set_ylabel("Regret@1")
    ax_cvregret.set_ylabel("CV Mean Regret@1")
    ax_move.set_ylabel("Mean phase TV")
    ax_move.set_xlabel("Observed runs used for fitting")
    ax_move.set_xticks(list(SUBSET_SIZES))
    ax_move.set_xlim(min(SUBSET_SIZES), max(SUBSET_SIZES))

    for axis in (ax_bpb, ax_regret, ax_cvregret, ax_move):
        axis.grid(True, alpha=0.25)
        handles = axis.get_lines()
        labels = [handle.get_label() for handle in handles if not handle.get_label().startswith("_")]
        axis.legend(handles, labels, loc="best", frameon=True)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("raw_retuned", "top8actual_hull", "all_observed_hull"),
        default="top8actual_hull",
    )
    args = parser.parse_args()

    packet = load_packet()
    frame = _curve_points(args.variant).sort_values("subset_size")
    output_root = Path(__file__).resolve().parents[1] / "reference_outputs"
    curve_csv = output_root / f"convergence_{args.variant}.csv"
    summary_json = output_root / f"convergence_{args.variant}.json"
    plot_path = output_root / f"convergence_{args.variant}.png"

    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(curve_csv, index=False)
    _plot(frame, plot_path, best_observed_bpb=float(np.min(packet.base.y)))
    summary_json.write_text(
        json.dumps(
            {
                "variant": args.variant,
                "curve_points_csv": str(curve_csv),
                "plot": str(plot_path),
                "best_observed_bpb": float(np.min(packet.base.y)),
                "rows": frame.to_dict(orient="records"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
