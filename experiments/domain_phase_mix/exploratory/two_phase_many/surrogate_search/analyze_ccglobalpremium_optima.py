# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze and plot the many-domain CCGlobalPremium optima."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.domain_phase_mix.static_batch_selection import average_phase_tv_distance
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    MANY_DOMAIN_TARGET,
    PENALTY_KIND_GROUP_LOG_THRESHOLD,
    PREMIUM_MODE_GLOBAL,
    SIGNAL_KIND_RETAINED_TOTAL,
    SIGNAL_KIND_THRESHOLD_TOTAL,
    evaluate_cc_model,
    load_two_phase_many_packet,
    optimize_cc_globalpremium_model,
)

SCRIPT_DIR = Path(__file__).resolve().parent
SUMMARY_JSON = SCRIPT_DIR / "ccglobalpremium_optima_summary.json"
SUMMARY_CSV = SCRIPT_DIR / "ccglobalpremium_optima_summary.csv"
WEIGHTS_CSV = SCRIPT_DIR / "ccglobalpremium_optima_weights.csv"
PLOT_PNG = SCRIPT_DIR / "ccglobalpremium_optima_comparison.png"


def _phase_entropy(weights: np.ndarray) -> float:
    clipped = np.clip(np.asarray(weights, dtype=float), 1e-12, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def _row_weights(frame: pd.DataFrame, domain_names: list[str], row_idx: int) -> np.ndarray:
    row = frame.iloc[row_idx]
    return np.asarray(
        [
            [float(row[f"phase_0_{domain_name}"]) for domain_name in domain_names],
            [float(row[f"phase_1_{domain_name}"]) for domain_name in domain_names],
        ],
        dtype=float,
    )


def _domain_plot_order(domain_names: list[str]) -> list[int]:
    cc_topics = sorted(
        {
            domain_name[len("dolma3_cc/") : -len("_high")]
            for domain_name in domain_names
            if domain_name.startswith("dolma3_cc/") and domain_name.endswith("_high")
        }
    )
    cc_order = []
    name_to_index = {domain_name: idx for idx, domain_name in enumerate(domain_names)}
    for topic in cc_topics:
        cc_order.append(name_to_index[f"dolma3_cc/{topic}_high"])
        cc_order.append(name_to_index[f"dolma3_cc/{topic}_low"])

    cc_indices = set(cc_order)
    non_cc_order = [idx for idx, domain_name in enumerate(domain_names) if idx not in cc_indices]
    return non_cc_order + cc_order


def _display_domain_label(domain_name: str) -> str:
    if domain_name.startswith("dolma3_cc/"):
        topic, quality = domain_name[len("dolma3_cc/") :].rsplit("_", maxsplit=1)
        return f"cc/{topic} {quality}"
    return domain_name.replace("dolma3_", "").replace("dolmino_", "")


def _params_threshold() -> dict[str, float | str]:
    return {
        "signal_kind": SIGNAL_KIND_THRESHOLD_TOTAL,
        "alpha": 8.0,
        "eta": 5.0,
        "lam": 0.0,
        "sig_tau": 0.25,
        "premium_mode": PREMIUM_MODE_GLOBAL,
        "pen_kind": PENALTY_KIND_GROUP_LOG_THRESHOLD,
        "tau": 2.0,
        "reg": 0.01,
    }


def _params_retained() -> dict[str, float | str]:
    return {
        "signal_kind": SIGNAL_KIND_RETAINED_TOTAL,
        "alpha": 8.0,
        "eta": 3.0,
        "lam": 1.0,
        "sig_tau": 0.0,
        "premium_mode": PREMIUM_MODE_GLOBAL,
        "pen_kind": PENALTY_KIND_GROUP_LOG_THRESHOLD,
        "tau": 1.0,
        "reg": 0.01,
    }


def _analyze_one(
    *,
    data,
    model_name: str,
    params: dict[str, float | str],
    observed_weights: np.ndarray,
    best_idx: int,
) -> tuple[dict[str, object], np.ndarray]:
    row, model = evaluate_cc_model(data, model_name, params)
    result, phase0, phase1 = optimize_cc_globalpremium_model(model, data, seed=0)
    optimum = np.stack([phase0, phase1], axis=0)
    distances = average_phase_tv_distance(observed_weights, optimum[None, :, :])
    nearest_idx = int(np.argmin(distances))
    best_value = float(data.y[best_idx])

    top_phase0 = (
        pd.DataFrame(
            {
                "domain": data.domain_names,
                "weight": phase0,
                "epochs": phase0 * data.c0,
            }
        )
        .sort_values(["weight", "epochs"], ascending=False)
        .head(10)
        .to_dict(orient="records")
    )
    top_phase1 = (
        pd.DataFrame(
            {
                "domain": data.domain_names,
                "weight": phase1,
                "epochs": phase1 * data.c1,
            }
        )
        .sort_values(["weight", "epochs"], ascending=False)
        .head(10)
        .to_dict(orient="records")
    )

    summary = {
        "model": model_name,
        "objective_metric": MANY_DOMAIN_TARGET,
        "predicted_optimum_value": float(result.fun),
        "observed_best_run_name": str(data.frame.iloc[best_idx][data.name_col]),
        "observed_best_value": best_value,
        "gap_below_observed_best": float(result.fun) - best_value,
        "phase0_max_weight": float(np.max(phase0)),
        "phase1_max_weight": float(np.max(phase1)),
        "phase0_entropy": _phase_entropy(phase0),
        "phase1_entropy": _phase_entropy(phase1),
        "nearest_observed_run_name": str(data.frame.iloc[nearest_idx][data.name_col]),
        "nearest_observed_value": float(data.y[nearest_idx]),
        "nearest_observed_tv_distance": float(distances[nearest_idx]),
        "top_phase0_domains": top_phase0,
        "top_phase1_domains": top_phase1,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "cv_rmse": float(row["cv_rmse"]),
        "cv_regret_at_1": float(row["cv_regret_at_1"]),
    }
    return summary, optimum


def _plot(optima: dict[str, np.ndarray], best_name: str, best_weights: np.ndarray, domain_names: list[str]) -> None:
    schedules = [
        optima["CCGlobalPremium-Threshold"],
        optima["CCGlobalPremium-RetainedTotal"],
        best_weights,
    ]
    cmap = plt.get_cmap("RdYlGn_r")
    colors = [cmap(0.9), cmap(0.15), cmap(0.55)]
    labels = ["Threshold", "RetainedTotal", f"Best observed\n{best_name}"]
    ordered_indices = _domain_plot_order(domain_names)
    ordered_labels = [_display_domain_label(domain_names[idx]) for idx in ordered_indices]

    fig, axes = plt.subplots(1, 2, figsize=(18, 15), sharey=True)
    bar_height = 0.22

    for phase_idx, ax in enumerate(axes):
        weight_vectors = [schedule[phase_idx] for schedule in schedules]
        y = np.arange(len(ordered_indices))
        for schedule_idx, (weights, color, label) in enumerate(zip(weight_vectors, colors, labels, strict=True)):
            ax.barh(
                y + (schedule_idx - 1) * bar_height,
                weights[ordered_indices],
                height=bar_height,
                color=color,
                alpha=0.9,
                label=label if phase_idx == 0 else None,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(ordered_labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mixture weight")
        ax.set_title(f"Phase {phase_idx}")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

    axes[0].legend(loc="lower right", frameon=False)
    fig.suptitle("Many-domain uncheatable BPB: CCGlobalPremium predicted optima vs best observed", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(PLOT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = load_two_phase_many_packet(target=MANY_DOMAIN_TARGET)
    observed_weights = data.w
    best_idx = int(np.argmin(data.y))
    best_name = str(data.frame.iloc[best_idx][data.name_col])
    best_weights = _row_weights(data.frame, data.domain_names, best_idx)

    threshold_summary, threshold_optimum = _analyze_one(
        data=data,
        model_name="CCGlobalPremium-Threshold",
        params=_params_threshold(),
        observed_weights=observed_weights,
        best_idx=best_idx,
    )
    retained_summary, retained_optimum = _analyze_one(
        data=data,
        model_name="CCGlobalPremium-RetainedTotal",
        params=_params_retained(),
        observed_weights=observed_weights,
        best_idx=best_idx,
    )

    weights_rows = []
    for model_name, optimum in [
        ("CCGlobalPremium-Threshold", threshold_optimum),
        ("CCGlobalPremium-RetainedTotal", retained_optimum),
    ]:
        for domain_name, phase0, phase1, c0, c1 in zip(
            data.domain_names, optimum[0], optimum[1], data.c0, data.c1, strict=True
        ):
            weights_rows.append(
                {
                    "model": model_name,
                    "domain": domain_name,
                    "phase0_weight": float(phase0),
                    "phase0_epochs": float(phase0 * c0),
                    "phase1_weight": float(phase1),
                    "phase1_epochs": float(phase1 * c1),
                }
            )
    pd.DataFrame(weights_rows).to_csv(WEIGHTS_CSV, index=False)

    summary_rows = [threshold_summary, retained_summary]
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)
    SUMMARY_JSON.write_text(json.dumps(summary_rows, indent=2))

    _plot(
        {
            "CCGlobalPremium-Threshold": threshold_optimum,
            "CCGlobalPremium-RetainedTotal": retained_optimum,
        },
        best_name,
        best_weights,
        data.domain_names,
    )

    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"Plot: {PLOT_PNG}")


if __name__ == "__main__":
    main()
