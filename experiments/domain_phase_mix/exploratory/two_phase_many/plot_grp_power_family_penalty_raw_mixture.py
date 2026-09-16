# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "matplotlib", "numpy", "pandas"]
# ///
"""Plot the raw-optimum mixture for the power-family-penalty GRP variant."""

from __future__ import annotations

import json
from pathlib import Path

import fsspec
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.plot_grp_vs_proportional import (
    TEXT_MUTED_COLOR,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_penalty_raw_optima_baselines import (
    GENERICFAMILY_PENALTY_RAW_OPTIMA_SOURCE_EXPERIMENT,
    genericfamily_penalty_raw_optimum_summary,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_PNG = SCRIPT_DIR / "grp_power_family_penalty_raw_mixture.png"
SUMMARY_JSON = SCRIPT_DIR / "grp_power_family_penalty_raw_mixture_summary.json"
WEIGHTS_CSV = SCRIPT_DIR / "grp_power_family_penalty_raw_mixture_weights.csv"

VARIANT_NAME = "power_family_penalty"
RUN_NAME = "baseline_genericfamily_power_family_penalty_raw_optimum"
CHECKPOINT_ROOT = "marin-us-east5/checkpoints/" + GENERICFAMILY_PENALTY_RAW_OPTIMA_SOURCE_EXPERIMENT
TOP_K = 14
BAR_EDGE_COLOR = "#0f172a"
GRID_COLOR = "#cbd5e1"
FAMILY_ORDER = ("broad_text", "tech_code", "reasoning")


def _domain_label(domain_name: str) -> str:
    if domain_name.startswith("dolma3_cc/"):
        topic, quality = domain_name[len("dolma3_cc/") :].rsplit("_", maxsplit=1)
        return f"cc/{topic.replace('_', ' ')} {quality}"
    return domain_name.removeprefix("dolma3_").removeprefix("dolmino_").replace("_", " ")


def _weights_array(
    phase_weights: dict[str, dict[str, float]],
    domain_names: list[str],
) -> np.ndarray:
    return np.asarray(
        [
            [float(phase_weights["phase_0"][domain_name]) for domain_name in domain_names],
            [float(phase_weights["phase_1"][domain_name]) for domain_name in domain_names],
        ],
        dtype=float,
    )


def _validated_bpb() -> float | None:
    fs = fsspec.filesystem("gs")
    matches = sorted(fs.glob(f"{CHECKPOINT_ROOT}/{RUN_NAME}-*/checkpoints/eval_metrics.jsonl"))
    if not matches:
        return None
    payload: dict[str, float] | None = None
    with fs.open(matches[-1], "r") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
    if payload is None:
        return None
    value = payload.get("eval/uncheatable_eval/bpb")
    return None if value is None else float(value)


def _top_indices(weights: np.ndarray, top_k: int) -> np.ndarray:
    return np.argsort(-weights)[:top_k]


def _plot_phase(ax, *, weights: np.ndarray, domain_names: list[str], phase_idx: int, color: tuple[float, ...]) -> None:
    top_idx = _top_indices(weights[phase_idx], TOP_K)
    labels = [_domain_label(domain_names[idx]) for idx in top_idx]
    values = weights[phase_idx, top_idx]
    y = np.arange(len(top_idx))
    ax.barh(
        y,
        values,
        color=color,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.45,
        alpha=0.96,
    )
    for row_idx, value in enumerate(values):
        ax.text(
            float(value) + max(0.0016, float(np.max(values)) * 0.03),
            row_idx,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=10.5,
            color=BAR_EDGE_COLOR,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mixture Weight", fontsize=11.5, fontweight="bold")
    ax.set_title(f"Phase {phase_idx}: top-weight domains", fontsize=14.5, fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle="--", color=GRID_COLOR, alpha=0.65, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_family_shares(ax, family_shares: dict[str, float], cmap) -> None:
    phases = ("phase0", "phase1")
    x = np.arange(len(phases))
    width = 0.22
    for idx, family_name in enumerate(FAMILY_ORDER):
        values = [float(family_shares[f"{phase}_{family_name}"]) for phase in phases]
        ax.bar(
            x + (idx - 1) * width,
            values,
            width=width,
            color=cmap(0.2 + 0.28 * idx),
            edgecolor=BAR_EDGE_COLOR,
            linewidth=0.4,
            label=family_name.replace("_", " "),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["phase 0", "phase 1"], fontsize=11)
    ax.set_ylabel("Family Share", fontsize=11.5, fontweight="bold")
    ax.set_title("Family shares", fontsize=14.5, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", color=GRID_COLOR, alpha=0.65, linewidth=0.8)
    ax.legend(loc="best", frameon=True, fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    summary = genericfamily_penalty_raw_optimum_summary(VARIANT_NAME)
    domain_names = list(summary.phase_weights["phase_0"])
    weights = _weights_array(summary.phase_weights, domain_names)
    validated_bpb = _validated_bpb()
    cmap = plt.get_cmap("RdYlGn_r")

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 1.0], hspace=0.26, wspace=0.24)
    ax_phase0 = fig.add_subplot(grid[0, 0])
    ax_phase1 = fig.add_subplot(grid[1, 0])
    ax_family = fig.add_subplot(grid[0, 1])
    ax_text = fig.add_subplot(grid[1, 1])

    _plot_phase(ax_phase0, weights=weights, domain_names=domain_names, phase_idx=0, color=cmap(0.22))
    _plot_phase(ax_phase1, weights=weights, domain_names=domain_names, phase_idx=1, color=cmap(0.72))
    _plot_family_shares(ax_family, summary.family_shares, cmap)

    fig.suptitle("GRP power-family penalty raw optimum", fontsize=22, fontweight="bold", y=0.99)
    subtitle = f"Predicted BPB {summary.raw_predicted_optimum_value:.6f}"
    if validated_bpb is not None:
        subtitle += f"   |   validated BPB {validated_bpb:.6f}"
    fig.text(0.5, 0.958, subtitle, ha="center", va="center", fontsize=13, color=TEXT_MUTED_COLOR)

    ax_text.axis("off")
    optimism_gap = None if validated_bpb is None else validated_bpb - summary.raw_predicted_optimum_value
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                f"Variant: {summary.variant_name}",
                f"Raw predicted BPB: {summary.raw_predicted_optimum_value:.6f}",
                (
                    "Validated BPB: pending"
                    if validated_bpb is None
                    else f"Validated BPB: {validated_bpb:.6f} (gap +{optimism_gap:.6f})"
                ),
                f"Nearest observed run: {summary.nearest_observed_run_name}",
                f"Nearest observed BPB: {summary.nearest_observed_value:.6f}",
                f"Nearest observed TV: {summary.nearest_observed_tv_distance:.3f}",
                f"Tuning objective: {summary.tuning_objective:.6f}",
                f"CV RMSE: {summary.tuning_cv_rmse:.6f}",
                f"CV DepOpt@8: {summary.tuning_cv_depopt_best8:.6f}",
                (
                    "Raw support below 1e-4: "
                    f"phase0={int(np.sum(weights[0] < 1e-4))}, "
                    f"phase1={int(np.sum(weights[1] < 1e-4))}"
                ),
            ]
        ),
        ha="left",
        va="top",
        fontsize=12,
        color=BAR_EDGE_COLOR,
        family="monospace",
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(PLOT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)

    weight_rows: list[dict[str, object]] = []
    for phase_idx, phase_name in enumerate(("phase_0", "phase_1")):
        for domain_idx, domain_name in enumerate(domain_names):
            weight_rows.append(
                {
                    "variant": VARIANT_NAME,
                    "phase": phase_name,
                    "domain_name": domain_name,
                    "weight": float(weights[phase_idx, domain_idx]),
                }
            )
    pd.DataFrame(weight_rows).to_csv(WEIGHTS_CSV, index=False)

    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "variant": VARIANT_NAME,
                "raw_predicted_bpb": summary.raw_predicted_optimum_value,
                "validated_bpb": validated_bpb,
                "nearest_observed_run_name": summary.nearest_observed_run_name,
                "nearest_observed_bpb": summary.nearest_observed_value,
                "nearest_observed_tv": summary.nearest_observed_tv_distance,
                "family_shares": summary.family_shares,
                "plot_png": str(PLOT_PNG),
                "weights_csv": str(WEIGHTS_CSV),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
