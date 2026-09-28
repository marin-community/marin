# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "matplotlib", "numpy", "pandas"]
# ///
"""Plot and summarize raw-optimum GRP penalty variants."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import fsspec
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    load_generic_family_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_penalty_calibration import (
    build_penalty_calibration_surrogate,
    optimize_penalty_calibration_model,
    penalty_calibration_variant_parameter_counts,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.plot_grp_vs_proportional import (
    TEXT_MUTED_COLOR,
)
from experiments.domain_phase_mix.static_batch_selection import average_phase_tv_distance

SCRIPT_DIR = Path(__file__).resolve().parent
BEST_CSV = SCRIPT_DIR / "grp_penalty_calibration_variants_best.csv"
PLOT_PNG = SCRIPT_DIR / "grp_penalty_raw_optima_comparison.png"
SUMMARY_JSON = SCRIPT_DIR / "grp_penalty_raw_optima_comparison_summary.json"
WEIGHTS_CSV = SCRIPT_DIR / "grp_penalty_raw_optima_comparison_weights.csv"

CHECKPOINT_ROOT = "marin-us-east5/checkpoints"
SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_genericfamily_penalty_raw_optima_uncheatable_bpb"
VARIANTS = (
    "power_family_penalty",
    "power_boxcox_family_penalty",
    "power_family_penalty_global_ftotal_pairces",
)
VALIDATED_RUN_NAMES = {
    "power_family_penalty": "baseline_genericfamily_power_family_penalty_raw_optimum",
    "power_family_penalty_global_ftotal_pairces": (
        "baseline_genericfamily_power_family_penalty_global_ftotal_pairces_raw_optimum"
    ),
}
DISPLAY_NAMES = {
    "power_family_penalty": "power-family penalty",
    "power_boxcox_family_penalty": "power/Box-Cox family penalty",
    "power_family_penalty_global_ftotal_pairces": "power-family penalty + pair-CES",
}
BAR_EDGE_COLOR = "#0f172a"
GRID_COLOR = "#cbd5e1"
TOP_K_PER_PHASE = 12


@dataclass(frozen=True)
class VariantSummary:
    variant: str
    display_name: str
    raw_predicted_bpb: float
    validated_bpb: float | None
    nearest_observed_run_name: str
    nearest_observed_bpb: float
    nearest_observed_tv: float
    total_param_count: int
    weights: np.ndarray


def _domain_label(domain_name: str) -> str:
    if domain_name.startswith("dolma3_cc/"):
        topic, quality = domain_name[len("dolma3_cc/") :].rsplit("_", maxsplit=1)
        return f"cc/{topic.replace('_', ' ')} {quality}"
    label = domain_name.removeprefix("dolma3_").removeprefix("dolmino_")
    return label.replace("_", " ")


def _params_from_row(row: dict[str, str]) -> dict[str, float]:
    keys = (
        "eta",
        "lam",
        "reg",
        "beta",
        "alpha",
        "tau",
        "tau_broad_text",
        "tau_tech_code",
        "tau_reasoning",
        "a_broad_text",
        "a_tech_code",
        "a_reasoning",
        "pair_rho",
    )
    return {key: float(row[key]) for key in keys if row.get(key, "") != ""}


def _validated_bpb(fs: fsspec.AbstractFileSystem, variant: str) -> float | None:
    run_name = VALIDATED_RUN_NAMES.get(variant)
    if run_name is None:
        return None
    pattern = f"{CHECKPOINT_ROOT}/{SOURCE_EXPERIMENT}/{run_name}-*/checkpoints/eval_metrics.jsonl"
    matches = sorted(fs.glob(pattern))
    if not matches:
        return None
    last_payload: dict[str, object] | None = None
    with fs.open(matches[-1], "r") as handle:
        for line in handle:
            if line.strip():
                last_payload = json.loads(line)
    if last_payload is None:
        return None
    value = last_payload.get("eval/uncheatable_eval/bpb")
    return None if value is None else float(value)


def _variant_summaries() -> list[VariantSummary]:
    with BEST_CSV.open() as handle:
        rows = list(csv.DictReader(handle))
    packet = load_generic_family_packet()
    fs = fsspec.filesystem("gs")

    summaries: list[VariantSummary] = []
    for variant in VARIANTS:
        row = next(candidate for candidate in rows if candidate["variant"] == variant and candidate["stage"] == "refine")
        params = _params_from_row(row)
        model = build_penalty_calibration_surrogate(packet, params=params, variant_name=variant).fit(
            packet.base.w, packet.base.y
        )
        result, phase0, phase1 = optimize_penalty_calibration_model(packet, model, seed=0)
        weights = np.stack([phase0, phase1], axis=0)
        distances = average_phase_tv_distance(packet.base.w, weights[None, :, :])
        nearest_idx = int(np.argmin(distances))
        counts = penalty_calibration_variant_parameter_counts(packet, variant)
        summaries.append(
            VariantSummary(
                variant=variant,
                display_name=DISPLAY_NAMES[variant],
                raw_predicted_bpb=float(result.fun),
                validated_bpb=_validated_bpb(fs, variant),
                nearest_observed_run_name=str(packet.base.frame.iloc[nearest_idx][packet.base.name_col]),
                nearest_observed_bpb=float(packet.base.y[nearest_idx]),
                nearest_observed_tv=float(distances[nearest_idx]),
                total_param_count=int(counts["total_param_count"]),
                weights=weights,
            )
        )
    return summaries


def _pairwise_phase_tv(summaries: list[VariantSummary]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(summaries)
    avg_phase_tv = np.zeros((n, n), dtype=float)
    phase0_tv = np.zeros((n, n), dtype=float)
    phase1_tv = np.zeros((n, n), dtype=float)
    for i, left in enumerate(summaries):
        for j, right in enumerate(summaries):
            phase0_tv[i, j] = 0.5 * np.abs(left.weights[0] - right.weights[0]).sum()
            phase1_tv[i, j] = 0.5 * np.abs(left.weights[1] - right.weights[1]).sum()
            avg_phase_tv[i, j] = 0.5 * (phase0_tv[i, j] + phase1_tv[i, j])
    return avg_phase_tv, phase0_tv, phase1_tv


def _phase_top_indices(summaries: list[VariantSummary], phase_idx: int) -> np.ndarray:
    stacked = np.stack([summary.weights[phase_idx] for summary in summaries], axis=0)
    scores = np.max(stacked, axis=0)
    return np.argsort(-scores)[:TOP_K_PER_PHASE]


def _variant_colors(summaries: list[VariantSummary]) -> dict[str, tuple[float, float, float, float]]:
    raw_preds = np.asarray([summary.raw_predicted_bpb for summary in summaries], dtype=float)
    norm = Normalize(vmin=float(raw_preds.min()), vmax=float(raw_preds.max()))
    cmap = plt.get_cmap("RdYlGn_r")
    return {summary.variant: cmap(norm(summary.raw_predicted_bpb)) for summary in summaries}


def _plot_phase_bars(ax, summaries: list[VariantSummary], *, phase_idx: int, domain_names: list[str]) -> None:
    indices = _phase_top_indices(summaries, phase_idx)
    labels = [_domain_label(domain_names[idx]) for idx in indices]
    colors = _variant_colors(summaries)
    y = np.arange(len(indices))
    bar_height = 0.22
    x_max = max(float(np.max(summary.weights[phase_idx, indices])) for summary in summaries)
    x_max = max(x_max * 1.25, 0.08)

    for offset_idx, summary in enumerate(summaries):
        offsets = y + (offset_idx - (len(summaries) - 1) / 2.0) * bar_height
        ax.barh(
            offsets,
            summary.weights[phase_idx, indices],
            height=bar_height,
            color=colors[summary.variant],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=0.4,
            alpha=0.96,
            label=summary.display_name,
        )

    ax.set_xlim(0.0, x_max)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mixture Weight", fontsize=12.5, fontweight="bold")
    ax.set_title(f"Phase {phase_idx}: top-weight domains", fontsize=16, fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle="--", color=GRID_COLOR, alpha=0.65, linewidth=0.8)
    ax.tick_params(axis="x", labelsize=11.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_tv_heatmap(ax, matrix: np.ndarray, labels: list[str]) -> None:
    image = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0.0, vmax=float(np.max(matrix)))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10.5)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_title("Pairwise average phase TV distance", fontsize=16, fontweight="bold", pad=10)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=11, color=BAR_EDGE_COLOR)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _summary_table_rows(summaries: list[VariantSummary]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary in summaries:
        rows.append(
            {
                "variant": summary.variant,
                "display_name": summary.display_name,
                "raw_predicted_bpb": summary.raw_predicted_bpb,
                "validated_bpb": summary.validated_bpb,
                "optimism_gap": (
                    None if summary.validated_bpb is None else summary.validated_bpb - summary.raw_predicted_bpb
                ),
                "nearest_observed_run_name": summary.nearest_observed_run_name,
                "nearest_observed_bpb": summary.nearest_observed_bpb,
                "nearest_observed_tv": summary.nearest_observed_tv,
                "total_param_count": summary.total_param_count,
            }
        )
    return rows


def main() -> None:
    summaries = _variant_summaries()
    packet = load_generic_family_packet()
    avg_phase_tv, phase0_tv, phase1_tv = _pairwise_phase_tv(summaries)
    labels = [summary.display_name for summary in summaries]

    fig = plt.figure(figsize=(22, 15), facecolor="white")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 1.0], hspace=0.24, wspace=0.22)

    ax_phase0 = fig.add_subplot(grid[0, 0])
    ax_phase1 = fig.add_subplot(grid[1, 0])
    ax_heatmap = fig.add_subplot(grid[0, 1])
    ax_text = fig.add_subplot(grid[1, 1])

    _plot_phase_bars(ax_phase0, summaries, phase_idx=0, domain_names=packet.base.domain_names)
    _plot_phase_bars(ax_phase1, summaries, phase_idx=1, domain_names=packet.base.domain_names)
    _plot_tv_heatmap(ax_heatmap, avg_phase_tv, labels)

    handles, legend_labels = ax_phase0.get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper center", ncol=3, frameon=False, fontsize=13.5, bbox_to_anchor=(0.5, 0.962)
    )
    fig.suptitle("GRP Penalty Raw-Optimum Comparison", fontsize=24, fontweight="bold", y=0.99)
    fig.text(
        0.5,
        0.962,
        "Raw predicted optima for three GRP follow-up variants, with finished validations where available",
        ha="center",
        va="center",
        fontsize=14,
        color=TEXT_MUTED_COLOR,
    )

    ax_text.axis("off")
    summary_lines = []
    for summary in summaries:
        validated = "pending"
        if summary.validated_bpb is not None:
            validated = f"{summary.validated_bpb:.6f} (gap +{summary.validated_bpb - summary.raw_predicted_bpb:.6f})"
        summary_lines.append(
            f"{summary.display_name}\n"
            f"  raw pred: {summary.raw_predicted_bpb:.6f}    validated: {validated}\n"
            f"  nearest support: {summary.nearest_observed_run_name} @ {summary.nearest_observed_bpb:.6f}"
            f"    TV={summary.nearest_observed_tv:.3f}    params={summary.total_param_count}"
        )
    summary_lines.append(
        "TV reading:\n"
        "  power/Box-Cox is materially different from the validated power-family raw optimum.\n"
        "  It is closer to neither winner than the two validated variants are to each other."
    )
    ax_text.text(
        0.0,
        1.0,
        "\n\n".join(summary_lines),
        ha="left",
        va="top",
        fontsize=12.5,
        color=BAR_EDGE_COLOR,
        family="monospace",
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(PLOT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)

    weight_rows: list[dict[str, object]] = []
    for summary in summaries:
        for phase_idx, phase_name in enumerate(("phase_0", "phase_1")):
            for domain_idx, domain_name in enumerate(packet.base.domain_names):
                weight_rows.append(
                    {
                        "variant": summary.variant,
                        "display_name": summary.display_name,
                        "phase": phase_name,
                        "domain_name": domain_name,
                        "weight": float(summary.weights[phase_idx, domain_idx]),
                    }
                )
    pd.DataFrame(weight_rows).to_csv(WEIGHTS_CSV, index=False)

    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "variant_rows": _summary_table_rows(summaries),
                "pairwise_avg_phase_tv": avg_phase_tv.tolist(),
                "pairwise_phase0_tv": phase0_tv.tolist(),
                "pairwise_phase1_tv": phase1_tv.tolist(),
                "plot_png": str(PLOT_PNG),
                "weights_csv": str(WEIGHTS_CSV),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
