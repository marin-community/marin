# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9"]
# ///
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Plot Uncheatable component changes and three-component optimum calibration.

Run with uv run experiments/domain_phase_mix/exploratory/two_phase_many/plot_component_tradeoffs_20260908.py
--paper-dir /path/to/data_mixing_paper_one_phase. Reads archived local observations and frozen predictions;
does not fit a model or fetch measurements. Outputs exact plotted values and source hashes beside the figure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from statistics import mean, stdev

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

SOURCE_ROOT = Path(__file__).resolve().parent / "reference_outputs"
OUTPUT_DIR = SOURCE_ROOT / "component_tradeoffs_20260908"
THREE_DIR = SOURCE_ROOT / "delphi_three_component_optimum_3e18_20260908"
FIGURE_STEM = "r8_component_tradeoffs"
FIGURE_SIZE = (5.5, 2.05)
MATCHED_FIGURE_SIZE = (5.5, 2.65)
FULL_CANDIDATE = "lwspu_u_snc_cap06"
COLORS = {"Code": "#0072B2", "Scientific": "#CC79A7", "Broad language": "#009E73"}
COMPONENTS = (
    ("github_cpp", "GitHub C++", "Code"),
    ("github_python", "GitHub Python", "Code"),
    ("arxiv_physics", "arXiv physics", "Scientific"),
    ("arxiv_computer_science", "arXiv CS", "Scientific"),
    ("ao3_english", "AO3", "Broad language"),
    ("wikipedia_english", "Wikipedia", "Broad language"),
    ("bbc_news", "BBC News", "Broad language"),
)
SOURCES = {
    "proportional": SOURCE_ROOT / "table9_reliability_20260905/proportional_uncheatable_components.csv",
    "full_optimum_seed0": SOURCE_ROOT / "delphi_frozen_procedure_validation_3e18_20260908/measured_results.csv",
    "full_optimum_seeds12": SOURCE_ROOT / "delphi_fairness_repeats_3e18_20260908/measured_results.csv",
    "byte_weights": SOURCE_ROOT / "delphi_one_phase_wspu_worsened_components_sweep_20260905/reference_predictions.csv",
    "three_measured": THREE_DIR / "measured_results.csv",
    "three_predicted_components": THREE_DIR / "predicted_uncheatable_components.csv",
    "three_predicted_objective": THREE_DIR / "predictions_uncheatable_worsened.csv",
    "three_predicted_full": THREE_DIR / "predictions_uncheatable.csv",
    "three_objective_weights": THREE_DIR / "objectives.csv",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_values(
    layout: str = "summary",
) -> tuple[list[dict[str, str | float | int]], list[dict[str, str | float | int]]]:
    """Compute means from the same eleven proportional and three optimum runs as the paper tables."""
    proportional = read_rows(SOURCES["proportional"])
    full = [
        row
        for key in ("full_optimum_seed0", "full_optimum_seeds12")
        for row in read_rows(SOURCES[key])
        if row["candidate_id"] == FULL_CANDIDATE and row["status"] == "measured"
    ]
    assert len(proportional) == 11 and len(full) == 3
    assert len({row["eval_metrics_uri"] for row in full}) == 3
    byte_weights = {
        row["component"]: float(row["byte_weight"])
        for row in read_rows(SOURCES["byte_weights"])
        if row["mixture"] == "proportional"
    }
    component_rows = []
    for component, label, group in COMPONENTS:
        baseline = mean(float(row[f"eval/uncheatable_eval/{component}/bpb"]) for row in proportional)
        optimum = [float(row[f"uncheatable_{component}_bpb"]) for row in full]
        observed = mean(optimum)
        component_rows.append(
            {
                "component": component,
                "label": label,
                "group": group,
                "byte_weight": byte_weights[component],
                "proportional_mean": baseline,
                "full_optimum_mean": observed,
                "full_optimum_sd": stdev(optimum),
                "delta_bpb": observed - baseline,
                "weighted_delta_bpb": byte_weights[component] * (observed - baseline),
                "proportional_runs": len(proportional),
                "optimum_runs": len(full),
            }
        )

    measured = read_rows(SOURCES["three_measured"])
    assert len(measured) == 3 and all(row["status"] == "measured" for row in measured)
    assert {int(row["trainer_seed"]) for row in measured} == {0, 1, 2}
    predictions = {
        key: next(row for row in read_rows(SOURCES[key]) if row["mixture"] == "three_component_optimum")
        for key in ("three_predicted_components", "three_predicted_objective", "three_predicted_full")
    }
    selected_weights = {
        row["component"].split("/")[2]: float(row["weight"]) for row in read_rows(SOURCES["three_objective_weights"])
    }
    selected_observations = [
        sum(weight * float(row[f"uncheatable_{component}_bpb"]) for component, weight in selected_weights.items())
        for row in measured
    ]
    calibration_inputs = (
        (
            "selected_three",
            "Selected three",
            float(predictions["three_predicted_objective"]["predicted_uncheatable_worsened"]),
            selected_observations,
        ),
        (
            "github_cpp",
            "GitHub C++",
            float(predictions["three_predicted_components"]["github_cpp"]),
            [float(row["uncheatable_github_cpp_bpb"]) for row in measured],
        ),
        (
            "github_python",
            "GitHub Python",
            float(predictions["three_predicted_components"]["github_python"]),
            [float(row["uncheatable_github_python_bpb"]) for row in measured],
        ),
        (
            "full_aggregate",
            "Full aggregate",
            float(predictions["three_predicted_full"]["predicted_uncheatable"]),
            [float(row["uncheatable_bpb"]) for row in measured],
        ),
    )
    if layout == "matched-components":
        baseline = mean(float(row["eval/uncheatable_eval/bpb"]) for row in proportional)
        full_values = [float(row["uncheatable_bpb"]) for row in full]
        observed = mean(full_values)
        component_rows.append(
            {
                "component": "full_aggregate",
                "label": "Full aggregate",
                "group": "Aggregate",
                "byte_weight": 1.0,
                "proportional_mean": baseline,
                "full_optimum_mean": observed,
                "full_optimum_sd": stdev(full_values),
                "delta_bpb": observed - baseline,
                "weighted_delta_bpb": observed - baseline,
                "proportional_runs": len(proportional),
                "optimum_runs": len(full),
            }
        )
        calibration_inputs = (
            *(
                (
                    component,
                    label,
                    float(predictions["three_predicted_components"][component]),
                    [float(row[f"uncheatable_{component}_bpb"]) for row in measured],
                )
                for component, label, _ in COMPONENTS
            ),
            calibration_inputs[-1],
        )
    calibration_rows = [
        {
            "evaluation": component,
            "label": label,
            "predicted_bpb": prediction,
            "observed_bpb": mean(observations),
            "observed_sd": stdev(observations),
            "observed_minus_predicted_bpb": mean(observations) - prediction,
            "runs": len(observations),
        }
        for component, label, prediction, observations in calibration_inputs
    ]
    baseline_components = {row["component"]: row["proportional_mean"] for row in component_rows}
    baselines = {
        **baseline_components,
        "selected_three": sum(weight * baseline_components[component] for component, weight in selected_weights.items()),
        "github_cpp": baseline_components["github_cpp"],
        "github_python": baseline_components["github_python"],
        "full_aggregate": mean(float(row["eval/uncheatable_eval/bpb"]) for row in proportional),
    }
    for row in calibration_rows:
        baseline = baselines[row["evaluation"]]
        row["proportional_mean"] = baseline
        row["observed_delta_bpb"] = row["observed_bpb"] - baseline
        row["predicted_delta_bpb"] = row["predicted_bpb"] - baseline
    full_predictions = next(
        row for row in read_rows(SOURCES["three_predicted_components"]) if row["mixture"] == "full_uncheatable_optimum"
    )
    full_predicted_aggregate = float(
        next(row for row in read_rows(SOURCES["three_predicted_full"]) if row["mixture"] == "full_uncheatable_optimum")[
            "predicted_uncheatable"
        ]
    )
    for row in component_rows:
        prediction = (
            full_predicted_aggregate
            if row["component"] == "full_aggregate"
            else float(full_predictions[row["component"]])
        )
        row["predicted_bpb"] = prediction
        row["predicted_delta_bpb"] = prediction - row["proportional_mean"]
        row["observed_minus_predicted_bpb"] = row["full_optimum_mean"] - prediction
    return component_rows, calibration_rows


def signed_tick(value: float, position: int) -> str:
    return "0" if abs(value) < 1e-10 else f"{value:+.2f}".replace("-", "\u2212")


def draw(
    component_rows: list[dict],
    calibration_rows: list[dict],
    output_dir: Path,
    panel_b: str,
    figure_stem: str,
    layout: str = "summary",
    show_predictions: str = "panel-b",
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.usetex": False,
            "font.size": 7,
            "axes.grid": False,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure_size = MATCHED_FIGURE_SIZE if layout == "matched-components" else FIGURE_SIZE
    fig = plt.figure(figsize=figure_size, facecolor="white")
    if layout == "matched-components":
        left = fig.add_axes((0.245, 0.205, 0.300, 0.650))
        right = fig.add_axes((0.602, 0.205, 0.380, 0.650))
    else:
        left = fig.add_axes((0.245, 0.245, 0.225, 0.585))
        right = fig.add_axes((0.704, 0.245, 0.278, 0.585))
    fig.text(0.02, 0.94, "A · Optimized for Uncheatable Eval", fontsize=8, fontweight="bold")
    if layout == "matched-components":
        fig.text(0.982, 0.94, "B · Optimized for broad language", ha="right", fontsize=8, fontweight="bold")
        fig.text(0.02, 0.885, "Percentages: byte weights in aggregate", fontsize=6.5, color="#444444")
    else:
        fig.text(0.535, 0.94, "B · Optimized for broad language", fontsize=8, fontweight="bold")
    for axis in (left, right):
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#777777")
        axis.spines["bottom"].set_linewidth(0.6)
        axis.tick_params(axis="y", length=0, pad=4)
        axis.tick_params(axis="x", length=2, pad=3, width=0.6)
        axis.axvline(0, color="#555555", linewidth=0.7, zorder=1)
        axis.xaxis.set_major_formatter(FuncFormatter(signed_tick))
        axis.xaxis.grid(True, linewidth=0.45, color="#DDDDDD", zorder=0)
        axis.invert_yaxis()

    positions = [0, 1, 2.3, 3.3, 4.6, 5.6, 6.6]
    if layout == "matched-components":
        positions.append(8.0)
    values = [float(row["delta_bpb"]) for row in component_rows]
    plot_colors = {**COLORS, "Aggregate": "#555555"}
    colors = [plot_colors[str(row["group"])] for row in component_rows]
    left.barh(positions, values, height=0.53, color=colors, zorder=2)
    if show_predictions == "both":
        left_predictions = [float(row["predicted_delta_bpb"]) for row in component_rows]
        left.hlines(positions, left_predictions, values, color="#333333", linewidth=0.7, linestyles="--", zorder=3)
        left.plot(
            left_predictions,
            positions,
            "D",
            color="#333333",
            markerfacecolor="white",
            markersize=3.2,
            markeredgewidth=0.6,
            zorder=4,
        )
    left.set_yticks(
        positions,
        [
            (
                row["label"]
                if row["component"] == "full_aggregate"
                else f"{row['label']} ({100 * float(row['byte_weight']):.1f}%)"
            )
            for row in component_rows
        ],
    )
    left.set_ylim(8.6 if layout == "matched-components" else 7.15, -0.55)
    left.set_xlim(-0.178, 0.060)
    left.set_xticks([-0.15, -0.05, 0.05])
    left.set_xlabel("Change from proportional (BPB)", labelpad=4)

    value_key = "observed_delta_bpb" if panel_b == "change-from-proportional" else "observed_minus_predicted_bpb"
    gaps = [float(row[value_key]) for row in calibration_rows]
    right_positions = positions if layout == "matched-components" else list(range(4))
    right_colors = colors if layout == "matched-components" else ["#009E73", "#D55E00", "#D55E00", "#555555"]
    right.barh(
        right_positions, gaps, height=0.53 if layout == "matched-components" else 0.44, color=right_colors, zorder=2
    )
    if layout == "matched-components":
        right.set_yticks(positions, [""] * len(positions))
        right.set_ylim(left.get_ylim())
        for axis in (left, right):
            axis.axhline(7.3, color="#DDDDDD", linewidth=0.6, zorder=0)
    else:
        right.set_yticks(right_positions, [row["label"] for row in calibration_rows])
        right.set_ylim(3.7, -0.55)
    if panel_b == "change-from-proportional":
        predictions = [float(row["predicted_delta_bpb"]) for row in calibration_rows]
        right.hlines(right_positions, predictions, gaps, color="#333333", linewidth=0.7, linestyles="--", zorder=3)
        right.plot(
            predictions,
            right_positions,
            "D",
            color="#333333",
            markerfacecolor="white",
            markersize=3.2,
            markeredgewidth=0.6,
            zorder=4,
        )
        right.set_xlim(-0.10 if layout == "matched-components" else -0.055, 0.84)
        right.set_xticks([0, 0.2, 0.4, 0.6])
        right.set_xlabel("Change from proportional (BPB)", labelpad=4)
    else:
        right.set_xlim(-0.055, 0.54)
        right.set_xticks([0, 0.2, 0.4])
        right.set_xlabel("Observed \u2212 predicted (BPB)", labelpad=4)
    annotation_ends = [max(value, 0) for value in gaps]
    if layout == "matched-components":
        annotation_ends = [max(value, prediction, 0) for value, prediction in zip(gaps, predictions, strict=True)]
    for y, value, end in zip(right_positions, gaps, annotation_ends, strict=True):
        right.annotate(
            f"{value:+.3f}".replace("-", "\u2212"),
            (end, y),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=6.5,
        )
    handles = [Line2D([], [], color=color, lw=3, label=group) for group, color in COLORS.items()]
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.025, -0.005),
        ncol=3,
        frameon=False,
        fontsize=6.5,
        handlelength=1.0,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    if panel_b == "change-from-proportional":
        fig.legend(
            handles=[
                Patch(facecolor="#555555", edgecolor="none", label="Observed"),
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="D",
                    color="#333333",
                    markerfacecolor="white",
                    markeredgewidth=0.6,
                    markersize=3.2,
                    label="Predicted",
                ),
            ],
            loc="lower left",
            bbox_to_anchor=(0.602 if layout == "matched-components" else 0.535, -0.005),
            ncol=2,
            frameon=False,
            fontsize=6.5,
            handlelength=1.0,
            columnspacing=1.0,
            handletextpad=0.4,
        )
    else:
        fig.text(0.535, 0.047, "Positive: loss underestimated", fontsize=6.5, color="#444444")
    fig.savefig(output_dir / f"{figure_stem}.pdf")
    fig.savefig(output_dir / f"{figure_stem}.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--paper-dir", type=Path)
    parser.add_argument(
        "--panel-b", choices=("prediction-error", "change-from-proportional"), default="prediction-error"
    )
    parser.add_argument("--layout", choices=("summary", "matched-components"), default="summary")
    parser.add_argument("--show-predictions", choices=("panel-b", "both"), default="panel-b")
    args = parser.parse_args()
    if args.layout == "matched-components" or args.show_predictions == "both":
        args.panel_b = "change-from-proportional"
    figure_stem = FIGURE_STEM + ("_proportional" if args.panel_b == "change-from-proportional" else "")
    if args.layout == "matched-components":
        figure_stem += "_all_components"
    if args.show_predictions == "both":
        figure_stem += "_both_predictions"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    components, calibration = load_values(args.layout)
    write_rows(args.output_dir / "component_changes.csv", components)
    write_rows(args.output_dir / "three_component_calibration.csv", calibration)
    draw(components, calibration, args.output_dir, args.panel_b, figure_stem, args.layout, args.show_predictions)
    provenance = {
        "figure_inches": MATCHED_FIGURE_SIZE if args.layout == "matched-components" else FIGURE_SIZE,
        "layout": args.layout,
        "show_predictions": args.show_predictions,
        "full_optimum_prediction_row": "full_uncheatable_optimum",
        "source_files": {
            key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for key, path in SOURCES.items()
        },
        "comparison": "Panel A compares the full Uncheatable optimum with proportional, not with Olmix.",
        "panel_b": args.panel_b,
        "panel_b_baseline": "Observed and predicted changes subtract the same eleven-run proportional mean.",
        "weights": (
            "Panel A labels show exact byte weights rounded to one decimal percent; bars are unweighted BPB changes."
        ),
        "uncertainty": (
            "Bars are observed means over three trainer seeds relative to the eleven-run proportional mean. "
            "Observed SDs remain in the plotted-data CSVs; no uncertainty marks are drawn."
        ),
        "code_share_of_full_gain": (
            sum(row["weighted_delta_bpb"] for row in components if row["group"] == "Code")
            / sum(row["weighted_delta_bpb"] for row in components if row["group"] != "Aggregate")
        ),
        "nominal_cap_identifiers": (
            "Archived cap06 labels identify policies whose cap was inactive; "
            "the manuscript reports the unconstrained optima."
        ),
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if args.paper_dir:
        for extension in ("pdf", "png"):
            shutil.copy2(args.output_dir / f"{figure_stem}.{extension}", args.paper_dir / "figures")
    print(f"Wrote {args.output_dir / figure_stem}")


if __name__ == "__main__":
    main()
