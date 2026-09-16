# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy>=1.7",
#   "fsspec>=2025.7",
#   "gcsfs>=2025.7",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///
"""Compare model-family raw optima after the expanded Delphi 3e18 fit."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_delphi_phase_policy_sample_efficiency_20260721 as analysis,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    plot_delphi_expanded_fit_raw_optima_20260721 as raw_optima,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/delphi_expanded_fit_raw_optimum_model_comparison_20260721"
REFERENCE_MODELS = {"uncheatable": "separate_heads", "table9": "compact_retained_state"}
MODEL_COLORS = {
    "effective_exposure": "#a50026",
    "separate_heads": "#d73027",
    "compact_retained_state": "#fdae61",
    "bucket_family_grp": "#66bd63",
    "hierarchical_phase_bucket_replay": "#006837",
}
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}, "responsive": True}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--optimizer-starts", type=int, default=16)
    parser.add_argument("--optimizer-seed", type=int, default=20260721)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fit_one(
    output_dir: str,
    target: str,
    model_id: str,
    optimizer_starts: int,
    optimizer_seed: int,
    overwrite: bool,
) -> tuple[str, str, str, str]:
    """Fit and optimize one model-target pair in an isolated artifact directory."""
    candidate_dir = Path(output_dir) / "candidates" / target / model_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    spec = raw_optima.CandidateSpec(
        target=target,
        model_id=model_id,
        label=f"{analysis.TARGET_COLUMNS[target]} / {analysis.MODEL_LABELS[model_id]}",
    )
    _result, provenance = raw_optima.fit_candidate(
        spec,
        candidate_dir,
        optimizer_starts,
        optimizer_seed,
        overwrite,
    )
    provenance_path = candidate_dir / "training_provenance.csv"
    provenance.to_csv(provenance_path, index=False)
    summary_path, weights_path = raw_optima.candidate_paths(candidate_dir, spec)
    return target, model_id, str(summary_path), str(weights_path)


def load_results(paths: list[tuple[str, str, str, str]]) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    summaries: list[dict[str, Any]] = []
    weights: dict[tuple[str, str], pd.DataFrame] = {}
    for target, model_id, summary_path, weights_path in paths:
        summary = json.loads(Path(summary_path).read_text())
        summaries.append(summary)
        weights[(target, model_id)] = pd.read_csv(weights_path)
    frame = pd.DataFrame(summaries).sort_values(["target", "model"]).reset_index(drop=True)
    return frame, weights


def policy_distances(
    summaries: pd.DataFrame,
    weights: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in analysis.TARGETS:
        target_rows = summaries[summaries["target"] == target]
        models = target_rows["model"].tolist()
        for left_index, left_model in enumerate(models):
            left = weights[(target, left_model)]
            for right_model in models[left_index:]:
                right = weights[(target, right_model)]
                phase0_tv = 0.5 * np.abs(left["phase_0_weight"] - right["phase_0_weight"]).sum()
                phase1_tv = 0.5 * np.abs(left["phase_1_weight"] - right["phase_1_weight"]).sum()
                aggregate_tv = 0.5 * np.abs(left["aggregate_weight"] - right["aggregate_weight"]).sum()
                rows.append(
                    {
                        "target": target,
                        "left_model": left_model,
                        "right_model": right_model,
                        "phase_0_tv": float(phase0_tv),
                        "phase_1_tv": float(phase1_tv),
                        "weighted_policy_tv": float(0.8 * phase0_tv + 0.2 * phase1_tv),
                        "aggregate_tv": float(aggregate_tv),
                    }
                )
    return pd.DataFrame(rows)


def reference_distances(summaries: pd.DataFrame, distances: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, reference_model in REFERENCE_MODELS.items():
        for model_id in analysis.MODEL_IDS:
            mask = (distances["target"] == target) & (
                ((distances["left_model"] == reference_model) & (distances["right_model"] == model_id))
                | ((distances["left_model"] == model_id) & (distances["right_model"] == reference_model))
            )
            distance = distances[mask].iloc[0]
            summary = summaries[(summaries["target"] == target) & (summaries["model"] == model_id)].iloc[0]
            rows.append(
                {
                    "target": target,
                    "reference_model": reference_model,
                    "model": model_id,
                    "model_label": analysis.MODEL_LABELS[model_id],
                    "weighted_policy_tv_to_reference": distance["weighted_policy_tv"],
                    "aggregate_tv_to_reference": distance["aggregate_tv"],
                    "predicted_bpb": summary["predicted_bpb"],
                    "max_bucket_weight": summary["max_bucket_weight"],
                    "max_simulated_epochs": summary["max_simulated_epochs"],
                    "phase_total_variation": summary["phase_total_variation"],
                    "nearest_fit_policy_tv": summary["nearest_fit_policy_tv"],
                    "standardized_fit_support_distance": summary["standardized_fit_support_distance"],
                }
            )
    return pd.DataFrame(rows)


def symmetric_matrix(distances: pd.DataFrame, target: str, value: str) -> tuple[list[str], np.ndarray]:
    models = list(analysis.MODEL_IDS)
    matrix = np.zeros((len(models), len(models)), dtype=float)
    index = {model_id: model_index for model_index, model_id in enumerate(models)}
    for row in distances[distances["target"] == target].itertuples(index=False):
        left = index[row.left_model]
        right = index[row.right_model]
        matrix[left, right] = float(getattr(row, value))
        matrix[right, left] = float(getattr(row, value))
    return [analysis.MODEL_LABELS[model_id] for model_id in models], matrix


def plot_pairwise_tv(distances: pd.DataFrame, output_dir: Path) -> Path:
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Uncheatable", "Table-9"), horizontal_spacing=0.12)
    for column, target in enumerate(analysis.TARGETS, start=1):
        labels, matrix = symmetric_matrix(distances, target, "weighted_policy_tv")
        figure.add_trace(
            go.Heatmap(
                z=matrix,
                x=labels,
                y=labels,
                zmin=0,
                zmax=max(0.5, float(distances["weighted_policy_tv"].max())),
                colorscale="RdYlGn_r",
                text=np.vectorize(lambda value: f"{value:.3f}")(matrix),
                texttemplate="%{text}",
                colorbar={"title": "weighted policy TV"} if column == 2 else None,
                showscale=column == 2,
                hovertemplate="%{y}<br>%{x}<br>TV=%{z:.5f}<extra></extra>",
            ),
            row=1,
            col=column,
        )
    figure.update_layout(
        title={
            "text": (
                "Do expanded-fit raw optima converge across model families?"
                "<br><span style='font-size:14px;color:#5f6b76'>"
                "Weighted policy TV = 0.8 TV(phase 0) + 0.2 TV(phase 1); zero means identical policies.</span>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_white",
        width=1700,
        height=850,
        margin={"l": 190, "r": 140, "t": 130, "b": 190},
        paper_bgcolor="#fbfaf6",
    )
    path = output_dir / "pairwise_raw_optimum_policy_tv.html"
    path.write_text(pio.to_html(figure, include_plotlyjs=True, full_html=True, config=PLOT_CONFIG))
    return path


def plot_health(reference: pd.DataFrame, output_dir: Path) -> Path:
    metrics = (
        ("predicted_bpb", "Predicted BPB"),
        ("max_simulated_epochs", "Max simulated epochs"),
        ("phase_total_variation", "Phase TV"),
        ("standardized_fit_support_distance", "Support distance"),
    )
    figure = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=[label for _target in analysis.TARGETS for _field, label in metrics],
        vertical_spacing=0.17,
        horizontal_spacing=0.08,
        row_titles=("Uncheatable", "Table-9"),
    )
    for row_index, target in enumerate(analysis.TARGETS, start=1):
        target_frame = reference[reference["target"] == target]
        colors = [MODEL_COLORS[model_id] for model_id in target_frame["model"]]
        for column_index, (field, _label) in enumerate(metrics, start=1):
            figure.add_trace(
                go.Bar(
                    x=target_frame["model_label"],
                    y=target_frame[field],
                    marker_color=colors,
                    customdata=np.column_stack(
                        [
                            target_frame["weighted_policy_tv_to_reference"],
                            target_frame["aggregate_tv_to_reference"],
                        ]
                    ),
                    hovertemplate=(
                        "%{x}<br>%{y:.6f}<br>policy TV to plotted reference=%{customdata[0]:.5f}"
                        "<br>aggregate TV to reference=%{customdata[1]:.5f}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row_index,
                col=column_index,
            )
            figure.update_xaxes(tickangle=-25, row=row_index, col=column_index)
    figure.update_layout(
        title={
            "text": "Expanded Delphi 3e18 raw-optimum health by model family",
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_white",
        width=2200,
        height=1200,
        margin={"l": 120, "r": 120, "t": 120, "b": 180},
        paper_bgcolor="#fbfaf6",
    )
    path = output_dir / "raw_optimum_model_health.html"
    path.write_text(pio.to_html(figure, include_plotlyjs=True, full_html=True, config=PLOT_CONFIG))
    return path


def write_report(
    summaries: pd.DataFrame,
    distances: pd.DataFrame,
    reference: pd.DataFrame,
    output_dir: Path,
) -> Path:
    sections = [
        "# Expanded Delphi 3e18 raw-optimum model comparison",
        "",
        (
            "All five model families were refit on the same 998 unique Delphi 3e18 policies "
            "(280 original two-phase, 238 independently trained tied, and 480 phase-varying extension policies). "
            "The optimizer used the same multistart protocol for every target-model pair."
        ),
        "",
    ]
    for target in analysis.TARGETS:
        target_summary = summaries[summaries["target"] == target][
            [
                "model_label",
                "predicted_bpb",
                "max_bucket_weight",
                "max_simulated_epochs",
                "phase_total_variation",
                "aggregate_tv_to_proportional",
                "nearest_fit_policy_tv",
                "standardized_fit_support_distance",
            ]
        ].copy()
        target_reference = reference[reference["target"] == target][
            ["model_label", "weighted_policy_tv_to_reference", "aggregate_tv_to_reference"]
        ]
        table = target_summary.merge(target_reference, on="model_label")
        pairwise = distances[(distances["target"] == target) & (distances["left_model"] != distances["right_model"])]
        sections.extend(
            [
                f"## {target}",
                "",
                table.to_markdown(index=False, floatfmt=".6f"),
                "",
                (
                    f"Pairwise weighted-policy TV: median {pairwise['weighted_policy_tv'].median():.4f}, "
                    f"minimum {pairwise['weighted_policy_tv'].min():.4f}, "
                    f"maximum {pairwise['weighted_policy_tv'].max():.4f}."
                ),
                "",
            ]
        )
    path = output_dir / "report.md"
    path.write_text("\n".join(sections) + "\n")
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(target, model_id) for target in analysis.TARGETS for model_id in analysis.MODEL_IDS]
    completed: list[tuple[str, str, str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                fit_one,
                str(args.output_dir),
                target,
                model_id,
                args.optimizer_starts,
                args.optimizer_seed,
                args.overwrite,
            ): (target, model_id)
            for target, model_id in jobs
        }
        for future in as_completed(futures):
            target, model_id = futures[future]
            result = future.result()
            completed.append(result)
            print(f"completed {target}/{model_id}", flush=True)

    summaries, weights = load_results(completed)
    distances = policy_distances(summaries, weights)
    reference = reference_distances(summaries, distances)
    summaries.to_csv(args.output_dir / "raw_optimum_model_comparison.csv", index=False)
    distances.to_csv(args.output_dir / "pairwise_raw_optimum_policy_tv.csv", index=False)
    reference.to_csv(args.output_dir / "raw_optimum_distance_to_plotted_reference.csv", index=False)
    plot_pairwise_tv(distances, args.output_dir)
    plot_health(reference, args.output_dir)
    report = write_report(summaries, distances, reference, args.output_dir)
    print(reference.to_string(index=False), flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    main()
