# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "kaleido==0.2.1",
#   "numpy",
#   "pandas",
#   "plotly",
#   "tabulate",
# ]
# ///

"""Compare raw and unweighted quartic-ridge gain in the dense WSD80 replay panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    plot_starcoder_wsd80_dense_horizon_replay_gain_20260811 as raw_plot,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_dense_horizon_replay_surface_sensitivity_20260811"

SURFACE_DEGREE = 4
SURFACE_FOLDS = 5
SURFACE_RIDGE_GRID = np.logspace(-6, 2, 17)
SURFACE_GRID_SIZE = 197
MIN_UNTIED_CONTRAST = 0.04
PHASE_0_FRACTION = 0.8
EXPECTED_COORDINATES_PER_BLOCK = 125


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-policies", type=Path, default=raw_plot.DEFAULT_SELECTED_POLICIES)
    parser.add_argument("--coverage-observations", type=Path, default=raw_plot.DEFAULT_COVERAGE_OBSERVATIONS)
    parser.add_argument("--design", type=Path, default=raw_plot.DEFAULT_DESIGN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _surface_features(aggregate: np.ndarray, contrast: np.ndarray) -> np.ndarray:
    """Return the preregistered quartic basis in aggregate and raw contrast."""
    normalized_aggregate = (aggregate - 0.5) / 0.5
    terms = []
    for degree in range(SURFACE_DEGREE + 1):
        for aggregate_degree in range(degree + 1):
            contrast_degree = degree - aggregate_degree
            terms.append(normalized_aggregate**aggregate_degree * contrast**contrast_degree)
    return np.column_stack(terms)


def _coordinates_to_features(coordinates: np.ndarray) -> np.ndarray:
    p0 = coordinates[:, 0]
    p1 = coordinates[:, 1]
    aggregate = PHASE_0_FRACTION * p0 + (1.0 - PHASE_0_FRACTION) * p1
    contrast = p1 - p0
    return _surface_features(aggregate, contrast)


def _ridge_operator(features: np.ndarray, ridge: float) -> np.ndarray:
    penalty = np.eye(features.shape[1], dtype=float) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(features.T @ features + penalty, features.T)


def _spatial_folds(coordinates: np.ndarray) -> np.ndarray:
    p0_bin = np.floor(5 * coordinates[:, 0]).astype(int)
    p1_bin = np.floor(5 * coordinates[:, 1]).astype(int)
    return (p0_bin + 2 * p1_bin) % SURFACE_FOLDS


def _select_ridge(coordinates: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    folds = _spatial_folds(coordinates)
    if set(folds) != set(range(SURFACE_FOLDS)):
        raise ValueError("The preregistered spatial-CV rule produced an empty fold")
    features = _coordinates_to_features(coordinates)
    scores = []
    for ridge in SURFACE_RIDGE_GRID:
        squared_errors = []
        for fold in range(SURFACE_FOLDS):
            train = folds != fold
            test = ~train
            coefficients = _ridge_operator(features[train], float(ridge)) @ target[train]
            squared_errors.extend(np.square(features[test] @ coefficients - target[test]))
        scores.append(float(np.sqrt(np.mean(squared_errors))))
    selected = int(np.argmin(scores))
    return float(SURFACE_RIDGE_GRID[selected]), scores[selected]


def _convex_hull(coordinates: np.ndarray) -> np.ndarray:
    points = sorted({(float(row[0]), float(row[1])) for row in coordinates})
    if len(points) < 3:
        raise ValueError("At least three unique coordinates are required for a surface hull")

    def cross(origin: tuple[float, float], left: tuple[float, float], right: tuple[float, float]) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (right[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _inside_convex_hull(points: np.ndarray, hull: np.ndarray) -> np.ndarray:
    inside = np.ones(len(points), dtype=bool)
    for start, end in zip(hull, np.roll(hull, -1, axis=0), strict=True):
        edge = end - start
        cross = edge[0] * (points[:, 1] - start[1]) - edge[1] * (points[:, 0] - start[0])
        inside &= cross >= -1e-12
    return inside


def _optimization_grids(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, SURFACE_GRID_SIZE)
    p0, p1 = np.meshgrid(axis, axis, indexing="ij")
    untied = np.column_stack((p0.ravel(), p1.ravel()))
    untied = untied[np.abs(untied[:, 1] - untied[:, 0]) >= MIN_UNTIED_CONTRAST]
    hull = _convex_hull(coordinates)
    untied = untied[_inside_convex_hull(untied, hull)]
    tied = np.column_stack((axis, axis))
    tied = tied[_inside_convex_hull(tied, hull)]
    if untied.size == 0 or tied.size == 0:
        raise ValueError("The empirical hull contains no optimization candidates")
    return untied, tied


def _nearest_distance(point: np.ndarray, coordinates: np.ndarray) -> float:
    return float(np.linalg.norm(coordinates - point, axis=1).min())


def fit_unweighted_surfaces(coverage_path: Path) -> pd.DataFrame:
    """Fit the preregistered basis without unavailable calibration-derived weights."""
    coverage = pd.read_csv(coverage_path)
    required = {
        "cell_id",
        "support_id",
        "coordinate_id",
        "phase_0_starcoder",
        "phase_1_starcoder",
        "bpb",
    }
    missing = required - set(coverage.columns)
    if missing:
        raise ValueError(f"Coverage table is missing fields: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for (cell_id, support_id), group in coverage.groupby(["cell_id", "support_id"], sort=True):
        if len(group) != EXPECTED_COORDINATES_PER_BLOCK or group["coordinate_id"].duplicated().any():
            raise ValueError(f"{cell_id}/{support_id}: expected 125 unique coordinates")
        coordinates = group[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
        target = group["bpb"].to_numpy(dtype=float)
        if not np.isfinite(target).all():
            raise ValueError(f"{cell_id}/{support_id}: non-finite BPB")

        ridge, spatial_cv_rmse = _select_ridge(coordinates, target)
        features = _coordinates_to_features(coordinates)
        coefficients = _ridge_operator(features, ridge) @ target
        untied_grid, tied_grid = _optimization_grids(coordinates)
        untied_prediction = _coordinates_to_features(untied_grid) @ coefficients
        tied_prediction = _coordinates_to_features(tied_grid) @ coefficients
        untied_index = int(np.argmin(untied_prediction))
        tied_index = int(np.argmin(tied_prediction))
        untied = untied_grid[untied_index]
        tied = tied_grid[tied_index]
        rows.append(
            {
                "cell_id": cell_id,
                "support_id": support_id,
                "selected_ridge": ridge,
                "unweighted_spatial_cv_rmse": spatial_cv_rmse,
                "surface_tied_p": float(tied[0]),
                "surface_tied_bpb": float(tied_prediction[tied_index]),
                "surface_untied_p0": float(untied[0]),
                "surface_untied_p1": float(untied[1]),
                "surface_untied_bpb": float(untied_prediction[untied_index]),
                "surface_global_two_phase_gain_bpb": float(
                    tied_prediction[tied_index] - untied_prediction[untied_index]
                ),
                "surface_tied_nearest_design_l2": _nearest_distance(tied, coordinates),
                "surface_untied_nearest_design_l2": _nearest_distance(untied, coordinates),
            }
        )
    fitted = pd.DataFrame(rows)
    if len(fitted) != raw_plot.EXPECTED_BLOCKS:
        raise ValueError(f"Expected {raw_plot.EXPECTED_BLOCKS} fitted surfaces, got {len(fitted)}")
    return fitted


def load_comparison(
    selected_path: Path,
    coverage_path: Path,
    design_path: Path,
) -> pd.DataFrame:
    """Join raw-grid and unweighted continuous-surface estimates."""
    design = json.loads(design_path.read_text(encoding="utf-8"))
    estimator = design["analysis_contract"]["surface_estimator"]
    if estimator["basis"] != "all aggregate_and_raw_contrast_monomials_through_total_degree_four":
        raise ValueError("Unexpected preregistered surface basis")
    if design["analysis_contract"]["optimization"]["axis_grid_size"] != SURFACE_GRID_SIZE:
        raise ValueError("Unexpected preregistered optimization grid")

    raw = raw_plot.load_summary(selected_path, coverage_path, design_path)
    fitted = fit_unweighted_surfaces(coverage_path)
    comparison = raw.merge(fitted, on=["cell_id", "support_id"], validate="one_to_one")
    comparison["surface_minus_raw_gain_bpb"] = (
        comparison["surface_global_two_phase_gain_bpb"] - comparison["raw_two_phase_gain_bpb"]
    )
    return comparison.sort_values(["support_order", "rung"]).reset_index(drop=True)


def _custom_data(group: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            group["cell_id"],
            group["support_id"].map(raw_plot.SUPPORT_LABELS),
            group["total_parameter_tpp"],
            group["raw_two_phase_gain_bpb"].map(lambda value: f"{float(value):+.6f}"),
            group["surface_global_two_phase_gain_bpb"].map(lambda value: f"{float(value):+.6f}"),
            group["surface_minus_raw_gain_bpb"].map(lambda value: f"{float(value):+.6f}"),
            group["surface_tied_p"],
            group["surface_tied_bpb"],
            group["surface_untied_p0"],
            group["surface_untied_p1"],
            group["surface_untied_bpb"],
            group["selected_ridge"],
            group["unweighted_spatial_cv_rmse"],
            group["surface_tied_nearest_design_l2"],
            group["surface_untied_nearest_design_l2"],
        ]
    )


def _add_panel_trace(
    figure: go.Figure,
    group: pd.DataFrame,
    support_id: str,
    column: int,
    y_column: str,
    show_legend: bool,
) -> None:
    figure.add_trace(
        go.Scatter(
            x=group["materialized_tokens_b"],
            y=group[y_column],
            mode="lines+markers+text",
            name=raw_plot.SUPPORT_LABELS[support_id],
            legendgroup=support_id,
            showlegend=show_legend,
            line={
                "color": raw_plot.SUPPORT_COLORS[support_id],
                "width": 3.2 if support_id in {"full", "m100", "m400"} else 2.0,
            },
            marker={
                "color": raw_plot.PLOT_BACKGROUND,
                "size": 31,
                "symbol": "circle",
                "line": {"color": raw_plot.SUPPORT_COLORS[support_id], "width": 4.0},
            },
            text=[raw_plot.SUPPORT_MARKER_LABELS[support_id]] * len(group),
            textposition="middle center",
            textfont={
                "color": raw_plot.SUPPORT_COLORS[support_id],
                "family": "Avenir Next Condensed, Arial Narrow, sans-serif",
                "size": 9,
            },
            customdata=_custom_data(group),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "Materialized tokens: %{x:.3f}B<br>"
                "Total-parameter TPP: %{customdata[2]:.2f}<br><br>"
                "Raw grid global two-phase gain: %{customdata[3]} BPB<br>"
                "Unweighted surface global two-phase gain: %{customdata[4]} BPB<br>"
                "Surface - raw: %{customdata[5]} BPB<br><br>"
                "<b>Unweighted quartic-ridge optimum</b><br>"
                "Tied: p=%{customdata[6]:.4f}, %{customdata[7]:.6f} BPB<br>"
                "Untied: p0=%{customdata[8]:.4f}, p1=%{customdata[9]:.4f}, %{customdata[10]:.6f} BPB<br>"
                "Ridge: %{customdata[11]:.6g}; spatial-CV RMSE: %{customdata[12]:.6f}<br>"
                "Nearest sampled L2: tied %{customdata[13]:.4f}, untied %{customdata[14]:.4f}<br>"
                "Sensitivity only: calibration-derived variance weights are not available"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=column,
    )


def build_figure(comparison: pd.DataFrame) -> go.Figure:
    """Build a side-by-side raw versus continuous-surface sensitivity plot."""
    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        subplot_titles=(
            "<b>Raw common-grid minima</b><br><sup>one discovery seed · descriptive</sup>",
            "<b>Unweighted quartic-ridge sensitivity</b><br><sup>fails sanity check · not an estimate</sup>",
        ),
    )
    for support_id in raw_plot.SUPPORT_ORDER:
        group = comparison.loc[comparison["support_id"].eq(support_id)].sort_values("rung")
        if len(group) != raw_plot.EXPECTED_CELLS:
            raise ValueError(f"{support_id}: expected {raw_plot.EXPECTED_CELLS} horizon rows")
        _add_panel_trace(figure, group, support_id, 1, "raw_two_phase_gain_bpb", False)
        _add_panel_trace(figure, group, support_id, 2, "surface_global_two_phase_gain_bpb", True)

    values = comparison[["raw_two_phase_gain_bpb", "surface_global_two_phase_gain_bpb"]].to_numpy(dtype=float)
    y_min = float(values.min())
    y_max = float(values.max())
    y_padding = 0.16 * max(y_max - y_min, 1e-4)
    for column in (1, 2):
        figure.add_hline(y=0.0, line={"color": raw_plot.ZERO_COLOR, "width": 2.0}, row=1, col=column)

    figure.update_layout(
        title={
            "text": (
                "<b>StarCoder WSD80 global two-phase gain: observed grid versus failed smooth sensitivity</b><br>"
                "<sup>Fixed N = 210M · four token horizons · seven StarCoder repetition regimes · "
                "same 125 coordinates per block</sup>"
            ),
            "x": 0.04,
            "xanchor": "left",
            "font": {
                "size": 28,
                "color": raw_plot.PAPER_TEXT,
                "family": "Georgia, Times New Roman, serif",
            },
        },
        width=1940,
        height=1180,
        paper_bgcolor=raw_plot.PAPER_BACKGROUND,
        plot_bgcolor=raw_plot.PLOT_BACKGROUND,
        font={"family": "Avenir Next, Source Sans Pro, sans-serif", "size": 14, "color": raw_plot.PAPER_TEXT},
        margin={"l": 120, "r": 400, "t": 150, "b": 190},
        hoverlabel={"bgcolor": raw_plot.PLOT_BACKGROUND, "font": {"size": 13, "color": raw_plot.PAPER_TEXT}},
        legend={
            "title": {"text": "<b>StarCoder simulated-epoching<br>repetition multiplier</b>"},
            "x": 1.015,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "bgcolor": "rgba(255,253,248,0.96)",
            "bordercolor": raw_plot.GRID_COLOR,
            "borderwidth": 1.5,
            "font": {"size": 13},
            "itemsizing": "constant",
        },
        annotations=[
            *figure.layout.annotations,
            {
                "text": (
                    "Left: raw selected-grid gain, retained as the primary observed result. "
                    "Right: unweighted sensitivity using the preregistered quartic basis, spatial-CV ridge, "
                    "convex-hull domain, tied restriction, and |contrast| >= 0.04.<br>"
                    "The frozen primary surface additionally requires calibration-derived heteroskedastic weights "
                    "and wild-bootstrap uncertainty; those repeats are not yet materialized. Do not use the right "
                    "panel for scaling claims."
                ),
                "x": 0.5,
                "xref": "paper",
                "y": -0.13,
                "yref": "paper",
                "showarrow": False,
                "xanchor": "center",
                "align": "center",
                "font": {"size": 13, "color": raw_plot.PAPER_TEXT},
            },
        ],
    )
    tick_rows = comparison.drop_duplicates("rung").sort_values("rung")
    for column in (1, 2):
        figure.update_xaxes(
            type="log",
            title_text="Materialized training tokens D",
            tickmode="array",
            tickvals=tick_rows["materialized_tokens_b"],
            ticktext=[f"{value:.2f}B" for value in tick_rows["materialized_tokens_b"]],
            gridcolor=raw_plot.GRID_COLOR,
            zeroline=False,
            showline=True,
            linecolor=raw_plot.PAPER_TEXT,
            linewidth=1.2,
            ticks="outside",
            row=1,
            col=column,
        )
    figure.update_yaxes(
        title_text="Global two-phase gain (BPB)<br><sup>tied optimum - untied optimum; higher is better</sup>",
        range=[y_min - y_padding, y_max + y_padding],
        tickformat="+.3f",
        gridcolor=raw_plot.GRID_COLOR,
        zeroline=False,
        showline=True,
        linecolor=raw_plot.PAPER_TEXT,
        linewidth=1.2,
        ticks="outside",
        row=1,
        col=1,
    )
    return figure


def write_report(output_dir: Path, comparison: pd.DataFrame) -> None:
    table = comparison[
        [
            "cell_id",
            "support_id",
            "materialized_tokens_b",
            "raw_two_phase_gain_bpb",
            "surface_global_two_phase_gain_bpb",
            "surface_minus_raw_gain_bpb",
            "selected_ridge",
            "unweighted_spatial_cv_rmse",
            "surface_tied_p",
            "surface_untied_p0",
            "surface_untied_p1",
            "surface_untied_nearest_design_l2",
        ]
    ]
    sign_disagreements = int(
        (comparison["raw_two_phase_gain_bpb"].gt(0.0) != comparison["surface_global_two_phase_gain_bpb"].gt(0.0)).sum()
    )
    lines = [
        "# StarCoder WSD80 dense horizon-by-repetition surface sensitivity",
        "",
        "- The existing raw-grid plot is unchanged and remains the primary observed result.",
        (
            "- This artifact compares it with an unweighted quartic-ridge surface using the preregistered basis, "
            "spatial folds, ridge grid, convex-hull optimization domain, tied restriction, and minimum untied "
            "absolute contrast."
        ),
        (
            "- It is not the frozen primary surface estimator: the 564 calibration repeats needed to estimate "
            "heteroskedastic inverse-variance weights and wild-bootstrap uncertainty are not materialized."
        ),
        "- Raw minima are selection-biased; smoothed minima trade lower point noise for functional-form bias.",
        "",
        "## Verdict",
        "",
        "Do not use this unweighted smooth surface for the scaling claim.",
        "",
        (
            f"- Mean raw gain: {comparison['raw_two_phase_gain_bpb'].mean():+.7f} BPB; mean smooth gain: "
            f"{comparison['surface_global_two_phase_gain_bpb'].mean():+.7f} BPB."
        ),
        f"- Raw and smooth gain signs disagree in {sign_disagreements}/{len(comparison)} blocks.",
        (
            f"- Unweighted spatial-CV RMSE ranges from "
            f"{comparison['unweighted_spatial_cv_rmse'].min():.7f} to "
            f"{comparison['unweighted_spatial_cv_rmse'].max():.7f} BPB, much larger than the measured gains."
        ),
        (
            "- The smooth optimization therefore amplifies boundary and functional-form error into spurious "
            "continuous minima. Keep the raw common-grid result primary until the frozen weighted estimator and "
            "fresh-seed confirmation are available."
        ),
        "",
        "## Comparison",
        "",
        table.to_markdown(index=False, floatfmt=".7f"),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    comparison = load_comparison(args.selected_policies, args.coverage_observations, args.design)
    figure = build_figure(comparison)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "raw_and_unweighted_surface_gain.csv", index=False)
    figure.write_html(
        args.output_dir / "starcoder_wsd80_dense_horizon_replay_surface_sensitivity.html",
        include_plotlyjs=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "starcoder_wsd80_dense_horizon_replay_surface_sensitivity",
                "height": 2360,
                "width": 3880,
                "scale": 4,
            },
        },
    )
    figure.write_image(
        args.output_dir / "starcoder_wsd80_dense_horizon_replay_surface_sensitivity.png",
        width=1940,
        height=1180,
        scale=2,
    )
    write_report(args.output_dir, comparison)


if __name__ == "__main__":
    main()
