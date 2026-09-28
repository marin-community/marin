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

"""Plot fitted one-phase/two-phase optimum scaling in the matched N,D grid."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from starcoder_wsd80_epoch_accounting import (
    simulated_materialized_epochs,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
PANEL_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_matched_nd_stage1_20260731"
STAGE3_RESULTS = PANEL_DIR / "stage3_dense_surface_results_20260802"
DEFAULT_DISCOVERY_SUMMARY = STAGE3_RESULTS / "cell_discovery_summary.csv"
DEFAULT_FITTED_CANDIDATES = STAGE3_RESULTS / "fitted_surface_candidates.csv"
DEFAULT_SOURCE_DESIGN = PANEL_DIR / "stage2_results_20260801" / "source_design.json"
DEFAULT_STAGE1_SUMMARY = PANEL_DIR / "results_20260801" / "cell_summary.csv"
DEFAULT_CONFIRMATION_SUMMARY = PANEL_DIR / "confirmation_results_20260801" / "cell_confirmation_summary.csv"
DEFAULT_CONFIRMATION_DESIGN = PANEL_DIR / "confirmation_design_20260801" / "run_manifest.csv"
DEFAULT_OUTPUT_DIR = PANEL_DIR / "optimum_scaling_20260802"

PHASE_0_FRACTION = 0.8
TRACK_ORDER = ("increase_d", "increase_n", "increase_nd")
TRACK_LABELS = {
    "increase_d": "Fixed N · increase D",
    "increase_n": "Fixed D · increase N",
    "increase_nd": "Increase N and D",
}
AXES = {
    "parameters_m": ("Model parameters N (millions)", "total_parameters", 1e6),
    "tokens_b": ("Materialized tokens D (billions)", "materialized_tokens", 1e9),
    "compute_e18": ("Training compute (10<sup>18</sup> FLOPs)", "compute_flops", 1e18),
    "total_parameter_tpp": ("Total-parameter tokens per parameter D/N", "total_parameter_tpp", 1.0),
}
VARYING_TRACKS = {
    "parameters_m": {"increase_n", "increase_nd"},
    "tokens_b": {"increase_d", "increase_nd"},
    "compute_e18": set(TRACK_ORDER),
    "total_parameter_tpp": set(TRACK_ORDER),
}
METRICS = {
    "optimum_l2_distance": "L2 distance between discovered optima",
    "policy_class_gap_bpb": "Two Phase Gain (BPB; larger is better)",
}

PAPER_BACKGROUND = "#F7F3E8"
PLOT_BACKGROUND = "#FFFDF8"
PAPER_TEXT = "#17324D"
GRID_COLOR = "#D8D1C2"
CONFIRMATION_COLOR = "#111827"
METRIC_COLORS = dict(zip(METRICS, sample_colorscale("RdYlGn_r", (0.08, 0.90)), strict=True))
METRIC_SYMBOLS = {
    "optimum_l2_distance": "circle",
    "policy_class_gap_bpb": "diamond",
}
EXPORT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "starcoder_wsd80_matched_nd_optimum_scaling",
        "height": 1800,
        "width": 2400,
        "scale": 4,
    },
}


@dataclass(frozen=True)
class CurveFit:
    """One descriptive nonnegative log-linear scaling fit."""

    axis: str
    metric: str
    track: str
    intercept: float
    slope_per_decade: float
    r_squared: float
    point_count: int
    x_min: float
    x_max: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-summary", type=Path, default=DEFAULT_DISCOVERY_SUMMARY)
    parser.add_argument("--fitted-candidates", type=Path, default=DEFAULT_FITTED_CANDIDATES)
    parser.add_argument("--source-design", type=Path, default=DEFAULT_SOURCE_DESIGN)
    parser.add_argument("--stage1-summary", type=Path, default=DEFAULT_STAGE1_SUMMARY)
    parser.add_argument("--confirmation-summary", type=Path, default=DEFAULT_CONFIRMATION_SUMMARY)
    parser.add_argument("--confirmation-design", type=Path, default=DEFAULT_CONFIRMATION_DESIGN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _add_epoch_columns(frame: pd.DataFrame, *, prefix: str, phase_0: str, phase_1: str) -> None:
    coordinates = zip(frame[phase_0], frame[phase_1], strict=True)
    epoch_rows = [simulated_materialized_epochs(p0, p1) for p0, p1 in coordinates]
    frame[f"{prefix}_starcoder_phase_0_simulated_epochs"] = [row.starcoder.phase_0 for row in epoch_rows]
    frame[f"{prefix}_starcoder_phase_1_simulated_epochs"] = [row.starcoder.phase_1 for row in epoch_rows]
    frame[f"{prefix}_starcoder_total_simulated_epochs"] = [row.starcoder.total for row in epoch_rows]
    frame[f"{prefix}_nemotron_phase_0_simulated_epochs"] = [row.nemotron.phase_0 for row in epoch_rows]
    frame[f"{prefix}_nemotron_phase_1_simulated_epochs"] = [row.nemotron.phase_1 for row in epoch_rows]
    frame[f"{prefix}_nemotron_total_simulated_epochs"] = [row.nemotron.total for row in epoch_rows]


def _cell_table(source_design_path: Path) -> pd.DataFrame:
    design = json.loads(source_design_path.read_text(encoding="utf-8"))
    cells = pd.DataFrame(design["source_cells"] if "source_cells" in design else design["cells"])
    required = {
        "cell_id",
        "compute_flops",
        "materialized_tokens",
        "non_embedding_parameters",
        "rung",
        "total_parameters",
        "track_memberships",
    }
    missing = required - set(cells.columns)
    if missing:
        raise ValueError(f"Source design is missing cell fields: {sorted(missing)}")
    if len(cells) != 10 or cells["cell_id"].nunique() != 10:
        raise ValueError("Matched-N,D source design must contain ten unique cells")
    return cells


def _tracks_for_cell(cell: pd.Series) -> tuple[str, ...]:
    memberships = cell["track_memberships"]
    if isinstance(memberships, str):
        memberships = json.loads(memberships.replace("'", '"'))
    tracks = tuple(str(value) for value in memberships)
    unknown = set(tracks) - set(TRACK_ORDER)
    if unknown:
        raise ValueError(f"{cell['cell_id']}: unknown track memberships {sorted(unknown)}")
    return tracks


def discovered_optima(
    discovery_path: Path,
    fitted_candidates_path: Path,
    source_design_path: Path,
    stage1_path: Path,
) -> pd.DataFrame:
    """Return one frozen smooth-surface optimum comparison per N,D cell."""
    discovery = pd.read_csv(discovery_path)
    candidates = pd.read_csv(fitted_candidates_path)
    cells = _cell_table(source_design_path)
    required = {
        "cell_id",
        "best_tied_bpb",
        "best_tied_weight",
        "best_untied_bpb",
        "best_untied_p0",
        "best_untied_p1",
    }
    missing = required - set(discovery.columns)
    if missing:
        raise ValueError(f"Discovery summary is missing fields: {sorted(missing)}")
    candidate_fields = {
        "cell_id",
        "fitted_untied_p0",
        "fitted_untied_p1",
        "fitted_untied_bpb",
        "fitted_tied_weight",
        "fitted_tied_bpb",
        "fitted_gain_tied_minus_untied_bpb",
        "bootstrap_gain_p05",
        "bootstrap_gain_p50",
        "bootstrap_gain_p95",
        "bootstrap_positive_gain_probability",
        "bootstrap_candidate_l2_p90",
        "confirmation_eligible",
    }
    missing = candidate_fields - set(candidates.columns)
    if missing:
        raise ValueError(f"Fitted-candidate table is missing fields: {sorted(missing)}")
    if len(candidates) != 10 or candidates["cell_id"].nunique() != 10:
        raise ValueError("Expected one frozen fitted candidate per N,D cell")
    frame = discovery.merge(
        cells[
            [
                "cell_id",
                "compute_flops",
                "materialized_tokens",
                "non_embedding_parameters",
                "rung",
                "total_parameters",
                "track_memberships",
            ]
        ],
        on="cell_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_design"),
    )
    for column in ("materialized_tokens", "rung", "total_parameters"):
        design_column = f"{column}_design"
        if design_column in frame.columns:
            if not np.allclose(frame[column], frame[design_column], rtol=0.0, atol=0.0):
                raise ValueError(f"Discovery and source design disagree on {column}")
            frame = frame.drop(columns=design_column)

    frame = frame.merge(candidates, on="cell_id", validate="one_to_one")
    frame = frame.rename(
        columns={
            "best_tied_weight": "raw_best_tied_weight",
            "best_tied_bpb": "raw_best_tied_bpb",
            "best_untied_p0": "raw_best_untied_p0",
            "best_untied_p1": "raw_best_untied_p1",
            "best_untied_bpb": "raw_best_untied_bpb",
        }
    )
    frame["raw_policy_class_gap_bpb"] = np.maximum(0.0, frame["raw_best_tied_bpb"] - frame["raw_best_untied_bpb"])
    frame["best_tied_p0"] = frame["fitted_tied_weight"]
    frame["best_tied_p1"] = frame["fitted_tied_weight"]
    frame["best_tied_bpb"] = frame["fitted_tied_bpb"]
    frame["two_phase_p0"] = frame["fitted_untied_p0"]
    frame["two_phase_p1"] = frame["fitted_untied_p1"]
    frame["two_phase_bpb"] = frame["fitted_untied_bpb"]
    frame["policy_class_gap_bpb"] = frame["fitted_gain_tied_minus_untied_bpb"]
    frame["optimum_l2_distance"] = np.hypot(
        frame["two_phase_p0"] - frame["best_tied_p0"],
        frame["two_phase_p1"] - frame["best_tied_p1"],
    )
    frame["aggregate_two_phase_starcoder"] = (
        PHASE_0_FRACTION * frame["two_phase_p0"] + (1.0 - PHASE_0_FRACTION) * frame["two_phase_p1"]
    )
    _add_epoch_columns(frame, prefix="tied", phase_0="best_tied_p0", phase_1="best_tied_p1")
    _add_epoch_columns(frame, prefix="two_phase", phase_0="two_phase_p0", phase_1="two_phase_p1")
    _add_epoch_columns(frame, prefix="raw_tied", phase_0="raw_best_tied_weight", phase_1="raw_best_tied_weight")
    _add_epoch_columns(frame, prefix="raw_two_phase", phase_0="raw_best_untied_p0", phase_1="raw_best_untied_p1")
    frame["total_parameter_tpp"] = frame["materialized_tokens"] / frame["total_parameters"]
    frame["non_embedding_tpp"] = frame["materialized_tokens"] / frame["non_embedding_parameters"]
    stage1 = pd.read_csv(stage1_path)
    if {"cell_id", "observed_policy_class_gap_bpb"} - set(stage1.columns):
        raise ValueError("Stage-1 summary is missing its cell ID or observed policy-class gap")
    stage1 = stage1[["cell_id", "observed_policy_class_gap_bpb"]].copy()
    stage1["stage1_policy_class_gain_bpb"] = np.maximum(0.0, -stage1["observed_policy_class_gap_bpb"])
    frame = frame.merge(
        stage1[["cell_id", "stage1_policy_class_gain_bpb"]],
        on="cell_id",
        how="left",
        validate="one_to_one",
    )
    if frame["stage1_policy_class_gain_bpb"].isna().any():
        raise ValueError("Stage-1 summary does not cover every matched-N,D cell")
    if (frame[["policy_class_gap_bpb", "optimum_l2_distance", "raw_policy_class_gap_bpb"]] < -1e-12).any().any():
        raise ValueError("Nested policy-class metrics must be nonnegative")
    return frame.sort_values(["rung", "cell_id"]).reset_index(drop=True)


def expanded_tracks(frame: pd.DataFrame) -> pd.DataFrame:
    """Duplicate the shared rung into each of its three scaling tracks."""
    rows = []
    for _, row in frame.iterrows():
        for track in _tracks_for_cell(row):
            record = row.to_dict()
            record["track"] = track
            rows.append(record)
    expanded = pd.DataFrame(rows)
    counts = expanded.groupby("track").size().to_dict()
    if counts != {track: 4 for track in TRACK_ORDER}:
        raise ValueError(f"Every scaling track must contain four cells, got {counts}")
    return expanded.sort_values(["track", "rung"]).reset_index(drop=True)


def _fit_curve(x: np.ndarray, y: np.ndarray, *, axis: str, metric: str, track: str) -> CurveFit:
    if len(x) != 4 or len(np.unique(x)) != 4:
        raise ValueError(f"{axis}/{track}: expected four unique scaling points")
    log_x = np.log10(x)
    slope, intercept = np.polyfit(log_x, y, deg=1)
    fitted = np.maximum(0.0, intercept + slope * log_x)
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 if total <= 1e-18 and residual <= 1e-18 else 1.0 - residual / total if total > 0 else np.nan
    return CurveFit(
        axis=axis,
        metric=metric,
        track=track,
        intercept=float(intercept),
        slope_per_decade=float(slope),
        r_squared=float(r_squared),
        point_count=len(x),
        x_min=float(x.min()),
        x_max=float(x.max()),
    )


def scaling_fits(expanded: pd.DataFrame) -> pd.DataFrame:
    fits = []
    for axis, (_, column, divisor) in AXES.items():
        for metric in METRICS:
            for track in TRACK_ORDER:
                if track not in VARYING_TRACKS[axis]:
                    continue
                group = expanded.loc[expanded["track"].eq(track)].sort_values("rung")
                x = group[column].to_numpy(dtype=float) / divisor
                y = group[metric].to_numpy(dtype=float)
                fits.append(_fit_curve(x, y, axis=axis, metric=metric, track=track).__dict__)
    return pd.DataFrame(fits)


def _custom_data(group: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            group["cell_id"],
            group["track"].map(TRACK_LABELS),
            group["rung"],
            group["total_parameters"] / 1e6,
            group["materialized_tokens"] / 1e9,
            group["compute_flops"] / 1e18,
            group["total_parameter_tpp"],
            group["best_tied_p0"],
            group["best_tied_bpb"],
            group["two_phase_p0"],
            group["two_phase_p1"],
            group["aggregate_two_phase_starcoder"],
            group["two_phase_p1"] - group["two_phase_p0"],
            group["two_phase_bpb"],
            group["policy_class_gap_bpb"],
            group["optimum_l2_distance"],
            group["non_embedding_tpp"],
            group["bootstrap_positive_gain_probability"],
            group["confirmation_eligible"],
            group["tied_starcoder_phase_0_simulated_epochs"],
            group["tied_starcoder_phase_1_simulated_epochs"],
            group["tied_starcoder_total_simulated_epochs"],
            group["tied_nemotron_phase_0_simulated_epochs"],
            group["tied_nemotron_phase_1_simulated_epochs"],
            group["tied_nemotron_total_simulated_epochs"],
            group["two_phase_starcoder_phase_0_simulated_epochs"],
            group["two_phase_starcoder_phase_1_simulated_epochs"],
            group["two_phase_starcoder_total_simulated_epochs"],
            group["two_phase_nemotron_phase_0_simulated_epochs"],
            group["two_phase_nemotron_phase_1_simulated_epochs"],
            group["two_phase_nemotron_total_simulated_epochs"],
        ]
    )


def _hover_template(axis_title: str, metric: str) -> str:
    common = (
        "<b>%{customdata[0]}</b><br>"
        + "%{customdata[1]} · rung %{customdata[2]:.0f}<br>"
        + f"{axis_title}: %{{x:.4g}}<br>"
        + "N: %{customdata[3]:.1f}M · D: %{customdata[4]:.3f}B<br>"
        + "Compute: %{customdata[5]:.3f}e18 FLOPs<br>"
        + "Total/non-embedding TPP: %{customdata[6]:.3f} / %{customdata[16]:.3f}<br><br>"
    )
    if metric == "optimum_l2_distance":
        detail = (
            "<b>Quartic-ridge fitted optima</b><br>"
            "Tied: p=%{customdata[7]:.3f}<br>"
            "Tied StarCoder epochs: %{customdata[19]:.3f} early + %{customdata[20]:.3f} late = %{customdata[21]:.3f}<br>"
            "Tied Nemotron epochs: %{customdata[22]:.3f} early + %{customdata[23]:.3f} late = %{customdata[24]:.3f}<br>"
            "Two phase: p0=%{customdata[9]:.3f}, p1=%{customdata[10]:.3f}<br>"
            "2p StarCoder epochs: %{customdata[25]:.3f} early + %{customdata[26]:.3f} late = %{customdata[27]:.3f}<br>"
            "2p Nemotron epochs: %{customdata[28]:.3f} early + %{customdata[29]:.3f} late = %{customdata[30]:.3f}<br>"
            "L2 separation: %{customdata[15]:.6f}"
        )
    elif metric == "policy_class_gap_bpb":
        detail = (
            "<b>Quartic-ridge fitted-surface estimate</b><br>"
            "Tied optimum: p=%{customdata[7]:.3f}, %{customdata[8]:.6f} BPB<br>"
            "Tied StarCoder epochs: %{customdata[19]:.3f} early + %{customdata[20]:.3f} late = %{customdata[21]:.3f}<br>"
            "Tied Nemotron epochs: %{customdata[22]:.3f} early + %{customdata[23]:.3f} late = %{customdata[24]:.3f}<br>"
            "Two-phase optimum: p0=%{customdata[9]:.3f}, p1=%{customdata[10]:.3f}, %{customdata[13]:.6f} BPB<br>"
            "2p StarCoder epochs: %{customdata[25]:.3f} early + %{customdata[26]:.3f} late = %{customdata[27]:.3f}<br>"
            "2p Nemotron epochs: %{customdata[28]:.3f} early + %{customdata[29]:.3f} late = %{customdata[30]:.3f}<br>"
            "Gain: %{customdata[14]:.6f} BPB<br>"
            "Model-bootstrap P(gain&gt;0): %{customdata[17]:.3f}<br>"
            "Passes frozen discovery gate: %{customdata[18]}"
        )
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return common + detail + "<extra></extra>"


def _confirmation_summary(path: Path, design_path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = {"cell_id", "mean_gain_bpb", "ci95_low", "ci95_high", "confirmed"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Confirmation summary is missing fields: {sorted(missing)}")
    design = pd.read_csv(design_path)
    required_design = ("cell_id", "role", "phase_0_starcoder", "phase_1_starcoder")
    missing = set(required_design) - set(design.columns)
    if missing:
        raise ValueError(f"Confirmation design is missing fields: {sorted(missing)}")
    coordinates = design[list(required_design)].drop_duplicates()
    if coordinates.duplicated(["cell_id", "role"]).any():
        raise ValueError("Confirmation design has multiple coordinates for one cell and role")
    candidate = coordinates.loc[coordinates["role"].eq("untied_candidate")].rename(
        columns={"phase_0_starcoder": "candidate_p0", "phase_1_starcoder": "candidate_p1"}
    )
    comparator = coordinates.loc[coordinates["role"].eq("tied_comparator")].rename(
        columns={"phase_0_starcoder": "comparator_p0", "phase_1_starcoder": "comparator_p1"}
    )
    frame = frame.merge(candidate[["cell_id", "candidate_p0", "candidate_p1"]], on="cell_id", validate="one_to_one")
    frame = frame.merge(comparator[["cell_id", "comparator_p0", "comparator_p1"]], on="cell_id", validate="one_to_one")
    _add_epoch_columns(frame, prefix="candidate", phase_0="candidate_p0", phase_1="candidate_p1")
    _add_epoch_columns(frame, prefix="comparator", phase_0="comparator_p0", phase_1="comparator_p1")
    return frame


def _gain_label(value: float) -> str:
    return "0" if value <= 5e-8 else f"+{value:.5f}"


def build_figure(
    expanded: pd.DataFrame,
    confirmation: pd.DataFrame,
) -> go.Figure:
    """Build one scaling-setting row with two metric traces per active panel."""
    figure = make_subplots(
        rows=3,
        cols=4,
        specs=[[{"secondary_y": True} for _ in range(4)] for _ in range(3)],
        vertical_spacing=0.11,
        horizontal_spacing=0.08,
        subplot_titles=(
            "Model size N",
            "Token horizon D",
            "Total compute",
            "Tokens per parameter D/N",
            "Model size N",
            "Token horizon D",
            "Total compute",
            "Tokens per parameter D/N",
            "Model size N",
            "Token horizon D",
            "Total compute",
            "Tokens per parameter D/N",
        ),
    )
    distance_top = 1.12 * float(expanded["optimum_l2_distance"].max())
    gap_values = [
        float(expanded["policy_class_gap_bpb"].max()),
        float(expanded["raw_policy_class_gap_bpb"].max()),
    ]
    if not confirmation.empty:
        gap_values.append(float(confirmation["ci95_high"].max()))
    gap_top = 1.35 * max(gap_values)
    axes = tuple(AXES)
    column_centers = (0.095, 0.365, 0.635, 0.905)
    row_centers = (0.87, 0.50, 0.13)

    for row_index, track in enumerate(TRACK_ORDER, start=1):
        group = expanded.loc[expanded["track"].eq(track)].sort_values("rung")
        for column_index, axis in enumerate(axes, start=1):
            axis_title, x_column, divisor = AXES[axis]
            if track not in VARYING_TRACKS[axis]:
                fixed_value = float(group[x_column].iloc[0]) / divisor
                short_axis = "N" if axis == "parameters_m" else "D"
                figure.add_annotation(
                    text=f"<b>{short_axis} held fixed</b><br>{fixed_value:.4g}",
                    x=column_centers[column_index - 1],
                    y=row_centers[row_index - 1],
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font={"size": 16, "color": PAPER_TEXT},
                )
                figure.update_xaxes(visible=False, row=row_index, col=column_index)
                figure.update_yaxes(visible=False, row=row_index, col=column_index, secondary_y=False)
                figure.update_yaxes(visible=False, row=row_index, col=column_index, secondary_y=True)
                continue

            x = group[x_column].to_numpy(dtype=float) / divisor
            for metric, secondary_y in (
                ("optimum_l2_distance", False),
                ("policy_class_gap_bpb", True),
            ):
                metric_label = (
                    "Quartic-ridge optimum separation"
                    if metric == "optimum_l2_distance"
                    else "Quartic-ridge fitted-surface gain"
                )
                y = group[metric].to_numpy(dtype=float)
                is_gain = metric == "policy_class_gap_bpb"
                figure.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines+markers+text" if is_gain else "lines+markers",
                        name=metric_label,
                        legendgroup=metric,
                        showlegend=row_index == 1 and column_index == 2,
                        line={"color": METRIC_COLORS[metric], "width": 1.7},
                        marker={
                            "color": METRIC_COLORS[metric],
                            "line": {"color": METRIC_COLORS[metric], "width": 2.0},
                            "size": 10,
                            "symbol": METRIC_SYMBOLS[metric],
                        },
                        text=[_gain_label(value) for value in y] if is_gain else None,
                        textposition="top center",
                        textfont={"color": METRIC_COLORS[metric], "size": 11},
                        cliponaxis=False,
                        customdata=_custom_data(group),
                        hovertemplate=_hover_template(axis_title, metric),
                    ),
                    row=row_index,
                    col=column_index,
                    secondary_y=secondary_y,
                )
                if is_gain:
                    figure.add_trace(
                        go.Scatter(
                            x=x,
                            y=group["raw_policy_class_gap_bpb"],
                            mode="markers",
                            name="Raw observed min-vs-min gain",
                            legendgroup="raw-gain",
                            showlegend=row_index == 1 and column_index == 2,
                            marker={
                                "color": METRIC_COLORS[metric],
                                "size": 8,
                                "symbol": "circle-open",
                                "opacity": 0.55,
                            },
                            customdata=np.column_stack(
                                [
                                    group["cell_id"],
                                    group["track"].map(TRACK_LABELS),
                                    group["raw_best_tied_bpb"],
                                    group["raw_best_untied_bpb"],
                                    group["raw_best_tied_weight"],
                                    group["raw_best_untied_p0"],
                                    group["raw_best_untied_p1"],
                                    group["raw_tied_starcoder_phase_0_simulated_epochs"],
                                    group["raw_tied_starcoder_phase_1_simulated_epochs"],
                                    group["raw_tied_starcoder_total_simulated_epochs"],
                                    group["raw_tied_nemotron_phase_0_simulated_epochs"],
                                    group["raw_tied_nemotron_phase_1_simulated_epochs"],
                                    group["raw_tied_nemotron_total_simulated_epochs"],
                                    group["raw_two_phase_starcoder_phase_0_simulated_epochs"],
                                    group["raw_two_phase_starcoder_phase_1_simulated_epochs"],
                                    group["raw_two_phase_starcoder_total_simulated_epochs"],
                                    group["raw_two_phase_nemotron_phase_0_simulated_epochs"],
                                    group["raw_two_phase_nemotron_phase_1_simulated_epochs"],
                                    group["raw_two_phase_nemotron_total_simulated_epochs"],
                                ]
                            ),
                            hovertemplate=(
                                "<b>%{customdata[0]}</b><br>"
                                "%{customdata[1]}<br>"
                                "Raw observed gain: %{y:.6f} BPB<br>"
                                "Raw tied minimum: p=%{customdata[4]:.3f}, %{customdata[2]:.6f} BPB<br>"
                                "Tied StarCoder epochs: %{customdata[7]:.3f} early + "
                                "%{customdata[8]:.3f} late = %{customdata[9]:.3f}<br>"
                                "Tied Nemotron epochs: %{customdata[10]:.3f} early + "
                                "%{customdata[11]:.3f} late = %{customdata[12]:.3f}<br>"
                                "Raw untied minimum: p0=%{customdata[5]:.3f}, p1=%{customdata[6]:.3f}, "
                                "%{customdata[3]:.6f} BPB<br>"
                                "2p StarCoder epochs: %{customdata[13]:.3f} early + "
                                "%{customdata[14]:.3f} late = %{customdata[15]:.3f}<br>"
                                "2p Nemotron epochs: %{customdata[16]:.3f} early + "
                                "%{customdata[17]:.3f} late = %{customdata[18]:.3f}"
                                "<extra>one reference seed per coordinate; selected minima</extra>"
                            ),
                        ),
                        row=row_index,
                        col=column_index,
                        secondary_y=True,
                    )
            if track == "increase_d" and not confirmation.empty:
                merged = confirmation.merge(
                    group.drop_duplicates("cell_id"),
                    on="cell_id",
                    how="left",
                    validate="one_to_one",
                )
                confirmation_x = merged[x_column].to_numpy(dtype=float) / divisor
                mean = merged["mean_gain_bpb"].to_numpy(dtype=float)
                figure.add_trace(
                    go.Scatter(
                        x=confirmation_x,
                        y=mean,
                        mode="markers",
                        name="Stage-2 pair · 8 fresh seeds",
                        legendgroup="confirmation",
                        showlegend=column_index == 2,
                        marker={
                            "color": CONFIRMATION_COLOR,
                            "size": 15,
                            "symbol": "star",
                            "line": {"color": PLOT_BACKGROUND, "width": 1.2},
                        },
                        error_y={
                            "type": "data",
                            "array": merged["ci95_high"].to_numpy(dtype=float) - mean,
                            "arrayminus": mean - merged["ci95_low"].to_numpy(dtype=float),
                            "color": CONFIRMATION_COLOR,
                            "thickness": 1.5,
                        },
                        customdata=np.column_stack(
                            [
                                merged["cell_id"],
                                merged["ci95_low"],
                                merged["ci95_high"],
                                merged["confirmed"],
                                merged["candidate_p0"],
                                merged["candidate_p1"],
                                merged["comparator_p0"],
                                merged["comparator_p1"],
                                merged["candidate_starcoder_phase_0_simulated_epochs"],
                                merged["candidate_starcoder_phase_1_simulated_epochs"],
                                merged["candidate_starcoder_total_simulated_epochs"],
                                merged["candidate_nemotron_phase_0_simulated_epochs"],
                                merged["candidate_nemotron_phase_1_simulated_epochs"],
                                merged["candidate_nemotron_total_simulated_epochs"],
                                merged["comparator_starcoder_phase_0_simulated_epochs"],
                                merged["comparator_starcoder_phase_1_simulated_epochs"],
                                merged["comparator_starcoder_total_simulated_epochs"],
                                merged["comparator_nemotron_phase_0_simulated_epochs"],
                                merged["comparator_nemotron_phase_1_simulated_epochs"],
                                merged["comparator_nemotron_total_simulated_epochs"],
                            ]
                        ),
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            + f"{axis_title}: %{{x:.4g}}<br>"
                            + "Stage-2 selected policy pair<br>"
                            + "8 fresh paired seeds<br>"
                            + "Candidate: p0=%{customdata[4]:.3f}, p1=%{customdata[5]:.3f}<br>"
                            + "Candidate StarCoder epochs: %{customdata[8]:.3f} early + "
                            + "%{customdata[9]:.3f} late = %{customdata[10]:.3f}<br>"
                            + "Candidate Nemotron epochs: %{customdata[11]:.3f} early + "
                            + "%{customdata[12]:.3f} late = %{customdata[13]:.3f}<br>"
                            + "Comparator: p0=%{customdata[6]:.3f}, p1=%{customdata[7]:.3f}<br>"
                            + "Comparator StarCoder epochs: %{customdata[14]:.3f} early + "
                            + "%{customdata[15]:.3f} late = %{customdata[16]:.3f}<br>"
                            + "Comparator Nemotron epochs: %{customdata[17]:.3f} early + "
                            + "%{customdata[18]:.3f} late = %{customdata[19]:.3f}<br>"
                            + "Paired mean gain: %{y:.6f} BPB<br>"
                            + "95% CI: [%{customdata[1]:.6f}, %{customdata[2]:.6f}]<br>"
                            + "Confirmation passed: %{customdata[3]}"
                            + "<extra>near, not identical to, the later Stage-3 fitted argmin</extra>"
                        ),
                    ),
                    row=row_index,
                    col=column_index,
                    secondary_y=True,
                )

            figure.update_xaxes(
                type="log",
                title_text=axis_title,
                gridcolor=GRID_COLOR,
                zeroline=False,
                showline=True,
                linecolor=PAPER_TEXT,
                ticks="outside",
                row=row_index,
                col=column_index,
            )
            figure.update_yaxes(
                title_text="L2 distance",
                range=[0.0, distance_top],
                gridcolor=GRID_COLOR,
                zeroline=True,
                zerolinecolor=PAPER_TEXT,
                zerolinewidth=1.2,
                showline=True,
                linecolor=PAPER_TEXT,
                ticks="outside",
                tickformat=".3f",
                row=row_index,
                col=column_index,
                secondary_y=False,
            )
            figure.update_yaxes(
                title_text="Two Phase Gain (BPB)<br><sup>larger is better</sup>",
                range=[0.0, gap_top],
                showgrid=False,
                zeroline=True,
                zerolinecolor=PAPER_TEXT,
                zerolinewidth=1.2,
                showline=True,
                linecolor=PAPER_TEXT,
                ticks="outside",
                tickformat=".4f",
                row=row_index,
                col=column_index,
                secondary_y=True,
            )

    confirmation_note = (
        "The black star is a Stage-2 selected policy pair evaluated on eight fresh paired seeds; "
        "it is near, but not identical to, the later Stage-3 fitted-surface argmin."
        if not confirmation.empty
        else "The promoted high-TPP fixed-N cell is awaiting its 8-pair fresh-seed confirmation."
    )
    row_annotations = [
        {
            "text": TRACK_LABELS[track],
            "x": -0.075,
            "xref": "paper",
            "y": center,
            "yref": "paper",
            "showarrow": False,
            "textangle": -90,
            "font": {"family": "Georgia, serif", "size": 19, "color": PAPER_TEXT},
        }
        for track, center in zip(TRACK_ORDER, row_centers, strict=True)
    ]
    figure.update_layout(
        title={
            "text": (
                "StarCoder 80/20 WSD phase-optimum scaling"
                "<br><sup>714-run dense discovery panel · quartic-ridge optima from the frozen "
                "spatial-CV procedure</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
            "pad": {"b": 24},
            "font": {"family": "Georgia, serif", "size": 28, "color": PAPER_TEXT},
        },
        width=2400,
        height=1800,
        paper_bgcolor=PAPER_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        font={"family": "Avenir Next, Helvetica Neue, sans-serif", "size": 13, "color": PAPER_TEXT},
        hoverlabel={"font": {"family": "Avenir Next, Helvetica Neue, sans-serif", "size": 13}},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.02,
            "yanchor": "bottom",
            "bgcolor": "rgba(255,255,255,0.72)",
            "bordercolor": GRID_COLOR,
            "borderwidth": 1,
        },
        margin={"l": 170, "r": 100, "t": 260, "b": 270},
        annotations=[
            *figure.layout.annotations,
            *row_annotations,
            {
                "text": (
                    "Filled diamonds and their solid connecting lines show gain between the preregistered "
                    "smoothed tied and untied optima; residual-bootstrap diagnostics "
                    "remain in the companion table but are not plotted.<br>"
                    "Open circles show the descriptive, selection-biased gain between the lowest raw tied and "
                    "untied observations.<br>"
                    f"{confirmation_note}<br>Exact fitted candidate locations remain unstable in several cells."
                    "<br>The D/N column uses total parameters and summarizes each intervention track separately; "
                    "it does not assume TPP is a sufficient pooled scaling variable."
                ),
                "x": 0.5,
                "xref": "paper",
                "y": -0.15,
                "yref": "paper",
                "showarrow": False,
                "xanchor": "center",
                "align": "center",
                "font": {"size": 13, "color": PAPER_TEXT},
            },
        ],
    )
    return figure


def write_report(
    output_dir: Path,
    optima: pd.DataFrame,
    fits: pd.DataFrame,
    confirmation: pd.DataFrame,
) -> None:
    eligible = optima.loc[optima["confirmation_eligible"]]
    lines = [
        "# StarCoder WSD80 matched-N,D optimum scaling",
        "",
        "- Source: completed 714-run Stage-1/2/3 discovery panel over ten N,D cells.",
        (
            f"- Frozen smooth-gain gate: {len(eligible)}/{len(optima)} cells. The sole eligible cell is the "
            "high-token fixed-N endpoint."
        ),
        "- Smooth optima come from the preregistered quartic-ridge surface selected by deterministic spatial CV; "
        "raw selected minima are retained only as descriptive, selection-biased points.",
        (
            "- Gain intervals are the frozen leverage-corrected residual bootstrap with ridge held fixed. "
            "They remain in the table but are omitted from the headline plot. Candidate-location bootstrap "
            "displacement is reported separately and is large in several cells."
        ),
        "- Scaling curves are nonnegative log-linear OLS summaries over four cells per varying track. With n=4 and "
        "partly collinear N,D interventions, they are descriptive rather than inferential scaling laws.",
        "- The total-parameter D/N panels preserve the three intervention tracks separately. They do not assert that "
        "TPP is sufficient to collapse model-size and token-horizon effects.",
        (
            "- The Stage-2 selected pair's eight-fresh-seed confirmation is overlaid; it is not the exact "
            "later Stage-3 fitted-surface argmin."
            if not confirmation.empty
            else "- The promoted high-TPP fixed-N discovery winner is still awaiting fresh-seed confirmation."
        ),
        "",
        "## Per-cell discovered optima",
        "",
        optima[
            [
                "cell_id",
                "rung",
                "total_parameters",
                "materialized_tokens",
                "compute_flops",
                "total_parameter_tpp",
                "best_tied_p0",
                "best_tied_bpb",
                "two_phase_p0",
                "two_phase_p1",
                "two_phase_bpb",
                "optimum_l2_distance",
                "policy_class_gap_bpb",
                "bootstrap_gain_p05",
                "bootstrap_gain_p95",
                "bootstrap_positive_gain_probability",
                "bootstrap_candidate_l2_p90",
                "raw_policy_class_gap_bpb",
                "stage1_policy_class_gain_bpb",
                "confirmation_eligible",
            ]
        ].to_markdown(index=False, floatfmt=".7f"),
        "",
        "## Descriptive curve fits",
        "",
        fits.to_markdown(index=False, floatfmt=".7f"),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    optima = discovered_optima(
        args.discovery_summary,
        args.fitted_candidates,
        args.source_design,
        args.stage1_summary,
    )
    expanded = expanded_tracks(optima)
    fits = scaling_fits(expanded)
    confirmation = _confirmation_summary(args.confirmation_summary, args.confirmation_design)
    figure = build_figure(expanded, confirmation)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    optima.to_csv(args.output_dir / "discovered_optimum_scaling.csv", index=False)
    fits.to_csv(args.output_dir / "descriptive_scaling_curve_fits.csv", index=False)
    figure.write_html(
        args.output_dir / "starcoder_wsd80_matched_nd_optimum_scaling.html",
        include_plotlyjs=True,
        config=EXPORT_CONFIG,
    )
    figure.write_image(args.output_dir / "starcoder_wsd80_matched_nd_optimum_scaling.png", scale=2)
    write_report(args.output_dir, optima, fits, confirmation)


if __name__ == "__main__":
    main()
