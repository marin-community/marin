# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gcsfs>=2025.1",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scipy>=1.14",
# ]
# ///

"""Analyze exact endpoints from the StarCoder coupled-onset dense surfaces."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import gcsfs
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.spatial import Delaunay
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
DESIGN_PATH = REPO_ROOT / (
    "experiments/domain_phase_mix/starcoder_wsd80_coupled_onset_dense_surface_design_20260830.json.gz"
)
OUTPUT_DIR = REPO_ROOT / (
    "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "starcoder_wsd80_coupled_onset_dense_surface_results_20260901"
)
CHECKPOINT_ROOT = (
    "gs://marin-us-central2/checkpoints/pinlin_calvin_xu/data_mixture/"
    "starcoder_wsd80_coupled_onset_dense_surfaces_central2_v4_20260830"
)
CHECKPOINT_VERSION = "2026.08.30.1"
IRIS_ROOT = "/calvinxu/dm-starcoder-wsd80-coupled-onset-dense-surface-central2-v4-v1-20260830"
EXPECTED_DESIGN_SHA256 = "423bfb51e546181f78c04863b6298d57a32a38b33b25eae6a1d1464a737010eb"
EXPECTED_ENDPOINT_STEP = 28_259
EXPECTED_RUNS = 375
EXPECTED_COORDINATES = 125
ARMS = ("coupled_0p60", "coupled_0p80", "coupled_0p90")
REFERENCE_ARM = "coupled_0p80"
ARM_LABELS = {
    "coupled_0p60": "0.60T",
    "coupled_0p80": "0.80T",
    "coupled_0p90": "0.90T",
}
SURFACE_METRICS_3D = (
    ("programming_languages_bpb", "Programming Languages BPB"),
    ("c4_bpb", "C4 BPB"),
)
PRIMARY_METRIC = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
BROAD_METRIC = "eval/paloma/c4_en-llama3/bpb"
UNSCALED_METRICS = {
    "programming_languages_bpb": PRIMARY_METRIC,
    "c4_bpb": BROAD_METRIC,
    "uncheatable_bpb": "eval/uncheatable_eval/bpb",
    "github_cpp_bpb": "eval/uncheatable_eval/github_cpp-llama3/bpb",
    "github_python_bpb": "eval/uncheatable_eval/github_python-llama3/bpb",
}
SELECTION_CLASSES = {"tied": 26, "eligible_untied": 94, "ineligible_near_tied": 5}
HISTORICAL_TIED_COORDINATE = "c109"
HISTORICAL_UNTIED_COORDINATE = "c020"
PAIRED_GAIN_NOISE_SD_BPB = 0.001182
SURFACE_EDGE_TOLERANCE = 1e-8
PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 4},
}


def canonical_sha256(value: Any) -> str:
    """Return a stable hash for a JSON-compatible value."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_design() -> tuple[dict[str, Any], pd.DataFrame]:
    """Load and validate the frozen design."""
    payload = json.loads(gzip.decompress(DESIGN_PATH.read_bytes()))
    claimed_hash = payload.pop("design_sha256", None)
    observed_hash = canonical_sha256(payload)
    if claimed_hash != EXPECTED_DESIGN_SHA256 or observed_hash != EXPECTED_DESIGN_SHA256:
        raise ValueError(f"Design hash drifted: {claimed_hash=} {observed_hash=}")
    runs = pd.DataFrame(payload["runs"])
    if len(runs) != EXPECTED_RUNS or runs["row_id"].duplicated().any() or runs["run_name"].duplicated().any():
        raise ValueError("Frozen run inventory is incomplete or non-unique")
    if tuple(runs["arm_id"].drop_duplicates()) != ARMS:
        raise ValueError("Arm order or identity drifted")
    for arm in ARMS:
        arm_rows = runs.loc[runs["arm_id"].eq(arm)]
        observed_classes = arm_rows["selection_class"].value_counts().to_dict()
        if len(arm_rows) != EXPECTED_COORDINATES or observed_classes != SELECTION_CLASSES:
            raise ValueError(f"{arm}: coordinate inventory drifted: {observed_classes}")
    return payload, runs


def _endpoint_uri(run_name: str) -> str:
    return f"{CHECKPOINT_ROOT}/{run_name}/{CHECKPOINT_VERSION}/checkpoints/eval_metrics.jsonl"


def _read_endpoint(filesystem: gcsfs.GCSFileSystem, record: dict[str, Any]) -> dict[str, Any]:
    uri = _endpoint_uri(str(record["run_name"]))
    with filesystem.open(uri.removeprefix("gs://"), "rb") as handle:
        rows = [json.loads(line) for line in handle.read().decode().splitlines() if line.strip()]
    endpoints = [row for row in rows if int(row.get("step", -1)) == EXPECTED_ENDPOINT_STEP]
    if len(endpoints) != 1:
        raise ValueError(f"{record['run_name']}: expected one exact endpoint, found {len(endpoints)}")
    endpoint = endpoints[0]
    metrics = {name: float(endpoint[key]) for name, key in UNSCALED_METRICS.items()}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"{record['run_name']}: non-finite endpoint metric")
    return {
        **record,
        **metrics,
        "endpoint_step": EXPECTED_ENDPOINT_STEP,
        "eval_metrics_uri": uri,
    }


def collect_endpoints(runs: pd.DataFrame) -> pd.DataFrame:
    """Read all exact endpoints concurrently from durable GCS metrics."""
    filesystem = gcsfs.GCSFileSystem(token="google_default")
    records = runs.to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=32) as pool:
        observations = list(pool.map(lambda record: _read_endpoint(filesystem, record), records))
    frame = pd.DataFrame(observations).sort_values("run_order").reset_index(drop=True)
    if len(frame) != EXPECTED_RUNS or frame["coordinate_id"].nunique() != EXPECTED_COORDINATES:
        raise ValueError("Exact endpoint inventory is incomplete")
    if frame.groupby("coordinate_id")["arm_id"].nunique().ne(len(ARMS)).any():
        raise ValueError("At least one coordinate is not crossed over all arms")
    return frame


def selected_policies(observations: pd.DataFrame) -> pd.DataFrame:
    """Apply the preregistered raw-grid selection rule to each arm."""
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_rows = observations.loc[observations["arm_id"].eq(arm)]
        tied = arm_rows.loc[arm_rows["selection_class"].eq("tied")]
        untied = arm_rows.loc[arm_rows["selection_class"].eq("eligible_untied")]
        selected_tied = tied.loc[tied["programming_languages_bpb"].idxmin()]
        selected_untied = untied.loc[untied["programming_languages_bpb"].idxmin()]
        gain = float(selected_tied["programming_languages_bpb"] - selected_untied["programming_languages_bpb"])
        for policy_type, selected in (("tied", selected_tied), ("eligible_untied", selected_untied)):
            rows.append(
                {
                    "arm_id": arm,
                    "requested_onset_fraction": float(selected["requested_onset_fraction"]),
                    "realized_onset_fraction": float(selected["realized_onset_fraction"]),
                    "policy_type": policy_type,
                    "coordinate_id": selected["coordinate_id"],
                    "programming_languages_bpb": float(selected["programming_languages_bpb"]),
                    "c4_bpb": float(selected["c4_bpb"]),
                    "aggregate_starcoder": float(selected["aggregate_starcoder"]),
                    "phase_0_starcoder": float(selected["phase_0_starcoder"]),
                    "phase_1_starcoder": float(selected["phase_1_starcoder"]),
                    "phase_contrast": float(selected["phase_contrast"]),
                    "normalized_fiber_position": float(selected["normalized_fiber_position"]),
                    "raw_grid_gain_bpb": gain,
                }
            )
    selected = pd.DataFrame(rows)
    if len(selected) != 2 * len(ARMS):
        raise ValueError("Raw-grid selection failed")
    return selected


def arm_summary(observations: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """Summarize the selected gain and fixed-coordinate descriptive control."""
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_rows = observations.loc[observations["arm_id"].eq(arm)].set_index("coordinate_id")
        chosen = selected.loc[selected["arm_id"].eq(arm)].set_index("policy_type")
        tied = chosen.loc["tied"]
        untied = chosen.loc["eligible_untied"]
        tied_values = np.sort(
            arm_rows.loc[arm_rows["selection_class"].eq("tied"), "programming_languages_bpb"].to_numpy(dtype=float)
        )
        untied_values = np.sort(
            arm_rows.loc[arm_rows["selection_class"].eq("eligible_untied"), "programming_languages_bpb"].to_numpy(
                dtype=float
            )
        )
        historical_gain = float(
            arm_rows.loc[HISTORICAL_TIED_COORDINATE, "programming_languages_bpb"]
            - arm_rows.loc[HISTORICAL_UNTIED_COORDINATE, "programming_languages_bpb"]
        )
        rows.append(
            {
                "arm_id": arm,
                "requested_onset_fraction": float(tied["requested_onset_fraction"]),
                "realized_onset_fraction": float(tied["realized_onset_fraction"]),
                "boundary_step": int(observations.loc[observations["arm_id"].eq(arm), "boundary_step"].iloc[0]),
                "phase1_lr_integral": float(
                    observations.loc[observations["arm_id"].eq(arm), "optimizer"].iloc[0][
                        "normalized_phase_1_lr_integral"
                    ]
                ),
                "selected_tied_coordinate": tied["coordinate_id"],
                "selected_untied_coordinate": untied["coordinate_id"],
                "selected_tied_bpb": float(tied["programming_languages_bpb"]),
                "selected_untied_bpb": float(untied["programming_languages_bpb"]),
                "selected_gain_bpb": float(tied["raw_grid_gain_bpb"]),
                "selected_c4_gain_bpb": float(tied["c4_bpb"] - untied["c4_bpb"]),
                "selected_tied_margin_to_second_bpb": float(tied_values[1] - tied_values[0]),
                "selected_untied_margin_to_second_bpb": float(untied_values[1] - untied_values[0]),
                "selected_absolute_phase_contrast": abs(float(untied["phase_contrast"])),
                "selected_absolute_normalized_fiber": abs(float(untied["normalized_fiber_position"])),
                "historical_c109_minus_c020_gain_bpb": historical_gain,
                "selection_inflation_vs_historical_pair_bpb": float(tied["raw_grid_gain_bpb"]) - historical_gain,
                "selected_gain_in_paired_noise_sd": float(tied["raw_grid_gain_bpb"]) / PAIRED_GAIN_NOISE_SD_BPB,
                "historical_gain_in_paired_noise_sd": historical_gain / PAIRED_GAIN_NOISE_SD_BPB,
            }
        )
    return pd.DataFrame(rows)


def c4_noninferiority_sensitivity(observations: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """Select the best programming branch that does not regress C4 versus the selected tied policy."""
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_rows = observations.loc[observations["arm_id"].eq(arm)]
        tied = selected.loc[selected["arm_id"].eq(arm) & selected["policy_type"].eq("tied")].iloc[0]
        eligible = arm_rows.loc[
            arm_rows["selection_class"].eq("eligible_untied") & arm_rows["c4_bpb"].le(float(tied["c4_bpb"]))
        ]
        chosen = eligible.loc[eligible["programming_languages_bpb"].idxmin()]
        rows.append(
            {
                "arm_id": arm,
                "requested_onset_fraction": float(chosen["requested_onset_fraction"]),
                "tied_coordinate": tied["coordinate_id"],
                "selected_untied_coordinate": chosen["coordinate_id"],
                "eligible_untied_coordinates": len(eligible),
                "programming_gain_bpb": float(tied["programming_languages_bpb"] - chosen["programming_languages_bpb"]),
                "c4_gain_bpb": float(tied["c4_bpb"] - chosen["c4_bpb"]),
            }
        )
    return pd.DataFrame(rows)


def expected_asymmetric_null_gain() -> float:
    """Approximate best-of-bank bias under iid equal-mean Gaussian policy noise."""
    policy_noise_sd = PAIRED_GAIN_NOISE_SD_BPB / math.sqrt(2.0)

    def blom_expected_max(sample_count: int) -> float:
        probability = (sample_count - 0.375) / (sample_count + 0.25)
        return NormalDist().inv_cdf(probability)

    return policy_noise_sd * (
        blom_expected_max(SELECTION_CLASSES["eligible_untied"]) - blom_expected_max(SELECTION_CLASSES["tied"])
    )


def matched_coordinate_deformation(observations: pd.DataFrame) -> pd.DataFrame:
    """Describe how each complete surface changes relative to the 0.80 reference arm."""
    wide = observations.pivot(
        index="coordinate_id",
        columns="arm_id",
        values="programming_languages_bpb",
    )
    rows: list[dict[str, Any]] = []
    reference = wide[REFERENCE_ARM].to_numpy(dtype=float)
    for arm in ARMS:
        values = wide[arm].to_numpy(dtype=float)
        delta = values - reference
        rows.append(
            {
                "arm_id": arm,
                "reference_arm": REFERENCE_ARM,
                "coordinates": len(delta),
                "mean_delta_bpb": float(delta.mean()),
                "median_delta_bpb": float(np.median(delta)),
                "centered_sd_bpb": float(np.std(delta - delta.mean(), ddof=1)),
                "q05_delta_bpb": float(np.quantile(delta, 0.05)),
                "q95_delta_bpb": float(np.quantile(delta, 0.95)),
                "spearman_with_reference": float(spearmanr(values, reference).statistic),
            }
        )
    return pd.DataFrame(rows)


def _surface_plot(observations: pd.DataFrame, selected: pd.DataFrame) -> Path:
    minimum = float(observations["programming_languages_bpb"].min())
    maximum = float(observations["programming_languages_bpb"].quantile(0.98))
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Coupled onset 0.60T", "Coupled onset 0.80T", "Coupled onset 0.90T"),
        horizontal_spacing=0.06,
    )
    for column, arm in enumerate(ARMS, start=1):
        arm_rows = observations.loc[observations["arm_id"].eq(arm)]
        figure.add_trace(
            go.Scatter(
                x=arm_rows["aggregate_starcoder"],
                y=arm_rows["normalized_fiber_position"],
                mode="markers",
                showlegend=False,
                marker={
                    "size": 9,
                    "color": arm_rows["programming_languages_bpb"],
                    "colorscale": "RdYlGn_r",
                    "cmin": minimum,
                    "cmax": maximum,
                    "showscale": column == 3,
                    "colorbar": {"title": "Programming<br>BPB"} if column == 3 else None,
                    "line": {"width": 0.3, "color": "#17324D"},
                },
                customdata=np.stack([arm_rows["coordinate_id"], arm_rows["programming_languages_bpb"]], axis=1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>Aggregate StarCoder %{x:.3f}"
                    "<br>Normalized fiber %{y:.3f}<br>BPB %{customdata[1]:.6f}<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )
        chosen = selected.loc[selected["arm_id"].eq(arm)]
        for policy_type, symbol, label in (
            ("tied", "diamond-open", "Selected tied"),
            ("eligible_untied", "star-open", "Selected untied"),
        ):
            point = chosen.loc[chosen["policy_type"].eq(policy_type)]
            figure.add_trace(
                go.Scatter(
                    x=point["aggregate_starcoder"],
                    y=point["normalized_fiber_position"],
                    mode="markers",
                    name=label,
                    legendgroup=label,
                    showlegend=column == 1,
                    marker={"size": 18, "symbol": symbol, "color": "#102E45", "line": {"width": 2}},
                    hovertemplate=f"{label}<extra></extra>",
                ),
                row=1,
                col=column,
            )
    figure.update_layout(
        title="StarCoder WSD80 programming-language endpoint surfaces",
        height=650,
        margin={"l": 70, "r": 100, "t": 120, "b": 80},
        paper_bgcolor="#F8F3E8",
        plot_bgcolor="#F8F3E8",
        font={"family": "Avenir Next, sans-serif", "size": 15, "color": "#17324D"},
        legend={"orientation": "h", "y": 1.08, "x": 0.5, "xanchor": "center"},
    )
    figure.update_xaxes(title="Aggregate StarCoder share", gridcolor="#DCE5EA")
    figure.update_yaxes(title="Normalized phase fiber", range=[-1.05, 1.05], gridcolor="#DCE5EA")
    path = OUTPUT_DIR / "endpoint_surfaces.html"
    pio.write_html(figure, path, include_plotlyjs=True, full_html=True, config=PLOT_CONFIG)
    return path


def _surface_metric_range(values: pd.Series) -> tuple[float, float]:
    minimum = float(values.min())
    maximum = float(values.max())
    padding = max(0.02 * (maximum - minimum), 0.002)
    return minimum - padding, maximum + padding


def _surface_interior_metric_range(observations: pd.DataFrame, metric: str) -> tuple[float, float]:
    interior = observations.loc[
        observations["phase_0_starcoder"].between(
            SURFACE_EDGE_TOLERANCE,
            1.0 - SURFACE_EDGE_TOLERANCE,
            inclusive="neither",
        )
        & observations["phase_1_starcoder"].between(
            SURFACE_EDGE_TOLERANCE,
            1.0 - SURFACE_EDGE_TOLERANCE,
            inclusive="neither",
        ),
        metric,
    ]
    if interior.empty:
        raise ValueError("Cannot focus the surface without strict-interior observations")
    return _surface_metric_range(interior)


def _surface_scene(metric_label: str, metric_range: tuple[float, float]) -> dict[str, Any]:
    pane = {
        "range": [0.0, 1.0],
        "gridcolor": "#FFFFFF",
        "backgroundcolor": "#EAF0F2",
        "showbackground": True,
        "zeroline": False,
        "tickfont": {"size": 11},
    }
    return {
        "xaxis": {**pane, "title": {"text": "Phase 0<br>StarCoder share", "font": {"size": 12}}},
        "yaxis": {**pane, "title": {"text": "Phase 1<br>StarCoder share", "font": {"size": 12}}},
        "zaxis": {
            "title": {"text": "BPB", "font": {"size": 12}},
            "range": list(metric_range),
            "gridcolor": "#FFFFFF",
            "backgroundcolor": "#EAF0F2",
            "showbackground": True,
            "zeroline": False,
            "tickfont": {"size": 11},
        },
        "camera": {"eye": {"x": -1.5, "y": -1.5, "z": 1.15}},
        "aspectmode": "manual",
        "aspectratio": {"x": 1.0, "y": 1.0, "z": 0.72},
        "uirevision": "coupled-onset-3d-surface",
    }


def _surface_3d_figure(observations: pd.DataFrame, summary: pd.DataFrame) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "scene"}, {"type": "scene"}],
            [{"type": "xy", "colspan": 2}, None],
        ],
        subplot_titles=(
            *(label for _, label in SURFACE_METRICS_3D),
            "Selected Programming Languages 2p advantage over 1p",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.13,
        row_heights=[0.72, 0.28],
    )
    trace_indices: dict[str, list[int]] = {arm: [] for arm in ARMS}
    full_metric_ranges = {metric: _surface_metric_range(observations[metric]) for metric, _ in SURFACE_METRICS_3D}
    focused_metric_ranges = {
        metric: _surface_interior_metric_range(observations, metric) for metric, _ in SURFACE_METRICS_3D
    }

    for arm in ARMS:
        arm_rows = observations.loc[observations["arm_id"].eq(arm)].reset_index(drop=True)
        points = arm_rows[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
        triangles = Delaunay(points).simplices
        visible = arm == REFERENCE_ARM
        for column, (metric, metric_label) in enumerate(SURFACE_METRICS_3D, start=1):
            values = arm_rows[metric].to_numpy(dtype=float)
            color_axis = "coloraxis" if column == 1 else "coloraxis2"
            trace_indices[arm].append(len(figure.data))
            figure.add_trace(
                go.Mesh3d(
                    x=arm_rows["phase_0_starcoder"],
                    y=arm_rows["phase_1_starcoder"],
                    z=values,
                    i=triangles[:, 0],
                    j=triangles[:, 1],
                    k=triangles[:, 2],
                    intensity=values,
                    coloraxis=color_axis,
                    opacity=0.48,
                    flatshading=False,
                    showscale=True,
                    hoverinfo="skip",
                    name="Linear Delaunay mesh",
                    showlegend=False,
                    visible=visible,
                ),
                row=1,
                col=column,
            )
            customdata = list(
                zip(
                    arm_rows["coordinate_id"],
                    arm_rows["aggregate_starcoder"],
                    arm_rows["normalized_fiber_position"],
                    arm_rows["selection_class"],
                    strict=True,
                )
            )
            trace_indices[arm].append(len(figure.data))
            figure.add_trace(
                go.Scatter3d(
                    x=arm_rows["phase_0_starcoder"],
                    y=arm_rows["phase_1_starcoder"],
                    z=values,
                    mode="markers",
                    marker={
                        "size": 3.8,
                        "color": values,
                        "coloraxis": color_axis,
                        "line": {"color": "#FFFFFF", "width": 0.7},
                        "showscale": False,
                    },
                    customdata=customdata,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b>"
                        "<br>Phase 0 StarCoder %{x:.4f}"
                        "<br>Phase 1 StarCoder %{y:.4f}"
                        f"<br>{metric_label} %{{z:.6f}}"
                        "<br>Aggregate StarCoder %{customdata[1]:.4f}"
                        "<br>Normalized fiber %{customdata[2]:+.3f}"
                        "<br>Design class %{customdata[3]}<extra></extra>"
                    ),
                    name="Measured endpoints",
                    legendgroup="measured-endpoints",
                    showlegend=column == 1,
                    visible=visible,
                ),
                row=1,
                col=column,
            )
            optima = (
                (
                    "tied",
                    "Best observed 1p",
                    "diamond",
                    "#17324D",
                    arm_rows.loc[arm_rows["selection_class"].eq("tied")],
                ),
                (
                    "eligible-2p",
                    "Best observed 2p (eligible)",
                    "x",
                    "#D95A32",
                    arm_rows.loc[arm_rows["selection_class"].eq("eligible_untied")],
                ),
            )
            for legend_group, label, symbol, color, candidates in optima:
                optimum = candidates.loc[candidates[metric].idxmin()]
                trace_indices[arm].append(len(figure.data))
                figure.add_trace(
                    go.Scatter3d(
                        x=[optimum["phase_0_starcoder"]],
                        y=[optimum["phase_1_starcoder"]],
                        z=[optimum[metric]],
                        mode="markers",
                        marker={
                            "size": 8.5,
                            "symbol": symbol,
                            "color": color,
                            "line": {"color": "#FFFFFF", "width": 1.3},
                        },
                        text=[
                            "<br>".join(
                                [
                                    f"<b>{label}</b>",
                                    f"Coordinate {optimum['coordinate_id']}",
                                    f"Phase 0 StarCoder {optimum['phase_0_starcoder']:.4f}",
                                    f"Phase 1 StarCoder {optimum['phase_1_starcoder']:.4f}",
                                    f"{metric_label} {optimum[metric]:.6f}",
                                ]
                            )
                        ],
                        hoverinfo="text",
                        name=label,
                        legendgroup=legend_group,
                        showlegend=column == 1,
                        visible=visible,
                    ),
                    row=1,
                    col=column,
                )

    gain_rows = summary.set_index("arm_id").loc[list(ARMS)].reset_index()
    gain_customdata = list(
        zip(
            gain_rows["selected_tied_bpb"],
            gain_rows["selected_untied_bpb"],
            gain_rows["selected_tied_coordinate"],
            gain_rows["selected_untied_coordinate"],
            strict=True,
        )
    )
    persistent_trace_indices = [len(figure.data)]
    figure.add_trace(
        go.Scatter(
            x=gain_rows["requested_onset_fraction"],
            y=gain_rows["selected_gain_bpb"],
            mode="lines+markers+text",
            line={"color": "#D95A32", "width": 3},
            marker={"size": 10, "color": "#D95A32", "line": {"color": "#FFFFFF", "width": 1}},
            text=[f"{gain:+.4f}" for gain in gain_rows["selected_gain_bpb"]],
            textposition="top center",
            textfont={"size": 12},
            cliponaxis=False,
            customdata=gain_customdata,
            hovertemplate=(
                "<b>Coupled onset %{x:.2f}T</b>"
                "<br>2p advantage %{y:+.6f} BPB"
                "<br>Best 1p %{customdata[0]:.6f} (%{customdata[2]})"
                "<br>Best eligible 2p %{customdata[1]:.6f} (%{customdata[3]})"
                "<extra></extra>"
            ),
            name="Selected 2p advantage",
            legendgroup="selected-gain",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    for arm in ARMS:
        point = gain_rows.loc[gain_rows["arm_id"].eq(arm)].iloc[0]
        trace_indices[arm].append(len(figure.data))
        figure.add_trace(
            go.Scatter(
                x=[point["requested_onset_fraction"]],
                y=[point["selected_gain_bpb"]],
                mode="markers",
                marker={
                    "size": 18,
                    "symbol": "circle-open",
                    "color": "#17324D",
                    "line": {"color": "#17324D", "width": 3},
                },
                hoverinfo="skip",
                name="Selected onset",
                showlegend=False,
                visible=arm == REFERENCE_ARM,
            ),
            row=2,
            col=1,
        )

    steps: list[dict[str, Any]] = []
    for arm in ARMS:
        visibility = [False] * len(figure.data)
        for trace_index in persistent_trace_indices:
            visibility[trace_index] = True
        for trace_index in trace_indices[arm]:
            visibility[trace_index] = True
        steps.append(
            {
                "label": ARM_LABELS[arm],
                "method": "update",
                "args": [{"visible": visibility}],
            }
        )

    subplot_titles = list(figure.layout.annotations)
    subplot_titles[0].update(x=0.22, y=0.775, font={"size": 18, "color": "#17324D"})
    subplot_titles[1].update(x=0.78, y=0.775, font={"size": 18, "color": "#17324D"})
    subplot_titles[2].update(x=0.5, y=0.295, font={"size": 17, "color": "#17324D"})

    figure.update_layout(
        title={"text": "Endpoint response surfaces by coupled phase/LR onset", "x": 0.5, "y": 0.99},
        height=1260,
        margin={"l": 115, "r": 105, "t": 120, "b": 250},
        paper_bgcolor="#F8F3E8",
        font={"family": "Avenir Next, sans-serif", "size": 14, "color": "#17324D"},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.825,
            "yanchor": "middle",
            "font": {"size": 13},
        },
        scene={
            **_surface_scene(SURFACE_METRICS_3D[0][1], focused_metric_ranges[SURFACE_METRICS_3D[0][0]]),
            "domain": {"x": [0.0, 0.44], "y": [0.40, 0.74]},
        },
        scene2={
            **_surface_scene(SURFACE_METRICS_3D[1][1], focused_metric_ranges[SURFACE_METRICS_3D[1][0]]),
            "domain": {"x": [0.56, 1.0], "y": [0.40, 0.74]},
        },
        coloraxis={
            "colorscale": "RdYlGn_r",
            "cmin": focused_metric_ranges[SURFACE_METRICS_3D[0][0]][0],
            "cmax": focused_metric_ranges[SURFACE_METRICS_3D[0][0]][1],
            "colorbar": {
                "title": "BPB",
                "len": 0.28,
                "thickness": 12,
                "x": 0.455,
                "y": 0.57,
                "tickfont": {"size": 11},
            },
        },
        coloraxis2={
            "colorscale": "RdYlGn_r",
            "cmin": focused_metric_ranges[SURFACE_METRICS_3D[1][0]][0],
            "cmax": focused_metric_ranges[SURFACE_METRICS_3D[1][0]][1],
            "colorbar": {
                "title": "BPB",
                "len": 0.28,
                "thickness": 12,
                "x": 1.005,
                "y": 0.57,
                "tickfont": {"size": 11},
            },
        },
        sliders=[
            {
                "active": ARMS.index(REFERENCE_ARM),
                "steps": steps,
                "x": 0.24,
                "len": 0.52,
                "y": 0.94,
                "xanchor": "left",
                "yanchor": "bottom",
                "currentvalue": {"visible": False},
                "pad": {"t": 0, "b": 0},
                "transition": {"duration": 0},
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "active": 0,
                "direction": "right",
                "showactive": True,
                "x": 0.5,
                "xanchor": "center",
                "y": -0.09,
                "yanchor": "top",
                "bgcolor": "#F8F3E8",
                "bordercolor": "#9AABAF",
                "buttons": [
                    {
                        "label": "Hide edge outliers",
                        "method": "relayout",
                        "args": [
                            {
                                "scene.zaxis.range": list(focused_metric_ranges[SURFACE_METRICS_3D[0][0]]),
                                "scene2.zaxis.range": list(focused_metric_ranges[SURFACE_METRICS_3D[1][0]]),
                                "coloraxis.cmin": focused_metric_ranges[SURFACE_METRICS_3D[0][0]][0],
                                "coloraxis.cmax": focused_metric_ranges[SURFACE_METRICS_3D[0][0]][1],
                                "coloraxis2.cmin": focused_metric_ranges[SURFACE_METRICS_3D[1][0]][0],
                                "coloraxis2.cmax": focused_metric_ranges[SURFACE_METRICS_3D[1][0]][1],
                            }
                        ],
                    },
                    {
                        "label": "Show full range",
                        "method": "relayout",
                        "args": [
                            {
                                "scene.zaxis.range": list(full_metric_ranges[SURFACE_METRICS_3D[0][0]]),
                                "scene2.zaxis.range": list(full_metric_ranges[SURFACE_METRICS_3D[1][0]]),
                                "coloraxis.cmin": full_metric_ranges[SURFACE_METRICS_3D[0][0]][0],
                                "coloraxis.cmax": full_metric_ranges[SURFACE_METRICS_3D[0][0]][1],
                                "coloraxis2.cmin": full_metric_ranges[SURFACE_METRICS_3D[1][0]][0],
                                "coloraxis2.cmax": full_metric_ranges[SURFACE_METRICS_3D[1][0]][1],
                            }
                        ],
                    },
                ],
            }
        ],
        annotations=[
            *subplot_titles,
            {
                "text": (
                    "Top: piecewise-linear meshes through measured endpoints.<br>"
                    "Bottom: positive advantage means best 1p BPB minus best eligible 2p BPB; "
                    "values are selected discovery minima."
                ),
                "x": 0.5,
                "xref": "paper",
                "y": -0.205,
                "yref": "paper",
                "showarrow": False,
                "align": "center",
                "font": {"size": 13, "color": "#536A78"},
            },
        ],
    )
    gain_values = gain_rows["selected_gain_bpb"].to_numpy(dtype=float)
    gain_padding = max(0.2 * float(np.ptp(gain_values)), 0.0005)
    figure.update_xaxes(
        title="Coupled phase/LR onset",
        tickvals=gain_rows["requested_onset_fraction"],
        ticktext=[ARM_LABELS[arm] for arm in ARMS],
        range=[0.57, 0.93],
        gridcolor="#DCE5EA",
        domain=[0.06, 0.94],
        row=2,
        col=1,
    )
    figure.update_yaxes(
        title="2p advantage (BPB)",
        range=[min(0.0, float(gain_values.min()) - gain_padding), float(gain_values.max()) + 2.0 * gain_padding],
        tickformat="+.4f",
        gridcolor="#DCE5EA",
        zeroline=True,
        zerolinecolor="#17324D",
        zerolinewidth=1.5,
        domain=[0.05, 0.25],
        row=2,
        col=1,
    )
    return figure


def _surface_3d_plot(observations: pd.DataFrame, summary: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "endpoint_surfaces_3d.html"
    pio.write_html(
        _surface_3d_figure(observations, summary),
        path,
        include_plotlyjs=True,
        full_html=True,
        config=PLOT_CONFIG,
    )
    return path


def _gain_plot(summary: pd.DataFrame, deformation: pd.DataFrame) -> Path:
    colors = px.colors.sample_colorscale("RdYlGn_r", [0.15, 0.85])
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Raw-grid two-phase gain", "Matched-coordinate shift versus 0.80T"),
        horizontal_spacing=0.12,
    )
    figure.add_trace(
        go.Scatter(
            x=summary["requested_onset_fraction"],
            y=summary["selected_gain_bpb"],
            mode="lines+markers+text",
            text=summary["selected_untied_coordinate"],
            textposition="top center",
            line={"color": colors[0], "width": 3},
            marker={"size": 11},
            name="Selected raw-grid gain",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=deformation["arm_id"],
            y=deformation["mean_delta_bpb"],
            mode="markers",
            error_y={
                "type": "data",
                "array": deformation["q95_delta_bpb"] - deformation["mean_delta_bpb"],
                "arrayminus": deformation["mean_delta_bpb"] - deformation["q05_delta_bpb"],
            },
            marker={"size": 12, "color": colors[1]},
            name="Mean and 5-95% coordinate shift",
        ),
        row=1,
        col=2,
    )
    figure.add_hline(y=0.0, line={"color": "#17324D", "width": 1}, row="all", col="all")
    figure.update_layout(
        title="Coupled onset discovery summaries",
        height=600,
        margin={"l": 75, "r": 40, "t": 120, "b": 80},
        paper_bgcolor="#F8F3E8",
        plot_bgcolor="#F8F3E8",
        font={"family": "Avenir Next, sans-serif", "size": 15, "color": "#17324D"},
        showlegend=False,
    )
    figure.update_xaxes(gridcolor="#DCE5EA")
    figure.update_yaxes(title="BPB difference", gridcolor="#DCE5EA")
    path = OUTPUT_DIR / "selected_gain_and_deformation.html"
    pio.write_html(figure, path, include_plotlyjs=True, full_html=True, config=PLOT_CONFIG)
    return path


def _summary_table(frame: pd.DataFrame) -> str:
    lines = [
        "| coupled onset | tied coordinate | untied coordinate | tied BPB | untied BPB | "
        "primary gain | C4 gain | |fiber| |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.requested_onset_fraction:.2f}T | {row.selected_tied_coordinate} | "
            f"{row.selected_untied_coordinate} | {row.selected_tied_bpb:.6f} | "
            f"{row.selected_untied_bpb:.6f} | {row.selected_gain_bpb:+.6f} | "
            f"{row.selected_c4_gain_bpb:+.6f} | "
            f"{row.selected_absolute_normalized_fiber:.3f} |"
        )
    return "\n".join(lines)


def _c4_sensitivity_table(frame: pd.DataFrame) -> str:
    lines = [
        "| coupled onset | C4-safe untied coordinate | eligible rows | programming gain | C4 gain |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.requested_onset_fraction:.2f}T | {row.selected_untied_coordinate} | "
            f"{row.eligible_untied_coordinates} | {row.programming_gain_bpb:+.6f} | {row.c4_gain_bpb:+.6f} |"
        )
    return "\n".join(lines)


def write_outputs() -> None:
    """Collect exact endpoints and write the frozen discovery analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload, runs = load_design()
    observations = collect_endpoints(runs)
    selected = selected_policies(observations)
    summary = arm_summary(observations, selected)
    deformation = matched_coordinate_deformation(observations)
    c4_sensitivity = c4_noninferiority_sensitivity(observations, selected)
    null_gain_bias = expected_asymmetric_null_gain()

    observations_path = OUTPUT_DIR / "observations.csv"
    selected_path = OUTPUT_DIR / "selected_policies.csv"
    arm_path = OUTPUT_DIR / "arm_summary.csv"
    deformation_path = OUTPUT_DIR / "matched_coordinate_deformation.csv"
    c4_sensitivity_path = OUTPUT_DIR / "c4_noninferiority_sensitivity.csv"
    observations.drop(columns=["optimizer", "coordinate_sources"]).to_csv(observations_path, index=False)
    selected.to_csv(selected_path, index=False)
    summary.to_csv(arm_path, index=False)
    deformation.to_csv(deformation_path, index=False)
    c4_sensitivity.to_csv(c4_sensitivity_path, index=False)
    surface_plot = _surface_plot(observations, selected)
    surface_3d_plot = _surface_3d_plot(observations, summary)
    gain_plot = _gain_plot(summary, deformation)

    gains = summary.set_index("arm_id")["selected_gain_bpb"]
    ordered = bool(gains["coupled_0p60"] >= gains["coupled_0p80"] >= gains["coupled_0p90"])
    analysis = {
        "generated_at": datetime.now(UTC).isoformat(),
        "design_sha256": EXPECTED_DESIGN_SHA256,
        "iris_root": IRIS_ROOT,
        "checkpoint_root": CHECKPOINT_ROOT,
        "endpoint_step": EXPECTED_ENDPOINT_STEP,
        "complete_rows": len(observations),
        "coordinates_per_arm": EXPECTED_COORDINATES,
        "selected_gain_order_matches_hypothesis": ordered,
        "paired_gain_noise_sd_bpb": PAIRED_GAIN_NOISE_SD_BPB,
        "approximate_asymmetric_null_gain_bpb": null_gain_bias,
        "arm_summary": summary.to_dict(orient="records"),
        "matched_coordinate_deformation": deformation.to_dict(orient="records"),
        "posthoc_c4_noninferiority_sensitivity": c4_sensitivity.to_dict(orient="records"),
        "inference_status": "discovery_only_pending_fresh_eight_seed_confirmation",
        "confirmation_inventory_if_all_selected_policies_are_unique": 3 * 2 * 8,
        "claim_boundary": payload["claim_boundary"],
        "limitations": [
            "All discovery rows use one common trainer/data seed.",
            "The 125 coordinates are design points, not stochastic replicates.",
            "Each arm's minima are selected on the same surface used to report the raw gain.",
            (
                "The untied minimum searches 94 eligible coordinates versus 26 tied coordinates, "
                "creating asymmetric winner bias."
            ),
            (
                "The Gaussian best-of-bank bias calculation is only a scale diagnostic; "
                "designed coordinates are correlated and nonexchangeable."
            ),
            "The intervention couples phase-2 duration and LR-decay duration.",
        ],
    }
    analysis_path = OUTPUT_DIR / "analysis_summary.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")

    gain_values = ", ".join(
        f"{row.requested_onset_fraction:.2f}T: {row.selected_gain_bpb:+.6f} BPB"
        for row in summary.itertuples(index=False)
    )
    best_arm = summary.loc[summary["selected_untied_bpb"].idxmin()]
    fixed_pair_values = ", ".join(
        f"{row.requested_onset_fraction:.2f}T: {row.historical_c109_minus_c020_gain_bpb:+.6f} BPB"
        for row in summary.itertuples(index=False)
    )
    selection_inflation_values = ", ".join(
        f"{row.requested_onset_fraction:.2f}T: {row.selection_inflation_vs_historical_pair_bpb:+.6f} BPB"
        for row in summary.itertuples(index=False)
    )
    summary_by_arm = summary.set_index("arm_id")
    fixed_080_minus_060_sd = (
        summary_by_arm.loc["coupled_0p80", "historical_c109_minus_c020_gain_bpb"]
        - summary_by_arm.loc["coupled_0p60", "historical_c109_minus_c020_gain_bpb"]
    ) / (math.sqrt(2.0) * PAIRED_GAIN_NOISE_SD_BPB)
    fixed_080_minus_090_sd = (
        summary_by_arm.loc["coupled_0p80", "historical_c109_minus_c020_gain_bpb"]
        - summary_by_arm.loc["coupled_0p90", "historical_c109_minus_c020_gain_bpb"]
    ) / (math.sqrt(2.0) * PAIRED_GAIN_NOISE_SD_BPB)
    reference_deformation = deformation.set_index("arm_id")
    order_text = "matches" if ordered else "does not match"
    report = f"""# StarCoder WSD80 coupled phase/LR-onset dense surfaces

## Result

All {EXPECTED_RUNS} exact step-{EXPECTED_ENDPOINT_STEP} endpoints are complete. The discovery minima give
{gain_values}. The raw ordering therefore **{order_text}** the preregistered hypothesis that an earlier coupled
onset produces a larger two-phase advantage.

More specifically, moving the coupled onset earlier from 0.80T to 0.60T does not increase the selected programming
gain; it decreases it. The 0.80T and 0.90T selected gains differ by only
`{abs(gains['coupled_0p80'] - gains['coupled_0p90']):.6f}` BPB and are unresolved at discovery precision.

The best absolute programming endpoint is `{best_arm.selected_untied_bpb:.6f}` at the
{best_arm.requested_onset_fraction:.2f}T onset. The historical fixed `c109` tied versus `c020` untied comparison
also lacks the proposed order ({fixed_pair_values}). This fixed pair was chosen at 0.80T and transported to the
other surfaces, so it is descriptive rather than an independent confirmation. Its 0.80T-minus-0.60T contrast is
about `{fixed_080_minus_060_sd:.1f}` paired-noise SDs in the adverse direction, while 0.80T minus 0.90T is only
`{fixed_080_minus_090_sd:.1f}` SDs.

These are selected, one-seed discovery gains, not confirmatory effect estimates. The next inferential gate is the
frozen eight-seed comparison of each selected tied and untied policy.

- [Open the interactive 3D endpoint surfaces](endpoint_surfaces_3d.html). The slider changes the coupled phase/LR
  onset for both Programming Languages and C4. Each translucent surface is a piecewise-linear Delaunay mesh, not a
  fitted response model; the overlaid markers are the 125 measured endpoints. The range toggle hides edge-policy
  spikes above the strict-interior response envelope by default, rescales the colors with the focused range, and can
  restore the complete vertical and color ranges. Diamond and cross markers identify each metric's best observed 1p
  and preregistered-eligible 2p policies.
- [Open the three 2D endpoint surfaces](endpoint_surfaces.html).
- [Open the selected-gain and matched-deformation summary](selected_gain_and_deformation.html).

## Selected policies

{_summary_table(summary)}

The primary selection ignores C4. The 0.60T policy improves both reported metrics, the 0.80T policy trades
`0.004406` C4 BPB for programming gain, and the 0.90T policy trades `0.120262` C4 BPB. A post-hoc sensitivity
screen that requires the untied policy to be no worse than its selected tied baseline on C4 gives:

{_c4_sensitivity_table(c4_sensitivity)}

This sensitivity screen was not preregistered and does not establish a joint-objective winner. It does show that
the 0.90T programming optimum is not decision-relevant under even a zero-tolerance C4 constraint.

## What changes across onset arms

The common 125 coordinate IDs preserve aggregate StarCoder exposure and normalized fiber position, so matched
differences measure deformation of the endpoint surface rather than a change in the sampled coordinate bank.
The deformation table reports each arm relative to the 0.80T reference. Rank correlations and centered spread
separate a mostly global level shift from policy-specific reshaping. The programming-BPB rank correlation is
{reference_deformation.loc['coupled_0p60', 'spearman_with_reference']:.3f} for 0.60T versus 0.80T and
{reference_deformation.loc['coupled_0p90', 'spearman_with_reference']:.3f} for 0.90T versus 0.80T. Thus the 0.60T
intervention substantially reshapes policy ranking rather than simply translating the surface.

The selected untied minima are shallow: the best-to-second-best gaps are
{summary.iloc[0]['selected_untied_margin_to_second_bpb']:.6f},
{summary.iloc[1]['selected_untied_margin_to_second_bpb']:.6f}, and
{summary.iloc[2]['selected_untied_margin_to_second_bpb']:.6f} BPB. Their exact coordinate identities should not be
treated as stable before replication. The C4-gain column uses the same sign convention as the primary gain
(tied minus untied); the late-onset programming optima show a broad-language tradeoff, especially at 0.90T.

Selection inflation relative to the transported historical pair is {selection_inflation_values}. It is largest at
0.60T, yet 0.60T still has the smallest selected gain. That makes the adverse 0.60T ordering harder to attribute
solely to selected-minimum inflation, although it does not de-bias the reported gains.

The treatment does not isolate optimizer schedule from phase duration: moving the onset changes both the amount of
phase-2 training and the amount of learning-rate mass available in phase 2. The earlier fixed-boundary LR-onset
experiment is the orthogonal LR-only control.

## Limitations and next gate

All discovery rows share one trainer/data seed, and raw minima have winner's-curse bias. That bias is asymmetric:
each untied minimum searches 94 eligible coordinates while each tied minimum searches 26. The positive raw gains
therefore cannot establish a two-phase advantage. Under an intentionally simplified iid equal-mean Gaussian null,
the unequal bank sizes alone imply about `{null_gain_bias:.6f}` BPB of positive gain. The real design points are
correlated and nonexchangeable, so this is a scale diagnostic rather than a correction. Surface coordinates are not
independent stochastic replicates, so no p-value or confidence interval is attached to the discovery ordering.

The preregistered confirmation requires 48 fresh runs: three arms by tied versus untied by eight reserved seeds.
The scientifically clean default is to preserve that gate. If decision relevance and compute dominate, a formal
amendment could retain only 0.60T and 0.80T for 32 runs; silently dropping 0.90T after seeing the data would not be
defensible.

## Provenance

- Iris root: `{IRIS_ROOT}`
- Exact endpoint: step `{EXPECTED_ENDPOINT_STEP}`
- Checkpoint root: `{CHECKPOINT_ROOT}`
- Design SHA-256: `{EXPECTED_DESIGN_SHA256}`
"""
    report_path = OUTPUT_DIR / "results.md"
    report_path.write_text(report)

    output_paths = (
        observations_path,
        selected_path,
        arm_path,
        deformation_path,
        c4_sensitivity_path,
        analysis_path,
        report_path,
        surface_plot,
        surface_3d_plot,
        gain_plot,
    )
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output_paths}
    print(json.dumps({"outputs": hashes, "analysis": analysis}, indent=2))


if __name__ == "__main__":
    write_outputs()
