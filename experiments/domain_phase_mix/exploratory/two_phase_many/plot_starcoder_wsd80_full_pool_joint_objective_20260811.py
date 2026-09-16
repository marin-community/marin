# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
#   "tabulate",
# ]
# ///

"""Inspect full-pool WSD80 surfaces under joint code and broad-web objectives."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.interpolate import LinearNDInterpolator

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
SOURCE_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_dense_support_empirical_optimum_confirmation_design_20260811"
DEFAULT_COVERAGE = SOURCE_DIR / "coverage_observations.csv"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_full_pool_joint_objective_20260811"

GCS_ROOT = (
    "gs://marin-us-central1/checkpoints/pinlin_calvin_xu/data_mixture/" "starcoder_wsd80_dense_support_surfaces_20260808"
)
CHECKPOINT_VERSION = "2026.07.11"
EXPECTED_CELLS = 4
EXPECTED_COORDINATES = 125
EXPECTED_ROWS = EXPECTED_CELLS * EXPECTED_COORDINATES
PHASE_0_FRACTION = 0.8
PHASE_1_FRACTION = 0.2

METRICS = {
    "code": "eval/paloma/dolma_100_programing_languages-llama3/bpb",
    "c4_100": "eval/paloma/c4_100_domains-llama3/bpb",
    "c4_en": "eval/paloma/c4_en-llama3/bpb",
    "refinedweb": "eval/paloma/falcon-refinedweb-llama3/bpb",
    "dolma15": "eval/paloma/dolma-v1_5-llama3/bpb",
    "paloma_macro": "eval/paloma/macro_bpb",
    "uncheatable": "eval/uncheatable_eval/bpb",
}
WEB_LABELS = {
    "c4_100": "C4-100 Domains",
    "c4_en": "C4 English",
    "refinedweb": "Falcon RefinedWeb",
    "dolma15": "Dolma 1.5",
    "paloma_macro": "Paloma macro",
    "uncheatable": "Uncheatable",
}
SENSITIVITY_WEB_METRICS = ("c4_100", "c4_en", "refinedweb", "dolma15", "paloma_macro", "uncheatable")
PRIMARY_WEB_METRIC = "c4_100"
PRIMARY_LAMBDA = 0.5
LAMBDA_GRID = np.linspace(0.0, 1.0, 21)
DISPLAY_GRID_SIZE = 101
DISPLAY_DELTA_CAP = 0.05

PAPER_BACKGROUND = "#F7F3E8"
PLOT_BACKGROUND = "#FFFDF8"
PAPER_TEXT = "#17324D"
GRID_COLOR = "#D8D1C2"
TIED_COLOR = "#F2B134"
UNTIED_COLOR = "#D95F3B"
OBSERVED_COLOR = "#17324D"
HORIZON_COLORS = ("#2D6A4F", "#40916C", "#D97706", "#B91C1C")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--refresh-metrics", action="store_true")
    parser.add_argument("--fetch-workers", type=int, default=32)
    return parser.parse_args()


def _metric_uri(run_name: str) -> str:
    return f"{GCS_ROOT}/{run_name}/{CHECKPOINT_VERSION}/checkpoints/eval_metrics.jsonl"


def _fetch_metrics(run_name: str) -> dict[str, object]:
    uri = _metric_uri(run_name)
    result = subprocess.run(
        ["gcloud", "storage", "cat", uri],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to read {uri}: {result.stderr.strip()}")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No evaluation rows in {uri}")
    payload = json.loads(lines[-1])
    values = {name: payload.get(key) for name, key in METRICS.items()}
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(f"{run_name} is missing metrics: {missing}")
    return {"run_name": run_name, "metric_uri": uri, **values}


def _load_coverage(path: Path) -> pd.DataFrame:
    coverage = pd.read_csv(path)
    required = {
        "cell_id",
        "coordinate_id",
        "run_name",
        "support_id",
        "phase_0_starcoder",
        "phase_1_starcoder",
        "aggregate_starcoder",
        "materialized_tokens",
        "bpb",
    }
    missing = required - set(coverage.columns)
    if missing:
        raise ValueError(f"Coverage table is missing fields: {sorted(missing)}")
    coverage = coverage.loc[coverage["support_id"].eq("full")].copy()
    if len(coverage) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} full-pool rows, got {len(coverage)}")
    counts = coverage.groupby("cell_id").size()
    if len(counts) != EXPECTED_CELLS or not counts.eq(EXPECTED_COORDINATES).all():
        raise ValueError(f"Expected {EXPECTED_COORDINATES} coordinates in each of {EXPECTED_CELLS} cells")
    if coverage["run_name"].duplicated().any():
        raise ValueError("Full-pool run names must be unique")
    return coverage


def _materialize_metrics(
    coverage: pd.DataFrame,
    path: Path,
    *,
    refresh: bool,
    workers: int,
) -> pd.DataFrame:
    expected_runs = set(coverage["run_name"])
    if path.exists() and not refresh:
        metrics = pd.read_csv(path)
        if set(metrics["run_name"]) == expected_runs and not metrics[list(METRICS)].isna().any().any():
            return metrics

    rows: list[dict[str, object]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_metrics, run_name): run_name for run_name in sorted(expected_runs)}
        for future in as_completed(futures):
            run_name = futures[future]
            try:
                rows.append(future.result())
            except Exception as error:
                errors.append(f"{run_name}: {error}")
    if errors:
        raise RuntimeError("Metric materialization failed:\n" + "\n".join(errors[:20]))

    metrics = pd.DataFrame(rows).sort_values("run_name").reset_index(drop=True)
    if len(metrics) != EXPECTED_ROWS or set(metrics["run_name"]) != expected_runs:
        raise ValueError("Materialized metric rows do not match the full-pool design")
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(path, index=False)
    return metrics


def load_observations(
    coverage_path: Path,
    output_dir: Path,
    *,
    refresh: bool,
    workers: int,
) -> pd.DataFrame:
    """Return one complete multi-metric row for every full-pool coordinate."""
    coverage = _load_coverage(coverage_path)
    metrics_path = output_dir / "full_pool_metric_observations.csv"
    metrics = _materialize_metrics(coverage, metrics_path, refresh=refresh, workers=workers)
    observations = coverage.merge(metrics, on="run_name", validate="one_to_one")
    if not np.allclose(observations["bpb"], observations["code"], atol=1e-7, rtol=0.0):
        raise ValueError("Persisted Programming Languages BPB disagrees with the coverage table")
    observations["policy_class"] = np.where(
        np.isclose(observations["phase_0_starcoder"], observations["phase_1_starcoder"]),
        "tied",
        "untied",
    )
    expected_aggregate = (
        PHASE_0_FRACTION * observations["phase_0_starcoder"] + PHASE_1_FRACTION * observations["phase_1_starcoder"]
    )
    if not np.allclose(expected_aggregate, observations["aggregate_starcoder"], atol=1e-6, rtol=0.0):
        raise ValueError("Aggregate StarCoder weights do not match the 80/20 phase fractions")
    observations["materialized_tokens_b"] = observations["materialized_tokens"] / 1e9
    observations["wandb_url"] = observations["run_name"].map(
        lambda run_name: f"https://wandb.ai/marin-community/marin/runs/{run_name}"
    )
    return observations.sort_values(["materialized_tokens", "coordinate_id"]).reset_index(drop=True)


def _objective(group: pd.DataFrame, web_metric: str, lambda_code: float) -> pd.Series:
    return lambda_code * group["code"] + (1.0 - lambda_code) * group[web_metric]


def _robust_objective(group: pd.DataFrame, web_metric: str, lambda_code: float) -> pd.Series:
    standardized: dict[str, pd.Series] = {}
    for metric in ("code", web_metric):
        q25, q75 = np.quantile(group[metric], [0.25, 0.75])
        scale = float(q75 - q25)
        if scale <= 0.0:
            raise ValueError(f"{metric} has non-positive IQR")
        standardized[metric] = (group[metric] - float(np.median(group[metric]))) / scale
    return lambda_code * standardized["code"] + (1.0 - lambda_code) * standardized[web_metric]


def _is_dominated_by_tied(group: pd.DataFrame, untied_index: int, web_metric: str) -> bool:
    row = group.loc[untied_index]
    tied = group.loc[group["policy_class"].eq("tied")]
    weakly_better = tied["code"].le(row["code"]) & tied[web_metric].le(row[web_metric])
    strictly_better = tied["code"].lt(row["code"]) | tied[web_metric].lt(row[web_metric])
    return bool((weakly_better & strictly_better).any())


def summarize_objectives(observations: pd.DataFrame) -> pd.DataFrame:
    """Compute raw observed tied and off-diagonal minima for each objective."""
    rows: list[dict[str, object]] = []
    for web_metric in SENSITIVITY_WEB_METRICS:
        for cell_id, group in observations.groupby("cell_id", sort=False):
            tied = group["policy_class"].eq("tied")
            for lambda_code in LAMBDA_GRID:
                objective = _objective(group, web_metric, float(lambda_code))
                robust_objective = _robust_objective(group, web_metric, float(lambda_code))
                tied_index = int(objective.loc[tied].idxmin())
                untied_index = int(objective.loc[~tied].idxmin())
                tied_row = group.loc[tied_index]
                untied_row = group.loc[untied_index]
                raw_gain = float(objective.loc[tied_index] - objective.loc[untied_index])
                robust_gain = float(robust_objective.loc[tied_index] - robust_objective.loc[untied_index])
                rows.append(
                    {
                        "cell_id": cell_id,
                        "materialized_tokens_b": tied_row["materialized_tokens_b"],
                        "web_metric": web_metric,
                        "lambda_code": float(lambda_code),
                        "raw_off_diagonal_gain_bpb": raw_gain,
                        "raw_global_two_phase_gain_bpb": max(0.0, raw_gain),
                        "robust_standardized_gain": robust_gain,
                        "tied_coordinate_id": tied_row["coordinate_id"],
                        "tied_p": tied_row["phase_0_starcoder"],
                        "tied_objective_bpb": objective.loc[tied_index],
                        "untied_coordinate_id": untied_row["coordinate_id"],
                        "untied_p0": untied_row["phase_0_starcoder"],
                        "untied_p1": untied_row["phase_1_starcoder"],
                        "untied_aggregate": untied_row["aggregate_starcoder"],
                        "untied_objective_bpb": objective.loc[untied_index],
                        "tied_minus_untied_code_bpb": tied_row["code"] - untied_row["code"],
                        "tied_minus_untied_web_bpb": tied_row[web_metric] - untied_row[web_metric],
                        "untied_dominated_by_sampled_tied": _is_dominated_by_tied(group, untied_index, web_metric),
                    }
                )
    return pd.DataFrame(rows).sort_values(["web_metric", "materialized_tokens_b", "lambda_code"])


def _linear_surface(group: pd.DataFrame, values: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, DISPLAY_GRID_SIZE)
    p0, p1 = np.meshgrid(axis, axis, indexing="xy")
    coordinates = group[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
    interpolator = LinearNDInterpolator(coordinates, values.to_numpy(dtype=float), fill_value=np.nan)
    prediction = np.asarray(interpolator(p0, p1), dtype=float)
    return p0, p1, prediction


def _pareto_front(group: pd.DataFrame, web_metric: str) -> pd.DataFrame:
    ordered = group.sort_values(["code", web_metric])
    keep: list[int] = []
    best_web = float("inf")
    for index, row in ordered.iterrows():
        if float(row[web_metric]) < best_web:
            keep.append(index)
            best_web = float(row[web_metric])
    return group.loc[keep].sort_values("code")


def build_surface_figure(observations: pd.DataFrame, summary: pd.DataFrame) -> go.Figure:
    """Build four raw equal-weight joint-objective surfaces and Pareto views."""
    horizons = observations[["cell_id", "materialized_tokens_b"]].drop_duplicates().sort_values("materialized_tokens_b")
    titles: list[str] = []
    for row in horizons.itertuples(index=False):
        titles.extend(
            [
                f"D={row.materialized_tokens_b:.2f}B · 50/50 average BPB surface",
                f"D={row.materialized_tokens_b:.2f}B · code/web frontier zoom",
            ]
        )
    figure = make_subplots(
        rows=EXPECTED_CELLS,
        cols=2,
        subplot_titles=titles,
        column_widths=[0.58, 0.42],
        horizontal_spacing=0.09,
        vertical_spacing=0.075,
    )

    primary_summary = summary.loc[
        summary["web_metric"].eq(PRIMARY_WEB_METRIC) & np.isclose(summary["lambda_code"], PRIMARY_LAMBDA)
    ].set_index("cell_id")
    for row_index, cell in enumerate(horizons.itertuples(index=False), start=1):
        group = observations.loc[observations["cell_id"].eq(cell.cell_id)].copy()
        objective = _objective(group, PRIMARY_WEB_METRIC, PRIMARY_LAMBDA)
        surface_p0, surface_p1, surface = _linear_surface(group, objective)
        minimum = float(objective.min())
        display_delta = np.minimum(surface - minimum, DISPLAY_DELTA_CAP)
        observed_delta = np.minimum(objective - minimum, DISPLAY_DELTA_CAP)
        selected = primary_summary.loc[cell.cell_id]

        custom = np.column_stack(
            (
                group["coordinate_id"],
                group["run_name"],
                group["aggregate_starcoder"],
                group["code"],
                group[PRIMARY_WEB_METRIC],
                objective,
            )
        )
        figure.add_trace(
            go.Contour(
                x=surface_p0[0],
                y=surface_p1[:, 0],
                z=display_delta,
                coloraxis="coloraxis",
                contours={"start": 0.0, "end": DISPLAY_DELTA_CAP, "size": 0.005, "coloring": "heatmap"},
                customdata=surface,
                hovertemplate=(
                    "p0=%{x:.3f}<br>p1=%{y:.3f}<br>" "Piecewise-linear 50/50 BPB=%{customdata:.6f}<extra></extra>"
                ),
                showscale=False,
                showlegend=False,
            ),
            row=row_index,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=group["phase_0_starcoder"],
                y=group["phase_1_starcoder"],
                mode="markers",
                name="Observed coordinate",
                marker={
                    "size": 6,
                    "color": observed_delta,
                    "coloraxis": "coloraxis",
                    "line": {"color": OBSERVED_COLOR, "width": 0.7},
                },
                customdata=custom,
                hovertemplate=(
                    "<b>%{customdata[0]}</b> · %{customdata[1]}<br>p0=%{x:.4f}; p1=%{y:.4f}<br>"
                    "aggregate=%{customdata[2]:.4f}<br>Programming=%{customdata[3]:.6f}<br>"
                    "C4-100=%{customdata[4]:.6f}<br>50/50 average=%{customdata[5]:.6f}<extra></extra>"
                ),
                showlegend=row_index == 1,
                legendgroup="observed",
            ),
            row=row_index,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                mode="lines",
                name="Tied policies",
                line={"color": TIED_COLOR, "width": 2.2, "dash": "dash"},
                hoverinfo="skip",
                showlegend=row_index == 1,
                legendgroup="tied-line",
            ),
            row=row_index,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=[selected["tied_p"]],
                y=[selected["tied_p"]],
                mode="markers",
                name="Best sampled tied",
                marker={"size": 16, "symbol": "x", "color": TIED_COLOR, "line": {"width": 3}},
                hovertemplate=(
                    f"Best sampled tied<br>p={selected['tied_p']:.4f}<br>"
                    f"50/50 average={selected['tied_objective_bpb']:.6f} BPB<extra></extra>"
                ),
                showlegend=row_index == 1,
                legendgroup="tied-min",
            ),
            row=row_index,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=[selected["untied_p0"]],
                y=[selected["untied_p1"]],
                mode="markers",
                name="Best sampled off-diagonal",
                marker={
                    "size": 18,
                    "symbol": "star",
                    "color": UNTIED_COLOR,
                    "line": {"color": "white", "width": 1.2},
                },
                hovertemplate=(
                    f"Best sampled off-diagonal<br>p0={selected['untied_p0']:.4f}; "
                    f"p1={selected['untied_p1']:.4f}<br>aggregate={selected['untied_aggregate']:.4f}<br>"
                    f"50/50 average={selected['untied_objective_bpb']:.6f} BPB<br>"
                    f"raw global two-phase gain={selected['raw_global_two_phase_gain_bpb']:+.6f} BPB"
                    "<extra></extra>"
                ),
                showlegend=row_index == 1,
                legendgroup="untied-min",
            ),
            row=row_index,
            col=1,
        )

        for policy_class, symbol, color in (
            ("tied", "circle", TIED_COLOR),
            ("untied", "circle-open", OBSERVED_COLOR),
        ):
            policy = group.loc[group["policy_class"].eq(policy_class)]
            figure.add_trace(
                go.Scatter(
                    x=policy["code"],
                    y=policy[PRIMARY_WEB_METRIC],
                    mode="markers",
                    name=f"{policy_class.title()} observations",
                    marker={"size": 7, "symbol": symbol, "color": color, "opacity": 0.75},
                    customdata=np.column_stack(
                        (policy["coordinate_id"], policy["phase_0_starcoder"], policy["phase_1_starcoder"])
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>Programming=%{x:.6f}<br>C4-100=%{y:.6f}<br>"
                        "p0=%{customdata[1]:.4f}; p1=%{customdata[2]:.4f}<extra></extra>"
                    ),
                    showlegend=row_index == 1,
                    legendgroup=f"pareto-{policy_class}",
                ),
                row=row_index,
                col=2,
            )
        all_front = _pareto_front(group, PRIMARY_WEB_METRIC)
        tied_front = _pareto_front(group.loc[group["policy_class"].eq("tied")], PRIMARY_WEB_METRIC)
        for front, name, color, dash in (
            (all_front, "All-policy Pareto frontier", UNTIED_COLOR, "solid"),
            (tied_front, "Sampled tied Pareto frontier", TIED_COLOR, "dash"),
        ):
            figure.add_trace(
                go.Scatter(
                    x=front["code"],
                    y=front[PRIMARY_WEB_METRIC],
                    mode="lines",
                    name=name,
                    line={"color": color, "width": 2.4, "dash": dash},
                    hoverinfo="skip",
                    showlegend=row_index == 1,
                    legendgroup=name,
                ),
                row=row_index,
                col=2,
            )

        figure.update_xaxes(title_text="Phase 0 StarCoder weight", range=[-0.02, 1.02], row=row_index, col=1)
        figure.update_yaxes(title_text="Phase 1 StarCoder weight", range=[-0.02, 1.02], row=row_index, col=1)
        code_min = float(group["code"].min())
        code_limit = float(group["code"].quantile(0.8))
        web_min = float(group[PRIMARY_WEB_METRIC].min())
        web_limit = float(group[PRIMARY_WEB_METRIC].quantile(0.8))
        code_margin = 0.04 * (code_limit - code_min)
        web_margin = 0.04 * (web_limit - web_min)
        figure.update_xaxes(
            title_text="Programming Languages BPB",
            range=[code_min - code_margin, code_limit + code_margin],
            row=row_index,
            col=2,
        )
        figure.update_yaxes(
            title_text="C4-100 Domains BPB",
            range=[web_min - web_margin, web_limit + web_margin],
            row=row_index,
            col=2,
        )

    figure.update_layout(
        width=1500,
        height=2350,
        paper_bgcolor=PAPER_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        font={"family": "Avenir Next, Helvetica Neue, sans-serif", "color": PAPER_TEXT, "size": 14},
        margin={"l": 95, "r": 70, "t": 115, "b": 80},
        coloraxis={
            "colorscale": "RdYlGn_r",
            "cmin": 0.0,
            "cmax": DISPLAY_DELTA_CAP,
            "colorbar": {"title": "50/50 BPB<br>above raw min", "thickness": 18, "len": 0.3, "y": 0.98},
        },
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.01,
            "yanchor": "bottom",
            "bgcolor": "rgba(255,253,248,0.94)",
            "bordercolor": GRID_COLOR,
            "borderwidth": 1,
        },
        hoverlabel={"bgcolor": PLOT_BACKGROUND, "font": {"color": PAPER_TEXT, "size": 13}},
    )
    figure.update_xaxes(gridcolor=GRID_COLOR, zeroline=False)
    figure.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)
    return figure


def build_lambda_figure(summary: pd.DataFrame) -> go.Figure:
    """Plot raw global two-phase gain across the code-versus-web weight sweep."""
    primary = summary.loc[summary["web_metric"].eq(PRIMARY_WEB_METRIC)]
    figure = go.Figure()
    for color, (_cell_id, group) in zip(HORIZON_COLORS, primary.groupby("cell_id", sort=False), strict=True):
        group = group.sort_values("lambda_code")
        figure.add_trace(
            go.Scatter(
                x=group["lambda_code"],
                y=group["raw_off_diagonal_gain_bpb"],
                mode="lines+markers",
                name=f"D={group['materialized_tokens_b'].iloc[0]:.2f}B",
                line={"color": color, "width": 3},
                marker={"size": 8, "color": color},
                customdata=np.column_stack(
                    (
                        group["tied_coordinate_id"],
                        group["untied_coordinate_id"],
                        group["untied_p0"],
                        group["untied_p1"],
                        group["tied_minus_untied_code_bpb"],
                        group["tied_minus_untied_web_bpb"],
                    )
                ),
                hovertemplate=(
                    "lambda(code)=%{x:.2f}<br>off-diagonal minus tied gain=%{y:+.6f} BPB<br>"
                    "tied=%{customdata[0]}<br>off-diagonal=%{customdata[1]} "
                    "(%{customdata[2]:.4f}, %{customdata[3]:.4f})<br>"
                    "Programming contribution: tied-untied=%{customdata[4]:+.6f} BPB<br>"
                    "C4 contribution: tied-untied=%{customdata[5]:+.6f} BPB<extra></extra>"
                ),
            )
        )
    figure.add_hline(y=0.0, line={"color": PAPER_TEXT, "width": 2})
    figure.add_vline(x=PRIMARY_LAMBDA, line={"color": "#777777", "width": 1.5, "dash": "dot"})
    figure.add_annotation(
        x=PRIMARY_LAMBDA,
        y=1.0,
        yref="paper",
        text="equal BPB weight",
        showarrow=False,
        yshift=12,
        font={"color": "#555555"},
    )
    figure.update_layout(
        title={
            "text": (
                "<b>Raw sampled off-diagonal advantage depends on the evaluation objective</b><br>"
                "<sup>J_lambda = lambda Programming Languages BPB + (1-lambda) C4-100 Domains BPB · "
                "positive is a sampled global two-phase gain</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        width=1450,
        height=700,
        paper_bgcolor=PAPER_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        font={"family": "Avenir Next, Helvetica Neue, sans-serif", "color": PAPER_TEXT, "size": 15},
        margin={"l": 100, "r": 55, "t": 145, "b": 90},
        xaxis={
            "title": "Programming Languages weight lambda (0 = C4 only; 1 = Programming only)",
            "range": [-0.02, 1.02],
            "gridcolor": GRID_COLOR,
        },
        yaxis={
            "title": "Best sampled tied minus best sampled off-diagonal objective (BPB; higher is better)",
            "gridcolor": GRID_COLOR,
            "zeroline": False,
        },
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.08,
            "bgcolor": "rgba(255,253,248,0.95)",
            "bordercolor": GRID_COLOR,
            "borderwidth": 1,
        },
        hoverlabel={"bgcolor": PLOT_BACKGROUND, "font": {"color": PAPER_TEXT, "size": 13}},
    )
    return figure


def _sensitivity_table(summary: pd.DataFrame) -> pd.DataFrame:
    equal_weight = summary.loc[np.isclose(summary["lambda_code"], PRIMARY_LAMBDA)].copy()
    grouped = equal_weight.groupby("web_metric")["raw_off_diagonal_gain_bpb"]
    result = grouped.agg(
        positive_horizons=lambda values: int((values > 0.0).sum()),
        mean_gain_bpb="mean",
        min_gain_bpb="min",
        max_gain_bpb="max",
    ).reset_index()
    result["web_target"] = result["web_metric"].map(WEB_LABELS)
    return result[["web_metric", "web_target", "positive_horizons", "mean_gain_bpb", "min_gain_bpb", "max_gain_bpb"]]


def _primary_table(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "materialized_tokens_b",
        "raw_off_diagonal_gain_bpb",
        "raw_global_two_phase_gain_bpb",
        "tied_coordinate_id",
        "tied_p",
        "untied_coordinate_id",
        "untied_p0",
        "untied_p1",
        "untied_aggregate",
        "tied_minus_untied_code_bpb",
        "tied_minus_untied_web_bpb",
        "untied_dominated_by_sampled_tied",
    ]
    return summary.loc[
        summary["web_metric"].eq(PRIMARY_WEB_METRIC) & np.isclose(summary["lambda_code"], PRIMARY_LAMBDA),
        columns,
    ].sort_values("materialized_tokens_b")


def _format_table(frame: pd.DataFrame) -> str:
    formatted = frame.copy()
    for column in formatted.select_dtypes(include=[np.number]).columns:
        formatted[column] = formatted[column].map(lambda value: f"{float(value):.6f}")
    return formatted.to_html(index=False, classes="data-table", border=0, escape=True)


def write_report(
    output_dir: Path,
    surface_figure: go.Figure,
    lambda_figure: go.Figure,
    primary: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    surface_html = pio.to_html(
        surface_figure,
        full_html=False,
        include_plotlyjs="inline",
        config={"responsive": True, "displaylogo": False, "toImageButtonOptions": {"scale": 4}},
    )
    lambda_html = pio.to_html(
        lambda_figure,
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True, "displaylogo": False, "toImageButtonOptions": {"scale": 4}},
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WSD80 full-pool code/web objective</title>
<style>
:root {{ --paper: {PAPER_BACKGROUND}; --ink: {PAPER_TEXT}; --card: {PLOT_BACKGROUND}; --line: {GRID_COLOR}; }}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: "Avenir Next", "Helvetica Neue", sans-serif;
}}
main {{ max-width: 1560px; margin: 0 auto; padding: 42px 30px 80px; }}
h1, h2 {{ font-family: Georgia, "Times New Roman", serif; }}
h1 {{ font-size: clamp(34px, 5vw, 62px); margin: 0 0 12px; }}
h2 {{ font-size: 32px; margin: 0 0 14px; }}
.lede {{ max-width: 1100px; font-size: 19px; line-height: 1.55; color: #506272; }}
.cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; margin: 30px 0; }}
.card {{ background: var(--card); border: 1px solid var(--line); padding: 22px; }}
.card b {{ display: block; margin-bottom: 8px; text-transform: uppercase; letter-spacing: .08em; font-size: 13px; }}
.card p {{ margin: 0; line-height: 1.5; }}
.plot-card, .table-card {{ background: var(--card); border: 1px solid var(--line); margin: 24px 0; overflow-x: auto; }}
.table-card {{ padding: 26px; }}
.data-table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
.data-table th, .data-table td {{
  border-bottom: 1px solid var(--line);
  padding: 9px 10px;
  text-align: right;
  white-space: nowrap;
}}
.data-table th:first-child,
.data-table td:first-child,
.data-table th:nth-child(2),
.data-table td:nth-child(2) {{ text-align: left; }}
.warning {{
  border-left: 5px solid #D97706;
  background: #FFF7E6;
  padding: 18px 22px;
  line-height: 1.55;
  margin: 24px 0;
}}
code {{ background: #ECE6D8; padding: 2px 5px; }}
@media (max-width: 900px) {{ .cards {{ grid-template-columns: 1fr; }} main {{ padding: 24px 12px 60px; }} }}
</style>
</head>
<body><main>
<h1>Full-pool code/web policy surfaces</h1>
<p class="lede">
  The no-forced-replay StarCoder WSD80 panel showed no sampled global two-phase advantage on Programming
  Languages BPB alone. This diagnostic asks a different, preregisterable question: does a joint code and
  broad-web objective expose an off-diagonal policy when both datasets matter?
</p>
<div class="cards">
  <div class="card">
    <b>Primary objective</b>
    <p><strong>50% Programming Languages BPB + 50% C4-100 Domains BPB.</strong> Both terms are in BPB, so the
    primary average needs no fitted calibration.</p>
  </div>
  <div class="card">
    <b>What is measured</b>
    <p>Every optimum is selected from the same 125 actually trained coordinates per horizon: 26 tied and 99
    off-diagonal.</p>
  </div>
  <div class="card">
    <b>What is not proved</b>
    <p>C4 is a broad-web proxy, not a direct Nemotron-CC target. A code/web off-diagonal optimum is compatible
    with temporal conflict, but does not by itself identify gradient conflict causally.</p>
  </div>
</div>
<div class="warning">
  <strong>Selection caveat.</strong> Off-diagonal policies outnumber tied policies 99 to 26, and each coordinate
  currently has one discovery seed. Small positive gains can therefore arise from unequal search multiplicity
  and noise. Piecewise-linear contours below are visual guides only; they are never used to select or score an
  optimum.
</div>
<section><h2>Equal-weight joint objective</h2><div class="plot-card">{surface_html}</div></section>
<section><h2>Objective-weight sensitivity</h2><div class="plot-card">{lambda_html}</div></section>
<section class="table-card"><h2>Primary raw minima</h2>{_format_table(primary)}</section>
<section class="table-card"><h2>Broad-web proxy sensitivity at 50/50 weight</h2>{_format_table(sensitivity)}</section>
<div class="warning">
  <strong>Interpretation.</strong> The equal-weight objective selects an off-diagonal point at every horizon, but
  the selected policy generally improves one component while worsening the other. This is a better sampled
  scalar tradeoff, not a policy that dominates the tied solution on both metrics. Confirmation requires fresh
  repeats at the frozen selected coordinates and matched tied comparators.
</div>
</main></body></html>"""
    (output_dir / "starcoder_wsd80_full_pool_joint_objective.html").write_text(html, encoding="utf-8")


def write_markdown_report(output_dir: Path, primary: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    report = f"""# StarCoder WSD80 full-pool code/web objective

## Question

The full physical StarCoder pool removes forced StarCoder repetition. Programming Languages BPB alone has no
raw sampled global two-phase advantage at any of the four token horizons. This exploratory diagnostic tests
whether a joint code and broad-web objective has an off-diagonal optimum.

## Frozen analysis

- Primary web proxy: `eval/paloma/c4_100_domains-llama3/bpb`.
- Primary scalar objective: `0.5 * Programming Languages BPB + 0.5 * C4-100 Domains BPB`.
- Sensitivity weights: lambda(code) from 0 to 1 in increments of 0.05.
- Selection: raw minima over the 26 tied and 99 off-diagonal trained coordinates in each horizon.
- Piecewise-linear interpolation is visualization-only and does not determine any reported minimum.

## Primary result

{primary.to_markdown(index=False, floatfmt=".6f")}

The best sampled off-diagonal coordinate beats the best sampled tied coordinate on the equal-weight objective
in all four horizons. The gains are small, from {primary['raw_off_diagonal_gain_bpb'].min():.6f} to
{primary['raw_off_diagonal_gain_bpb'].max():.6f} BPB. In each horizon, the selected policy trades one component
against the other rather than improving both simultaneously.

## Metric sensitivity

{sensitivity.to_markdown(index=False, floatfmt=".6f")}

The sign is consistent across all four horizons for C4-100, C4 English, and RefinedWeb. It is less consistent
for broader aggregates. This makes the effect more credible as a code-versus-web scheduling tradeoff, but it
remains a one-seed discovery result.

## Limits

1. The off-diagonal search has 99 candidates versus 26 tied candidates, so unequal multiplicity favors finding
   a lower noisy off-diagonal minimum.
2. C4-100 Domains is a broad-web proxy, not a direct Nemotron-CC heldout target.
3. A joint-objective off-diagonal optimum is compatible with temporal gradient conflict or target revaluation,
   but does not causally establish either mechanism.
4. Fresh repeats at the frozen selected off-diagonal and tied coordinates are required before interpreting the
   0.0009-0.0024 BPB gains as expected performance.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations = load_observations(
        args.coverage,
        args.output_dir,
        refresh=args.refresh_metrics,
        workers=args.fetch_workers,
    )
    summary = summarize_objectives(observations)
    summary.to_csv(args.output_dir / "joint_objective_raw_minima.csv", index=False)
    primary = _primary_table(summary)
    sensitivity = _sensitivity_table(summary)
    primary.to_csv(args.output_dir / "primary_equal_weight_minima.csv", index=False)
    sensitivity.to_csv(args.output_dir / "web_proxy_sensitivity.csv", index=False)

    surface_figure = build_surface_figure(observations, summary)
    lambda_figure = build_lambda_figure(summary)
    write_report(args.output_dir, surface_figure, lambda_figure, primary, sensitivity)
    write_markdown_report(args.output_dir, primary, sensitivity)
    print(f"Wrote {args.output_dir / 'starcoder_wsd80_full_pool_joint_objective.html'}")


if __name__ == "__main__":
    main()
