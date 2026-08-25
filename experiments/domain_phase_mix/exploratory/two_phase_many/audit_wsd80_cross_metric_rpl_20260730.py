# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
# ]
# ///
"""Test whether retained-power-law WSD80 geometry transfers across smooth BPB targets.

The nonlinear RPL shape was selected on Paloma Dolma 100 Programming Languages BPB. This audit freezes
that shape and ridge, then refits only the target-specific linear response head for every BPB metric
available on the same checkpoints. That is a stricter transfer test than independently tuning the
nonlinear dynamics for each target.

The surface is heteroskedastic and its extreme boundaries are much noisier than its interior. Pooled
RMSE is therefore descriptive rather than decisive. The audit preregisters separate boundary,
interior, lower-tail, and optimum-neighborhood scores, plus discrete out-of-fold policy-selection
regret and full-fit optimum geometry.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_wsd80_incumbents_20260728 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_model_20260728 as rpl,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    starcoder_wsd80_panel_20260728 as wsd80,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = wsd80.REFERENCE_OUTPUTS / "wsd80_cross_metric_rpl_20260730"
ALL_METRICS_CSV = wsd80.SURFACE_DIR / "wsd80_all_bpb_metrics.csv"
PRIMARY_TARGET = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
OUTER_SPLITS = 3
OUTER_SEED = 0
BOUNDARY_MARGIN = 0.025
LOWER_TAIL_QUANTILE = 0.20
OPTIMUM_RADIUS = 0.15
OPTIMUM_GRID = 201
EXPORT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
REPRESENTATIVE_METRICS = (
    PRIMARY_TARGET,
    "eval/uncheatable_eval/github_python-llama3/bpb",
    "eval/uncheatable_eval/github_cpp-llama3/bpb",
    "eval/uncheatable_eval/arxiv_computer_science-llama3/bpb",
    "eval/uncheatable_eval/macro_bpb",
    "eval/paloma/macro_bpb",
    "eval/paloma/c4_en-llama3/bpb",
    "eval/paloma/dolma-v1_5-llama3/bpb",
    "eval/paloma/falcon-refinedweb-llama3/bpb",
)

DATASET_LABELS = {
    "4chan": "4chan",
    "ao3_english": "AO3 English",
    "arxiv_computer_science": "arXiv Computer Science",
    "arxiv_physics": "arXiv Physics",
    "bbc_news": "BBC News",
    "c4_100_domains": "C4 100 Domains",
    "c4_en": "C4 English",
    "dolma-v1_5": "Dolma v1.5",
    "dolma_100_programing_languages": "Dolma 100 Programming Languages",
    "dolma_100_subreddits": "Dolma 100 Subreddits",
    "falcon-refinedweb": "Falcon RefinedWeb",
    "gab": "Gab",
    "github_cpp": "GitHub C++",
    "github_python": "GitHub Python",
    "m2d2_s2orc_unsplit": "M2D2 S2ORC",
    "m2d2_wikipedia_unsplit": "M2D2 Wikipedia",
    "manosphere_meta_sep": "Manosphere",
    "mc4": "mC4",
    "ptb": "Penn Treebank",
    "redpajama": "RedPajama",
    "twitterAAE_HELM_fixed": "TwitterAAE",
    "wikipedia_english": "Wikipedia English",
    "wikitext_103": "WikiText-103",
}


@dataclass(frozen=True)
class FrozenRpl:
    """A nonlinear RPL shape and ridge frozen from the primary-target fit."""

    protocol: str
    shape: rpl.Shape
    ridge: float


FROZEN_MODELS = {
    "random": FrozenRpl(
        protocol="random",
        shape=rpl.Shape(
            benefit_exponent=1.0,
            benefit_offset=0.01,
            damage_exponent=2.0,
            damage_threshold=0.0,
            retention=10.0,
            late_multiplier=4.0,
            ordering_channel=True,
        ),
        ridge=1e-4,
    ),
    "blocked": FrozenRpl(
        protocol="blocked",
        shape=rpl.Shape(
            benefit_exponent=1.0,
            benefit_offset=0.01,
            damage_exponent=2.0,
            damage_threshold=0.0,
            retention=10.0,
            late_multiplier=2.0,
            ordering_channel=True,
        ),
        ridge=1e-4,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid", type=int, default=OPTIMUM_GRID)
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Optional exact metric columns. By default every complete BPB metric is audited.",
    )
    parser.add_argument(
        "--retune-representative",
        action="store_true",
        help=(
            "Also run target-specific nested random-fold shape selection on the representative "
            "metric basket. This is substantially slower than the frozen-shape transfer audit."
        ),
    )
    parser.add_argument(
        "--include-no-ordering-ablation",
        action="store_true",
        help=(
            "Also evaluate the random-fold frozen shape with the derivative-based ordering and "
            "asymmetry block removed. Retention and the late multiplier remain active."
        ),
    )
    parser.add_argument("--retained-workers", type=int, default=8)
    return parser.parse_args()


def metric_label(metric: str) -> str:
    aggregate = {
        "eval/bpb": "All eval datasets · micro",
        "eval/macro_bpb": "All eval datasets · macro",
        "eval/paloma/bpb": "Paloma · micro",
        "eval/paloma/macro_bpb": "Paloma · macro",
        "eval/uncheatable_eval/bpb": "Uncheatable · micro",
        "eval/uncheatable_eval/macro_bpb": "Uncheatable · macro",
    }.get(metric)
    if aggregate is not None:
        return aggregate
    match = re.fullmatch(r"eval/(?P<suite>paloma|uncheatable_eval)/(?P<dataset>.+)-llama3/bpb", metric)
    if match is None:
        return metric
    suite = "Paloma" if match.group("suite") == "paloma" else "Uncheatable"
    dataset = DATASET_LABELS.get(match.group("dataset"), match.group("dataset").replace("_", " ").title())
    return f"{suite} · {dataset}"


def metric_suite(metric: str) -> str:
    if "/uncheatable_eval/" in metric:
        return "Uncheatable"
    if "/paloma/" in metric:
        return "Paloma"
    return "All eval"


def load_metric_panel() -> tuple[wsd80.Panel, pd.DataFrame, tuple[str, ...]]:
    panel = wsd80.load_surface()
    metrics = pd.read_csv(ALL_METRICS_CSV)
    metric_columns = tuple(column for column in metrics if column != "wandb_run_id" and column.endswith("bpb"))
    merged = panel.frame.merge(metrics, on="wandb_run_id", how="left", validate="one_to_one")
    assert len(merged) == len(panel.frame)
    complete = tuple(column for column in metric_columns if merged[column].notna().all())
    assert PRIMARY_TARGET in complete
    return panel, merged, complete


def load_metric_replicates(metrics: tuple[str, ...]) -> pd.DataFrame:
    observations = wsd80.load_fiber_replicates()
    values = pd.read_csv(ALL_METRICS_CSV)
    merged = observations.drop(columns=[column for column in metrics if column in observations]).merge(
        values[["wandb_run_id", *metrics]],
        on="wandb_run_id",
        how="left",
        validate="one_to_one",
    )
    assert merged[list(metrics)].notna().all().all()
    return merged


def pooled_seed_sigma(replicates: pd.DataFrame, metric: str) -> float:
    variances = []
    degrees = []
    for _coordinate, block in replicates.groupby(["phase_0_starcoder", "phase_1_starcoder"]):
        if len(block) < 2:
            continue
        variances.append(float(block[metric].var(ddof=1)))
        degrees.append(len(block) - 1)
    assert variances
    return float(np.sqrt(np.average(variances, weights=degrees)))


def fit_fixed_head(
    design: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    geometry: rpl.Geometry,
    frozen: FrozenRpl,
) -> tuple[float, np.ndarray]:
    return rpl.solve_head(
        design[indices],
        target[indices],
        frozen.ridge,
        rpl.penalty_multipliers(geometry, frozen.shape),
    )


def out_of_fold_predictions(
    panel: wsd80.Panel,
    target: np.ndarray,
    geometry: rpl.Geometry,
    frozen: FrozenRpl,
) -> np.ndarray:
    design = rpl.design_matrix(panel.weights, geometry, frozen.shape)
    indices = np.arange(len(target))
    fold_builder = benchmark.random_folds if frozen.protocol == "random" else benchmark.mixture_blocked_folds
    folds = fold_builder(panel.weights, indices, OUTER_SPLITS, OUTER_SEED)
    prediction = np.full(len(target), np.nan)
    for train, test in folds:
        intercept, coefficients = fit_fixed_head(design, target, train, geometry, frozen)
        prediction[test] = intercept + design[test] @ coefficients
    assert np.isfinite(prediction).all()
    return prediction


def fit_full_model(
    panel: wsd80.Panel,
    target: np.ndarray,
    geometry: rpl.Geometry,
    frozen: FrozenRpl,
) -> rpl.Fitted:
    design = rpl.design_matrix(panel.weights, geometry, frozen.shape)
    intercept, coefficients = fit_fixed_head(design, target, np.arange(len(target)), geometry, frozen)
    return rpl.Fitted(
        shape=frozen.shape,
        ridge=frozen.ridge,
        intercept=intercept,
        coefficients=coefficients,
        geometry=geometry,
    )


def local_folds(
    weights: np.ndarray,
    global_indices: np.ndarray,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Random inner folds expressed as masks over a sliced outer-training panel."""
    global_folds = benchmark.random_folds(weights, global_indices, OUTER_SPLITS, seed)
    return tuple(
        (
            np.isin(global_indices, train),
            np.isin(global_indices, test),
        )
        for train, test in global_folds
    )


def retuned_out_of_fold_predictions(
    panel: wsd80.Panel,
    target: np.ndarray,
    geometry: rpl.Geometry,
    workers: int,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    """Nested random-fold prediction with the unchanged RPL grid selected per target."""
    indices = np.arange(len(target))
    prediction = np.full(len(target), np.nan)
    parameters: list[dict[str, float | int]] = []
    outer = benchmark.random_folds(panel.weights, indices, OUTER_SPLITS, OUTER_SEED)
    for fold_id, (train, test) in enumerate(outer):
        model = rpl.fit(
            panel.weights[train],
            target[train],
            geometry,
            local_folds(panel.weights, train, 10_000 + fold_id),
            workers=workers,
        )
        prediction[test] = model.predict(panel.weights[test])
        parameters.append(
            {
                "fold": fold_id,
                "benefit_exponent": model.shape.benefit_exponent,
                "benefit_offset": model.shape.benefit_offset,
                "damage_exponent": model.shape.damage_exponent,
                "damage_threshold": model.shape.damage_threshold,
                "retention": model.shape.retention,
                "late_multiplier": model.shape.late_multiplier,
                "ordering_channel": int(model.shape.ordering_channel),
                "ridge": model.ridge,
            }
        )
    assert np.isfinite(prediction).all()
    return prediction, parameters


def retuned_full_model(
    panel: wsd80.Panel,
    target: np.ndarray,
    geometry: rpl.Geometry,
    workers: int,
) -> rpl.Fitted:
    indices = np.arange(len(target))
    return rpl.fit(
        panel.weights,
        target,
        geometry,
        local_folds(panel.weights, indices, 20_000),
        workers=workers,
    )


def boundary_rows(panel: wsd80.Panel) -> np.ndarray:
    phase_0 = panel.phase_0[:, 1]
    phase_1 = panel.phase_1[:, 1]
    return (
        (phase_0 <= BOUNDARY_MARGIN)
        | (phase_1 <= BOUNDARY_MARGIN)
        | (phase_0 >= 1.0 - BOUNDARY_MARGIN)
        | (phase_1 >= 1.0 - BOUNDARY_MARGIN)
    )


def subset_masks(panel: wsd80.Panel, target: np.ndarray) -> tuple[dict[str, np.ndarray], int]:
    boundary = boundary_rows(panel)
    interior = ~boundary
    interior_rows = np.flatnonzero(interior)
    best_interior = int(interior_rows[np.argmin(target[interior_rows])])
    threshold = float(np.quantile(target[interior], LOWER_TAIL_QUANTILE))
    distance = np.hypot(
        panel.phase_0[:, 1] - panel.phase_0[best_interior, 1],
        panel.phase_1[:, 1] - panel.phase_1[best_interior, 1],
    )
    masks = {
        "all": np.ones(len(target), dtype=bool),
        "boundary": boundary,
        "interior": interior,
        "lower_tail": interior & (target <= threshold),
        "optimum_neighborhood": interior & (distance <= OPTIMUM_RADIUS),
    }
    assert all(mask.any() for mask in masks.values())
    return masks, best_interior


def score_predictions(
    metric: str,
    protocol: str,
    target: np.ndarray,
    prediction: np.ndarray,
    masks: dict[str, np.ndarray],
    sigma: float,
) -> list[dict[str, float | int | str]]:
    rows = []
    for subset, mask in masks.items():
        residual = prediction[mask] - target[mask]
        correlation = spearmanr(prediction[mask], target[mask]).statistic if mask.sum() > 1 else np.nan
        rows.append(
            {
                "metric": metric,
                "label": metric_label(metric),
                "suite": metric_suite(metric),
                "protocol": protocol,
                "subset": subset,
                "rows": int(mask.sum()),
                "sigma": sigma,
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "rmse_sigma": float(np.sqrt(np.mean(residual**2)) / sigma),
                "mae": float(np.mean(np.abs(residual))),
                "median_absolute": float(np.median(np.abs(residual))),
                "median_absolute_sigma": float(np.median(np.abs(residual)) / sigma),
                "bias": float(np.mean(residual)),
                "spearman": float(correlation),
            }
        )
    return rows


def discrete_selection(
    metric: str,
    protocol: str,
    target: np.ndarray,
    prediction: np.ndarray,
    masks: dict[str, np.ndarray],
    panel: wsd80.Panel,
) -> dict[str, float | int | str]:
    interior = np.flatnonzero(masks["interior"])
    ranked = interior[np.argsort(prediction[interior])]
    selected = int(ranked[0])
    best = int(interior[np.argmin(target[interior])])
    top_five = ranked[: min(5, len(ranked))]
    return {
        "metric": metric,
        "label": metric_label(metric),
        "suite": metric_suite(metric),
        "protocol": protocol,
        "selected_index": selected,
        "selected_phase_0": float(panel.phase_0[selected, 1]),
        "selected_phase_1": float(panel.phase_1[selected, 1]),
        "selected_observed_bpb": float(target[selected]),
        "best_interior_index": best,
        "best_interior_phase_0": float(panel.phase_0[best, 1]),
        "best_interior_phase_1": float(panel.phase_1[best, 1]),
        "best_interior_bpb": float(target[best]),
        "regret_at_1": float(target[selected] - target[best]),
        "regret_at_5": float(np.min(target[top_five]) - target[best]),
        "selected_distance": float(
            np.hypot(
                panel.phase_0[selected, 1] - panel.phase_0[best, 1],
                panel.phase_1[selected, 1] - panel.phase_1[best, 1],
            )
        ),
    }


def continuous_optimum(
    metric: str,
    protocol: str,
    target: np.ndarray,
    model: rpl.Fitted,
    panel: wsd80.Panel,
    grid: int,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    axis = np.linspace(0.0, 1.0, grid)
    phase_0, phase_1 = np.meshgrid(axis, axis, indexing="ij")
    weights = benchmark.grid_weights(phase_0.ravel(), phase_1.ravel())
    prediction = model.predict(weights)
    interior = (
        (phase_0.ravel() > BOUNDARY_MARGIN)
        & (phase_1.ravel() > BOUNDARY_MARGIN)
        & (phase_0.ravel() < 1.0 - BOUNDARY_MARGIN)
        & (phase_1.ravel() < 1.0 - BOUNDARY_MARGIN)
    )
    best_grid = int(np.argmin(prediction))
    interior_rows = np.flatnonzero(interior)
    best_grid_interior = int(interior_rows[np.argmin(prediction[interior_rows])])

    tied_rows = np.flatnonzero(np.isclose(panel.phase_0[:, 1], panel.phase_1[:, 1]))
    observed_best = int(np.argmin(target))
    observed_best_interior_rows = np.flatnonzero(~boundary_rows(panel))
    observed_best_interior = int(observed_best_interior_rows[np.argmin(target[observed_best_interior_rows])])
    observed_best_tied = int(tied_rows[np.argmin(target[tied_rows])])

    tied_axis = np.linspace(0.0, 1.0, grid * grid)
    tied_prediction = model.predict(benchmark.grid_weights(tied_axis, tied_axis))
    row = {
        "metric": metric,
        "label": metric_label(metric),
        "suite": metric_suite(metric),
        "protocol": protocol,
        "observed_best_phase_0": float(panel.phase_0[observed_best, 1]),
        "observed_best_phase_1": float(panel.phase_1[observed_best, 1]),
        "observed_best_bpb": float(target[observed_best]),
        "observed_best_is_boundary": int(boundary_rows(panel)[observed_best]),
        "observed_best_interior_phase_0": float(panel.phase_0[observed_best_interior, 1]),
        "observed_best_interior_phase_1": float(panel.phase_1[observed_best_interior, 1]),
        "observed_best_interior_bpb": float(target[observed_best_interior]),
        "observed_best_tied_bpb": float(target[observed_best_tied]),
        "observed_sampled_two_phase_gain": float(target[observed_best_tied] - target[observed_best]),
        "predicted_best_phase_0": float(phase_0.ravel()[best_grid]),
        "predicted_best_phase_1": float(phase_1.ravel()[best_grid]),
        "predicted_best_bpb": float(prediction[best_grid]),
        "predicted_best_is_boundary": int(not interior[best_grid]),
        "predicted_best_interior_phase_0": float(phase_0.ravel()[best_grid_interior]),
        "predicted_best_interior_phase_1": float(phase_1.ravel()[best_grid_interior]),
        "predicted_best_interior_bpb": float(prediction[best_grid_interior]),
        "predicted_best_tied_bpb": float(np.min(tied_prediction)),
        "predicted_two_phase_gain": float(np.min(tied_prediction) - np.min(prediction)),
        "optimum_distance_all": float(
            np.hypot(
                phase_0.ravel()[best_grid] - panel.phase_0[observed_best, 1],
                phase_1.ravel()[best_grid] - panel.phase_1[observed_best, 1],
            )
        ),
        "optimum_distance_interior": float(
            np.hypot(
                phase_0.ravel()[best_grid_interior] - panel.phase_0[observed_best_interior, 1],
                phase_1.ravel()[best_grid_interior] - panel.phase_1[observed_best_interior, 1],
            )
        ),
    }
    grid_frame = pd.DataFrame(
        {
            "metric": metric,
            "protocol": protocol,
            "phase_0": phase_0.ravel(),
            "phase_1": phase_1.ravel(),
            "prediction": prediction,
        }
    )
    return row, grid_frame


def heatmap(metrics: pd.DataFrame, output_path: Path) -> None:
    subsets = ("all", "boundary", "interior", "lower_tail", "optimum_neighborhood")
    labels = (
        metrics[["metric", "label", "suite"]].drop_duplicates().sort_values(["suite", "label"]).reset_index(drop=True)
    )
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Random-fold interpolation", "Blocked-region transfer"),
        horizontal_spacing=0.13,
    )
    for column, protocol in enumerate(("random", "blocked"), start=1):
        block = metrics[metrics["protocol"] == protocol]
        table = (
            labels[["metric"]]
            .merge(block.pivot(index="metric", columns="subset", values="median_absolute_sigma"), on="metric")
            .set_index("metric")
            .loc[labels["metric"], list(subsets)]
        )
        figure.add_trace(
            go.Heatmap(
                z=table.to_numpy(),
                x=["All", "Boundary", "Interior", "Lower 20%", "Optimum neighborhood"],
                y=labels["label"],
                colorscale="RdYlGn_r",
                zmin=0.0,
                zmax=float(np.nanquantile(metrics["median_absolute_sigma"], 0.90)),
                colorbar={"title": "Median |error| / seed SD", "x": 1.02} if column == 2 else None,
                showscale=column == 2,
                hovertemplate="%{y}<br>%{x}<br>%{z:.2f} seed SD<extra></extra>",
            ),
            row=1,
            col=column,
        )
    figure.update_layout(
        title="Frozen-shape retained power law across WSD80 metrics",
        height=max(900, 31 * len(labels)),
        width=1500,
        margin={"l": 260, "r": 100, "t": 100, "b": 80},
        paper_bgcolor="#fffdf7",
        plot_bgcolor="#fffdf7",
    )
    figure.write_html(output_path, include_plotlyjs=True, config=EXPORT_CONFIG)


def selection_figure(optima: pd.DataFrame, selection: pd.DataFrame, output_path: Path) -> None:
    random_optima = optima[optima["protocol"] == "random"].copy()
    random_selection = selection[selection["protocol"] == "random"].copy()
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Two-phase gain", "OOF interior policy-selection regret"),
    )
    palette = {"Paloma": "#d95f02", "Uncheatable": "#1b9e77", "All eval": "#7570b3"}
    for suite, block in random_optima.groupby("suite"):
        figure.add_trace(
            go.Scatter(
                x=block["observed_sampled_two_phase_gain"],
                y=block["predicted_two_phase_gain"],
                mode="markers",
                name=suite,
                legendgroup=suite,
                marker={"size": 10, "color": palette[suite], "line": {"color": "#17324d", "width": 1}},
                customdata=np.column_stack([block["label"]]),
                hovertemplate=(
                    "%{customdata[0]}<br>observed sampled gain %{x:.5f}" "<br>predicted gain %{y:.5f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        selected = random_selection[random_selection["suite"] == suite]
        figure.add_trace(
            go.Scatter(
                x=selected["regret_at_1"],
                y=selected["regret_at_5"],
                mode="markers",
                name=suite,
                legendgroup=suite,
                showlegend=False,
                marker={"size": 10, "color": palette[suite], "line": {"color": "#17324d", "width": 1}},
                customdata=np.column_stack([selected["label"]]),
                hovertemplate="%{customdata[0]}<br>Regret@1 %{x:.5f}<br>Regret@5 %{y:.5f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    gain_min = float(
        min(
            random_optima["observed_sampled_two_phase_gain"].min(),
            random_optima["predicted_two_phase_gain"].min(),
        )
    )
    gain_max = float(
        max(
            random_optima["observed_sampled_two_phase_gain"].max(),
            random_optima["predicted_two_phase_gain"].max(),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[gain_min, gain_max],
            y=[gain_min, gain_max],
            mode="lines",
            line={"color": "#6b7785", "dash": "dash"},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    figure.update_xaxes(title="Observed sampled two-phase gain (BPB)", row=1, col=1)
    figure.update_yaxes(title="Predicted two-phase gain (BPB)", row=1, col=1)
    figure.update_xaxes(title="Regret@1 (BPB)", row=1, col=2)
    figure.update_yaxes(title="Regret@5 (BPB)", row=1, col=2)
    figure.update_layout(
        title="Cross-metric optimum behavior with primary RPL shape frozen",
        width=1450,
        height=620,
        paper_bgcolor="#fffdf7",
        plot_bgcolor="#fffdf7",
    )
    figure.write_html(output_path, include_plotlyjs=True, config=EXPORT_CONFIG)


def surface_gallery(
    panel: wsd80.Panel,
    targets: dict[str, np.ndarray],
    models: dict[str, rpl.Fitted],
    grids: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    metrics = list(targets)
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Observed checkpoints", "Frozen-shape fitted surface", "Prediction residual"),
        horizontal_spacing=0.08,
    )
    traces_per_metric = 3
    for metric_index, metric in enumerate(metrics):
        target = targets[metric]
        model = models[metric]
        prediction = model.predict(panel.weights)
        grid = grids[metric]
        axis = np.sort(grid["phase_0"].unique())
        z = grid.pivot(index="phase_0", columns="phase_1", values="prediction").loc[axis, axis].to_numpy()
        low = float(min(target.min(), grid["prediction"].min()))
        high = float(max(target.max(), grid["prediction"].max()))
        visible = metric_index == 0
        figure.add_trace(
            go.Scattergl(
                x=panel.phase_0[:, 1],
                y=panel.phase_1[:, 1],
                mode="markers",
                visible=visible,
                marker={
                    "size": 8,
                    "color": target,
                    "colorscale": "RdYlGn_r",
                    "cmin": low,
                    "cmax": high,
                    "line": {"color": "#17324d", "width": 0.5},
                    "showscale": False,
                },
                customdata=np.column_stack([target]),
                hovertemplate="p0=%{x:.3f}<br>p1=%{y:.3f}<br>observed=%{customdata[0]:.5f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Heatmap(
                x=axis,
                y=axis,
                z=z.T,
                visible=visible,
                colorscale="RdYlGn_r",
                zmin=low,
                zmax=high,
                showscale=False,
                hovertemplate="p0=%{x:.3f}<br>p1=%{y:.3f}<br>predicted=%{z:.5f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        residual = prediction - target
        limit = float(np.quantile(np.abs(residual), 0.98))
        figure.add_trace(
            go.Scattergl(
                x=panel.phase_0[:, 1],
                y=panel.phase_1[:, 1],
                mode="markers",
                visible=visible,
                marker={
                    "size": 8,
                    "color": residual,
                    "colorscale": "RdYlGn_r",
                    "cmin": -limit,
                    "cmax": limit,
                    "line": {"color": "#17324d", "width": 0.5},
                    "showscale": True,
                    "colorbar": {"title": "predicted - observed"},
                },
                customdata=np.column_stack([target, prediction]),
                hovertemplate=(
                    "p0=%{x:.3f}<br>p1=%{y:.3f}<br>observed=%{customdata[0]:.5f}"
                    "<br>predicted=%{customdata[1]:.5f}<br>residual=%{marker.color:.5f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=3,
        )
    buttons = []
    for metric_index, metric in enumerate(metrics):
        visible = [False] * (traces_per_metric * len(metrics))
        start = traces_per_metric * metric_index
        visible[start : start + traces_per_metric] = [True] * traces_per_metric
        buttons.append(
            {
                "label": metric_label(metric),
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"WSD80 surface transfer · {metric_label(metric)} BPB"},
                ],
            }
        )
    figure.update_xaxes(title="Phase 0 StarCoder share", range=[0, 1])
    figure.update_yaxes(title="Phase 1 StarCoder share", range=[0, 1], scaleanchor=None)
    figure.update_layout(
        title=f"WSD80 surface transfer · {metric_label(metrics[0])} BPB",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.01,
                "y": 1.16,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
        width=1600,
        height=620,
        margin={"l": 90, "r": 100, "t": 150, "b": 80},
        paper_bgcolor="#fffdf7",
        plot_bgcolor="#fffdf7",
    )
    figure.write_html(output_path, include_plotlyjs=True, config=EXPORT_CONFIG)


def render_report(
    metrics: pd.DataFrame,
    selection: pd.DataFrame,
    optima: pd.DataFrame,
    primary_sigma: float,
    retuned_metrics: pd.DataFrame | None,
    retuned_selection: pd.DataFrame | None,
) -> str:
    random = metrics[metrics["protocol"] == "random"]
    blocked = metrics[metrics["protocol"] == "blocked"]
    lines = [
        "# WSD80 cross-metric retained-power-law audit",
        "",
        "## Protocol",
        "",
        (
            "The nonlinear RPL shape and ridge were frozen from the Paloma Dolma 100 Programming "
            "Languages full-panel fit. Each metric refits only its linear response amplitudes and "
            "intercept. The primary-target frozen result is therefore a structural reference, not "
            "out-of-target evidence; its nested target-specific result is the honest primary check. "
            "The random- and blocked-fold references use separately selected frozen shapes. "
            f"Boundary rows have either phase share within {BOUNDARY_MARGIN:.3f} of 0 or 1. "
            f"The lower tail is the best {LOWER_TAIL_QUANTILE:.0%} of interior observations; the "
            f"optimum neighborhood is within Euclidean radius {OPTIMUM_RADIUS:.2f} of the best "
            "observed interior coordinate. These target-derived geometric subsets are descriptive, "
            "while all fitted predictions remain out of fold."
        ),
        "",
        f"Primary-target pooled seed SD: `{primary_sigma:.6f}` BPB.",
        "",
        "## Median transfer diagnostics across metrics",
        "",
        (
            "| protocol | subset | median RMSE / metric seed SD | "
            "median absolute error / metric seed SD | median Spearman |"
        ),
        "|---|---|---:|---:|---:|",
    ]
    for protocol, frame in (("random", random), ("blocked", blocked)):
        for subset in ("all", "boundary", "interior", "lower_tail", "optimum_neighborhood"):
            block = frame[frame["subset"] == subset]
            lines.append(
                f"| {protocol} | {subset} | {block['rmse_sigma'].median():.3f} | "
                f"{block['median_absolute_sigma'].median():.3f} | {block['spearman'].median():.3f} |"
            )
    random_selection = selection[selection["protocol"] == "random"]
    random_optima = optima[optima["protocol"] == "random"]
    lines.extend(
        [
            "",
            "## Policy selection",
            "",
            f"- Median random-fold interior Regret@1: `{random_selection['regret_at_1'].median():.6f}` BPB.",
            f"- Median random-fold interior Regret@5: `{random_selection['regret_at_5'].median():.6f}` BPB.",
            (
                f"- Metrics whose observed sampled optimum is interior: "
                f"`{int((random_optima['observed_best_is_boundary'] == 0).sum())} / {len(random_optima)}`."
            ),
            (
                f"- Metrics whose fitted continuous optimum is interior: "
                f"`{int((random_optima['predicted_best_is_boundary'] == 0).sum())} / {len(random_optima)}`."
            ),
            "",
            "## Selected metrics",
            "",
            (
                "| metric | interior median / SD | lower-tail median / SD | "
                "optimum-neighborhood median / SD | Regret@1 | observed gain | predicted gain |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    selected_labels = (
        "Paloma · Dolma 100 Programming Languages",
        "Uncheatable · GitHub Python",
        "Uncheatable · GitHub C++",
        "Uncheatable · arXiv Computer Science",
        "Uncheatable · macro",
        "Paloma · macro",
        "Paloma · C4 English",
        "Paloma · Dolma v1.5",
        "Paloma · Falcon RefinedWeb",
    )
    for label in selected_labels:
        metric_rows = random[(random["label"] == label)]
        if metric_rows.empty:
            continue
        by_subset = metric_rows.set_index("subset")
        chosen = random_selection[random_selection["label"] == label].iloc[0]
        optimum = random_optima[random_optima["label"] == label].iloc[0]
        lines.append(
            f"| {label} | {by_subset.loc['interior', 'median_absolute_sigma']:.3f} | "
            f"{by_subset.loc['lower_tail', 'median_absolute_sigma']:.3f} | "
            f"{by_subset.loc['optimum_neighborhood', 'median_absolute_sigma']:.3f} | "
            f"{chosen['regret_at_1']:.6f} | {optimum['observed_sampled_two_phase_gain']:.6f} | "
            f"{optimum['predicted_two_phase_gain']:.6f} |"
        )
    no_ordering = metrics[metrics["protocol"] == "random_no_ordering"]
    if not no_ordering.empty:
        no_ordering_selection = selection[selection["protocol"] == "random_no_ordering"]
        no_ordering_optima = optima[optima["protocol"] == "random_no_ordering"]
        lines.extend(
            [
                "",
                "## Derivative-based ordering ablation",
                "",
                (
                    "This ablation removes the ordering and asymmetry columns while preserving the "
                    "retention gate, late multiplier, aggregate response, and concentration gap."
                ),
                "",
                (
                    "| metric | full interior median BPB | no-ordering interior median BPB | "
                    "full Regret@1 | no-ordering Regret@1 | full predicted gain | "
                    "no-ordering predicted gain |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label in selected_labels:
            full = random[(random["label"] == label)].set_index("subset")
            ablated = no_ordering[(no_ordering["label"] == label)].set_index("subset")
            if full.empty or ablated.empty:
                continue
            full_pick = random_selection[random_selection["label"] == label].iloc[0]
            ablated_pick = no_ordering_selection[no_ordering_selection["label"] == label].iloc[0]
            full_optimum = random_optima[random_optima["label"] == label].iloc[0]
            ablated_optimum = no_ordering_optima[no_ordering_optima["label"] == label].iloc[0]
            lines.append(
                f"| {label} | {full.loc['interior', 'median_absolute']:.6f} | "
                f"{ablated.loc['interior', 'median_absolute']:.6f} | "
                f"{full_pick['regret_at_1']:.6f} | {ablated_pick['regret_at_1']:.6f} | "
                f"{full_optimum['predicted_two_phase_gain']:.6f} | "
                f"{ablated_optimum['predicted_two_phase_gain']:.6f} |"
            )
    if retuned_metrics is not None and retuned_selection is not None:
        lines.extend(
            [
                "",
                "## Target-specific nonlinear refits",
                "",
                (
                    "This second pass keeps the RPL equation and candidate grid fixed, but selects "
                    "nonlinear shape and ridge independently inside each target's outer training folds."
                ),
                "",
                (
                    "| metric | frozen interior median / SD | retuned interior median / SD | "
                    "frozen lower-tail median / SD | retuned lower-tail median / SD | "
                    "frozen Regret@1 | retuned Regret@1 |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label in selected_labels:
            frozen = random[random["label"] == label]
            retuned = retuned_metrics[retuned_metrics["label"] == label]
            if frozen.empty or retuned.empty:
                continue
            frozen_by_subset = frozen.set_index("subset")
            retuned_by_subset = retuned.set_index("subset")
            frozen_pick = random_selection[random_selection["label"] == label].iloc[0]
            retuned_pick = retuned_selection[retuned_selection["label"] == label].iloc[0]
            lines.append(
                f"| {label} | {frozen_by_subset.loc['interior', 'median_absolute_sigma']:.3f} | "
                f"{retuned_by_subset.loc['interior', 'median_absolute_sigma']:.3f} | "
                f"{frozen_by_subset.loc['lower_tail', 'median_absolute_sigma']:.3f} | "
                f"{retuned_by_subset.loc['lower_tail', 'median_absolute_sigma']:.3f} | "
                f"{frozen_pick['regret_at_1']:.6f} | {retuned_pick['regret_at_1']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `cross_metric_scores.csv`: OOF errors by metric, protocol, and geometric subset.",
            "- `cross_metric_selection.csv`: discrete OOF policy-selection diagnostics.",
            "- `cross_metric_optima.csv`: observed and full-fit continuous optimum geometry.",
            "- `cross_metric_error_heatmap.html`: boundary versus interior error.",
            "- `cross_metric_selection.html`: gain and selection-regret diagnostics.",
            "- `cross_metric_surface_gallery.html`: observed, fitted, and residual surfaces for every metric.",
        ]
    )
    if retuned_metrics is not None:
        lines.extend(
            [
                "- `retuned_cross_metric_scores.csv`: nested target-specific OOF errors.",
                "- `retuned_cross_metric_selection.csv`: nested target-specific policy selection.",
                "- `retuned_cross_metric_optima.csv`: target-specific full-fit optimum geometry.",
                "- `retuned_cross_metric_parameters.csv`: selected nonlinear parameters by fold and full fit.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel, frame, available = load_metric_panel()
    metrics = tuple(args.metrics) if args.metrics else available
    missing = sorted(set(metrics) - set(available))
    if missing:
        raise ValueError(f"metrics are unavailable or incomplete: {missing}")
    replicates = load_metric_replicates(metrics)
    geometry = rpl.Geometry(
        c0=panel.c0,
        c1=panel.c1,
        phase_0_fraction=wsd80.REALIZED_PHASE_0_FRACTION,
    )

    metric_rows: list[dict[str, float | int | str]] = []
    selection_rows: list[dict[str, float | int | str]] = []
    optimum_rows: list[dict[str, float | int | str]] = []
    targets: dict[str, np.ndarray] = {}
    gallery_models: dict[str, rpl.Fitted] = {}
    gallery_grids: dict[str, pd.DataFrame] = {}
    noise_rows = []
    frozen_models = dict(FROZEN_MODELS)
    if args.include_no_ordering_ablation:
        random_model = FROZEN_MODELS["random"]
        frozen_models["random_no_ordering"] = FrozenRpl(
            protocol="random",
            shape=replace(random_model.shape, ordering_channel=False),
            ridge=random_model.ridge,
        )
    for metric_index, metric in enumerate(metrics, start=1):
        print(f"metric {metric_index}/{len(metrics)}: {metric_label(metric)}", flush=True)
        target = frame[metric].to_numpy(dtype=float)
        targets[metric] = target
        sigma = pooled_seed_sigma(replicates, metric)
        noise_rows.append(
            {
                "metric": metric,
                "label": metric_label(metric),
                "suite": metric_suite(metric),
                "pooled_seed_sigma": sigma,
            }
        )
        masks, _best_interior = subset_masks(panel, target)
        for protocol, frozen in frozen_models.items():
            oof = out_of_fold_predictions(panel, target, geometry, frozen)
            metric_rows.extend(score_predictions(metric, protocol, target, oof, masks, sigma))
            selection_rows.append(discrete_selection(metric, protocol, target, oof, masks, panel))
            model = fit_full_model(panel, target, geometry, frozen)
            optimum, grid = continuous_optimum(metric, protocol, target, model, panel, args.grid)
            optimum_rows.append(optimum)
            if protocol == "random":
                gallery_models[metric] = model
                gallery_grids[metric] = grid

    metrics_frame = pd.DataFrame(metric_rows)
    selection_frame = pd.DataFrame(selection_rows)
    optima_frame = pd.DataFrame(optimum_rows)
    noise_frame = pd.DataFrame(noise_rows)
    metrics_frame.to_csv(args.output_dir / "cross_metric_scores.csv", index=False)
    selection_frame.to_csv(args.output_dir / "cross_metric_selection.csv", index=False)
    optima_frame.to_csv(args.output_dir / "cross_metric_optima.csv", index=False)
    noise_frame.to_csv(args.output_dir / "cross_metric_seed_noise.csv", index=False)

    heatmap(metrics_frame, args.output_dir / "cross_metric_error_heatmap.html")
    selection_figure(optima_frame, selection_frame, args.output_dir / "cross_metric_selection.html")
    surface_gallery(
        panel,
        targets,
        gallery_models,
        gallery_grids,
        args.output_dir / "cross_metric_surface_gallery.html",
    )

    primary_sigma = float(noise_frame.loc[noise_frame["metric"] == PRIMARY_TARGET, "pooled_seed_sigma"].iloc[0])
    retuned_metrics_frame = None
    retuned_selection_frame = None
    if args.retune_representative:
        if args.retained_workers < 1:
            raise ValueError("--retained-workers must be positive")
        retuned_metric_rows: list[dict[str, float | int | str]] = []
        retuned_selection_rows: list[dict[str, float | int | str]] = []
        retuned_optimum_rows: list[dict[str, float | int | str]] = []
        retuned_parameter_rows: list[dict[str, float | int | str]] = []
        for metric_index, metric in enumerate(REPRESENTATIVE_METRICS, start=1):
            if metric not in targets:
                continue
            print(
                f"retuned metric {metric_index}/{len(REPRESENTATIVE_METRICS)}: {metric_label(metric)}",
                flush=True,
            )
            target = targets[metric]
            sigma = float(noise_frame.loc[noise_frame["metric"] == metric, "pooled_seed_sigma"].iloc[0])
            masks, _best_interior = subset_masks(panel, target)
            oof, parameters = retuned_out_of_fold_predictions(
                panel,
                target,
                geometry,
                args.retained_workers,
            )
            retuned_metric_rows.extend(score_predictions(metric, "retuned_random", target, oof, masks, sigma))
            retuned_selection_rows.append(discrete_selection(metric, "retuned_random", target, oof, masks, panel))
            for row in parameters:
                retuned_parameter_rows.append({"metric": metric, "label": metric_label(metric), **row})
            full_model = retuned_full_model(panel, target, geometry, args.retained_workers)
            optimum, _grid = continuous_optimum(
                metric,
                "retuned_random",
                target,
                full_model,
                panel,
                args.grid,
            )
            retuned_optimum_rows.append(optimum)
            retuned_parameter_rows.append(
                {
                    "metric": metric,
                    "label": metric_label(metric),
                    "fold": -1,
                    "benefit_exponent": full_model.shape.benefit_exponent,
                    "benefit_offset": full_model.shape.benefit_offset,
                    "damage_exponent": full_model.shape.damage_exponent,
                    "damage_threshold": full_model.shape.damage_threshold,
                    "retention": full_model.shape.retention,
                    "late_multiplier": full_model.shape.late_multiplier,
                    "ordering_channel": int(full_model.shape.ordering_channel),
                    "ridge": full_model.ridge,
                }
            )
        retuned_metrics_frame = pd.DataFrame(retuned_metric_rows)
        retuned_selection_frame = pd.DataFrame(retuned_selection_rows)
        retuned_metrics_frame.to_csv(args.output_dir / "retuned_cross_metric_scores.csv", index=False)
        retuned_selection_frame.to_csv(
            args.output_dir / "retuned_cross_metric_selection.csv",
            index=False,
        )
        pd.DataFrame(retuned_optimum_rows).to_csv(
            args.output_dir / "retuned_cross_metric_optima.csv",
            index=False,
        )
        pd.DataFrame(retuned_parameter_rows).to_csv(
            args.output_dir / "retuned_cross_metric_parameters.csv",
            index=False,
        )

    report = render_report(
        metrics_frame,
        selection_frame,
        optima_frame,
        primary_sigma,
        retuned_metrics_frame,
        retuned_selection_frame,
    )
    (args.output_dir / "report.md").write_text(report + "\n")
    summary = {
        "metrics": len(metrics),
        "fit_rows": len(panel.y),
        "repeat_coordinates": int((replicates.groupby(["phase_0_starcoder", "phase_1_starcoder"]).size() >= 2).sum()),
        "boundary_margin": BOUNDARY_MARGIN,
        "lower_tail_quantile": LOWER_TAIL_QUANTILE,
        "optimum_radius": OPTIMUM_RADIUS,
        "outer_splits": OUTER_SPLITS,
        "outer_seed": OUTER_SEED,
        "primary_target": PRIMARY_TARGET,
        "primary_seed_sigma": primary_sigma,
        "median_metrics": (
            metrics_frame.groupby(["protocol", "subset"])[["rmse_sigma", "median_absolute_sigma", "spearman"]]
            .median()
            .reset_index()
            .to_dict(orient="records")
        ),
    }
    if retuned_metrics_frame is not None and retuned_selection_frame is not None:
        summary["retuned_median_metrics"] = (
            retuned_metrics_frame.groupby(["protocol", "subset"])[["rmse_sigma", "median_absolute_sigma", "spearman"]]
            .median()
            .reset_index()
            .to_dict(orient="records")
        )
        summary["retuned_median_regret_at_1"] = float(retuned_selection_frame["regret_at_1"].median())
        summary["retuned_median_regret_at_5"] = float(retuned_selection_frame["regret_at_5"].median())
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
