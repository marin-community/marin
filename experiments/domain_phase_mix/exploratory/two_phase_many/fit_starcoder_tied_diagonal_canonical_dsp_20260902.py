# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Fit canonical single-phase DSP to the StarCoder tied-diagonal curves."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix import olmix_loglinear_fit as olmix_loglinear  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_dsp_single_phase_ladder_20260824 as dsp_ladder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    starcoder_wsd80_epoch_accounting as epoch_accounting,
)

SOURCE_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_fixed_model_tied_diagonal_20260730" / "results_20260731"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_tied_diagonal_canonical_dsp_20260902"
TOKEN_BUDGETS = (1_000_000_000, 2_000_000_000, 4_000_000_000, 8_000_000_000)
OUTER_FOLDS = 5
INNER_FOLDS = 3
FOLD_SEED = 20_260_902
DENSE_POINTS = 1001
PLOTLY_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}, "responsive": True}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--restarts", type=int, default=48)
    parser.add_argument(
        "--reuse-existing-dsp",
        action="store_true",
        help="Reuse validated DSP outputs in --output-dir and only fit the inexpensive OLMix overlay.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observations = pd.read_csv(source_dir / "tied_diagonal_observations.csv")
    optima = pd.read_csv(source_dir / "tied_optima.csv")
    noise = pd.read_csv(source_dir / "repeat_noise.csv")
    expected_weights = np.arange(21, dtype=float) / 20.0
    if tuple(sorted(observations["token_budget_requested"].unique())) != TOKEN_BUDGETS:
        raise ValueError("StarCoder tied panel does not contain the frozen four token budgets")
    for budget, group in observations.groupby("token_budget_requested", sort=True):
        weights = np.sort(group["weight"].to_numpy(float))
        if len(group) != len(expected_weights) or not np.allclose(weights, expected_weights, atol=1e-12):
            raise ValueError(f"{budget}: expected the complete 21-point tied diagonal")
    if set(optima["token_budget_requested"]) != set(TOKEN_BUDGETS):
        raise ValueError("Frozen tied optima are incomplete")
    if set(noise["token_budget_requested"]) != set(TOKEN_BUDGETS):
        raise ValueError("Frozen repeat-noise estimates are incomplete")
    return observations, optima, noise


def exposures(starcoder_share: np.ndarray) -> np.ndarray:
    epoch_scales = np.asarray(
        [
            epoch_accounting.SIMULATED_EPOCH_TARGET_BUDGET / epoch_accounting.NEMOTRON_SOURCE_TOKENS,
            epoch_accounting.SIMULATED_EPOCH_TARGET_BUDGET / epoch_accounting.STARCODER_SOURCE_TOKENS,
        ],
        dtype=float,
    )
    return tied_weights(starcoder_share) * epoch_scales[None, :]


def tied_weights(starcoder_share: np.ndarray) -> np.ndarray:
    """Return Nemotron and StarCoder weights on the tied single-phase edge."""
    return np.column_stack([1.0 - starcoder_share, starcoder_share])


def interleaved_folds(row_count: int, fold_count: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    labels = np.arange(row_count) % fold_count
    rows = np.arange(row_count)
    return tuple((rows[labels != fold], rows[labels == fold]) for fold in range(fold_count))


def fit_canonical(
    exposure: np.ndarray,
    response: np.ndarray,
    *,
    seed: int,
    maxiter: int,
    restarts: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    canonical = next(rung for rung in dsp_ladder.LADDER if rung.name == "canonical")
    return dsp_ladder.fit_rung(
        exposure,
        response,
        canonical,
        interleaved_folds(len(response), INNER_FOLDS),
        (),
        seed=seed,
        maxiter=maxiter,
        restarts=restarts,
    )


def predict_canonical(
    exposure: np.ndarray,
    vector: np.ndarray,
    intercept: float,
    coefficients: np.ndarray,
) -> np.ndarray:
    canonical = next(rung for rung in dsp_ladder.LADDER if rung.name == "canonical")
    return intercept + dsp_ladder.rung_design(exposure, vector, canonical, exposure.shape[1]) @ coefficients


def calibration(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    if np.ptp(predicted) <= 1e-12:
        return float(actual.mean()), 0.0
    fit = stats.linregress(predicted, actual)
    return float(fit.intercept), float(fit.slope)


def fit_curves(
    observations: pd.DataFrame,
    optima: pd.DataFrame,
    noise: pd.DataFrame,
    *,
    maxiter: int,
    restarts: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, float | int | str]] = []
    dense_rows: list[dict[str, float | int]] = []
    metric_rows: list[dict[str, float | int | bool]] = []
    parameter_rows: list[dict[str, float | int | str]] = []
    optimum_lookup = optima.set_index("token_budget_requested")
    noise_lookup = noise.set_index("token_budget_requested")

    for budget, unsorted_group in observations.groupby("token_budget_requested", sort=True):
        group = unsorted_group.sort_values("weight").reset_index(drop=True)
        weights = group["weight"].to_numpy(float)
        response = group["starcoder_bpb"].to_numpy(float)
        exposure = exposures(weights)
        outer_splits = interleaved_folds(len(group), OUTER_FOLDS)
        oof = np.full(len(group), np.nan)
        fold_labels = np.full(len(group), -1, dtype=int)

        for fold, (train, test) in enumerate(outer_splits):
            vector, intercept, coefficients = fit_canonical(
                exposure[train],
                response[train],
                seed=FOLD_SEED + int(budget // 1_000_000) + fold,
                maxiter=maxiter,
                restarts=restarts,
            )
            oof[test] = predict_canonical(exposure[test], vector, intercept, coefficients)
            fold_labels[test] = fold
        if not np.isfinite(oof).all() or np.any(fold_labels < 0):
            raise ValueError(f"{budget}: incomplete out-of-fold predictions")

        vector, intercept, coefficients = fit_canonical(
            exposure,
            response,
            seed=FOLD_SEED + int(budget // 1_000_000),
            maxiter=maxiter,
            restarts=restarts,
        )
        dense_weight = np.linspace(0.0, 1.0, DENSE_POINTS)
        dense_prediction = predict_canonical(exposures(dense_weight), vector, intercept, coefficients)
        full_at_observed = predict_canonical(exposure, vector, intercept, coefficients)

        for index, row in group.iterrows():
            prediction_rows.append(
                {
                    "token_budget_requested": int(budget),
                    "weight": float(row["weight"]),
                    "observed_bpb": float(row["starcoder_bpb"]),
                    "oof_prediction_bpb": float(oof[index]),
                    "full_fit_prediction_bpb": float(full_at_observed[index]),
                    "outer_fold": int(fold_labels[index]),
                    "run_name": str(row["run_name"]),
                    "source": str(row["source"]),
                }
            )
        for weight, prediction in zip(dense_weight, dense_prediction, strict=True):
            dense_rows.append(
                {
                    "token_budget_requested": int(budget),
                    "weight": float(weight),
                    "full_fit_prediction_bpb": float(prediction),
                }
            )

        observed_min = optimum_lookup.loc[budget]
        repeat_sd = float(noise_lookup.loc[budget, "repeat_sd_bpb"])
        oof_selected = int(np.argmin(oof))
        dense_selected = int(np.argmin(dense_prediction))
        calibration_intercept, calibration_slope = calibration(response, oof)
        spearman = float(stats.spearmanr(oof, response).statistic)
        rmse = float(np.sqrt(np.mean((oof - response) ** 2)))
        full_fit_rmse = float(np.sqrt(np.mean((full_at_observed - response) ** 2)))
        interior = (weights > 0.0) & (weights < 1.0)
        oof_interior_rmse = float(np.sqrt(np.mean((oof[interior] - response[interior]) ** 2)))
        metric_rows.append(
            {
                "token_budget_requested": int(budget),
                "rows": len(group),
                "oof_rmse": rmse,
                "oof_interior_rmse": oof_interior_rmse,
                "oof_spearman": spearman,
                "calibration_intercept": calibration_intercept,
                "calibration_slope": calibration_slope,
                "repeat_sd_bpb": repeat_sd,
                "oof_rmse_over_repeat_sd": rmse / repeat_sd,
                "full_fit_rmse": full_fit_rmse,
                "full_fit_rmse_over_repeat_sd": full_fit_rmse / repeat_sd,
                "observed_grid_min_weight": float(observed_min["sampled_min_weight"]),
                "observed_grid_min_bpb": float(observed_min["sampled_min_bpb"]),
                "one_sd_basin_low": float(observed_min["one_sd_basin_low"]),
                "one_sd_basin_high": float(observed_min["one_sd_basin_high"]),
                "oof_selected_weight": float(weights[oof_selected]),
                "oof_selected_predicted_bpb": float(oof[oof_selected]),
                "oof_selected_actual_bpb": float(response[oof_selected]),
                "oof_selection_regret": float(response[oof_selected] - observed_min["sampled_min_bpb"]),
                "oof_selected_in_one_sd_basin": bool(
                    observed_min["one_sd_basin_low"] - 1e-12
                    <= weights[oof_selected]
                    <= observed_min["one_sd_basin_high"] + 1e-12
                ),
                "full_fit_dense_min_weight": float(dense_weight[dense_selected]),
                "full_fit_dense_min_predicted_bpb": float(dense_prediction[dense_selected]),
            }
        )
        names = ("log_rate_nemotron", "log_rate_starcoder", "threshold_nemotron", "threshold_starcoder")
        coefficient_names = (
            "benefit_amplitude_nemotron",
            "benefit_amplitude_starcoder",
            "harm_amplitude_nemotron",
            "harm_amplitude_starcoder",
        )
        parameter_rows.append(
            {
                "token_budget_requested": int(budget),
                "parameter": "intercept",
                "value": float(intercept),
            }
        )
        parameter_rows.extend(
            {"token_budget_requested": int(budget), "parameter": name, "value": float(value)}
            for name, value in zip(names, vector, strict=True)
        )
        parameter_rows.extend(
            {"token_budget_requested": int(budget), "parameter": name, "value": float(value)}
            for name, value in zip(coefficient_names, coefficients, strict=True)
        )

    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(dense_rows),
        pd.DataFrame(metric_rows),
        pd.DataFrame(parameter_rows),
    )


def load_existing_dsp_outputs(
    output_dir: Path,
    source_dir: Path,
    *,
    maxiter: int,
    restarts: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a previously completed DSP fit after validating its frozen inputs."""
    protocol_path = output_dir / "protocol.json"
    if not protocol_path.exists():
        raise ValueError(f"Cannot reuse DSP outputs without {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_hashes = {
        filename: file_sha256(source_dir / filename)
        for filename in ("tied_diagonal_observations.csv", "tied_optima.csv", "repeat_noise.csv")
    }
    if protocol.get("input_hashes") != expected_hashes:
        raise ValueError("Existing DSP outputs were fit on different input files")
    if protocol.get("maxiter") != maxiter or protocol.get("restarts") != restarts:
        raise ValueError("Existing DSP optimizer settings do not match --maxiter and --restarts")

    predictions = pd.read_csv(output_dir / "predictions.csv")
    dense = pd.read_csv(output_dir / "dense_curves.csv")
    metrics = pd.read_csv(output_dir / "metrics.csv")
    parameters = pd.read_csv(output_dir / "full_fit_parameters.csv")
    if len(predictions) != len(TOKEN_BUDGETS) * 21 or len(dense) != len(TOKEN_BUDGETS) * DENSE_POINTS:
        raise ValueError("Existing DSP outputs do not contain the complete four-curve panel")
    if set(metrics["token_budget_requested"]) != set(TOKEN_BUDGETS):
        raise ValueError("Existing DSP metrics do not contain the frozen four token budgets")
    return predictions, dense, metrics, parameters


def add_olmix_overlays(
    observations: pd.DataFrame,
    predictions: pd.DataFrame,
    dense: pd.DataFrame,
    metrics: pd.DataFrame,
    parameters: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit exact full-data OLMix log-linear curves on the same four panels."""
    predictions = predictions.drop(columns=[column for column in predictions if column.startswith("olmix_")]).copy()
    dense = dense.drop(columns=[column for column in dense if column.startswith("olmix_")]).copy()
    metrics = metrics.drop(columns=[column for column in metrics if column.startswith("olmix_")]).copy()
    parameters = parameters.loc[~parameters["parameter"].str.startswith("olmix_")].copy()
    predictions["olmix_full_fit_prediction_bpb"] = np.nan
    dense["olmix_full_fit_prediction_bpb"] = np.nan
    parameter_rows: list[dict[str, float | int | str]] = []

    for budget, unsorted_group in observations.groupby("token_budget_requested", sort=True):
        group = unsorted_group.sort_values("weight")
        weights = group["weight"].to_numpy(float)
        response = group["starcoder_bpb"].to_numpy(float)
        fit = olmix_loglinear.fit_olmix_loglinear_model(
            tied_weights(weights),
            response,
            seed=FOLD_SEED + int(budget // 1_000_000),
            n_starts=olmix_loglinear.FIT_N_STARTS,
        )
        fitted_at_observed = fit.predict(tied_weights(weights))

        point_mask = predictions["token_budget_requested"].eq(budget)
        point_weights = predictions.loc[point_mask, "weight"].to_numpy(float)
        predictions.loc[point_mask, "olmix_full_fit_prediction_bpb"] = fit.predict(tied_weights(point_weights))

        dense_mask = dense["token_budget_requested"].eq(budget)
        dense_weights = dense.loc[dense_mask, "weight"].to_numpy(float)
        dense_prediction = fit.predict(tied_weights(dense_weights))
        dense.loc[dense_mask, "olmix_full_fit_prediction_bpb"] = dense_prediction

        metric_rows = metrics.index[metrics["token_budget_requested"].eq(budget)]
        if len(metric_rows) != 1:
            raise ValueError(f"{budget}: expected exactly one DSP metric row")
        metric_row = metric_rows[0]
        repeat_sd = float(metrics.loc[metric_row, "repeat_sd_bpb"])
        rmse = float(np.sqrt(np.mean((fitted_at_observed - response) ** 2)))
        dense_min = int(np.argmin(dense_prediction))
        metrics.loc[metric_row, "olmix_full_fit_rmse"] = rmse
        metrics.loc[metric_row, "olmix_full_fit_rmse_over_repeat_sd"] = rmse / repeat_sd
        metrics.loc[metric_row, "olmix_full_fit_dense_min_weight"] = float(dense_weights[dense_min])
        metrics.loc[metric_row, "olmix_full_fit_dense_min_predicted_bpb"] = float(dense_prediction[dense_min])
        metrics.loc[metric_row, "olmix_huber_loss"] = fit.huber_loss

        parameter_rows.extend(
            [
                {
                    "token_budget_requested": int(budget),
                    "parameter": "olmix_log_c",
                    "value": fit.log_c,
                },
                {
                    "token_budget_requested": int(budget),
                    "parameter": "olmix_beta_nemotron",
                    "value": fit.coefficients[0],
                },
                {
                    "token_budget_requested": int(budget),
                    "parameter": "olmix_beta_starcoder",
                    "value": fit.coefficients[1],
                },
                {
                    "token_budget_requested": int(budget),
                    "parameter": "olmix_huber_loss",
                    "value": fit.huber_loss,
                },
            ]
        )

    if predictions["olmix_full_fit_prediction_bpb"].isna().any() or dense["olmix_full_fit_prediction_bpb"].isna().any():
        raise ValueError("OLMix overlay predictions are incomplete")
    parameters = pd.concat([parameters, pd.DataFrame(parameter_rows)], ignore_index=True)
    parameters = parameters.sort_values(["token_budget_requested", "parameter"]).reset_index(drop=True)
    return predictions, dense, metrics, parameters


def build_figure(
    predictions: pd.DataFrame,
    dense: pd.DataFrame,
    metrics: pd.DataFrame,
) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"{budget // 1_000_000_000}B materialized tokens" for budget in TOKEN_BUDGETS],
        horizontal_spacing=0.09,
        vertical_spacing=0.22,
    )
    for annotation in figure.layout.annotations:
        annotation.update(yshift=24)
    colors = sample_colorscale("RdYlGn_r", np.linspace(0.12, 0.88, len(TOKEN_BUDGETS)))
    starcoder_epoch_scale = epoch_accounting.SIMULATED_EPOCH_TARGET_BUDGET / epoch_accounting.STARCODER_SOURCE_TOKENS
    for index, budget in enumerate(TOKEN_BUDGETS):
        row = index // 2 + 1
        column = index % 2 + 1
        points = predictions.loc[predictions["token_budget_requested"].eq(budget)].sort_values("weight")
        curve = dense.loc[dense["token_budget_requested"].eq(budget)].sort_values("weight")
        metric = metrics.loc[metrics["token_budget_requested"].eq(budget)].iloc[0]
        color = colors[index]
        figure.add_vrect(
            x0=metric["one_sd_basin_low"],
            x1=metric["one_sd_basin_high"],
            fillcolor="#2a9d8f",
            opacity=0.10,
            line_width=0,
            row=row,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=curve["weight"],
                y=curve["full_fit_prediction_bpb"],
                mode="lines",
                line={"color": color, "width": 3.5},
                name="DSP full fit",
                legendgroup="full-fit",
                showlegend=index == 0,
                customdata=curve["weight"].to_numpy() * starcoder_epoch_scale,
                hovertemplate=(
                    "StarCoder p=%{x:.3f}<br>StarCoder materialized epochs=%{customdata:.3f}"
                    "<br>full-fit BPB=%{y:.6f}<extra></extra>"
                ),
            ),
            row=row,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=curve["weight"],
                y=curve["olmix_full_fit_prediction_bpb"],
                mode="lines",
                line={"color": "#277da1", "width": 3.0, "dash": "dash"},
                name="OLMix full fit",
                legendgroup="olmix-full-fit",
                showlegend=index == 0,
                customdata=curve["weight"].to_numpy() * starcoder_epoch_scale,
                hovertemplate=(
                    "StarCoder p=%{x:.3f}<br>StarCoder materialized epochs=%{customdata:.3f}"
                    "<br>OLMix full-fit BPB=%{y:.6f}<extra></extra>"
                ),
            ),
            row=row,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=points["weight"],
                y=points["oof_prediction_bpb"],
                mode="lines+markers",
                line={"color": "#8d99ae", "width": 1.5, "dash": "dot"},
                marker={"color": "#fbf8f0", "line": {"color": "#58657a", "width": 1.5}, "size": 8},
                name="DSP out-of-fold",
                legendgroup="oof",
                showlegend=index == 0,
                customdata=(
                    points.assign(starcoder_epochs=points["weight"] * starcoder_epoch_scale)[
                        ["starcoder_epochs", "outer_fold", "observed_bpb"]
                    ].to_numpy(dtype=object)
                ),
                hovertemplate=(
                    "StarCoder p=%{x:.2f}<br>StarCoder materialized epochs=%{customdata[0]:.3f}"
                    "<br>OOF BPB=%{y:.6f}<br>observed=%{customdata[2]:.6f}"
                    "<br>held-out fold=%{customdata[1]}<extra></extra>"
                ),
            ),
            row=row,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=points["weight"],
                y=points["observed_bpb"],
                mode="markers",
                marker={"color": "#173042", "size": 8},
                name="Observed",
                legendgroup="observed",
                showlegend=index == 0,
                customdata=(
                    points.assign(starcoder_epochs=points["weight"] * starcoder_epoch_scale)[
                        ["starcoder_epochs", "run_name", "source"]
                    ].to_numpy(dtype=object)
                ),
                hovertemplate=(
                    "StarCoder p=%{x:.2f}<br>StarCoder materialized epochs=%{customdata[0]:.3f}"
                    "<br>observed BPB=%{y:.6f}<br>%{customdata[1]}<br>%{customdata[2]}<extra></extra>"
                ),
            ),
            row=row,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=[metric["observed_grid_min_weight"]],
                y=[metric["observed_grid_min_bpb"]],
                mode="markers",
                marker={"color": "#173042", "size": 15, "symbol": "star"},
                name="Observed minimum",
                legendgroup="observed-min",
                showlegend=index == 0,
                customdata=[metric["observed_grid_min_weight"] * starcoder_epoch_scale],
                hovertemplate=(
                    "observed minimum<br>StarCoder p=%{x:.2f}"
                    "<br>StarCoder materialized epochs=%{customdata:.3f}<br>BPB=%{y:.6f}<extra></extra>"
                ),
            ),
            row=row,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=[metric["full_fit_dense_min_weight"]],
                y=[metric["full_fit_dense_min_predicted_bpb"]],
                mode="markers",
                marker={"color": "#e76f51", "size": 13, "symbol": "x", "line": {"width": 2}},
                name="DSP fitted minimum",
                legendgroup="predicted-min",
                showlegend=index == 0,
                customdata=[metric["full_fit_dense_min_weight"] * starcoder_epoch_scale],
                hovertemplate=(
                    "DSP minimum<br>StarCoder p=%{x:.3f}"
                    "<br>StarCoder materialized epochs=%{customdata:.3f}"
                    "<br>predicted BPB=%{y:.6f}<extra></extra>"
                ),
            ),
            row=row,
            col=column,
        )
        selected_point = points.loc[points["weight"].eq(metric["oof_selected_weight"])].iloc[0]
        figure.add_trace(
            go.Scatter(
                x=[metric["oof_selected_weight"]],
                y=[selected_point["observed_bpb"]],
                mode="markers",
                marker={
                    "color": "#fbf8f0",
                    "line": {"color": "#6d597a", "width": 2.5},
                    "size": 13,
                    "symbol": "diamond",
                },
                name="DSP OOF pick",
                legendgroup="oof-selected",
                showlegend=index == 0,
                customdata=[metric["oof_selected_weight"] * starcoder_epoch_scale],
                hovertemplate=(
                    "OOF-selected grid point<br>StarCoder p=%{x:.2f}"
                    "<br>StarCoder materialized epochs=%{customdata:.3f}<br>observed BPB=%{y:.6f}"
                    f"<br>regret={metric['oof_selection_regret']:+.6f}<extra></extra>"
                ),
            ),
            row=row,
            col=column,
        )
        figure.add_annotation(
            x=0.5,
            y=0.98,
            xref=f"x{index + 1 if index else ''} domain",
            yref=f"y{index + 1 if index else ''} domain",
            text=(
                f"DSP fit: {metric['full_fit_rmse']:.4f} RMSE "
                f"({metric['full_fit_rmse_over_repeat_sd']:.1f}x noise)"
                f"<br>OLMix: {metric['olmix_full_fit_rmse']:.4f} RMSE "
                f"({metric['olmix_full_fit_rmse_over_repeat_sd']:.1f}x noise)"
                f"<br>DSP OOF: {metric['oof_rmse']:.4f}; interior {metric['oof_interior_rmse']:.4f}"
                f"<br>rank rho {metric['oof_spearman']:.3f}; selected p={metric['oof_selected_weight']:.2f}"
            ),
            showarrow=False,
            align="left",
            xanchor="center",
            yanchor="top",
            bgcolor="rgba(251,248,240,0.88)",
            bordercolor="#d5cdbf",
            borderwidth=1,
            font={"size": 11, "color": "#173042"},
        )
        figure.update_xaxes(title="StarCoder fraction p", range=[-0.02, 1.02], row=row, col=column)
        figure.update_yaxes(title="Programming Languages BPB", row=row, col=column)

        base_axis_index = index + 1
        base_xaxis = "x" if base_axis_index == 1 else f"x{base_axis_index}"
        base_yaxis = "y" if base_axis_index == 1 else f"y{base_axis_index}"
        top_axis_index = len(TOKEN_BUDGETS) + base_axis_index
        top_xaxis = f"x{top_axis_index}"
        figure.update_layout(
            {
                f"xaxis{top_axis_index}": {
                    "anchor": base_yaxis,
                    "overlaying": base_xaxis,
                    "matches": base_xaxis,
                    "side": "top",
                    "range": [-0.02, 1.02],
                    "tickmode": "array",
                    "tickvals": [0.0, 0.5, 1.0],
                    "ticktext": [f"{fraction * starcoder_epoch_scale:.1f} ep" for fraction in (0.0, 0.5, 1.0)],
                    "ticks": "outside",
                    "ticklen": 4,
                    "tickfont": {"size": 11, "color": "#5f7180"},
                    "showgrid": False,
                    "zeroline": False,
                }
            }
        )
        figure.add_trace(
            go.Scatter(
                x=[0.0, 1.0],
                y=[metric["observed_grid_min_bpb"], metric["observed_grid_min_bpb"]],
                mode="markers",
                marker={"opacity": 0.0},
                hoverinfo="skip",
                showlegend=False,
                xaxis=top_xaxis,
                yaxis=base_yaxis,
            )
        )

    figure.update_layout(
        title={
            "text": (
                "StarCoder-Nemotron tied diagonal: DSP versus OLMix"
                "<br><sup>Nemotron fraction is 1-p. Solid DSP and dashed OLMix use all 21 rows; dotted markers are "
                "held-out DSP predictions.<br>Lower ticks are p; upper ticks are StarCoder materialized epochs. "
                "Green bands are frozen one-repeat-SD basins.</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_white",
        height=1200,
        margin={"l": 90, "r": 45, "t": 175, "b": 205},
        font={"family": "Avenir Next, Helvetica Neue, sans-serif", "color": "#173042", "size": 14},
        paper_bgcolor="#fbf8f0",
        plot_bgcolor="#fbf8f0",
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.16,
            "yanchor": "top",
            "entrywidth": 145,
            "entrywidthmode": "pixels",
            "bgcolor": "rgba(251,248,240,0.92)",
            "bordercolor": "#d5cdbf",
            "borderwidth": 1,
        },
        hoverlabel={"font_size": 13},
    )
    figure.update_xaxes(gridcolor="#ded8ca", zerolinecolor="#173042", automargin=True, title_standoff=12)
    figure.update_yaxes(gridcolor="#ded8ca", zerolinecolor="#173042", automargin=True, title_standoff=12)
    return figure


def write_report(output_dir: Path, metrics: pd.DataFrame) -> None:
    table = metrics.copy()
    table["budget"] = table["token_budget_requested"].map(lambda value: f"{value / 1e9:g}B")
    table["OOF RMSE"] = table["oof_rmse"].map(lambda value: f"{value:.6f}")
    table["DSP full-fit RMSE"] = table["full_fit_rmse"].map(lambda value: f"{value:.6f}")
    table["OLMix full-fit RMSE"] = table["olmix_full_fit_rmse"].map(lambda value: f"{value:.6f}")
    table["OOF interior RMSE"] = table["oof_interior_rmse"].map(lambda value: f"{value:.6f}")
    table["RMSE / repeat SD"] = table["oof_rmse_over_repeat_sd"].map(lambda value: f"{value:.2f}")
    table["OLMix RMSE / repeat SD"] = table["olmix_full_fit_rmse_over_repeat_sd"].map(lambda value: f"{value:.2f}")
    table["Spearman"] = table["oof_spearman"].map(lambda value: f"{value:.3f}")
    table["observed min p"] = table["observed_grid_min_weight"].map(lambda value: f"{value:.2f}")
    table["OLMix min p"] = table["olmix_full_fit_dense_min_weight"].map(lambda value: f"{value:.2f}")
    table["OOF selected p"] = table["oof_selected_weight"].map(lambda value: f"{value:.2f}")
    table["selection regret"] = table["oof_selection_regret"].map(lambda value: f"{value:+.6f}")
    table["inside 1-SD basin"] = table["oof_selected_in_one_sd_basin"].map(lambda value: "yes" if value else "no")
    table = table[
        [
            "budget",
            "DSP full-fit RMSE",
            "OLMix full-fit RMSE",
            "OOF RMSE",
            "OOF interior RMSE",
            "RMSE / repeat SD",
            "OLMix RMSE / repeat SD",
            "Spearman",
            "observed min p",
            "OLMix min p",
            "OOF selected p",
            "selection regret",
            "inside 1-SD basin",
        ]
    ]
    lines = [
        "# Canonical DSP versus OLMix on the StarCoder tied diagonal",
        "",
        "Each of the four 21-point curves is fit separately. The input is StarCoder fraction `p`; Nemotron is "
        "`1-p`. Canonical DSP receives the two simulated-materialized epoch exposures and fits one saturation "
        "rate, one harm threshold, one nonnegative benefit amplitude, and one nonnegative harm amplitude per "
        "bucket, plus an intercept.",
        "",
        "The OLMix overlay is the repository's exact positive log-linear law "
        "`L(p) = c + exp(beta_N * (1-p) + beta_S * p)`, fit with summed Huber loss and 48 starts. It is a "
        "descriptive full-data fit, directly comparable to the smooth full-data DSP curve. On this one-dimensional "
        "simplex edge, OLMix is necessarily monotone (or flat), so it cannot represent an interior minimum.",
        "",
        "The lower x-axis remains StarCoder fraction `p`. The synchronized upper axis reports StarCoder "
        "materialized epochs, `p * 26.4579`, under the historical simulated-support construction shared by all "
        "four token-budget panels.",
        "",
        "Primary diagnostics use one frozen interleaved five-fold partition. The smooth full-data curves in "
        "`index.html` are descriptive and are not used for the DSP out-of-fold scores or selections. No OLMix OOF "
        "claims are made in this visual capacity check.",
        "",
        table.to_markdown(index=False),
        "",
        "The selection regret is the measured BPB at the grid point selected by the 21 out-of-fold predictions "
        "minus the measured minimum on the same 21-point grid.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.maxiter < 1 or args.restarts < 1:
        raise ValueError("maxiter and restarts must be positive")
    observations, optima, noise = load_inputs(args.source_dir)
    if args.reuse_existing_dsp:
        predictions, dense, metrics, parameters = load_existing_dsp_outputs(
            args.output_dir,
            args.source_dir,
            maxiter=args.maxiter,
            restarts=args.restarts,
        )
    else:
        predictions, dense, metrics, parameters = fit_curves(
            observations,
            optima,
            noise,
            maxiter=args.maxiter,
            restarts=args.restarts,
        )
    predictions, dense, metrics, parameters = add_olmix_overlays(
        observations,
        predictions,
        dense,
        metrics,
        parameters,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    dense.to_csv(args.output_dir / "dense_curves.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    parameters.to_csv(args.output_dir / "full_fit_parameters.csv", index=False)
    fold_assignments = predictions[["token_budget_requested", "weight", "outer_fold"]]
    fold_assignments.to_csv(args.output_dir / "fold_assignments.csv", index=False)
    protocol = {
        "schema_version": 3,
        "model": "canonical single-phase DSP with exact OLMix log-linear overlay",
        "target": "Programming Languages BPB",
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "fold_seed": FOLD_SEED,
        "fold_assignment": "ordered row index modulo fold count, shared across budgets",
        "maxiter": args.maxiter,
        "restarts": args.restarts,
        "dsp_outputs_reused": args.reuse_existing_dsp,
        "olmix": {
            "law": "c + exp(beta_nemotron * (1-p) + beta_starcoder * p)",
            "loss": "summed Huber",
            "huber_delta": olmix_loglinear.DEFAULT_HUBER_DELTA,
            "starts": olmix_loglinear.FIT_N_STARTS,
            "seed": "fold_seed + token_budget_millions",
            "scope": "descriptive full-data overlay only",
        },
        "simulated_epoch_target_budget": epoch_accounting.SIMULATED_EPOCH_TARGET_BUDGET,
        "epoch_scales": {
            "nemotron": epoch_accounting.SIMULATED_EPOCH_TARGET_BUDGET / epoch_accounting.NEMOTRON_SOURCE_TOKENS,
            "starcoder": epoch_accounting.SIMULATED_EPOCH_TARGET_BUDGET / epoch_accounting.STARCODER_SOURCE_TOKENS,
        },
        "materialized_epoch_axis": {
            "lower_axis": "StarCoder fraction p",
            "upper_axis": "StarCoder materialized epochs",
            "formula": "p * simulated_epoch_target_budget / full_starcoder_source_tokens",
            "full_starcoder_source_tokens": epoch_accounting.STARCODER_SOURCE_TOKENS,
        },
        "input_hashes": {
            filename: file_sha256(args.source_dir / filename)
            for filename in ("tied_diagonal_observations.csv", "tied_optima.csv", "repeat_noise.csv")
        },
        "optimizer_sha256": file_sha256(Path(dsp_ladder.__file__).resolve()),
        "olmix_optimizer_sha256": file_sha256(Path(olmix_loglinear.__file__).resolve()),
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    figure = build_figure(predictions, dense, metrics)
    figure.write_html(args.output_dir / "index.html", include_plotlyjs=True, config=PLOTLY_CONFIG)
    write_report(args.output_dir, metrics)
    print(metrics.to_string(index=False), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
