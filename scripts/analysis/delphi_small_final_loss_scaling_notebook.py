# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
# ]
# ///

"""Marimo notebook for small-Delphi endpoint scaling diagnostics.

Run:
    uv run --with marimo marimo edit scripts/analysis/delphi_small_final_loss_scaling_notebook.py
"""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import math
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return Path, go, make_subplots, math, mo, np, pd


@app.cell
def _(Path, mo, pd):
    output_dir = Path("midtrain_analysis_outputs/small_final_loss_scaling")
    endpoints_path = output_dir / "endpoints.csv"
    fits_path = output_dir / "fit_summary.csv"
    targets_path = output_dir / "extrapolation_targets.csv"
    predictions_path = output_dir / "extrapolation_predictions.csv"
    trajectory_points_path = output_dir / "trajectory_points.csv"
    trajectory_predictions_path = output_dir / "trajectory_prefix_predictions.csv"
    trajectory_summary_path = output_dir / "trajectory_prefix_summary.csv"
    trajectory_selection_path = output_dir / "trajectory_method_selection.csv"

    mo.stop(
        not endpoints_path.exists() or not fits_path.exists(),
        mo.md(
            f"""
            Missing endpoint-fit outputs under `{output_dir}`.

            Run:

            ```bash
            uv run python scripts/analysis/delphi_small_final_loss_scaling.py
            ```
            """
        ),
    )

    endpoints_raw = pd.read_csv(endpoints_path, dtype={"scale": str, "lr": str})
    fits_raw = pd.read_csv(fits_path, dtype={"lr": str})
    targets_raw = pd.read_csv(targets_path, dtype={"scale": str, "lr": str}) if targets_path.exists() else pd.DataFrame()
    predictions_raw = (
        pd.read_csv(predictions_path, dtype={"target_scale": str, "lr": str})
        if predictions_path.exists()
        else pd.DataFrame()
    )
    trajectory_points_raw = (
        pd.read_csv(trajectory_points_path, dtype={"scale": str, "lr": str})
        if trajectory_points_path.exists()
        else pd.DataFrame()
    )
    trajectory_predictions_raw = (
        pd.read_csv(trajectory_predictions_path, dtype={"scale": str, "lr": str})
        if trajectory_predictions_path.exists()
        else pd.DataFrame()
    )
    trajectory_summary_raw = pd.read_csv(trajectory_summary_path) if trajectory_summary_path.exists() else pd.DataFrame()
    trajectory_selection_raw = (
        pd.read_csv(trajectory_selection_path) if trajectory_selection_path.exists() else pd.DataFrame()
    )

    fit_scale_order = ["3e18", "9e18", "2e19", "3e19", "9e19", "2e20"]
    heldout_scale_order = ["1e21", "1e22"]
    scale_order = fit_scale_order + heldout_scale_order
    mix_order = ["p33m67", "p50m50", "p67m33"]
    lr_order = ["33", "50", "67", "83"]
    fit_scale_flops = {
        "3e18": 3e18,
        "9e18": 9e18,
        "2e19": 2e19,
        "3e19": 3e19,
        "9e19": 9e19,
        "2e20": 2e20,
    }
    metric_options = {
        "Held-out math validation": "math_val_loss",
        "Aggregate eval/loss": "eval_loss",
        "Paloma macro": "paloma_macro_loss",
        "Paloma C4": "paloma_c4_loss",
        "Train loss": "train_loss",
    }
    fit_options = {
        "log(loss) vs log(compute)": "log_loss_vs_log_compute",
        "floor + A * compute^-alpha": "floor_plus_power",
    }
    return (
        endpoints_raw,
        fit_options,
        fit_scale_flops,
        fits_raw,
        heldout_scale_order,
        lr_order,
        metric_options,
        mix_order,
        output_dir,
        predictions_raw,
        scale_order,
        targets_raw,
        trajectory_points_raw,
        trajectory_predictions_raw,
        trajectory_selection_raw,
        trajectory_summary_raw,
    )


@app.cell
def _(mo, output_dir):
    mo.md(
        f"""
    # Small Delphi Endpoint Scaling

    This notebook is for judging whether final validation losses look
    scale-law-like before we fit full trajectories.

    Data source: `{output_dir}`.
    """
    )
    return


@app.cell
def _(endpoints_raw, mo, pd, scale_order):
    coverage_cells = endpoints_raw[["scale", "mix", "lr", "complete"]].drop_duplicates()
    coverage_cells["scale"] = pd.Categorical(coverage_cells["scale"], categories=scale_order, ordered=True)
    coverage_table = (
        coverage_cells.groupby("scale", observed=True)["complete"]
        .agg(complete="sum", selected_cells="count")
        .reset_index()
    )
    coverage_table["coverage"] = (
        coverage_table["complete"].astype(str) + "/" + coverage_table["selected_cells"].astype(str)
    )
    mo.md("## Coverage")
    return (coverage_table,)


@app.cell
def _(coverage_table):
    coverage_table[["scale", "coverage"]]
    return


@app.cell
def _(heldout_scale_order, mo, pd, targets_raw):
    if targets_raw.empty:
        target_coverage_table = pd.DataFrame(columns=["scale", "coverage"])
    else:
        target_cells = targets_raw[["scale", "mix", "lr", "complete"]].drop_duplicates().copy()
        target_cells["scale"] = pd.Categorical(target_cells["scale"], categories=heldout_scale_order, ordered=True)
        target_coverage_table = (
            target_cells.groupby("scale", observed=True)["complete"]
            .agg(complete="sum", selected_cells="count")
            .reset_index()
        )
        target_coverage_table["coverage"] = (
            target_coverage_table["complete"].astype(str) + "/" + target_coverage_table["selected_cells"].astype(str)
        )
    mo.md("## Held-Out Target Coverage")
    return (target_coverage_table,)


@app.cell
def _(target_coverage_table):
    target_coverage_table[["scale", "coverage"]]
    return


@app.cell
def _(fit_options, metric_options, mix_order, mo):
    metric_select = mo.ui.dropdown(
        options=list(metric_options),
        value="Held-out math validation",
        label="metric",
    )
    mix_select = mo.ui.dropdown(
        options=mix_order,
        value="p33m67",
        label="mix for detailed plot",
    )
    fit_select = mo.ui.radio(
        options=list(fit_options),
        value="log(loss) vs log(compute)",
        label="fit",
    )
    lr33 = mo.ui.checkbox(value=True, label="lr33")
    lr50 = mo.ui.checkbox(value=True, label="lr50")
    lr67 = mo.ui.checkbox(value=True, label="lr67")
    lr83 = mo.ui.checkbox(value=True, label="lr83")
    show_targets = mo.ui.checkbox(value=True, label="show 1e21/1e22")
    lr_controls = mo.vstack([lr33, lr50, lr67, lr83], gap=0.4)
    controls = mo.hstack(
        [
            metric_select,
            mix_select,
            fit_select,
            mo.vstack([mo.md("**learning rates**"), lr_controls], gap=0.2),
            mo.vstack([mo.md("**held-out**"), show_targets], gap=0.2),
        ],
        justify="start",
        gap=2,
    )
    controls  # noqa: B018
    return (
        fit_select,
        lr33,
        lr50,
        lr67,
        lr83,
        metric_select,
        mix_select,
        show_targets,
    )


@app.cell
def _(fit_options, fit_select, metric_options, metric_select):
    selected_metric_label = metric_options[metric_select.value]
    selected_fit_kind = fit_options[fit_select.value]
    return selected_fit_kind, selected_metric_label


@app.cell
def _(lr33, lr50, lr67, lr83):
    selected_lrs = tuple(
        lr
        for lr, enabled in (
            ("33", lr33.value),
            ("50", lr50.value),
            ("67", lr67.value),
            ("83", lr83.value),
        )
        if enabled
    )
    return (selected_lrs,)


@app.cell
def _(mo):
    mo.md(
        """
    ## Fit Quality Readout

    Start with the log-linear fit. The two most useful columns are:

    - `RMSE`: in raw loss units, on the same scale as the plotted y-axis.
    - `LOOCV RMSE`: leave-one-scale-out endpoint prediction error. This is
      the cleaner "would it predict an unseen scale?" diagnostic.

    `R2` is useful, but with monotone smooth scaling curves it can look
    good even when the extrapolation error is not acceptable.
    """
    )
    return


@app.cell
def _(fits_raw, pd, selected_fit_kind, selected_metric_label):
    selected_fit_table = fits_raw[
        fits_raw["metric_label"].eq(selected_metric_label) & fits_raw["fit_kind"].eq(selected_fit_kind)
    ].copy()
    selected_fit_table["recipe"] = selected_fit_table["mix"] + "-lr" + selected_fit_table["lr"].astype(str)
    selected_fit_table = selected_fit_table.sort_values(["loocv_rmse", "rmse"], na_position="last")
    selected_fit_view = selected_fit_table[
        [
            "recipe",
            "n",
            "scales",
            "exponent",
            "floor",
            "r2",
            "rmse",
            "loocv_rmse",
            "monotone_nonincreasing",
            "first_value",
            "last_value",
        ]
    ].copy()
    for _column in ["exponent", "floor", "r2", "rmse", "loocv_rmse", "first_value", "last_value"]:
        selected_fit_view[_column] = selected_fit_view[_column].map(
            lambda value: None if pd.isna(value) else round(value, 5)
        )
    selected_fit_view  # noqa: B018
    return (selected_fit_table,)


@app.cell
def _(
    endpoints_raw,
    fit_scale_flops,
    go,
    lr_order,
    make_subplots,
    mix_select,
    np,
    selected_fit_table,
    selected_lrs,
    selected_metric_label,
    show_targets,
    targets_raw,
):
    def predict_from_fit(fit_row, flops_values):
        compute_x = np.asarray(flops_values, dtype=float) / 1e18
        if fit_row["fit_kind"] == "log_loss_vs_log_compute":
            return fit_row["amplitude"] * np.power(compute_x, fit_row["exponent"])
        return fit_row["floor"] + fit_row["amplitude"] * np.power(compute_x, -fit_row["exponent"])

    def endpoint_residual_figure(endpoints, fits, targets, metric_label, mix_name, selected_lrs, show_heldout):
        metric_df = endpoints[endpoints["metric_label"].eq(metric_label)].copy()
        metric_df = metric_df[metric_df["mix"].eq(mix_name)]
        target_df = targets[targets["metric_label"].eq(metric_label)].copy() if not targets.empty else targets
        if not target_df.empty:
            target_df = target_df[target_df["mix"].eq(mix_name)]
        lr_colors = {"33": "#4C78A8", "50": "#F58518", "67": "#54A24B", "83": "#E45756"}
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.68, 0.32],
            subplot_titles=("Endpoint loss with fitted scaling curves", "Residuals: observed - fitted"),
        )
        x_max = max(fit_scale_flops.values())
        if show_heldout and not target_df.empty:
            x_max = max(x_max, target_df["scale_flops"].max())
        x_grid = np.geomspace(min(fit_scale_flops.values()), x_max, 240)
        for lr in lr_order:
            if lr not in selected_lrs:
                continue
            recipe_df = metric_df[metric_df["lr"].eq(lr)].sort_values("scale_flops")
            target_recipe_df = (
                target_df[target_df["lr"].eq(lr)].sort_values("scale_flops")
                if show_heldout and not target_df.empty
                else target_df.iloc[0:0]
            )
            if recipe_df.empty and target_recipe_df.empty:
                continue
            complete_df = recipe_df[recipe_df["complete"]]
            partial_df = recipe_df[~recipe_df["complete"]]
            name = f"{mix_name}-lr{lr}"
            color = lr_colors[lr]
            if not complete_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=complete_df["scale_flops"],
                        y=complete_df["value"],
                        mode="markers",
                        name=name,
                        legendgroup=name,
                        marker={"size": 9, "color": color},
                        customdata=complete_df[["scale", "run_name", "global_step", "expected_steps"]].to_numpy(),
                        hovertemplate=(
                            "%{customdata[0]}<br>%{customdata[1]}<br>"
                            "step=%{customdata[2]}/%{customdata[3]}<br>"
                            "loss=%{y:.5f}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )
            if not partial_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=partial_df["scale_flops"],
                        y=partial_df["value"],
                        mode="markers",
                        name=f"{name} partial",
                        legendgroup=name,
                        marker={"size": 10, "symbol": "x-open", "color": color},
                        showlegend=False,
                        customdata=partial_df[["scale", "run_name", "global_step", "expected_steps"]].to_numpy(),
                        hovertemplate=(
                            "%{customdata[0]} partial<br>%{customdata[1]}<br>"
                            "step=%{customdata[2]}/%{customdata[3]}<br>"
                            "current loss=%{y:.5f}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )
            fit_row = fits[fits["mix"].eq(mix_name) & fits["lr"].eq(lr)]
            if fit_row.empty:
                continue
            fit_row = fit_row.iloc[0]
            fit_y = predict_from_fit(fit_row, x_grid)
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=fit_y,
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    line={"color": color},
                    showlegend=False,
                    hovertemplate=f"{name}<br>fit=%{{y:.5f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            if not complete_df.empty:
                fitted_at_points = predict_from_fit(fit_row, complete_df["scale_flops"].to_numpy())
                residuals = complete_df["value"].to_numpy() - fitted_at_points
                fig.add_trace(
                    go.Scatter(
                        x=complete_df["scale_flops"],
                        y=residuals,
                        mode="markers+lines",
                        name=f"{name} residual",
                        legendgroup=name,
                        marker={"color": color},
                        line={"color": color},
                        showlegend=False,
                        customdata=complete_df[["scale"]].to_numpy(),
                        hovertemplate="%{customdata[0]}<br>residual=%{y:.5f}<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )
            if not target_recipe_df.empty:
                target_fitted = predict_from_fit(fit_row, target_recipe_df["scale_flops"].to_numpy())
                target_residuals = target_recipe_df["value"].to_numpy() - target_fitted
                target_symbols = np.where(target_recipe_df["complete"], "diamond", "x-open")
                fig.add_trace(
                    go.Scatter(
                        x=target_recipe_df["scale_flops"],
                        y=target_recipe_df["value"],
                        mode="markers",
                        name=f"{name} held-out",
                        legendgroup=name,
                        marker={"size": 12, "symbol": target_symbols, "color": color},
                        showlegend=False,
                        customdata=target_recipe_df[
                            ["scale", "target_kind", "run_name", "global_step", "expected_steps", "progress"]
                        ].to_numpy(),
                        hovertemplate=(
                            "%{customdata[0]} %{customdata[1]}<br>%{customdata[2]}<br>"
                            "step=%{customdata[3]}/%{customdata[4]} "
                            "(%{customdata[5]:.1%})<br>"
                            "observed=%{y:.5f}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=target_recipe_df["scale_flops"],
                        y=target_residuals,
                        mode="markers",
                        name=f"{name} held-out residual",
                        legendgroup=name,
                        marker={"size": 11, "symbol": target_symbols, "color": color},
                        showlegend=False,
                        customdata=target_recipe_df[["scale", "target_kind"]].to_numpy(),
                        hovertemplate="%{customdata[0]} %{customdata[1]}<br>residual=%{y:.5f}<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )
        for row in [1, 2]:
            fig.add_vline(x=max(fit_scale_flops.values()), line_dash="dot", line_color="#777", row=row, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="#666", row=2, col=1)
        fig.update_xaxes(type="log", title_text="isoflop scale", row=2, col=1)
        fig.update_yaxes(title_text=metric_label, row=1, col=1)
        fig.update_yaxes(title_text="loss residual", row=2, col=1)
        fig.update_layout(
            height=760,
            title=f"{metric_label}: {mix_name}",
            legend_title_text="recipe",
            showlegend=True,
            margin={"l": 60, "r": 30, "t": 90, "b": 50},
        )
        return fig

    selected_endpoint_figure = endpoint_residual_figure(
        endpoints_raw,
        selected_fit_table,
        targets_raw,
        selected_metric_label,
        mix_select.value,
        selected_lrs,
        show_targets.value,
    )
    selected_endpoint_figure  # noqa: B018
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Held-Out Extrapolation Error

    These rows are not included in the fit. They evaluate the selected
    small-ladder curve at `1e21` and `1e22`, then compare against the local
    Delphi midtraining endpoints. `best_prefix` means the target run did not
    reach the final planned step.
    """
    )
    return


@app.cell
def _(
    mix_select,
    pd,
    predictions_raw,
    selected_fit_kind,
    selected_lrs,
    selected_metric_label,
):
    if predictions_raw.empty:
        selected_prediction_view = pd.DataFrame()
    else:
        selected_predictions = predictions_raw[
            predictions_raw["metric_label"].eq(selected_metric_label)
            & predictions_raw["fit_kind"].eq(selected_fit_kind)
            & predictions_raw["mix"].eq(mix_select.value)
            & predictions_raw["lr"].isin(selected_lrs)
        ].copy()
        selected_predictions = selected_predictions.sort_values(["target_scale_flops", "lr"])
        selected_prediction_view = selected_predictions[
            [
                "target_scale",
                "recipe",
                "target_kind",
                "observed",
                "predicted",
                "error",
                "abs_error",
                "pct_error",
                "target_progress",
                "train_scales",
            ]
        ].copy()
        for _column in ["observed", "predicted", "error", "abs_error"]:
            selected_prediction_view[_column] = selected_prediction_view[_column].map(
                lambda value: None if pd.isna(value) else round(value, 5)
            )
        for _column in ["pct_error", "target_progress"]:
            selected_prediction_view[_column] = selected_prediction_view[_column].map(
                lambda value: None if pd.isna(value) else round(value, 3)
            )
    selected_prediction_view  # noqa: B018
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Within-Run Prefix Prediction

    This section asks a different question from the endpoint scaling fit:
    given only the first `x%` of a run, how well can we predict its final
    validation loss? Methods are tuned on completed small-ladder runs through
    `2e20`; `1e21` and `1e22` are held out.
    """
    )
    return


@app.cell
def _(pd, trajectory_selection_raw):
    if trajectory_selection_raw.empty:
        trajectory_selection_view = pd.DataFrame()
    else:
        trajectory_selection_view = trajectory_selection_raw[
            [
                "metric_label",
                "selected_method",
                "selected_prefix",
                "small_cv_mean_abs_error",
                "small_cv_best_mean_abs_error",
                "heldout_complete_n",
                "heldout_complete_mean_abs_error",
                "heldout_complete_mean_abs_pct_error",
            ]
        ].copy()
        for _column in [
            "selected_prefix",
            "small_cv_mean_abs_error",
            "small_cv_best_mean_abs_error",
            "heldout_complete_mean_abs_error",
            "heldout_complete_mean_abs_pct_error",
        ]:
            trajectory_selection_view[_column] = trajectory_selection_view[_column].map(
                lambda value: None if pd.isna(value) else round(value, 5)
            )
    trajectory_selection_view  # noqa: B018
    return


@app.cell
def _(mo, trajectory_predictions_raw):
    if trajectory_predictions_raw.empty:
        prefix_labels = ["0.10"]
        trajectory_method_options = ["template_by_recipe"]
    else:
        prefix_values = sorted(float(value) for value in trajectory_predictions_raw["prefix"].dropna().unique())
        prefix_labels = [f"{value:.2f}" for value in prefix_values]
        trajectory_method_options = sorted(trajectory_predictions_raw["method"].dropna().unique())
    prefix_default = "0.10" if "0.10" in prefix_labels else prefix_labels[0]
    method_default = (
        "template_by_recipe" if "template_by_recipe" in trajectory_method_options else trajectory_method_options[0]
    )
    trajectory_prefix_select = mo.ui.dropdown(
        options=prefix_labels,
        value=prefix_default,
        label="prefix fraction",
    )
    trajectory_method_select = mo.ui.dropdown(
        options=trajectory_method_options,
        value=method_default,
        label="prediction method",
    )
    mo.hstack([trajectory_prefix_select, trajectory_method_select], justify="start", gap=2)
    return trajectory_method_select, trajectory_prefix_select


@app.cell
def _(trajectory_method_select, trajectory_prefix_select):
    selected_trajectory_method = trajectory_method_select.value
    selected_trajectory_prefix = float(trajectory_prefix_select.value)
    return selected_trajectory_method, selected_trajectory_prefix


@app.cell
def _(
    pd,
    selected_metric_label,
    selected_trajectory_method,
    trajectory_summary_raw,
):
    if trajectory_summary_raw.empty:
        trajectory_tradeoff_view = pd.DataFrame()
    else:
        trajectory_tradeoff_view = trajectory_summary_raw[
            trajectory_summary_raw["metric_label"].eq(selected_metric_label)
            & trajectory_summary_raw["target_kind"].eq("complete")
            & trajectory_summary_raw["method"].eq(selected_trajectory_method)
        ].copy()
        trajectory_tradeoff_view = trajectory_tradeoff_view[
            [
                "eval_split",
                "prefix",
                "n",
                "mean_abs_error",
                "median_abs_error",
                "max_abs_error",
                "mean_abs_pct_error",
                "median_prefix_actual_tau",
            ]
        ].sort_values(["prefix", "eval_split"])
        for _column in [
            "prefix",
            "mean_abs_error",
            "median_abs_error",
            "max_abs_error",
            "mean_abs_pct_error",
            "median_prefix_actual_tau",
        ]:
            trajectory_tradeoff_view[_column] = trajectory_tradeoff_view[_column].map(
                lambda value: None if pd.isna(value) else round(value, 5)
            )
    trajectory_tradeoff_view  # noqa: B018
    return


@app.cell
def _(
    mix_select,
    pd,
    selected_lrs,
    selected_metric_label,
    selected_trajectory_method,
    selected_trajectory_prefix,
    trajectory_predictions_raw,
):
    if trajectory_predictions_raw.empty:
        trajectory_prediction_detail_view = pd.DataFrame()
    else:
        trajectory_prediction_detail_view = trajectory_predictions_raw[
            trajectory_predictions_raw["metric_label"].eq(selected_metric_label)
            & trajectory_predictions_raw["mix"].eq(mix_select.value)
            & trajectory_predictions_raw["lr"].isin(selected_lrs)
            & trajectory_predictions_raw["method"].eq(selected_trajectory_method)
            & trajectory_predictions_raw["prefix"].eq(selected_trajectory_prefix)
        ].copy()
        trajectory_prediction_detail_view = trajectory_prediction_detail_view.sort_values(
            ["eval_split", "scale", "lr", "complete"],
            ascending=[True, True, True, False],
        )
        trajectory_prediction_detail_view = trajectory_prediction_detail_view[
            [
                "eval_split",
                "scale",
                "recipe",
                "target_kind",
                "prefix_actual_tau",
                "target",
                "predicted",
                "error",
                "abs_error",
                "pct_error",
                "fit_n",
            ]
        ]
        for _column in ["prefix_actual_tau", "target", "predicted", "error", "abs_error", "pct_error"]:
            trajectory_prediction_detail_view[_column] = trajectory_prediction_detail_view[_column].map(
                lambda value: None if pd.isna(value) else round(value, 5)
            )
    trajectory_prediction_detail_view  # noqa: B018
    return


@app.cell
def _(
    go,
    mix_select,
    selected_lrs,
    selected_metric_label,
    selected_trajectory_method,
    selected_trajectory_prefix,
    trajectory_points_raw,
    trajectory_predictions_raw,
):
    def within_run_prediction_figure(points, predictions):
        fig = go.Figure()
        if points.empty or predictions.empty:
            fig.add_annotation(
                text="Run delphi_within_run_prediction.py to generate trajectory predictions.",
                showarrow=False,
            )
            return fig

        lr_colors = {"33": "#4C78A8", "50": "#F58518", "67": "#54A24B", "83": "#E45756"}
        plot_points = points[
            points["metric_label"].eq(selected_metric_label)
            & points["mix"].eq(mix_select.value)
            & points["lr"].isin(selected_lrs)
        ].copy()
        plot_predictions = predictions[
            predictions["metric_label"].eq(selected_metric_label)
            & predictions["mix"].eq(mix_select.value)
            & predictions["lr"].isin(selected_lrs)
            & predictions["method"].eq(selected_trajectory_method)
            & predictions["prefix"].eq(selected_trajectory_prefix)
        ].copy()
        if plot_points.empty:
            fig.add_annotation(text="No trajectory points for this selection.", showarrow=False)
            return fig

        seen_line_legends = set()
        for _, group in plot_points.groupby(["scale", "lr", "run_id"], observed=True, sort=False):
            group = group.sort_values("tau")
            lr = str(group.iloc[0]["lr"])
            eval_split = str(group.iloc[0]["eval_split"])
            is_heldout = eval_split == "heldout_large"
            legend_key = (lr, eval_split)
            fig.add_trace(
                go.Scatter(
                    x=group["tau"],
                    y=group["value"],
                    mode="lines",
                    name=f"lr{lr} {'held-out' if is_heldout else 'small'}",
                    legendgroup=f"lr{lr}-{eval_split}",
                    showlegend=legend_key not in seen_line_legends,
                    line={
                        "color": lr_colors.get(lr, "#666"),
                        "width": 3 if is_heldout else 1,
                        "dash": "solid" if is_heldout else "dot",
                    },
                    opacity=0.95 if is_heldout else 0.35,
                    customdata=group[["scale", "run_name", "step", "final_step"]].to_numpy(),
                    hovertemplate=(
                        "%{customdata[0]}<br>%{customdata[1]}<br>"
                        "step=%{customdata[2]}/%{customdata[3]}<br>"
                        "tau=%{x:.3f}<br>loss=%{y:.5f}<extra></extra>"
                    ),
                )
            )
            seen_line_legends.add(legend_key)

        if not plot_predictions.empty:
            for lr, group in plot_predictions.groupby("lr", observed=True):
                color = lr_colors.get(str(lr), "#666")
                fig.add_trace(
                    go.Scatter(
                        x=[1.0] * len(group),
                        y=group["target"],
                        mode="markers",
                        name=f"lr{lr} observed final",
                        legendgroup=f"lr{lr}-markers",
                        marker={"symbol": "diamond", "size": 9, "color": color},
                        customdata=group[["scale", "target_kind", "run_name"]].to_numpy(),
                        hovertemplate=(
                            "%{customdata[0]} %{customdata[1]}<br>%{customdata[2]}<br>"
                            "observed final=%{y:.5f}<extra></extra>"
                        ),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[1.0] * len(group),
                        y=group["predicted"],
                        mode="markers",
                        name=f"lr{lr} predicted final",
                        legendgroup=f"lr{lr}-markers",
                        marker={"symbol": "x", "size": 11, "color": color, "line": {"width": 2}},
                        customdata=group[["scale", "target_kind", "run_name", "error", "abs_error"]].to_numpy(),
                        hovertemplate=(
                            "%{customdata[0]} %{customdata[1]}<br>%{customdata[2]}<br>"
                            "predicted=%{y:.5f}<br>"
                            "error=%{customdata[3]:.5f}<br>"
                            "abs error=%{customdata[4]:.5f}<extra></extra>"
                        ),
                    )
                )

        fig.add_vline(
            x=selected_trajectory_prefix,
            line_dash="dot",
            line_color="#555",
            annotation_text=f"prefix {selected_trajectory_prefix:.0%}",
            annotation_position="top",
        )
        fig.update_xaxes(range=[0, 1.03], title_text="normalized training progress")
        fig.update_yaxes(title_text=selected_metric_label)
        fig.update_layout(
            height=620,
            title=f"Within-run prediction: {selected_metric_label}, {mix_select.value}, {selected_trajectory_method}",
            legend_title_text="trajectory / marker",
            margin={"l": 60, "r": 30, "t": 80, "b": 50},
        )
        return fig

    trajectory_prediction_figure = within_run_prediction_figure(trajectory_points_raw, trajectory_predictions_raw)
    trajectory_prediction_figure  # noqa: B018
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Leave-One-Scale-Out Diagnostics

    Each point below is a fit trained on all other completed scales for the
    same recipe, then evaluated on the held-out scale. This is more useful
    than an in-sample line overlay for judging whether the curve has real
    predictive value.
    """
    )
    return


@app.cell
def _(endpoints_raw, math, np, pd, selected_fit_kind, selected_metric_label):
    def log_linear_loo_records(endpoints, metric_label):
        records = []
        metric_df = endpoints[endpoints["metric_label"].eq(metric_label) & endpoints["complete"]].copy()
        for (mix, lr), group in metric_df.groupby(["mix", "lr"], observed=True):
            group = group.sort_values("scale_flops")
            if len(group) < 3:
                continue
            x_all = group["scale_flops"].to_numpy(dtype=float) / 1e18
            y_all = group["value"].to_numpy(dtype=float)
            if np.any(y_all <= 0):
                continue
            for row_index in range(len(group)):
                train_mask = np.ones(len(group), dtype=bool)
                train_mask[row_index] = False
                coeffs = np.polyfit(np.log(x_all[train_mask]), np.log(y_all[train_mask]), deg=1)
                pred = float(math.exp(coeffs[1] + coeffs[0] * math.log(x_all[row_index])))
                observed = float(y_all[row_index])
                row = group.iloc[row_index]
                records.append(
                    {
                        "mix": mix,
                        "lr": lr,
                        "recipe": f"{mix}-lr{lr}",
                        "scale": row["scale"],
                        "scale_flops": row["scale_flops"],
                        "observed": observed,
                        "predicted": pred,
                        "error": observed - pred,
                        "abs_error": abs(observed - pred),
                        "pct_error": 100 * (observed - pred) / observed,
                    }
                )
        return pd.DataFrame(records)

    loo_records = (
        log_linear_loo_records(endpoints_raw, selected_metric_label)
        if selected_fit_kind == "log_loss_vs_log_compute"
        else pd.DataFrame()
    )
    return (loo_records,)


@app.cell
def _(go, loo_records, selected_metric_label):
    if loo_records.empty:
        loo_figure = go.Figure()
        loo_figure.add_annotation(text="LOOCV view is implemented for the log-linear fit.", showarrow=False)
    else:
        loo_figure = go.Figure()
        for recipe, group in loo_records.groupby("recipe", observed=True):
            loo_figure.add_trace(
                go.Scatter(
                    x=group["observed"],
                    y=group["predicted"],
                    mode="markers",
                    name=recipe,
                    customdata=group[["scale", "error", "pct_error"]].to_numpy(),
                    hovertemplate=(
                        "%{customdata[0]}<br>"
                        "observed=%{x:.5f}<br>predicted=%{y:.5f}<br>"
                        "error=%{customdata[1]:.5f}<br>"
                        "pct error=%{customdata[2]:.2f}%<extra></extra>"
                    ),
                )
            )
        lo = min(loo_records["observed"].min(), loo_records["predicted"].min())
        hi = max(loo_records["observed"].max(), loo_records["predicted"].max())
        loo_figure.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                name="perfect prediction",
                line={"dash": "dot", "color": "#444"},
            )
        )
        loo_figure.update_layout(
            height=560,
            title=f"Leave-one-scale-out predictions: {selected_metric_label}",
            xaxis_title="observed endpoint loss",
            yaxis_title="predicted endpoint loss",
            legend_title_text="recipe",
        )
    loo_figure  # noqa: B018
    return


@app.cell
def _(loo_records):
    if loo_records.empty:
        loo_summary = loo_records
    else:
        loo_summary = (
            loo_records.groupby("recipe", observed=True)
            .agg(
                n=("abs_error", "size"),
                mean_abs_error=("abs_error", "mean"),
                max_abs_error=("abs_error", "max"),
                mean_abs_pct_error=("pct_error", lambda values: values.abs().mean()),
            )
            .reset_index()
            .sort_values("mean_abs_error")
        )
        for _column in ["mean_abs_error", "max_abs_error", "mean_abs_pct_error"]:
            loo_summary[_column] = loo_summary[_column].round(5)
    loo_summary  # noqa: B018
    return


@app.cell
def _(fits_raw, go, lr_order, mix_order, np, selected_metric_label):
    heatmap_df = fits_raw[
        fits_raw["metric_label"].eq(selected_metric_label) & fits_raw["fit_kind"].eq("log_loss_vs_log_compute")
    ].copy()
    heatmap_z = []
    heatmap_text = []
    for mix in mix_order:
        row_values = []
        row_text = []
        for lr in lr_order:
            row = heatmap_df[heatmap_df["mix"].eq(mix) & heatmap_df["lr"].eq(lr)]
            if row.empty:
                row_values.append(np.nan)
                row_text.append("")
            else:
                value = float(row.iloc[0]["exponent"])
                row_values.append(value)
                row_text.append(f"{value:.4f}")
        heatmap_z.append(row_values)
        heatmap_text.append(row_text)
    exponent_heatmap = go.Figure(
        data=go.Heatmap(
            z=heatmap_z,
            x=[f"lr{lr}" for lr in lr_order],
            y=mix_order,
            text=heatmap_text,
            texttemplate="%{text}",
            colorscale="RdBu",
            reversescale=True,
            colorbar={"title": "exponent"},
        )
    )
    exponent_heatmap.update_layout(
        height=360,
        title=f"Log-linear scaling exponent by recipe: {selected_metric_label}",
        xaxis_title="LR factor",
        yaxis_title="mix",
    )
    exponent_heatmap  # noqa: B018
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Reading This

    For held-out math validation, the endpoint scaling signal is strong if:

    - residuals are small compared with the differences between recipes;
    - leave-one-scale-out errors are close to the noise level we care about;
    - exponents are stable across neighboring recipes instead of jumping around.
    - held-out `1e21`/`1e22` residuals do not change sign or magnitude in a
      way that would alter the recipe choice.

    The floor-plus-power fit is included as a diagnostic. With only 5-6
    scales it can make the curve look nicer while learning a floor that is
    mostly a free parameter, so treat the log-linear fit as the first-pass
    baseline.
    """
    )
    return


if __name__ == "__main__":
    app.run()
