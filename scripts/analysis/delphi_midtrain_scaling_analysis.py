# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze Delphi/v5-isoflop midtraining W&B histories from the local dump.

This script is intentionally offline-only: it reads ``midtrain_wandb_data/`` and
does not call the W&B API. Outputs go to ``midtrain_analysis_outputs/``.

Rows logged as ``1e20`` are historical v5-isoflop runs, not canonical Delphi.
The W&B names are preserved for lookup compatibility.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_DIR = Path("midtrain_wandb_data")
OUT_DIR = Path("midtrain_analysis_outputs")
PLOTS_DIR = OUT_DIR / "plots"

CURRENT_PATTERN = re.compile(
    r"^delphi-(?P<scale>1e20|1e21|1e22)-(?P<mix>p33m67|p50m50|p67m33)-"
    r"(?P<budget>4p94b|9p25b|32p07b)-lr(?P<lr>0\.33|0\.5|0\.67|0\.83)-(?P<hash>.+)$"
)

EXPECTED_FINAL_STEP = {
    "1e20": 9412,
    "1e21": 4410,
    "1e22": 7646,
}
FINAL_STEP_TOLERANCE = 5

PRETRAIN_TOKENS_B = {
    "1e20": 24.67,
    "1e21": 46.27,
    "1e22": 160.37,
}

MODEL_FLOPS = {
    "1e20": 1e20,
    "1e21": 1e21,
    "1e22": 1e22,
}

MIDTRAIN_TOKENS_B = {
    "1e20": 4.94,
    "1e21": 9.25,
    "1e22": 32.07,
}

SCALE_ORDER = ["1e20", "1e21", "1e22"]
MIX_ORDER = ["p67m33", "p50m50", "p33m67"]
LR_ORDER = ["0.33", "0.5", "0.67", "0.83"]
MATH_METRIC = "eval/nemotron_cc_math_v1/4plus/loss"
EVAL_METRIC = "eval/loss"
PALOMA_MACRO_METRIC = "eval/paloma/macro_loss"

SCALE_CAVEAT_TEXT = (
    "Scale caveat: rows labeled `1e20` are historical v5-isoflop 3e20 runs "
    "from `isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5`, not canonical "
    "Delphi. The canonical Delphi 3e20 bucket winner is "
    "`isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6`. Treat 1e20-to-1e22 "
    "transfer claims as contaminated; the within-family Delphi comparison is "
    "1e21-to-1e22."
)

METRICS = [
    "eval/loss",
    "eval/nemotron_cc_math_v1/4plus/loss",
    "eval/paloma/macro_loss",
    "eval/paloma/c4_en/loss",
    "train/loss",
]

DISPLAY_METRIC = {
    EVAL_METRIC: "eval loss",
    MATH_METRIC: "math validation loss",
    PALOMA_MACRO_METRIC: "Paloma macro loss",
    "eval/paloma/c4_en/loss": "Paloma C4 loss",
    "train/loss": "train loss",
}

RECIPE_COLORS = {
    "p67m33 lr0.33": "#0f766e",
    "p67m33 lr0.5": "#14b8a6",
    "p67m33 lr0.67": "#2dd4bf",
    "p67m33 lr0.83": "#5eead4",
    "p50m50 lr0.33": "#334155",
    "p50m50 lr0.5": "#64748b",
    "p50m50 lr0.67": "#94a3b8",
    "p50m50 lr0.83": "#d97706",
    "p33m67 lr0.33": "#b91c1c",
    "p33m67 lr0.5": "#ea580c",
    "p33m67 lr0.67": "#f97316",
    "p33m67 lr0.83": "#f59e0b",
}

FORECAST_METHOD_TEXT = (
    "For incomplete 1e22 cells in the forecast-specific plots/tables, the forecast "
    "uses a shape-ratio rule: measure the cell's current improvement from its own "
    "step-0 baseline, estimate what fraction of final improvement is usually achieved "
    "by the same tau from completed historical v5-isoflop-3e20 and Delphi-1e21 "
    "reference curves, then divide the current "
    "improvement by that fraction and convert back to a final loss."
)

# These two current-sweep run IDs logged steps 0..799 to marin-community/marin
# before the W&B project routing fix. Later rows are in delphi-midtraining.
EARLY_MARIN_RUN_IDS = {
    "delphi-1e20-p67m33-4p94b-lr0.33-590ea1",
    "delphi-1e20-p67m33-4p94b-lr0.5-9e1229",
}


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    project: str
    run_dir: Path
    name: str
    state: str
    created_at: str
    url: str
    scale: str
    mix: str
    lr: str
    budget: str

    @property
    def cell(self) -> str:
        return f"{self.scale}-{self.mix}-lr{self.lr}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def is_complete_step(max_step: float | int | None, expected_final_step: int) -> bool:
    """Treat W&B histories that stopped within a few rows of the end as complete."""
    if max_step is None or not math.isfinite(float(max_step)):
        return False
    return int(max_step) >= expected_final_step - FINAL_STEP_TOLERANCE


def recipe_label(mix: Any, lr: Any) -> str:
    return f"{mix} lr{lr}"


def load_run_infos() -> list[RunInfo]:
    infos: list[RunInfo] = []

    def maybe_add(run_dir: Path, project: str) -> None:
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            return
        meta = read_json(metadata_path)
        run_id = str(meta["id"])
        name = str(meta.get("name") or run_id)
        match = CURRENT_PATTERN.match(name) or CURRENT_PATTERN.match(run_id)
        if match is None:
            return
        groups = match.groupdict()
        infos.append(
            RunInfo(
                run_id=run_id,
                project=project,
                run_dir=run_dir,
                name=name,
                state=str(meta.get("state", "")),
                created_at=str(meta.get("created_at", "")),
                url=str(meta.get("url", "")),
                scale=groups["scale"],
                mix=groups["mix"],
                lr=groups["lr"],
                budget=groups["budget"],
            )
        )

    for run_dir in sorted((DATA_DIR / "runs").glob("*")):
        maybe_add(run_dir, "delphi-midtraining")
    for run_dir in sorted((DATA_DIR / "projects" / "marin" / "runs").glob("*")):
        maybe_add(run_dir, "marin")
    return infos


def metric_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def load_history_long(info: RunInfo) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    history_path = info.run_dir / "history.jsonl"
    if not history_path.exists():
        return pd.DataFrame()
    with history_path.open() as f:
        for line in f:
            raw = json.loads(line)
            step = raw.get("global_step", raw.get("_step"))
            if step is None:
                continue
            try:
                step_i = int(step)
            except (TypeError, ValueError):
                continue
            for metric in METRICS:
                value = metric_value(raw, metric)
                if value is None:
                    continue
                rows.append(
                    {
                        "run_id": info.run_id,
                        "project": info.project,
                        "name": info.name,
                        "state": info.state,
                        "created_at": info.created_at,
                        "url": info.url,
                        "scale": info.scale,
                        "mix": info.mix,
                        "lr": info.lr,
                        "budget": info.budget,
                        "cell": info.cell,
                        "step": step_i,
                        "metric": metric,
                        "value": value,
                    }
                )
    return pd.DataFrame(rows)


def run_progress(info: RunInfo, history: pd.DataFrame) -> dict[str, Any]:
    summary = read_json(info.run_dir / "summary.json") if (info.run_dir / "summary.json").exists() else {}
    if history.empty:
        max_step = None
        min_step = None
    else:
        run_hist = history[history["run_id"].eq(info.run_id)]
        max_step = int(run_hist["step"].max()) if not run_hist.empty else None
        min_step = int(run_hist["step"].min()) if not run_hist.empty else None
    expected = EXPECTED_FINAL_STEP[info.scale]
    return {
        "run_id": info.run_id,
        "project": info.project,
        "name": info.name,
        "state": info.state,
        "created_at": info.created_at,
        "url": info.url,
        "scale": info.scale,
        "mix": info.mix,
        "lr": info.lr,
        "budget": info.budget,
        "cell": info.cell,
        "min_step": min_step,
        "max_step": max_step,
        "expected_final_step": expected,
        "progress": None if max_step is None else max_step / expected,
        "summary_step": summary.get("global_step", summary.get("_step")),
        "summary_eval_loss": summary.get("eval/loss"),
        "summary_train_loss": summary.get("train/loss"),
    }


def choose_best_prefix_runs(registry: pd.DataFrame) -> pd.DataFrame:
    """Mark the best available run for each current sweep cell.

    Finished complete runs win. For unfinished cells, the run with the longest
    usable history wins. This deliberately keeps crashed-but-informative prefixes
    for prediction while excluding empty failed attempts.
    """
    registry = registry.copy()
    registry["selected_finished_for_cell"] = False
    registry["best_prefix_for_cell"] = False
    for _, group in registry.groupby(["scale", "mix", "lr"], sort=False):
        usable = group[group["max_step"].notna()].copy()
        if usable.empty:
            continue
        complete = usable[usable["max_step"].combine(usable["expected_final_step"], is_complete_step)]
        if not complete.empty:
            idx = complete.sort_values(["max_step", "created_at"]).index[-1]
        else:
            idx = usable.sort_values(["max_step", "created_at"]).index[-1]
        registry.loc[idx, "best_prefix_for_cell"] = True
        if is_complete_step(registry.loc[idx, "max_step"], registry.loc[idx, "expected_final_step"]):
            registry.loc[idx, "selected_finished_for_cell"] = True
    return registry


def merge_cell_histories(history: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Merge histories into one best-observed curve per cell and metric."""
    selected_ids = set(registry.loc[registry["best_prefix_for_cell"], "run_id"])

    # Always include the accidental marin early-prefix shards before the primary
    # delphi-midtraining continuation for the same run IDs.
    early = history[history["run_id"].isin(EARLY_MARIN_RUN_IDS) & history["project"].eq("marin")]
    selected = history[history["run_id"].isin(selected_ids) & history["project"].eq("delphi-midtraining")]

    merged = pd.concat([early, selected], ignore_index=True)
    if merged.empty:
        return merged

    # Sort so delphi-midtraining overwrites marin on overlapping steps, matching
    # the README guidance.
    project_rank = {"marin": 0, "delphi-midtraining": 1}
    merged["project_rank"] = merged["project"].map(project_rank).fillna(0)
    merged = merged.sort_values(["scale", "mix", "lr", "metric", "step", "project_rank", "created_at"])
    merged = merged.drop_duplicates(["scale", "mix", "lr", "metric", "step"], keep="last")
    merged["expected_final_step"] = merged["scale"].map(EXPECTED_FINAL_STEP)
    merged["tau"] = merged["step"] / merged["expected_final_step"]
    merged["pretrain_tokens_b"] = merged["scale"].map(PRETRAIN_TOKENS_B)
    merged["midtrain_tokens_b"] = merged["scale"].map(MIDTRAIN_TOKENS_B)
    return merged.drop(columns=["project_rank"])


def add_baselines(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _keys, group in curves.groupby(["scale", "mix", "lr", "metric"], sort=False):
        group = group.sort_values("step").copy()
        baseline = float(group.iloc[0]["value"])
        final_observed = float(group.iloc[-1]["value"])
        group["baseline_value"] = baseline
        group["delta"] = group["value"] - baseline
        group["improvement"] = baseline - group["value"]
        group["final_observed_value"] = final_observed
        group["observed_final_improvement"] = baseline - final_observed
        denom = group["observed_final_improvement"].replace(0, np.nan)
        group["normalized_improvement"] = group["improvement"] / denom
        rows.append(group)
    return pd.concat(rows, ignore_index=True) if rows else curves


def interp_at_tau(group: pd.DataFrame, tau: float, column: str) -> float | None:
    clean = group.sort_values("tau")
    x = clean["tau"].to_numpy(dtype=float)
    y = clean[column].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or tau < x.min() or tau > x.max():
        return None
    return float(np.interp(tau, x, y))


def predict_with_tail_linear(group: pd.DataFrame) -> float | None:
    clean = group.sort_values("tau").tail(6)
    x = clean["tau"].to_numpy(dtype=float)
    y = clean["value"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return None
    slope, intercept = np.polyfit(x, y, deg=1)
    pred = float(slope * 1.0 + intercept)
    return pred if math.isfinite(pred) else None


def predict_with_shape_ratio(curves: pd.DataFrame, target: pd.DataFrame, metric: str) -> tuple[float | None, int]:
    row = target.sort_values("tau").iloc[-1]
    tau_current = float(row["tau"])
    current_improvement = float(row["improvement"])
    baseline = float(row["baseline_value"])
    scale = str(row["scale"])
    mix = str(row["mix"])
    lr = str(row["lr"])
    if scale != "1e22":
        return None, 0

    ratios: list[float] = []
    refs = curves[
        curves["metric"].eq(metric)
        & curves["mix"].eq(mix)
        & curves["lr"].eq(lr)
        & curves["scale"].isin(["1e20", "1e21"])
    ]
    for _, group in refs.groupby("scale", observed=True):
        if group.empty:
            continue
        final_step = EXPECTED_FINAL_STEP[str(group.iloc[0]["scale"])]
        if not is_complete_step(group["step"].max(), final_step):
            continue
        final_improvement = float(group.sort_values("step").iloc[-1]["improvement"])
        if abs(final_improvement) < 1e-8:
            continue
        improvement_at_tau = interp_at_tau(group, tau_current, "improvement")
        if improvement_at_tau is None:
            continue
        ratios.append(improvement_at_tau / final_improvement)

    # If same-cell references are missing, fall back to all completed cells at
    # the same metric. This only affects currently incomplete 1e21 reference
    # cells such as p67m33/lr0.5 and p67m33/lr0.67.
    if not ratios:
        refs = curves[curves["metric"].eq(metric) & curves["scale"].isin(["1e20", "1e21"])]
        for _, group in refs.groupby(["scale", "mix", "lr"], observed=True):
            if group.empty:
                continue
            final_step = EXPECTED_FINAL_STEP[str(group.iloc[0]["scale"])]
            if not is_complete_step(group["step"].max(), final_step):
                continue
            final_improvement = float(group.sort_values("step").iloc[-1]["improvement"])
            if abs(final_improvement) < 1e-8:
                continue
            improvement_at_tau = interp_at_tau(group, tau_current, "improvement")
            if improvement_at_tau is None:
                continue
            ratios.append(improvement_at_tau / final_improvement)

    ratios = [r for r in ratios if math.isfinite(r) and abs(r) > 1e-6]
    if not ratios:
        return None, 0
    ratio = float(np.median(ratios))
    predicted_final_improvement = current_improvement / ratio
    pred = baseline - predicted_final_improvement
    return (pred if math.isfinite(pred) else None), len(ratios)


def complete_cell_keys(curves: pd.DataFrame) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for (scale, mix, lr), group in curves.groupby(["scale", "mix", "lr"], sort=False, observed=True):
        if is_complete_step(group["step"].max(), EXPECTED_FINAL_STEP[str(scale)]):
            keys.add((str(scale), str(mix), str(lr)))
    return keys


def make_predictions(curves: pd.DataFrame, metric: str = EVAL_METRIC) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_curves = curves[curves["metric"].eq(metric)]
    completed_cells = complete_cell_keys(curves)
    for (scale, mix, lr), group in metric_curves.groupby(["scale", "mix", "lr"], sort=False, observed=True):
        if scale != "1e22":
            continue
        group = group.sort_values("step")
        if group.empty:
            continue
        last = group.iloc[-1]
        linear_pred = predict_with_tail_linear(group)
        shape_pred, n_shape = predict_with_shape_ratio(curves, group, metric)
        if (str(scale), str(mix), str(lr)) in completed_cells:
            pred_endpoint = float(last["value"])
            pred_method = "observed final"
        elif shape_pred is not None and math.isfinite(float(shape_pred)):
            pred_endpoint = float(shape_pred)
            pred_method = "shape ratio"
        elif linear_pred is not None and math.isfinite(float(linear_pred)):
            pred_endpoint = float(linear_pred)
            pred_method = "tail linear fallback"
        else:
            pred_endpoint = None
            pred_method = "none"
        rows.append(
            {
                "scale": scale,
                "mix": mix,
                "lr": lr,
                "metric": metric,
                "current_step": int(last["step"]),
                "expected_final_step": int(last["expected_final_step"]),
                "current_tau": float(last["tau"]),
                "current_value": float(last["value"]),
                "baseline_value": float(last["baseline_value"]),
                "current_improvement": float(last["improvement"]),
                "pred_tail_linear": linear_pred,
                "pred_shape_ratio": shape_pred,
                "shape_reference_count": n_shape,
                "pred_endpoint": pred_endpoint,
                "pred_method": pred_method,
            }
        )
    return pd.DataFrame(rows).sort_values(["mix", "lr"])


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["scale"] = pd.Categorical(df["scale"], SCALE_ORDER, ordered=True)
    df["mix"] = pd.Categorical(df["mix"], MIX_ORDER, ordered=True)
    df["lr"] = pd.Categorical(df["lr"], LR_ORDER, ordered=True)
    return df.sort_values(["scale", "mix", "lr"])


def write_plot(fig: go.Figure, name: str) -> None:
    path = PLOTS_DIR / name
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)


def plot_raw_curves(curves: pd.DataFrame) -> None:
    metric = EVAL_METRIC
    df = curves[curves["metric"].eq(metric)].copy()
    df["recipe"] = df.apply(lambda row: recipe_label(row["mix"], row["lr"]), axis=1)
    fig = px.line(
        ordered(df),
        x="tau",
        y="value",
        color="recipe",
        color_discrete_map=RECIPE_COLORS,
        facet_col="scale",
        hover_data=["step", "run_id", "project", "state"],
        title="Current sweep: eval/loss vs proportional progress (1e20 is v5-isoflop)",
        labels={"tau": "midtraining progress tau = step / final_step", "value": "eval/loss"},
    )
    fig.update_layout(height=520, legend_title_text="recipe")
    write_plot(fig, "current_eval_loss_by_scale.html")


def plot_metric_dashboard(curves: pd.DataFrame) -> None:
    df = curves[curves["metric"].isin([EVAL_METRIC, MATH_METRIC, "eval/paloma/macro_loss"])].copy()
    df["metric_name"] = df["metric"].map(DISPLAY_METRIC)
    df["recipe"] = df.apply(lambda row: recipe_label(row["mix"], row["lr"]), axis=1)
    fig = px.line(
        ordered(df),
        x="tau",
        y="value",
        color="recipe",
        color_discrete_map=RECIPE_COLORS,
        facet_row="metric_name",
        facet_col="scale",
        hover_data=["step", "run_id", "project"],
        title="Raw validation curves by logged scale and metric (1e20 is v5-isoflop)",
        labels={"tau": "midtraining progress", "value": "loss"},
    )
    fig.update_layout(height=950, legend_title_text="recipe")
    write_plot(fig, "raw_validation_curves_dashboard.html")


def plot_normalized_collapse(curves: pd.DataFrame) -> None:
    metric = MATH_METRIC
    df = curves[curves["metric"].eq(metric)].copy()
    # Only complete small-scale curves are valid for final-normalized collapse.
    df = df[df["scale"].isin(["1e20", "1e21"])]
    complete_keys = []
    for key, group in df.groupby(["scale", "mix", "lr"], observed=True):
        if is_complete_step(group["step"].max(), EXPECTED_FINAL_STEP[str(key[0])]):
            complete_keys.append(key)
    if not complete_keys:
        return
    complete = pd.concat([df[(df["scale"].eq(s)) & (df["mix"].eq(m)) & (df["lr"].eq(l))] for s, m, l in complete_keys])
    complete["recipe"] = complete.apply(lambda row: recipe_label(row["mix"], row["lr"]), axis=1)
    fig = px.line(
        ordered(complete),
        x="tau",
        y="normalized_improvement",
        color="recipe",
        color_discrete_map=RECIPE_COLORS,
        facet_col="scale",
        hover_data=["step", "value", "baseline_value", "improvement"],
        title="Shape-collapse check: normalized math improvement (1e20 is v5-isoflop)",
        labels={"tau": "midtraining progress", "normalized_improvement": "improvement / final improvement"},
    )
    fig.add_hline(y=1.0, line_dash="dot", opacity=0.5)
    fig.update_layout(height=520, legend_title_text="recipe")
    write_plot(fig, "normalized_math_loss_collapse_finished_1e20_1e21.html")


def plot_pareto(curves: pd.DataFrame) -> None:
    math_metric = MATH_METRIC
    paloma_metric = "eval/paloma/macro_loss"
    math_df = curves[curves["metric"].eq(math_metric)][
        ["scale", "mix", "lr", "step", "tau", "improvement", "run_id"]
    ].rename(columns={"improvement": "math_improvement"})
    paloma_df = curves[curves["metric"].eq(paloma_metric)][["scale", "mix", "lr", "step", "tau", "delta"]].rename(
        columns={"delta": "retention_damage"}
    )
    df = math_df.merge(paloma_df, on=["scale", "mix", "lr", "step", "tau"], how="inner")
    if df.empty:
        return
    df["recipe"] = df.apply(lambda row: recipe_label(row["mix"], row["lr"]), axis=1)
    fig = px.line(
        ordered(df),
        x="retention_damage",
        y="math_improvement",
        color="recipe",
        color_discrete_map=RECIPE_COLORS,
        facet_col="scale",
        hover_data=["step", "tau", "run_id"],
        title="Pareto trajectories: math improvement vs Paloma retention damage (1e20 is v5-isoflop)",
        labels={
            "retention_damage": "Paloma macro loss delta (positive = worse)",
            "math_improvement": "math validation loss improvement (positive = better)",
        },
        markers=True,
    )
    fig.add_vline(x=0, line_dash="dot", opacity=0.4)
    fig.add_hline(y=0, line_dash="dot", opacity=0.4)
    fig.update_layout(height=560, legend_title_text="recipe")
    write_plot(fig, "math_vs_paloma_pareto.html")


def plot_predictions(curves: pd.DataFrame, predictions: pd.DataFrame, metric: str, output_name: str) -> None:
    df = curves[curves["metric"].eq(metric) & curves["scale"].eq("1e22")].copy()
    metric_name = DISPLAY_METRIC.get(metric, metric)
    fig = make_subplots(
        rows=len(MIX_ORDER),
        cols=len(LR_ORDER),
        subplot_titles=[f"{mix} lr{lr}" for mix in MIX_ORDER for lr in LR_ORDER],
        shared_yaxes=False,
        shared_xaxes=True,
    )
    color = "#2563eb"
    for i, mix in enumerate(MIX_ORDER, start=1):
        for j, lr in enumerate(LR_ORDER, start=1):
            cell = df[df["mix"].eq(mix) & df["lr"].eq(lr)].sort_values("tau")
            if not cell.empty:
                fig.add_trace(
                    go.Scatter(
                        x=cell["tau"],
                        y=cell["value"],
                        mode="lines+markers",
                        name="observed curve",
                        legendgroup="observed",
                        showlegend=(i == 1 and j == 1),
                        line={"color": color},
                        hovertemplate="tau=%{x:.3f}<br>loss=%{y:.4f}<extra></extra>",
                    ),
                    row=i,
                    col=j,
                )
            pred = predictions[predictions["mix"].eq(mix) & predictions["lr"].eq(lr)]
            if not pred.empty:
                pred_row = pred.iloc[0]
                if pred_row["pred_method"] == "observed final":
                    continue
                value = pred_row.get("pred_endpoint")
                if value is None or not math.isfinite(float(value)):
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=[1.0],
                        y=[float(value)],
                        mode="markers",
                        name="forecast endpoint",
                        legendgroup="forecast",
                        showlegend=(i == 1 and j == 1),
                        marker={"size": 12, "color": "#111827", "symbol": "star"},
                        hovertemplate=(
                            f"{pred_row['pred_method']} forecast<br>" "tau=1<br>loss=%{y:.4f}<extra></extra>"
                        ),
                    ),
                    row=i,
                    col=j,
                )
    fig.update_xaxes(title_text="tau", range=[0, 1.03])
    fig.update_yaxes(title_text=metric_name)
    fig.update_layout(
        title=f"1e22 {metric_name} prefix curves with shape-ratio endpoint forecasts",
        height=760,
        legend_title_text="trace",
    )
    write_plot(fig, output_name)


def plot_endpoint_heatmap(curves: pd.DataFrame, predictions: pd.DataFrame, metric: str, output_name: str) -> None:
    endpoints = []
    completed_cells = complete_cell_keys(curves)
    for (scale, mix, lr), group in curves[curves["metric"].eq(metric)].groupby(
        ["scale", "mix", "lr"], sort=False, observed=True
    ):
        group = group.sort_values("step")
        if (str(scale), str(mix), str(lr)) in completed_cells:
            value = float(group.iloc[-1]["value"])
            kind = "observed final"
        elif scale == "1e22":
            pred = predictions[predictions["mix"].eq(mix) & predictions["lr"].eq(lr)]
            if pred.empty or pd.isna(pred.iloc[0]["pred_endpoint"]):
                continue
            value = float(pred.iloc[0]["pred_endpoint"])
            kind = str(pred.iloc[0]["pred_method"])
        else:
            continue
        endpoints.append({"scale": scale, "mix": mix, "lr": lr, "value": value, "kind": kind})
    df = pd.DataFrame(endpoints)
    if df.empty:
        return
    df["recipe"] = df.apply(lambda row: recipe_label(row["mix"], row["lr"]), axis=1)
    metric_name = DISPLAY_METRIC.get(metric, metric)
    fig = px.bar(
        ordered(df),
        x="recipe",
        y="value",
        color="kind",
        facet_col="scale",
        barmode="group",
        title=f"Observed/fill-in final {metric_name} by recipe (1e20 is v5-isoflop)",
        labels={"value": f"final {metric_name}", "recipe": "recipe"},
    )
    fig.update_layout(height=520)
    write_plot(fig, output_name)


def endpoint_scaling_rows(curves: pd.DataFrame) -> pd.DataFrame:
    completed_cells = complete_cell_keys(curves)
    rows: list[dict[str, Any]] = []
    metrics = [EVAL_METRIC, MATH_METRIC, PALOMA_MACRO_METRIC]
    for (metric, scale, mix, lr), group in curves[curves["metric"].isin(metrics)].groupby(
        ["metric", "scale", "mix", "lr"], sort=False, observed=True
    ):
        group = group.sort_values("step")
        last = group.iloc[-1]
        cell_key = (str(scale), str(mix), str(lr))
        if cell_key not in completed_cells:
            continue
        rows.append(
            {
                "metric": metric,
                "metric_name": DISPLAY_METRIC.get(metric, metric),
                "scale": str(scale),
                "model_flops": MODEL_FLOPS[str(scale)],
                "mix": str(mix),
                "lr": str(lr),
                "recipe": recipe_label(mix, lr),
                "value": float(last["value"]),
                "last_step": int(last["step"]),
                "tau": float(last["tau"]),
            }
        )
    return pd.DataFrame(rows)


def plot_final_loss_vs_model_flops(curves: pd.DataFrame) -> None:
    df = endpoint_scaling_rows(curves)
    if df.empty:
        return
    metric_order = [EVAL_METRIC, MATH_METRIC, PALOMA_MACRO_METRIC]
    fig = make_subplots(
        rows=1,
        cols=len(metric_order),
        subplot_titles=[DISPLAY_METRIC[metric] for metric in metric_order],
        shared_xaxes=True,
        shared_yaxes=False,
    )
    recipes = [recipe_label(mix, lr) for mix in MIX_ORDER for lr in LR_ORDER]
    for col, metric in enumerate(metric_order, start=1):
        metric_df = df[df["metric"].eq(metric)]
        for recipe in recipes:
            recipe_df = metric_df[metric_df["recipe"].eq(recipe)].sort_values("model_flops")
            if not recipe_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=recipe_df["model_flops"],
                        y=recipe_df["value"],
                        mode="lines+markers",
                        name=recipe,
                        legendgroup=recipe,
                        showlegend=(col == 1),
                        line={"color": RECIPE_COLORS.get(recipe)},
                        marker={"size": 9, "symbol": "circle"},
                        customdata=np.stack(
                            [
                                recipe_df["scale"].to_numpy(),
                                recipe_df["last_step"].to_numpy(),
                                recipe_df["tau"].to_numpy(),
                            ],
                            axis=-1,
                        ),
                        hovertemplate=(
                            "%{customdata[0]}<br>"
                            f"{recipe}<br>"
                            "observed end-of-training loss<br>"
                            "step=%{customdata[1]} tau=%{customdata[2]:.3f}<br>"
                            "loss=%{y:.4f}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=col,
                )
    for col in range(1, len(metric_order) + 1):
        fig.update_xaxes(
            type="log",
            tickmode="array",
            tickvals=[MODEL_FLOPS[scale] for scale in SCALE_ORDER],
            ticktext=SCALE_ORDER,
            title_text="logged model FLOPs scale",
            row=1,
            col=col,
        )
        fig.update_yaxes(title_text="final loss", row=1, col=col)
    fig.update_layout(
        title="Observed final validation loss vs logged model FLOPs by midtraining recipe",
        height=620,
        legend_title_text="recipe",
    )
    write_plot(fig, "final_validation_loss_vs_model_flops.html")


PLOT_LINKS = [
    ("Current eval/loss by logged scale", "plots/current_eval_loss_by_scale.html"),
    ("Raw validation curves", "plots/raw_validation_curves_dashboard.html"),
    ("Math trajectory collapse", "plots/normalized_math_loss_collapse_finished_1e20_1e21.html"),
    ("Math vs Paloma Pareto", "plots/math_vs_paloma_pareto.html"),
    ("Observed final validation loss vs model FLOPs", "plots/final_validation_loss_vs_model_flops.html"),
    ("1e22 eval/loss forecasts", "plots/1e22_eval_loss_predictions.html"),
    ("1e22 math forecasts", "plots/1e22_math_loss_predictions.html"),
    ("Endpoint eval/loss by recipe", "plots/endpoint_eval_loss_by_recipe.html"),
    ("Endpoint math loss by recipe", "plots/endpoint_math_loss_by_recipe.html"),
]


def forecast_table(predictions: pd.DataFrame) -> str:
    if predictions.empty:
        return "_No predictions available._"
    show = predictions[
        [
            "mix",
            "lr",
            "current_step",
            "current_tau",
            "current_value",
            "pred_shape_ratio",
            "pred_tail_linear",
            "pred_endpoint",
            "pred_method",
        ]
    ].copy()
    return show.to_markdown(index=False, floatfmt=".4f")


def write_summary(registry: pd.DataFrame, curves: pd.DataFrame, predictions_by_metric: dict[str, pd.DataFrame]) -> None:
    finished = registry[registry["selected_finished_for_cell"]]
    best_prefix = registry[registry["best_prefix_for_cell"]]
    lines = [
        "# Delphi Midtraining Scaling Analysis",
        "",
        SCALE_CAVEAT_TEXT,
        "",
        "Offline analysis built from `midtrain_wandb_data/`.",
        "",
        "## Run Coverage",
        "",
        f"- Current-sweep W&B attempts in registry: `{len(registry)}`",
        f"- Finished cells selected for analysis: `{len(finished)}`",
        f"- Best-prefix cells: `{len(best_prefix)}`",
        "",
        "Finished historical sweep cells by logged scale:",
        "",
    ]
    for scale in SCALE_ORDER:
        count = int(finished["scale"].eq(scale).sum())
        lines.append(f"- `{scale}`: `{count}/{len(MIX_ORDER) * len(LR_ORDER)}`")
    lines += [
        "",
        "## Data Caveat",
        "",
        SCALE_CAVEAT_TEXT,
        "",
        "The two `1e20 p67m33` runs with run ids `...lr0.33-590ea1` and",
        "`...lr0.5-9e1229` merge early steps from `marin-community/marin`",
        "with later histories from `marin-community/delphi-midtraining`.",
        "",
        f"A cell is treated as complete when W&B history reaches within `{FINAL_STEP_TOLERANCE}`",
        "steps of the expected final step. This keeps near-final runs like",
        "`1e22 p50m50 lr0.33` from being mislabeled as unfinished only because",
        "the final W&B row is missing.",
        "",
        "## Marker Semantics And Forecasting",
        "",
        "In `plots/final_validation_loss_vs_model_flops.html`, every point is an",
        "observed end-of-training loss. Prefix checkpoints and forecasted endpoints",
        "are intentionally excluded from that scaling-law view.",
        "",
        FORECAST_METHOD_TEXT,
        "",
        "## 1e22 Eval/Loss Forecasts",
        "",
    ]
    lines.append(forecast_table(predictions_by_metric.get(EVAL_METRIC, pd.DataFrame())))
    lines += [
        "",
        "## 1e22 Math Validation Forecasts",
        "",
        "These use the Nemotron CC math v1 4+ validation loss.",
        "",
    ]
    lines.append(forecast_table(predictions_by_metric.get(MATH_METRIC, pd.DataFrame())))
    lines += [
        "",
        "## Plots",
        "",
        *[f"- `{path}` — {label}" for label, path in PLOT_LINKS],
        "",
        "Use the `.parquet` tables for downstream pandas work when possible. CSV",
        "readers can otherwise interpret labels like `1e20` as scientific",
        'notation unless `dtype={"scale": str}` is passed. Keep the `1e20`',
        "string only as a W&B/logged-scale label; it is not a canonical Delphi",
        "base.",
        "",
        "Prediction caveat: incomplete 1e22 endpoints use the shape-ratio forecast",
        "when available and only fall back to tail-linear extrapolation if no",
        "shape reference exists. Completed 1e22 cells use observed final values.",
        "",
    ]
    (OUT_DIR / "analysis_summary.md").write_text("\n".join(lines))


def write_index(registry: pd.DataFrame, predictions_by_metric: dict[str, pd.DataFrame]) -> None:
    finished = registry[registry["selected_finished_for_cell"]]
    best_prefix = registry[registry["best_prefix_for_cell"]]
    rows = []
    for scale in SCALE_ORDER:
        finished_count = int(finished["scale"].eq(scale).sum())
        prefix_count = int(best_prefix["scale"].eq(scale).sum()) - finished_count
        rows.append((scale, finished_count, prefix_count, len(MIX_ORDER) * len(LR_ORDER)))
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plot_links = "\n".join(f'<li><a href="{path}">{label}</a></li>' for label, path in PLOT_LINKS)
    coverage_rows = "\n".join(
        f"<tr><td>{scale}</td><td>{finished_count}</td><td>{prefix_count}</td><td>{total}</td></tr>"
        for scale, finished_count, prefix_count, total in rows
    )
    eval_forecast_count = int(
        predictions_by_metric.get(EVAL_METRIC, pd.DataFrame())
        .get("pred_method", pd.Series(dtype=str))
        .ne("observed final")
        .sum()
    )
    math_forecast_count = int(
        predictions_by_metric.get(MATH_METRIC, pd.DataFrame())
        .get("pred_method", pd.Series(dtype=str))
        .ne("observed final")
        .sum()
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Delphi Midtraining Analysis - 1e20 Caveated</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      color: #111827;
      background: #f8fafc;
    }}
    header {{ background: #111827; color: white; padding: 24px 32px; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 24px; }}
    section {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 18px; margin-bottom: 16px; }}
    h1 {{ margin: 0 0 6px; font-size: 26px; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    p {{ margin: 0 0 10px; color: #4b5563; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    a {{ color: #1d4ed8; font-weight: 600; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{ background: #f3f4f6; padding: 1px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    <h1>Delphi Midtraining Analysis</h1>
    <p>
      Generated {generated} from <code>midtrain_wandb_data/</code>.
      Main outputs are in <code>midtrain_analysis_outputs/</code>.
    </p>
    <p>{SCALE_CAVEAT_TEXT}</p>
  </header>
  <main>
    <section>
      <h2>Coverage</h2>
      <table>
        <thead><tr><th>Scale</th><th>Finished</th><th>Prefix</th><th>Total</th></tr></thead>
        <tbody>{coverage_rows}</tbody>
      </table>
    </section>
    <section>
      <h2>Plots</h2>
      <ul>{plot_links}</ul>
    </section>
    <section>
      <h2>Notes</h2>
      <p>
        The trajectory-collapse plot now uses <code>{MATH_METRIC}</code>,
        not aggregate <code>{EVAL_METRIC}</code>.
      </p>
      <p>
        The FLOPs scaling plot includes only observed end-of-training losses.
        Prefix checkpoints and forecasted endpoints are intentionally excluded.
      </p>
      <p>{FORECAST_METHOD_TEXT}</p>
      <p>
        Incomplete 1e22 forecasts: {eval_forecast_count} eval/loss cells and
        {math_forecast_count} math cells use a forecast marker; completed cells
        use observed values.
      </p>
      <p>See <a href="analysis_summary.md">analysis_summary.md</a> for tables and caveats.</p>
    </section>
  </main>
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(html)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    infos = load_run_infos()
    histories = []
    progress_rows = []
    for info in infos:
        hist = load_history_long(info)
        histories.append(hist)
        progress_rows.append(run_progress(info, hist))
    history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    registry = choose_best_prefix_runs(pd.DataFrame(progress_rows))

    curves = merge_cell_histories(history, registry)
    curves = add_baselines(curves)
    curves = ordered(curves)

    predictions_eval = make_predictions(curves, metric=EVAL_METRIC)
    predictions_math = make_predictions(curves, metric=MATH_METRIC)
    predictions_paloma_macro = make_predictions(curves, metric=PALOMA_MACRO_METRIC)
    predictions_by_metric = {
        EVAL_METRIC: predictions_eval,
        MATH_METRIC: predictions_math,
        PALOMA_MACRO_METRIC: predictions_paloma_macro,
    }

    registry.to_csv(OUT_DIR / "midtrain_run_registry.csv", index=False)
    curves.to_csv(OUT_DIR / "midtrain_trajectory_deltas.csv", index=False)
    predictions_eval.to_csv(OUT_DIR / "prediction_1e22_eval_loss.csv", index=False)
    predictions_math.to_csv(OUT_DIR / "prediction_1e22_math_loss.csv", index=False)
    predictions_paloma_macro.to_csv(OUT_DIR / "prediction_1e22_paloma_macro_loss.csv", index=False)
    registry.to_parquet(OUT_DIR / "midtrain_run_registry.parquet", index=False)
    curves.to_parquet(OUT_DIR / "midtrain_trajectory_deltas.parquet", index=False)
    predictions_eval.to_parquet(OUT_DIR / "prediction_1e22_eval_loss.parquet", index=False)
    predictions_math.to_parquet(OUT_DIR / "prediction_1e22_math_loss.parquet", index=False)
    predictions_paloma_macro.to_parquet(OUT_DIR / "prediction_1e22_paloma_macro_loss.parquet", index=False)

    plot_raw_curves(curves)
    plot_metric_dashboard(curves)
    plot_normalized_collapse(curves)
    plot_pareto(curves)
    plot_final_loss_vs_model_flops(curves)
    plot_predictions(curves, predictions_eval, EVAL_METRIC, "1e22_eval_loss_predictions.html")
    plot_predictions(curves, predictions_math, MATH_METRIC, "1e22_math_loss_predictions.html")
    plot_endpoint_heatmap(curves, predictions_eval, EVAL_METRIC, "endpoint_eval_loss_by_recipe.html")
    plot_endpoint_heatmap(curves, predictions_math, MATH_METRIC, "endpoint_math_loss_by_recipe.html")
    write_summary(registry, curves, predictions_by_metric)
    write_index(registry, predictions_by_metric)

    print(f"Wrote registry: {OUT_DIR / 'midtrain_run_registry.csv'}")
    print(f"Wrote trajectories: {OUT_DIR / 'midtrain_trajectory_deltas.csv'}")
    print(f"Wrote predictions: {OUT_DIR / 'prediction_1e22_eval_loss.csv'}")
    print(f"Wrote predictions: {OUT_DIR / 'prediction_1e22_math_loss.csv'}")
    print(f"Wrote predictions: {OUT_DIR / 'prediction_1e22_paloma_macro_loss.csv'}")
    print(f"Wrote plots: {PLOTS_DIR}")
    print(f"Wrote summary: {OUT_DIR / 'analysis_summary.md'}")
    print(f"Wrote index: {OUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
