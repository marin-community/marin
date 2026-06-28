# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Compare K=0.20 fits on old validation losses vs clean-seen loss.

The clean-seen eval sweep also logged the old math-validation anchors for the
same checkpoints. This report runs the same K=0.20 fit registry on each target
loss so we can test whether the extrapolation problem is tied to the validation
target rather than the run set.

Run:
    uv run --with scipy --with plotly --with pandas --with gcsfs \
      python scripts/analysis/delphi_k020_old_vs_clean_val_report.py
"""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from delphi_isotoken_endpoint_scaling import DEFAULT_CUTOFF_SCALE, SCALE_ORDER
from delphi_k020_clean_seen_fit_family_report import (
    ANCHOR_METRIC,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    LR_LABELS,
    LR_ORDER,
    RESOURCE_LABELS,
    TARGET_METRIC,
    FailedFit,
    FitSpec,
    FittedModel,
    failure_table_html,
    finite_or_none,
    fit_model,
    fit_specs,
    load_clean_seen_summary,
    predict_model,
    sort_value,
    split_points,
    table_html,
)
from marin.scaling_laws.scaling_plots import MARKERS, PALETTE
from plotly.subplots import make_subplots

DEFAULT_OUTPUT_STEM = "delphi_k020_old_vs_clean_val_fit_report"
OLD_ENDPOINTS_PATH = Path("midtrain_analysis_outputs/small_final_loss_scaling/endpoints.csv")
OLD_ISOFLOP_PATH = DEFAULT_OUTPUT_DIR / "isoflop_k020_endpoints.csv"


@dataclass(frozen=True)
class TargetSpec:
    key: str
    label: str
    column: str
    description: str
    role: str


TARGETS = (
    TargetSpec(
        key="clean_seen",
        label="new clean-seen loss",
        column="clean_seen_loss",
        description="The new decontaminated validation loss. This is one of the two primary targets in this report.",
        role="primary",
    ),
    TargetSpec(
        key="old_4plus_anchor",
        label="old 4plus math val",
        column=ANCHOR_METRIC,
        description=(
            "The old math-validation target used in the earlier K=0.20 plots: "
            "eval/nemotron_cc_math_v1/4plus/loss. It is one named slice of the old Nemotron math validation set."
        ),
        role="primary",
    ),
    TargetSpec(
        key="old_eval_loss",
        label="old eval/loss aggregate",
        column="eval_loss",
        description=(
            "Auxiliary diagnostic only: the broad eval/loss aggregate emitted by the same eval jobs. "
            "This is not the main old math target we were fitting before."
        ),
        role="diagnostic",
    ),
    TargetSpec(
        key="old_macro_loss",
        label="old macro math val",
        column="macro_loss",
        description=(
            "Auxiliary diagnostic only: macro average across the old math-validation subtasks/slices. "
            "This differs from 4plus because 4plus is one named slice, while macro_loss averages across slices."
        ),
        role="diagnostic",
    ),
)
PRIMARY_TARGET_KEYS = tuple(target.key for target in TARGETS if target.role == "primary")
DIAGNOSTIC_TARGET_KEYS = tuple(target.key for target in TARGETS if target.role == "diagnostic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="CSV path or gs:// URI for the clean-seen summary.")
    parser.add_argument(
        "--fit-through-scale",
        choices=SCALE_ORDER[:-1],
        default=DEFAULT_CUTOFF_SCALE,
        help="Largest scale included in the training split.",
    )
    parser.add_argument("--old-endpoints", type=Path, default=OLD_ENDPOINTS_PATH)
    parser.add_argument("--old-isoflop", type=Path, default=OLD_ISOFLOP_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with fsspec.open(str(path), "w") as handle:
        handle.write(text)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    write_text(path, frame.to_csv(index=False))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, default=finite_or_none))


def records_for_json(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: finite_or_none(value) for key, value in record.items()} for record in frame.to_dict(orient="records")]


def target_points(base_points: pd.DataFrame, target: TargetSpec) -> pd.DataFrame:
    missing = [column for column in (target.column, "run", "scale", "lr") if column not in base_points.columns]
    if missing:
        raise ValueError(f"Missing required columns for {target.key}: {missing}")
    points = base_points.copy()
    points[TARGET_METRIC] = points[target.column].astype(float)
    points["target_key"] = target.key
    points["target_label"] = target.label
    points["target_column"] = target.column
    points["target_description"] = target.description
    return points


def comparable_fit_specs() -> list[FitSpec]:
    """Use scale/LR fits only, excluding anchor calibration for fair metric comparison."""

    return [spec for spec in fit_specs() if not spec.family.startswith("anchor")]


def prediction_frame(model: FittedModel, points: pd.DataFrame) -> pd.DataFrame:
    rows = points.copy()
    rows["model_key"] = model.spec.key
    rows["model_label"] = model.spec.label
    rows["family"] = model.spec.family
    rows["resource"] = RESOURCE_LABELS.get(model.spec.scale_feature, model.spec.scale_feature)
    rows["prediction"] = predict_model(model.spec, model.parameters, rows)
    rows["error"] = rows["prediction"] - rows[TARGET_METRIC]
    rows["error_pct"] = (rows["prediction"] / rows[TARGET_METRIC] - 1.0) * 100.0
    rows["abs_error_pct"] = rows["error_pct"].abs()
    return rows.sort_values(["target_key", "model_key", "split", "scale_flops", "lr_numeric"]).reset_index(drop=True)


def fit_one_target(
    points: pd.DataFrame,
    fit_through_scale: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[FittedModel], list[FailedFit]]:
    train, _ = split_points(points, fit_through_scale)
    prediction_frames: list[pd.DataFrame] = []
    fit_rows: list[dict[str, Any]] = []
    models: list[FittedModel] = []
    failures: list[FailedFit] = []
    target_key = str(points["target_key"].iloc[0])
    target_label = str(points["target_label"].iloc[0])
    for spec in comparable_fit_specs():
        try:
            model = fit_model(spec, train)
            predictions = prediction_frame(model, points)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            failures.append(FailedFit(spec=spec, error=f"{target_label}: {exc}"))
            continue
        models.append(model)
        prediction_frames.append(predictions)
        fit_rows.append(
            {
                "target_key": target_key,
                "target_label": target_label,
                "target_column": str(points["target_column"].iloc[0]),
                "model_key": spec.key,
                "model_label": spec.label,
                "family": spec.family,
                "resource": RESOURCE_LABELS.get(spec.scale_feature, spec.scale_feature),
                "train_n": model.train_n,
                "fit_r2": model.fit_r2,
                "fit_rmse": model.fit_rmse,
                "parameters": json.dumps(model.parameters, sort_keys=True),
            }
        )
    if not prediction_frames:
        raise ValueError(f"No fits succeeded for {target_label}")
    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(fit_rows), models, failures


def fit_all_targets(
    base_points: pd.DataFrame,
    fit_through_scale: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[FailedFit]]:
    prediction_frames: list[pd.DataFrame] = []
    fit_tables: list[pd.DataFrame] = []
    failures: list[FailedFit] = []
    for target in TARGETS:
        points = target_points(base_points, target)
        predictions, fit_table, _, target_failures = fit_one_target(points, fit_through_scale)
        prediction_frames.append(predictions)
        fit_tables.append(fit_table)
        failures.extend(target_failures)
    return pd.concat(prediction_frames, ignore_index=True), pd.concat(fit_tables, ignore_index=True), failures


def summarize_predictions(predictions: pd.DataFrame, fit_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["target_key", "target_label", "target_column", "model_key", "model_label", "family", "resource"]
    for group_key, frame in predictions.groupby(group_columns, sort=False):
        target_key, target_label, target_column, model_key, model_label, family, resource = group_key
        heldout = frame[frame["split"].eq("heldout")]
        one_e22 = heldout[heldout["scale"].eq("1e22")]
        fit_meta = fit_table[fit_table["target_key"].eq(target_key) & fit_table["model_key"].eq(model_key)].iloc[0]
        errors = heldout["error_pct"].to_numpy(dtype=float)
        loss_errors = heldout["error"].to_numpy(dtype=float)
        one_e22_errors = one_e22["error_pct"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "target_key": target_key,
            "target_label": target_label,
            "target_column": target_column,
            "model_key": model_key,
            "model_label": model_label,
            "family": family,
            "resource": resource,
            "train_n": int(fit_meta["train_n"]),
            "heldout_n": len(heldout),
            "heldout_mae_pct": float(np.mean(np.abs(errors))),
            "heldout_rmse_pct": math.sqrt(float(np.mean(errors**2))),
            "heldout_bias_pct": float(np.mean(errors)),
            "heldout_loss_rmse": math.sqrt(float(np.mean(loss_errors**2))),
            "1e22_mae_pct": float(np.mean(np.abs(one_e22_errors))),
            "1e22_rmse_pct": math.sqrt(float(np.mean(one_e22_errors**2))),
            "1e22_bias_pct": float(np.mean(one_e22_errors)),
            "fit_r2": None if pd.isna(fit_meta["fit_r2"]) else float(fit_meta["fit_r2"]),
            "fit_rmse": float(fit_meta["fit_rmse"]),
        }
        for lr in LR_ORDER:
            cell = one_e22[one_e22["lr"].eq(lr)]
            if cell.empty:
                continue
            label = lr.replace(".", "")
            row[f"1e22_{label}_actual"] = float(cell[TARGET_METRIC].iloc[0])
            row[f"1e22_{label}_prediction"] = float(cell["prediction"].iloc[0])
            row[f"1e22_{label}_error_pct"] = float(cell["error_pct"].iloc[0])
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["target_key", "heldout_mae_pct", "1e22_mae_pct", "model_label"])
        .reset_index(drop=True)
    )


def best_by_target(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.sort_values(["target_key", "heldout_mae_pct", "1e22_mae_pct", "model_label"])
        .groupby(["target_key", "target_label"], as_index=False, sort=False)
        .first()
        .sort_values("heldout_mae_pct")
        .reset_index(drop=True)
    )


def one_e22_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = predictions[predictions["scale"].eq("1e22")].copy()
    return (
        rows[
            [
                "target_label",
                "target_column",
                "model_label",
                "family",
                "resource",
                "lr",
                TARGET_METRIC,
                "prediction",
                "error_pct",
                "abs_error_pct",
                "run",
            ]
        ]
        .sort_values(["target_label", "abs_error_pct", "lr", "model_label"])
        .reset_index(drop=True)
    )


def lr_suffix_to_decimal(value: Any) -> str:
    text = str(value)
    if text in {"33", "50", "67", "83"}:
        return f"0.{text}"
    return f"{float(value):.2f}"


def cached_old_val_check(base_points: pd.DataFrame, old_endpoints_path: Path, old_isoflop_path: Path) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    if old_endpoints_path.exists():
        old = pd.read_csv(old_endpoints_path, dtype={"scale": str, "lr": str})
        small = old[
            old["metric_label"].eq("math_val_loss")
            & old["mix"].eq("p33m67")
            & old["complete"].astype(bool)
            & old["run_id"].astype(str).str.contains("-k0p20-", regex=False)
        ].copy()
        small["lr"] = small["lr"].map(lr_suffix_to_decimal)
        pieces.append(small[["scale", "lr", "value", "run_id"]])
    if old_isoflop_path.exists():
        isoflop = pd.read_csv(old_isoflop_path, dtype={"scale_label": str, "lr": str})
        heldout = isoflop[isoflop["scale_label"].isin(["1e21", "1e22"])].copy()
        heldout["scale"] = heldout["scale_label"]
        heldout["lr"] = heldout["lr"].map(lr_suffix_to_decimal)
        heldout["run_id"] = heldout["run_id"].astype(str)
        pieces.append(heldout[["scale", "lr", "value", "run_id"]])
    if not pieces:
        return pd.DataFrame()
    cached = pd.concat(pieces, ignore_index=True).drop_duplicates(["scale", "lr"], keep="last")
    observed = base_points[["scale", "lr", ANCHOR_METRIC, "run"]].copy()
    merged = observed.merge(cached, on=["scale", "lr"], how="inner")
    if merged.empty:
        return merged
    merged = merged.rename(
        columns={
            "value": "cached_old_math_val_loss",
            ANCHOR_METRIC: "same_sweep_old_4plus_loss",
            "run": "same_sweep_run",
            "run_id": "cached_old_run",
        }
    )
    merged["delta"] = merged["same_sweep_old_4plus_loss"] - merged["cached_old_math_val_loss"]
    merged["delta_pct"] = (merged["same_sweep_old_4plus_loss"] / merged["cached_old_math_val_loss"] - 1.0) * 100.0
    return merged.sort_values(["scale", "lr"]).reset_index(drop=True)


def loss_comparison_figure(points: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Actual losses by scale", "Clean-seen minus old 4plus anchor"),
        horizontal_spacing=0.1,
    )
    target_columns = [
        ("clean_seen_loss", "new clean-seen"),
        (ANCHOR_METRIC, "old 4plus"),
    ]
    colors = {"clean_seen_loss": PALETTE[0], ANCHOR_METRIC: PALETTE[1]}
    for column, label in target_columns:
        for lr in LR_ORDER:
            frame = points[points["lr"].eq(lr)].sort_values("scale_flops")
            fig.add_trace(
                go.Scatter(
                    x=frame["scale"],
                    y=frame[column],
                    mode="lines+markers",
                    name=f"{label} {LR_LABELS[lr]}",
                    legendgroup=f"{column}-{lr}",
                    line={
                        "color": colors[column],
                        "dash": (
                            "solid"
                            if lr == "0.67"
                            else "dot" if lr == "0.33" else "dash" if lr == "0.50" else "longdash"
                        ),
                    },
                    marker={"size": 7, "symbol": MARKERS[int(float(lr) * 100) % len(MARKERS)]},
                    hovertemplate="%{fullData.name}<br>scale=%{x}<br>loss=%{y:.6f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
    for lr in LR_ORDER:
        frame = points[points["lr"].eq(lr)].sort_values("scale_flops")
        delta = frame["clean_seen_loss"] - frame[ANCHOR_METRIC]
        fig.add_trace(
            go.Scatter(
                x=frame["scale"],
                y=delta,
                mode="lines+markers",
                name=f"clean - old 4plus {LR_LABELS[lr]}",
                legendgroup=f"delta-{lr}",
                line={"color": PALETTE[int(float(lr) * 100) % len(PALETTE)]},
                marker={"size": 8},
                hovertemplate="%{fullData.name}<br>scale=%{x}<br>delta=%{y:.6f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title_text="scale", categoryorder="array", categoryarray=SCALE_ORDER, row=1, col=1)
    fig.update_xaxes(title_text="scale", categoryorder="array", categoryarray=SCALE_ORDER, row=1, col=2)
    fig.update_yaxes(title_text="loss", row=1, col=1)
    fig.update_yaxes(title_text="loss delta", row=1, col=2)
    fig.update_layout(
        title="K=0.20 p33m67 validation losses",
        width=1500,
        height=620,
        legend={"orientation": "h", "y": -0.18},
        margin={"l": 70, "r": 30, "t": 90, "b": 150},
    )
    return fig


def fit_dropdown_figure(predictions: pd.DataFrame, summary: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Actual vs predicted", "Heldout prediction error", "1e22 actual vs predicted"),
        horizontal_spacing=0.09,
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}]],
    )
    ordered = summary.sort_values(["target_key", "heldout_mae_pct", "1e22_mae_pct", "model_label"])
    initial_pair = (str(ordered.iloc[0]["target_key"]), str(ordered.iloc[0]["model_key"]))
    trace_keys: list[tuple[str, str] | str] = []
    lr_colors = {lr: PALETTE[index % len(PALETTE)] for index, lr in enumerate(LR_ORDER)}

    def add_trace(trace: Any, target_key: str, model_key: str, row: int, col: int) -> None:
        trace.visible = (target_key, model_key) == initial_pair
        fig.add_trace(trace, row=row, col=col)
        trace_keys.append((target_key, model_key))

    def add_always(trace: Any, row: int, col: int) -> None:
        trace.visible = True
        fig.add_trace(trace, row=row, col=col)
        trace_keys.append("always")

    for (target_key, model_key), model_frame in predictions.groupby(["target_key", "model_key"], sort=False):
        for lr in LR_ORDER:
            for split in ("train", "heldout"):
                frame = model_frame[model_frame["lr"].eq(lr) & model_frame["split"].eq(split)]
                if frame.empty:
                    continue
                add_trace(
                    go.Scatter(
                        x=frame[TARGET_METRIC],
                        y=frame["prediction"],
                        mode="markers",
                        name=f"{LR_LABELS[lr]} {split}",
                        legendgroup=f"{target_key}-{model_key}-{lr}-{split}",
                        marker={
                            "symbol": MARKERS[0] if split == "train" else MARKERS[9],
                            "size": 9 if split == "train" else 13,
                            "color": lr_colors[lr],
                            "line": {"width": 1, "color": "#111827"},
                        },
                        customdata=np.stack(
                            [
                                frame["target_label"].astype(str),
                                frame["model_label"].astype(str),
                                frame["scale"].astype(str),
                                frame["lr"].astype(str),
                                frame["error_pct"].astype(float),
                                frame["run"].astype(str),
                            ],
                            axis=-1,
                        ),
                        hovertemplate=(
                            "%{customdata[0]}<br>%{customdata[1]}<br>%{customdata[5]}<br>"
                            "scale=%{customdata[2]} lr=%{customdata[3]}<br>actual=%{x:.5f}<br>pred=%{y:.5f}"
                            "<br>error=%{customdata[4]:+.2f}%<extra></extra>"
                        ),
                    ),
                    str(target_key),
                    str(model_key),
                    row=1,
                    col=1,
                )
        heldout = model_frame[model_frame["split"].eq("heldout")]
        for lr in LR_ORDER:
            frame = heldout[heldout["lr"].eq(lr)].sort_values("scale_flops")
            if frame.empty:
                continue
            add_trace(
                go.Scatter(
                    x=frame["scale"],
                    y=frame["error_pct"],
                    mode="lines+markers",
                    name=f"{LR_LABELS[lr]} heldout error",
                    legendgroup=f"{target_key}-{model_key}-{lr}-error",
                    marker={"symbol": MARKERS[int(float(lr) * 100) % len(MARKERS)], "size": 10, "color": lr_colors[lr]},
                    line={"color": lr_colors[lr]},
                    hovertemplate="%{fullData.name}<br>scale=%{x}<br>error=%{y:+.2f}%<extra></extra>",
                ),
                str(target_key),
                str(model_key),
                row=1,
                col=2,
            )
        one_e22 = model_frame[model_frame["scale"].eq("1e22")].sort_values("lr_numeric")
        if not one_e22.empty:
            add_trace(
                go.Bar(
                    x=one_e22["lr"],
                    y=one_e22[TARGET_METRIC],
                    name="1e22 actual",
                    marker={"color": "#94a3b8"},
                    offsetgroup="actual",
                    legendgroup=f"{target_key}-{model_key}-1e22",
                    hovertemplate="lr=%{x}<br>actual=%{y:.5f}<extra></extra>",
                ),
                str(target_key),
                str(model_key),
                row=1,
                col=3,
            )
            add_trace(
                go.Bar(
                    x=one_e22["lr"],
                    y=one_e22["prediction"],
                    name="1e22 predicted",
                    marker={"color": "#1877f2"},
                    offsetgroup="predicted",
                    legendgroup=f"{target_key}-{model_key}-1e22",
                    customdata=one_e22["error_pct"],
                    hovertemplate="lr=%{x}<br>pred=%{y:.5f}<br>error=%{customdata:+.2f}%<extra></extra>",
                ),
                str(target_key),
                str(model_key),
                row=1,
                col=3,
            )
    lower = float(min(predictions[TARGET_METRIC].min(), predictions["prediction"].min()))
    upper = float(max(predictions[TARGET_METRIC].max(), predictions["prediction"].max()))
    add_always(
        go.Scatter(
            x=[lower, upper],
            y=[lower, upper],
            mode="lines",
            name="perfect prediction",
            line={"color": "#475569", "dash": "dash"},
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line={"color": "#475569", "width": 1}, row=1, col=2)
    label_by_pair = {
        (str(row["target_key"]), str(row["model_key"])): f"{row['target_label']}: {row['model_label']}"
        for _, row in ordered.iterrows()
    }
    buttons = []
    for pair, label in label_by_pair.items():
        visible = [trace_key == "always" or trace_key == pair for trace_key in trace_keys]
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"Fit comparison<br><sup>{html.escape(label)}</sup>"},
                ],
            }
        )
    initial_label = label_by_pair[initial_pair]
    fig.update_xaxes(title_text="actual target loss", row=1, col=1)
    fig.update_yaxes(title_text="predicted target loss", row=1, col=1)
    fig.update_xaxes(title_text="heldout scale", categoryorder="array", categoryarray=SCALE_ORDER, row=1, col=2)
    fig.update_yaxes(title_text="prediction error (pred / actual - 1) [%]", row=1, col=2)
    fig.update_xaxes(title_text="1e22 LR factor", row=1, col=3)
    fig.update_yaxes(title_text="target loss", row=1, col=3)
    fig.update_layout(
        title=f"Fit comparison<br><sup>{html.escape(initial_label)}</sup>",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.0,
                "xanchor": "left",
                "y": 1.2,
                "yanchor": "top",
            }
        ],
        barmode="group",
        width=1550,
        height=700,
        legend={"orientation": "h", "y": -0.18},
        margin={"l": 70, "r": 30, "t": 110, "b": 150},
    )
    return fig


def targets_table_html(role: str) -> str:
    rows = []
    for target in TARGETS:
        if target.role != role:
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(target.label)}</td>"
            f"<td><code>{html.escape(target.column)}</code></td>"
            f"<td>{html.escape(target.description)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>target</th><th>column</th><th>meaning</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def target_description_html() -> str:
    return targets_table_html("primary")


def format_float(value: Any, digits: int = 3, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}{suffix}"


def simple_table_html(frame: pd.DataFrame, columns: list[tuple[str, str, int, str]]) -> str:
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label, _, _ in columns)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for key, _, digits, suffix in columns:
            value = row.get(key)
            if isinstance(value, str):
                rendered = html.escape(value)
            else:
                rendered = format_float(value, digits, suffix)
            cells.append(f'<td data-sort-value="{sort_value(value)}">{rendered}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def build_report_html(
    loss_fig: go.Figure,
    fit_fig: go.Figure,
    best: pd.DataFrame,
    summary: pd.DataFrame,
    one_e22: pd.DataFrame,
    cache_check: pd.DataFrame,
    failures: list[FailedFit],
    input_path: str,
    fit_through_scale: str,
) -> str:
    loss_html = loss_fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="k020-loss-comparison")
    fit_html = fit_fig.to_html(include_plotlyjs=False, full_html=False, div_id="k020-fit-comparison")
    best_columns = [
        ("target_label", "target", 3, ""),
        ("target_column", "column", 3, ""),
        ("model_label", "best heldout model", 3, ""),
        ("resource", "resource", 3, ""),
        ("heldout_mae_pct", "heldout MAE", 2, "%"),
        ("1e22_mae_pct", "1e22 MAE", 2, "%"),
        ("1e22_050_actual", "1e22 lr0.50 actual", 5, ""),
        ("1e22_050_prediction", "1e22 lr0.50 pred", 5, ""),
        ("1e22_050_error_pct", "1e22 lr0.50 err", 2, "%"),
        ("1e22_067_actual", "1e22 lr0.67 actual", 5, ""),
        ("1e22_067_prediction", "1e22 lr0.67 pred", 5, ""),
        ("1e22_067_error_pct", "1e22 lr0.67 err", 2, "%"),
    ]
    summary_columns = [
        ("target_label", "target", 3, ""),
        ("model_label", "model", 3, ""),
        ("family", "family", 3, ""),
        ("resource", "resource", 3, ""),
        ("heldout_n", "heldout n", 0, ""),
        ("heldout_mae_pct", "heldout MAE", 2, "%"),
        ("heldout_rmse_pct", "heldout RMSE", 2, "%"),
        ("1e22_mae_pct", "1e22 MAE", 2, "%"),
        ("1e22_050_prediction", "1e22 lr0.50 pred", 5, ""),
        ("1e22_050_error_pct", "1e22 lr0.50 err", 2, "%"),
        ("1e22_067_prediction", "1e22 lr0.67 pred", 5, ""),
        ("1e22_067_error_pct", "1e22 lr0.67 err", 2, "%"),
    ]
    one_e22_columns = [
        ("target_label", "target", 3, ""),
        ("model_label", "model", 3, ""),
        ("family", "family", 3, ""),
        ("resource", "resource", 3, ""),
        ("lr", "LR", 2, ""),
        (TARGET_METRIC, "actual", 5, ""),
        ("prediction", "prediction", 5, ""),
        ("error_pct", "signed error", 2, "%"),
        ("abs_error_pct", "abs error", 2, "%"),
        ("run", "run", 3, ""),
    ]
    cache_columns = [
        ("scale", "scale", 3, ""),
        ("lr", "LR", 2, ""),
        ("same_sweep_old_4plus_loss", "same-sweep old 4plus", 6, ""),
        ("cached_old_math_val_loss", "cached old math val", 6, ""),
        ("delta", "delta", 6, ""),
        ("delta_pct", "delta", 3, "%"),
    ]
    failure_html = failure_table_html(failures)
    cache_html = (
        "<p>No cached old-val overlap found.</p>" if cache_check.empty else simple_table_html(cache_check, cache_columns)
    )
    primary_best = best[best["target_key"].isin(PRIMARY_TARGET_KEYS)].reset_index(drop=True)
    primary_summary = summary[summary["target_key"].isin(PRIMARY_TARGET_KEYS)].reset_index(drop=True)
    primary_one_e22 = one_e22[one_e22["target_label"].isin(primary_best["target_label"])].reset_index(drop=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Delphi K=0.20 Old vs Clean Validation Fits</title>
  <style>
    :root {{
      color-scheme: light;
      --text: #172033;
      --muted: #5f6b7a;
      --border: #d8dee8;
      --panel: #f7f9fc;
      --accent: #1877f2;
    }}
    body {{
      margin: 0;
      color: var(--text);
      background: #fff;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{
      max-width: 1580px;
      margin: 0 auto;
      padding: 28px 32px 48px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 24px 0 10px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 10px 0 8px;
      font-size: 15px;
      letter-spacing: 0;
    }}
    p {{
      margin: 0 0 10px;
    }}
    code {{
      padding: 1px 4px;
      border-radius: 4px;
      background: #edf1f7;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.95em;
    }}
    .lede {{
      max-width: 1160px;
      color: var(--muted);
      font-size: 15px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(340px, 0.9fr) minmax(520px, 1.1fr);
      gap: 14px;
      margin: 20px 0 18px;
    }}
    .panel {{
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px 16px;
    }}
    .panel ul {{
      margin: 0;
      padding-left: 18px;
    }}
    .panel li {{
      margin: 5px 0;
    }}
    .callout {{
      border-left: 4px solid var(--accent);
      background: #f2f7ff;
      padding: 12px 14px;
      margin: 18px 0;
      max-width: 1180px;
    }}
    .table-wrap {{
      overflow-x: auto;
      margin: 8px 0 20px;
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
      background: #fff;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      background: #f8fafc;
      position: sticky;
      top: 0;
    }}
    table.sortable th {{
      cursor: pointer;
      user-select: none;
    }}
    table.sortable th button {{
      appearance: none;
      border: 0;
      padding: 0;
      margin: 0;
      background: transparent;
      color: inherit;
      font: inherit;
      cursor: pointer;
      text-align: left;
      white-space: nowrap;
    }}
    .sort-indicator {{
      display: inline-block;
      width: 3ch;
      margin-left: 5px;
      color: var(--accent);
    }}
    th[aria-sort="ascending"] .sort-indicator::after {{
      content: "asc";
    }}
    th[aria-sort="descending"] .sort-indicator::after {{
      content: "desc";
    }}
    details {{
      margin: 18px 0;
    }}
    summary {{
      cursor: pointer;
      font-weight: 600;
    }}
    @media (max-width: 900px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
      main {{
        padding: 22px 16px 40px;
      }}
    }}
  </style>
</head>
<body>
<main>
  <h1>Delphi K=0.20 old-vs-clean validation fits</h1>
  <p class="lede">
    Same p33m67 K=0.20 checkpoints, same train/heldout split, two primary validation targets.
    Input: <code>{html.escape(input_path)}</code>. Fits train through <code>{html.escape(fit_through_scale)}</code>;
    <code>1e21</code> and <code>1e22</code> are held out.
  </p>
  <div class="callout">
    <strong>Primary comparison:</strong> old 4plus math validation versus new clean-seen validation.
    The old 4plus metric is the one used in the earlier K=0.20 plots. Other emitted eval metrics are hidden from
    this page so the comparison stays focused.
  </div>
  <div class="grid">
    <section class="panel">
      <h2>Fit Setup</h2>
      <ul>
        <li>Rows: 36 = 9 scales x 4 LR factors.</li>
        <li>Training rows: scales through <code>3e20</code>.</li>
        <li>Heldout rows: <code>1e21</code> and <code>1e22</code>.</li>
        <li>Primary targets: <code>clean_seen_loss</code> and <code>anchor_4plus_loss</code>.</li>
        <li>Fit families: same scale/LR families as the clean-seen report; anchor-calibration fits are excluded so both targets use the same comparable models.</li>
      </ul>
    </section>
    <section class="panel">
      <h2>Targets</h2>
      {target_description_html()}
    </section>
  </div>
  <h2>Loss Curves</h2>
  {loss_html}
  <h2>Primary Comparison</h2>
  <p class="lede">
    This is the two-row answer to the question: same checkpoints and fit code, old 4plus target versus new clean-seen target.
  </p>
  <div class="table-wrap">
    {table_html(primary_best, best_columns, default_sort_key="heldout_mae_pct")}
  </div>
  <h2>Fit Explorer: Primary Targets Only</h2>
  {fit_html}
  <details open>
    <summary>Primary model summaries</summary>
    <div class="table-wrap">
      {table_html(primary_summary, summary_columns, default_sort_key="heldout_mae_pct")}
    </div>
  </details>
  <details open>
    <summary>Primary 1e22 predictions</summary>
    <div class="table-wrap">
      {table_html(primary_one_e22, one_e22_columns, default_sort_key="abs_error_pct")}
    </div>
  </details>
  <details>
    <summary>Cached old-val sanity check</summary>
    <p class="lede">
      Compares the same-sweep old 4plus anchor against the older cached <code>math_val_loss</code> values where
      cells overlap. Small deltas here mean the comparison is not explained by a cache/key mismatch.
    </p>
    <div class="table-wrap">{cache_html}</div>
  </details>
  <details>
    <summary>Fit failures</summary>
    <div class="table-wrap">{failure_html}</div>
  </details>
</main>
<script>
  function cellSortValue(row, index) {{
    const cell = row.cells[index];
    if (!cell) {{
      return {{missing: true, text: "", number: Number.NaN}};
    }}
    const raw = cell.dataset.sortValue || "";
    if (raw === "") {{
      return {{missing: true, text: "", number: Number.NaN}};
    }}
    const number = Number(raw);
    return {{
      missing: false,
      text: raw,
      number: Number.isFinite(number) ? number : Number.NaN,
    }};
  }}

  function compareRows(left, right, index, direction) {{
    const leftValue = cellSortValue(left, index);
    const rightValue = cellSortValue(right, index);
    if (leftValue.missing && rightValue.missing) {{
      return 0;
    }}
    if (leftValue.missing) {{
      return 1;
    }}
    if (rightValue.missing) {{
      return -1;
    }}
    const bothNumeric = Number.isFinite(leftValue.number) && Number.isFinite(rightValue.number);
    const result = bothNumeric
      ? leftValue.number - rightValue.number
      : leftValue.text.localeCompare(rightValue.text, undefined, {{numeric: true, sensitivity: "base"}});
    return direction === "asc" ? result : -result;
  }}

  function sortTable(table, index, direction) {{
    const body = table.tBodies[0];
    const rows = Array.from(body.rows);
    rows.sort((left, right) => compareRows(left, right, index, direction));
    for (const row of rows) {{
      body.appendChild(row);
    }}
    for (const header of table.tHead.rows[0].cells) {{
      header.removeAttribute("aria-sort");
      header.dataset.direction = "";
    }}
    const activeHeader = table.tHead.rows[0].cells[index];
    activeHeader.setAttribute("aria-sort", direction === "asc" ? "ascending" : "descending");
    activeHeader.dataset.direction = direction;
  }}

  function installSortableTables() {{
    for (const table of document.querySelectorAll("table.sortable")) {{
      const headers = Array.from(table.tHead.rows[0].cells);
      for (const header of headers) {{
        const index = Number(header.dataset.sortIndex);
        header.addEventListener("click", () => {{
          const current = header.dataset.direction;
          const next = current === "asc" ? "desc" : "asc";
          sortTable(table, index, next);
        }});
      }}
      const defaultIndex = Number(table.dataset.defaultSortIndex);
      if (Number.isInteger(defaultIndex)) {{
        sortTable(table, defaultIndex, table.dataset.defaultSortDirection || "asc");
      }}
    }}
  }}

  installSortableTables();
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    base_points = load_clean_seen_summary(args.input)
    predictions, fit_table, failures = fit_all_targets(base_points, args.fit_through_scale)
    summary = summarize_predictions(predictions, fit_table)
    best = best_by_target(summary)
    primary_summary = summary[summary["target_key"].isin(PRIMARY_TARGET_KEYS)].reset_index(drop=True)
    primary_predictions = predictions[predictions["target_key"].isin(PRIMARY_TARGET_KEYS)].reset_index(drop=True)
    one_e22 = one_e22_predictions(predictions)
    cache_check = cached_old_val_check(base_points, args.old_endpoints, args.old_isoflop)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / args.output_stem
    write_csv(prefix.with_name(f"{prefix.name}_points.csv"), base_points)
    write_csv(prefix.with_name(f"{prefix.name}_predictions.csv"), predictions)
    write_csv(prefix.with_name(f"{prefix.name}_summary.csv"), summary)
    write_csv(prefix.with_name(f"{prefix.name}_best_by_target.csv"), best)
    write_csv(prefix.with_name(f"{prefix.name}_primary_summary.csv"), primary_summary)
    write_csv(prefix.with_name(f"{prefix.name}_1e22_predictions.csv"), one_e22)
    write_csv(prefix.with_name(f"{prefix.name}_cached_old_val_check.csv"), cache_check)
    write_json(
        prefix.with_name(f"{prefix.name}_fit.json"),
        {
            "input": args.input,
            "fit_through_scale": args.fit_through_scale,
            "targets": [target.__dict__ for target in TARGETS],
            "fit_table": records_for_json(fit_table),
            "summary": records_for_json(summary),
            "failures": [
                {"target_or_model": failure.spec.label, "family": failure.spec.family, "error": failure.error}
                for failure in failures
            ],
        },
    )
    html_path = prefix.with_suffix(".html")
    write_text(
        html_path,
        build_report_html(
            loss_comparison_figure(base_points),
            fit_dropdown_figure(primary_predictions, primary_summary),
            best,
            summary,
            one_e22,
            cache_check,
            failures,
            args.input,
            args.fit_through_scale,
        ),
    )
    print(best.to_string(index=False))
    print()
    print(one_e22.head(24).to_string(index=False))
    print()
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
