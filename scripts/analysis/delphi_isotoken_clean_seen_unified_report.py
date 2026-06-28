# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a unified old-4plus vs clean-seen report for Delphi iso-token reruns.

This compares the trusted p33m67/lr0.50 iso-token clean-seen evals for 1B, 2B,
4B, and 8B midtraining token budgets against the K=0.20 iso-FLOP ladder.

Run:
    uv run --with scipy --with plotly --with pandas --with gcsfs \
      python scripts/analysis/delphi_isotoken_clean_seen_unified_report.py
"""

from __future__ import annotations

import argparse
import html
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from build_delphi_midtraining_interactive_report import fit_floor_power, floor_power_model
from delphi_isotoken_endpoint_scaling import ALL_SCALE_FLOPS, DEFAULT_CUTOFF_SCALE, HELD_OUT_SCALES, SCALE_ORDER
from delphi_k020_clean_seen_fit_family_report import records_for_json, table_html, write_csv, write_json, write_text
from delphi_small_final_loss_scaling import MATH_FRACTION, MIDTRAIN_BUDGET_FRACTION, SCALE_PRETRAIN_TOKENS_B
from marin.scaling_laws.scaling_plots import MARKERS, PALETTE
from plotly.subplots import make_subplots

DEFAULT_ISOTOKEN_INPUT = (
    "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/"
    "evals_clean_seen_1e22_isotoken_p33m67_lr50/summary_p33m67_isotoken_clean_seen_1e22.csv"
)
DEFAULT_K020_INPUT = (
    "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/"
    "evals_clean_seen_1e22_k020/summary_p33m67_clean_seen_1e22_k020.csv"
)
DEFAULT_OUTPUT_DIR = Path("sk_midtrain_analysis_fable")
DEFAULT_OUTPUT_STEM = "delphi_isotoken_clean_seen_unified_report"

MIX = "p33m67"
LR_FACTOR = 0.50
MATH_MIX_FRACTION = MATH_FRACTION[MIX]
TRUSTED_ISOTOKEN_BUDGETS = ("1b", "2b", "4b", "8b")
FIT_MODEL_LABEL = "per-series Chinchilla on pretraining FLOPs"

PER_RUN_RESULT_PATHS = (
    "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/"
    "evals_clean_seen_1e22_isotoken_p33m67_lr50/"
    "delphi-1e22-p33m67-tok1b-lr50-a008/step-237/metrics.jsonl/eval_results.json",
    "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/"
    "evals_clean_seen_1e22_isotoken_p33m67_lr50/"
    "delphi-1e22-p33m67-tok2b-lr50-a003/step-476/metrics.jsonl/eval_results.json",
    "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/"
    "evals_clean_seen_1e22_isotoken_p33m67_lr50/"
    "delphi-1e22-p33m67-tok4b-lr50-a002/step-953/metrics.jsonl/eval_results.json",
    "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/"
    "evals_clean_seen_1e22_isotoken_p33m67_lr50/"
    "delphi-1e22-p33m67-tok8b-lr50-a001/step-1906/metrics.jsonl/eval_results.json",
)


@dataclass(frozen=True)
class TargetSpec:
    key: str
    label: str
    column: str
    source_column: str
    description: str


@dataclass(frozen=True)
class SeriesSpec:
    key: str
    label: str
    color: str
    marker: str
    short_label: str
    description: str


TARGETS = (
    TargetSpec(
        key="old_4plus",
        label="old 4plus validation",
        column="old_4plus_loss",
        source_column="anchor_4plus_loss",
        description=(
            "The earlier Nemotron math 4plus validation slice, measured in the same new eval jobs. "
            "This is the old target used by the iso-token PNG."
        ),
    ),
    TargetSpec(
        key="clean_seen",
        label="new clean-seen validation",
        column="clean_seen_loss",
        source_column="clean_seen_loss",
        description="The new decontaminated clean-seen 1e22 validation loss.",
    ),
)

SERIES = (
    SeriesSpec(
        key="tok1b",
        label="iso-token 1B",
        short_label="1B",
        color="#6baed6",
        marker=MARKERS[0],
        description="Fixed 1B total midtraining tokens at p33m67/lr0.50.",
    ),
    SeriesSpec(
        key="tok2b",
        label="iso-token 2B",
        short_label="2B",
        color="#3182bd",
        marker=MARKERS[1],
        description="Fixed 2B total midtraining tokens at p33m67/lr0.50.",
    ),
    SeriesSpec(
        key="tok4b",
        label="iso-token 4B",
        short_label="4B",
        color="#08519c",
        marker=MARKERS[2],
        description="Fixed 4B total midtraining tokens at p33m67/lr0.50.",
    ),
    SeriesSpec(
        key="tok8b",
        label="iso-token 8B",
        short_label="8B",
        color="#08306b",
        marker=MARKERS[3],
        description="Fixed 8B total midtraining tokens at p33m67/lr0.50.",
    ),
    SeriesSpec(
        key="k0p20",
        label="K=0.20 iso-FLOP",
        short_label="K=0.20",
        color=PALETTE[1],
        marker=MARKERS[4],
        description=(
            "The original K=0.20 ladder at p33m67/lr0.50. Its token budget grows with scale; "
            "the 1e22 point uses about 32B total midtraining tokens."
        ),
    ),
)
SERIES_BY_KEY = {series.key: series for series in SERIES}
TARGET_BY_KEY = {target.key: target for target in TARGETS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isotoken-input", default=DEFAULT_ISOTOKEN_INPUT)
    parser.add_argument("--k020-input", default=DEFAULT_K020_INPUT)
    parser.add_argument(
        "--fit-through-scale",
        choices=SCALE_ORDER[:-1],
        default=DEFAULT_CUTOFF_SCALE,
        help="Largest scale included in the training split.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    return parser.parse_args()


def scale_label_for(value: Any) -> str:
    if isinstance(value, str) and value in ALL_SCALE_FLOPS:
        return value
    numeric = float(value)
    for label, flops in ALL_SCALE_FLOPS.items():
        if math.isclose(numeric, flops, rel_tol=1e-9):
            return label
    raise ValueError(f"Unknown scale value: {value!r}")


def budget_tokens_b(budget: str) -> float:
    if not budget.endswith("b"):
        raise ValueError(f"Only billion-token budgets are expected here, got {budget!r}")
    return float(budget.removesuffix("b"))


def normalize_common(frame: pd.DataFrame, fit_through_scale: str) -> pd.DataFrame:
    cutoff = ALL_SCALE_FLOPS[fit_through_scale]
    out = frame.copy()
    out["scale"] = out["scale"].map(scale_label_for)
    out["scale_flops"] = out["scale"].map(ALL_SCALE_FLOPS).astype(float)
    out["scale_order"] = out["scale"].map({scale: index for index, scale in enumerate(SCALE_ORDER)})
    out["split"] = np.where(out["scale_flops"] <= cutoff + 1.0, "fit", "heldout")
    out["midtrain_math_tokens_b"] = out["midtrain_tokens_b"] * MATH_MIX_FRACTION
    return out.sort_values(["series_order", "scale_order"]).reset_index(drop=True)


def load_isotoken_points(path: str, fit_through_scale: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "scale",
        "budget",
        "run",
        "step",
        "complete_by_step",
        "actual_tokens",
        "clean_seen_loss",
        "anchor_4plus_loss",
        "eval_results_path",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required iso-token columns: {missing}")

    out = frame[frame["budget"].isin(TRUSTED_ISOTOKEN_BUDGETS)].copy()
    incomplete = out[~out["complete_by_step"].astype(bool)]
    if not incomplete.empty:
        bad = incomplete[["scale", "budget", "run", "step"]].to_dict(orient="records")
        raise ValueError(f"Trusted iso-token rows include incomplete evals: {bad}")

    out["series"] = "tok" + out["budget"].astype(str)
    out["series_order"] = out["series"].map({series.key: index for index, series in enumerate(SERIES)})
    out["series_label"] = out["series"].map(lambda key: SERIES_BY_KEY[key].label)
    out["series_short_label"] = out["series"].map(lambda key: SERIES_BY_KEY[key].short_label)
    out["source_family"] = "iso-token"
    out["nominal_midtrain_tokens_b"] = out["budget"].map(budget_tokens_b)
    out["midtrain_tokens_b"] = out["actual_tokens"].astype(float) / 1e9
    out["lr_factor"] = LR_FACTOR
    out["old_4plus_loss"] = out["anchor_4plus_loss"].astype(float)
    out["clean_seen_loss"] = out["clean_seen_loss"].astype(float)
    out["notes"] = "trusted clean-seen iso-token row"
    return normalize_common(out, fit_through_scale)


def load_k020_points(path: str, fit_through_scale: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"scale", "lr_factor", "run", "step", "clean_seen_loss", "anchor_4plus_loss", "eval_results_path"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required K=0.20 columns: {missing}")

    out = frame[np.isclose(frame["lr_factor"].astype(float), LR_FACTOR)].copy()
    if out.empty:
        raise ValueError(f"No K=0.20 rows found for lr_factor={LR_FACTOR}")
    out["scale"] = out["scale"].map(scale_label_for)
    out["series"] = "k0p20"
    out["series_order"] = len(SERIES) - 1
    out["series_label"] = SERIES_BY_KEY["k0p20"].label
    out["series_short_label"] = SERIES_BY_KEY["k0p20"].short_label
    out["source_family"] = "iso-FLOP"
    out["budget"] = "k0p20"
    out["actual_tokens"] = np.nan
    out["nominal_midtrain_tokens_b"] = out["scale"].map(SCALE_PRETRAIN_TOKENS_B).astype(float) * MIDTRAIN_BUDGET_FRACTION
    out["midtrain_tokens_b"] = out["nominal_midtrain_tokens_b"]
    out["old_4plus_loss"] = out["anchor_4plus_loss"].astype(float)
    out["clean_seen_loss"] = out["clean_seen_loss"].astype(float)
    out["notes"] = "K=0.20 comparison row at lr0.50"
    return normalize_common(out, fit_through_scale)


def load_points(isotoken_path: str, k020_path: str, fit_through_scale: str) -> pd.DataFrame:
    points = pd.concat(
        [load_isotoken_points(isotoken_path, fit_through_scale), load_k020_points(k020_path, fit_through_scale)],
        ignore_index=True,
    )
    expected = set(SCALE_ORDER)
    for series in SERIES:
        found = set(points.loc[points["series"].eq(series.key), "scale"])
        if found != expected:
            raise ValueError(f"Series {series.key} has scales {sorted(found)} but expected {sorted(expected)}")
    return points.sort_values(["series_order", "scale_order"]).reset_index(drop=True)


def predict_floor_power(fit: dict[str, float], xs: np.ndarray) -> np.ndarray:
    return floor_power_model(xs / 1e18, fit["floor"], fit["amplitude"], fit["alpha"])


def fit_all(points: pd.DataFrame, fit_through_scale: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    fit_rows: list[dict[str, Any]] = []

    for target in TARGETS:
        for series in SERIES:
            frame = points[points["series"].eq(series.key)].sort_values("scale_flops")
            train = frame[frame["scale_flops"] <= ALL_SCALE_FLOPS[fit_through_scale] + 1.0]
            fit = fit_floor_power(
                train["scale_flops"].to_numpy(dtype=float),
                train[target.column].to_numpy(dtype=float),
            )
            if fit is None:
                raise ValueError(f"Fit failed for {series.key}/{target.key}")

            predictions = frame.copy()
            predictions["target_key"] = target.key
            predictions["target_label"] = target.label
            predictions["target_column"] = target.column
            predictions["actual"] = predictions[target.column].astype(float)
            predictions["prediction"] = predict_floor_power(fit, predictions["scale_flops"].to_numpy(dtype=float))
            predictions["error"] = predictions["prediction"] - predictions["actual"]
            predictions["error_pct"] = (predictions["prediction"] / predictions["actual"] - 1.0) * 100.0
            predictions["abs_error_pct"] = predictions["error_pct"].abs()
            prediction_frames.append(predictions)

            fit_rows.append(
                {
                    "target_key": target.key,
                    "target_label": target.label,
                    "target_column": target.column,
                    "series": series.key,
                    "series_label": series.label,
                    "model": FIT_MODEL_LABEL,
                    "fit_through_scale": fit_through_scale,
                    "train_n": int(fit["n"]),
                    "fit_r2": fit["r2"],
                    "fit_rmse": fit["rmse"],
                    "floor": fit["floor"],
                    "amplitude": fit["amplitude"],
                    "alpha": fit["alpha"],
                }
            )

    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(fit_rows)


def heldout_summary(predictions: pd.DataFrame, fit_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (target_key, series_key), frame in predictions.groupby(["target_key", "series"], sort=False):
        held = frame[frame["scale"].isin(HELD_OUT_SCALES)].copy()
        one_e21 = held[held["scale"].eq("1e21")]
        one_e22 = held[held["scale"].eq("1e22")]
        fit = fit_table[fit_table["target_key"].eq(target_key) & fit_table["series"].eq(series_key)].iloc[0]
        errors = held["error_pct"].to_numpy(dtype=float)
        row = {
            "target_key": target_key,
            "target_label": TARGET_BY_KEY[target_key].label,
            "series": series_key,
            "series_label": SERIES_BY_KEY[series_key].label,
            "model": FIT_MODEL_LABEL,
            "train_n": int(fit["train_n"]),
            "fit_r2": fit["fit_r2"],
            "fit_rmse": fit["fit_rmse"],
            "heldout_mae_pct": float(np.mean(np.abs(errors))),
            "heldout_bias_pct": float(np.mean(errors)),
            "actual_1e21": float(one_e21["actual"].iloc[0]),
            "pred_1e21": float(one_e21["prediction"].iloc[0]),
            "error_1e21_pct": float(one_e21["error_pct"].iloc[0]),
            "actual_1e22": float(one_e22["actual"].iloc[0]),
            "pred_1e22": float(one_e22["prediction"].iloc[0]),
            "error_1e22_pct": float(one_e22["error_pct"].iloc[0]),
            "abs_error_1e22_pct": abs(float(one_e22["error_pct"].iloc[0])),
            "floor": float(fit["floor"]),
            "amplitude": float(fit["amplitude"]),
            "alpha": float(fit["alpha"]),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["target_key", "abs_error_1e22_pct", "series_label"]).reset_index(drop=True)


def one_e22_actuals(points: pd.DataFrame) -> pd.DataFrame:
    rows = points[points["scale"].eq("1e22")].copy()
    rows["clean_minus_old"] = rows["clean_seen_loss"] - rows["old_4plus_loss"]
    rows["clean_over_old_pct"] = (rows["clean_seen_loss"] / rows["old_4plus_loss"] - 1.0) * 100.0
    rows["nominal_midtrain_math_tokens_b"] = rows["nominal_midtrain_tokens_b"] * MATH_MIX_FRACTION
    return rows[
        [
            "series_label",
            "source_family",
            "nominal_midtrain_tokens_b",
            "nominal_midtrain_math_tokens_b",
            "old_4plus_loss",
            "clean_seen_loss",
            "clean_minus_old",
            "clean_over_old_pct",
            "run",
            "step",
        ]
    ].sort_values("nominal_midtrain_tokens_b")


def target_definition_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target": target.label,
                "report_column": target.column,
                "source_column": target.source_column,
                "meaning": target.description,
            }
            for target in TARGETS
        ]
    )


def series_definition_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "series": series.label,
                "included_scales": "3e18 through 1e22",
                "fit_train_scales": "3e18 through 3e20",
                "meaning": series.description,
            }
            for series in SERIES
        ]
    )


def scaling_figure(points: pd.DataFrame, fit_table: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[target.label for target in TARGETS],
        horizontal_spacing=0.08,
    )
    xs = np.logspace(math.log10(min(ALL_SCALE_FLOPS.values())), math.log10(max(ALL_SCALE_FLOPS.values())), 240)
    for col, target in enumerate(TARGETS, start=1):
        for series in SERIES:
            spec_frame = points[points["series"].eq(series.key)].sort_values("scale_flops")
            customdata = np.column_stack(
                [
                    spec_frame["scale"],
                    spec_frame["run"],
                    spec_frame["nominal_midtrain_tokens_b"],
                    spec_frame["midtrain_math_tokens_b"],
                    spec_frame["split"],
                ]
            )
            fig.add_trace(
                go.Scatter(
                    x=spec_frame["scale_flops"],
                    y=spec_frame[target.column],
                    mode="lines+markers",
                    name=f"{series.label} actual",
                    legendgroup=series.key,
                    showlegend=col == 1,
                    line=dict(color=series.color, width=2),
                    marker=dict(symbol=series.marker, size=8, color=series.color),
                    customdata=customdata,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        f"{html.escape(series.label)}<br>"
                        f"{html.escape(target.label)}=%{{y:.6f}}<br>"
                        "total midtrain tokens=%{customdata[2]:.3f}B<br>"
                        "math midtrain tokens=%{customdata[3]:.3f}B<br>"
                        "split=%{customdata[4]}<br>"
                        "run=%{customdata[1]}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
            fit = fit_table[fit_table["target_key"].eq(target.key) & fit_table["series"].eq(series.key)].iloc[0]
            fit_payload = {"floor": fit["floor"], "amplitude": fit["amplitude"], "alpha": fit["alpha"]}
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=predict_floor_power(fit_payload, xs),
                    mode="lines",
                    name=f"{series.label} fit",
                    legendgroup=series.key,
                    showlegend=False,
                    line=dict(color=series.color, dash="dot", width=2),
                    hovertemplate=(
                        f"{html.escape(series.label)} fit<br>"
                        f"{html.escape(target.label)}=%{{y:.6f}}<br>"
                        "scale=%{x:.2e}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )

    for col in (1, 2):
        fig.add_vline(x=ALL_SCALE_FLOPS[DEFAULT_CUTOFF_SCALE], line_dash="dash", line_color="#94a3b8", row=1, col=col)
        fig.update_xaxes(type="log", title_text="base pretraining FLOPs", row=1, col=col)
        fig.update_yaxes(title_text="loss", row=1, col=col)
    fig.update_layout(
        template="plotly_white",
        height=560,
        margin=dict(l=55, r=35, t=70, b=60),
        legend=dict(orientation="h", y=-0.18, x=0),
    )
    return fig


def token_budget_figure(points: pd.DataFrame) -> go.Figure:
    one_e22 = points[points["scale"].eq("1e22")].copy().sort_values("nominal_midtrain_tokens_b")
    one_e22["clean_minus_old"] = one_e22["clean_seen_loss"] - one_e22["old_4plus_loss"]
    customdata = np.column_stack([one_e22["series_label"], one_e22["run"], one_e22["step"]])
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["1e22 actual losses", "1e22 clean-seen minus old 4plus"],
        horizontal_spacing=0.12,
    )
    for target, color in (("old_4plus_loss", "#475569"), ("clean_seen_loss", "#1877F2")):
        label = "old 4plus validation" if target == "old_4plus_loss" else "new clean-seen validation"
        fig.add_trace(
            go.Scatter(
                x=one_e22["nominal_midtrain_tokens_b"],
                y=one_e22[target],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=9, color=color),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    f"{html.escape(label)}=%{{y:.6f}}<br>"
                    "total midtraining tokens=%{x:.3f}B<br>"
                    "run=%{customdata[1]}<br>"
                    "step=%{customdata[2]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=one_e22["nominal_midtrain_tokens_b"],
            y=one_e22["clean_minus_old"],
            mode="lines+markers",
            name="clean - old",
            line=dict(color="#E42C97", width=2),
            marker=dict(size=9, color="#E42C97"),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>" "delta=%{y:.6f}<br>" "total midtraining tokens=%{x:.3f}B<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    for col in (1, 2):
        fig.update_xaxes(type="log", title_text="nominal total midtraining tokens (B)", row=1, col=col)
        fig.update_yaxes(title_text="loss" if col == 1 else "loss delta", row=1, col=col)
    fig.update_layout(
        template="plotly_white",
        height=480,
        margin=dict(l=55, r=35, t=70, b=60),
        legend=dict(orientation="h", y=-0.20, x=0),
    )
    return fig


def delta_figure(points: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for series in SERIES:
        frame = points[points["series"].eq(series.key)].sort_values("scale_flops").copy()
        frame["clean_minus_old"] = frame["clean_seen_loss"] - frame["old_4plus_loss"]
        fig.add_trace(
            go.Scatter(
                x=frame["scale_flops"],
                y=frame["clean_minus_old"],
                mode="lines+markers",
                name=series.label,
                line=dict(color=series.color, width=2),
                marker=dict(symbol=series.marker, size=8, color=series.color),
                customdata=np.column_stack([frame["scale"], frame["nominal_midtrain_tokens_b"]]),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    f"{html.escape(series.label)}<br>"
                    "clean - old=%{y:.6f}<br>"
                    "total midtrain tokens=%{customdata[1]:.3f}B<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        template="plotly_white",
        height=430,
        margin=dict(l=55, r=35, t=40, b=60),
        legend=dict(orientation="h", y=-0.25, x=0),
        xaxis=dict(type="log", title="base pretraining FLOPs"),
        yaxis=dict(title="clean-seen loss minus old 4plus loss"),
    )
    return fig


def source_html(isotoken_input: str, k020_input: str) -> str:
    per_run_items = "\n".join(f"<li><code>{html.escape(path)}</code></li>" for path in PER_RUN_RESULT_PATHS)
    return f"""
      <details>
        <summary>Source files</summary>
        <ul>
          <li>Iso-token clean-seen summary: <code>{html.escape(isotoken_input)}</code></li>
          <li>K=0.20 clean-seen summary: <code>{html.escape(k020_input)}</code></li>
          {per_run_items}
        </ul>
      </details>
    """


def render_html(
    *,
    points: pd.DataFrame,
    fit_table: pd.DataFrame,
    summary: pd.DataFrame,
    one_e22: pd.DataFrame,
    isotoken_input: str,
    k020_input: str,
    fit_through_scale: str,
) -> str:
    scaling_html = scaling_figure(points, fit_table).to_html(
        include_plotlyjs="cdn", full_html=False, div_id="scaling-fits"
    )
    token_budget_html = token_budget_figure(points).to_html(
        include_plotlyjs=False,
        full_html=False,
        div_id="one-e22-token-budget",
    )
    delta_html = delta_figure(points).to_html(include_plotlyjs=False, full_html=False, div_id="clean-old-delta")

    target_columns = [
        ("target", "target", 3, ""),
        ("report_column", "report column", 3, ""),
        ("source_column", "source column", 3, ""),
        ("meaning", "meaning", 3, ""),
    ]
    series_columns = [
        ("series", "series", 3, ""),
        ("included_scales", "included scales", 3, ""),
        ("fit_train_scales", "fit train scales", 3, ""),
        ("meaning", "meaning", 3, ""),
    ]
    one_e22_columns = [
        ("series_label", "series", 3, ""),
        ("source_family", "family", 3, ""),
        ("nominal_midtrain_tokens_b", "total tokens B", 3, ""),
        ("nominal_midtrain_math_tokens_b", "math tokens B", 3, ""),
        ("old_4plus_loss", "old 4plus loss", 6, ""),
        ("clean_seen_loss", "clean-seen loss", 6, ""),
        ("clean_minus_old", "clean - old", 6, ""),
        ("clean_over_old_pct", "clean/old minus 1", 2, "%"),
        ("run", "run", 3, ""),
        ("step", "step", 0, ""),
    ]
    summary_columns = [
        ("target_label", "target", 3, ""),
        ("series_label", "series", 3, ""),
        ("train_n", "train n", 0, ""),
        ("fit_r2", "fit R2", 4, ""),
        ("fit_rmse", "fit RMSE", 6, ""),
        ("actual_1e21", "1e21 actual", 6, ""),
        ("pred_1e21", "1e21 pred", 6, ""),
        ("error_1e21_pct", "1e21 err", 2, "%"),
        ("actual_1e22", "1e22 actual", 6, ""),
        ("pred_1e22", "1e22 pred", 6, ""),
        ("error_1e22_pct", "1e22 err", 2, "%"),
        ("heldout_mae_pct", "heldout MAE", 2, "%"),
        ("alpha", "alpha", 4, ""),
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Delphi iso-token clean-seen validation report</title>
  <style>
    :root {{
      --text: #162033;
      --muted: #5f6b7a;
      --border: #d8dee8;
      --panel: #f8fafc;
      --accent: #1877f2;
    }}
    body {{
      margin: 0;
      color: var(--text);
      background: #ffffff;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 28px 32px 48px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 28px 0 10px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    p {{
      max-width: 1120px;
      color: var(--muted);
    }}
    code {{
      background: #eef2f7;
      padding: 1px 5px;
      border-radius: 5px;
      color: #1f2937;
    }}
    .lede {{
      font-size: 15px;
      margin-bottom: 16px;
    }}
    .callout {{
      border-left: 4px solid var(--accent);
      background: #f2f7ff;
      padding: 12px 14px;
      margin: 16px 0;
      max-width: 1180px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 18px;
      align-items: start;
    }}
    .panel {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
      background: var(--panel);
    }}
    .panel h2 {{
      margin-top: 0;
    }}
    .plot-wrap {{
      border-top: 1px solid var(--border);
      padding-top: 12px;
      margin-top: 12px;
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
    td:last-child {{
      white-space: normal;
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
      font-weight: 650;
    }}
    li {{
      margin: 5px 0;
    }}
  </style>
</head>
<body>
<main>
  <h1>Delphi iso-token clean-seen validation report</h1>
  <p class="lede">
    Unified p33m67/lr0.50 comparison for fixed-token iso-token reruns and the K=0.20 ladder.
    The fits are <code>L = E + A * (C/1e18)^(-alpha)</code>, trained through
    <code>{html.escape(fit_through_scale)}</code>; <code>1e21</code> and <code>1e22</code>
    are held out.
  </p>
  <div class="callout">
    <strong>Important:</strong> this report uses the completed canonical iso-token clean-seen
    summary with 36 rows: 9 scales x 4 fixed-token budgets. The corrected 1B/1e22 row is
    <code>delphi-1e22-p33m67-tok1b-lr50-a008</code> at step 237.
  </div>
  <div class="grid">
    <section class="panel">
      <h2>Targets</h2>
      <div class="table-wrap">
        {table_html(target_definition_table(), target_columns)}
      </div>
    </section>
    <section class="panel">
      <h2>Series</h2>
      <div class="table-wrap">
        {table_html(series_definition_table(), series_columns)}
      </div>
    </section>
  </div>
  <h2>Scaling Fits</h2>
  <p>
    Dotted lines are fits using the same Chinchilla floor-plus-power form as the earlier
    iso-token plot. Solid markers are actual eval losses. The vertical dashed line marks
    the last training scale for the fit.
  </p>
  <div class="plot-wrap">{scaling_html}</div>
  <h2>1e22 Token Budget View</h2>
  <p>
    This is the direct readout for the 1e22 question: fixed 1B, 2B, 4B, and 8B token budgets
    versus the K=0.20 1e22 point at about 32B total midtraining tokens.
  </p>
  <div class="plot-wrap">{token_budget_html}</div>
  <h2>Clean-Old Gap Across Scale</h2>
  <p>
    The gap is <code>clean_seen_loss - old_4plus_loss</code>. Larger positive values mean
    the old 4plus validation target was more optimistic relative to the clean-seen target.
  </p>
  <div class="plot-wrap">{delta_html}</div>
  <h2>1e22 Actual Losses</h2>
  <div class="table-wrap">
    {table_html(one_e22, one_e22_columns, default_sort_key="clean_seen_loss")}
  </div>
  <h2>Heldout Fit Errors</h2>
  <p>
    Error is <code>prediction / actual - 1</code>. Positive means the fit predicted a worse
    loss than the actual heldout loss.
  </p>
  <div class="table-wrap">
    {table_html(summary, summary_columns, default_sort_key="abs_error_1e22_pct")}
  </div>
  <details>
    <summary>Rows not shown</summary>
    <p>
      The old-validation-only PNG also included 500M iso-token runs. This report does not include
      500M because there is no clean-seen 1e22 500M row in the canonical iso-token clean-seen summary.
    </p>
  </details>
  {source_html(isotoken_input, k020_input)}
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


def build_outputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    points = load_points(args.isotoken_input, args.k020_input, args.fit_through_scale)
    predictions, fit_table = fit_all(points, args.fit_through_scale)
    summary = heldout_summary(predictions, fit_table)
    one_e22 = one_e22_actuals(points)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    points_path = args.output_dir / f"{args.output_stem}_points.csv"
    predictions_path = args.output_dir / f"{args.output_stem}_predictions.csv"
    summary_path = args.output_dir / f"{args.output_stem}_fit_summary.csv"
    json_path = args.output_dir / f"{args.output_stem}.json"
    html_path = args.output_dir / f"{args.output_stem}.html"

    write_csv(points_path, points)
    write_csv(predictions_path, predictions)
    write_csv(summary_path, summary)
    write_json(
        json_path,
        {
            "isotoken_input": args.isotoken_input,
            "k020_input": args.k020_input,
            "fit_through_scale": args.fit_through_scale,
            "trusted_isotoken_budgets": list(TRUSTED_ISOTOKEN_BUDGETS),
            "summary_note": "The canonical iso-token clean-seen summary has 36 rows and no partial rows.",
            "per_run_result_paths_checked": list(PER_RUN_RESULT_PATHS),
            "one_e22_actuals": records_for_json(one_e22),
            "fit_summary": records_for_json(summary),
        },
    )
    html_text = render_html(
        points=points,
        fit_table=fit_table,
        summary=summary,
        one_e22=one_e22,
        isotoken_input=args.isotoken_input,
        k020_input=args.k020_input,
        fit_through_scale=args.fit_through_scale,
    )
    write_text(html_path, html_text)
    return points, predictions, summary, one_e22, html_path


def main() -> None:
    args = parse_args()
    _, _, summary, one_e22, html_path = build_outputs(args)
    print(f"wrote {html_path}")
    print("\n1e22 actuals:")
    print(
        one_e22[
            [
                "series_label",
                "old_4plus_loss",
                "clean_seen_loss",
                "clean_minus_old",
                "clean_over_old_pct",
            ]
        ].to_string(index=False)
    )
    print("\n1e22 fit errors:")
    print(
        summary[
            [
                "target_label",
                "series_label",
                "actual_1e22",
                "pred_1e22",
                "error_1e22_pct",
                "heldout_mae_pct",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
