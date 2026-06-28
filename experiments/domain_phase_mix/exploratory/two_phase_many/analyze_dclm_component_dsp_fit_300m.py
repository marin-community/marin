# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas",
#     "plotly",
#     "scipy",
#     "scikit-learn",
# ]
# ///
"""Diagnose why DSP fits DCLM Core macro poorly.

This script fits DSP to each DCLM Core centered-accuracy component separately,
then asks whether averaging component-wise OOF predictions recovers the macro
target better than fitting the macro directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr

from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dclm_core_dsp_300m import (
    DCLM_MATRIX_CSV,
    METADATA_CSV,
    RAW_MATRIX_CSV,
    TARGET_COLUMN,
    load_fit_frame,
    prediction_metrics,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "dclm_core_component_dsp_diagnostic_20260614_repeatcopy128"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-matrix-csv", type=Path, default=RAW_MATRIX_CSV)
    parser.add_argument("--dclm-matrix-csv", type=Path, default=DCLM_MATRIX_CSV)
    parser.add_argument("--metadata-csv", type=Path, default=METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--variant", choices=sorted(dsp.VARIANTS), default="no_penalty")
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--coarse-top-k", type=int, default=3)
    parser.add_argument("--basin-hopping-iters", type=int, default=0)
    return parser.parse_args()


def dclm_component_columns(frame: pd.DataFrame) -> list[str]:
    """Return DCLM component centered-accuracy columns in stable order."""
    columns = [
        column
        for column in frame.columns
        if column.startswith("lm_eval/dclm_core/")
        and column.endswith("/centered_accuracy")
        and column != TARGET_COLUMN
    ]
    return sorted(columns)


def task_label(column: str) -> str:
    """Extract compact DCLM task label."""
    prefix = "lm_eval/dclm_core/"
    suffix = "/centered_accuracy"
    return column.removeprefix(prefix).removesuffix(suffix)


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute ordinary \(R^2\)."""
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")


def merge_raw_and_dclm(raw_matrix_csv: Path, dclm_matrix_csv: Path) -> pd.DataFrame:
    """Merge 300M weights and DCLM component targets."""
    raw = pd.read_csv(raw_matrix_csv, low_memory=False)
    dclm = pd.read_csv(dclm_matrix_csv, low_memory=False)
    dclm_cols = ["run_name", TARGET_COLUMN, *dclm_component_columns(dclm)]
    return raw.merge(dclm[dclm_cols], on="run_name", how="left", validate="one_to_one")


def fit_target(
    joined: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    target_column: str,
    variant: dsp.DSPVariant,
    maxiter: int,
    coarse_top_k: int,
    basin_hopping_iters: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit one DCLM target and return metrics plus row-wise predictions."""
    fit_frame = joined.loc[
        joined["row_kind"].eq("signal") & joined["status"].eq("completed") & joined[target_column].notna()
    ].copy()
    fit_frame["objective_metric"] = -pd.to_numeric(fit_frame[target_column], errors="raise")
    packet = dsp.packet_from_frame(fit_frame.reset_index(drop=True), metadata)
    model, _trace = dsp.fit_variant(
        packet,
        variant,
        maxiter=maxiter,
        coarse_top_k=coarse_top_k,
        basin_hopping_iters=basin_hopping_iters,
    )
    train_score = -dsp.predict(model, packet.w)
    oof_score = -dsp.oof_predictions(packet, model)
    actual = -packet.y
    proportional_mask = packet.frame["run_name"].eq("baseline_proportional").to_numpy()
    row = {
        "target_column": target_column,
        "task": "macro" if target_column == TARGET_COLUMN else task_label(target_column),
        "variant": variant.name,
        "fit_row_count": int(len(packet.y)),
        "target_mean": float(np.mean(actual)),
        "target_std": float(np.std(actual, ddof=1)),
        "target_min": float(np.min(actual)),
        "target_max": float(np.max(actual)),
        "target_range": float(np.max(actual) - np.min(actual)),
        "best_observed_run_name": str(packet.frame.iloc[int(np.argmax(actual))]["run_name"]),
        "best_observed_score": float(np.max(actual)),
        "proportional_score": float(actual[proportional_mask][0]) if proportional_mask.any() else np.nan,
        **prediction_metrics(actual, train_score, "train"),
        **prediction_metrics(actual, oof_score, "oof"),
    }
    predictions = packet.frame[["run_name"]].copy()
    predictions["target_column"] = target_column
    predictions["task"] = row["task"]
    predictions["actual"] = actual
    predictions["train_pred"] = train_score
    predictions["oof_pred"] = oof_score
    return row, predictions


def macro_from_components(component_predictions: pd.DataFrame, actual_macro: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    """Average component-wise predictions and compare to actual macro."""
    wide_actual = component_predictions.pivot(index="run_name", columns="task", values="actual")
    wide_oof = component_predictions.pivot(index="run_name", columns="task", values="oof_pred")
    wide_train = component_predictions.pivot(index="run_name", columns="task", values="train_pred")
    out = actual_macro[["run_name", TARGET_COLUMN]].copy()
    out = out.merge(
        wide_actual.mean(axis=1).rename("component_actual_macro").reset_index(), on="run_name", how="inner"
    )
    out = out.merge(wide_oof.mean(axis=1).rename("component_oof_macro").reset_index(), on="run_name", how="inner")
    out = out.merge(wide_train.mean(axis=1).rename("component_train_macro").reset_index(), on="run_name", how="inner")
    metrics = {
        **prediction_metrics(
            out[TARGET_COLUMN].to_numpy(dtype=float),
            out["component_oof_macro"].to_numpy(dtype=float),
            "component_ensemble_oof_macro",
        ),
        **prediction_metrics(
            out[TARGET_COLUMN].to_numpy(dtype=float),
            out["component_train_macro"].to_numpy(dtype=float),
            "component_ensemble_train_macro",
        ),
        "component_actual_macro_max_abs_diff": float(
            np.max(np.abs(out[TARGET_COLUMN].to_numpy(dtype=float) - out["component_actual_macro"].to_numpy(dtype=float)))
        ),
    }
    return metrics, out


def write_component_plot(component_summary: pd.DataFrame, macro_predictions: pd.DataFrame, output_path: Path) -> None:
    """Write component fit and macro-from-components diagnostics."""
    ordered = component_summary.sort_values("oof_spearman", ascending=True)
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("component OOF Spearman", "component OOF R2", "component-ensemble macro"),
        horizontal_spacing=0.12,
        column_widths=[0.34, 0.28, 0.38],
    )
    fig.add_trace(
        go.Bar(
            x=ordered["oof_spearman"],
            y=ordered["task"],
            orientation="h",
            marker={"color": ordered["oof_spearman"], "colorscale": "RdYlGn_r"},
            name="OOF Spearman",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=ordered["oof_r2"],
            y=ordered["task"],
            orientation="h",
            marker={"color": ordered["oof_r2"], "colorscale": "RdYlGn_r"},
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=macro_predictions[TARGET_COLUMN],
            y=macro_predictions["component_oof_macro"],
            mode="markers",
            text=macro_predictions["run_name"],
            marker={
                "size": 8,
                "color": macro_predictions[TARGET_COLUMN],
                "colorscale": "RdYlGn_r",
                "showscale": True,
                "colorbar": {"title": "actual macro"},
            },
            name="component ensemble",
        ),
        row=1,
        col=3,
    )
    min_value = float(min(macro_predictions[TARGET_COLUMN].min(), macro_predictions["component_oof_macro"].min()))
    max_value = float(max(macro_predictions[TARGET_COLUMN].max(), macro_predictions["component_oof_macro"].max()))
    fig.add_trace(
        go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            line={"dash": "dash", "color": "black"},
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.add_vline(x=0.0, line_dash="dash", line_color="black", row=1, col=1)
    fig.add_vline(x=0.0, line_dash="dash", line_color="black", row=1, col=2)
    fig.update_xaxes(title_text="Spearman", row=1, col=1)
    fig.update_xaxes(title_text="R2", row=1, col=2)
    fig.update_xaxes(title_text="actual macro", row=1, col=3)
    fig.update_yaxes(title_text="component", row=1, col=1)
    fig.update_yaxes(title_text="component OOF macro", row=1, col=3)
    fig.update_layout(
        template="plotly_white",
        width=1650,
        height=max(720, 28 * len(component_summary)),
        margin={"l": 300, "r": 40, "t": 80, "b": 80},
        title={"text": "DCLM component DSP diagnostic", "x": 0.5},
        showlegend=False,
    )
    fig.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(
    output_dir: Path,
    summary: dict[str, Any],
    component_summary: pd.DataFrame,
    macro_comparison: pd.DataFrame,
) -> None:
    """Write Markdown summary."""
    top_components = component_summary.sort_values("oof_spearman", ascending=False).head(8)
    weak_components = component_summary.sort_values("oof_spearman", ascending=True).head(8)
    lines = [
        "# DCLM Component DSP Diagnostic",
        "",
        "Question: does direct DCLM macro fitting fail because component scores are individually modelable but aggregate poorly, or because the components themselves are weakly modeled?",
        "",
        "## Summary",
        "",
        f"- Variant: `{summary['variant']}`.",
        f"- Components: `{summary['component_count']}`.",
        f"- Median component OOF Spearman: `{summary['component_oof_spearman_median']:.4f}`.",
        f"- Mean component OOF Spearman: `{summary['component_oof_spearman_mean']:.4f}`.",
        f"- Components with OOF Spearman >= 0.5: `{summary['component_oof_spearman_ge_0p5_count']}`.",
        f"- Components with positive OOF R2: `{summary['component_positive_oof_r2_count']}`.",
        f"- Component-wise OOF ensemble macro Spearman: `{summary['component_ensemble_oof_macro_spearman']:.4f}`.",
        f"- Component-wise OOF ensemble macro R2: `{summary['component_ensemble_oof_macro_r2']:.4f}`.",
        f"- Actual macro reconstructed from component actuals max absolute diff: `{summary['component_actual_macro_max_abs_diff']:.3e}`.",
        "",
        "Interpretation: if the component-wise OOF ensemble macro is also weak, the failure is not primarily a direct-macro aggregation bug.",
        "",
        "## Best Component Fits",
        "",
    ]
    for _, row in top_components.iterrows():
        lines.append(
            f"- `{row['task']}`: OOF Spearman `{row['oof_spearman']:.4f}`, OOF R2 `{row['oof_r2']:.4f}`, range `{row['target_range']:.4f}`."
        )
    lines.extend(["", "## Weakest Component Fits", ""])
    for _, row in weak_components.iterrows():
        lines.append(
            f"- `{row['task']}`: OOF Spearman `{row['oof_spearman']:.4f}`, OOF R2 `{row['oof_r2']:.4f}`, range `{row['target_range']:.4f}`."
        )
    lines.extend(
        [
            "",
            "## Top Macro Rows",
            "",
        ]
    )
    for _, row in macro_comparison.sort_values(TARGET_COLUMN, ascending=False).head(8).iterrows():
        lines.append(
            f"- `{row['run_name']}`: actual `{row[TARGET_COLUMN]:.6f}`, component OOF macro `{row['component_oof_macro']:.6f}`."
        )
    lines.append("")
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the component diagnostic."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata_csv)
    joined = merge_raw_and_dclm(args.raw_matrix_csv, args.dclm_matrix_csv)
    component_columns = dclm_component_columns(joined)
    if not component_columns:
        raise ValueError("No DCLM component centered-accuracy columns found")
    variant = dsp.VARIANTS[args.variant]

    component_rows = []
    prediction_frames = []
    for index, column in enumerate(component_columns, start=1):
        print(f"[{index}/{len(component_columns)}] fitting {task_label(column)}", flush=True)
        row, predictions = fit_target(
            joined,
            metadata,
            target_column=column,
            variant=variant,
            maxiter=args.maxiter,
            coarse_top_k=args.coarse_top_k,
            basin_hopping_iters=args.basin_hopping_iters,
        )
        component_rows.append(row)
        prediction_frames.append(predictions)

    component_summary = pd.DataFrame.from_records(component_rows).sort_values("task")
    component_predictions = pd.concat(prediction_frames, ignore_index=True)
    macro_frame = load_fit_frame(args.raw_matrix_csv, args.dclm_matrix_csv, TARGET_COLUMN)
    macro_metrics, macro_predictions = macro_from_components(
        component_predictions,
        macro_frame[["run_name", TARGET_COLUMN]].copy(),
    )
    summary = {
        "variant": variant.name,
        "component_count": int(len(component_summary)),
        "component_oof_spearman_mean": float(component_summary["oof_spearman"].mean()),
        "component_oof_spearman_median": float(component_summary["oof_spearman"].median()),
        "component_oof_spearman_min": float(component_summary["oof_spearman"].min()),
        "component_oof_spearman_max": float(component_summary["oof_spearman"].max()),
        "component_oof_spearman_ge_0p5_count": int((component_summary["oof_spearman"] >= 0.5).sum()),
        "component_positive_oof_r2_count": int((component_summary["oof_r2"] > 0.0).sum()),
        **macro_metrics,
    }
    component_summary.to_csv(args.output_dir / "component_fit_summary.csv", index=False)
    component_predictions.to_csv(args.output_dir / "component_predictions_long.csv", index=False)
    macro_predictions.to_csv(args.output_dir / "component_ensemble_macro_predictions.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_component_plot(component_summary, macro_predictions, args.output_dir / "component_dsp_fit_diagnostics.html")
    write_report(args.output_dir, summary, component_summary, macro_predictions)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
