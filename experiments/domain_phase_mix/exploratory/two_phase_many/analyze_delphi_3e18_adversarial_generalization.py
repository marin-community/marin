# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scipy>=1.14",
# ]
# ///
"""Audit frozen Delphi 3e18 surrogate fits on newly appended heldouts."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT = SCRIPT_DIR / "mixture_fit_debugger/src/generated/dashboard_data.json"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/delphi_3e18_adversarial_generalization_20260718"
SWARM_ID = "delphi_3e18"
NEW_ONE_PHASE_PANEL = "delphi_one_phase_augmented_swarm_3e18_20260715"
ADVERSARIAL_PANEL = "delphi_3e18_adversarial_stress_panel_20260716"
NEW_PANELS = frozenset((NEW_ONE_PHASE_PANEL, ADVERSARIAL_PANEL))
LOWER_TAIL_FRACTION = 0.15
LOWER_TAIL_MIN_COUNT = 10
OPTIMISM_THRESHOLD = 0.05
PLOT_MODELS = (
    "hierarchical_phase_bucket_replay",
    "bucket_family_grp",
    "compact_retained_state",
    "effective_exposure",
    "separate_heads",
)
SPLIT_LABELS = {
    "historical": "Historical heldouts",
    "new_one_phase": "New one-phase swarm",
    "adversarial": "Adversarial stress panel",
    "adversarial_target_matched": "Adversarial, target-matched",
    "adversarial_cross_target": "Adversarial, cross-target",
    "expanded": "All coordinate-disjoint heldouts",
}
SPLIT_COLORS = {
    "historical": "#4f86c6",
    "new_one_phase": "#ef9b3f",
    "adversarial": "#bd3e3e",
}


def finite_pairs(observed: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    return observed[valid], predicted[valid]


def regret_at_k(observed: np.ndarray, predicted: np.ndarray, k: int) -> float:
    selected = np.argsort(predicted)[: min(k, len(predicted))]
    return float(np.min(observed[selected]) - np.min(observed))


def metric_summary(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    observed, predicted = finite_pairs(observed, predicted)
    if len(observed) < 3:
        raise ValueError(f"At least three finite observations are required, got {len(observed)}")
    residual = predicted - observed
    calibration_slope, calibration_intercept = np.polyfit(predicted, observed, deg=1)
    lower_tail_count = min(
        len(observed),
        max(LOWER_TAIL_MIN_COUNT, math.ceil(LOWER_TAIL_FRACTION * len(observed))),
    )
    lower_tail = np.argsort(predicted)[:lower_tail_count]
    lower_tail_optimism = observed[lower_tail] - predicted[lower_tail]
    selected = int(np.argmin(predicted))
    optimism = observed - predicted
    return {
        "n": len(observed),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias_predicted_minus_observed": float(np.mean(residual)),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "calibration_slope_observed_on_predicted": float(calibration_slope),
        "calibration_intercept_observed_on_predicted": float(calibration_intercept),
        "regret_at_1": regret_at_k(observed, predicted, 1),
        "regret_at_3": regret_at_k(observed, predicted, 3),
        "regret_at_5": regret_at_k(observed, predicted, 5),
        "selected_optimism": float(optimism[selected]),
        "lower_tail_optimism": float(np.mean(np.maximum(lower_tail_optimism, 0.0))),
        "low_tail_rmse": float(np.sqrt(np.mean(lower_tail_optimism**2))),
        "optimism_gt_0p05_count": int(np.sum(optimism > OPTIMISM_THRESHOLD)),
        "optimism_gt_0p05_fraction": float(np.mean(optimism > OPTIMISM_THRESHOLD)),
        "worst_optimism": float(np.max(optimism)),
        "max_absolute_error": float(np.max(np.abs(residual))),
    }


def split_name(row: dict[str, Any]) -> str | None:
    if row["split"] != "heldout" or row["isSharedAlias"]:
        return None
    if row["panel"] == NEW_ONE_PHASE_PANEL:
        return "new_one_phase"
    if row["panel"] == ADVERSARIAL_PANEL:
        return "adversarial"
    return "historical"


def records_for_model(
    rows: list[dict[str, Any]],
    predicted: Iterable[float | None],
    target: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row, prediction in zip(rows, predicted, strict=True):
        split = split_name(row)
        observed = row["observed"].get(target)
        if split is None or observed is None or prediction is None:
            continue
        observed_float = float(observed)
        predicted_float = float(prediction)
        if not math.isfinite(observed_float) or not math.isfinite(predicted_float):
            continue
        records.append(
            {
                "row_id": row["id"],
                "name": row["name"],
                "panel": row["panel"],
                "split": split,
                "policy_family": row["policyFamily"],
                "phase_family": row["phaseFamily"],
                "candidate_target": row["candidateTarget"],
                "method": row["method"],
                "wandb_url": row["wandbUrl"],
                "observed": observed_float,
                "predicted": predicted_float,
                "prediction_residual": predicted_float - observed_float,
                "optimism": observed_float - predicted_float,
                "phase_tv": row["diagnostics"]["phaseTv"],
                "aggregate_tv_to_proportional": row["diagnostics"]["aggregateTvToProportional"],
                "aggregate_kl_to_proportional": row["diagnostics"]["aggregateKlToProportional"],
                "max_epoch": row["diagnostics"]["maxEpoch"],
                "support_distance": row["diagnostics"]["supportDistance"],
            }
        )
    return pd.DataFrame.from_records(records)


def model_metrics(bundle: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    swarm = bundle["swarms"][SWARM_ID]
    rows = swarm["rows"]
    model_labels = {model: metadata["label"] for model, metadata in bundle["models"].items()}
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for target in swarm["targets"]:
        models = swarm["predictions"][target]["two_phase"]
        for model, payload in models.items():
            records = records_for_model(rows, payload["prediction"], target)
            records.insert(0, "model", model)
            records.insert(1, "model_label", model_labels[model])
            records.insert(2, "target", target)
            prediction_rows.append(records)
            subsets = {
                "historical": records.loc[records["split"] == "historical"],
                "new_one_phase": records.loc[records["split"] == "new_one_phase"],
                "adversarial": records.loc[records["split"] == "adversarial"],
                "adversarial_target_matched": records.loc[
                    (records["split"] == "adversarial") & (records["candidate_target"] == target)
                ],
                "adversarial_cross_target": records.loc[
                    (records["split"] == "adversarial") & (records["candidate_target"] != target)
                ],
                "expanded": records,
            }
            for split, subset in subsets.items():
                summary = metric_summary(
                    subset["observed"].to_numpy(dtype=float),
                    subset["predicted"].to_numpy(dtype=float),
                )
                metric_rows.append(
                    {
                        "target": target,
                        "model": model,
                        "model_label": model_labels[model],
                        "split": split,
                        **summary,
                    }
                )
    return pd.DataFrame.from_records(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def plot_calibration(bundle: dict[str, Any], predictions: pd.DataFrame) -> None:
    target_labels = {target: metadata["label"] for target, metadata in bundle["swarms"][SWARM_ID]["targets"].items()}
    available_models = [model for model in PLOT_MODELS if model in set(predictions["model"])]
    for target, target_label in target_labels.items():
        figure = make_subplots(
            rows=len(available_models),
            cols=2,
            subplot_titles=[
                title
                for model in available_models
                for title in (
                    f"{bundle['models'][model]['label']}: observed vs predicted",
                    f"{bundle['models'][model]['label']}: residual vs observed",
                )
            ],
            vertical_spacing=0.055,
        )
        target_rows = predictions.loc[predictions["target"] == target]
        for row_index, model in enumerate(available_models, start=1):
            model_rows = target_rows.loc[target_rows["model"] == model]
            for split in ("historical", "new_one_phase", "adversarial"):
                subset = model_rows.loc[model_rows["split"] == split]
                customdata = np.column_stack(
                    (
                        subset["name"],
                        subset["panel"],
                        subset["optimism"],
                        subset["max_epoch"],
                        subset["phase_tv"],
                    )
                )
                figure.add_trace(
                    go.Scatter(
                        x=subset["predicted"],
                        y=subset["observed"],
                        mode="markers",
                        name=SPLIT_LABELS[split],
                        legendgroup=split,
                        showlegend=row_index == 1,
                        marker={"color": SPLIT_COLORS[split], "size": 7, "opacity": 0.72},
                        customdata=customdata,
                        hovertemplate=(
                            "%{customdata[0]}<br>%{customdata[1]}<br>predicted=%{x:.5f}<br>"
                            "observed=%{y:.5f}<br>optimism=%{customdata[2]:.5f}<br>"
                            "max epoch=%{customdata[3]:.2f}<br>phase TV=%{customdata[4]:.3f}<extra></extra>"
                        ),
                    ),
                    row=row_index,
                    col=1,
                )
                figure.add_trace(
                    go.Scatter(
                        x=subset["observed"],
                        y=subset["prediction_residual"],
                        mode="markers",
                        name=SPLIT_LABELS[split],
                        legendgroup=split,
                        showlegend=False,
                        marker={"color": SPLIT_COLORS[split], "size": 7, "opacity": 0.72},
                        customdata=customdata,
                        hovertemplate=(
                            "%{customdata[0]}<br>%{customdata[1]}<br>observed=%{x:.5f}<br>"
                            "predicted - observed=%{y:.5f}<br>max epoch=%{customdata[3]:.2f}<br>"
                            "phase TV=%{customdata[4]:.3f}<extra></extra>"
                        ),
                    ),
                    row=row_index,
                    col=2,
                )
            bounds = [
                float(min(model_rows["observed"].min(), model_rows["predicted"].min())),
                float(max(model_rows["observed"].max(), model_rows["predicted"].max())),
            ]
            figure.add_trace(
                go.Scatter(
                    x=bounds,
                    y=bounds,
                    mode="lines",
                    line={"color": "#667681", "dash": "dash"},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row_index,
                col=1,
            )
            figure.add_hline(y=0.0, line={"color": "#667681", "dash": "dash"}, row=row_index, col=2)
        figure.update_xaxes(title_text="Predicted BPB", col=1)
        figure.update_yaxes(title_text="Observed BPB", col=1)
        figure.update_xaxes(title_text="Observed BPB", col=2)
        figure.update_yaxes(title_text="Predicted - observed", col=2)
        figure.update_layout(
            title=f"Frozen Delphi 3e18 surrogate generalization: {target_label}",
            template="plotly_white",
            height=390 * len(available_models),
            width=1450,
            margin={"l": 80, "r": 40, "t": 110, "b": 65},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.015, "xanchor": "left", "x": 0},
        )
        figure.write_html(
            OUTPUT_DIR / f"{target}_calibration.html",
            include_plotlyjs=True,
            config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 4}},
        )


def fmt(value: Any, digits: int = 5) -> str:
    return f"{float(value):.{digits}f}"


def markdown_table(rows: list[tuple[Any, ...]], headers: tuple[str, ...]) -> str:
    lines = [f"| {' | '.join(headers)} |", f"| {' | '.join('---' for _ in headers)} |"]
    lines.extend(f"| {' | '.join(str(value) for value in row)} |" for row in rows)
    return "\n".join(lines)


def write_report(bundle: dict[str, Any], metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    swarm = bundle["swarms"][SWARM_ID]
    report = [
        "# Delphi 3e18 adversarial generalization audit",
        "",
        (
            f"Every surrogate remains fit exclusively on the unchanged {swarm['dataset']['fitDesignCount']}-row "
            "Delphi 3e18 swarm. This report evaluates frozen fits on the append-only heldout archive: "
            f"{int((predictions['split'] == 'historical').sum() / (len(bundle['models']) * len(swarm['targets'])))} "
            "historical coordinates, 238 newly trained one-phase coordinates, and 120 sealed adversarial "
            "coordinates. Exact-coordinate aliases are excluded."
        ),
        "",
        (
            "Calibration is the OLS slope in observed BPB = intercept + slope * predicted BPB. For lower-is-better "
            "BPB, optimism is observed minus predicted; optimism above 0.05 means the surrogate understated loss "
            "by more than 0.05 BPB. The heldout archive is intervention-designed rather than IID, so these numbers "
            "measure transfer to the explored policies, not population risk."
        ),
    ]
    for target, target_metadata in swarm["targets"].items():
        report.extend(["", f"## {target_metadata['label']}", ""])
        target_metrics = metrics.loc[metrics["target"] == target]
        report.append("### New one-phase panel")
        report.append("")
        one_phase = target_metrics.loc[target_metrics["split"] == "new_one_phase"].sort_values(
            ["rmse", "worst_optimism"]
        )
        report.append(
            markdown_table(
                [
                    (
                        row.model_label,
                        int(row.n),
                        fmt(row.rmse),
                        fmt(row.bias_predicted_minus_observed),
                        fmt(row.calibration_slope_observed_on_predicted, 3),
                        int(row.optimism_gt_0p05_count),
                        fmt(row.worst_optimism),
                        fmt(row.regret_at_1),
                    )
                    for row in one_phase.itertuples()
                ],
                ("Model", "n", "RMSE", "Bias", "Cal. slope", "Opt. >.05", "Worst opt.", "Regret@1"),
            )
        )
        report.extend(["", "### Sealed adversarial panel", ""])
        adversarial = target_metrics.loc[target_metrics["split"] == "adversarial"].sort_values(
            ["rmse", "worst_optimism"]
        )
        report.append(
            markdown_table(
                [
                    (
                        row.model_label,
                        int(row.n),
                        fmt(row.rmse),
                        fmt(row.bias_predicted_minus_observed),
                        fmt(row.calibration_slope_observed_on_predicted, 3),
                        int(row.optimism_gt_0p05_count),
                        fmt(row.worst_optimism),
                        fmt(row.regret_at_1),
                    )
                    for row in adversarial.itertuples()
                ],
                ("Model", "n", "RMSE", "Bias", "Cal. slope", "Opt. >.05", "Worst opt.", "Regret@1"),
            )
        )
        report.extend(["", "### Target-matched adversarial policies", ""])
        target_matched = target_metrics.loc[target_metrics["split"] == "adversarial_target_matched"].sort_values(
            ["rmse", "worst_optimism"]
        )
        report.append(
            markdown_table(
                [
                    (
                        row.model_label,
                        int(row.n),
                        fmt(row.rmse),
                        fmt(row.bias_predicted_minus_observed),
                        fmt(row.calibration_slope_observed_on_predicted, 3),
                        int(row.optimism_gt_0p05_count),
                        fmt(row.worst_optimism),
                        fmt(row.regret_at_1),
                    )
                    for row in target_matched.itertuples()
                ],
                ("Model", "n", "RMSE", "Bias", "Cal. slope", "Opt. >.05", "Worst opt.", "Regret@1"),
            )
        )
        report.extend(["", "### Expanded append-only archive", ""])
        expanded = target_metrics.loc[target_metrics["split"] == "expanded"].sort_values(["rmse", "worst_optimism"])
        report.append(
            markdown_table(
                [
                    (
                        row.model_label,
                        int(row.n),
                        fmt(row.rmse),
                        fmt(row.bias_predicted_minus_observed),
                        fmt(row.calibration_slope_observed_on_predicted, 3),
                        int(row.optimism_gt_0p05_count),
                        fmt(row.worst_optimism),
                        fmt(row.regret_at_1),
                    )
                    for row in expanded.itertuples()
                ],
                ("Model", "n", "RMSE", "Bias", "Cal. slope", "Opt. >.05", "Worst opt.", "Regret@1"),
            )
        )
    report.extend(
        [
            "",
            "## Files",
            "",
            "- `model_split_metrics.csv`: all metrics by target, model, and heldout split.",
            "- `heldout_predictions.csv`: row-level predictions and exposure diagnostics.",
            "- `worst_adversarial_predictions.csv`: the ten most optimistic adversarial predictions per target/model.",
            "- `*_calibration.html`: interactive observed-versus-predicted and residual plots.",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    bundle = json.loads(INPUT.read_text())
    metrics, predictions = model_metrics(bundle)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "model_split_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "heldout_predictions.csv", index=False)
    worst = (
        predictions.loc[predictions["split"] == "adversarial"]
        .sort_values(["target", "model", "optimism"], ascending=[True, True, False])
        .groupby(["target", "model"], sort=False)
        .head(10)
    )
    worst.to_csv(OUTPUT_DIR / "worst_adversarial_predictions.csv", index=False)
    plot_calibration(bundle, predictions)
    write_report(bundle, metrics, predictions)
    summary = {
        "fit_rows": bundle["swarms"][SWARM_ID]["dataset"]["fitDesignCount"],
        "coordinate_disjoint_heldouts": bundle["swarms"][SWARM_ID]["dataset"]["heldoutCount"],
        "new_one_phase_heldouts": int(
            predictions.loc[predictions["model"] == predictions["model"].iloc[0], "split"].eq("new_one_phase").sum()
            / len(bundle["swarms"][SWARM_ID]["targets"])
        ),
        "adversarial_heldouts": int(
            predictions.loc[predictions["model"] == predictions["model"].iloc[0], "split"].eq("adversarial").sum()
            / len(bundle["swarms"][SWARM_ID]["targets"])
        ),
        "models": int(metrics["model"].nunique()),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
