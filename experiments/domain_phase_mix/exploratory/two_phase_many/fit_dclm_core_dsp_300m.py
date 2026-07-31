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
"""Fit DSP to the completed 300M DCLM Core target and optimize its mixture.

DCLM Core centered accuracy is higher-is-better. The standalone DSP
implementation is loss-shaped and minimizes its target, so this script fits
``objective_metric = -centered_accuracy_macro`` and reports all user-facing
scores back in higher-is-better centered-accuracy units.
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

from experiments.domain_phase_mix.exploratory.two_phase_many import dclm_matrix_guard
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_MATRIX_CSV = SCRIPT_DIR / "metric_registry" / "raw_metric_matrix_300m" / "raw_metric_matrix_300m.csv"
DCLM_MATRIX_CSV = (
    SCRIPT_DIR
    / "metric_registry"
    / "300m_dclm_core_completion"
    / "300m_dclm_core_eval_results_full_after_retry8_bigbench_rescored_repeatcopy128.csv"
)
METADATA_CSV = SCRIPT_DIR / "two_phase_many_epoch_metadata.csv"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "dclm_core_dsp_300m_20260614_repeatcopy128"
TARGET_COLUMN = "lm_eval/dclm_core/centered_accuracy_macro"
TASK_COUNT_COLUMN = "lm_eval/dclm_core/task_count"
MISSING_TASK_COUNT_COLUMN = "lm_eval/dclm_core/missing_task_count"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-matrix-csv", type=Path, default=RAW_MATRIX_CSV)
    parser.add_argument("--dclm-matrix-csv", type=Path, default=DCLM_MATRIX_CSV)
    parser.add_argument("--metadata-csv", type=Path, default=METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--variant", choices=sorted(dsp.VARIANTS), default="effective_exposure")
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--coarse-top-k", type=int, default=4)
    parser.add_argument("--basin-hopping-iters", type=int, default=0)
    parser.add_argument("--optimum-starts", type=int, default=240)
    parser.add_argument("--max-observed-starts", type=int, default=241)
    parser.add_argument("--stability-seeds", type=int, default=5)
    parser.add_argument("--stability-starts", type=int, default=120)
    return parser.parse_args()


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute ordinary \(R^2\)."""
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")


def prediction_metrics(actual: np.ndarray, predicted: np.ndarray, prefix: str) -> dict[str, float]:
    """Summarize predictions in higher-is-better target units."""
    residual = predicted - actual
    return {
        f"{prefix}_rmse": float(np.sqrt(np.mean(residual**2))),
        f"{prefix}_mae": float(np.mean(np.abs(residual))),
        f"{prefix}_r2": r2_score(actual, predicted),
        f"{prefix}_pearson": float(pearsonr(actual, predicted).statistic),
        f"{prefix}_spearman": float(spearmanr(actual, predicted).statistic),
    }


def average_phase_tv(left: np.ndarray, right: np.ndarray) -> float:
    """Return average total variation across two phases."""
    return float(np.abs(left - right).sum() / (2.0 * left.shape[0]))


def entropy(weights: np.ndarray) -> float:
    """Return Shannon entropy for a simplex vector."""
    clipped = np.clip(np.asarray(weights, dtype=float), 1e-300, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def phase_stats(weights: np.ndarray, *, prefix: str) -> dict[str, float | int]:
    """Compute basic support and concentration statistics for two-phase weights."""
    return {
        f"{prefix}_phase0_support_gt_1e3": int(np.sum(weights[0] > 1e-3)),
        f"{prefix}_phase1_support_gt_1e3": int(np.sum(weights[1] > 1e-3)),
        f"{prefix}_phase0_entropy": entropy(weights[0]),
        f"{prefix}_phase1_entropy": entropy(weights[1]),
        f"{prefix}_phase0_effective_support": float(np.exp(entropy(weights[0]))),
        f"{prefix}_phase1_effective_support": float(np.exp(entropy(weights[1]))),
        f"{prefix}_phase0_max_weight": float(np.max(weights[0])),
        f"{prefix}_phase1_max_weight": float(np.max(weights[1])),
    }


def load_fit_frame(raw_matrix_csv: Path, dclm_matrix_csv: Path, target_column: str) -> pd.DataFrame:
    """Join the 300M weight matrix to the completed DCLM Core overlay."""
    raw = pd.read_csv(raw_matrix_csv, low_memory=False)
    dclm = pd.read_csv(dclm_matrix_csv, low_memory=False)
    dclm_matrix_guard.validate_corrected_dclm_matrix(dclm, dclm_matrix_csv)
    if target_column not in dclm.columns:
        raise ValueError(f"DCLM matrix is missing target column {target_column!r}")
    dclm_columns = ["run_name", target_column]
    for column in (TASK_COUNT_COLUMN, MISSING_TASK_COUNT_COLUMN):
        if column in dclm.columns:
            dclm_columns.append(column)
    joined = raw.merge(dclm[dclm_columns], on="run_name", how="left", validate="one_to_one")
    mask = joined["row_kind"].eq("signal") & joined["status"].eq("completed") & joined[target_column].notna()
    fit_frame = joined.loc[mask].copy()
    fit_frame["objective_metric"] = -pd.to_numeric(fit_frame[target_column], errors="raise")
    if fit_frame.empty:
        raise ValueError("No completed signal rows with non-null DCLM Core target")
    if fit_frame["run_name"].duplicated().any():
        duplicate_names = fit_frame.loc[fit_frame["run_name"].duplicated(), "run_name"].tolist()
        raise ValueError(f"Duplicate run_name values in fit frame: {duplicate_names[:10]}")
    return fit_frame.reset_index(drop=True)


def prediction_frame(packet: dsp.PacketData, model: dsp.FittedDSPModel, oof_loss: np.ndarray) -> pd.DataFrame:
    """Return observed-row predictions in higher-is-better units."""
    train_score = -dsp.predict(model, packet.w)
    oof_score = -oof_loss
    actual_score = -packet.y
    frame = packet.frame[["run_name", "registry_run_key"]].copy()
    frame["actual_dclm_centered_accuracy_macro"] = actual_score
    frame["train_pred_dclm_centered_accuracy_macro"] = train_score
    frame["oof_pred_dclm_centered_accuracy_macro"] = oof_score
    frame["train_residual_pred_minus_actual"] = train_score - actual_score
    frame["oof_residual_pred_minus_actual"] = oof_score - actual_score
    frame["actual_rank_desc"] = frame["actual_dclm_centered_accuracy_macro"].rank(method="min", ascending=False)
    frame["oof_rank_desc"] = frame["oof_pred_dclm_centered_accuracy_macro"].rank(method="min", ascending=False)
    return frame.sort_values("oof_pred_dclm_centered_accuracy_macro", ascending=False).reset_index(drop=True)


def weights_with_model_params(
    model: dsp.FittedDSPModel,
    weights: np.ndarray,
    proportional: np.ndarray,
) -> pd.DataFrame:
    """Create a per-domain raw optimum table with model parameters and deltas."""
    tau = np.asarray(model.params.get("tau", np.zeros(len(model.domain_names))), dtype=float)
    gamma = float(model.params["gamma"]) if "gamma" in model.params else np.nan
    rows = []
    for index, domain_name in enumerate(model.domain_names):
        rows.append(
            {
                "domain_name": domain_name,
                "phase_0_weight": float(weights[0, index]),
                "phase_1_weight": float(weights[1, index]),
                "phase_0_delta_vs_proportional": float(weights[0, index] - proportional[0, index]),
                "phase_1_delta_vs_proportional": float(weights[1, index] - proportional[1, index]),
                "phase_0_effective_epochs": float(weights[0, index] * model.c0[index]),
                "phase_1_effective_epochs": float(weights[1, index] * model.c1[index]),
                "total_effective_epochs": float(weights[0, index] * model.c0[index] + weights[1, index] * model.c1[index]),
                "benefit_coef": float(model.benefit_coef[index]),
                "penalty_coef": float(model.penalty_coef[index]),
                "rho": float(np.asarray(model.params["rho"])[index]),
                "tau": float(tau[index]),
                "gamma": gamma,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("total_effective_epochs", ascending=False)


def write_fit_plot(predictions: pd.DataFrame, output_path: Path) -> None:
    """Write observed-vs-predicted and residual diagnostics."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("OOF prediction vs actual", "OOF residual vs actual rank"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Scatter(
            x=predictions["actual_dclm_centered_accuracy_macro"],
            y=predictions["oof_pred_dclm_centered_accuracy_macro"],
            mode="markers",
            text=predictions["run_name"],
            marker={"size": 8, "color": predictions["actual_rank_desc"], "colorscale": "RdYlGn_r"},
            name="run",
        ),
        row=1,
        col=1,
    )
    min_value = float(
        min(
            predictions["actual_dclm_centered_accuracy_macro"].min(),
            predictions["oof_pred_dclm_centered_accuracy_macro"].min(),
        )
    )
    max_value = float(
        max(
            predictions["actual_dclm_centered_accuracy_macro"].max(),
            predictions["oof_pred_dclm_centered_accuracy_macro"].max(),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            line={"dash": "dash", "color": "black"},
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=predictions["actual_rank_desc"],
            y=predictions["oof_residual_pred_minus_actual"],
            mode="markers",
            text=predictions["run_name"],
            marker={"size": 8, "color": predictions["actual_dclm_centered_accuracy_macro"], "colorscale": "RdYlGn_r"},
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="black", row=1, col=2)
    fig.update_xaxes(title_text="actual centered accuracy macro", row=1, col=1)
    fig.update_yaxes(title_text="OOF predicted centered accuracy macro", row=1, col=1)
    fig.update_xaxes(title_text="actual rank, 1 is best", autorange="reversed", row=1, col=2)
    fig.update_yaxes(title_text="OOF prediction - actual", row=1, col=2)
    fig.update_layout(template="plotly_white", width=1300, height=560, title={"text": "DCLM Core DSP fit", "x": 0.5})
    fig.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_weights_plot(weights: pd.DataFrame, output_path: Path) -> None:
    """Write raw optimum weight and epoch plots."""
    ordered = weights.sort_values("total_effective_epochs", ascending=True)
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=("phase weights", "effective epochs"),
        horizontal_spacing=0.08,
    )
    fig.add_trace(
        go.Bar(
            x=ordered["phase_0_weight"],
            y=ordered["domain_name"],
            orientation="h",
            name="phase 0",
            marker={"color": "rgba(44, 127, 184, 0.85)"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=ordered["phase_1_weight"],
            y=ordered["domain_name"],
            orientation="h",
            name="phase 1",
            marker={"color": "rgba(217, 95, 14, 0.85)"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=ordered["phase_0_effective_epochs"],
            y=ordered["domain_name"],
            orientation="h",
            showlegend=False,
            marker={"color": "rgba(44, 127, 184, 0.85)"},
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=ordered["phase_1_effective_epochs"],
            y=ordered["domain_name"],
            orientation="h",
            showlegend=False,
            marker={"color": "rgba(217, 95, 14, 0.85)"},
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="weight", row=1, col=1)
    fig.update_xaxes(title_text="materialized effective epochs", row=1, col=2)
    fig.update_layout(
        template="plotly_white",
        barmode="group",
        width=1450,
        height=max(760, 24 * len(ordered)),
        margin={"l": 330, "r": 40, "t": 75, "b": 80},
        title={"text": "DCLM Core DSP raw optimum", "x": 0.5},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.08},
    )
    fig.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def run_stability(
    model: dsp.FittedDSPModel,
    observed_start_weights: np.ndarray,
    *,
    stability_seeds: int,
    stability_starts: int,
    max_observed_starts: int,
) -> pd.DataFrame:
    """Run raw optimization from several seeds to diagnose optimum stability."""
    rows = []
    for seed in range(stability_seeds):
        result, weights = dsp.optimize_raw(
            model,
            num_starts=stability_starts,
            seed=seed,
            observed_start_weights=observed_start_weights,
            max_observed_starts=max_observed_starts,
        )
        rows.append(
            {
                "seed": seed,
                "loss_value": float(result.fun),
                "pred_dclm_centered_accuracy_macro": float(-result.fun),
                **phase_stats(weights, prefix="raw"),
            }
        )
    return pd.DataFrame.from_records(rows)


def write_report(output_dir: Path, summary: dict[str, Any], top_weights: pd.DataFrame) -> None:
    """Write a concise Markdown report."""
    top_lines = [
        (
            f"- `{row['domain_name']}`: w0 `{row['phase_0_weight']:.4f}`, "
            f"w1 `{row['phase_1_weight']:.4f}`, epochs `{row['total_effective_epochs']:.3f}`."
        )
        for _, row in top_weights.head(12).iterrows()
    ]
    lines = [
        "# DCLM Core DSP Fit 300M",
        "",
        "Target: `lm_eval/dclm_core/centered_accuracy_macro`.",
        "DSP was fit to the sign-flipped target because its implementation is loss-shaped.",
        "",
        "## Fit",
        "",
        f"- Rows: `{summary['fit_row_count']}`.",
        f"- Variant: `{summary['variant']}`.",
        f"- Total parameter count: `{summary['total_param_count']}`.",
        (
            f"- OOF RMSE / R2 / Spearman: `{summary['score_oof_rmse']:.6f}` / "
            f"`{summary['score_oof_r2']:.4f}` / `{summary['score_oof_spearman']:.4f}`."
        ),
        (
            f"- Train RMSE / R2 / Spearman: `{summary['score_train_rmse']:.6f}` / "
            f"`{summary['score_train_r2']:.4f}` / `{summary['score_train_spearman']:.4f}`."
        ),
        "",
        "## Optimum",
        "",
        f"- Observed best: `{summary['best_observed_run_name']}` at `{summary['best_observed_score']:.6f}`.",
        f"- Proportional actual: `{summary['proportional_actual_score']:.6f}`.",
        f"- Raw DSP optimum predicted score: `{summary['raw_predicted_score']:.6f}`.",
        f"- Raw optimum nearest observed row: `{summary['raw_nearest_observed_run_name']}`.",
        f"- Raw optimum TV to nearest observed row: `{summary['raw_nearest_observed_tv']:.4f}`.",
        f"- Raw optimum TV to proportional: `{summary['raw_tv_to_proportional']:.4f}`.",
        "",
        "Top raw-optimum domains by materialized epochs:",
        "",
        *top_lines,
        "",
        "## Caveat",
        "",
        (
            "This is a surrogate optimum, not a validated mixture. The DCLM target has a narrow observed "
            "range at 300M, so the raw DSP optimum should be treated as an extrapolative candidate until "
            "it is checked against held-out scaling points or a direct run."
        ),
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the DCLM Core DSP fit and raw optimization."""
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata_csv)
    fit_frame = load_fit_frame(args.raw_matrix_csv, args.dclm_matrix_csv, args.target_column)
    packet = dsp.packet_from_frame(fit_frame, metadata)
    variant = dsp.VARIANTS[args.variant]
    model, tuning = dsp.fit_variant(
        packet,
        variant,
        maxiter=args.maxiter,
        coarse_top_k=args.coarse_top_k,
        basin_hopping_iters=args.basin_hopping_iters,
    )
    raw_result, raw_weights = dsp.optimize_raw(
        model,
        num_starts=args.optimum_starts,
        observed_start_weights=packet.w,
        max_observed_starts=args.max_observed_starts,
    )
    model_metrics = dsp.metrics(packet, model, raw_result, raw_weights)
    oof_loss = dsp.oof_predictions(packet, model)
    train_score = -dsp.predict(model, packet.w)
    oof_score = -oof_loss
    actual_score = -packet.y
    predictions = prediction_frame(packet, model, oof_loss)

    proportional_rows = packet.frame["run_name"].eq("baseline_proportional")
    if not proportional_rows.any():
        raise ValueError("Fit frame does not contain baseline_proportional")
    proportional_weight = packet.w[int(np.flatnonzero(proportional_rows)[0])]
    best_observed_index = int(np.argmax(actual_score))
    nearest_to_raw = model_metrics["raw_nearest_observed_run_name"]
    raw_weights_frame = weights_with_model_params(model, raw_weights, proportional_weight)
    stability = run_stability(
        model,
        packet.w,
        stability_seeds=args.stability_seeds,
        stability_starts=args.stability_starts,
        max_observed_starts=args.max_observed_starts,
    )

    score_train_metrics = prediction_metrics(actual_score, train_score, "score_train")
    score_oof_metrics = prediction_metrics(actual_score, oof_score, "score_oof")
    proportional_actual = float(actual_score[proportional_rows.to_numpy()][0])
    proportional_pred = float(-dsp.predict(model, proportional_weight[None, :, :])[0])
    raw_predicted_score = float(-raw_result.fun)
    summary: dict[str, Any] = {
        "target_column": args.target_column,
        "variant": variant.name,
        "fit_row_count": int(len(packet.y)),
        "excluded_missing_dclm_count": int(
            pd.read_csv(args.dclm_matrix_csv, usecols=["run_name", args.target_column])[args.target_column].isna().sum()
        ),
        "total_param_count": int(model.total_param_count),
        "m_dependent_params_per_domain": int(model.m_dependent_params_per_domain),
        "maxiter": int(args.maxiter),
        "coarse_top_k": int(args.coarse_top_k),
        "basin_hopping_iters": int(args.basin_hopping_iters),
        "optimum_starts": int(args.optimum_starts),
        "best_observed_run_name": str(packet.frame.iloc[best_observed_index]["run_name"]),
        "best_observed_score": float(actual_score[best_observed_index]),
        "proportional_actual_score": proportional_actual,
        "proportional_train_pred_score": proportional_pred,
        "raw_predicted_score": raw_predicted_score,
        "raw_predicted_delta_vs_proportional_actual": raw_predicted_score - proportional_actual,
        "raw_predicted_delta_vs_proportional_pred": raw_predicted_score - proportional_pred,
        "raw_nearest_observed_run_name": str(nearest_to_raw),
        "raw_nearest_observed_tv": float(model_metrics["raw_nearest_observed_tv"]),
        "raw_nearest_observed_score": float(-model_metrics["raw_nearest_observed_value"]),
        "raw_tv_to_proportional": average_phase_tv(raw_weights, proportional_weight),
        "raw_optimization_success": bool(raw_result.success),
        "raw_optimization_message": str(raw_result.message),
        "fitted_gamma": float(model.params["gamma"]) if "gamma" in model.params else np.nan,
        "stability_pred_score_min": float(stability["pred_dclm_centered_accuracy_macro"].min()),
        "stability_pred_score_median": float(stability["pred_dclm_centered_accuracy_macro"].median()),
        "stability_pred_score_max": float(stability["pred_dclm_centered_accuracy_macro"].max()),
        **score_train_metrics,
        **score_oof_metrics,
        **phase_stats(raw_weights, prefix="raw"),
    }

    tuning.to_csv(output_dir / "tuning_trace.csv", index=False)
    predictions.to_csv(output_dir / "observed_predictions.csv", index=False)
    raw_weights_frame.to_csv(output_dir / "raw_optimum_weights.csv", index=False)
    stability.to_csv(output_dir / "raw_optimum_stability.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "model.json").write_text(json.dumps(dsp.model_to_json(model, model_metrics), indent=2), encoding="utf-8")
    write_fit_plot(predictions, output_dir / "dclm_dsp_fit_diagnostics.html")
    write_weights_plot(raw_weights_frame, output_dir / "dclm_dsp_raw_optimum_weights.html")
    write_report(output_dir, summary, raw_weights_frame)
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Wrote DCLM Core DSP artifacts to {output_dir}")


if __name__ == "__main__":
    main()
