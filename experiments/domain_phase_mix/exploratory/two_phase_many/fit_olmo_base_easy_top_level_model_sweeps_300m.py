# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Fit OLMoBaseEval Easy top-level BPB models on the corrected 300M panel.

The fit panel is the deletion-augmented 300M convention:

- 241 ex-ante qsplit/signal rows, excluding the adaptive OLMix row;
- 39 proportional domain-deletion rows;
- the proportional row target is replaced by the mean of the original
  proportional checkpoint plus the 10 proportional-noise repeats.

Targets are the OLMoBaseEval Easy top-level QA/code/math BPBs and their simple
3-way macro. The script sweeps:

- effective-exposure DSP linear-head L2 regularization;
- OLMix Huber delta, using the reference log-linear functional form.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as olmix,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_top_level_dsp_300m as olmo_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "olmo_base_easy_top_level_model_sweeps_300m_20260625"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
TARGET_ORDER = [*olmo_dsp.TARGET_KEYS.keys(), olmo_dsp.MACRO_TARGET]


@dataclass(frozen=True)
class ModelSummary:
    model_family: str
    variant: str
    target_name: str
    target_metric: str
    hyperparameter_name: str
    hyperparameter_value: float
    n_rows: int
    n_signal_rows: int
    n_deletion_rows: int
    n_proportional_reference_rows: int
    proportional_reference_mean: float
    proportional_reference_std: float | None
    train_rmse: float
    train_mae: float
    train_pearson: float
    train_spearman: float
    oof_rmse: float
    oof_mae: float
    oof_pearson: float
    oof_spearman: float
    fold_mean_regret_at_1: float
    lower_tail_optimism: float
    low_tail_rmse: float
    best_observed_run_name: str
    best_observed_value: float
    predicted_best_observed_run_name: str
    predicted_best_observed_actual_value: float
    predicted_best_observed_predicted_value: float
    selection_score: float
    notes: str


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dsp-linear-reg-values", default="1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2")
    parser.add_argument("--olmix-huber-deltas", default="0.005,0.01,0.02,0.05")
    parser.add_argument("--maxiter", type=int, default=dsp.FIT_MAXITER)
    parser.add_argument("--coarse-top-k", type=int, default=dsp.START_TOP_K)
    parser.add_argument("--basin-hopping-iters", type=int, default=1)
    parser.add_argument("--olmix-starts", type=int, default=24)
    return parser.parse_args()


def regression_metrics(y: np.ndarray, y_hat: np.ndarray) -> tuple[float, float, float, float]:
    residual = y_hat - y
    rmse = float(np.sqrt(np.mean(residual * residual)))
    mae = float(np.mean(np.abs(residual)))
    pearson = float(pearsonr(y, y_hat).statistic) if np.std(y) > 0.0 and np.std(y_hat) > 0.0 else float("nan")
    spearman = float(spearmanr(y, y_hat).statistic) if np.std(y) > 0.0 and np.std(y_hat) > 0.0 else float("nan")
    return rmse, mae, pearson, spearman


def selection_score(oof_rmse: float, lower_tail_optimism: float) -> float:
    return float(oof_rmse + 0.5 * max(lower_tail_optimism, 0.0))


def build_all_target_panels(
    columns: list[str],
    domains: list[str],
    token_counts: np.ndarray,
) -> dict[str, tuple[pd.DataFrame, dict[str, Any], dsp.PacketData]]:
    metrics = olmo_dsp.load_all_300m_olmo_metrics()
    target_wide = olmo_dsp.build_target_wide(metrics)
    panels: dict[str, tuple[pd.DataFrame, dict[str, Any], dsp.PacketData]] = {}
    for target_name in TARGET_ORDER:
        panel, metadata = olmo_dsp.build_fit_panel(columns, target_wide, target_name)
        packet = olmo_dsp.build_dsp_packet(panel, columns, domains, token_counts, target_name)
        panels[target_name] = (panel, metadata, packet)
    return panels


def oof_dsp_predictions(
    packet: dsp.PacketData, model: dsp.FittedDSPModel
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    folds = olmix.kfold_indices(len(packet.y), n_splits=olmix.N_SPLITS, seed=olmix.CV_SEED)
    oof = np.zeros_like(packet.y, dtype=float)
    for train_idx, test_idx in folds:
        fold_model = dsp.fit_linear_head(
            packet.w[train_idx],
            packet.y[train_idx],
            packet,
            model.variant,
            model.params,
        )
        oof[test_idx] = dsp.predict(fold_model, packet.w[test_idx])
    return oof, folds


def summarize_predictions(
    *,
    model_family: str,
    variant: str,
    target_name: str,
    target_metric: str,
    hyperparameter_name: str,
    hyperparameter_value: float,
    panel: pd.DataFrame,
    metadata: dict[str, Any],
    observed: np.ndarray,
    train_prediction: np.ndarray,
    oof_prediction: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    notes: str,
) -> ModelSummary:
    train_rmse, train_mae, train_pearson, train_spearman = regression_metrics(observed, train_prediction)
    oof_metrics = olmix.predictive_diagnostics(observed, oof_prediction, folds)
    best_idx = int(np.argmin(observed))
    pred_best_idx = int(np.argmin(oof_prediction))
    oof_rmse = float(oof_metrics["rmse"])
    lower_tail_optimism = float(oof_metrics["lower_tail_optimism"])
    return ModelSummary(
        model_family=model_family,
        variant=variant,
        target_name=target_name,
        target_metric=target_metric,
        hyperparameter_name=hyperparameter_name,
        hyperparameter_value=float(hyperparameter_value),
        n_rows=int(len(panel)),
        n_signal_rows=int(panel["panel_source"].eq("qsplit_signal").sum()),
        n_deletion_rows=int(panel["panel_source"].eq("domain_deletion").sum()),
        n_proportional_reference_rows=int(metadata["n_proportional_reference_rows"]),
        proportional_reference_mean=float(metadata["proportional_reference_mean"]),
        proportional_reference_std=metadata["proportional_reference_std"],
        train_rmse=train_rmse,
        train_mae=train_mae,
        train_pearson=train_pearson,
        train_spearman=train_spearman,
        oof_rmse=oof_rmse,
        oof_mae=float(oof_metrics["mae"]),
        oof_pearson=float(oof_metrics["pearson"]),
        oof_spearman=float(oof_metrics["spearman"]),
        fold_mean_regret_at_1=float(oof_metrics["fold_mean_regret_at_1"]),
        lower_tail_optimism=lower_tail_optimism,
        low_tail_rmse=float(oof_metrics["low_tail_rmse"]),
        best_observed_run_name=str(panel.iloc[best_idx]["run_name"]),
        best_observed_value=float(observed[best_idx]),
        predicted_best_observed_run_name=str(panel.iloc[pred_best_idx]["run_name"]),
        predicted_best_observed_actual_value=float(observed[pred_best_idx]),
        predicted_best_observed_predicted_value=float(oof_prediction[pred_best_idx]),
        selection_score=selection_score(oof_rmse, lower_tail_optimism),
        notes=notes,
    )


def fit_effective_exposure_dsp(
    *,
    panel: pd.DataFrame,
    metadata: dict[str, Any],
    packet: dsp.PacketData,
    target_name: str,
    target_metric: str,
    linear_reg: float,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ModelSummary, pd.DataFrame]:
    original_linear_reg = dsp.LINEAR_REG
    dsp.LINEAR_REG = float(linear_reg)
    try:
        model, tuning = dsp.fit_variant(
            packet,
            dsp.VARIANTS["effective_exposure"],
            maxiter=int(args.maxiter),
            coarse_top_k=int(args.coarse_top_k),
            basin_hopping_iters=int(args.basin_hopping_iters),
        )
        train_prediction = dsp.predict(model, packet.w)
        oof_prediction, folds = oof_dsp_predictions(packet, model)
    finally:
        dsp.LINEAR_REG = original_linear_reg

    target_dir = output_dir / "dsp_effective_exposure" / target_name / f"linear_reg_{linear_reg:g}"
    target_dir.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(target_dir / "tuning.csv", index=False)
    (target_dir / "model.json").write_text(
        json.dumps(
            dsp.model_to_json(
                model,
                {
                    "target": target_metric,
                    "variant": "effective_exposure",
                    "linear_reg": linear_reg,
                },
            ),
            indent=2,
        )
        + "\n"
    )
    summary = summarize_predictions(
        model_family="dsp",
        variant="effective_exposure",
        target_name=target_name,
        target_metric=target_metric,
        hyperparameter_name="linear_reg",
        hyperparameter_value=linear_reg,
        panel=panel,
        metadata=metadata,
        observed=packet.y,
        train_prediction=train_prediction,
        oof_prediction=oof_prediction,
        folds=folds,
        notes="DSP effective-exposure variant; nonlinear parameters retuned for each L2 value.",
    )
    predictions = panel[["run_name", "source_experiment", "panel_source", target_name]].copy()
    predictions["model_family"] = "dsp"
    predictions["variant"] = "effective_exposure"
    predictions["target_name"] = target_name
    predictions["hyperparameter_name"] = "linear_reg"
    predictions["hyperparameter_value"] = float(linear_reg)
    predictions["train_prediction"] = train_prediction
    predictions["oof_prediction"] = oof_prediction
    predictions["train_residual"] = train_prediction - packet.y
    predictions["oof_residual"] = oof_prediction - packet.y
    predictions.to_csv(target_dir / "predictions.csv", index=False)
    return summary, predictions


def fit_olmix(
    *,
    panel: pd.DataFrame,
    metadata: dict[str, Any],
    columns: list[str],
    domains: list[str],
    target_name: str,
    target_metric: str,
    huber_delta: float,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ModelSummary, pd.DataFrame]:
    target = pd.to_numeric(panel[target_name], errors="coerce").to_numpy(dtype=float)
    weights = panel[columns].astype(float).to_numpy().reshape(len(panel), 2, len(domains))
    log_c, coefficients, huber_loss = olmix.fit_olmix_loglinear(
        weights,
        target,
        delta=float(huber_delta),
        seed=olmix.FIT_SEED,
        n_starts=int(args.olmix_starts),
        verbose=False,
    )
    train_prediction = olmix.predict(log_c, coefficients, weights)
    oof_prediction, folds = olmix.fit_oof_predictions(
        weights,
        target,
        delta=float(huber_delta),
        seed=olmix.CV_SEED,
        n_starts=int(args.olmix_starts),
    )
    target_dir = output_dir / "olmix" / target_name / f"huber_delta_{huber_delta:g}"
    target_dir.mkdir(parents=True, exist_ok=True)
    model_payload = {
        "target": target_metric,
        "huber_delta": float(huber_delta),
        "fit_huber_loss": float(huber_loss),
        "log_c": float(log_c),
        "coefficients": coefficients.tolist(),
        "phase_fractions": olmix.PHASE_FRACTIONS.tolist(),
        "fit_n_starts": int(args.olmix_starts),
        "functional_form": "exp(log_c) + exp(<coefficients, two_phase_weights>)",
    }
    (target_dir / "model.json").write_text(json.dumps(model_payload, indent=2) + "\n")
    summary = summarize_predictions(
        model_family="olmix",
        variant="loglinear",
        target_name=target_name,
        target_metric=target_metric,
        hyperparameter_name="huber_delta",
        hyperparameter_value=huber_delta,
        panel=panel,
        metadata=metadata,
        observed=target,
        train_prediction=train_prediction,
        oof_prediction=oof_prediction,
        folds=folds,
        notes="Reference OLMix log-linear functional form; Huber delta swept. OLMix has no official L2 term here.",
    )
    predictions = panel[["run_name", "source_experiment", "panel_source", target_name]].copy()
    predictions["model_family"] = "olmix"
    predictions["variant"] = "loglinear"
    predictions["target_name"] = target_name
    predictions["hyperparameter_name"] = "huber_delta"
    predictions["hyperparameter_value"] = float(huber_delta)
    predictions["train_prediction"] = train_prediction
    predictions["oof_prediction"] = oof_prediction
    predictions["train_residual"] = train_prediction - target
    predictions["oof_residual"] = oof_prediction - target
    predictions.to_csv(target_dir / "predictions.csv", index=False)
    return summary, predictions


def best_rows(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.sort_values(["model_family", "target_name", "selection_score", "oof_rmse"])
        .groupby(["model_family", "target_name"], as_index=False)
        .head(1)
        .sort_values(["target_name", "model_family"])
    )


def plot_sweep_metrics(output_dir: Path, summary: pd.DataFrame) -> None:
    for family, hp_name in (("dsp", "linear_reg"), ("olmix", "huber_delta")):
        subset = summary[summary["model_family"].eq(family)].copy()
        if subset.empty:
            continue
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("OOF RMSE", "OOF Spearman", "fold regret@1", "lower-tail optimism"),
        )
        colors = {
            "olmo_base_easy_qa_bpb": "#1f77b4",
            "olmo_base_easy_code_bpb": "#d62728",
            "olmo_base_easy_math_bpb": "#2ca02c",
            olmo_dsp.MACRO_TARGET: "#9467bd",
        }
        for target_name in TARGET_ORDER:
            view = subset[subset["target_name"].eq(target_name)].sort_values("hyperparameter_value")
            x = view["hyperparameter_value"]
            name = target_name.replace("olmo_base_easy_", "").replace("_bpb", "")
            color = colors[target_name]
            fig.add_trace(
                go.Scatter(x=x, y=view["oof_rmse"], mode="lines+markers", name=name, marker_color=color), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=x, y=view["oof_spearman"], mode="lines+markers", name=name, marker_color=color, showlegend=False
                ),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=view["fold_mean_regret_at_1"],
                    mode="lines+markers",
                    name=name,
                    marker_color=color,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=view["lower_tail_optimism"],
                    mode="lines+markers",
                    name=name,
                    marker_color=color,
                    showlegend=False,
                ),
                row=2,
                col=2,
            )
        fig.update_xaxes(title_text=hp_name, type="log" if family == "dsp" else None)
        fig.update_layout(
            title=f"{family.upper()} hyperparameter sweep on OLMoBaseEval Easy top-level BPBs",
            template="plotly_white",
            width=1400,
            height=900,
        )
        fig.write_html(output_dir / f"{family}_{hp_name}_sweep_metrics.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def plot_best_scatters(output_dir: Path, summary: pd.DataFrame, predictions: pd.DataFrame) -> None:
    best = best_rows(summary)
    keys = set(
        zip(
            best["model_family"],
            best["target_name"],
            best["hyperparameter_name"],
            best["hyperparameter_value"].round(12),
        )
    )
    selected = predictions[
        [
            (row.model_family, row.target_name, row.hyperparameter_name, round(float(row.hyperparameter_value), 12))
            in keys
            for row in predictions.itertuples(index=False)
        ]
    ].copy()
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[target.replace("olmo_base_easy_", "").replace("_bpb", "") for target in TARGET_ORDER],
    )
    colors = {"dsp": "#d62728", "olmix": "#1f77b4"}
    symbols = {"qsplit_signal": "circle", "domain_deletion": "diamond"}
    for idx, target_name in enumerate(TARGET_ORDER):
        row = idx // 2 + 1
        col = idx % 2 + 1
        observed_col = target_name
        view = selected[selected["target_name"].eq(target_name)].copy()
        lo = float(np.nanmin(view[[observed_col, "oof_prediction"]].to_numpy()))
        hi = float(np.nanmax(view[[observed_col, "oof_prediction"]].to_numpy()))
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                line={"dash": "dash", "color": "#666"},
                name="y=x" if idx == 0 else None,
                showlegend=idx == 0,
            ),
            row=row,
            col=col,
        )
        for family in ("dsp", "olmix"):
            family_view = view[view["model_family"].eq(family)]
            if family_view.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=family_view[observed_col],
                    y=family_view["oof_prediction"],
                    mode="markers",
                    marker={
                        "size": 7,
                        "opacity": 0.72,
                        "color": colors[family],
                        "symbol": [symbols.get(value, "circle") for value in family_view["panel_source"]],
                    },
                    text=family_view["run_name"],
                    customdata=np.stack(
                        [
                            family_view["hyperparameter_value"].astype(float).to_numpy(),
                            family_view["panel_source"].astype(str).to_numpy(),
                        ],
                        axis=1,
                    ),
                    name=family,
                    showlegend=idx == 0,
                    hovertemplate=(
                        "run=%{text}<br>panel=%{customdata[1]}<br>actual=%{x:.6f}<br>"
                        "oof=%{y:.6f}<br>hp=%{customdata[0]:.4g}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
    fig.update_xaxes(title_text="Observed BPB")
    fig.update_yaxes(title_text="OOF predicted BPB")
    fig.update_layout(
        title="Best selected OOF scatter: effective-exposure DSP vs OLMix",
        template="plotly_white",
        width=1400,
        height=1050,
    )
    fig.write_html(output_dir / "best_model_oof_scatter.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def plot_summary_bars(output_dir: Path, summary: pd.DataFrame) -> None:
    best = best_rows(summary)
    fig = make_subplots(rows=1, cols=3, subplot_titles=("OOF RMSE", "OOF Spearman", "fold regret@1"))
    colors = {"dsp": "#d62728", "olmix": "#1f77b4"}
    for family in ("dsp", "olmix"):
        view = best[best["model_family"].eq(family)]
        fig.add_trace(
            go.Bar(x=view["target_name"], y=view["oof_rmse"], name=family, marker_color=colors[family]),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=view["target_name"], y=view["oof_spearman"], name=family, marker_color=colors[family], showlegend=False
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=view["target_name"],
                y=view["fold_mean_regret_at_1"],
                name=family,
                marker_color=colors[family],
                showlegend=False,
            ),
            row=1,
            col=3,
        )
    fig.update_layout(
        title="Best selected fits by target",
        template="plotly_white",
        width=1500,
        height=650,
        barmode="group",
    )
    fig.write_html(output_dir / "best_model_metric_summary.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(output_dir: Path, summary: pd.DataFrame) -> None:
    best = best_rows(summary)
    columns = [
        "target_name",
        "model_family",
        "hyperparameter_name",
        "hyperparameter_value",
        "oof_spearman",
        "oof_rmse",
        "fold_mean_regret_at_1",
        "lower_tail_optimism",
        "low_tail_rmse",
        "train_spearman",
        "train_rmse",
        "selection_score",
        "predicted_best_observed_run_name",
        "predicted_best_observed_actual_value",
    ]
    lines = [
        "# OLMoBaseEval Easy Top-Level Model Sweeps",
        "",
        "Fit panel: 241 ex-ante qsplit/signal rows plus 39 domain-deletion rows; "
        "the adaptive OLMix row is excluded; the proportional target is replaced by the 11-row proportional-reference mean.",
        "",
        "Model-selection score is `oof_rmse + 0.5 * max(lower_tail_optimism, 0)`.",
        "",
        "## Best selected rows",
        "",
        best[columns].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Artifacts",
        "",
        "- `fit_summary.csv`",
        "- `fit_predictions.csv`",
        "- `best_selected_fit_summary.csv`",
        "- `best_model_oof_scatter.html`",
        "- `best_model_metric_summary.html`",
        "- `dsp_linear_reg_sweep_metrics.html`",
        "- `olmix_huber_delta_sweep_metrics.html`",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    linear_regs = parse_float_list(args.dsp_linear_reg_values)
    huber_deltas = parse_float_list(args.olmix_huber_deltas)
    _signal, columns, domains, _natural = olmix.load_raw_signal_panel()
    token_counts = olmix.load_domain_token_counts(domains)
    panels = build_all_target_panels(columns, domains, token_counts)

    summaries: list[ModelSummary] = []
    predictions: list[pd.DataFrame] = []
    for target_name in TARGET_ORDER:
        panel, metadata, packet = panels[target_name]
        target_metric = str(metadata["target_metric"])
        panel_dir = args.output_dir / "fit_panels" / target_name
        panel_dir.mkdir(parents=True, exist_ok=True)
        panel.to_csv(panel_dir / "fit_panel.csv", index=False)
        print(f"Target {target_name}: {len(panel)} rows", flush=True)

        for linear_reg in linear_regs:
            print(f"  DSP effective_exposure LINEAR_REG={linear_reg:g}", flush=True)
            summary, pred = fit_effective_exposure_dsp(
                panel=panel,
                metadata=metadata,
                packet=packet,
                target_name=target_name,
                target_metric=target_metric,
                linear_reg=linear_reg,
                args=args,
                output_dir=args.output_dir,
            )
            summaries.append(summary)
            predictions.append(pred)

        for huber_delta in huber_deltas:
            print(f"  OLMix Huber delta={huber_delta:g}", flush=True)
            summary, pred = fit_olmix(
                panel=panel,
                metadata=metadata,
                columns=columns,
                domains=domains,
                target_name=target_name,
                target_metric=target_metric,
                huber_delta=huber_delta,
                args=args,
                output_dir=args.output_dir,
            )
            summaries.append(summary)
            predictions.append(pred)

    summary_frame = pd.DataFrame([asdict(summary) for summary in summaries])
    predictions_frame = pd.concat(predictions, ignore_index=True)
    best = best_rows(summary_frame)
    summary_frame.to_csv(args.output_dir / "fit_summary.csv", index=False)
    predictions_frame.to_csv(args.output_dir / "fit_predictions.csv", index=False)
    best.to_csv(args.output_dir / "best_selected_fit_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "targets": TARGET_ORDER,
                "dsp_linear_reg_values": linear_regs,
                "olmix_huber_deltas": huber_deltas,
                "fit_rows_per_target": {target: int(len(panels[target][0])) for target in TARGET_ORDER},
                "best_rows": best.to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n"
    )
    plot_sweep_metrics(args.output_dir, summary_frame)
    plot_best_scatters(args.output_dir, summary_frame, predictions_frame)
    plot_summary_bars(args.output_dir, summary_frame)
    write_report(args.output_dir, summary_frame)
    print(
        best[
            [
                "target_name",
                "model_family",
                "hyperparameter_name",
                "hyperparameter_value",
                "oof_spearman",
                "oof_rmse",
                "fold_mean_regret_at_1",
                "lower_tail_optimism",
                "low_tail_rmse",
                "train_spearman",
                "train_rmse",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
