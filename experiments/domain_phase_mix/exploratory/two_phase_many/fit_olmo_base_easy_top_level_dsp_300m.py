# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Fit DSP variants to OLMoBaseEval Easy top-level BPB aggregates at 300M.

The fit panel matches the deletion-augmented 300M convention used by the
OLMix/DSP uncheatable-BPB comparison:

- 241 ex-ante qsplit/signal rows, excluding the adaptive OLMix row;
- 39 proportional domain-deletion rows;
- the proportional row target is replaced by the mean of the original
  proportional checkpoint plus the 10 proportional-noise repeats.

Targets are the evaluator-emitted OLMoBaseEval Easy top-level aggregate BPBs:
QA, code, math, and their simple 3-way macro.
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

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_olmo_base_easy_domain_ablation_pvalues as olmo_pvalues,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    fit_olmix_reference_deletion_augmented_300m as olmix,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "olmo_base_easy_top_level_dsp_300m_20260625"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}

TARGET_KEYS = {
    "olmo_base_easy_qa_bpb": "olmo_base_eval/easy_bpb/olmobase_easy_qa/bpb",
    "olmo_base_easy_code_bpb": "olmo_base_eval/easy_bpb/olmobase_easy_code/bpb",
    "olmo_base_easy_math_bpb": "olmo_base_eval/easy_bpb/olmobase_easy_math/bpb",
}
MACRO_TARGET = "olmo_base_easy_top3_macro_bpb"
VARIANT_NAMES = ("canonical", "effective_exposure")


@dataclass(frozen=True)
class FitSummary:
    target_name: str
    target_metric: str
    variant: str
    variant_description: str
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
    train_best_observed_run_name: str
    train_best_observed_value: float
    predicted_best_observed_run_name: str
    predicted_best_observed_actual_value: float
    predicted_best_observed_predicted_value: float
    total_param_count: int
    m_dependent_params_per_domain: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--maxiter", type=int, default=dsp.FIT_MAXITER)
    parser.add_argument("--coarse-top-k", type=int, default=dsp.START_TOP_K)
    parser.add_argument("--basin-hopping-iters", type=int, default=3)
    return parser.parse_args()


def load_all_300m_olmo_metrics() -> pd.DataFrame:
    manifest = olmo_pvalues.load_manifests(
        olmo_pvalues.DEFAULT_716_MANIFEST,
        olmo_pvalues.DEFAULT_PROP_NOISE_MANIFEST,
    )
    wanted = manifest[
        manifest["scale"].eq(olmo_pvalues.SCALE)
        & (
            manifest["panel"].eq("parity")
            | (
                manifest["panel"].eq(olmo_pvalues.PCTRL_PANEL)
                & manifest["run_name"].str.startswith("pctrl_del_", na=False)
            )
            | manifest["panel"].eq(olmo_pvalues.PROP_NOISE_PANEL)
        )
    ].copy()
    if wanted.empty:
        raise ValueError("No 300M OLMoBaseEval rows found")

    rows: list[dict[str, Any]] = []
    missing_outputs: list[str] = []
    for row in wanted.itertuples(index=False):
        try:
            metric_map = olmo_pvalues.metric_map_for_output(
                olmo_pvalues.DEFAULT_METRICS_ROOT,
                str(row.output_name),
            )
        except FileNotFoundError:
            missing_outputs.append(str(row.output_name))
            continue
        for metric in metric_map.values():
            if metric.benchmark_key not in set(TARGET_KEYS.values()):
                continue
            rows.append(
                {
                    "scale": str(row.scale),
                    "panel": str(row.panel),
                    "run_name": str(row.run_name),
                    "output_name": str(row.output_name),
                    "wandb_run_id": str(row.wandb_run_id),
                    "target_metric": metric.benchmark_key,
                    "olmo_task": metric.olmo_task,
                    "value_bpb": metric.value_bpb,
                }
            )
    if missing_outputs:
        raise ValueError(f"Missing metrics.json for {len(missing_outputs)} outputs; examples={missing_outputs[:8]}")
    metrics = pd.DataFrame(rows)
    expected_runs = wanted["run_name"].nunique()
    counts = metrics.groupby("target_metric")["run_name"].nunique()
    missing_counts = {
        metric: int(expected_runs - count) for metric, count in counts.items() if int(count) != expected_runs
    }
    if missing_counts:
        print(
            f"WARNING: top-level BPB coverage is incomplete for some targets: {missing_counts}; "
            "fitting target-specific complete cases.",
            flush=True,
        )
    return metrics


def build_target_wide(metrics: pd.DataFrame) -> pd.DataFrame:
    name_by_key = {value: key for key, value in TARGET_KEYS.items()}
    top = metrics.copy()
    top["target_name"] = top["target_metric"].map(name_by_key)
    index_columns = ["panel", "run_name", "output_name", "wandb_run_id"]
    if top.duplicated(index_columns + ["target_name"]).any():
        duplicated = top.loc[
            top.duplicated(index_columns + ["target_name"], keep=False), index_columns + ["target_name"]
        ]
        raise ValueError(f"Duplicate top-level metric rows:\n{duplicated.head(20)}")
    wide = top.set_index(index_columns + ["target_name"])["value_bpb"].unstack("target_name").reset_index()
    wide = wide.reset_index()
    missing_targets = sorted(set(TARGET_KEYS).difference(wide.columns))
    if missing_targets:
        raise ValueError(f"Missing top-level target columns: {missing_targets}")
    wide[MACRO_TARGET] = wide[list(TARGET_KEYS)].mean(axis=1, skipna=False)
    return wide


def build_fit_panel(columns: list[str], targets: pd.DataFrame, target_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    signal, _, _, _ = olmix.load_raw_signal_panel()
    signal = signal.copy()
    signal["panel_source"] = "qsplit_signal"
    deletion = olmix.load_deletion_weights(columns)
    deletion["panel_source"] = "domain_deletion"

    fit_rows = pd.concat(
        [
            signal[["run_name", "source_experiment", "panel_source", *columns]],
            deletion[["run_name", "source_experiment", "panel_source", *columns]],
        ],
        ignore_index=True,
    )
    target_values = targets[["panel", "run_name", target_name]].copy()
    fit_target = target_values[target_values["panel"].isin(["parity", olmo_pvalues.PCTRL_PANEL])]
    fit_rows = fit_rows.merge(fit_target[["run_name", target_name]], on="run_name", how="left", validate="one_to_one")

    reference_values = target_values[
        (target_values["panel"].eq("parity") & target_values["run_name"].eq(olmo_pvalues.BASELINE_RUN_NAME))
        | target_values["panel"].eq(olmo_pvalues.PROP_NOISE_PANEL)
    ][target_name]
    fit_rows, ref_n, ref_mean, ref_std = olmix.replace_proportional_target_with_reference_mean(
        fit_rows,
        target_column=target_name,
        reference=reference_values,
    )
    fit_rows = fit_rows[pd.to_numeric(fit_rows[target_name], errors="coerce").notna()].reset_index(drop=True)
    if len(fit_rows) < 279:
        raise ValueError(f"Expected at least 279 fit rows for {target_name}, found {len(fit_rows)}")
    signal_count = int(fit_rows["panel_source"].eq("qsplit_signal").sum())
    if signal_count < 240:
        raise ValueError(f"Expected at least 240 signal rows for {target_name}, found {signal_count}")
    if int(fit_rows["panel_source"].eq("domain_deletion").sum()) != 39:
        raise ValueError(f"Expected 39 deletion rows for {target_name}")
    if ref_n != 11 or ref_mean is None:
        raise ValueError(f"Expected 11 proportional reference rows for {target_name}, found {ref_n}")
    metadata = {
        "target_name": target_name,
        "target_metric": target_name if target_name == MACRO_TARGET else TARGET_KEYS[target_name],
        "n_proportional_reference_rows": ref_n,
        "proportional_reference_mean": ref_mean,
        "proportional_reference_std": ref_std,
    }
    return fit_rows, metadata


def build_dsp_packet(
    panel: pd.DataFrame, columns: list[str], domains: list[str], token_counts: np.ndarray, target_name: str
) -> dsp.PacketData:
    weights = panel[columns].astype(float).to_numpy().reshape(len(panel), 2, len(domains))
    weights = dsp.normalize_weights(weights)
    phase_epoch_multipliers = olmix.PHASE_FRACTIONS[:, None] * float(olmix.load_target_budget()) / token_counts[None, :]
    return dsp.PacketData(
        frame=panel.reset_index(drop=True),
        name_col="run_name",
        y=pd.to_numeric(panel[target_name], errors="coerce").to_numpy(dtype=float),
        w=weights,
        m=len(domains),
        c0=phase_epoch_multipliers[0],
        c1=phase_epoch_multipliers[1],
        domain_names=list(domains),
    )


def regression_metrics(y: np.ndarray, y_hat: np.ndarray) -> tuple[float, float, float, float]:
    residual = y_hat - y
    rmse = float(np.sqrt(np.mean(residual * residual)))
    mae = float(np.mean(np.abs(residual)))
    pearson = float(pearsonr(y, y_hat).statistic) if np.std(y) > 0.0 and np.std(y_hat) > 0.0 else float("nan")
    spearman = float(spearmanr(y, y_hat).statistic) if np.std(y) > 0.0 and np.std(y_hat) > 0.0 else float("nan")
    return rmse, mae, pearson, spearman


def fit_dsp_oof_predictions(
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


def fit_one(
    *,
    packet: dsp.PacketData,
    panel: pd.DataFrame,
    target_name: str,
    target_metric: str,
    metadata: dict[str, Any],
    variant_name: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[FitSummary, pd.DataFrame]:
    variant = dsp.VARIANTS[variant_name]
    print(f"Fitting {variant_name} on {target_name}", flush=True)
    model, tuning = dsp.fit_variant(
        packet,
        variant,
        maxiter=int(args.maxiter),
        coarse_top_k=int(args.coarse_top_k),
        basin_hopping_iters=int(args.basin_hopping_iters),
    )
    target_dir = output_dir / target_name / variant_name
    target_dir.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(target_dir / "dsp_tuning.csv", index=False)
    (target_dir / "model.json").write_text(
        json.dumps(dsp.model_to_json(model, {"target": target_metric, "variant": variant_name}), indent=2)
    )

    train_pred = dsp.predict(model, packet.w)
    oof_pred, folds = fit_dsp_oof_predictions(packet, model)
    train_rmse, train_mae, train_pearson, train_spearman = regression_metrics(packet.y, train_pred)
    oof_metrics = olmix.predictive_diagnostics(packet.y, oof_pred, folds)

    best_idx = int(np.argmin(packet.y))
    pred_best_idx = int(np.argmin(oof_pred))
    summary = FitSummary(
        target_name=target_name,
        target_metric=target_metric,
        variant=variant_name,
        variant_description=variant.description,
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
        oof_rmse=float(oof_metrics["rmse"]),
        oof_mae=float(oof_metrics["mae"]),
        oof_pearson=float(oof_metrics["pearson"]),
        oof_spearman=float(oof_metrics["spearman"]),
        fold_mean_regret_at_1=float(oof_metrics["fold_mean_regret_at_1"]),
        lower_tail_optimism=float(oof_metrics["lower_tail_optimism"]),
        low_tail_rmse=float(oof_metrics["low_tail_rmse"]),
        train_best_observed_run_name=str(panel.iloc[best_idx]["run_name"]),
        train_best_observed_value=float(packet.y[best_idx]),
        predicted_best_observed_run_name=str(panel.iloc[pred_best_idx]["run_name"]),
        predicted_best_observed_actual_value=float(packet.y[pred_best_idx]),
        predicted_best_observed_predicted_value=float(oof_pred[pred_best_idx]),
        total_param_count=int(model.total_param_count),
        m_dependent_params_per_domain=int(model.m_dependent_params_per_domain),
    )

    predictions = panel[["run_name", "source_experiment", "panel_source", target_name]].copy()
    predictions["target_name"] = target_name
    predictions["variant"] = variant_name
    predictions["train_prediction"] = train_pred
    predictions["oof_prediction"] = oof_pred
    predictions["train_residual"] = train_pred - packet.y
    predictions["oof_residual"] = oof_pred - packet.y
    predictions.to_csv(target_dir / "fit_predictions.csv", index=False)
    return summary, predictions


def write_fit_scatter(output_dir: Path, predictions: pd.DataFrame) -> None:
    targets = list(TARGET_KEYS) + [MACRO_TARGET]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[target.replace("olmo_base_easy_", "").replace("_bpb", "") for target in targets],
    )
    color_by_variant = {"canonical": "#1f77b4", "effective_exposure": "#d62728"}
    for idx, target_name in enumerate(targets):
        row = idx // 2 + 1
        col = idx % 2 + 1
        subset = predictions[predictions["target_name"].eq(target_name)]
        lo = float(np.nanmin(subset[[target_name, "oof_prediction"]].to_numpy()))
        hi = float(np.nanmax(subset[[target_name, "oof_prediction"]].to_numpy()))
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
        for variant_name in VARIANT_NAMES:
            view = subset[subset["variant"].eq(variant_name)]
            fig.add_trace(
                go.Scatter(
                    x=view[target_name],
                    y=view["oof_prediction"],
                    mode="markers",
                    marker={"size": 6, "opacity": 0.75, "color": color_by_variant[variant_name]},
                    text=view["run_name"],
                    name=variant_name,
                    showlegend=idx == 0,
                    hovertemplate="run=%{text}<br>actual=%{x:.6f}<br>oof=%{y:.6f}<extra></extra>",
                ),
                row=row,
                col=col,
            )
    fig.update_xaxes(title_text="Observed BPB")
    fig.update_yaxes(title_text="OOF predicted BPB")
    fig.update_layout(
        title="OLMoBaseEval Easy top-level aggregate BPB: DSP OOF predictions",
        template="plotly_white",
        width=1280,
        height=1000,
    )
    fig.write_html(output_dir / "top_level_dsp_oof_fit_scatter.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_summary_plots(output_dir: Path, summary: pd.DataFrame) -> None:
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("OOF Spearman", "OOF RMSE", "Fold regret@1"),
    )
    colors = {"canonical": "#1f77b4", "effective_exposure": "#d62728"}
    for variant_name in VARIANT_NAMES:
        view = summary[summary["variant"].eq(variant_name)]
        fig.add_trace(
            go.Bar(x=view["target_name"], y=view["oof_spearman"], name=variant_name, marker_color=colors[variant_name]),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=view["target_name"],
                y=view["oof_rmse"],
                name=variant_name,
                marker_color=colors[variant_name],
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=view["target_name"],
                y=view["fold_mean_regret_at_1"],
                name=variant_name,
                marker_color=colors[variant_name],
                showlegend=False,
            ),
            row=1,
            col=3,
        )
    fig.update_layout(
        title="Canonical vs effective-exposure DSP on OLMoBaseEval Easy top-level BPBs",
        template="plotly_white",
        width=1500,
        height=620,
        barmode="group",
    )
    fig.write_html(output_dir / "top_level_dsp_metric_summary.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(output_dir: Path, summary: pd.DataFrame) -> None:
    key_columns = [
        "target_name",
        "variant",
        "oof_spearman",
        "oof_rmse",
        "fold_mean_regret_at_1",
        "lower_tail_optimism",
        "low_tail_rmse",
        "train_spearman",
        "train_rmse",
        "proportional_reference_mean",
        "predicted_best_observed_run_name",
        "predicted_best_observed_actual_value",
    ]
    lines = [
        "# OLMoBaseEval Easy Top-Level BPB DSP Fits",
        "",
        "Targets are the evaluator-emitted top-level aggregate BPBs for QA, code, math, plus a simple 3-way macro.",
        "",
        "Fit panel: 241 ex-ante qsplit/signal rows plus 39 domain-deletion rows. The adaptive OLMix row is excluded. The proportional row is replaced by the 11-row proportional reference mean.",
        "",
        "OOF predictions reuse the full-data nonlinear DSP geometry and refit the linear head per fold, matching the existing DSP/OLMix comparison convention.",
        "",
        summary[key_columns].to_markdown(index=False, floatfmt=".6f"),
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _signal, columns, domains, _natural = olmix.load_raw_signal_panel()
    token_counts = olmix.load_domain_token_counts(domains)
    metrics = load_all_300m_olmo_metrics()
    target_wide = build_target_wide(metrics)

    summaries: list[FitSummary] = []
    prediction_frames: list[pd.DataFrame] = []
    for target_name in [*TARGET_KEYS.keys(), MACRO_TARGET]:
        panel, metadata = build_fit_panel(columns, target_wide, target_name)
        packet = build_dsp_packet(panel, columns, domains, token_counts, target_name)
        target_metric = str(metadata["target_metric"])
        target_panel_dir = args.output_dir / target_name
        target_panel_dir.mkdir(parents=True, exist_ok=True)
        panel[["run_name", "source_experiment", "panel_source", target_name]].to_csv(
            target_panel_dir / "fit_panel.csv",
            index=False,
        )
        for variant_name in VARIANT_NAMES:
            summary, predictions = fit_one(
                packet=packet,
                panel=panel,
                target_name=target_name,
                target_metric=target_metric,
                metadata=metadata,
                variant_name=variant_name,
                args=args,
                output_dir=args.output_dir,
            )
            summaries.append(summary)
            prediction_frames.append(predictions)

    summary_frame = pd.DataFrame([asdict(summary) for summary in summaries])
    predictions_frame = pd.concat(prediction_frames, ignore_index=True)
    summary_frame.to_csv(args.output_dir / "top_level_dsp_fit_summary.csv", index=False)
    predictions_frame.to_csv(args.output_dir / "top_level_dsp_fit_predictions.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps([asdict(summary) for summary in summaries], indent=2))
    write_fit_scatter(args.output_dir, predictions_frame)
    write_summary_plots(args.output_dir, summary_frame)
    write_report(args.output_dir, summary_frame)
    print(
        summary_frame[
            [
                "target_name",
                "variant",
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
