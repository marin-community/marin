# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Diagnose whether uncheatable DSP heldout optimism is a bad phase multiplier.

This script asks a narrow question: for the old two-phase effective-exposure
DSP fit on Uncheatable BPB, can retuning the scalar phase-1 multiplier explain
the poor predictions on the exposure-repair validation points, or is the failure
more structural?

It uses the 300M deletion-augmented uncheatable fit panel and the four completed
3e18 exposure-repair validation mixtures. It does not launch jobs or read remote
state.
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

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_dsp_l2_kl_sweep_deletion_augmented_300m as l2_sweep,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_dsp_vs_olmix_deletion_augmented_300m as dsp_compare,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as olmix,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "dsp_uncheatable_eta_diagnostic_20260703"
DEFAULT_REPAIR_RESULTS = (
    REFERENCE_OUTPUTS / "delphi_dsp_exposure_repair_validation_20260702" / "results_from_wandb.csv"
)
DEFAULT_REPAIR_MIXTURE_DIR = (
    REFERENCE_OUTPUTS / "dsp_exposure_repair_validation_mixtures_20260702" / "mixtures"
)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
LINEAR_REG = 0.01


@dataclass(frozen=True)
class ModelSummary:
    model_name: str
    variant: str
    phase_parameter_summary: str
    train_rmse: float
    train_spearman: float
    oof_rmse: float
    oof_spearman: float
    oof_lower_tail_optimism: float
    oof_low_tail_rmse: float
    heldout_all_mae: float
    heldout_all_rmse: float
    heldout_uncheatable_mae: float
    heldout_uncheatable_rmse: float
    heldout_predicted_best_uncheatable: str
    heldout_actual_best_uncheatable: str
    heldout_best_order_correct: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repair-results", type=Path, default=DEFAULT_REPAIR_RESULTS)
    parser.add_argument("--repair-mixture-dir", type=Path, default=DEFAULT_REPAIR_MIXTURE_DIR)
    parser.add_argument("--maxiter", type=int, default=dsp.FIT_MAXITER)
    parser.add_argument("--coarse-top-k", type=int, default=dsp.START_TOP_K)
    parser.add_argument("--basin-hopping-iters", type=int, default=1)
    return parser.parse_args()


def phase_summary(model: dsp.FittedDSPModel) -> str:
    params = model.params
    keys = [
        "gamma",
        "gamma_saturation",
        "gamma_penalty",
        "lambda_retention",
        "eta",
        "eta_saturation",
        "eta_penalty",
        "phi",
    ]
    parts = []
    for key in keys:
        if key in params:
            parts.append(f"{key}={float(params[key]):.6g}")
    return ", ".join(parts) if parts else "none"


def load_packet() -> tuple[dsp.PacketData, pd.DataFrame, list[str], np.ndarray, np.ndarray, int]:
    _signal, columns, domains, natural = olmix.load_raw_signal_panel()
    target_budget = olmix.load_target_budget()
    token_counts = olmix.load_domain_token_counts(domains)
    panel, _metadata = olmix.build_uncheatable_panel(columns)
    packet = dsp_compare.build_dsp_packet(panel, columns, domains, token_counts, target_budget)
    return packet, panel, domains, natural, token_counts, target_budget


def fit_model(packet: dsp.PacketData, variant_key: str, args: argparse.Namespace) -> dsp.FittedDSPModel:
    original_linear_reg = dsp.LINEAR_REG
    dsp.LINEAR_REG = LINEAR_REG
    try:
        model, _train_metrics, _oof_metrics, _oof_pred, _tuning = l2_sweep.fit_one_dsp_model(
            packet,
            variant_key=variant_key,
            linear_reg=LINEAR_REG,
            maxiter=int(args.maxiter),
            coarse_top_k=int(args.coarse_top_k),
            basin_hopping_iters=int(args.basin_hopping_iters),
        )
        return model
    finally:
        dsp.LINEAR_REG = original_linear_reg


def fixed_gamma_model(packet: dsp.PacketData, base_model: dsp.FittedDSPModel, gamma: float) -> dsp.FittedDSPModel:
    params: dict[str, float | np.ndarray] = {}
    for key, value in base_model.params.items():
        if isinstance(value, np.ndarray):
            params[key] = np.asarray(value, dtype=float).copy()
        else:
            params[key] = float(value)
    params["gamma"] = float(gamma)
    original_linear_reg = dsp.LINEAR_REG
    dsp.LINEAR_REG = LINEAR_REG
    try:
        return dsp.fit_linear_head(packet.w, packet.y, packet, base_model.variant, params)
    finally:
        dsp.LINEAR_REG = original_linear_reg


def repair_weights(model: dsp.FittedDSPModel, mixture_path: Path) -> np.ndarray:
    frame = pd.read_csv(mixture_path).set_index("domain")
    missing = sorted(set(model.domain_names).difference(frame.index))
    if missing:
        raise ValueError(f"{mixture_path} missing domains: {missing[:8]}")
    ordered = frame.loc[model.domain_names]
    weights = ordered[["phase_0_weight", "phase_1_weight"]].to_numpy(dtype=float).T
    return dsp.normalize_weights(weights[None, :, :])


def load_heldout(args: argparse.Namespace) -> pd.DataFrame:
    results = pd.read_csv(args.repair_results)
    rows: list[dict[str, Any]] = []
    for row in results.itertuples(index=False):
        mixture = str(row.mixture)
        mixture_path = args.repair_mixture_dir / f"{mixture}.csv"
        if not mixture_path.exists():
            raise FileNotFoundError(mixture_path)
        rows.append(
            {
                "mixture": mixture,
                "mixture_path": mixture_path,
                "uncheatable_bpb": float(row.uncheatable_bpb),
                "table9_macro_bpb": float(row.table9_macro_bpb),
                "is_uncheatable_objective": mixture.startswith("dsp_uncheatable_"),
            }
        )
    return pd.DataFrame(rows)


def oof_metrics(packet: dsp.PacketData, model: dsp.FittedDSPModel) -> dict[str, float]:
    oof, folds = dsp_compare.fit_dsp_oof_predictions(packet, model)
    return olmix.predictive_diagnostics(packet.y, oof, folds)


def summarize_model(
    *,
    model_name: str,
    model: dsp.FittedDSPModel,
    packet: dsp.PacketData,
    heldout: pd.DataFrame,
) -> tuple[ModelSummary, pd.DataFrame]:
    train_pred = dsp.predict(model, packet.w)
    train_rmse, _train_mae, _train_pearson, train_spearman = dsp_compare.regression_metrics(packet.y, train_pred)
    metrics = oof_metrics(packet, model)

    prediction_rows: list[dict[str, Any]] = []
    for row in heldout.itertuples(index=False):
        weights = repair_weights(model, Path(row.mixture_path))
        prediction = float(dsp.predict(model, weights)[0])
        actual = float(row.uncheatable_bpb)
        prediction_rows.append(
            {
                "model_name": model_name,
                "mixture": str(row.mixture),
                "is_uncheatable_objective": bool(row.is_uncheatable_objective),
                "actual_uncheatable_bpb": actual,
                "predicted_uncheatable_bpb": prediction,
                "prediction_error": prediction - actual,
                "absolute_error": abs(prediction - actual),
            }
        )
    predictions = pd.DataFrame(prediction_rows)
    all_residual = predictions["prediction_error"].to_numpy(dtype=float)
    uncheatable_predictions = predictions[predictions["is_uncheatable_objective"]].copy()
    uncheatable_residual = uncheatable_predictions["prediction_error"].to_numpy(dtype=float)
    predicted_best = str(
        uncheatable_predictions.loc[uncheatable_predictions["predicted_uncheatable_bpb"].idxmin(), "mixture"]
    )
    actual_best = str(
        uncheatable_predictions.loc[uncheatable_predictions["actual_uncheatable_bpb"].idxmin(), "mixture"]
    )
    summary = ModelSummary(
        model_name=model_name,
        variant=model.variant.name,
        phase_parameter_summary=phase_summary(model),
        train_rmse=float(train_rmse),
        train_spearman=float(train_spearman),
        oof_rmse=float(metrics["rmse"]),
        oof_spearman=float(metrics["spearman"]),
        oof_lower_tail_optimism=float(metrics["lower_tail_optimism"]),
        oof_low_tail_rmse=float(metrics["low_tail_rmse"]),
        heldout_all_mae=float(np.mean(np.abs(all_residual))),
        heldout_all_rmse=float(np.sqrt(np.mean(all_residual * all_residual))),
        heldout_uncheatable_mae=float(np.mean(np.abs(uncheatable_residual))),
        heldout_uncheatable_rmse=float(np.sqrt(np.mean(uncheatable_residual * uncheatable_residual))),
        heldout_predicted_best_uncheatable=predicted_best,
        heldout_actual_best_uncheatable=actual_best,
        heldout_best_order_correct=predicted_best == actual_best,
    )
    return summary, predictions


def write_gamma_plot(output_dir: Path, gamma_sweep: pd.DataFrame, selected_gamma: float) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=gamma_sweep["gamma"],
            y=gamma_sweep["oof_rmse"],
            mode="lines+markers",
            name="OOF RMSE",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=gamma_sweep["gamma"],
            y=gamma_sweep["heldout_uncheatable_mae"],
            mode="lines+markers",
            name="heldout uncheatable MAE",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=gamma_sweep["gamma"],
            y=gamma_sweep["heldout_all_mae"],
            mode="lines+markers",
            name="heldout all repair MAE",
        )
    )
    fig.add_vline(
        x=selected_gamma,
        line_dash="dash",
        line_color="#475569",
        annotation_text=f"fitted gamma={selected_gamma:.3g}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Uncheatable effective-exposure DSP: fixed-shape gamma sweep",
        xaxis_title="phase-1 exposure multiplier gamma",
        yaxis_title="BPB-scale error",
        xaxis_type="log",
        template="plotly_white",
        width=1100,
        height=650,
    )
    fig.write_html(output_dir / "uncheatable_fixed_shape_gamma_sweep.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(
    output_dir: Path,
    selected_gamma: float,
    gamma_sweep: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> None:
    best_heldout = gamma_sweep.loc[gamma_sweep["heldout_uncheatable_mae"].idxmin()]
    best_oof = gamma_sweep.loc[gamma_sweep["oof_selection_score"].idxmin()]
    lines = [
        "# Uncheatable DSP phase-multiplier diagnostic",
        "",
        "Question: can retuning the effective-exposure phase-1 multiplier explain the repaired-candidate heldout miss?",
        "",
        f"Old fitted effective-exposure gamma: `{selected_gamma:.6g}`.",
        f"Best fixed-shape gamma by uncheatable heldout MAE: `{float(best_heldout['gamma']):.6g}` with MAE `{float(best_heldout['heldout_uncheatable_mae']):.6f}`.",
        f"Best fixed-shape gamma by OOF selection score: `{float(best_oof['gamma']):.6g}` with heldout MAE `{float(best_oof['heldout_uncheatable_mae']):.6f}`.",
        "",
        "## Variant summary",
        "",
        model_summary[
            [
                "model_name",
                "phase_parameter_summary",
                "oof_rmse",
                "oof_spearman",
                "heldout_uncheatable_mae",
                "heldout_all_mae",
                "heldout_predicted_best_uncheatable",
                "heldout_best_order_correct",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
        "If the best fixed-shape gamma remains badly overoptimistic, the issue is not merely that the scalar was tuned to the wrong value. If a no-phase or retained variant predicts heldouts much better while similar OOF diagnostics hold, the current effective-exposure functional form is the wrong inductive bias for frontier proposals.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packet, _panel, _domains, _natural, _token_counts, _target_budget = load_packet()
    heldout = load_heldout(args)

    variants = ["effective_exposure", "canonical", "no_phase", "retained_effective_exposure", "split_saturation_penalty"]
    summaries: list[ModelSummary] = []
    prediction_frames: list[pd.DataFrame] = []
    fitted_models: dict[str, dsp.FittedDSPModel] = {}
    for variant in variants:
        print(f"Fitting {variant}", flush=True)
        model = fit_model(packet, variant, args)
        fitted_models[variant] = model
        summary, predictions = summarize_model(model_name=variant, model=model, packet=packet, heldout=heldout)
        summaries.append(summary)
        prediction_frames.append(predictions)

    effective = fitted_models["effective_exposure"]
    selected_gamma = float(effective.params["gamma"])
    gamma_values = np.unique(
        np.concatenate(
            [
                np.geomspace(1e-4, 100.0, 49),
                np.asarray([selected_gamma], dtype=float),
            ]
        )
    )
    gamma_rows: list[dict[str, Any]] = []
    for gamma in gamma_values:
        model = fixed_gamma_model(packet, effective, float(gamma))
        summary, predictions = summarize_model(
            model_name=f"effective_exposure_fixed_gamma_{gamma:.6g}",
            model=model,
            packet=packet,
            heldout=heldout,
        )
        row = asdict(summary)
        row["gamma"] = float(gamma)
        row["oof_selection_score"] = row["oof_rmse"] + 0.5 * row["oof_lower_tail_optimism"]
        gamma_rows.append(row)
        prediction_frames.append(predictions)

    model_summary = pd.DataFrame([asdict(summary) for summary in summaries])
    heldout_predictions = pd.concat(prediction_frames, ignore_index=True)
    gamma_sweep = pd.DataFrame(gamma_rows).sort_values("gamma").reset_index(drop=True)
    model_summary.to_csv(args.output_dir / "uncheatable_variant_heldout_summary.csv", index=False)
    heldout_predictions.to_csv(args.output_dir / "uncheatable_heldout_predictions.csv", index=False)
    gamma_sweep.to_csv(args.output_dir / "uncheatable_fixed_shape_gamma_sweep.csv", index=False)
    write_gamma_plot(args.output_dir, gamma_sweep, selected_gamma)
    write_report(args.output_dir, selected_gamma, gamma_sweep, model_summary)
    metadata = {
        "linear_reg": LINEAR_REG,
        "target_metric": olmix.UNCHEATABLE_TARGET,
        "selected_effective_exposure_gamma": selected_gamma,
        "heldout_rows": heldout["mixture"].tolist(),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print("Variant summary")
    print(model_summary.to_string(index=False))
    print("Best gamma by heldout_uncheatable_mae")
    print(gamma_sweep.loc[gamma_sweep["heldout_uncheatable_mae"].idxmin()].to_string())
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
