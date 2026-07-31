# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Diagnose satiating late-exposure DSP variants for Uncheatable BPB.

This is a local-only diagnostic. It tests the mechanistic hypothesis that the
phase-1 benefit has high marginal value for small late exposure but satiates as
late exposure grows. That targets the current failure mode where constant-gamma
effective-exposure DSP is good in-support but badly overoptimistic on repaired
frontier candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_dsp_vs_olmix_deletion_augmented_300m as dsp_compare,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as olmix,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import diagnose_dsp_uncheatable_eta_heldout as eta_diag  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "dsp_uncheatable_satiating_late_bonus_20260703"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
LINEAR_REG = 0.01
TARGET = "eval/uncheatable_eval/bpb"


@dataclass(frozen=True)
class VariantRow:
    model_name: str
    variant_key: str
    phase_params: str
    train_rmse: float
    train_spearman: float
    oof_rmse: float
    oof_spearman: float
    oof_lower_tail_optimism: float
    oof_low_tail_rmse: float
    heldout_all_mae: float
    heldout_uncheatable_mae: float
    heldout_max_optimism: float
    heldout_gap_error: float
    heldout_best_order_correct: bool
    leave_good_rmse: float
    leave_good_signed_optimism: float
    raw_predicted_optimum: float | None
    raw_nearest_observed_tv: float | None
    raw_nearest_observed_value: float | None
    pass_gates: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repair-results", type=Path, default=eta_diag.DEFAULT_REPAIR_RESULTS)
    parser.add_argument("--repair-mixture-dir", type=Path, default=eta_diag.DEFAULT_REPAIR_MIXTURE_DIR)
    parser.add_argument("--maxiter", type=int, default=dsp.FIT_MAXITER)
    parser.add_argument("--coarse-top-k", type=int, default=dsp.START_TOP_K)
    parser.add_argument("--basin-hopping-iters", type=int, default=0)
    parser.add_argument("--good-frac", type=float, default=0.15)
    return parser.parse_args()


def regression_metrics(y: np.ndarray, y_hat: np.ndarray) -> tuple[float, float, float, float]:
    residual = np.asarray(y_hat, dtype=float) - np.asarray(y, dtype=float)
    rmse = float(np.sqrt(np.mean(residual * residual)))
    mae = float(np.mean(np.abs(residual)))
    pearson = float(pearsonr(y, y_hat).statistic) if np.std(y) > 0.0 and np.std(y_hat) > 0.0 else float("nan")
    spearman = float(spearmanr(y, y_hat).statistic) if np.std(y) > 0.0 and np.std(y_hat) > 0.0 else float("nan")
    return rmse, mae, pearson, spearman


def subset_packet(packet: dsp.PacketData, idx: np.ndarray) -> dsp.PacketData:
    return replace(
        packet,
        frame=packet.frame.iloc[idx].reset_index(drop=True),
        y=packet.y[idx],
        w=packet.w[idx],
    )


def fit_model(packet: dsp.PacketData, variant_key: str, args: argparse.Namespace) -> dsp.FittedDSPModel:
    original_linear_reg = dsp.LINEAR_REG
    dsp.LINEAR_REG = LINEAR_REG
    try:
        model, _trace = dsp.fit_variant(
            packet,
            dsp.VARIANTS[variant_key],
            maxiter=int(args.maxiter),
            coarse_top_k=int(args.coarse_top_k),
            basin_hopping_iters=int(args.basin_hopping_iters),
        )
        return model
    finally:
        dsp.LINEAR_REG = original_linear_reg


def fit_linear_head(packet: dsp.PacketData, model: dsp.FittedDSPModel, idx: np.ndarray) -> dsp.FittedDSPModel:
    original_linear_reg = dsp.LINEAR_REG
    dsp.LINEAR_REG = LINEAR_REG
    try:
        return dsp.fit_linear_head(packet.w[idx], packet.y[idx], packet, model.variant, model.params)
    finally:
        dsp.LINEAR_REG = original_linear_reg


def phase_summary(model: dsp.FittedDSPModel) -> str:
    keys = [
        "gamma",
        "gamma_bonus",
        "beta_late",
        "late_satiation_scale",
        "lambda_retention",
        "eta",
        "phi",
    ]
    parts = []
    for key in keys:
        if key in model.params:
            parts.append(f"{key}={float(model.params[key]):.6g}")
    return ", ".join(parts) if parts else "none"


def oof_predictions(packet: dsp.PacketData, model: dsp.FittedDSPModel) -> np.ndarray:
    oof, _folds = dsp_compare.fit_dsp_oof_predictions(packet, model)
    return np.asarray(oof, dtype=float)


def lower_tail_metrics(y: np.ndarray, y_hat: np.ndarray) -> tuple[float, float]:
    tail_count = max(5, int(np.ceil(dsp.LOWER_TAIL_FRAC * len(y))))
    tail_idx = np.argsort(y_hat)[:tail_count]
    residual = np.asarray(y_hat, dtype=float) - np.asarray(y, dtype=float)
    optimism = float(np.mean(np.maximum(y[tail_idx] - y_hat[tail_idx], 0.0)))
    low_tail_rmse = float(np.sqrt(np.mean(residual[tail_idx] * residual[tail_idx])))
    return optimism, low_tail_rmse


def repaired_predictions(model: dsp.FittedDSPModel, heldout: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in heldout.itertuples(index=False):
        weights = eta_diag.repair_weights(model, Path(row.mixture_path))
        pred = float(dsp.predict(model, weights)[0])
        actual = float(row.uncheatable_bpb)
        rows.append(
            {
                "model_name": model.variant.name,
                "mixture": str(row.mixture),
                "is_uncheatable_objective": bool(row.is_uncheatable_objective),
                "actual_uncheatable_bpb": actual,
                "predicted_uncheatable_bpb": pred,
                "prediction_error": pred - actual,
                "optimism": actual - pred,
                "absolute_error": abs(pred - actual),
            }
        )
    return pd.DataFrame(rows)


def leave_good_out(packet: dsp.PacketData, variant_key: str, args: argparse.Namespace) -> tuple[float, float, pd.DataFrame]:
    holdout_count = max(5, int(np.ceil(float(args.good_frac) * len(packet.y))))
    test_idx = np.argsort(packet.y)[:holdout_count]
    train_idx = np.setdiff1d(np.arange(len(packet.y)), test_idx)
    train_packet = subset_packet(packet, train_idx)
    model = fit_model(train_packet, variant_key, args)
    pred = dsp.predict(model, packet.w[test_idx])
    y = packet.y[test_idx]
    rmse, _mae, _pearson, _spearman = regression_metrics(y, pred)
    signed_optimism = float(np.mean(y - pred))
    frame = pd.DataFrame(
        {
            "variant_key": variant_key,
            "run_name": packet.frame.iloc[test_idx][packet.name_col].to_numpy(),
            "actual": y,
            "predicted": pred,
            "prediction_error": pred - y,
            "is_leave_good_holdout": True,
        }
    )
    return rmse, signed_optimism, frame


def raw_frontier(model: dsp.FittedDSPModel, packet: dsp.PacketData, variant_key: str) -> tuple[float | None, float | None, float | None]:
    if variant_key not in {"no_phase", "effective_exposure", "satiating_late_bonus"}:
        return None, None, None
    result, weights = dsp.optimize_raw(
        model,
        num_starts=8,
        observed_start_weights=packet.w,
        max_observed_starts=16,
        observed_jitter_scale=0.01,
    )
    distances = dsp.average_phase_tv_distance(packet.w, weights[None, :, :])
    nearest_idx = int(np.argmin(distances))
    return float(result.fun), float(distances[nearest_idx]), float(packet.y[nearest_idx])


def gate_row(summary: dict[str, Any], heldout_predictions: pd.DataFrame) -> bool:
    uncheatable = heldout_predictions[heldout_predictions["is_uncheatable_objective"]].copy()
    targeted = uncheatable.loc[uncheatable["mixture"].str.endswith("exposure_targeted")].iloc[0]
    all_deficits = uncheatable.loc[uncheatable["mixture"].str.endswith("exposure_all_deficits")].iloc[0]
    predicted_gap = float(targeted.predicted_uncheatable_bpb - all_deficits.predicted_uncheatable_bpb)
    actual_gap = float(targeted.actual_uncheatable_bpb - all_deficits.actual_uncheatable_bpb)
    raw_ok = summary["raw_predicted_optimum"] is None or float(summary["raw_predicted_optimum"]) >= 0.935
    return bool(
        summary["oof_spearman"] >= 0.70
        and summary["heldout_uncheatable_mae"] <= 0.030
        and summary["heldout_max_optimism"] <= 0.035
        and (np.sign(predicted_gap) == np.sign(actual_gap) or abs(predicted_gap - actual_gap) <= 0.004)
        and raw_ok
        and summary["leave_good_rmse"] <= 0.015
        and summary["leave_good_signed_optimism"] <= 0.005
    )


def summarize_variant(
    packet: dsp.PacketData,
    heldout: pd.DataFrame,
    variant_key: str,
    args: argparse.Namespace,
) -> tuple[VariantRow, pd.DataFrame, pd.DataFrame]:
    print(f"Fitting {variant_key}", flush=True)
    model = fit_model(packet, variant_key, args)
    train_pred = dsp.predict(model, packet.w)
    train_rmse, _train_mae, _train_pearson, train_spearman = regression_metrics(packet.y, train_pred)
    oof = oof_predictions(packet, model)
    oof_rmse, _oof_mae, _oof_pearson, oof_spearman = regression_metrics(packet.y, oof)
    oof_optimism, oof_low_tail_rmse = lower_tail_metrics(packet.y, oof)
    repaired = repaired_predictions(model, heldout)
    uncheatable = repaired[repaired["is_uncheatable_objective"]].copy()
    predicted_best = str(uncheatable.loc[uncheatable["predicted_uncheatable_bpb"].idxmin(), "mixture"])
    actual_best = str(uncheatable.loc[uncheatable["actual_uncheatable_bpb"].idxmin(), "mixture"])
    targeted = uncheatable.loc[uncheatable["mixture"].str.endswith("exposure_targeted")].iloc[0]
    all_deficits = uncheatable.loc[uncheatable["mixture"].str.endswith("exposure_all_deficits")].iloc[0]
    predicted_gap = float(targeted.predicted_uncheatable_bpb - all_deficits.predicted_uncheatable_bpb)
    actual_gap = float(targeted.actual_uncheatable_bpb - all_deficits.actual_uncheatable_bpb)
    leave_rmse, leave_optimism, leave_frame = leave_good_out(packet, variant_key, args)
    raw_pred, raw_tv, raw_nearest = raw_frontier(model, packet, variant_key)
    row_dict = {
        "model_name": model.variant.name,
        "variant_key": variant_key,
        "phase_params": phase_summary(model),
        "train_rmse": float(train_rmse),
        "train_spearman": float(train_spearman),
        "oof_rmse": float(oof_rmse),
        "oof_spearman": float(oof_spearman),
        "oof_lower_tail_optimism": float(oof_optimism),
        "oof_low_tail_rmse": float(oof_low_tail_rmse),
        "heldout_all_mae": float(repaired["absolute_error"].mean()),
        "heldout_uncheatable_mae": float(uncheatable["absolute_error"].mean()),
        "heldout_max_optimism": float(uncheatable["optimism"].max()),
        "heldout_gap_error": predicted_gap - actual_gap,
        "heldout_best_order_correct": predicted_best == actual_best,
        "leave_good_rmse": leave_rmse,
        "leave_good_signed_optimism": leave_optimism,
        "raw_predicted_optimum": raw_pred,
        "raw_nearest_observed_tv": raw_tv,
        "raw_nearest_observed_value": raw_nearest,
    }
    row_dict["pass_gates"] = gate_row(row_dict, repaired)
    repaired.insert(0, "variant_key", variant_key)
    return VariantRow(**row_dict), repaired, leave_frame


def beta_scale_sweep(
    packet: dsp.PacketData,
    heldout: pd.DataFrame,
    model: dsp.FittedDSPModel,
) -> pd.DataFrame:
    if model.variant.phase_mode != dsp.PhaseMode.SATIATING_LATE_BONUS:
        return pd.DataFrame()
    base_scale = float(model.params["late_satiation_scale"])
    rows: list[dict[str, Any]] = []
    for beta in [0.0, 0.25, 0.5, 0.75, 1.0 - 1e-4]:
        for scale_mult in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
            params = dict(model.params)
            params["beta_late"] = beta
            params["late_satiation_scale"] = base_scale * scale_mult
            sweep_model = dsp.fit_linear_head(packet.w, packet.y, packet, model.variant, params)
            oof = oof_predictions(packet, sweep_model)
            oof_rmse, _mae, _pearson, oof_spearman = regression_metrics(packet.y, oof)
            oof_optimism, _low_rmse = lower_tail_metrics(packet.y, oof)
            repaired = repaired_predictions(sweep_model, heldout)
            uncheatable = repaired[repaired["is_uncheatable_objective"]].copy()
            targeted = uncheatable.loc[uncheatable["mixture"].str.endswith("exposure_targeted")].iloc[0]
            all_deficits = uncheatable.loc[uncheatable["mixture"].str.endswith("exposure_all_deficits")].iloc[0]
            rows.append(
                {
                    "beta_late": beta,
                    "scale_mult": scale_mult,
                    "late_satiation_scale": params["late_satiation_scale"],
                    "oof_rmse": float(oof_rmse),
                    "oof_spearman": float(oof_spearman),
                    "oof_lower_tail_optimism": float(oof_optimism),
                    "selection_score": float(oof_rmse + 0.5 * oof_optimism),
                    "heldout_uncheatable_mae": float(uncheatable["absolute_error"].mean()),
                    "heldout_max_optimism": float(uncheatable["optimism"].max()),
                    "predicted_gap": float(targeted.predicted_uncheatable_bpb - all_deficits.predicted_uncheatable_bpb),
                    "actual_gap": float(targeted.actual_uncheatable_bpb - all_deficits.actual_uncheatable_bpb),
                }
            )
    return pd.DataFrame(rows)


def write_plots(output_dir: Path, summary: pd.DataFrame, heldout: pd.DataFrame, sweep: pd.DataFrame) -> None:
    fig = px.scatter(
        summary,
        x="oof_spearman",
        y="heldout_uncheatable_mae",
        color="variant_key",
        symbol="pass_gates",
        hover_data=["phase_params", "leave_good_rmse", "raw_predicted_optimum", "heldout_best_order_correct"],
        title="Satiating late-exposure DSP diagnostic",
        template="plotly_white",
    )
    fig.add_hline(y=0.030, line_dash="dot", line_color="gray", annotation_text="heldout MAE gate")
    fig.add_vline(x=0.70, line_dash="dot", line_color="gray", annotation_text="OOF Spearman gate")
    fig.update_layout(width=1100, height=700)
    fig.write_html(output_dir / "satiating_variant_gate_scatter.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    fig2 = px.bar(
        heldout,
        x="mixture",
        y="prediction_error",
        color="variant_key",
        barmode="group",
        title="Repaired heldout prediction error",
        template="plotly_white",
    )
    fig2.update_layout(width=1300, height=650, xaxis_tickangle=-20)
    fig2.write_html(output_dir / "satiating_heldout_errors.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    if not sweep.empty:
        fig3 = px.scatter(
            sweep,
            x="selection_score",
            y="heldout_uncheatable_mae",
            color="beta_late",
            size="scale_mult",
            hover_data=["oof_spearman", "heldout_max_optimism", "predicted_gap", "actual_gap"],
            title="Satiating late bonus beta/scale tension sweep",
            template="plotly_white",
            color_continuous_scale="RdYlGn_r",
        )
        fig3.add_hline(y=0.030, line_dash="dot", line_color="gray")
        fig3.update_layout(width=1100, height=700)
        fig3.write_html(output_dir / "satiating_beta_scale_tension_sweep.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(output_dir: Path, summary: pd.DataFrame, sweep: pd.DataFrame) -> None:
    lines = [
        "# Satiating late-exposure DSP diagnostic",
        "",
        "This tests whether phase-1 exposure should receive a concave refresh bonus instead of a constant effective-exposure multiplier.",
        "",
        "## Variant gates",
        "",
        summary[
            [
                "variant_key",
                "phase_params",
                "oof_spearman",
                "heldout_uncheatable_mae",
                "heldout_max_optimism",
                "heldout_gap_error",
                "leave_good_rmse",
                "leave_good_signed_optimism",
                "raw_predicted_optimum",
                "pass_gates",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
    ]
    if not sweep.empty:
        best = sweep.sort_values("selection_score").head(10)
        lines.extend(
            [
                "",
                "## Best beta/scale sweep points",
                "",
                best.to_markdown(index=False, floatfmt=".6f"),
            ]
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packet, _panel, _domains, _natural, _token_counts, _target_budget = eta_diag.load_packet()
    heldout = eta_diag.load_heldout(args)
    variant_keys = ["no_phase", "effective_exposure", "dsre_satiety", "satiating_late_bonus"]

    rows: list[VariantRow] = []
    heldout_frames: list[pd.DataFrame] = []
    leave_good_frames: list[pd.DataFrame] = []
    fitted_models: dict[str, dsp.FittedDSPModel] = {}
    for variant_key in variant_keys:
        row, repaired, leave_good = summarize_variant(packet, heldout, variant_key, args)
        rows.append(row)
        heldout_frames.append(repaired)
        leave_good_frames.append(leave_good)
        fitted_models[variant_key] = fit_model(packet, variant_key, args)

    summary = pd.DataFrame([asdict(row) for row in rows])
    heldout_predictions = pd.concat(heldout_frames, ignore_index=True)
    leave_good = pd.concat(leave_good_frames, ignore_index=True)
    sweep = beta_scale_sweep(packet, heldout, fitted_models["satiating_late_bonus"])

    summary.to_csv(args.output_dir / "satiating_variant_summary.csv", index=False)
    heldout_predictions.to_csv(args.output_dir / "satiating_heldout_predictions.csv", index=False)
    leave_good.to_csv(args.output_dir / "satiating_leave_good_out_predictions.csv", index=False)
    sweep.to_csv(args.output_dir / "satiating_beta_scale_sweep.csv", index=False)
    write_plots(args.output_dir, summary, heldout_predictions, sweep)
    write_report(args.output_dir, summary, sweep)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "target": TARGET,
                "linear_reg": LINEAR_REG,
                "variant_keys": variant_keys,
                "fit_rows": int(len(packet.y)),
                "repair_results": str(args.repair_results),
                "repair_mixture_dir": str(args.repair_mixture_dir),
            },
            indent=2,
        )
        + "\n"
    )
    print(summary.sort_values(["pass_gates", "heldout_uncheatable_mae"], ascending=[False, True]).to_string(index=False))


if __name__ == "__main__":
    main()
