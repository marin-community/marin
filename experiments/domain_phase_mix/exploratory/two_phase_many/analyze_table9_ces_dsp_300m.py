# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scipy", "scikit-learn"]
# ///
"""CES benefit-aggregation diagnostic for Table-9 DSP.

The core DSP head is additive across domain signals. This diagnostic tests a
minimal CES alternative without introducing free per-domain CES weights: fit an
additive DSP model on each train fold, use its nonnegative benefit coefficients
as CES weights, and refit only a scalar CES utility coefficient plus the usual
additive penalty coefficients. This keeps the test local and avoids turning CES
into a new high-dimensional optimizer.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import nnls

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_table9_phase_split_dsp_300m as phase_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_olmo_base_easy_per_component_dsp_decision_300m as component_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_paper_faithful_olmix_300m as paper_olmix,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_top_level_dsp_300m as top_level_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "table9_dsp_ces_diagnostic_20260702"
MACRO_TARGET = "table9_macro_bpb"
LINEAR_REG = 1e-4
LOWER_TAIL_FRAC = 0.15
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class CESHead:
    """Fitted CES utility head for fixed DSP nonlinear parameters."""

    rho_ces: float
    ces_weights: np.ndarray
    intercept: float
    utility_coef: float
    penalty_coef: np.ndarray


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-variants", default="effective_exposure,retained_effective_exposure")
    parser.add_argument("--linear-reg-values", default="0.0001,0.001,0.01")
    parser.add_argument("--rho-ces-values", default="-2,-1,-0.5,0,0.5,1,2,4")
    parser.add_argument("--maxiter", type=int, default=12)
    parser.add_argument("--coarse-top-k", type=int, default=2)
    return parser.parse_args()


def normalized_ces_weights(coef: np.ndarray) -> np.ndarray:
    weights = np.maximum(np.asarray(coef, dtype=float), 0.0)
    weights = weights + 1e-8
    return weights / weights.sum()


def ces_aggregate(signal: np.ndarray, ces_weights: np.ndarray, rho_ces: float) -> np.ndarray:
    clipped = np.clip(signal, 1e-9, None)
    weights = ces_weights[None, :]
    if abs(rho_ces) < 1e-9:
        return np.exp(np.sum(weights * np.log(clipped), axis=1))
    moment = np.sum(weights * np.power(clipped, rho_ces), axis=1)
    return np.power(np.clip(moment, 1e-300, None), 1.0 / rho_ces)


def fit_ces_head(
    signal: np.ndarray,
    penalty: np.ndarray,
    targets: np.ndarray,
    *,
    ces_weights: np.ndarray,
    rho_ces: float,
) -> CESHead:
    utility = ces_aggregate(signal, ces_weights, rho_ces)
    design = np.hstack([-utility[:, None], penalty])
    design_mean = design.mean(axis=0, keepdims=True)
    target_mean = float(targets.mean())
    centered_design = design - design_mean
    centered_targets = targets - target_mean
    if LINEAR_REG > 0.0:
        centered_design = np.vstack([centered_design, np.sqrt(LINEAR_REG) * np.eye(centered_design.shape[1])])
        centered_targets = np.concatenate([centered_targets, np.zeros(centered_design.shape[1], dtype=float)])
    coef, _ = nnls(centered_design, centered_targets)
    intercept = float(target_mean - (design_mean @ coef).item())
    return CESHead(
        rho_ces=float(rho_ces),
        ces_weights=np.asarray(ces_weights, dtype=float),
        intercept=intercept,
        utility_coef=float(coef[0]),
        penalty_coef=np.asarray(coef[1:], dtype=float),
    )


def predict_ces(model: dsp.FittedDSPModel, head: CESHead, weights: np.ndarray) -> np.ndarray:
    signal, penalty = dsp.features(weights, model.c0, model.c1, model.variant, model.params)
    utility = ces_aggregate(signal, head.ces_weights, head.rho_ces)
    return np.asarray(head.intercept - head.utility_coef * utility + penalty @ head.penalty_coef, dtype=float)


def ces_profile_objective(targets: np.ndarray, pred: np.ndarray) -> float:
    residual = pred - targets
    rmse = float(np.sqrt(np.mean(residual**2)))
    tail_count = max(5, int(np.ceil(LOWER_TAIL_FRAC * len(targets))))
    tail_idx = np.argsort(pred)[:tail_count]
    optimism = float(np.mean(np.maximum(targets[tail_idx] - pred[tail_idx], 0.0)))
    return rmse + 0.5 * optimism


def fit_best_ces_head(
    packet: dsp.PacketData,
    additive_model: dsp.FittedDSPModel,
    rho_grid: list[float],
) -> tuple[CESHead, pd.DataFrame]:
    signal, penalty = dsp.features(packet.w, packet.c0, packet.c1, additive_model.variant, additive_model.params)
    ces_weights = normalized_ces_weights(additive_model.benefit_coef)
    rows = []
    best_head: CESHead | None = None
    best_objective = float("inf")
    for rho_ces in rho_grid:
        head = fit_ces_head(signal, penalty, packet.y, ces_weights=ces_weights, rho_ces=rho_ces)
        pred = predict_ces(additive_model, head, packet.w)
        objective = ces_profile_objective(packet.y, pred)
        rows.append({"rho_ces": rho_ces, "train_objective": objective})
        if objective < best_objective:
            best_objective = objective
            best_head = head
    if best_head is None:
        raise RuntimeError("No CES head was fitted")
    return best_head, pd.DataFrame(rows)


def run_nested_ces(
    packet: dsp.PacketData,
    variant_key: str,
    linear_reg: float,
    folds: list[tuple[np.ndarray, np.ndarray]],
    rho_grid: list[float],
    *,
    maxiter: int,
    coarse_top_k: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    out = np.zeros_like(packet.y, dtype=float)
    rows = []
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        print(f"    CES nested fold {fold_id + 1}/{len(folds)}", flush=True)
        train_packet = phase_dsp.subset_packet(packet, train_idx)
        additive_model, _ = phase_dsp.fit_variant_with_l2(
            train_packet,
            variant_key,
            linear_reg,
            maxiter=maxiter,
            coarse_top_k=coarse_top_k,
            basin_hopping_iters=0,
        )
        head, rho_summary = fit_best_ces_head(train_packet, additive_model, rho_grid)
        out[test_idx] = predict_ces(additive_model, head, packet.w[test_idx])
        rows.append(
            {
                "fold_id": fold_id,
                "rho_ces": head.rho_ces,
                "utility_coef": head.utility_coef,
                "rho_train_objective": float(rho_summary["train_objective"].min()),
            }
        )
    return out, rows


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    signal, columns, domains, _natural = base.load_raw_signal_panel()
    token_counts = base.load_domain_token_counts(domains)
    panel, metadata = paper_olmix.build_fit_panel(columns)
    packet = top_level_dsp.build_dsp_packet(panel, columns, domains, token_counts, MACRO_TARGET)
    folds = component_dsp.panel_stratified_folds(panel, n_splits=phase_dsp.N_SPLITS, seed=phase_dsp.CV_SEED)

    rho_grid = parse_float_list(args.rho_ces_values)
    summary_rows = []
    pred_rows = []
    fold_rows = []
    for variant_key in parse_str_list(args.base_variants):
        for linear_reg in parse_float_list(args.linear_reg_values):
            print(f"Fitting CES {variant_key} linear_reg={linear_reg}", flush=True)
            oof, fold_info = run_nested_ces(
                packet,
                variant_key,
                linear_reg,
                folds,
                rho_grid,
                maxiter=args.maxiter,
                coarse_top_k=args.coarse_top_k,
            )
            rmse, spearman = phase_dsp.regression_metrics(packet.y, oof)
            selected_idx = int(np.argmin(oof))
            observed_best_idx = int(np.argmin(packet.y))
            summary_rows.append(
                {
                    "base_variant_key": variant_key,
                    "linear_reg": linear_reg,
                    "nested_oof_rmse": rmse,
                    "nested_oof_spearman": spearman,
                    "nested_fold_mean_regret_at_1": phase_dsp.fold_mean_regret_at_k(packet.y, oof, folds, 1),
                    "nested_global_regret_at_1": phase_dsp.global_regret_at_k(packet.y, oof, 1),
                    "nested_global_regret_at_3": phase_dsp.global_regret_at_k(packet.y, oof, 3),
                    "nested_lower_tail_optimism": phase_dsp.lower_tail_optimism(packet.y, oof)[0],
                    "nested_low_tail_rmse": phase_dsp.lower_tail_optimism(packet.y, oof)[1],
                    "nested_selected_run_name": str(packet.frame.iloc[selected_idx]["run_name"]),
                    "nested_selected_actual_bpb": float(packet.y[selected_idx]),
                    "nested_selected_actual_rank": int(np.argsort(packet.y).tolist().index(selected_idx) + 1),
                    "best_observed_run_name": str(packet.frame.iloc[observed_best_idx]["run_name"]),
                    "best_observed_bpb": float(packet.y[observed_best_idx]),
                    "rho_ces_nested_mean": float(np.mean([row["rho_ces"] for row in fold_info])),
                    "rho_ces_nested_std": float(np.std([row["rho_ces"] for row in fold_info])),
                }
            )
            for row in fold_info:
                fold_rows.append({"base_variant_key": variant_key, "linear_reg": linear_reg, **row})
            pred_rows.extend(
                {
                    "base_variant_key": variant_key,
                    "linear_reg": linear_reg,
                    "run_name": str(packet.frame.iloc[idx]["run_name"]),
                    "observed_bpb": float(packet.y[idx]),
                    "predicted_bpb": float(oof[idx]),
                }
                for idx in range(len(packet.y))
            )

    summary = pd.DataFrame(summary_rows).sort_values(["nested_fold_mean_regret_at_1", "nested_oof_rmse"])
    predictions = pd.DataFrame(pred_rows)
    fold_summary = pd.DataFrame(fold_rows)
    summary.to_csv(output_dir / "ces_nested_cv_summary.csv", index=False)
    predictions.to_csv(output_dir / "ces_nested_cv_predictions.csv", index=False)
    fold_summary.to_csv(output_dir / "ces_fold_rho_summary.csv", index=False)

    fig = go.Figure()
    for _, row in summary.iterrows():
        subset = predictions[
            predictions["base_variant_key"].eq(row["base_variant_key"])
            & predictions["linear_reg"].eq(row["linear_reg"])
        ]
        fig.add_trace(
            go.Scatter(
                x=subset["observed_bpb"],
                y=subset["predicted_bpb"],
                mode="markers",
                name=f"{row['base_variant_key']} L2={row['linear_reg']}",
                text=subset["run_name"],
                hovertemplate="%{text}<br>observed=%{x:.4f}<br>predicted=%{y:.4f}<extra></extra>",
            )
        )
    lo = float(min(predictions["observed_bpb"].min(), predictions["predicted_bpb"].min()))
    hi = float(max(predictions["observed_bpb"].max(), predictions["predicted_bpb"].max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y=x", line={"dash": "dash"}))
    fig.update_layout(
        title="CES benefit aggregation: nested OOF Table-9 macro BPB",
        xaxis_title="Observed Table-9 macro BPB",
        yaxis_title="Nested OOF predicted BPB",
        width=1050,
        height=720,
    )
    fig.write_html(output_dir / "ces_nested_oof_scatter.html", include_plotlyjs="cdn", config=PLOT_CONFIG)
    print(summary.to_string(index=False))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
