# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Sweep Huber linear-head fits for canonical DSP on the 300M panel."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import basinhopping, minimize, nnls

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_dsp_vs_olmix_deletion_augmented_300m as dsp_compare,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as olmix,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "dsp_huber_sweep_deletion_augmented_300m_20260625"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class HuberFitResult:
    """A fitted Huber DSP model and fitting trace."""

    model: dsp.FittedDSPModel
    irls_iterations: int
    max_weight_change: float


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--huber-deltas",
        default="0.0025,0.005,0.0075,0.01,0.015,0.02,0.03,0.05",
        help="Comma-separated Huber transition points in BPB units.",
    )
    parser.add_argument("--linear-reg", type=float, default=1e-4)
    parser.add_argument("--maxiter", type=int, default=dsp.FIT_MAXITER)
    parser.add_argument("--coarse-top-k", type=int, default=dsp.START_TOP_K)
    parser.add_argument("--basin-hopping-iters", type=int, default=1)
    parser.add_argument("--irls-maxiter", type=int, default=40)
    parser.add_argument(
        "--retune-nonlinear",
        action="store_true",
        help="Retune DSP nonlinear parameters under Huber loss. Slow; default reuses the L2 nonlinear fit.",
    )
    return parser.parse_args()


def huber_loss(residual: np.ndarray, delta: float) -> np.ndarray:
    """Elementwise Huber loss."""
    abs_residual = np.abs(residual)
    quadratic = abs_residual <= delta
    return np.where(quadratic, 0.5 * residual * residual, delta * (abs_residual - 0.5 * delta))


def build_design(
    weights: np.ndarray,
    packet: dsp.PacketData,
    variant: dsp.DSPVariant,
    params: dict[str, float | np.ndarray],
) -> np.ndarray:
    """Build the linear-head design matrix for fixed DSP nonlinear parameters."""
    signal, penalty = dsp.features(weights, packet.c0, packet.c1, variant, params)
    if variant.penalty_mode == dsp.PenaltyMode.NONE:
        return -signal
    return np.hstack([-signal, penalty])


def weighted_nnls_linear_head(
    design: np.ndarray,
    targets: np.ndarray,
    observation_weights: np.ndarray,
    linear_reg: float,
) -> tuple[float, np.ndarray]:
    """Solve a weighted, intercept-centered NNLS linear head."""
    weights = np.asarray(observation_weights, dtype=float)
    if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        raise ValueError("Observation weights must be nonnegative with positive total mass")
    weights = weights / float(np.mean(weights))
    weight_sum = float(np.sum(weights))
    design_mean = np.sum(design * weights[:, None], axis=0, keepdims=True) / weight_sum
    target_mean = float(np.sum(targets * weights) / weight_sum)
    centered_design = design - design_mean
    centered_targets = targets - target_mean
    root_weights = np.sqrt(weights)
    weighted_design = centered_design * root_weights[:, None]
    weighted_targets = centered_targets * root_weights
    if linear_reg > 0.0:
        weighted_design = np.vstack([weighted_design, np.sqrt(linear_reg) * np.eye(weighted_design.shape[1])])
        weighted_targets = np.concatenate([weighted_targets, np.zeros(weighted_design.shape[1], dtype=float)])
    coef, _ = nnls(weighted_design, weighted_targets)
    intercept = float(target_mean - (design_mean @ coef).item())
    return intercept, np.asarray(coef, dtype=float)


def fit_huber_linear_head(
    weights: np.ndarray,
    targets: np.ndarray,
    packet: dsp.PacketData,
    variant: dsp.DSPVariant,
    params: dict[str, float | np.ndarray],
    *,
    delta: float,
    linear_reg: float,
    irls_maxiter: int,
) -> HuberFitResult:
    """Fit a nonnegative Huber linear head by IRLS for fixed nonlinear parameters."""
    design = build_design(weights, packet, variant, params)
    observation_weights = np.ones(len(targets), dtype=float)
    last_observation_weights = observation_weights.copy()
    intercept = float(np.mean(targets))
    coef = np.zeros(design.shape[1], dtype=float)
    max_weight_change = float("inf")
    for iteration in range(1, irls_maxiter + 1):
        intercept, coef = weighted_nnls_linear_head(design, targets, observation_weights, linear_reg)
        residual = intercept + design @ coef - targets
        abs_residual = np.maximum(np.abs(residual), 1e-12)
        observation_weights = np.minimum(1.0, delta / abs_residual)
        max_weight_change = float(np.max(np.abs(observation_weights - last_observation_weights)))
        if max_weight_change < 1e-7:
            break
        last_observation_weights = observation_weights.copy()

    benefit_coef = np.asarray(coef[: packet.m], dtype=float)
    if variant.penalty_mode == dsp.PenaltyMode.NONE:
        penalty_coef = np.zeros(packet.m, dtype=float)
    else:
        penalty_coef = np.asarray(coef[packet.m :], dtype=float)
    model = dsp.FittedDSPModel(
        variant=variant,
        params=params,
        intercept=intercept,
        benefit_coef=benefit_coef,
        penalty_coef=penalty_coef,
        domain_names=list(packet.domain_names),
        c0=np.asarray(packet.c0, dtype=float),
        c1=np.asarray(packet.c1, dtype=float),
    )
    return HuberFitResult(model=model, irls_iterations=iteration, max_weight_change=max_weight_change)


def profile_objective_huber(
    packet: dsp.PacketData,
    variant: dsp.DSPVariant,
    theta: np.ndarray,
    *,
    delta: float,
    linear_reg: float,
    irls_maxiter: int,
) -> float:
    """Huber-profile DSP tuning objective."""
    params = dsp.unpack_theta(theta, variant, packet.m)
    fit = fit_huber_linear_head(
        packet.w,
        packet.y,
        packet,
        variant,
        params,
        delta=delta,
        linear_reg=linear_reg,
        irls_maxiter=irls_maxiter,
    )
    pred = dsp.predict(fit.model, packet.w)
    residual = pred - packet.y
    robust_scale = float(np.sqrt(2.0 * np.mean(huber_loss(residual, delta))))
    tail_count = max(5, int(np.ceil(dsp.LOWER_TAIL_FRAC * len(packet.y))))
    tail_idx = np.argsort(pred)[:tail_count]
    optimism = float(np.mean(np.maximum(packet.y[tail_idx] - pred[tail_idx], 0.0)))
    return robust_scale + 0.5 * optimism


def fit_variant_huber(
    packet: dsp.PacketData,
    variant: dsp.DSPVariant,
    *,
    delta: float,
    linear_reg: float,
    maxiter: int,
    coarse_top_k: int,
    basin_hopping_iters: int,
    irls_maxiter: int,
) -> tuple[HuberFitResult, pd.DataFrame]:
    """Tune nonlinear parameters and fit final Huber DSP model."""
    starts = dsp.start_bank(packet, variant)
    coord_bounds = dsp.bounds(variant, packet.m)

    def objective(theta: np.ndarray) -> float:
        return profile_objective_huber(
            packet,
            variant,
            np.asarray(theta, dtype=float),
            delta=delta,
            linear_reg=linear_reg,
            irls_maxiter=irls_maxiter,
        )

    coarse_rows = [
        {"stage": "coarse", "start_id": start_id, "objective": objective(start)} for start_id, start in enumerate(starts)
    ]
    ranked = sorted(coarse_rows, key=lambda row: float(row["objective"]))
    rows = list(coarse_rows)
    best_objective = float("inf")
    best_theta: np.ndarray | None = None

    for rank, row in enumerate(ranked[:coarse_top_k]):
        start = starts[int(row["start_id"])]
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=coord_bounds,
            options={"maxiter": maxiter, "ftol": 1e-7, "maxls": 20},
        )
        rows.append(
            {
                "stage": "refine",
                "chosen_rank": rank,
                "start_id": int(row["start_id"]),
                "objective": float(result.fun),
                "success": bool(result.success),
                "message": str(result.message),
            }
        )
        if float(result.fun) < best_objective:
            best_objective = float(result.fun)
            best_theta = np.asarray(result.x, dtype=float)

    if best_theta is None:
        raise RuntimeError(f"No Huber fit result for {variant.name} delta={delta}")

    if basin_hopping_iters > 0:
        hop_result = basinhopping(
            objective,
            best_theta,
            niter=basin_hopping_iters,
            stepsize=0.15,
            minimizer_kwargs={
                "method": "L-BFGS-B",
                "bounds": coord_bounds,
                "options": {"maxiter": max(8, maxiter // 4), "ftol": 1e-7},
            },
            seed=dsp.CV_SEED,
        )
        rows.append(
            {
                "stage": "basin_hopping_diagnostic",
                "chosen_rank": -1,
                "start_id": -1,
                "objective": float(hop_result.fun),
                "success": bool(hop_result.lowest_optimization_result.success),
                "message": str(hop_result.message),
            }
        )
        if float(hop_result.fun) < best_objective:
            best_theta = np.asarray(hop_result.x, dtype=float)

    params = dsp.unpack_theta(best_theta, variant, packet.m)
    fit = fit_huber_linear_head(
        packet.w,
        packet.y,
        packet,
        variant,
        params,
        delta=delta,
        linear_reg=linear_reg,
        irls_maxiter=irls_maxiter,
    )
    return fit, pd.DataFrame.from_records(rows)


def load_l2_nonlinear_params() -> dict[str, float | np.ndarray]:
    """Load canonical DSP nonlinear parameters from the existing L2 comparison fit."""
    path = SCRIPT_DIR / "reference_outputs" / "dsp_vs_olmix_deletion_augmented_300m_20260625" / "dsp_model.json"
    with path.open() as f:
        payload = json.load(f)
    params = payload["params"]
    return {
        "rho": np.asarray(params["rho"], dtype=float),
        "tau": np.asarray(params["tau"], dtype=float),
        "gamma": float(params["gamma"]),
    }


def oof_huber_predictions(
    packet: dsp.PacketData,
    model: dsp.FittedDSPModel,
    *,
    delta: float,
    linear_reg: float,
    irls_maxiter: int,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Compute OOF predictions with fixed nonlinear params and Huber fold heads."""
    folds = olmix.kfold_indices(len(packet.y), n_splits=olmix.N_SPLITS, seed=olmix.CV_SEED)
    oof = np.zeros_like(packet.y, dtype=float)
    for train_idx, test_idx in folds:
        fold_fit = fit_huber_linear_head(
            packet.w[train_idx],
            packet.y[train_idx],
            packet,
            model.variant,
            model.params,
            delta=delta,
            linear_reg=linear_reg,
            irls_maxiter=irls_maxiter,
        )
        oof[test_idx] = dsp.predict(fold_fit.model, packet.w[test_idx])
    return oof, folds


def selection_score(row: pd.Series) -> float:
    return float(row["oof_rmse"] + 0.5 * row["lower_tail_optimism"])


def write_sweep_plot(output_dir: Path, summary: pd.DataFrame, best_oof_frame: pd.DataFrame) -> None:
    """Write sweep diagnostics plots."""
    x = summary["huber_delta"].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=summary["baseline_l2_oof_rmse"], mode="lines", name="L2 OOF RMSE"))
    fig.add_trace(go.Scatter(x=x, y=summary["oof_rmse"], mode="lines+markers", name="Huber OOF RMSE"))
    fig.add_trace(go.Scatter(x=x, y=summary["low_tail_rmse"], mode="lines+markers", name="Huber low-tail RMSE"))
    fig.add_trace(
        go.Scatter(x=x, y=summary["lower_tail_optimism"], mode="lines+markers", name="Huber lower-tail optimism")
    )
    fig.add_trace(go.Scatter(x=x, y=summary["fold_mean_regret_at_1"], mode="lines+markers", name="Huber fold regret@1"))
    fig.update_layout(
        title="DSP Huber linear-head sweep",
        xaxis_title="Huber delta (BPB)",
        yaxis_title="BPB-scale diagnostic",
        template="plotly_white",
        width=1050,
        height=720,
    )
    fig.write_html(output_dir / "dsp_huber_fit_sweep_diagnostics.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x, y=summary["baseline_l2_oof_spearman"], mode="lines", name="L2 OOF Spearman"))
    fig2.add_trace(go.Scatter(x=x, y=summary["oof_spearman"], mode="lines+markers", name="Huber OOF Spearman"))
    fig2.add_trace(go.Scatter(x=x, y=summary["train_spearman"], mode="lines+markers", name="Huber train Spearman"))
    fig2.update_layout(
        title="DSP Huber linear-head sweep: rank fit",
        xaxis_title="Huber delta (BPB)",
        yaxis_title="Spearman",
        template="plotly_white",
        width=1050,
        height=650,
    )
    fig2.write_html(output_dir / "dsp_huber_fit_sweep_spearman.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    best = summary.loc[summary["selection_score"].idxmin()]
    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(
            x=best_oof_frame["observed"],
            y=best_oof_frame["oof_predicted"],
            mode="markers",
            marker={
                "size": 8,
                "color": best_oof_frame["residual_abs"],
                "colorscale": "RdYlGn_r",
                "reversescale": True,
            },
            text=best_oof_frame["run_name"],
            hovertemplate="run=%{text}<br>observed=%{x:.4f}<br>oof=%{y:.4f}<extra></extra>",
        )
    )
    min_value = float(min(best_oof_frame["observed"].min(), best_oof_frame["oof_predicted"].min()))
    max_value = float(max(best_oof_frame["observed"].max(), best_oof_frame["oof_predicted"].max()))
    fig3.add_trace(go.Scatter(x=[min_value, max_value], y=[min_value, max_value], mode="lines", name="identity"))
    fig3.update_layout(
        title=f"Best Huber OOF predictions (delta={float(best['huber_delta']):g})",
        xaxis_title="Observed BPB",
        yaxis_title="OOF predicted BPB",
        template="plotly_white",
        width=850,
        height=760,
    )
    fig3.write_html(output_dir / "dsp_huber_best_oof_scatter.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(output_dir: Path, summary: pd.DataFrame, baseline: pd.Series, *, retune_nonlinear: bool) -> None:
    """Write a concise Markdown report."""
    best = summary.loc[summary["selection_score"].idxmin()]
    lines = [
        "# DSP Huber linear-head sweep",
        "",
        "Panel: deletion-augmented 300M Uncheatable BPB panel. Nonlinear form is canonical DSP.",
        "",
        f"Huber fits use `LINEAR_REG={float(best['linear_reg']):g}` and sweep Huber `delta` in BPB units.",
        "The Huber fit replaces the fixed-parameter NNLS least-squares linear head with IRLS-weighted NNLS.",
        (
            "DSP nonlinear parameters were retuned under the Huber profile objective."
            if retune_nonlinear
            else "DSP nonlinear parameters are reused from the existing best L2 canonical fit; only the linear head is refit."
        ),
        "",
        "## Best Huber row",
        "",
        best[
            [
                "huber_delta",
                "train_spearman",
                "train_rmse",
                "oof_spearman",
                "oof_rmse",
                "fold_mean_regret_at_1",
                "lower_tail_optimism",
                "low_tail_rmse",
                "selection_score",
            ]
        ]
        .to_frame()
        .T.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Baseline L2 row",
        "",
        baseline[
            [
                "linear_reg",
                "train_spearman",
                "train_rmse",
                "oof_spearman",
                "oof_rmse",
                "fold_mean_regret_at_1",
                "lower_tail_optimism",
                "low_tail_rmse",
                "selection_score",
            ]
        ]
        .to_frame()
        .T.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Full Huber sweep",
        "",
        summary[
            [
                "huber_delta",
                "train_spearman",
                "train_rmse",
                "oof_spearman",
                "oof_rmse",
                "fold_mean_regret_at_1",
                "lower_tail_optimism",
                "low_tail_rmse",
                "selection_score",
                "mean_irls_iterations",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    deltas = parse_float_list(args.huber_deltas)

    _signal, columns, domains, _natural = olmix.load_raw_signal_panel()
    target_budget = olmix.load_target_budget()
    token_counts = olmix.load_domain_token_counts(domains)
    panel, metadata = olmix.build_uncheatable_panel(columns)
    packet = dsp_compare.build_dsp_packet(panel, columns, domains, token_counts, target_budget)
    variant = dsp.VARIANTS["canonical"]

    baseline_path = (
        SCRIPT_DIR
        / "reference_outputs"
        / "dsp_l2_kl_sweep_deletion_augmented_300m_20260625"
        / "dsp_l2_fit_sweep_summary.csv"
    )
    baseline_frame = pd.read_csv(baseline_path)
    baseline_frame["selection_score"] = baseline_frame["oof_rmse"] + 0.5 * baseline_frame["lower_tail_optimism"]
    baseline = baseline_frame.loc[baseline_frame["selection_score"].idxmin()]

    rows: list[dict[str, Any]] = []
    best_oof_frame: pd.DataFrame | None = None
    for delta in deltas:
        print(f"Fitting Huber DSP delta={delta:g}", flush=True)
        if args.retune_nonlinear:
            fit, trace = fit_variant_huber(
                packet,
                variant,
                delta=float(delta),
                linear_reg=float(args.linear_reg),
                maxiter=int(args.maxiter),
                coarse_top_k=int(args.coarse_top_k),
                basin_hopping_iters=int(args.basin_hopping_iters),
                irls_maxiter=int(args.irls_maxiter),
            )
            trace.to_csv(output_dir / f"dsp_huber_tuning_delta_{delta:g}.csv", index=False)
            nonlinear_source = "retuned_huber"
        else:
            fit = fit_huber_linear_head(
                packet.w,
                packet.y,
                packet,
                variant,
                load_l2_nonlinear_params(),
                delta=float(delta),
                linear_reg=float(args.linear_reg),
                irls_maxiter=int(args.irls_maxiter),
            )
            nonlinear_source = "l2_fixed"
        train_pred = dsp.predict(fit.model, packet.w)
        train_rmse, train_mae, train_pearson, train_spearman = dsp_compare.regression_metrics(packet.y, train_pred)
        oof_pred, folds = oof_huber_predictions(
            packet,
            fit.model,
            delta=float(delta),
            linear_reg=float(args.linear_reg),
            irls_maxiter=int(args.irls_maxiter),
        )
        oof_metrics = olmix.predictive_diagnostics(packet.y, oof_pred, folds)
        residual = oof_pred - packet.y
        row = {
            "huber_delta": float(delta),
            "linear_reg": float(args.linear_reg),
            "train_rmse": float(train_rmse),
            "train_mae": float(train_mae),
            "train_pearson": float(train_pearson),
            "train_spearman": float(train_spearman),
            "oof_rmse": float(oof_metrics["rmse"]),
            "oof_mae": float(oof_metrics["mae"]),
            "oof_pearson": float(oof_metrics["pearson"]),
            "oof_spearman": float(oof_metrics["spearman"]),
            "fold_mean_regret_at_1": float(oof_metrics["fold_mean_regret_at_1"]),
            "lower_tail_optimism": float(oof_metrics["lower_tail_optimism"]),
            "low_tail_rmse": float(oof_metrics["low_tail_rmse"]),
            "selection_score": float(oof_metrics["rmse"] + 0.5 * oof_metrics["lower_tail_optimism"]),
            "full_fit_irls_iterations": int(fit.irls_iterations),
            "full_fit_max_weight_change": float(fit.max_weight_change),
            "mean_irls_iterations": float(fit.irls_iterations),
            "baseline_l2_oof_rmse": float(baseline["oof_rmse"]),
            "baseline_l2_oof_spearman": float(baseline["oof_spearman"]),
            "baseline_l2_selection_score": float(baseline["selection_score"]),
            "nonlinear_source": nonlinear_source,
        }
        rows.append(row)
        oof_frame = pd.DataFrame(
            {
                "run_name": panel["run_name"].astype(str),
                "observed": packet.y,
                "oof_predicted": oof_pred,
                "residual": residual,
                "residual_abs": np.abs(residual),
                "huber_delta": float(delta),
            }
        )
        oof_frame.to_csv(output_dir / f"dsp_huber_oof_predictions_delta_{delta:g}.csv", index=False)
        if best_oof_frame is None or row["selection_score"] < float(best_oof_frame.attrs["selection_score"]):
            oof_frame.attrs["selection_score"] = row["selection_score"]
            best_oof_frame = oof_frame

    summary = pd.DataFrame(rows).sort_values("huber_delta").reset_index(drop=True)
    if best_oof_frame is None:
        raise RuntimeError("No Huber OOF predictions were generated")
    write_sweep_plot(output_dir, summary, best_oof_frame)

    summary.to_csv(output_dir / "dsp_huber_fit_sweep_summary.csv", index=False)
    write_report(output_dir, summary, baseline, retune_nonlinear=bool(args.retune_nonlinear))
    with (output_dir / "metadata.json").open("w") as f:
        json.dump(
            {
                "target_metric": olmix.UNCHEATABLE_TARGET,
                "panel_rows": int(len(panel)),
                "phase_fractions": olmix.PHASE_FRACTIONS.tolist(),
                "huber_deltas": deltas,
                "linear_reg": float(args.linear_reg),
                "retune_nonlinear": bool(args.retune_nonlinear),
                "nonlinear_source": "retuned_huber" if args.retune_nonlinear else "l2_fixed",
                "n_proportional_reference_rows": int(metadata["n_proportional_reference_rows"]),
                "proportional_reference_mean": metadata["proportional_reference_mean"],
                "proportional_reference_std": metadata["proportional_reference_std"],
                "baseline_l2_summary_path": str(baseline_path),
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(summary.to_string(index=False))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
