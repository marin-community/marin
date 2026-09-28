# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit the P3 surrogate on the two-phase StarCoder packet.

This mirrors the earlier ``grp_starcoder_u_shape_fit`` diagnostic, but uses the
collaborator P3 form:

    y = b + sum_d beta_d (w0_d + eta w1_d)^a
        - gamma0 sum_d (c0_d w0_d)^p
        - gamma1 sum_d (c1_d w1_d)^p

The target is BPB, so lower is better. Linear coefficients are fit with ridge;
``eta``, ``a``, ``p``, and the ridge penalty are selected by 5-fold CV.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.starcoder_grp import (
    load_completed_two_phase_starcoder_packet,
    subset_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    STARCODER_TARGET,
    PacketData,
    regression_metrics,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PREFIX = SCRIPT_DIR / "p3_starcoder_fit"

ETA_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0)
A_GRID = tuple(np.linspace(0.5, 2.0, 8))
P_GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
ALPHA_GRID = tuple(np.logspace(-4, 4, 17))
CV_SPLITS = 5
CV_SEED = 0
GRID_SIZE = 1001
EPS = 1e-4


@dataclass(frozen=True)
class P3Params:
    """P3 nonlinear and ridge parameters."""

    eta: float
    a: float
    p: float
    ridge_alpha: float


@dataclass
class P3Model:
    """Fitted P3 linear head with stored standardization."""

    params: P3Params
    active_mask: np.ndarray
    design_mean: np.ndarray
    design_std: np.ndarray
    coef: np.ndarray
    intercept: float

    def predict(self, weights: np.ndarray, c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
        """Predict BPB for two-phase weights."""
        design = build_p3_design(weights, c0, c1, self.params)
        active = design[:, self.active_mask]
        standardized = (active - self.design_mean) / self.design_std
        return np.asarray(self.intercept + standardized @ self.coef, dtype=float)


def build_p3_design(weights: np.ndarray, c0: np.ndarray, c1: np.ndarray, params: P3Params) -> np.ndarray:
    """Build P3 design features for two-phase weights."""
    w0 = weights[:, 0, :]
    w1 = weights[:, 1, :]
    combined_exposure = np.maximum(w0 + params.eta * w1, EPS)
    signal = np.power(combined_exposure, params.a)
    phase0_epochs = np.maximum(w0 * c0[None, :], EPS)
    phase1_epochs = np.maximum(w1 * c1[None, :], EPS)
    penalty0 = np.power(phase0_epochs, params.p).sum(axis=1, keepdims=True)
    penalty1 = np.power(phase1_epochs, params.p).sum(axis=1, keepdims=True)
    return np.column_stack([signal, -penalty0, -penalty1]).astype(float)


def fit_p3_head(weights: np.ndarray, y: np.ndarray, c0: np.ndarray, c1: np.ndarray, params: P3Params) -> P3Model:
    """Fit the ridge linear head for fixed P3 nonlinear parameters."""
    design = build_p3_design(weights, c0, c1, params)
    active_mask = design.std(axis=0) > 1e-12
    if not active_mask.any():
        raise ValueError("P3 design has no active features")

    active = design[:, active_mask]
    design_mean = active.mean(axis=0)
    design_std = active.std(axis=0)
    standardized = (active - design_mean) / design_std
    centered_y = y - y.mean()
    eye = np.eye(standardized.shape[1])
    coef = np.linalg.solve(
        standardized.T @ standardized + params.ridge_alpha * eye,
        standardized.T @ centered_y,
    )
    intercept = float(y.mean())
    return P3Model(
        params=params,
        active_mask=active_mask,
        design_mean=design_mean,
        design_std=design_std,
        coef=coef,
        intercept=intercept,
    )


def cv_predict_p3(packet: PacketData, params: P3Params) -> tuple[np.ndarray, list[float]]:
    """Return out-of-fold P3 predictions and per-fold regret."""
    kf = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_SEED)
    oof = np.zeros_like(packet.y, dtype=float)
    fold_regrets: list[float] = []
    for train_idx, test_idx in kf.split(packet.w):
        model = fit_p3_head(packet.w[train_idx], packet.y[train_idx], packet.c0, packet.c1, params)
        pred = model.predict(packet.w[test_idx], packet.c0, packet.c1)
        oof[test_idx] = pred
        chosen = int(np.argmin(pred))
        fold_regrets.append(float(packet.y[test_idx][chosen] - np.min(packet.y[test_idx])))
    return oof, fold_regrets


def metrics_from_prediction(packet: PacketData, pred: np.ndarray) -> dict[str, Any]:
    """Return scalar fit metrics for a prediction vector."""
    out = regression_metrics(packet.frame, packet.name_col, packet.y, pred)
    out["pearson"] = float(np.corrcoef(packet.y, pred)[0, 1])
    return out


def select_p3_params(packet: PacketData) -> tuple[P3Params, pd.DataFrame, np.ndarray, list[float]]:
    """Grid-search P3 parameters by 5-fold CV RMSE."""
    rows: list[dict[str, float]] = []
    best_params: P3Params | None = None
    best_pred: np.ndarray | None = None
    best_fold_regrets: list[float] | None = None
    best_rmse = float("inf")

    for eta in ETA_GRID:
        for a in A_GRID:
            for p in P_GRID:
                for ridge_alpha in ALPHA_GRID:
                    params = P3Params(eta=float(eta), a=float(a), p=float(p), ridge_alpha=float(ridge_alpha))
                    pred, fold_regrets = cv_predict_p3(packet, params)
                    residual = pred - packet.y
                    rmse = float(np.sqrt(np.mean(residual**2)))
                    r2 = float(1.0 - np.sum(residual**2) / np.sum((packet.y - packet.y.mean()) ** 2))
                    spearman = float(spearmanr(packet.y, pred).statistic)
                    foldmean_regret = float(np.mean(fold_regrets))
                    rows.append(
                        {
                            "eta": params.eta,
                            "a": params.a,
                            "p": params.p,
                            "ridge_alpha": params.ridge_alpha,
                            "cv_rmse": rmse,
                            "cv_r2": r2,
                            "cv_spearman": spearman,
                            "cv_foldmean_regret_at_1": foldmean_regret,
                        }
                    )
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = params
                        best_pred = pred
                        best_fold_regrets = fold_regrets

    if best_params is None or best_pred is None or best_fold_regrets is None:
        raise RuntimeError("P3 parameter search did not produce a candidate")
    return best_params, pd.DataFrame(rows), best_pred, best_fold_regrets


def build_u_slice_weights(starcoder_weights: np.ndarray) -> np.ndarray:
    """Return two-phase weights on the phase-0 Nemotron-only slice."""
    weights = np.zeros((len(starcoder_weights), 2, 2), dtype=float)
    weights[:, 0, 0] = 1.0
    weights[:, 1, 0] = 1.0 - starcoder_weights
    weights[:, 1, 1] = starcoder_weights
    return weights


def p3_parameter_count() -> dict[str, int]:
    """Return parameter-count variants for P3."""
    return {
        "linear_coefficients": 4,
        "intercept": 1,
        "shape_parameters": 3,
        "ridge_hyperparameter": 1,
        "total_without_ridge_hyperparameter": 8,
        "total_with_ridge_hyperparameter": 9,
    }


def write_prediction_plot(
    packet: PacketData,
    train_pred: np.ndarray,
    cv_pred: np.ndarray,
    output_html: Path,
    output_png: Path,
) -> None:
    """Write predicted-vs-actual diagnostics."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Full-data fit", "5-fold OOF prediction"))
    for col, pred, title in [(1, train_pred, "full"), (2, cv_pred, "oof")]:
        fig.add_trace(
            go.Scatter(
                x=packet.y,
                y=pred,
                text=packet.frame[packet.name_col],
                mode="markers",
                marker={
                    "size": 8,
                    "color": packet.frame["phase_1_starcoder"],
                    "colorscale": "RdYlGn_r",
                    "showscale": col == 2,
                    "colorbar": {"title": "phase-1<br>StarCoder"},
                    "line": {"width": 0.4, "color": "white"},
                },
                name=title,
                showlegend=False,
            ),
            row=1,
            col=col,
        )
    lo = float(min(packet.y.min(), train_pred.min(), cv_pred.min()))
    hi = float(max(packet.y.max(), train_pred.max(), cv_pred.max()))
    parity_line = go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line={"color": "black", "dash": "dash"})
    fig.add_trace(parity_line, row=1, col=1)
    fig.add_trace(parity_line, row=1, col=2)
    fig.update_xaxes(title_text="Actual BPB")
    fig.update_yaxes(title_text="Predicted BPB")
    fig.update_layout(title="P3 on two-phase StarCoder data", width=1300, height=560, template="plotly_white")
    fig.write_html(output_html)
    fig.write_image(output_png, scale=2)


def write_u_shape_plot(
    subset_packet_data: PacketData,
    subset_model: P3Model,
    full_model: P3Model,
    output_html: Path,
    output_png: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Write U-shaped slice diagnostics and return summary data."""
    x_obs = subset_packet_data.frame["phase_1_starcoder"].to_numpy(dtype=float)
    order = np.argsort(x_obs)
    sorted_subset = subset_packet(subset_packet_data, order)
    x_obs = sorted_subset.frame["phase_1_starcoder"].to_numpy(dtype=float)
    y_obs = sorted_subset.y

    x_grid = np.linspace(0.0, 1.0, GRID_SIZE, dtype=float)
    w_grid = build_u_slice_weights(x_grid)
    subset_curve = subset_model.predict(w_grid, subset_packet_data.c0, subset_packet_data.c1)
    full_curve = full_model.predict(w_grid, subset_packet_data.c0, subset_packet_data.c1)
    subset_pred = subset_model.predict(sorted_subset.w, subset_packet_data.c0, subset_packet_data.c1)
    full_pred = full_model.predict(sorted_subset.w, subset_packet_data.c0, subset_packet_data.c1)

    output_frame = sorted_subset.frame.copy()
    output_frame["p3_subset_fit_prediction"] = subset_pred
    output_frame["p3_all_data_fit_prediction"] = full_pred

    starcoder_phase1_epochs = float(subset_packet_data.c1[1])
    summary = {
        "subset_fit_metrics_on_subset": metrics_from_prediction(sorted_subset, subset_pred),
        "all_data_fit_metrics_on_subset": metrics_from_prediction(sorted_subset, full_pred),
        "observed_subset_min": {
            "phase_1_starcoder": float(x_obs[np.argmin(y_obs)]),
            "bpb": float(np.min(y_obs)),
        },
        "subset_fit_slice_min": {
            "phase_1_starcoder": float(x_grid[np.argmin(subset_curve)]),
            "bpb": float(np.min(subset_curve)),
            "epochs": float(x_grid[np.argmin(subset_curve)] * starcoder_phase1_epochs),
        },
        "all_data_fit_slice_min": {
            "phase_1_starcoder": float(x_grid[np.argmin(full_curve)]),
            "bpb": float(np.min(full_curve)),
            "epochs": float(x_grid[np.argmin(full_curve)] * starcoder_phase1_epochs),
        },
    }

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_obs,
            y=y_obs,
            mode="markers+lines",
            marker={"size": 8, "color": "#111827"},
            line={"color": "#9ca3af", "dash": "dash"},
            name="Observed U-slice",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=subset_curve,
            mode="lines",
            line={"width": 3, "color": "#1b7837"},
            name="P3 fit on U-slice only",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=full_curve,
            mode="lines",
            line={"width": 3, "color": "#b2182b"},
            name="P3 fit on all StarCoder runs",
        )
    )
    fig.update_layout(
        title="P3 StarCoder U-shaped slice diagnostic",
        xaxis_title="Phase-1 StarCoder weight",
        yaxis_title=STARCODER_TARGET,
        width=950,
        height=580,
        template="plotly_white",
    )
    fig.write_html(output_html)
    fig.write_image(output_png, scale=2)
    return summary, output_frame


def main() -> None:
    """Fit P3 and write diagnostics."""
    packet = load_completed_two_phase_starcoder_packet(target=STARCODER_TARGET)

    best_params, grid, cv_pred, cv_fold_regrets = select_p3_params(packet)
    full_model = fit_p3_head(packet.w, packet.y, packet.c0, packet.c1, best_params)
    train_pred = full_model.predict(packet.w, packet.c0, packet.c1)

    grid.to_csv(OUTPUT_PREFIX.with_name(f"{OUTPUT_PREFIX.name}_grid.csv"), index=False)
    pd.DataFrame(
        {
            packet.name_col: packet.frame[packet.name_col],
            "actual": packet.y,
            "p3_train_prediction": train_pred,
            "p3_oof_prediction": cv_pred,
            "phase_0_starcoder": packet.frame["phase_0_starcoder"],
            "phase_1_starcoder": packet.frame["phase_1_starcoder"],
        }
    ).to_csv(OUTPUT_PREFIX.with_name(f"{OUTPUT_PREFIX.name}_predictions.csv"), index=False)

    slice_mask = packet.frame["phase_0_nemotron_full"].round(4).eq(1.0).to_numpy(dtype=bool)
    subset_packet_data = subset_packet(packet, slice_mask)
    subset_params, _subset_grid, _subset_cv_pred, _subset_fold_regrets = select_p3_params(subset_packet_data)
    subset_model = fit_p3_head(
        subset_packet_data.w,
        subset_packet_data.y,
        subset_packet_data.c0,
        subset_packet_data.c1,
        subset_params,
    )

    u_summary, u_frame = write_u_shape_plot(
        subset_packet_data,
        subset_model,
        full_model,
        OUTPUT_PREFIX.with_name(f"{OUTPUT_PREFIX.name}_u_shape.html"),
        OUTPUT_PREFIX.with_name(f"{OUTPUT_PREFIX.name}_u_shape.png"),
    )
    u_frame.to_csv(OUTPUT_PREFIX.with_name(f"{OUTPUT_PREFIX.name}_u_shape_predictions.csv"), index=False)

    write_prediction_plot(
        packet,
        train_pred,
        cv_pred,
        OUTPUT_PREFIX.with_name(f"{OUTPUT_PREFIX.name}_predicted_vs_actual.html"),
        OUTPUT_PREFIX.with_name(f"{OUTPUT_PREFIX.name}_predicted_vs_actual.png"),
    )

    train_metrics = metrics_from_prediction(packet, train_pred)
    cv_metrics = metrics_from_prediction(packet, cv_pred)
    cv_metrics["cv_foldmean_regret_at_1"] = float(np.mean(cv_fold_regrets))
    cv_metrics["cv_foldmedian_regret_at_1"] = float(np.median(cv_fold_regrets))
    cv_metrics["cv_foldmax_regret_at_1"] = float(np.max(cv_fold_regrets))

    summary = {
        "target": STARCODER_TARGET,
        "dataset": "two_phase_starcoder",
        "model": "P3",
        "n_runs": len(packet.y),
        "n_subset_runs": len(subset_packet_data.y),
        "selected_params": best_params.__dict__,
        "subset_selected_params": subset_params.__dict__,
        "n_params": p3_parameter_count(),
        "train": train_metrics,
        "cv": cv_metrics,
        "u_shape": u_summary,
        "feature_names": [
            "signal_nemotron_full",
            "signal_starcoder",
            "negative_phase0_concentration",
            "negative_phase1_concentration",
        ],
        "linear_head": {
            "active_mask": full_model.active_mask.tolist(),
            "coef_standardized": full_model.coef.tolist(),
            "intercept": full_model.intercept,
        },
    }
    OUTPUT_PREFIX.with_name(f"{OUTPUT_PREFIX.name}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
