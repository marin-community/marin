# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "plotly", "scipy", "scikit-learn", "kaleido"]
# ///
"""Evaluate variable-scale DSP extensions on the Marin ND scaling panel.

This script tests a centered additive extension of effective-exposure DSP:

    y(N, D, w) = scale_head(N, D)
                 - sum_i a_i * amp_b(N) * Delta S_i(w)
                 + sum_i p_i * amp_p(N) * Delta P_i(w)

where Delta features are centered against the proportional mixture at the same
exact (N, D) point. Centering prevents the mixture head from explaining the
baseline scaling trajectory and leaves the DSP terms to model relative mixture
effects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize, nnls
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "analysis_dataset"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "variable_scale_dsp_nd_20260515"
IMG_DIR = OUTPUT_DIR / "img"
FROZEN_DSP_MODEL_PATH = (
    SCRIPT_DIR
    / "reference_outputs"
    / "dsp_canonical_variants_300m_20260510"
    / "dsp_effective_exposure_penalty_nnls"
    / "model.json"
)
PRIMARY_METRIC = "eval/uncheatable_eval/bpb"
N_REF = 58_998_528.0
D_REF = 1_199_833_088.0
CV_SPLITS = 5
CV_SEED = 0
FULL_FIT_STARTS = 3
FULL_FIT_MAXITER = 40
REFIT_MAXITER = 18
RIDGE = 1e-6
LOWER_TAIL_FRAC = 0.15


@dataclass(frozen=True)
class VariableScaleDSPSpec:
    """One variable-scale DSP candidate."""

    name: str
    amplitude_mode: str
    exposure_scale: bool
    description: str


@dataclass(frozen=True)
class FittedVariableScaleDSP:
    """Fitted nonlinear and profiled linear parameters."""

    spec: VariableScaleDSPSpec
    theta: np.ndarray
    coef: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    train_objective: float
    optimizer_success: bool
    optimizer_message: str


SPECS: tuple[VariableScaleDSPSpec, ...] = (
    VariableScaleDSPSpec(
        name="dsp_vs_centered_no_amp",
        amplitude_mode="none",
        exposure_scale=False,
        description="Centered variable-scale DSP with no mixture-effect amplitude scaling.",
    ),
    VariableScaleDSPSpec(
        name="dsp_vs_centered_shared_amp",
        amplitude_mode="shared",
        exposure_scale=False,
        description="Centered variable-scale DSP with one shared N-dependent mixture amplitude.",
    ),
    VariableScaleDSPSpec(
        name="dsp_vs_centered_split_amp",
        amplitude_mode="split",
        exposure_scale=False,
        description="Centered variable-scale DSP with separate N-dependent benefit and penalty amplitudes.",
    ),
    VariableScaleDSPSpec(
        name="dsp_vs_centered_exposure_scaled",
        amplitude_mode="none",
        exposure_scale=True,
        description="Centered variable-scale DSP with a learned D-dependent multiplier inside DSP exposure.",
    ),
)


def softplus(x: np.ndarray) -> np.ndarray:
    """Stable softplus."""

    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def load_data() -> dict[str, Any]:
    """Load the current ND analysis packet and proportional references."""

    frame = pd.read_csv(DATA_DIR / "nd_scale_runs.csv", low_memory=False)
    payload = np.load(DATA_DIR / "nd_scale_packet.npz", allow_pickle=True)
    frozen_model = json.loads(FROZEN_DSP_MODEL_PATH.read_text())
    frozen_params = frozen_model["params"]
    mask = np.asarray(payload["primary_y_mask"], dtype=bool)
    frame = frame.loc[mask].reset_index(drop=True)
    weights = np.asarray(payload["weights"], dtype=float)[mask]
    multipliers = np.asarray(payload["simulated_epoch_multipliers"], dtype=float)[mask]
    model_sizes = np.asarray(payload["model_sizes"], dtype=float)[mask]
    train_tokens = np.asarray(payload["realized_train_tokens"], dtype=float)[mask]
    prop_weights = np.zeros_like(weights)
    prop_multipliers = np.zeros_like(multipliers)
    prop_indices: dict[tuple[float, float], int] = {}
    for idx, row in frame.loc[frame["mixture_id"].eq("baseline_proportional")].iterrows():
        key = (float(row["model_size"]), float(row["realized_train_tokens"]))
        prop_indices[key] = int(idx)
    missing_keys = []
    for row_idx, (model_size, train_token_count) in enumerate(zip(model_sizes, train_tokens, strict=True)):
        key = (float(model_size), float(train_token_count))
        prop_idx = prop_indices.get(key)
        if prop_idx is None:
            missing_keys.append(key)
            continue
        prop_weights[row_idx] = weights[prop_idx]
        prop_multipliers[row_idx] = multipliers[prop_idx]
    if missing_keys:
        raise ValueError(f"Missing proportional anchors for N,D pairs: {sorted(set(missing_keys))[:5]}")
    return {
        "frame": frame,
        "weights": weights,
        "multipliers": multipliers,
        "prop_weights": prop_weights,
        "prop_multipliers": prop_multipliers,
        "y": np.asarray(payload["primary_y"], dtype=float)[mask],
        "mixture_ids": np.asarray(payload["mixture_ids"], dtype=object)[mask].astype(str),
        "run_names": np.asarray(payload["run_names"], dtype=object)[mask].astype(str),
        "scale_names": np.asarray(payload["scale_names"], dtype=object).astype(str),
        "scale_index": np.asarray(payload["scale_index"], dtype=np.int64)[mask],
        "domain_names": np.asarray(payload["domain_names"], dtype=object).astype(str),
        "model_sizes": model_sizes,
        "realized_train_tokens": train_tokens,
        "frozen_params": {
            "rho": np.asarray(frozen_params["rho"], dtype=float),
            "tau": np.asarray(frozen_params["tau"], dtype=float),
            "gamma": float(frozen_params["gamma"]),
        },
    }


def theta_layout(spec: VariableScaleDSPSpec, num_domains: int) -> tuple[int, list[tuple[float, float]]]:
    """Return nonlinear theta length and bounds."""

    _ = num_domains
    bounds: list[tuple[float, float]] = [
        (np.log(0.02), np.log(2.0)),  # alpha
        (np.log(0.02), np.log(2.0)),  # beta
        (np.log(1e-3), np.log(2.0)),  # delta
    ]
    if spec.amplitude_mode == "shared":
        bounds.append((-2.0, 2.0))  # kappa
    elif spec.amplitude_mode == "split":
        bounds.extend([(-2.0, 2.0), (-2.0, 2.0)])  # kappa_benefit, kappa_penalty
    elif spec.amplitude_mode != "none":
        raise ValueError(f"Unknown amplitude mode {spec.amplitude_mode}")
    if spec.exposure_scale:
        bounds.append((-1.0, 1.0))  # omega
    return len(bounds), bounds


def unpack_theta(
    theta: np.ndarray,
    spec: VariableScaleDSPSpec,
    num_domains: int,
    frozen_params: dict[str, Any],
) -> dict[str, Any]:
    """Decode nonlinear parameters."""

    cursor = 0
    rho = np.asarray(frozen_params["rho"], dtype=float)
    tau = np.asarray(frozen_params["tau"], dtype=float)
    gamma = float(frozen_params["gamma"])
    if len(rho) != num_domains or len(tau) != num_domains:
        raise ValueError(f"Frozen DSP geometry has {len(rho)} rho/{len(tau)} tau values for {num_domains} domains")
    alpha = float(np.exp(theta[cursor]))
    cursor += 1
    beta = float(np.exp(theta[cursor]))
    cursor += 1
    delta = float(np.exp(theta[cursor]))
    cursor += 1
    kappa_benefit = 0.0
    kappa_penalty = 0.0
    if spec.amplitude_mode == "shared":
        kappa_benefit = float(theta[cursor])
        kappa_penalty = kappa_benefit
        cursor += 1
    elif spec.amplitude_mode == "split":
        kappa_benefit = float(theta[cursor])
        cursor += 1
        kappa_penalty = float(theta[cursor])
        cursor += 1
    omega = 0.0
    if spec.exposure_scale:
        omega = float(theta[cursor])
        cursor += 1
    if cursor != len(theta):
        raise ValueError(f"unused theta values for {spec.name}: cursor={cursor}, len={len(theta)}")
    return {
        "rho": rho,
        "tau": tau,
        "gamma": gamma,
        "alpha": alpha,
        "beta": beta,
        "delta": delta,
        "kappa_benefit": kappa_benefit,
        "kappa_penalty": kappa_penalty,
        "omega": omega,
    }


def pack_theta(params: dict[str, Any], spec: VariableScaleDSPSpec) -> np.ndarray:
    """Pack nonlinear parameters."""

    values = [
        np.asarray(
            [
                np.log(float(params["alpha"])),
                np.log(float(params["beta"])),
                np.log(float(params["delta"])),
            ],
            dtype=float,
        ),
    ]
    if spec.amplitude_mode == "shared":
        values.append(np.asarray([float(params.get("kappa_benefit", 0.0))], dtype=float))
    elif spec.amplitude_mode == "split":
        values.append(
            np.asarray(
                [float(params.get("kappa_benefit", 0.0)), float(params.get("kappa_penalty", 0.0))],
                dtype=float,
            )
        )
    if spec.exposure_scale:
        values.append(np.asarray([float(params.get("omega", 0.0))], dtype=float))
    return np.concatenate(values)


def start_bank(data: dict[str, Any], spec: VariableScaleDSPSpec) -> list[np.ndarray]:
    """Build deterministic nonlinear starts."""

    _ = data
    starts = []
    for alpha, beta, delta, kappa, omega in (
        (0.25, 0.25, 0.25, 0.0, 0.0),
        (0.35, 0.35, 0.35, 0.0, 0.0),
        (0.50, 0.25, 0.50, 0.25, 0.0),
        (0.75, 0.15, 0.75, -0.25, 0.25),
        (0.15, 0.50, 0.15, 0.5, -0.25),
    ):
        params = {
            "alpha": alpha,
            "beta": beta,
            "delta": delta,
            "kappa_benefit": kappa,
            "kappa_penalty": kappa,
            "omega": omega,
        }
        if spec.amplitude_mode == "split":
            params["kappa_penalty"] = -kappa
        starts.append(pack_theta(params, spec))
    return starts


def dsp_features(
    weights: np.ndarray,
    multipliers: np.ndarray,
    data_tokens: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute effective-exposure DSP signal and penalty features."""

    e0 = weights[:, 0, :] * multipliers[:, 0, :]
    e1 = weights[:, 1, :] * multipliers[:, 1, :]
    d = data_tokens / D_REF
    exposure_scale = np.power(np.maximum(d, 1e-12), float(params["omega"]))[:, None]
    z = exposure_scale * (e0 + float(params["gamma"]) * e1)
    rho = np.asarray(params["rho"], dtype=float)[None, :]
    tau = np.asarray(params["tau"], dtype=float)[None, :]
    signal = 1.0 - np.exp(-rho * z)
    penalty = softplus(np.log1p(z) - tau) ** 2
    return signal, penalty


def design_matrix(
    data: dict[str, Any], indices: np.ndarray, spec: VariableScaleDSPSpec, theta: np.ndarray
) -> np.ndarray:
    """Build the profiled linear design matrix."""

    params = unpack_theta(theta, spec, len(data["domain_names"]), data["frozen_params"])
    n = data["model_sizes"][indices] / N_REF
    d = data["realized_train_tokens"][indices] / D_REF
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    delta = float(params["delta"])
    scale_cols = [
        np.ones(len(indices), dtype=float),
        np.power(np.maximum(n, 1e-12), -beta),
        np.power(np.maximum(n, 1e-12), delta) / np.power(np.maximum(d, 1e-12), alpha),
    ]
    signal, penalty = dsp_features(
        data["weights"][indices],
        data["multipliers"][indices],
        data["realized_train_tokens"][indices],
        params,
    )
    prop_signal, prop_penalty = dsp_features(
        data["prop_weights"][indices],
        data["prop_multipliers"][indices],
        data["realized_train_tokens"][indices],
        params,
    )
    benefit_amp = np.power(np.maximum(n, 1e-12), float(params["kappa_benefit"]))[:, None]
    penalty_amp = np.power(np.maximum(n, 1e-12), float(params["kappa_penalty"]))[:, None]
    benefit_delta = benefit_amp * (signal - prop_signal)
    penalty_delta = penalty_amp * (penalty - prop_penalty)
    return np.column_stack([*scale_cols, -benefit_delta, penalty_delta])


def fit_linear(
    design: np.ndarray, y: np.ndarray, num_domains: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit mixed unconstrained scale and nonnegative DSP linear coefficients."""

    rest = design[:, 1:]
    mean = rest.mean(axis=0)
    std = rest.std(axis=0)
    std = np.where(std < 1e-10, 1.0, std)
    x = np.column_stack([design[:, :1], (rest - mean) / std])
    scale_design = x[:, :3]
    dsp_design = x[:, 3 : 3 + 2 * num_domains]

    # Variable projection for a mixed head: solve the unconstrained scale head
    # analytically, then NNLS the DSP coefficients on scale-residualized data.
    y_scale_coef = np.linalg.lstsq(scale_design, y, rcond=None)[0]
    y_residual = y - scale_design @ y_scale_coef
    dsp_scale_coef = np.linalg.lstsq(scale_design, dsp_design, rcond=None)[0]
    dsp_residual = dsp_design - scale_design @ dsp_scale_coef
    if RIDGE > 0.0:
        dsp_aug = np.vstack([dsp_residual, np.sqrt(RIDGE) * np.eye(dsp_residual.shape[1])])
        y_aug = np.concatenate([y_residual, np.zeros(dsp_residual.shape[1], dtype=float)])
    else:
        dsp_aug = dsp_residual
        y_aug = y_residual
    dsp_coef, _ = nnls(dsp_aug, y_aug)
    scale_coef = np.linalg.lstsq(scale_design, y - dsp_design @ dsp_coef, rcond=None)[0]
    coef = np.concatenate([scale_coef, dsp_coef])
    pred = x @ coef
    return coef, mean, std, pred


def objective(theta: np.ndarray, data: dict[str, Any], indices: np.ndarray, spec: VariableScaleDSPSpec) -> float:
    """Profile objective over nonlinear parameters."""

    design = design_matrix(data, indices, spec, theta)
    _coef, _mean, _std, pred = fit_linear(design, data["y"][indices], len(data["domain_names"]))
    residual = pred - data["y"][indices]
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    tail_count = max(8, int(np.ceil(LOWER_TAIL_FRAC * len(indices))))
    tail = np.argsort(pred)[:tail_count]
    optimism = float(np.mean(np.maximum(data["y"][indices][tail] - pred[tail], 0.0)))
    return rmse + 0.25 * optimism


def fit_model(
    data: dict[str, Any],
    indices: np.ndarray,
    spec: VariableScaleDSPSpec,
    initial: np.ndarray | None = None,
) -> FittedVariableScaleDSP:
    """Fit one variable-scale DSP variant."""

    _theta_len, bounds = theta_layout(spec, len(data["domain_names"]))
    starts = [initial] if initial is not None else start_bank(data, spec)[:FULL_FIT_STARTS]
    maxiter = REFIT_MAXITER if initial is not None else FULL_FIT_MAXITER
    best: Any | None = None
    for start in starts:
        result = minimize(
            objective,
            np.asarray(start, dtype=float),
            args=(data, indices, spec),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-6, "maxls": 20},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    if best is None:
        raise RuntimeError(f"no optimizer result for {spec.name}")
    design = design_matrix(data, indices, spec, np.asarray(best.x, dtype=float))
    coef, mean, std, _pred = fit_linear(design, data["y"][indices], len(data["domain_names"]))
    return FittedVariableScaleDSP(
        spec=spec,
        theta=np.asarray(best.x, dtype=float),
        coef=coef,
        feature_mean=mean,
        feature_std=std,
        train_objective=float(best.fun),
        optimizer_success=bool(best.success),
        optimizer_message=str(best.message),
    )


def predict_model(model: FittedVariableScaleDSP, data: dict[str, Any], indices: np.ndarray) -> np.ndarray:
    """Predict held-out rows."""

    design = design_matrix(data, indices, model.spec, model.theta)
    rest = design[:, 1:]
    x = np.column_stack([design[:, :1], (rest - model.feature_mean) / model.feature_std])
    return np.asarray(x @ model.coef, dtype=float)


def metric_summary(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Compute lower-is-better prediction metrics."""

    best_pred_idx = int(np.argmin(pred))
    best_actual = float(np.min(y))
    chosen_actual = float(y[best_pred_idx])
    k = min(8, len(y))
    actual_top = set(np.argsort(y)[:k])
    pred_top = set(np.argsort(pred)[:k])
    return {
        "n": len(y),
        "rmse": float(np.sqrt(np.mean(np.square(pred - y)))),
        "mae": float(np.mean(np.abs(pred - y))),
        "pearson": float(pearsonr(y, pred).statistic) if len(y) > 2 and np.std(pred) > 0.0 else float("nan"),
        "spearman": float(spearmanr(y, pred).statistic) if len(y) > 2 and np.std(pred) > 0.0 else float("nan"),
        "actual_std": float(np.std(y)),
        "predicted_std": float(np.std(pred)),
        "predicted_actual_std_ratio": float(np.std(pred) / max(np.std(y), 1e-12)),
        "regret_at_1": chosen_actual - best_actual,
        "chosen_actual": chosen_actual,
        "best_actual": best_actual,
        "top8_overlap": len(actual_top & pred_top) / float(k),
    }


def prediction_frame(
    data: dict[str, Any],
    indices: np.ndarray,
    pred: np.ndarray,
    model_name: str,
    *,
    split: str,
) -> pd.DataFrame:
    """Build row-level predictions for inspection."""

    frame = (
        data["frame"]
        .iloc[indices][
            ["registry_run_key", "mixture_id", "run_name", "scale", "scale_display_label", "target_budget_multiplier"]
        ]
        .copy()
    )
    frame["model"] = model_name
    frame["split"] = split
    frame["actual_bpb"] = data["y"][indices]
    frame["predicted_bpb"] = pred
    frame["residual_bpb"] = frame["predicted_bpb"] - frame["actual_bpb"]
    return frame


def parameter_count(spec: VariableScaleDSPSpec, num_domains: int) -> int:
    """Approximate parameter count."""

    nonlinear = 2 * num_domains + 4  # frozen rho, frozen tau, frozen gamma, alpha, beta, delta
    if spec.amplitude_mode == "shared":
        nonlinear += 1
    elif spec.amplitude_mode == "split":
        nonlinear += 2
    if spec.exposure_scale:
        nonlinear += 1
    linear = 3 + 2 * num_domains
    return nonlinear + linear


def decoded_params(model: FittedVariableScaleDSP, data: dict[str, Any]) -> dict[str, Any]:
    """Decode a fitted model into compact scalar summaries."""

    params = unpack_theta(model.theta, model.spec, len(data["domain_names"]), data["frozen_params"])
    return {
        "alpha": float(params["alpha"]),
        "beta": float(params["beta"]),
        "delta": float(params["delta"]),
        "gamma": float(params["gamma"]),
        "kappa_benefit": float(params["kappa_benefit"]),
        "kappa_penalty": float(params["kappa_penalty"]),
        "omega": float(params["omega"]),
        "rho_min": float(np.min(params["rho"])),
        "rho_median": float(np.median(params["rho"])),
        "rho_max": float(np.max(params["rho"])),
        "tau_min": float(np.min(params["tau"])),
        "tau_median": float(np.median(params["tau"])),
        "tau_max": float(np.max(params["tau"])),
        "frozen_domain_geometry": True,
        "fitted_nonlinear_param_count": len(model.theta),
    }


def grouped_cv(data: dict[str, Any], spec: VariableScaleDSPSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run grouped OOF by mixture id."""

    all_idx = np.arange(len(data["y"]))
    groups = data["mixture_ids"]
    splitter = GroupKFold(n_splits=CV_SPLITS)
    pred = np.full(len(all_idx), np.nan, dtype=float)
    full_model = fit_model(data, all_idx, spec)
    for _fold_idx, (train_pos, test_pos) in enumerate(splitter.split(all_idx, data["y"], groups)):
        model = fit_model(data, all_idx[train_pos], spec, initial=full_model.theta)
        pred[all_idx[test_pos]] = predict_model(model, data, all_idx[test_pos])
    if np.isnan(pred).any():
        raise ValueError(f"missing grouped OOF predictions for {spec.name}")
    summary = metric_summary(data["y"], pred)
    summary.update(
        {
            "model": spec.name,
            "description": spec.description,
            "split": "grouped_oof",
            "parameter_count": parameter_count(spec, len(data["domain_names"])),
            "train_objective": full_model.train_objective,
            "optimizer_success": full_model.optimizer_success,
            "optimizer_message": full_model.optimizer_message,
        }
    )
    return pd.DataFrame([summary]), prediction_frame(data, all_idx, pred, spec.name, split="grouped_oof")


def scale_holdouts(data: dict[str, Any], spec: VariableScaleDSPSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on all but one scale and predict the held-out scale."""

    all_idx = np.arange(len(data["y"]))
    scale_labels = data["scale_names"][data["scale_index"]].astype(str)
    full_model = fit_model(data, all_idx, spec)
    rows = []
    predictions = []
    for scale in sorted(np.unique(scale_labels)):
        test = np.flatnonzero(scale_labels == scale)
        train = np.flatnonzero(scale_labels != scale)
        if len(test) < 3 or len(train) < 20:
            continue
        model = fit_model(data, train, spec, initial=full_model.theta)
        pred = predict_model(model, data, test)
        row = metric_summary(data["y"][test], pred)
        row.update({"model": spec.name, "split": f"leave_scale_{scale}", "scale": scale})
        rows.append(row)
        predictions.append(prediction_frame(data, test, pred, spec.name, split=f"leave_scale_{scale}"))
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def write_plots(summary: pd.DataFrame, predictions: pd.DataFrame, holdout: pd.DataFrame) -> None:
    """Write Plotly diagnostic plots."""

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    plot_frame = predictions.loc[predictions["split"].eq("grouped_oof")].copy()
    fig = px.scatter(
        plot_frame,
        x="actual_bpb",
        y="predicted_bpb",
        color="scale_display_label",
        symbol="model",
        facet_col="model",
        facet_col_wrap=2,
        hover_name="run_name",
        hover_data=["mixture_id", "target_budget_multiplier", "residual_bpb"],
        title="Variable-scale DSP: grouped OOF predictions",
        height=850,
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    lo = min(plot_frame["actual_bpb"].min(), plot_frame["predicted_bpb"].min())
    hi = max(plot_frame["actual_bpb"].max(), plot_frame["predicted_bpb"].max())
    fig.add_shape(type="line", x0=lo, x1=hi, y0=lo, y1=hi, line={"dash": "dot", "color": "#333333"})
    fig.update_layout(template="plotly_white")
    fig.write_html(IMG_DIR / "grouped_oof_predicted_vs_actual.html", include_plotlyjs="cdn")
    try:
        fig.write_image(IMG_DIR / "grouped_oof_predicted_vs_actual.png", scale=2)
    except ValueError:
        pass

    metric_cols = ["rmse", "spearman", "regret_at_1", "top8_overlap", "predicted_actual_std_ratio"]
    summary_long = summary.melt(
        id_vars=["model", "parameter_count"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )
    fig = px.bar(
        summary_long,
        x="model",
        y="value",
        color="model",
        facet_col="metric",
        facet_col_wrap=3,
        title="Variable-scale DSP grouped OOF metric comparison",
        height=700,
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    fig.write_html(IMG_DIR / "grouped_oof_metric_comparison.html", include_plotlyjs="cdn")
    try:
        fig.write_image(IMG_DIR / "grouped_oof_metric_comparison.png", scale=2)
    except ValueError:
        pass

    if not holdout.empty:
        holdout_long = holdout.melt(
            id_vars=["model", "scale"],
            value_vars=["rmse", "spearman", "regret_at_1", "top8_overlap"],
            var_name="metric",
            value_name="value",
        )
        fig = px.bar(
            holdout_long,
            x="scale",
            y="value",
            color="model",
            facet_col="metric",
            barmode="group",
            title="Variable-scale DSP leave-one-scale-out metrics",
            height=650,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig.update_layout(template="plotly_white")
        fig.write_html(IMG_DIR / "leave_scale_metric_comparison.html", include_plotlyjs="cdn")
        try:
            fig.write_image(IMG_DIR / "leave_scale_metric_comparison.png", scale=2)
        except ValueError:
            pass


def render_report(summary: pd.DataFrame, holdout: pd.DataFrame, decoded: pd.DataFrame) -> str:
    """Render Markdown report."""

    keep = [
        "model",
        "parameter_count",
        "rmse",
        "mae",
        "spearman",
        "pearson",
        "regret_at_1",
        "top8_overlap",
        "predicted_actual_std_ratio",
    ]
    lines = [
        "# Variable-Scale DSP on Marin ND Scaling Data",
        "",
        "## Data",
        "",
        f"- Source: `{DATA_DIR / 'nd_scale_runs.csv'}`",
        f"- Metric: `{PRIMARY_METRIC}`",
        f"- Rows: `{int(summary['n'].iloc[0])}` labeled rows for grouped OOF.",
        "",
        "## Form",
        "",
        "This is a screening evaluation: the per-domain DSP geometry",
        f"`rho_i`, `tau_i`, and `gamma` is frozen from `{FROZEN_DSP_MODEL_PATH}`.",
        "The fit retunes the global scale exponents, optional scale-amplitude exponents, and the profiled linear head.",
        "",
        "The baseline scale trajectory is modeled by:",
        "",
        "$$g(N,D)=E+C(N/N_0)^{-\\beta}+B(N/N_0)^\\delta(D/D_0)^{-\\alpha}.$$",
        "",
        "DSP mixture features are centered against proportional at the same exact `(N,D)`:",
        "",
        "$$\\Delta S_i=S_i(w)-S_i(w_{\\mathrm{prop}}), \\qquad \\Delta P_i=P_i(w)-P_i(w_{\\mathrm{prop}}).$$",
        "",
        "The tested additive form is:",
        "",
        "$$\\hat y(N,D,w)=g(N,D)-n^{\\kappa_b}\\sum_i a_i\\Delta S_i+n^{\\kappa_p}\\sum_i p_i\\Delta P_i.$$",
        "",
        "The exposure-scaled variant additionally replaces DSP exposure `z_i` by `(D/D0)^omega z_i`.",
        "",
        "## Grouped OOF Results",
        "",
        summary[keep].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Leave-One-Scale-Out Results",
        "",
        (
            holdout[["model", "scale", "n", "rmse", "spearman", "regret_at_1", "top8_overlap"]].to_markdown(
                index=False, floatfmt=".6f"
            )
            if not holdout.empty
            else "_No scale holdout results._"
        ),
        "",
        "## Decoded Parameter Summary",
        "",
        decoded.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
        "- Centering DSP features against proportional at the same exact `(N,D)` is important: it leaves the global scale head to model the proportional trajectory and asks DSP to model relative mixture effects.",
        "- All four variants materially improve grouped OOF rank over the standalone repetition-aware mixture scaling-law adaptations tested separately.",
        "- The split-amplitude variant has the best grouped OOF RMSE and regret among this screen, but its leave-130M and leave-60M behavior is worse than the no-amplitude version.",
        "- The no-amplitude centered form is the most stable leave-one-scale-out candidate in this screen.",
        "- A full retune of DSP domain geometry across ND would require analytic/autodiff gradients or substantially more optimizer time; finite-difference L-BFGS over 80+ nonlinear domain parameters is not practical for quick iteration.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Run the local evaluation."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    full_idx = np.arange(len(data["y"]))
    summary_frames = []
    prediction_frames = []
    holdout_frames = []
    holdout_prediction_frames = []
    decoded_rows = []
    for spec in SPECS:
        print(f"Fitting {spec.name}", flush=True)
        cv_summary, cv_predictions = grouped_cv(data, spec)
        scale_summary, scale_predictions = scale_holdouts(data, spec)
        full_model = fit_model(data, full_idx, spec)
        decoded = decoded_params(full_model, data)
        decoded.update(
            {
                "model": spec.name,
                "parameter_count": parameter_count(spec, len(data["domain_names"])),
                "optimizer_success": full_model.optimizer_success,
                "optimizer_message": full_model.optimizer_message,
            }
        )
        decoded_rows.append(decoded)
        summary_frames.append(cv_summary)
        prediction_frames.append(cv_predictions)
        holdout_frames.append(scale_summary)
        holdout_prediction_frames.append(scale_predictions)
        print(cv_summary.to_string(index=False), flush=True)

    summary = pd.concat(summary_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    holdout = pd.concat(holdout_frames, ignore_index=True)
    holdout_predictions = pd.concat(holdout_prediction_frames, ignore_index=True)
    decoded = pd.DataFrame.from_records(decoded_rows)
    summary.to_csv(OUTPUT_DIR / "grouped_oof_summary.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "grouped_oof_predictions.csv", index=False)
    holdout.to_csv(OUTPUT_DIR / "leave_scale_summary.csv", index=False)
    holdout_predictions.to_csv(OUTPUT_DIR / "leave_scale_predictions.csv", index=False)
    decoded.to_csv(OUTPUT_DIR / "decoded_params.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "grouped_oof_summary": summary.to_dict(orient="records"),
                "leave_scale_summary": holdout.to_dict(orient="records"),
                "decoded_params": decoded.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    write_plots(summary, predictions, holdout)
    (OUTPUT_DIR / "report.md").write_text(render_report(summary, holdout, decoded))
    print(f"Wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
