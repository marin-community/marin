# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "plotly", "scipy", "scikit-learn", "kaleido"]
# ///
"""Evaluate the repetition-aware variable-size mixture law on Marin ND data.

The source paper's variable-size law is a one-target-vs-generic formula. This
script tests the closest Marin translations on the current ND scaling panel:

    L = E + C / N^beta + B * N^delta / D_eff^alpha + optional linear mix head.

For a multi-domain two-phase mixture, each domain contributes an effective token
count derived from its total exposure and an Apple-style repetition discount.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "analysis_dataset"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "repetition_aware_variable_size_law_nd_20260514"
IMG_DIR = OUTPUT_DIR / "img"
PRIMARY_METRIC = "eval/uncheatable_eval/bpb"
PHASE_FRACTIONS = np.asarray([0.8, 0.2], dtype=float)
N_REF = 58_998_528.0
D_REF = 1_199_833_088.0
RIDGE = 1e-6
CV_SPLITS = 5
CV_SEED = 0
MAXITER = 180


@dataclass(frozen=True)
class LawSpec:
    """One tested adaptation of the variable-size repetition-aware law."""

    name: str
    use_domain_tau: bool
    use_domain_linear_head: bool
    per_domain_r1: bool
    description: str


@dataclass(frozen=True)
class FittedLaw:
    """Fitted nonlinear and linear parameters."""

    spec: LawSpec
    theta: np.ndarray
    coef: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    train_objective: float
    optimizer_success: bool
    optimizer_message: str


SPECS = (
    LawSpec(
        name="scale_only_chinchilla_size",
        use_domain_tau=False,
        use_domain_linear_head=False,
        per_domain_r1=False,
        description="No mixture dependence: E + C/N^beta + B*N^delta/D^alpha.",
    ),
    LawSpec(
        name="raml_uniform_domain_value_shared_r1",
        use_domain_tau=False,
        use_domain_linear_head=False,
        per_domain_r1=False,
        description="Paper-style repetition discount summed over domains, with all domains equally valuable.",
    ),
    LawSpec(
        name="raml_domain_value_shared_r1",
        use_domain_tau=True,
        use_domain_linear_head=False,
        per_domain_r1=False,
        description="Shared r1 plus per-domain positive value weights inside D_eff.",
    ),
    LawSpec(
        name="raml_domain_value_linear_mix_shared_r1",
        use_domain_tau=True,
        use_domain_linear_head=True,
        per_domain_r1=False,
        description="Shared r1, per-domain value weights inside D_eff, plus a signed linear mixture head.",
    ),
    LawSpec(
        name="raml_domain_value_linear_mix_per_domain_r1",
        use_domain_tau=True,
        use_domain_linear_head=True,
        per_domain_r1=True,
        description="Per-domain r1, per-domain value weights inside D_eff, plus a signed linear mixture head.",
    ),
)


def load_data() -> dict[str, Any]:
    """Load the current ND analysis packet."""

    frame = pd.read_csv(DATA_DIR / "nd_scale_runs.csv", low_memory=False)
    payload = np.load(DATA_DIR / "nd_scale_packet.npz", allow_pickle=True)
    mask = np.asarray(payload["primary_y_mask"], dtype=bool)
    return {
        "frame": frame.loc[mask].reset_index(drop=True),
        "weights": np.asarray(payload["weights"], dtype=float)[mask],
        "y": np.asarray(payload["primary_y"], dtype=float)[mask],
        "mixture_ids": np.asarray(payload["mixture_ids"], dtype=object)[mask].astype(str),
        "run_names": np.asarray(payload["run_names"], dtype=object)[mask].astype(str),
        "scale_names": np.asarray(payload["scale_names"], dtype=object),
        "scale_index": np.asarray(payload["scale_index"], dtype=np.int64)[mask],
        "domain_names": np.asarray(payload["domain_names"], dtype=object).astype(str),
        "model_sizes": np.asarray(payload["model_sizes"], dtype=float)[mask],
        "realized_train_tokens": np.asarray(payload["realized_train_tokens"], dtype=float)[mask],
        "simulated_epoch_multipliers": np.asarray(payload["simulated_epoch_multipliers"], dtype=float)[mask],
    }


def effective_repeat(r: np.ndarray, r1: np.ndarray | float) -> np.ndarray:
    """Apple-style effective repeat count, piecewise extended to r < 1."""

    r_arr = np.asarray(r, dtype=float)
    r1_arr = np.asarray(r1, dtype=float)
    repeated = 1.0 + r1_arr * (1.0 - np.exp(-np.maximum(r_arr - 1.0, 0.0) / r1_arr))
    return np.where(r_arr <= 1.0, r_arr, repeated)


def theta_layout(spec: LawSpec, num_domains: int) -> tuple[int, list[tuple[float, float]]]:
    """Return theta length and L-BFGS-B bounds."""

    bounds: list[tuple[float, float]] = [
        (np.log(0.02), np.log(2.0)),  # alpha
        (np.log(0.02), np.log(2.0)),  # beta
        (np.log(1e-3), np.log(2.0)),  # delta
    ]
    if spec.name == "scale_only_chinchilla_size":
        return 3, bounds
    if spec.per_domain_r1:
        bounds.extend([(np.log(1e-3), np.log(100.0)) for _ in range(num_domains)])
    else:
        bounds.append((np.log(1e-3), np.log(100.0)))
    if spec.use_domain_tau:
        bounds.extend([(-4.0, 4.0) for _ in range(num_domains)])
    return len(bounds), bounds


def unpack_theta(theta: np.ndarray, spec: LawSpec, num_domains: int) -> dict[str, np.ndarray | float]:
    """Decode nonlinear parameters."""

    cursor = 0
    alpha = float(np.exp(theta[cursor]))
    cursor += 1
    beta = float(np.exp(theta[cursor]))
    cursor += 1
    delta = float(np.exp(theta[cursor]))
    cursor += 1
    if spec.name == "scale_only_chinchilla_size":
        return {"alpha": alpha, "beta": beta, "delta": delta, "r1": np.inf, "tau": np.ones(num_domains)}
    if spec.per_domain_r1:
        r1 = np.exp(theta[cursor : cursor + num_domains])
        cursor += num_domains
    else:
        r1 = float(np.exp(theta[cursor]))
        cursor += 1
    if spec.use_domain_tau:
        log_tau = theta[cursor : cursor + num_domains]
        cursor += num_domains
        log_tau = log_tau - float(np.mean(log_tau))
        tau = np.exp(log_tau)
    else:
        tau = np.ones(num_domains, dtype=float)
    if cursor != len(theta):
        raise ValueError(f"unused theta values for {spec.name}: cursor={cursor}, len={len(theta)}")
    return {"alpha": alpha, "beta": beta, "delta": delta, "r1": r1, "tau": tau}


def start_bank(spec: LawSpec, num_domains: int) -> list[np.ndarray]:
    """Deterministic nonlinear starts."""

    theta_len, _bounds = theta_layout(spec, num_domains)
    starts: list[np.ndarray] = []
    for alpha, beta, delta, r1 in [
        (0.25, 0.25, 0.25, 1.0),
        (0.35, 0.35, 0.35, 3.0),
        (0.50, 0.25, 0.50, 10.0),
        (0.15, 0.50, 0.15, 30.0),
        (0.75, 0.15, 0.75, 5.0),
    ]:
        values = [np.log(alpha), np.log(beta), np.log(delta)]
        if spec.name != "scale_only_chinchilla_size":
            if spec.per_domain_r1:
                values.extend([np.log(r1)] * num_domains)
            else:
                values.append(np.log(r1))
            if spec.use_domain_tau:
                values.extend([0.0] * num_domains)
        start = np.asarray(values, dtype=float)
        if len(start) != theta_len:
            raise ValueError(f"bad start length for {spec.name}: {len(start)} != {theta_len}")
        starts.append(start)
    return starts


def domain_effective_tokens(
    data: dict[str, Any], indices: np.ndarray, params: dict[str, Any], spec: LawSpec
) -> np.ndarray:
    """Compute per-domain effective tokens for each row."""

    weights = data["weights"][indices]
    total_domain_fraction = np.einsum("p,npd->nd", PHASE_FRACTIONS, weights)
    total_repeats = np.sum(weights * data["simulated_epoch_multipliers"][indices], axis=1)
    tokens_seen = total_domain_fraction * data["realized_train_tokens"][indices, None]
    if spec.name == "scale_only_chinchilla_size":
        return tokens_seen
    repeat = np.maximum(total_repeats, 1e-12)
    effective_r = effective_repeat(repeat, params["r1"])
    return tokens_seen * (effective_r / repeat)


def design_matrix(data: dict[str, Any], indices: np.ndarray, spec: LawSpec, theta: np.ndarray) -> np.ndarray:
    """Build linear design for a nonlinear theta."""

    num_domains = len(data["domain_names"])
    params = unpack_theta(theta, spec, num_domains)
    n = data["model_sizes"][indices] / N_REF
    if spec.name == "scale_only_chinchilla_size":
        d_eff = data["realized_train_tokens"][indices] / D_REF
    else:
        domain_tokens = domain_effective_tokens(data, indices, params, spec)
        tau = np.asarray(params["tau"], dtype=float)
        d_eff = np.maximum(domain_tokens @ tau / D_REF, 1e-12)
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    delta = float(params["delta"])
    cols = [
        np.ones(len(indices), dtype=float),
        np.power(np.maximum(n, 1e-12), -beta),
        np.power(np.maximum(n, 1e-12), delta) / np.power(d_eff, alpha),
    ]
    if spec.use_domain_linear_head:
        h = np.einsum("p,npd->nd", PHASE_FRACTIONS, data["weights"][indices])
        cols.extend([h[:, j] for j in range(h.shape[1])])
    return np.column_stack(cols)


def fit_linear(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a lightly regularized linear head on standardized non-intercept features."""

    intercept = design[:, :1]
    rest = design[:, 1:]
    mean = rest.mean(axis=0)
    std = rest.std(axis=0)
    std = np.where(std < 1e-10, 1.0, std)
    x = np.column_stack([intercept, (rest - mean) / std])
    penalty = np.sqrt(RIDGE) * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    x_aug = np.vstack([x, penalty])
    y_aug = np.concatenate([y, np.zeros(x.shape[1], dtype=float)])
    coef = np.linalg.lstsq(x_aug, y_aug, rcond=None)[0]
    pred = x @ coef
    return coef, mean, std, pred


def objective(theta: np.ndarray, data: dict[str, Any], indices: np.ndarray, spec: LawSpec) -> float:
    """Profile objective over nonlinear parameters."""

    design = design_matrix(data, indices, spec, theta)
    _coef, _mean, _std, pred = fit_linear(design, data["y"][indices])
    residual = pred - data["y"][indices]
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    tail_count = max(8, int(np.ceil(0.15 * len(indices))))
    tail = np.argsort(pred)[:tail_count]
    optimism = float(np.mean(np.maximum(data["y"][indices][tail] - pred[tail], 0.0)))
    return rmse + 0.25 * optimism


def fit_law(data: dict[str, Any], indices: np.ndarray, spec: LawSpec, initial: np.ndarray | None = None) -> FittedLaw:
    """Fit one law spec."""

    _theta_len, bounds = theta_layout(spec, len(data["domain_names"]))
    starts = [initial] if initial is not None else start_bank(spec, len(data["domain_names"]))
    best: Any | None = None
    for start in starts:
        result = minimize(
            objective,
            start,
            args=(data, indices, spec),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": MAXITER, "ftol": 1e-10, "gtol": 1e-6},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    if best is None:
        raise RuntimeError(f"no optimizer result for {spec.name}")
    design = design_matrix(data, indices, spec, best.x)
    coef, mean, std, _pred = fit_linear(design, data["y"][indices])
    return FittedLaw(
        spec=spec,
        theta=np.asarray(best.x, dtype=float),
        coef=np.asarray(coef, dtype=float),
        feature_mean=np.asarray(mean, dtype=float),
        feature_std=np.asarray(std, dtype=float),
        train_objective=float(best.fun),
        optimizer_success=bool(best.success),
        optimizer_message=str(best.message),
    )


def predict_law(model: FittedLaw, data: dict[str, Any], indices: np.ndarray) -> np.ndarray:
    """Predict rows with a fitted law."""

    design = design_matrix(data, indices, model.spec, model.theta)
    rest = design[:, 1:]
    x = np.column_stack([design[:, :1], (rest - model.feature_mean) / model.feature_std])
    return x @ model.coef


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
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
        "pearson": float(pearsonr(y, pred).statistic) if len(y) > 2 and np.std(pred) > 0 else float("nan"),
        "spearman": float(spearmanr(y, pred).statistic) if len(y) > 2 and np.std(pred) > 0 else float("nan"),
        "actual_std": float(np.std(y)),
        "predicted_std": float(np.std(pred)),
        "predicted_actual_std_ratio": float(np.std(pred) / max(np.std(y), 1e-12)),
        "regret_at_1": chosen_actual - best_actual,
        "chosen_actual": chosen_actual,
        "best_actual": best_actual,
        "top8_overlap": len(actual_top & pred_top) / float(k),
    }


def grouped_cv(data: dict[str, Any], spec: LawSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run grouped OOF by mixture id."""

    all_idx = np.arange(len(data["y"]))
    groups = data["mixture_ids"]
    splitter = GroupKFold(n_splits=CV_SPLITS)
    pred = np.full(len(all_idx), np.nan, dtype=float)
    fold_rows = []
    full_model = fit_law(data, all_idx, spec)
    for fold_idx, (train_pos, test_pos) in enumerate(splitter.split(all_idx, data["y"], groups)):
        model = fit_law(data, all_idx[train_pos], spec, initial=full_model.theta)
        fold_pred = predict_law(model, data, all_idx[test_pos])
        pred[test_pos] = fold_pred
        row = metrics(data["y"][test_pos], fold_pred)
        row.update({"model": spec.name, "fold": fold_idx, "optimizer_success": model.optimizer_success})
        fold_rows.append(row)
    if np.isnan(pred).any():
        raise ValueError(f"missing OOF predictions for {spec.name}")
    pred_frame = prediction_frame(data, all_idx, pred, spec.name, split="grouped_oof")
    summary = metrics(data["y"], pred)
    summary.update(
        {
            "model": spec.name,
            "split": "grouped_oof",
            "parameter_count": parameter_count(spec, len(data["domain_names"])),
            "train_objective": full_model.train_objective,
            "optimizer_success": full_model.optimizer_success,
            "optimizer_message": full_model.optimizer_message,
        }
    )
    return pd.DataFrame([summary]), pred_frame


def scale_holdouts(data: dict[str, Any], spec: LawSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on all but one scale and predict the held-out scale."""

    all_idx = np.arange(len(data["y"]))
    scale_labels = data["scale_names"][data["scale_index"]].astype(str)
    full_model = fit_law(data, all_idx, spec)
    rows = []
    predictions = []
    for scale in sorted(np.unique(scale_labels)):
        test = np.flatnonzero(scale_labels == scale)
        train = np.flatnonzero(scale_labels != scale)
        if len(test) < 3 or len(train) < 20:
            continue
        model = fit_law(data, train, spec, initial=full_model.theta)
        pred = predict_law(model, data, test)
        row = metrics(data["y"][test], pred)
        row.update({"model": spec.name, "split": f"leave_scale_{scale}", "scale": scale})
        rows.append(row)
        predictions.append(prediction_frame(data, test, pred, spec.name, split=f"leave_scale_{scale}"))
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def prediction_frame(
    data: dict[str, Any], indices: np.ndarray, pred: np.ndarray, model_name: str, *, split: str
) -> pd.DataFrame:
    """Build row-level prediction frame."""

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


def parameter_count(spec: LawSpec, num_domains: int) -> int:
    """Approximate fitted parameter count."""

    nonlinear = 3
    if spec.name != "scale_only_chinchilla_size":
        nonlinear += num_domains if spec.per_domain_r1 else 1
        nonlinear += num_domains if spec.use_domain_tau else 0
    linear = 3 + (num_domains if spec.use_domain_linear_head else 0)
    return nonlinear + linear


def decoded_params(model: FittedLaw, num_domains: int) -> dict[str, Any]:
    """Decode fitted parameters for JSON output."""

    params = unpack_theta(model.theta, model.spec, num_domains)
    result: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            result[f"{key}_min"] = float(np.min(value))
            result[f"{key}_median"] = float(np.median(value))
            result[f"{key}_max"] = float(np.max(value))
        else:
            result[key] = float(value)
    return result


def write_plots(summary: pd.DataFrame, predictions: pd.DataFrame, holdout: pd.DataFrame) -> None:
    """Write diagnostic plots."""

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
        title="Repetition-aware variable-size law: grouped OOF predictions",
        height=850,
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
        title="Grouped OOF metric comparison",
        height=700,
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
            title="Leave-one-scale-out metrics",
            height=650,
        )
        fig.update_layout(template="plotly_white")
        fig.write_html(IMG_DIR / "leave_scale_metric_comparison.html", include_plotlyjs="cdn")
        try:
            fig.write_image(IMG_DIR / "leave_scale_metric_comparison.png", scale=2)
        except ValueError:
            pass


def report(summary: pd.DataFrame, holdout: pd.DataFrame, decoded: pd.DataFrame) -> str:
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
        "# Repetition-Aware Variable-Size Law on Marin ND Scaling Data",
        "",
        "## Data",
        "",
        f"- Source: `{DATA_DIR / 'nd_scale_runs.csv'}`",
        f"- Metric: `{PRIMARY_METRIC}`",
        f"- Rows: `{int(summary['n'].iloc[0])}` labeled rows for grouped OOF.",
        "",
        "## Tested Forms",
        "",
        "Base paper form:",
        "",
        "$$L = E + C/N^{\\beta} + B N^{\\delta}/D_{\\mathrm{eff}}^{\\alpha} + \\gamma h.$$",
        "",
        "For Marin's 39-domain, two-phase mixtures, domain exposure is aggregated as:",
        "",
        "$$h_i = 0.8w_{0i} + 0.2w_{1i}, \\qquad r_i = w_{0i}c_{0i} + w_{1i}c_{1i},$$",
        "",
        "$$D_{i,\\mathrm{eff}} = h_iD\\,\\frac{r_{i,\\mathrm{eff}}}{r_i}, \\qquad r_{i,\\mathrm{eff}} = r_i \\; (r_i \\le 1), \\quad 1 + r_1(1-e^{-(r_i-1)/r_1}) \\; (r_i > 1).$$",
        "",
        "The multi-domain variants use either equal domain value, positive per-domain value weights inside `D_eff`, or an additional signed linear mixture head.",
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
        "## Decoded Nonlinear Parameter Summary",
        "",
        decoded.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
        "- The literal variable-size law is too low-dimensional for Marin mixtures if domains are treated as equally valuable.",
        "- Positive per-domain value weights are the most stable improvement over the literal paper form.",
        "- A signed linear mixture head plus per-domain repetition constants gives the best grouped OOF fit here, but the shared-r1 signed-head variant is badly miscalibrated and the best variant hits optimizer budget limits.",
        "- These RAML adaptations are useful baselines/backbones for scale/repetition structure, not replacements for DSP-style mixture structure without additional optimum and perturbation-geometry validation.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    summary_frames = []
    prediction_frames = []
    holdout_frames = []
    holdout_prediction_frames = []
    decoded_rows = []
    full_indices = np.arange(len(data["y"]))
    for spec in SPECS:
        print(f"Fitting {spec.name}", flush=True)
        cv_summary, cv_predictions = grouped_cv(data, spec)
        scale_summary, scale_predictions = scale_holdouts(data, spec)
        full_model = fit_law(data, full_indices, spec)
        decoded = decoded_params(full_model, len(data["domain_names"]))
        decoded.update({"model": spec.name, "parameter_count": parameter_count(spec, len(data["domain_names"]))})
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
    (OUTPUT_DIR / "report.md").write_text(report(summary, holdout, decoded))
    print(f"Wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
