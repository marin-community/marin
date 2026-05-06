# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit and optimize P3 on the 60M/1.2B fit swarm.

This is a local diagnostic for comparing the collaborator P3 form against the
validated GRP no-L2 optimum on the same 242-row 60M swarm.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many.dataset_metadata import build_two_phase_many_loop_config
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.materialize_fit_dataset import (
    materialize_fit_dataset,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GENERIC_FAMILY_NAMES,
)
from experiments.domain_phase_mix.static_batch_selection import build_dataset_spec_from_frame

ROOT = Path(__file__).resolve().parent
SUMMARY_JSON = ROOT / "p3_60m_swarm_summary.json"
PREDICTIONS_CSV = ROOT / "p3_60m_swarm_predictions.csv"
GRID_CSV = ROOT / "p3_60m_swarm_grid.csv"
OPTIMUM_CSV = ROOT / "p3_60m_swarm_raw_optimum_weights.csv"
TOP_DOMAINS_CSV = ROOT / "p3_60m_swarm_raw_optimum_top_domains.csv"
PREDICTED_ACTUAL_HTML = ROOT / "p3_60m_swarm_predicted_vs_actual.html"
PREDICTED_ACTUAL_PNG = ROOT / "p3_60m_swarm_predicted_vs_actual.png"
OPTIMUM_HTML = ROOT / "p3_60m_swarm_raw_optimum_top_domains.html"
OPTIMUM_PNG = ROOT / "p3_60m_swarm_raw_optimum_top_domains.png"

GRP_WEIGHTS_CSV = (
    ROOT.parents[1]
    / "grp_power_family_penalty_no_l2_60m_vs_300m_fit_weights.csv"
)
TARGET_METRIC = "eval/uncheatable_eval/bpb"
SCALE = "60m_1p2b"
COHORT = "signal"
RUN_SET = "fit_swarm_60m_default"
MODEL_TARGET_COLUMN = "model_target"

ETA_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0)
A_GRID = tuple(np.linspace(0.5, 2.0, 8))
P_GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
RIDGE_GRID = tuple(np.logspace(-4, 4, 17))
CV_SPLITS = 5
CV_SEED = 0
RANDOM_STARTS = 96
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
    """Fitted standardized ridge head for P3."""

    params: P3Params
    active_mask: np.ndarray
    design_mean: np.ndarray
    design_std: np.ndarray
    coef: np.ndarray
    intercept: float

    def predict(self, weights: np.ndarray, c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
        """Predict objective metric for weights shaped ``[rows, phases, domains]``."""
        design = build_p3_design(weights, c0, c1, self.params)
        active = design[:, self.active_mask]
        standardized = (active - self.design_mean) / self.design_std
        return np.asarray(self.intercept + standardized @ self.coef, dtype=float)


@dataclass(frozen=True)
class FitData:
    """Fit-ready 60M swarm data."""

    frame: pd.DataFrame
    y: np.ndarray
    weights: np.ndarray
    c0: np.ndarray
    c1: np.ndarray
    domains: list[str]
    name_col: str


def load_fit_data() -> FitData:
    """Load the 242-row 60M fit swarm as a dense matrix."""
    frame = materialize_fit_dataset(
        TARGET_METRIC,
        scale=SCALE,
        cohort=COHORT,
        run_set=RUN_SET,
    )
    frame = frame.copy()
    frame[MODEL_TARGET_COLUMN] = frame["objective_metric"]
    loop = build_two_phase_many_loop_config(objective_metric=MODEL_TARGET_COLUMN, name="p3_60m_swarm")
    spec = build_dataset_spec_from_frame(
        frame,
        objective_metric=MODEL_TARGET_COLUMN,
        name="p3_60m_swarm",
        loop=loop,
    )
    name_col = "candidate_run_name" if "candidate_run_name" in frame.columns else "run_name"
    return FitData(
        frame=frame.reset_index(drop=True),
        y=np.asarray(spec.y, dtype=float),
        weights=np.asarray(spec.weights, dtype=float),
        c0=np.asarray(spec.epoch_multipliers[0], dtype=float),
        c1=np.asarray(spec.epoch_multipliers[1], dtype=float),
        domains=list(spec.domain_names),
        name_col=name_col,
    )


def build_p3_design(weights: np.ndarray, c0: np.ndarray, c1: np.ndarray, params: P3Params) -> np.ndarray:
    """Build P3 features: per-domain exposure signals and two concentration penalties."""
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
    """Fit a ridge linear head for fixed P3 nonlinear parameters."""
    design = build_p3_design(weights, c0, c1, params)
    active_mask = design.std(axis=0) > 1e-12
    if not active_mask.any():
        raise ValueError("P3 design has no active features")
    active = design[:, active_mask]
    design_mean = active.mean(axis=0)
    design_std = active.std(axis=0)
    standardized = (active - design_mean) / design_std
    centered_y = y - y.mean()
    coef = np.linalg.solve(
        standardized.T @ standardized + params.ridge_alpha * np.eye(standardized.shape[1]),
        standardized.T @ centered_y,
    )
    return P3Model(
        params=params,
        active_mask=active_mask,
        design_mean=design_mean,
        design_std=design_std,
        coef=coef,
        intercept=float(y.mean()),
    )


def prediction_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Return standard regression and rank metrics."""
    residual = pred - y
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - np.sum(residual**2) / np.sum((y - y.mean()) ** 2)),
        "pearson": float(np.corrcoef(y, pred)[0, 1]),
        "spearman": float(stats.spearmanr(y, pred).statistic),
    }


def cv_predict(data: FitData, params: P3Params) -> tuple[np.ndarray, list[float]]:
    """Return 5-fold OOF predictions and foldwise regret."""
    kf = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_SEED)
    oof = np.zeros_like(data.y)
    regrets: list[float] = []
    for train_idx, test_idx in kf.split(data.weights):
        model = fit_p3_head(data.weights[train_idx], data.y[train_idx], data.c0, data.c1, params)
        pred = model.predict(data.weights[test_idx], data.c0, data.c1)
        oof[test_idx] = pred
        chosen = int(np.argmin(pred))
        regrets.append(float(data.y[test_idx][chosen] - np.min(data.y[test_idx])))
    return oof, regrets


def select_params(data: FitData) -> tuple[P3Params, pd.DataFrame, np.ndarray, list[float]]:
    """Grid-search P3 hyperparameters by OOF RMSE."""
    rows: list[dict[str, float]] = []
    best_params: P3Params | None = None
    best_pred: np.ndarray | None = None
    best_regrets: list[float] | None = None
    best_rmse = float("inf")

    for eta in ETA_GRID:
        for a in A_GRID:
            for p in P_GRID:
                for ridge_alpha in RIDGE_GRID:
                    params = P3Params(float(eta), float(a), float(p), float(ridge_alpha))
                    pred, regrets = cv_predict(data, params)
                    metrics = prediction_metrics(data.y, pred)
                    row = {
                        "eta": params.eta,
                        "a": params.a,
                        "p": params.p,
                        "ridge_alpha": params.ridge_alpha,
                        "cv_foldmean_regret_at_1": float(np.mean(regrets)),
                        **{f"cv_{key}": value for key, value in metrics.items()},
                    }
                    rows.append(row)
                    if metrics["rmse"] < best_rmse:
                        best_rmse = metrics["rmse"]
                        best_params = params
                        best_pred = pred
                        best_regrets = regrets

    if best_params is None or best_pred is None or best_regrets is None:
        raise RuntimeError("No P3 parameters selected")
    return best_params, pd.DataFrame(rows), best_pred, best_regrets


def softmax(logits: np.ndarray) -> np.ndarray:
    """Stable row softmax."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def weights_from_logits(logits: np.ndarray, domain_count: int) -> np.ndarray:
    """Map concatenated phase logits to normalized two-phase weights."""
    return softmax(logits.reshape(2, domain_count))[None, :, :]


def logits_from_weights(weights: np.ndarray) -> np.ndarray:
    """Map normalized phase weights to centered logits."""
    clipped = np.clip(weights, 1e-12, 1.0)
    logits = np.log(clipped)
    logits -= logits.mean(axis=1, keepdims=True)
    return logits.reshape(-1)


def objective_from_logits(logits: np.ndarray, model: P3Model, data: FitData) -> float:
    """Objective minimized by the raw deployment optimizer."""
    weights = weights_from_logits(logits, len(data.domains))
    return float(model.predict(weights, data.c0, data.c1)[0])


def family_map(domains: list[str]) -> dict[str, list[int]]:
    """Return the same coarse family partition used by the GRP no-L2 diagnostics."""
    families = {family: [] for family in GENERIC_FAMILY_NAMES}
    for idx, domain in enumerate(domains):
        is_broad = (
            domain.startswith("dolma3_cc/")
            or domain
            in {
                "dolma3_wikipedia",
                "dolmino_common_crawl_hq",
                "dolmino_olmocr_pdfs_hq",
                "dolmino_stem_heavy_crawl",
            }
            or domain.endswith("synth_qa")
        )
        is_tech = any(token in domain for token in ("stack_edu", "synth_code", "synth_math")) or domain in {
            "dolma3_arxiv",
            "dolma3_finemath_3plus",
        }
        is_reasoning = domain in {"dolmino_synth_instruction", "dolmino_synth_thinking"}
        if is_broad:
            families["broad_text"].append(idx)
        if is_tech:
            families["tech_code"].append(idx)
        if is_reasoning:
            families["reasoning"].append(idx)
    return families


def mixture_diagnostics(weights: np.ndarray, data: FitData, prefix: str) -> dict[str, float]:
    """Return support and family-share diagnostics for one two-phase mixture."""
    families = family_map(data.domains)
    out: dict[str, float] = {}
    for phase_idx in range(2):
        phase = weights[phase_idx]
        entropy = -float(np.sum(phase * np.log(np.clip(phase, 1e-12, 1.0))))
        out[f"{prefix}_phase{phase_idx}_max_weight"] = float(phase.max())
        out[f"{prefix}_phase{phase_idx}_support_below_1e4"] = float(np.sum(phase > 1e-4))
        out[f"{prefix}_phase{phase_idx}_entropy"] = entropy
        for family_name, indices in families.items():
            out[f"{prefix}_phase{phase_idx}_{family_name}"] = float(phase[indices].sum())
    return out


def average_phase_tv(a: np.ndarray, b: np.ndarray) -> float:
    """Return average total variation across the two phases."""
    return float(0.5 * np.mean(np.sum(np.abs(a - b), axis=1)))


def optimize_raw(model: P3Model, data: FitData, grp_weights: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Optimize P3 over the two phase simplexes with multiple starts."""
    starts: list[np.ndarray] = []
    uniform = np.full((2, len(data.domains)), 1.0 / len(data.domains), dtype=float)
    starts.append(logits_from_weights(uniform))
    starts.append(logits_from_weights(grp_weights))
    for _, row in data.frame.nsmallest(12, "objective_metric").iterrows():
        weights = np.stack(
            [
                row[[f"phase_0_{domain}" for domain in data.domains]].to_numpy(dtype=float),
                row[[f"phase_1_{domain}" for domain in data.domains]].to_numpy(dtype=float),
            ],
            axis=0,
        )
        starts.append(logits_from_weights(weights))
    rng = np.random.default_rng(0)
    for concentration in (0.2, 0.5, 1.0, 2.0):
        for _ in range(RANDOM_STARTS // 4):
            sampled = rng.dirichlet(np.full(len(data.domains), concentration), size=2)
            starts.append(logits_from_weights(sampled))

    best_result = None
    rows: list[dict[str, float | int | bool | str]] = []
    for idx, start in enumerate(starts):
        result = minimize(
            objective_from_logits,
            start,
            args=(model, data),
            method="L-BFGS-B",
            options={"maxiter": 3000, "maxfun": 200000, "ftol": 1e-12, "gtol": 1e-9, "maxls": 80},
        )
        rows.append(
            {
                "start_index": idx,
                "success": bool(result.success),
                "fun": float(result.fun),
                "nit": int(result.nit),
                "message": str(result.message),
            }
        )
        if best_result is None or float(result.fun) < float(best_result.fun):
            best_result = result
    if best_result is None:
        raise RuntimeError("Raw optimization did not run")
    pd.DataFrame(rows).to_csv(ROOT / "p3_60m_swarm_raw_optimizer_starts.csv", index=False)
    optimum_weights = weights_from_logits(best_result.x, len(data.domains))[0]
    return optimum_weights, {
        "optimizer_success": bool(best_result.success),
        "optimizer_message": str(best_result.message),
        "optimizer_value": float(best_result.fun),
        "optimizer_nit": int(best_result.nit),
    }


def load_grp_weights(data: FitData) -> np.ndarray:
    """Load the validated 60M GRP no-L2 raw optimum weights."""
    frame = pd.read_csv(GRP_WEIGHTS_CSV)
    indexed = frame.set_index("domain").loc[data.domains]
    return np.stack(
        [
            indexed["fit60_phase0_weight"].to_numpy(dtype=float),
            indexed["fit60_phase1_weight"].to_numpy(dtype=float),
        ],
        axis=0,
    )


def nearest_observed(weights: np.ndarray, data: FitData) -> dict[str, float | str]:
    """Return nearest observed mixture and its measured value."""
    distances = []
    for _, row in data.frame.iterrows():
        observed = np.stack(
            [
                row[[f"phase_0_{domain}" for domain in data.domains]].to_numpy(dtype=float),
                row[[f"phase_1_{domain}" for domain in data.domains]].to_numpy(dtype=float),
            ],
            axis=0,
        )
        distances.append(average_phase_tv(weights, observed))
    nearest_idx = int(np.argmin(distances))
    return {
        "nearest_observed_run_name": str(data.frame.iloc[nearest_idx][data.name_col]),
        "nearest_observed_value": float(data.y[nearest_idx]),
        "nearest_observed_tv": float(distances[nearest_idx]),
    }


def write_prediction_plot(data: FitData, train_pred: np.ndarray, oof_pred: np.ndarray) -> None:
    """Write predicted-vs-actual diagnostics."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Full-data fit", "5-fold OOF"))
    for col, pred, label in [(1, train_pred, "train"), (2, oof_pred, "oof")]:
        fig.add_trace(
            go.Scatter(
                x=data.y,
                y=pred,
                text=data.frame[data.name_col],
                mode="markers",
                marker={
                    "size": 7,
                    "color": data.frame["phase_1_dolma3_stack_edu"],
                    "colorscale": "RdYlGn_r",
                    "showscale": col == 2,
                    "colorbar": {"title": "phase-1<br>stack_edu"},
                    "line": {"width": 0.3, "color": "white"},
                },
                name=label,
                showlegend=False,
            ),
            row=1,
            col=col,
        )
    lo = float(min(data.y.min(), train_pred.min(), oof_pred.min()))
    hi = float(max(data.y.max(), train_pred.max(), oof_pred.max()))
    for col in (1, 2):
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line={"color": "black", "dash": "dash"}), row=1, col=col)
    fig.update_xaxes(title_text="Actual BPB")
    fig.update_yaxes(title_text="Predicted BPB")
    fig.update_layout(title="P3 on 60M/1.2B fit swarm", width=1300, height=560, template="plotly_white")
    fig.write_html(PREDICTED_ACTUAL_HTML)
    fig.write_image(PREDICTED_ACTUAL_PNG, scale=2)


def write_optimum_plot(optimum: np.ndarray, grp: np.ndarray, data: FitData) -> None:
    """Write a top-domain phase comparison for P3 and GRP optima."""
    rows = []
    for idx, domain in enumerate(data.domains):
        rows.append(
            {
                "domain": domain,
                "p3_phase0": optimum[0, idx],
                "p3_phase1": optimum[1, idx],
                "grp_phase0": grp[0, idx],
                "grp_phase1": grp[1, idx],
                "p3_total": optimum[:, idx].mean(),
                "grp_total": grp[:, idx].mean(),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OPTIMUM_CSV, index=False)
    top = frame.assign(max_total=lambda df: df[["p3_total", "grp_total"]].max(axis=1)).nlargest(20, "max_total")
    top.to_csv(TOP_DOMAINS_CSV, index=False)

    fig = go.Figure()
    x = top["domain"].tolist()
    for column, name, color in [
        ("p3_phase0", "P3 phase 0", "#1b7837"),
        ("p3_phase1", "P3 phase 1", "#5aae61"),
        ("grp_phase0", "GRP phase 0", "#762a83"),
        ("grp_phase1", "GRP phase 1", "#af8dc3"),
    ]:
        fig.add_trace(go.Bar(x=x, y=top[column], name=name, marker_color=color))
    fig.update_layout(
        title="Raw P3 optimum vs validated GRP no-L2 optimum, top domains",
        xaxis_title="Domain",
        yaxis_title="Phase weight",
        barmode="group",
        width=1450,
        height=720,
        template="plotly_white",
    )
    fig.update_xaxes(tickangle=35)
    fig.write_html(OPTIMUM_HTML)
    fig.write_image(OPTIMUM_PNG, scale=2)


def main() -> None:
    """Run the local P3 fit and raw optimization."""
    data = load_fit_data()
    selected, grid, oof_pred, fold_regrets = select_params(data)
    grid.to_csv(GRID_CSV, index=False)

    model = fit_p3_head(data.weights, data.y, data.c0, data.c1, selected)
    train_pred = model.predict(data.weights, data.c0, data.c1)
    write_prediction_plot(data, train_pred, oof_pred)
    pd.DataFrame(
        {
            "run_name": data.frame[data.name_col],
            "actual": data.y,
            "p3_train_prediction": train_pred,
            "p3_oof_prediction": oof_pred,
            "residual_oof": oof_pred - data.y,
        }
    ).to_csv(PREDICTIONS_CSV, index=False)

    grp_weights = load_grp_weights(data)
    optimum, optimizer_info = optimize_raw(model, data, grp_weights)
    write_optimum_plot(optimum, grp_weights, data)

    p3_nearest = nearest_observed(optimum, data)
    grp_nearest = nearest_observed(grp_weights, data)
    p3_predicted_value = float(model.predict(optimum[None, :, :], data.c0, data.c1)[0])
    grp_predicted_by_p3 = float(model.predict(grp_weights[None, :, :], data.c0, data.c1)[0])
    observed_best_idx = int(np.argmin(data.y))
    observed_best = {
        "run_name": str(data.frame.iloc[observed_best_idx][data.name_col]),
        "value": float(data.y[observed_best_idx]),
    }

    summary = {
        "target": TARGET_METRIC,
        "scale": SCALE,
        "run_set": RUN_SET,
        "row_count": int(len(data.y)),
        "model": "P3",
        "parameter_count": {
            "linear_coefficients": int(len(model.coef)),
            "intercept": 1,
            "shape_parameters": 3,
            "ridge_hyperparameter": 1,
            "total_without_ridge_hyperparameter": int(len(model.coef) + 1 + 3),
            "total_with_ridge_hyperparameter": int(len(model.coef) + 1 + 4),
        },
        "selected_params": selected.__dict__,
        "linear_head": {
            "coef_standardized": model.coef.tolist(),
            "intercept": model.intercept,
        },
        "train": prediction_metrics(data.y, train_pred),
        "cv": {
            **prediction_metrics(data.y, oof_pred),
            "foldmean_regret_at_1": float(np.mean(fold_regrets)),
            "foldmedian_regret_at_1": float(np.median(fold_regrets)),
            "foldmax_regret_at_1": float(np.max(fold_regrets)),
        },
        "observed_best": observed_best,
        "raw_optimum": {
            **optimizer_info,
            "predicted_value": p3_predicted_value,
            **p3_nearest,
            **mixture_diagnostics(optimum, data, "p3"),
        },
        "validated_grp_no_l2_optimum": {
            "predicted_by_p3": grp_predicted_by_p3,
            "phase_tv_vs_p3": average_phase_tv(grp_weights, optimum),
            **grp_nearest,
            **mixture_diagnostics(grp_weights, data, "grp"),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
