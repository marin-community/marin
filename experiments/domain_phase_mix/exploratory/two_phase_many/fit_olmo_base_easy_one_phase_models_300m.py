# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Fit one-phase OLMix and effective-exposure DSP on OLMoBaseEval Table-9.

Inputs are the one-phase augmented parity panel materialized by
`materialize_olmo_base_easy_one_phase_parity_panel_300m.py`: 240 one-phase
qsplit rows, the shared phase-tied stratified baseline, 39 phase-constant
proportional-domain-deletion controls, and an 11-observation proportional
reference mean.

The output mirrors the two-phase Table-9 KL overlay plots: predicted BPB,
materialized epochs, and TV-to-proportional against KL. OLMix is fit
paper-faithfully per Table-9 component with a cap-4 repetition constraint. DSP
is fit directly to the Table-9 macro and optimized over a tied one-phase
simplex with KL-only proposal regularization.
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
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as base_olmix,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_paper_faithful_olmix_300m as paper_olmix,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
PANEL_PATH = (
    SCRIPT_DIR
    / "reference_outputs"
    / "olmo_base_easy_one_phase_parity_panel_300m_20260628"
    / "one_phase_augmented_fit_panel.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "olmo_base_easy_one_phase_model_sweeps_300m_20260628"
MACRO_TARGET = "table9_macro_bpb"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
DEFAULT_KL_GRID = (
    "0,0.001,0.0025,0.005,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.025,"
    "0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1.0"
)


@dataclass(frozen=True)
class DspFitSummary:
    model_family: str
    variant: str
    target_metric: str
    n_rows: int
    n_signal_rows: int
    n_deletion_rows: int
    n_proportional_reference_rows: int
    proportional_reference_mean: float
    proportional_reference_std: float
    linear_reg: float
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
    total_param_count: int
    m_dependent_params_per_domain: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=PANEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kl-grid", default=DEFAULT_KL_GRID)
    parser.add_argument("--olmix-huber-delta", type=float, default=0.01)
    parser.add_argument("--olmix-fit-n-starts", type=int, default=24)
    parser.add_argument("--dsp-linear-reg", type=float, default=1e-4)
    parser.add_argument("--dsp-maxiter", type=int, default=dsp.FIT_MAXITER)
    parser.add_argument("--dsp-coarse-top-k", type=int, default=dsp.START_TOP_K)
    parser.add_argument("--dsp-basin-hopping-iters", type=int, default=1)
    parser.add_argument("--dsp-raw-starts", type=int, default=40)
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("Expected at least one float")
    return sorted(set(parsed))


def phase_columns(frame: pd.DataFrame) -> list[str]:
    phase0 = [column for column in frame.columns if column.startswith("phase_0_")]
    phase1 = [column for column in frame.columns if column.startswith("phase_1_")]
    if len(phase0) != 39 or len(phase1) != 39:
        raise ValueError(f"Expected 39 domains in each phase, found {len(phase0)} and {len(phase1)}")
    return phase0 + phase1


def domain_names(columns: list[str]) -> list[str]:
    phase0 = [column.removeprefix("phase_0_") for column in columns if column.startswith("phase_0_")]
    phase1 = [column.removeprefix("phase_1_") for column in columns if column.startswith("phase_1_")]
    if phase0 != phase1:
        raise ValueError("Phase 0 and phase 1 domain columns are not aligned")
    return phase0


def load_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_csv(path, low_memory=False)
    if MACRO_TARGET not in panel.columns:
        raise ValueError(f"{path} is missing {MACRO_TARGET}")
    components = paper_olmix.table9_component_order()
    missing = [component for component in components if component not in panel.columns]
    if missing:
        raise ValueError(f"Panel is missing Table-9 component columns: {missing[:8]}")
    target_columns = [*components, MACRO_TARGET]
    if panel[target_columns].isna().any().any():
        bad_cols = panel[target_columns].columns[panel[target_columns].isna().any(axis=0)].tolist()
        raise ValueError(f"Panel has missing target values: {bad_cols[:8]}")
    if len(panel) != 280:
        raise ValueError(f"Expected 280 one-phase fit rows, found {len(panel)}")
    return panel


def single_weights(panel: pd.DataFrame, domains: list[str]) -> np.ndarray:
    phase0 = panel[[f"phase_0_{domain}" for domain in domains]].astype(float).to_numpy()
    phase1 = panel[[f"phase_1_{domain}" for domain in domains]].astype(float).to_numpy()
    phase0 = phase0 / phase0.sum(axis=1, keepdims=True)
    phase1 = phase1 / phase1.sum(axis=1, keepdims=True)
    max_delta = float(np.max(np.abs(phase0 - phase1)))
    if max_delta > 1e-10:
        raise ValueError(f"Expected one-phase/tied rows, but max phase delta is {max_delta:g}")
    return phase0


def proportional_reference(
    panel: pd.DataFrame, domains: list[str], panel_path: Path
) -> tuple[np.ndarray, float, float, int]:
    proportional = panel[panel["run_name"].eq("singleavg_baseline_proportional")]
    if len(proportional) != 1:
        raise ValueError("Expected exactly one singleavg_baseline_proportional row")
    natural = proportional.iloc[0][[f"phase_0_{domain}" for domain in domains]].astype(float).to_numpy()
    natural = natural / natural.sum()
    summary_path = panel_path.parent / "summary.json"
    if not summary_path.exists():
        raise ValueError(f"Missing proportional reference summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    ref_n = int(summary["proportional_reference_observation_count"])
    ref_mean = float(summary["proportional_reference_macro_mean"])
    ref_std = float(summary["proportional_reference_macro_std"])
    if ref_n != 11 or not np.isclose(ref_mean, float(proportional.iloc[0][MACRO_TARGET])):
        raise ValueError("Proportional target metadata is inconsistent with the fit panel")
    return natural, ref_mean, ref_std, ref_n


def count_rows(panel: pd.DataFrame) -> dict[str, int]:
    if "panel_source" not in panel.columns:
        raise ValueError("Panel is missing panel_source")
    return {
        "n_rows": len(panel),
        "n_signal_rows": int(
            panel["panel_source"].isin(["single_phase_qsplit_signal", "shared_stratified_baseline"]).sum()
        ),
        "n_deletion_rows": int(panel["panel_source"].eq("domain_deletion").sum()),
    }


def repeat_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim == 1:
        weights = weights[None, :]
    return np.stack([weights, weights], axis=1)


def simulated_epochs(weights: np.ndarray, token_counts: np.ndarray, target_budget: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim == 2:
        aggregate = weights
    elif weights.ndim == 3:
        aggregate = weights[:, 0, :]
    else:
        raise ValueError(f"Expected weights ndim 2 or 3, got {weights.ndim}")
    return float(target_budget) * aggregate / token_counts[None, :]


def kl_to_proportional(weights: np.ndarray, natural: np.ndarray) -> float:
    safe_w = np.clip(np.asarray(weights, dtype=float), 1e-12, 1.0)
    safe_p = np.clip(np.asarray(natural, dtype=float), 1e-12, 1.0)
    return float(np.sum(safe_w * (np.log(safe_w) - np.log(safe_p))))


def tv_to_proportional(weights: np.ndarray, natural: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(weights, dtype=float) - natural).sum())


def nearest_observed(
    panel: pd.DataFrame,
    observed_weights: np.ndarray,
    observed_target: np.ndarray,
    proposal: np.ndarray,
) -> tuple[str, float, float]:
    distances = 0.5 * np.abs(observed_weights - proposal[None, :]).sum(axis=1)
    idx = int(np.argmin(distances))
    return str(panel.iloc[idx]["run_name"]), float(observed_target[idx]), float(distances[idx])


def fit_olmix_components(
    panel: pd.DataFrame,
    weights: np.ndarray,
    *,
    huber_delta: float,
    fit_n_starts: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    components = paper_olmix.table9_component_order()
    targets = panel[components].astype(float).to_numpy()
    log_cs: list[float] = []
    coefficients: list[np.ndarray] = []
    train_predictions = np.zeros_like(targets, dtype=float)
    rows: list[dict[str, Any]] = []
    print(
        f"Fitting one-phase OLMix: {len(components)} components, delta={huber_delta:g}, starts={fit_n_starts}",
        flush=True,
    )
    for component_idx, component in enumerate(components, start=1):
        y = targets[:, component_idx - 1]
        log_c, coef, loss = base_olmix.fit_olmix_loglinear(
            weights,
            y,
            delta=huber_delta,
            seed=paper_olmix.FIT_SEED + component_idx,
            n_starts=fit_n_starts,
            verbose=False,
        )
        prediction = base_olmix.predict(log_c, coef, weights)
        train_predictions[:, component_idx - 1] = prediction
        rmse, mae, pearson, spearman = base_olmix.regression_metrics(y, prediction)
        log_cs.append(float(log_c))
        coefficients.append(np.asarray(coef, dtype=float))
        rows.append(
            {
                "component": component,
                "huber_delta": float(huber_delta),
                "fit_log_c": float(log_c),
                "fit_huber_loss": float(loss),
                "train_rmse": float(rmse),
                "train_mae": float(mae),
                "train_pearson": float(pearson),
                "train_spearman": float(spearman),
            }
        )
        if component_idx % 10 == 0 or component_idx == len(components):
            print(f"  OLMix component {component_idx}/{len(components)}", flush=True)
    return (
        np.asarray(log_cs, dtype=float),
        np.vstack(coefficients),
        pd.DataFrame(rows),
        train_predictions,
    )


def fit_dsp_model(
    panel: pd.DataFrame,
    weights: np.ndarray,
    domains: list[str],
    token_counts: np.ndarray,
    target_budget: int,
    *,
    linear_reg: float,
    maxiter: int,
    coarse_top_k: int,
    basin_hopping_iters: int,
) -> tuple[dsp.FittedDSPModel, pd.DataFrame, np.ndarray, np.ndarray, dict[str, float], dict[str, float]]:
    packet = dsp.PacketData(
        frame=panel.reset_index(drop=True),
        name_col="run_name",
        y=pd.to_numeric(panel[MACRO_TARGET], errors="raise").to_numpy(dtype=float),
        w=repeat_weights(weights),
        m=len(domains),
        c0=float(target_budget) / token_counts,
        c1=np.zeros(len(domains), dtype=float),
        domain_names=list(domains),
    )
    previous_reg = dsp.LINEAR_REG
    dsp.LINEAR_REG = float(linear_reg)
    try:
        print(f"Fitting one-phase DSP effective_exposure LINEAR_REG={linear_reg:g}", flush=True)
        model, tuning = dsp.fit_variant(
            packet,
            dsp.VARIANTS["effective_exposure"],
            maxiter=maxiter,
            coarse_top_k=coarse_top_k,
            basin_hopping_iters=basin_hopping_iters,
        )
    finally:
        dsp.LINEAR_REG = previous_reg
    train_pred = dsp.predict(model, packet.w)
    train_rmse, train_mae, train_pearson, train_spearman = base_olmix.regression_metrics(packet.y, train_pred)
    oof_pred = dsp.oof_predictions(packet, model)
    folds = base_olmix.kfold_indices(len(packet.y), n_splits=base_olmix.N_SPLITS, seed=base_olmix.CV_SEED)
    oof_metrics = base_olmix.predictive_diagnostics(packet.y, oof_pred, folds)
    train_metrics = {
        "rmse": float(train_rmse),
        "mae": float(train_mae),
        "pearson": float(train_pearson),
        "spearman": float(train_spearman),
    }
    return model, tuning, train_pred, oof_pred, train_metrics, {key: float(value) for key, value in oof_metrics.items()}


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    weights = np.exp(logits - np.max(logits))
    return weights / weights.sum()


def single_weights_to_logits(weights: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(weights, dtype=float), 1e-12, 1.0))


def optimize_dsp_kl(
    model: dsp.FittedDSPModel,
    natural: np.ndarray,
    *,
    kl_reg: float,
    starts: list[np.ndarray],
) -> tuple[np.ndarray, float, float, str]:
    def objective(logits: np.ndarray) -> float:
        weights = softmax(logits)
        prediction = float(dsp.predict(model, repeat_weights(weights))[0])
        return prediction + float(kl_reg) * kl_to_proportional(weights, natural)

    best: Any | None = None
    for start in starts:
        result = minimize(
            objective,
            single_weights_to_logits(start),
            method="L-BFGS-B",
            options={"maxiter": 900, "ftol": 1e-10, "maxls": 30},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    if best is None:
        raise RuntimeError("DSP KL optimization failed")
    weights = softmax(np.asarray(best.x, dtype=float))
    prediction = float(dsp.predict(model, repeat_weights(weights))[0])
    return weights, prediction, float(best.fun), str(best.message)


def raw_dsp_start_bank(
    model: dsp.FittedDSPModel,
    natural: np.ndarray,
    observed_weights: np.ndarray,
    observed_predictions: np.ndarray,
    *,
    num_random: int,
) -> list[np.ndarray]:
    starts: list[np.ndarray] = [natural]
    top_idx = np.argsort(observed_predictions)[: min(16, len(observed_predictions))]
    starts.extend(observed_weights[int(idx)] for idx in top_idx)
    raw_weights, _prediction, _regularized, _status = optimize_dsp_kl(
        model,
        natural,
        kl_reg=0.0,
        starts=starts,
    )
    starts.append(raw_weights)
    rng = np.random.default_rng(0)
    for _ in range(num_random):
        logits = rng.normal(0.0, 0.45, size=len(natural))
        starts.append(softmax(logits))
    return starts


def proposal_row(
    *,
    model_family: str,
    series: str,
    kl_reg: float,
    predicted_objective: float,
    regularized_objective: float,
    weights: np.ndarray,
    natural: np.ndarray,
    token_counts: np.ndarray,
    target_budget: int,
    panel: pd.DataFrame,
    observed_weights: np.ndarray,
    observed_target: np.ndarray,
    optimizer_status: str,
    huber_delta: float | None = None,
    linear_reg: float | None = None,
) -> dict[str, Any]:
    nearest_name, nearest_value, nearest_tv = nearest_observed(panel, observed_weights, observed_target, weights)
    epochs = simulated_epochs(weights[None, :], token_counts, target_budget)[0]
    ratios = weights / np.clip(natural, 1e-12, None)
    return {
        "model_family": model_family,
        "series": series,
        "huber_delta": huber_delta,
        "linear_reg": linear_reg,
        "kl_reg": float(kl_reg),
        "predicted_objective": float(predicted_objective),
        "regularized_objective": float(regularized_objective),
        "nearest_observed_run_name": nearest_name,
        "nearest_observed_value": nearest_value,
        "nearest_observed_tv": nearest_tv,
        "mean_phase_tv_to_proportional": tv_to_proportional(weights, natural),
        "max_epoch_multiplier": float(np.max(ratios)),
        "q95_epoch_multiplier": float(np.quantile(ratios, 0.95)),
        "max_simulated_epoch": float(np.max(epochs)),
        "q95_simulated_epoch": float(np.quantile(epochs, 0.95)),
        "max_weight": float(np.max(weights)),
        "min_weight": float(np.min(weights)),
        "optimizer_status": optimizer_status,
    }


def write_weights(
    output_dir: Path,
    variant: str,
    domains: list[str],
    weights: np.ndarray,
    natural: np.ndarray,
    token_counts: np.ndarray,
    target_budget: int,
) -> None:
    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    epochs = simulated_epochs(weights[None, :], token_counts, target_budget)[0]
    frame = pd.DataFrame(
        {
            "domain": domains,
            "proportional": natural,
            "phase_0_weight": weights,
            "phase_1_weight": weights,
            "aggregate_weight": weights,
            "available_tokens": token_counts,
            "simulated_epochs": epochs,
            "phase_0_epoch_multiplier": weights / np.clip(natural, 1e-12, None),
            "phase_1_epoch_multiplier": weights / np.clip(natural, 1e-12, None),
            "phase_0_delta": weights - natural,
            "phase_1_delta": weights - natural,
        }
    )
    frame["max_abs_delta"] = frame[["phase_0_delta", "phase_1_delta"]].abs().max(axis=1)
    frame.to_csv(variant_dir / "proposed_mixture_weights.csv", index=False)


def write_overlay_plots(output_dir: Path, overlay: pd.DataFrame) -> None:
    series_order = [
        "OLMix one-phase cap4",
        "DSP one-phase effective-exposure",
    ]
    colors = {
        "OLMix one-phase cap4": "#f97316",
        "DSP one-phase effective-exposure": "#2563eb",
    }
    for y_column, title, y_title, filename in (
        (
            "predicted_objective",
            "One-phase Table-9 macro KL sweep: predicted BPB",
            "Predicted Table-9 macro BPB (lower is better)",
            "table9_macro_kl_predicted_bpb_olmix_dsp_overlay.html",
        ),
        (
            "max_simulated_epoch",
            "One-phase Table-9 macro KL sweep: materialized epochs",
            "Max materialized epoch",
            "table9_macro_kl_epochs_olmix_dsp_overlay.html",
        ),
        (
            "mean_phase_tv_to_proportional",
            "One-phase Table-9 macro KL sweep: TV to proportional",
            "TV distance to proportional",
            "table9_macro_kl_tv_olmix_dsp_overlay.html",
        ),
    ):
        fig = go.Figure()
        for series in series_order:
            view = overlay[overlay["series"].eq(series)].sort_values("kl_reg")
            if view.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=view["kl_reg"],
                    y=view[y_column],
                    mode="lines+markers+text",
                    name=series,
                    text=[f"{value:.3g}" for value in view[y_column]],
                    textposition="top center",
                    line={"color": colors[series], "width": 3},
                    marker={"size": 9},
                    hovertemplate=(
                        "series=%{fullData.name}<br>KL=%{x:g}<br>"
                        f"{y_column}=%{{y:.6f}}<br>nearest=%{{customdata[0]}}<br>"
                        "nearest observed=%{customdata[1]:.6f}<extra></extra>"
                    ),
                    customdata=view[["nearest_observed_run_name", "nearest_observed_value"]].to_numpy(),
                )
            )
        fig.add_vline(x=0.05, line_dash="dash", line_color="#475569", annotation_text="KL=0.05")
        fig.update_layout(
            title=title,
            xaxis_title="KL coefficient",
            yaxis_title=y_title,
            xaxis_type="log" if overlay["kl_reg"].min() > 0 else None,
            template="plotly_white",
            width=1150,
            height=720,
        )
        fig.write_html(output_dir / filename, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_fit_scatter(
    output_dir: Path, panel: pd.DataFrame, actual: np.ndarray, train_pred: np.ndarray, oof_pred: np.ndarray
) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actual,
            y=train_pred,
            mode="markers",
            name="DSP train",
            marker={"size": 6, "color": "#93c5fd", "opacity": 0.65},
            text=panel["run_name"],
            hovertemplate="run=%{text}<br>actual=%{x:.6f}<br>train=%{y:.6f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=actual,
            y=oof_pred,
            mode="markers",
            name="DSP OOF",
            marker={"size": 8, "color": "#2563eb", "opacity": 0.78},
            text=panel["run_name"],
            hovertemplate="run=%{text}<br>actual=%{x:.6f}<br>oof=%{y:.6f}<extra></extra>",
        )
    )
    lo = float(np.nanmin(np.column_stack([actual, train_pred, oof_pred])))
    hi = float(np.nanmax(np.column_stack([actual, train_pred, oof_pred])))
    fig.add_trace(
        go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y=x", line={"dash": "dash", "color": "#64748b"})
    )
    fig.update_layout(
        title="One-phase Table-9 macro DSP fit",
        xaxis_title="Observed Table-9 macro BPB",
        yaxis_title="Predicted BPB",
        template="plotly_white",
        width=950,
        height=760,
    )
    fig.write_html(
        output_dir / "one_phase_dsp_table9_macro_fit_scatter.html", include_plotlyjs="cdn", config=PLOT_CONFIG
    )


def write_report(
    output_dir: Path,
    dsp_summary: DspFitSummary,
    component_summary: pd.DataFrame,
    overlay: pd.DataFrame,
) -> None:
    key_columns = [
        "series",
        "kl_reg",
        "predicted_objective",
        "regularized_objective",
        "max_simulated_epoch",
        "mean_phase_tv_to_proportional",
        "nearest_observed_run_name",
        "nearest_observed_value",
    ]
    lines = [
        "# One-phase OLMoBaseEval Table-9 model sweeps",
        "",
        "Panel: 240 one-phase qsplit rows plus 39 phase-constant domain-deletion controls. The proportional target is the 11-observation proportional mean.",
        "",
        "OLMix is paper-faithful at the objective granularity: one log-linear model per Table-9 component, optimized over the mean predicted component BPB with cap-4 repetition constraints.",
        "",
        "DSP is effective-exposure DSP fit directly to the Table-9 macro BPB. For a one-phase ablation, the DSP packet uses tied weights and total-budget exposure in `c0`; `c1` is zero, so no phase-asymmetric effect is available.",
        "",
        "## DSP fit",
        "",
        pd.DataFrame([asdict(dsp_summary)]).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## OLMix component fit summary",
        "",
        component_summary[["train_rmse", "train_spearman"]].describe().to_markdown(floatfmt=".6f"),
        "",
        "## KL sweep proposals",
        "",
        overlay[key_columns].to_markdown(index=False, floatfmt=".6f"),
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kl_grid = parse_float_list(args.kl_grid)
    panel = load_panel(args.panel)
    columns = phase_columns(panel)
    domains = domain_names(columns)
    weights = single_weights(panel, domains)
    natural, prop_mean, prop_std, prop_n = proportional_reference(panel, domains, args.panel)
    token_counts = base_olmix.load_domain_token_counts(domains)
    target_budget = base_olmix.load_target_budget()
    repetition_caps = base_olmix.repetition_weight_caps(
        token_counts,
        target_budget=target_budget,
        repetition_factor=base_olmix.REPETITION_FACTOR,
    )
    observed_target = panel[MACRO_TARGET].astype(float).to_numpy()
    row_counts = count_rows(panel)
    if row_counts["n_signal_rows"] != 241 or row_counts["n_deletion_rows"] != 39:
        raise ValueError(f"Unexpected one-phase panel composition: {row_counts}")

    log_cs, coefficients, component_summary, olmix_train_components = fit_olmix_components(
        panel,
        weights,
        huber_delta=float(args.olmix_huber_delta),
        fit_n_starts=int(args.olmix_fit_n_starts),
    )
    component_summary.to_csv(args.output_dir / "olmix_component_fit_summary.csv", index=False)
    olmix_macro_train = olmix_train_components.mean(axis=1)
    pd.DataFrame(
        {
            "run_name": panel["run_name"],
            "panel_source": panel["panel_source"],
            "observed_table9_macro_bpb": observed_target,
            "olmix_train_prediction": olmix_macro_train,
            "olmix_train_residual": olmix_macro_train - observed_target,
        }
    ).to_csv(args.output_dir / "olmix_macro_train_predictions.csv", index=False)

    dsp_model, dsp_tuning, dsp_train_pred, dsp_oof_pred, dsp_train_metrics, dsp_oof_metrics = fit_dsp_model(
        panel,
        weights,
        domains,
        token_counts,
        target_budget,
        linear_reg=float(args.dsp_linear_reg),
        maxiter=int(args.dsp_maxiter),
        coarse_top_k=int(args.dsp_coarse_top_k),
        basin_hopping_iters=int(args.dsp_basin_hopping_iters),
    )
    dsp_tuning.to_csv(args.output_dir / "dsp_tuning.csv", index=False)
    (args.output_dir / "dsp_model.json").write_text(
        json.dumps(
            dsp.model_to_json(dsp_model, {"target": MACRO_TARGET, "variant": "one_phase_effective_exposure"}),
            indent=2,
        )
    )
    pd.DataFrame(
        {
            "run_name": panel["run_name"],
            "panel_source": panel["panel_source"],
            "observed_table9_macro_bpb": observed_target,
            "dsp_train_prediction": dsp_train_pred,
            "dsp_oof_prediction": dsp_oof_pred,
            "dsp_train_residual": dsp_train_pred - observed_target,
            "dsp_oof_residual": dsp_oof_pred - observed_target,
        }
    ).to_csv(args.output_dir / "dsp_macro_fit_predictions.csv", index=False)
    write_fit_scatter(args.output_dir, panel, observed_target, dsp_train_pred, dsp_oof_pred)

    dsp_summary = DspFitSummary(
        model_family="DSP",
        variant="one_phase_effective_exposure",
        target_metric=MACRO_TARGET,
        n_rows=row_counts["n_rows"],
        n_signal_rows=row_counts["n_signal_rows"],
        n_deletion_rows=row_counts["n_deletion_rows"],
        n_proportional_reference_rows=prop_n,
        proportional_reference_mean=prop_mean,
        proportional_reference_std=prop_std,
        linear_reg=float(args.dsp_linear_reg),
        train_rmse=dsp_train_metrics["rmse"],
        train_mae=dsp_train_metrics["mae"],
        train_pearson=dsp_train_metrics["pearson"],
        train_spearman=dsp_train_metrics["spearman"],
        oof_rmse=dsp_oof_metrics["rmse"],
        oof_mae=dsp_oof_metrics["mae"],
        oof_pearson=dsp_oof_metrics["pearson"],
        oof_spearman=dsp_oof_metrics["spearman"],
        fold_mean_regret_at_1=dsp_oof_metrics["fold_mean_regret_at_1"],
        lower_tail_optimism=dsp_oof_metrics["lower_tail_optimism"],
        low_tail_rmse=dsp_oof_metrics["low_tail_rmse"],
        total_param_count=dsp_model.total_param_count,
        m_dependent_params_per_domain=dsp_model.m_dependent_params_per_domain,
    )
    pd.DataFrame([asdict(dsp_summary)]).to_csv(args.output_dir / "dsp_fit_summary.csv", index=False)

    dsp_starts = raw_dsp_start_bank(
        dsp_model,
        natural,
        weights,
        dsp_train_pred,
        num_random=int(args.dsp_raw_starts),
    )
    overlay_rows: list[dict[str, Any]] = []
    for kl_reg in kl_grid:
        print(f"Optimizing OLMix one-phase cap4 KL={kl_reg:g}", flush=True)
        olmix_two_phase, olmix_pred, olmix_reg, olmix_status = paper_olmix.solve_multi_single(
            log_cs,
            coefficients,
            natural=natural,
            kl_reg=float(kl_reg),
            repetition_caps=repetition_caps,
        )
        olmix_weights = np.asarray(olmix_two_phase[0], dtype=float)
        overlay_rows.append(
            proposal_row(
                model_family="OLMix",
                series="OLMix one-phase cap4",
                kl_reg=float(kl_reg),
                predicted_objective=float(olmix_pred),
                regularized_objective=float(olmix_reg),
                weights=olmix_weights,
                natural=natural,
                token_counts=token_counts,
                target_budget=target_budget,
                panel=panel,
                observed_weights=weights,
                observed_target=observed_target,
                optimizer_status=olmix_status,
                huber_delta=float(args.olmix_huber_delta),
            )
        )
        write_weights(
            args.output_dir,
            f"olmix_one_phase_cap4_delta{args.olmix_huber_delta:g}_kl{kl_reg:g}".replace(".", "p"),
            domains,
            olmix_weights,
            natural,
            token_counts,
            target_budget,
        )

        print(f"Optimizing DSP one-phase effective-exposure KL={kl_reg:g}", flush=True)
        dsp_weights, dsp_pred, dsp_reg, dsp_status = optimize_dsp_kl(
            dsp_model,
            natural,
            kl_reg=float(kl_reg),
            starts=dsp_starts,
        )
        overlay_rows.append(
            proposal_row(
                model_family="DSP",
                series="DSP one-phase effective-exposure",
                kl_reg=float(kl_reg),
                predicted_objective=float(dsp_pred),
                regularized_objective=float(dsp_reg),
                weights=dsp_weights,
                natural=natural,
                token_counts=token_counts,
                target_budget=target_budget,
                panel=panel,
                observed_weights=weights,
                observed_target=observed_target,
                optimizer_status=dsp_status,
                linear_reg=float(args.dsp_linear_reg),
            )
        )
        write_weights(
            args.output_dir,
            f"dsp_one_phase_effexp_linear_reg{args.dsp_linear_reg:g}_kl{kl_reg:g}".replace(".", "p"),
            domains,
            dsp_weights,
            natural,
            token_counts,
            target_budget,
        )

    overlay = pd.DataFrame(overlay_rows).sort_values(["series", "kl_reg"]).reset_index(drop=True)
    overlay.to_csv(args.output_dir / "one_phase_olmix_dsp_kl_sweep_summary.csv", index=False)
    overlay[overlay["model_family"].eq("OLMix")].to_csv(
        args.output_dir / "olmix_one_phase_kl_sweep_summary.csv", index=False
    )
    overlay[overlay["model_family"].eq("DSP")].to_csv(
        args.output_dir / "dsp_one_phase_kl_sweep_summary.csv", index=False
    )
    write_overlay_plots(args.output_dir, overlay)
    write_report(args.output_dir, dsp_summary, component_summary, overlay)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "input_panel": str(args.panel),
                "output_dir": str(args.output_dir),
                "target_metric": MACRO_TARGET,
                "row_counts": row_counts,
                "component_count": len(paper_olmix.table9_component_order()),
                "proportional_reference_n": prop_n,
                "proportional_reference_mean": prop_mean,
                "proportional_reference_std": prop_std,
                "olmix_huber_delta": float(args.olmix_huber_delta),
                "olmix_fit_n_starts": int(args.olmix_fit_n_starts),
                "olmix_repetition_factor": float(base_olmix.REPETITION_FACTOR),
                "dsp_variant": "effective_exposure",
                "dsp_linear_reg": float(args.dsp_linear_reg),
                "dsp_one_phase_exposure": "c0=target_budget/domain_tokens, c1=0, phase weights tied",
                "kl_grid": kl_grid,
                "artifacts": [
                    "one_phase_olmix_dsp_kl_sweep_summary.csv",
                    "olmix_component_fit_summary.csv",
                    "olmix_macro_train_predictions.csv",
                    "dsp_fit_summary.csv",
                    "dsp_macro_fit_predictions.csv",
                    "dsp_tuning.csv",
                    "dsp_model.json",
                    "table9_macro_kl_predicted_bpb_olmix_dsp_overlay.html",
                    "table9_macro_kl_epochs_olmix_dsp_overlay.html",
                    "table9_macro_kl_tv_olmix_dsp_overlay.html",
                    "report.md",
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(overlay[["series", "kl_reg", "predicted_objective", "max_simulated_epoch", "mean_phase_tv_to_proportional"]])
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
