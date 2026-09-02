# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Compare canonical DSP against OLMix on the 300M deletion-augmented panel."""

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
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    fit_olmix_reference_deletion_augmented_300m as olmix,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "dsp_vs_olmix_deletion_augmented_300m_20260625"
OLMIX_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "olmix_reference_deletion_augmented_300m_20260625"
TARGET_NAME = "uncheatable_eval_bpb"
TARGET_METRIC = olmix.UNCHEATABLE_TARGET
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class ProposalSummary:
    model_family: str
    variant: str
    target_metric: str
    n_rows: int
    n_signal_rows: int
    n_deletion_rows: int
    n_proportional_reference_rows: int
    proportional_reference_mean: float | None
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
    predicted_objective: float
    regularized_objective: float
    proportional_actual: float | None
    proportional_predicted: float
    best_observed_run_name: str
    best_observed_value: float
    nearest_observed_run_name: str
    nearest_observed_value: float
    nearest_observed_mean_phase_tv: float
    mean_phase_tv_to_proportional: float
    max_epoch_multiplier: float
    q95_epoch_multiplier: float
    repetition_factor: float | None
    target_budget_tokens: int
    max_simulated_epoch: float
    q95_simulated_epoch: float
    max_repetition_cap_violation: float | None
    max_weight: float
    min_weight: float
    optimizer_status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--maxiter", type=int, default=dsp.FIT_MAXITER)
    parser.add_argument("--coarse-top-k", type=int, default=dsp.START_TOP_K)
    parser.add_argument("--basin-hopping-iters", type=int, default=3)
    parser.add_argument("--raw-optimum-starts", type=int, default=48)
    parser.add_argument("--cap-kl-reg", type=float, default=olmix.KL_REG)
    return parser.parse_args()


def build_dsp_packet(
    panel: pd.DataFrame,
    columns: list[str],
    domains: list[str],
    token_counts: np.ndarray,
    target_budget: int,
) -> dsp.PacketData:
    weights = panel[columns].astype(float).to_numpy().reshape(len(panel), 2, len(domains))
    weights = dsp.normalize_weights(weights)
    phase_epoch_multipliers = olmix.PHASE_FRACTIONS[:, None] * float(target_budget) / token_counts[None, :]
    return dsp.PacketData(
        frame=panel.reset_index(drop=True),
        name_col="run_name",
        y=pd.to_numeric(panel[TARGET_METRIC], errors="coerce").to_numpy(dtype=float),
        w=weights,
        m=len(domains),
        c0=phase_epoch_multipliers[0],
        c1=phase_epoch_multipliers[1],
        domain_names=list(domains),
    )


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


def regression_metrics(y: np.ndarray, y_hat: np.ndarray) -> tuple[float, float, float, float]:
    residual = y_hat - y
    rmse = float(np.sqrt(np.mean(residual * residual)))
    mae = float(np.mean(np.abs(residual)))
    pearson = float(pearsonr(y, y_hat).statistic) if np.std(y) > 0.0 and np.std(y_hat) > 0.0 else float("nan")
    spearman = float(spearmanr(y, y_hat).statistic) if np.std(y) > 0.0 and np.std(y_hat) > 0.0 else float("nan")
    return rmse, mae, pearson, spearman


def blended_to_cap(
    raw_weights: np.ndarray,
    reference_weights: np.ndarray,
    repetition_caps: np.ndarray,
) -> np.ndarray:
    lo = 0.0
    hi = 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        candidate = (1.0 - mid) * reference_weights + mid * raw_weights
        aggregate = olmix.aggregate_phase_weights(candidate)
        if np.all(aggregate <= repetition_caps + 1e-12):
            lo = mid
        else:
            hi = mid
    return dsp.normalize_weights(((1.0 - lo) * reference_weights + lo * raw_weights)[None, :, :])[0]


def read_olmix_weights(variant: str) -> np.ndarray:
    path = OLMIX_OUTPUT_DIR / variant / "proposed_mixture_weights.csv"
    frame = pd.read_csv(path)
    weights = frame[["phase_0_weight", "phase_1_weight"]].to_numpy(dtype=float).T
    return dsp.normalize_weights(weights[None, :, :])[0]


def regularized_dsp_objective(
    model: dsp.FittedDSPModel,
    weights: np.ndarray,
    natural: np.ndarray,
    kl_reg: float,
) -> float:
    prediction = float(dsp.predict(model, weights[None, :, :])[0])
    kl = olmix.weighted_multiclass_kl(weights, natural, olmix.PHASE_FRACTIONS)
    return prediction + float(kl_reg) * kl


def optimize_dsp_cap4(
    model: dsp.FittedDSPModel,
    packet: dsp.PacketData,
    raw_weights: np.ndarray,
    natural: np.ndarray,
    repetition_caps: np.ndarray,
    *,
    kl_reg: float,
) -> tuple[np.ndarray, float, str]:
    m = len(natural)
    reference = np.stack([natural, natural], axis=0)
    starts: list[np.ndarray] = [
        reference,
        read_olmix_weights("uncheatable_eval_bpb_rep_cap4"),
        read_olmix_weights("uncheatable_eval_bpb_single_simplex_tied_phases_rep_cap4"),
        blended_to_cap(raw_weights, reference, repetition_caps),
    ]
    observed_predictions = dsp.predict(model, packet.w)
    observed_sim_epochs = np.asarray(
        [olmix.simulated_epochs(weights, np.ones(m), target_budget=1) for weights in packet.w],
        dtype=float,
    )
    feasible_observed = np.where(np.max(observed_sim_epochs / repetition_caps[None, :], axis=1) <= 1.0 + 1e-8)[0]
    for idx in feasible_observed[np.argsort(observed_predictions[feasible_observed])[:8]]:
        starts.append(packet.w[int(idx)])

    def pack(weights: np.ndarray) -> np.ndarray:
        return np.asarray(weights, dtype=float).reshape(-1)

    def unpack(x: np.ndarray) -> np.ndarray:
        weights = np.clip(np.asarray(x, dtype=float).reshape(2, m), 1e-12, None)
        return weights / weights.sum(axis=1, keepdims=True)

    def objective(x: np.ndarray) -> float:
        return regularized_dsp_objective(model, unpack(x), natural, kl_reg)

    def eq_phase0(x: np.ndarray) -> float:
        return float(np.sum(x[:m]) - 1.0)

    def eq_phase1(x: np.ndarray) -> float:
        return float(np.sum(x[m:]) - 1.0)

    def cap_constraint(x: np.ndarray) -> np.ndarray:
        weights = np.asarray(x, dtype=float).reshape(2, m)
        return repetition_caps - olmix.aggregate_phase_weights(weights)

    constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": eq_phase0},
        {"type": "eq", "fun": eq_phase1},
        {"type": "ineq", "fun": cap_constraint},
    ]
    bounds = [(1e-12, 1.0) for _ in range(2 * m)]
    best_result: Any | None = None
    best_weights: np.ndarray | None = None
    for start in starts:
        result = minimize(
            objective,
            pack(start),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1500, "ftol": 1e-11, "disp": False},
        )
        weights = unpack(np.asarray(result.x, dtype=float))
        feasible = np.max(olmix.aggregate_phase_weights(weights) - repetition_caps) <= 1e-7
        if feasible and (best_result is None or objective(pack(weights)) < objective(pack(best_weights))):  # type: ignore[arg-type]
            best_result = result
            best_weights = weights
    if best_weights is None or best_result is None:
        raise RuntimeError("DSP cap-4 optimization failed to find a feasible solution")
    return best_weights, float(objective(pack(best_weights))), str(best_result.message)


def proposal_summary(
    *,
    model_family: str,
    variant: str,
    model: dsp.FittedDSPModel,
    packet: dsp.PacketData,
    panel: pd.DataFrame,
    metadata: dict[str, Any],
    weights: np.ndarray,
    train_metrics: dict[str, float],
    oof_metrics: dict[str, float],
    natural: np.ndarray,
    token_counts: np.ndarray,
    target_budget: int,
    repetition_factor: float | None,
    regularized_objective: float,
    optimizer_status: str,
) -> ProposalSummary:
    reference = np.stack([natural, natural], axis=0)
    proposed_value = float(dsp.predict(model, weights[None, :, :])[0])
    distances = olmix.mean_phase_tv(packet.w, weights)
    nearest_idx = int(np.argmin(distances))
    best_idx = int(np.argmin(packet.y))
    prop_rows = panel.loc[panel["run_name"].eq("baseline_proportional"), TARGET_METRIC]
    prop_actual = float(prop_rows.iloc[0]) if len(prop_rows) else None
    prop_pred = float(dsp.predict(model, reference[None, :, :])[0])
    ratios = weights / np.clip(reference, 1e-12, None)
    sim_epochs = olmix.simulated_epochs(weights, token_counts, target_budget=target_budget)
    cap_violation = None if repetition_factor is None else float(np.max(sim_epochs - float(repetition_factor)))
    return ProposalSummary(
        model_family=model_family,
        variant=variant,
        target_metric=TARGET_METRIC,
        n_rows=int(len(panel)),
        n_signal_rows=int(panel["panel_source"].eq("qsplit_signal").sum()),
        n_deletion_rows=int(panel["panel_source"].eq("domain_deletion").sum()),
        n_proportional_reference_rows=int(metadata.get("n_proportional_reference_rows", 0)),
        proportional_reference_mean=metadata.get("proportional_reference_mean"),
        proportional_reference_std=metadata.get("proportional_reference_std"),
        train_rmse=float(train_metrics["rmse"]),
        train_mae=float(train_metrics["mae"]),
        train_pearson=float(train_metrics["pearson"]),
        train_spearman=float(train_metrics["spearman"]),
        oof_rmse=float(oof_metrics["rmse"]),
        oof_mae=float(oof_metrics["mae"]),
        oof_pearson=float(oof_metrics["pearson"]),
        oof_spearman=float(oof_metrics["spearman"]),
        fold_mean_regret_at_1=float(oof_metrics["fold_mean_regret_at_1"]),
        lower_tail_optimism=float(oof_metrics["lower_tail_optimism"]),
        low_tail_rmse=float(oof_metrics["low_tail_rmse"]),
        predicted_objective=proposed_value,
        regularized_objective=regularized_objective,
        proportional_actual=prop_actual,
        proportional_predicted=prop_pred,
        best_observed_run_name=str(panel.iloc[best_idx]["run_name"]),
        best_observed_value=float(packet.y[best_idx]),
        nearest_observed_run_name=str(panel.iloc[nearest_idx]["run_name"]),
        nearest_observed_value=float(packet.y[nearest_idx]),
        nearest_observed_mean_phase_tv=float(distances[nearest_idx]),
        mean_phase_tv_to_proportional=float(0.5 * np.abs(weights - reference).sum(axis=1).mean()),
        max_epoch_multiplier=float(np.max(ratios)),
        q95_epoch_multiplier=float(np.quantile(ratios, 0.95)),
        repetition_factor=repetition_factor,
        target_budget_tokens=int(target_budget),
        max_simulated_epoch=float(np.max(sim_epochs)),
        q95_simulated_epoch=float(np.quantile(sim_epochs, 0.95)),
        max_repetition_cap_violation=cap_violation,
        max_weight=float(np.max(weights)),
        min_weight=float(np.min(weights)),
        optimizer_status=optimizer_status,
    )


def write_dsp_weights(
    output_dir: Path,
    variant: str,
    weights: np.ndarray,
    domains: list[str],
    natural: np.ndarray,
    token_counts: np.ndarray,
    target_budget: int,
) -> None:
    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    ratios = weights / np.clip(np.stack([natural, natural], axis=0), 1e-12, None)
    sim_epochs = olmix.simulated_epochs(weights, token_counts, target_budget=target_budget)
    frame = pd.DataFrame(
        {
            "domain": domains,
            "proportional": natural,
            "phase_0_weight": weights[0],
            "phase_1_weight": weights[1],
            "aggregate_weight": olmix.aggregate_phase_weights(weights),
            "available_tokens": token_counts,
            "simulated_epochs": sim_epochs,
            "phase_0_epoch_multiplier": ratios[0],
            "phase_1_epoch_multiplier": ratios[1],
            "phase_0_delta": weights[0] - natural,
            "phase_1_delta": weights[1] - natural,
        }
    )
    frame["max_abs_delta"] = frame[["phase_0_delta", "phase_1_delta"]].abs().max(axis=1)
    frame.to_csv(variant_dir / "proposed_mixture_weights.csv", index=False)


def olmix_summary_rows() -> pd.DataFrame:
    frame = pd.read_csv(OLMIX_OUTPUT_DIR / "summary.csv")
    keep = frame[
        frame["target_name"].isin(
            [
                "uncheatable_eval_bpb",
                "uncheatable_eval_bpb_rep_cap4",
                "uncheatable_eval_bpb_single_simplex_tied_phases_rep_cap4",
            ]
        )
    ].copy()
    keep["model_family"] = "OLMix"
    keep["variant"] = keep["target_name"]
    keep["optimizer_status"] = keep["cvxpy_status"]
    return keep


def write_fit_scatter(
    output_dir: Path,
    panel: pd.DataFrame,
    y: np.ndarray,
    dsp_train_pred: np.ndarray,
    dsp_oof_pred: np.ndarray,
) -> None:
    olmix_pred_path = OLMIX_OUTPUT_DIR / "uncheatable_eval_bpb_rep_cap4" / "fit_panel_predictions.csv"
    olmix_pred = pd.read_csv(olmix_pred_path)
    plot_frame = panel[["run_name", "panel_source"]].copy()
    plot_frame["actual"] = y
    plot_frame["DSP train"] = dsp_train_pred
    plot_frame["DSP OOF"] = dsp_oof_pred
    plot_frame = plot_frame.merge(
        olmix_pred[["run_name", "olmix_prediction", "olmix_oof_prediction"]],
        on="run_name",
        how="left",
        validate="one_to_one",
    )
    plot_frame = plot_frame.rename(columns={"olmix_prediction": "OLMix train", "olmix_oof_prediction": "OLMix OOF"})

    fig = go.Figure()
    colors = {
        "DSP train": "#1f77b4",
        "DSP OOF": "#08519c",
        "OLMix train": "#ff7f0e",
        "OLMix OOF": "#a63603",
    }
    for column in ("DSP OOF", "OLMix OOF", "DSP train", "OLMix train"):
        fig.add_trace(
            go.Scatter(
                x=plot_frame["actual"],
                y=plot_frame[column],
                mode="markers",
                marker={"size": 8 if "OOF" in column else 6, "opacity": 0.82 if "OOF" in column else 0.45},
                marker_color=colors[column],
                name=column,
                text=plot_frame["run_name"],
                hovertemplate="run=%{text}<br>actual=%{x:.6f}<br>pred=%{y:.6f}<extra></extra>",
            )
        )
    lo = float(np.nanmin(plot_frame[["actual", "DSP OOF", "OLMix OOF"]].to_numpy()))
    hi = float(np.nanmax(plot_frame[["actual", "DSP OOF", "OLMix OOF"]].to_numpy()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line={"dash": "dash", "color": "#555"}, name="y=x"))
    fig.update_layout(
        title="Uncheatable BPB: DSP vs OLMix fit diagnostics",
        xaxis_title="Observed BPB",
        yaxis_title="Predicted BPB",
        template="plotly_white",
        width=1000,
        height=760,
    )
    fig.write_html(output_dir / "dsp_vs_olmix_fit_scatter.html", include_plotlyjs="cdn", config=PLOT_CONFIG)
    plot_frame.to_csv(output_dir / "dsp_vs_olmix_fit_predictions.csv", index=False)


def write_mixture_comparison(output_dir: Path, domains: list[str], natural: np.ndarray) -> None:
    rows: list[pd.DataFrame] = []
    for label, path in (
        ("OLMix two-phase cap4", OLMIX_OUTPUT_DIR / "uncheatable_eval_bpb_rep_cap4" / "proposed_mixture_weights.csv"),
        (
            "OLMix single tied cap4",
            OLMIX_OUTPUT_DIR
            / "uncheatable_eval_bpb_single_simplex_tied_phases_rep_cap4"
            / "proposed_mixture_weights.csv",
        ),
        ("DSP raw", output_dir / "dsp_canonical_raw" / "proposed_mixture_weights.csv"),
        ("DSP cap4 KL", output_dir / "dsp_canonical_rep_cap4_kl0p05" / "proposed_mixture_weights.csv"),
    ):
        frame = pd.read_csv(path)
        for phase in ("phase_0", "phase_1"):
            rows.append(
                pd.DataFrame(
                    {
                        "domain": frame["domain"],
                        "variant": f"{label} {phase[-1]}",
                        "epoch_multiplier": frame[f"{phase}_epoch_multiplier"],
                        "weight": frame[f"{phase}_weight"],
                    }
                )
            )
    comparison = pd.concat(rows, ignore_index=True)
    comparison.to_csv(output_dir / "dsp_vs_olmix_mixture_weights_long.csv", index=False)

    pivot = comparison.pivot(index="domain", columns="variant", values="epoch_multiplier").reindex(index=domains)
    z = np.log2(np.clip(pivot.to_numpy(dtype=float), 1e-9, None))
    text = np.vectorize(lambda value: f"{value:.1f}x")(pivot.to_numpy(dtype=float))
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=pivot.columns,
            y=pivot.index,
            colorscale="RdYlGn_r",
            zmid=0.0,
            colorbar={"title": "log2 epoch multiplier"},
            text=text,
            texttemplate="%{text}",
            hovertemplate="domain=%{y}<br>variant=%{x}<br>multiplier=%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Uncheatable BPB proposal mixtures: DSP vs OLMix",
        xaxis_title="Proposal",
        yaxis_title="Domain",
        template="plotly_white",
        width=1120,
        height=1280,
    )
    fig.write_html(output_dir / "dsp_vs_olmix_mixture_heatmap.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    wide = comparison.pivot(index="domain", columns="variant", values="epoch_multiplier")
    wide["max_abs_log2_multiplier"] = np.max(np.abs(np.log2(np.clip(wide.to_numpy(dtype=float), 1e-9, None))), axis=1)
    top = wide.sort_values("max_abs_log2_multiplier", ascending=False).head(30).iloc[::-1]
    fig2 = go.Figure()
    for column in [col for col in top.columns if col != "max_abs_log2_multiplier"]:
        fig2.add_trace(go.Bar(y=top.index, x=top[column], orientation="h", name=str(column)))
    fig2.add_vline(x=1.0, line_dash="dash", line_color="#444")
    fig2.update_layout(
        title="Largest exposure changes: DSP vs OLMix",
        xaxis_title="Epoch multiplier relative to proportional",
        yaxis_title="Domain",
        template="plotly_white",
        width=1250,
        height=980,
        barmode="group",
    )
    fig2.write_html(output_dir / "dsp_vs_olmix_epoch_multipliers.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(output_dir: Path, summary: pd.DataFrame) -> None:
    key_columns = [
        "model_family",
        "variant",
        "oof_spearman",
        "oof_rmse",
        "fold_mean_regret_at_1",
        "lower_tail_optimism",
        "low_tail_rmse",
        "predicted_objective",
        "regularized_objective",
        "proportional_actual",
        "mean_phase_tv_to_proportional",
        "max_simulated_epoch",
        "nearest_observed_run_name",
    ]
    lines = [
        "# DSP vs OLMix on deletion-augmented 300M Uncheatable BPB",
        "",
        "Fit panel: 241 ex-ante 300M qsplit/signal rows plus 39 domain-deletion rows.",
        "",
        "The `baseline_olmix_loglinear_uncheatable_bpb` adaptive row is excluded. The proportional target is the mean of 11 proportional observations.",
        "",
        "Diagnostic definitions match the OLMix reference artifact:",
        "",
        "- `oof_rmse` and `oof_spearman`: 5-fold out-of-fold predictions.",
        "- `fold_mean_regret_at_1`: held-out fold regret from selecting the lowest OOF-predicted BPB row within each fold.",
        "- `lower_tail_optimism`: mean `max(actual - predicted, 0)` over the lowest predicted 15% tail, with at least 5 rows.",
        "- `low_tail_rmse`: RMSE over the same lowest predicted tail.",
        "",
        summary[key_columns].to_markdown(index=False, floatfmt=".6f"),
        "",
        "DSP uses the canonical phase-benefit plus log-softplus-squared saturation penalty form from `standalone_code/dsp_exact.py`.",
        "",
        "The deployable DSP proposal shown here uses the same aggregate simulated-epoch cap of 4 and KL coefficient 0.05 as the OLMix cap-4 comparison.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _signal, columns, domains, natural = olmix.load_raw_signal_panel()
    target_budget = olmix.load_target_budget()
    token_counts = olmix.load_domain_token_counts(domains)
    repetition_caps = olmix.repetition_weight_caps(
        token_counts,
        target_budget=target_budget,
        repetition_factor=olmix.REPETITION_FACTOR,
    )
    panel, metadata = olmix.build_uncheatable_panel(columns)
    packet = build_dsp_packet(panel, columns, domains, token_counts, target_budget)

    model, tuning = dsp.fit_variant(
        packet,
        dsp.VARIANTS["canonical"],
        maxiter=int(args.maxiter),
        coarse_top_k=int(args.coarse_top_k),
        basin_hopping_iters=int(args.basin_hopping_iters),
    )
    tuning.to_csv(args.output_dir / "dsp_tuning.csv", index=False)
    train_pred = dsp.predict(model, packet.w)
    train_rmse, train_mae, train_pearson, train_spearman = regression_metrics(packet.y, train_pred)
    dsp_oof, folds = fit_dsp_oof_predictions(packet, model)
    oof_metrics = olmix.predictive_diagnostics(packet.y, dsp_oof, folds)
    train_metrics = {
        "rmse": train_rmse,
        "mae": train_mae,
        "pearson": train_pearson,
        "spearman": train_spearman,
    }

    raw_result, raw_weights = dsp.optimize_raw(
        model,
        num_starts=int(args.raw_optimum_starts),
        observed_start_weights=packet.w,
        max_observed_starts=80,
    )
    raw_regularized = regularized_dsp_objective(model, raw_weights, natural, float(args.cap_kl_reg))
    cap_weights, cap_regularized, cap_status = optimize_dsp_cap4(
        model,
        packet,
        raw_weights,
        natural,
        repetition_caps,
        kl_reg=float(args.cap_kl_reg),
    )

    raw_summary = proposal_summary(
        model_family="DSP",
        variant="dsp_canonical_raw",
        model=model,
        packet=packet,
        panel=panel,
        metadata=metadata,
        weights=raw_weights,
        train_metrics=train_metrics,
        oof_metrics=oof_metrics,
        natural=natural,
        token_counts=token_counts,
        target_budget=target_budget,
        repetition_factor=None,
        regularized_objective=raw_regularized,
        optimizer_status=str(raw_result.message),
    )
    cap_summary = proposal_summary(
        model_family="DSP",
        variant="dsp_canonical_rep_cap4_kl0p05",
        model=model,
        packet=packet,
        panel=panel,
        metadata=metadata,
        weights=cap_weights,
        train_metrics=train_metrics,
        oof_metrics=oof_metrics,
        natural=natural,
        token_counts=token_counts,
        target_budget=target_budget,
        repetition_factor=olmix.REPETITION_FACTOR,
        regularized_objective=cap_regularized,
        optimizer_status=cap_status,
    )

    write_dsp_weights(args.output_dir, raw_summary.variant, raw_weights, domains, natural, token_counts, target_budget)
    write_dsp_weights(args.output_dir, cap_summary.variant, cap_weights, domains, natural, token_counts, target_budget)
    observed = panel[["run_name", "source_experiment", "panel_source", TARGET_METRIC]].copy()
    observed["dsp_prediction"] = train_pred
    observed["dsp_oof_prediction"] = dsp_oof
    observed["residual"] = train_pred - packet.y
    observed["oof_residual"] = dsp_oof - packet.y
    observed.to_csv(args.output_dir / "dsp_fit_panel_predictions.csv", index=False)

    model_payload = dsp.model_to_json(model, {"target": TARGET_METRIC, "variant": "canonical"})
    (args.output_dir / "dsp_model.json").write_text(json.dumps(model_payload, indent=2))
    with (args.output_dir / "dsp_summary.json").open("w") as f:
        json.dump([asdict(raw_summary), asdict(cap_summary)], f, indent=2, sort_keys=True)

    dsp_rows = pd.DataFrame([asdict(raw_summary), asdict(cap_summary)])
    olmix_rows = olmix_summary_rows()
    summary = pd.concat([olmix_rows, dsp_rows], ignore_index=True, sort=False)
    summary.to_csv(args.output_dir / "dsp_vs_olmix_summary.csv", index=False)
    write_fit_scatter(args.output_dir, panel, packet.y, train_pred, dsp_oof)
    write_mixture_comparison(args.output_dir, domains, natural)
    write_report(args.output_dir, summary)
    print(
        summary[
            ["model_family", "variant", "oof_spearman", "oof_rmse", "predicted_objective", "max_simulated_epoch"]
        ].to_string(index=False)
    )
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
