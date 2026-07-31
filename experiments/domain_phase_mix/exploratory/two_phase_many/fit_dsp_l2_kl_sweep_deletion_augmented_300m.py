# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Sweep DSP linear-head L2 and KL-only proposal regularization."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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
    fit_dsp_vs_olmix_deletion_augmented_300m as dsp_compare,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as olmix,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "dsp_l2_kl_sweep_deletion_augmented_300m_20260625"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant", choices=sorted(dsp.VARIANTS), default="canonical")
    parser.add_argument("--linear-reg-values", default="1e-6,1e-5,1e-4,1e-3,1e-2")
    parser.add_argument("--kl-reg-values", default="0.01,0.025,0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--maxiter", type=int, default=dsp.FIT_MAXITER)
    parser.add_argument("--coarse-top-k", type=int, default=dsp.START_TOP_K)
    parser.add_argument("--basin-hopping-iters", type=int, default=1)
    parser.add_argument("--raw-optimum-starts", type=int, default=32)
    return parser.parse_args()


def softmax_pair(logits: np.ndarray, m: int) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    out = np.zeros((2, m), dtype=float)
    for phase_idx in range(2):
        phase_logits = logits[phase_idx * m : (phase_idx + 1) * m]
        weights = np.exp(phase_logits - np.max(phase_logits))
        out[phase_idx] = weights / weights.sum()
    return out


def weights_to_logits(weights: np.ndarray) -> np.ndarray:
    return np.log(np.clip(weights, 1e-12, 1.0)).reshape(-1)


def optimize_dsp_kl_only(
    model: dsp.FittedDSPModel,
    natural: np.ndarray,
    *,
    kl_reg: float,
    starts: list[np.ndarray],
) -> tuple[np.ndarray, float, str]:
    m = len(natural)

    def objective(logits: np.ndarray) -> float:
        weights = softmax_pair(logits, m)
        return dsp_compare.regularized_dsp_objective(model, weights, natural, kl_reg)

    best: Any | None = None
    for start_weights in starts:
        result = minimize(
            objective,
            weights_to_logits(start_weights),
            method="L-BFGS-B",
            options={"maxiter": 700, "ftol": 1e-10},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    if best is None:
        raise RuntimeError("KL-only DSP optimization failed")
    return softmax_pair(np.asarray(best.x, dtype=float), m), float(best.fun), str(best.message)


def fit_one_dsp_model(
    packet: dsp.PacketData,
    *,
    variant_key: str,
    linear_reg: float,
    maxiter: int,
    coarse_top_k: int,
    basin_hopping_iters: int,
) -> tuple[dsp.FittedDSPModel, dict[str, float], dict[str, float], np.ndarray, pd.DataFrame]:
    original_linear_reg = dsp.LINEAR_REG
    dsp.LINEAR_REG = float(linear_reg)
    try:
        model, tuning = dsp.fit_variant(
            packet,
            dsp.VARIANTS[variant_key],
            maxiter=maxiter,
            coarse_top_k=coarse_top_k,
            basin_hopping_iters=basin_hopping_iters,
        )
        train_pred = dsp.predict(model, packet.w)
        train_rmse, train_mae, train_pearson, train_spearman = dsp_compare.regression_metrics(packet.y, train_pred)
        oof_pred, folds = dsp_compare.fit_dsp_oof_predictions(packet, model)
        oof_metrics = olmix.predictive_diagnostics(packet.y, oof_pred, folds)
        train_metrics = {
            "rmse": float(train_rmse),
            "mae": float(train_mae),
            "pearson": float(train_pearson),
            "spearman": float(train_spearman),
        }
        return model, train_metrics, oof_metrics, oof_pred, tuning
    finally:
        dsp.LINEAR_REG = original_linear_reg


def model_selection_score(row: pd.Series) -> float:
    return float(row["oof_rmse"] + 0.5 * row["lower_tail_optimism"])


def write_l2_sweep_plot(output_dir: Path, summary: pd.DataFrame) -> None:
    x = summary["linear_reg"].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=summary["oof_rmse"], mode="lines+markers", name="OOF RMSE"))
    fig.add_trace(go.Scatter(x=x, y=summary["low_tail_rmse"], mode="lines+markers", name="low-tail RMSE"))
    fig.add_trace(go.Scatter(x=x, y=summary["lower_tail_optimism"], mode="lines+markers", name="lower-tail optimism"))
    fig.add_trace(go.Scatter(x=x, y=summary["fold_mean_regret_at_1"], mode="lines+markers", name="fold regret@1"))
    fig.update_layout(
        title="DSP linear-head L2 sweep",
        xaxis_title="LINEAR_REG",
        yaxis_title="BPB-scale diagnostic",
        template="plotly_white",
        width=1000,
        height=720,
    )
    fig.write_html(output_dir / "dsp_l2_fit_sweep_diagnostics.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x, y=summary["oof_spearman"], mode="lines+markers", name="OOF Spearman"))
    fig2.add_trace(go.Scatter(x=x, y=summary["train_spearman"], mode="lines+markers", name="train Spearman"))
    fig2.update_layout(
        title="DSP linear-head L2 sweep: rank fit",
        xaxis_title="LINEAR_REG",
        yaxis_title="Spearman",
        template="plotly_white",
        width=1000,
        height=640,
    )
    fig2.write_html(output_dir / "dsp_l2_fit_sweep_spearman.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_kl_sweep_plot(output_dir: Path, summary: pd.DataFrame) -> None:
    x = summary["kl_reg"].astype(str)
    fig0 = go.Figure()
    fig0.add_trace(
        go.Scatter(
            x=x,
            y=summary["predicted_objective"],
            mode="lines+markers+text",
            name="predicted BPB",
            text=[f"{value:.3f}" for value in summary["predicted_objective"]],
            textposition="top center",
            line={"color": "#2563eb", "width": 3},
            marker={"size": 10},
        )
    )
    kl_005 = summary.loc[np.isclose(summary["kl_reg"], 0.05), "predicted_objective"]
    if not kl_005.empty:
        fig0.add_hline(
            y=float(kl_005.iloc[0]),
            line_dash="dash",
            line_color="#475569",
            annotation_text="KL=0.05 reference",
            annotation_position="bottom right",
        )
    fig0.update_layout(
        title="DSP KL-only proposal sweep: predicted BPB",
        xaxis_title="KL coefficient",
        yaxis_title="Predicted Uncheatable BPB (lower is better)",
        template="plotly_white",
        width=1000,
        height=650,
    )
    fig0.write_html(output_dir / "dsp_kl_only_predicted_bpb_sweep.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=summary["predicted_objective"], mode="lines+markers", name="predicted BPB"))
    fig.add_trace(
        go.Scatter(x=x, y=summary["regularized_objective"], mode="lines+markers", name="regularized objective")
    )
    fig.update_layout(
        title="DSP KL-only proposal sweep: objective",
        xaxis_title="KL coefficient",
        yaxis_title="Objective",
        template="plotly_white",
        width=1000,
        height=650,
    )
    fig.write_html(output_dir / "dsp_kl_only_objective_sweep.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x, y=summary["max_simulated_epoch"], mode="lines+markers", name="max simulated epoch"))
    fig2.add_trace(go.Scatter(x=x, y=summary["q95_simulated_epoch"], mode="lines+markers", name="q95 simulated epoch"))
    fig2.add_hline(y=4.0, line_dash="dash", line_color="#444", annotation_text="cap-4 reference")
    fig2.update_layout(
        title="DSP KL-only proposal sweep: materialized epochs",
        xaxis_title="KL coefficient",
        yaxis_title="Simulated epochs",
        template="plotly_white",
        width=1000,
        height=650,
    )
    fig2.write_html(output_dir / "dsp_kl_only_epoch_sweep.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(x=x, y=summary["mean_phase_tv_to_proportional"], mode="lines+markers", name="TV to proportional")
    )
    fig3.update_layout(
        title="DSP KL-only proposal sweep: distance from proportional",
        xaxis_title="KL coefficient",
        yaxis_title="Mean phase TV",
        template="plotly_white",
        width=1000,
        height=600,
    )
    fig3.write_html(output_dir / "dsp_kl_only_tv_sweep.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_kl_mixture_heatmap(
    output_dir: Path, domains: list[str], variants: list[tuple[str, np.ndarray]], natural: np.ndarray
) -> None:
    rows = []
    reference = np.stack([natural, natural], axis=0)
    for label, weights in variants:
        ratios = weights / np.clip(reference, 1e-12, None)
        for phase_idx in range(2):
            rows.append(
                pd.DataFrame(
                    {
                        "domain": domains,
                        "variant": f"{label} p{phase_idx}",
                        "epoch_multiplier": ratios[phase_idx],
                    }
                )
            )
    frame = pd.concat(rows, ignore_index=True)
    pivot = frame.pivot(index="domain", columns="variant", values="epoch_multiplier").reindex(index=domains)
    z = np.log2(np.clip(pivot.to_numpy(dtype=float), 1e-9, None))
    text = np.vectorize(lambda value: f"{value:.1f}x")(pivot.to_numpy(dtype=float))
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=pivot.columns,
            y=pivot.index,
            colorscale="RdYlGn_r",
            zmid=0.0,
            text=text,
            texttemplate="%{text}",
            colorbar={"title": "log2 epoch multiplier"},
            hovertemplate="domain=%{y}<br>variant=%{x}<br>multiplier=%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="DSP KL-only proposal mixtures",
        xaxis_title="Proposal",
        yaxis_title="Domain",
        template="plotly_white",
        width=1050,
        height=1250,
    )
    fig.write_html(output_dir / "dsp_kl_only_mixture_heatmap.html", include_plotlyjs="cdn", config=PLOT_CONFIG)
    frame.to_csv(output_dir / "dsp_kl_only_mixture_weights_long.csv", index=False)


def write_report(output_dir: Path, fit_summary: pd.DataFrame, kl_summary: pd.DataFrame, selected_reg: float) -> None:
    lines = [
        "# DSP L2 and KL-only proposal sweep",
        "",
        "Panel: same 280-row deletion-augmented 300M Uncheatable BPB panel used for the OLMix and canonical DSP comparison.",
        "",
        "Two independent regularization questions are separated here:",
        "",
        "1. Fit regularization: `LINEAR_REG` in the DSP NNLS variable-projection linear head.",
        "2. Proposal regularization: KL to proportional during mixture optimization, with no explicit epoch cap.",
        "",
        f"Selected fit regularization for the KL-only proposal sweep: `{selected_reg:g}`.",
        "",
        "## L2 fit sweep",
        "",
        fit_summary[
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
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## KL-only proposal sweep",
        "",
        kl_summary[
            [
                "linear_reg",
                "kl_reg",
                "predicted_objective",
                "regularized_objective",
                "mean_phase_tv_to_proportional",
                "max_simulated_epoch",
                "q95_simulated_epoch",
                "nearest_observed_run_name",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "Interpretation should compare the KL-only maximum simulated epochs to the cap-4 reference. If KL-only needs a very large coefficient before max epochs become sane, then a hard cap or a stronger exposure-aware proposal prior is still justified.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    linear_regs = parse_float_list(args.linear_reg_values)
    kl_regs = parse_float_list(args.kl_reg_values)

    _signal, columns, domains, natural = olmix.load_raw_signal_panel()
    target_budget = olmix.load_target_budget()
    token_counts = olmix.load_domain_token_counts(domains)
    panel, metadata = olmix.build_uncheatable_panel(columns)
    packet = dsp_compare.build_dsp_packet(panel, columns, domains, token_counts, target_budget)
    reference_weights = np.stack([natural, natural], axis=0)

    fit_rows: list[dict[str, Any]] = []
    fitted_models: dict[float, dsp.FittedDSPModel] = {}
    fitted_oof: dict[float, np.ndarray] = {}
    for linear_reg in linear_regs:
        print(f"Fitting DSP with LINEAR_REG={linear_reg:g}", flush=True)
        model, train_metrics, oof_metrics, oof_pred, tuning = fit_one_dsp_model(
            packet,
            variant_key=str(args.variant),
            linear_reg=linear_reg,
            maxiter=int(args.maxiter),
            coarse_top_k=int(args.coarse_top_k),
            basin_hopping_iters=int(args.basin_hopping_iters),
        )
        fitted_models[linear_reg] = model
        fitted_oof[linear_reg] = oof_pred
        tuning.to_csv(output_dir / f"dsp_tuning_l2_{linear_reg:g}.csv", index=False)
        row = {
            "linear_reg": float(linear_reg),
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "train_pearson": train_metrics["pearson"],
            "train_spearman": train_metrics["spearman"],
            "oof_rmse": oof_metrics["rmse"],
            "oof_mae": oof_metrics["mae"],
            "oof_pearson": oof_metrics["pearson"],
            "oof_spearman": oof_metrics["spearman"],
            "fold_mean_regret_at_1": oof_metrics["fold_mean_regret_at_1"],
            "lower_tail_optimism": oof_metrics["lower_tail_optimism"],
            "low_tail_rmse": oof_metrics["low_tail_rmse"],
        }
        row["selection_score"] = float(row["oof_rmse"] + 0.5 * row["lower_tail_optimism"])
        fit_rows.append(row)

    fit_summary = pd.DataFrame(fit_rows).sort_values("linear_reg").reset_index(drop=True)
    fit_summary.to_csv(output_dir / "dsp_l2_fit_sweep_summary.csv", index=False)
    write_l2_sweep_plot(output_dir, fit_summary)

    selected_reg = float(fit_summary.iloc[int(np.argmin(fit_summary["selection_score"].to_numpy()))]["linear_reg"])
    selected_model = fitted_models[selected_reg]
    raw_result, raw_weights = dsp.optimize_raw(
        selected_model,
        num_starts=int(args.raw_optimum_starts),
        observed_start_weights=packet.w,
        max_observed_starts=80,
    )
    _ = raw_result
    starts = [
        reference_weights,
        raw_weights,
        dsp_compare.read_olmix_weights("uncheatable_eval_bpb_rep_cap4"),
        dsp_compare.read_olmix_weights("uncheatable_eval_bpb_single_simplex_tied_phases_rep_cap4"),
    ]
    train_pred = dsp.predict(selected_model, packet.w)
    train_rmse, train_mae, train_pearson, train_spearman = dsp_compare.regression_metrics(packet.y, train_pred)
    train_metrics = {
        "rmse": float(train_rmse),
        "mae": float(train_mae),
        "pearson": float(train_pearson),
        "spearman": float(train_spearman),
    }
    oof_pred = fitted_oof[selected_reg]
    folds = olmix.kfold_indices(len(packet.y), n_splits=olmix.N_SPLITS, seed=olmix.CV_SEED)
    oof_metrics = olmix.predictive_diagnostics(packet.y, oof_pred, folds)

    kl_rows: list[dict[str, Any]] = []
    kl_variants: list[tuple[str, np.ndarray]] = []
    for kl_reg in kl_regs:
        print(f"Optimizing KL-only proposal with lambda={kl_reg:g}", flush=True)
        weights, regularized_objective, status = optimize_dsp_kl_only(
            selected_model,
            natural,
            kl_reg=float(kl_reg),
            starts=starts,
        )
        summary = dsp_compare.proposal_summary(
            model_family=f"DSP {args.variant}",
            variant=f"dsp_{args.variant}_l2_{selected_reg:g}_kl_only_{kl_reg:g}",
            model=selected_model,
            packet=packet,
            panel=panel,
            metadata=metadata,
            weights=weights,
            train_metrics=train_metrics,
            oof_metrics=oof_metrics,
            natural=natural,
            token_counts=token_counts,
            target_budget=target_budget,
            repetition_factor=None,
            regularized_objective=float(regularized_objective),
            optimizer_status=status,
        )
        row = asdict(summary)
        row["linear_reg"] = float(selected_reg)
        row["kl_reg"] = float(kl_reg)
        kl_rows.append(row)
        if kl_reg in {0.05, 0.1, 0.2, 0.5, 1.0}:
            kl_variants.append((f"KL {kl_reg:g}", weights))
        dsp_compare.write_dsp_weights(
            output_dir,
            f"dsp_{args.variant}_l2_{selected_reg:g}_kl_only_{kl_reg:g}",
            weights,
            domains,
            natural,
            token_counts,
            target_budget,
        )

    kl_summary = pd.DataFrame(kl_rows).sort_values("kl_reg").reset_index(drop=True)
    kl_summary.to_csv(output_dir / "dsp_kl_only_proposal_sweep_summary.csv", index=False)
    write_kl_sweep_plot(output_dir, kl_summary)
    write_kl_mixture_heatmap(output_dir, domains, kl_variants, natural)
    write_report(output_dir, fit_summary, kl_summary, selected_reg)
    with (output_dir / "metadata.json").open("w") as f:
        json.dump(
            {
                "target_metric": olmix.UNCHEATABLE_TARGET,
                "variant": str(args.variant),
                "panel_rows": int(len(panel)),
                "phase_fractions": olmix.PHASE_FRACTIONS.tolist(),
                "linear_reg_values": linear_regs,
                "kl_reg_values": kl_regs,
                "selected_linear_reg": selected_reg,
                "selection_score": "oof_rmse + 0.5 * lower_tail_optimism",
                "n_proportional_reference_rows": int(metadata["n_proportional_reference_rows"]),
                "proportional_reference_mean": metadata["proportional_reference_mean"],
                "proportional_reference_std": metadata["proportional_reference_std"],
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print("L2 sweep")
    print(fit_summary.to_string(index=False))
    print("KL-only proposal sweep")
    print(
        kl_summary[
            [
                "linear_reg",
                "kl_reg",
                "predicted_objective",
                "regularized_objective",
                "mean_phase_tv_to_proportional",
                "max_simulated_epoch",
                "nearest_observed_run_name",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
