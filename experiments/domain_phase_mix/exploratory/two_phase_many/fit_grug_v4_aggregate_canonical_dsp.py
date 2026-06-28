# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit canonical DSP to the collaborator Grug-v4 aggregate target.

This keeps the collaborator gist's aggregate target construction fixed, but
replaces the gist's per-phase epoch-loss surrogate with the standalone DSP
implementation. The aggregate target is higher-is-better, while DSP is a
loss-like form, so this script fits DSP to ``-y_factor`` and reports predictions
back in higher-is-better units.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr

from experiments.domain_phase_mix.exploratory.two_phase_many.reproduce_collaborator_grug_v4_aggregate import (
    GRUG_DASHBOARD_WEIGHTS,
    OUTPUT_DIR,
    DatasetBundle,
    aggregate_targets,
    load_data,
    proportional_weights,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp

SENT_ZIP_INPUT_DIR = OUTPUT_DIR / "sent_zip_input/raw_metric_matrix_300m"
DEFAULT_RAW_CSV = SENT_ZIP_INPUT_DIR / "raw_metric_matrix_300m.csv"
DEFAULT_NOISE_CSV = SENT_ZIP_INPUT_DIR / "noise_baseline_run00097_300m.csv"
DEFAULT_METADATA_CSV = (
    Path(__file__).resolve().parents[4]
    / "experiments/domain_phase_mix/exploratory/two_phase_many/two_phase_many_epoch_metadata.csv"
)
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "canonical_dsp_sent_zip"
DEFAULT_VARIANT = "effective_exposure"
DEFAULT_V4_TRACK = "grug_moe_mix_v4"
DEFAULT_V4_HIDDEN_DIM = 512
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="sent_raw_metric_matrix_300m_zip")
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--noise-csv", type=Path, default=DEFAULT_NOISE_CSV)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant", choices=sorted(dsp.VARIANTS), default=DEFAULT_VARIANT)
    parser.add_argument("--maxiter", type=int, default=dsp.FIT_MAXITER)
    parser.add_argument("--coarse-top-k", type=int, default=dsp.START_TOP_K)
    parser.add_argument("--basin-hopping-iters", type=int, default=3)
    parser.add_argument("--optimum-starts", type=int, default=64)
    parser.add_argument("--max-observed-starts", type=int, default=64)
    parser.add_argument("--drop-incomplete-task-cols", action="store_true")
    return parser.parse_args()


def normalize(weights: np.ndarray) -> np.ndarray:
    """Normalize one simplex vector."""
    values = np.asarray(weights, dtype=float)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Cannot normalize non-positive weights")
    return values / total


def weights_to_packet(
    raw: pd.DataFrame,
    y_loss: np.ndarray,
    domains: list[str],
    w0: np.ndarray,
    w1: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
) -> dsp.PacketData:
    """Build a DSP packet from aggregate-target arrays."""
    frame = pd.DataFrame(
        {
            "run_name": raw["run_name"].astype(str),
            "run_id": raw["run_id"],
            "y_factor_loss_target": y_loss,
        }
    )
    weights = np.stack([w0, w1], axis=1)
    weights = dsp.normalize_weights(weights)
    return dsp.PacketData(
        frame=frame,
        name_col="run_name",
        y=np.asarray(y_loss, dtype=float),
        w=weights,
        m=len(domains),
        c0=np.asarray(c0, dtype=float),
        c1=np.asarray(c1, dtype=float),
        domain_names=list(domains),
    )


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute ordinary R^2."""
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")


def prediction_metrics(actual_y: np.ndarray, pred_y: np.ndarray, prefix: str) -> dict[str, float]:
    """Return fit metrics for higher-is-better aggregate predictions."""
    residual = pred_y - actual_y
    return {
        f"{prefix}_rmse": float(np.sqrt(np.mean(residual**2))),
        f"{prefix}_mae": float(np.mean(np.abs(residual))),
        f"{prefix}_r2": r2_score(actual_y, pred_y),
        f"{prefix}_pearson": float(pearsonr(actual_y, pred_y).statistic),
        f"{prefix}_spearman": float(spearmanr(actual_y, pred_y).statistic),
    }


def average_phase_tv(left: np.ndarray, right: np.ndarray) -> float:
    """Average total-variation distance across two phases."""
    return float(np.abs(left - right).sum() / (2.0 * left.shape[0]))


def entropy(weights: np.ndarray) -> float:
    """Shannon entropy."""
    clipped = np.clip(np.asarray(weights, dtype=float), 1e-300, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def phase_stats(weights: np.ndarray) -> dict[str, float]:
    """Summarize a two-phase mixture."""
    return {
        "phase0_support_gt_1e3": int(np.sum(weights[0] > 1e-3)),
        "phase1_support_gt_1e3": int(np.sum(weights[1] > 1e-3)),
        "phase0_entropy": entropy(weights[0]),
        "phase1_entropy": entropy(weights[1]),
        "phase0_effective_support": float(np.exp(entropy(weights[0]))),
        "phase1_effective_support": float(np.exp(entropy(weights[1]))),
        "phase0_max_weight": float(np.max(weights[0])),
        "phase1_max_weight": float(np.max(weights[1])),
    }


def read_dashboard_v4(domains: list[str]) -> np.ndarray | None:
    """Read dashboard v4 weights over the 39 training domains."""
    if not GRUG_DASHBOARD_WEIGHTS.exists():
        return None
    dashboard = pd.read_csv(GRUG_DASHBOARD_WEIGHTS)
    v4 = (
        dashboard[
            dashboard["track"].eq(DEFAULT_V4_TRACK)
            & dashboard["hidden_dim"].eq(DEFAULT_V4_HIDDEN_DIM)
            & dashboard["domain"].isin(domains)
        ]
        .pivot(index="domain", columns="phase", values="weight")
        .fillna(0.0)
    )
    if v4.empty:
        return None
    v4 = v4.reindex(domains).fillna(0.0)
    return np.stack(
        [
            normalize(v4["phase_0"].to_numpy(dtype=float)),
            normalize(v4["phase_1"].to_numpy(dtype=float)),
        ],
        axis=0,
    )


def read_reproduced_lcb(domains: list[str]) -> np.ndarray | None:
    """Read the previously reproduced collaborator LCB optimum if present."""
    path = OUTPUT_DIR / "sent_raw_metric_matrix_300m_zip/mixture_weights.csv"
    if not path.exists():
        return None
    weights = pd.read_csv(path)
    lcb = weights[weights["label"].eq("lcb_optimum")].set_index("domain")
    if lcb.empty:
        return None
    lcb = lcb.reindex(domains).fillna(0.0)
    return np.stack(
        [
            normalize(lcb["w0"].to_numpy(dtype=float)),
            normalize(lcb["w1"].to_numpy(dtype=float)),
        ],
        axis=0,
    )


def mixture_comparison(
    label: str,
    candidate: np.ndarray,
    reference: np.ndarray,
    pred_y_candidate: float,
    pred_y_reference: float,
) -> dict[str, float | str]:
    """Compare two mixtures."""
    flat_candidate = candidate.reshape(-1)
    flat_reference = reference.reshape(-1)
    return {
        "label": label,
        "mean_phase_tv": average_phase_tv(candidate, reference),
        "flat_phase_weight_correlation": float(np.corrcoef(flat_candidate, flat_reference)[0, 1]),
        "pred_y_factor": pred_y_candidate,
        "reference_pred_y_factor": pred_y_reference,
        "pred_delta_vs_reference": pred_y_candidate - pred_y_reference,
    }


def weights_frame(
    label: str,
    domains: list[str],
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    model: dsp.FittedDSPModel,
) -> pd.DataFrame:
    """Create a per-domain weight and parameter table."""
    tau = np.asarray(model.params.get("tau", np.zeros(len(domains))), dtype=float)
    rho = np.asarray(model.params["rho"], dtype=float)
    out = pd.DataFrame(
        {
            "label": label,
            "domain": domains,
            "phase_0_weight": weights[0],
            "phase_1_weight": weights[1],
            "total_weight": weights[0] + weights[1],
            "phase_0_epochs": weights[0] * c0,
            "phase_1_epochs": weights[1] * c1,
            "total_epochs": weights[0] * c0 + weights[1] * c1,
            "benefit_coef": model.benefit_coef,
            "penalty_coef": model.penalty_coef,
            "rho": rho,
            "tau": tau,
        }
    )
    return out.sort_values("total_epochs", ascending=False)


def observed_predictions_frame(packet: dsp.PacketData, model: dsp.FittedDSPModel, actual_y: np.ndarray) -> pd.DataFrame:
    """Write observed-row predictions in higher-is-better units."""
    pred_y = -dsp.predict(model, packet.w)
    out = packet.frame[["run_name", "run_id"]].copy()
    out["actual_y_factor"] = actual_y
    out["pred_y_factor"] = pred_y
    out["residual_pred_minus_actual"] = pred_y - actual_y
    out["actual_rank_desc"] = out["actual_y_factor"].rank(method="min", ascending=False)
    out["pred_rank_desc"] = out["pred_y_factor"].rank(method="min", ascending=False)
    return out.sort_values("pred_y_factor", ascending=False).reset_index(drop=True)


def write_mixture_plot(
    mixtures: dict[str, np.ndarray],
    domains: list[str],
    c0: np.ndarray,
    c1: np.ndarray,
    output: Path,
) -> None:
    """Plot mixture weights and effective epochs."""
    raw = mixtures["raw_dsp_optimum"]
    order = np.argsort(-(raw[0] * c0 + raw[1] * c1))
    names = [domains[index][:55] for index in order]
    colors = {
        "proportional": "rgba(24,119,242,0.85)",
        "dashboard_v4": "rgba(44,160,44,0.85)",
        "collaborator_lcb": "rgba(240,112,26,0.85)",
        "raw_dsp_optimum": "rgba(214,39,40,0.85)",
    }
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("phase weights", "effective epochs"))
    for label, weights in mixtures.items():
        color = colors.get(label, "rgba(80,80,80,0.75)")
        darker = color.replace("0.85", "1.0").replace("0.75", "1.0")
        fig.add_trace(
            go.Bar(x=weights[0][order], y=names, orientation="h", name=f"{label} phase 0", marker={"color": color}),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(x=weights[1][order], y=names, orientation="h", name=f"{label} phase 1", marker={"color": darker}),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=(weights[0] * c0)[order],
                y=names,
                orientation="h",
                showlegend=False,
                name=f"{label} phase 0",
                marker={"color": color},
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=(weights[1] * c1)[order],
                y=names,
                orientation="h",
                showlegend=False,
                name=f"{label} phase 1",
                marker={"color": darker},
            ),
            row=1,
            col=2,
        )
    fig.update_layout(
        template="plotly_white",
        barmode="group",
        height=max(720, 24 * len(domains)),
        width=1550,
        margin={"l": 330, "r": 30, "t": 75, "b": 90},
        title={"text": "Canonical DSP raw optimum for collaborator aggregate", "x": 0.5},
        legend={"orientation": "h", "yanchor": "top", "y": -0.06, "xanchor": "center", "x": 0.5},
    )
    fig.write_html(output, include_plotlyjs="cdn")


def write_report(output_dir: Path, summary: dict[str, Any], top_weights: pd.DataFrame) -> None:
    """Write a concise Markdown report."""
    top_lines = [
        (
            f"- `{row['domain']}`: w0 `{row['phase_0_weight']:.4f}`, "
            f"w1 `{row['phase_1_weight']:.4f}`, epochs `{row['total_epochs']:.3f}`."
        )
        for _, row in top_weights.head(12).iterrows()
    ]
    lines = [
        "# Canonical DSP Fit To Collaborator Aggregate",
        "",
        "Canonical form used here: `dsp_effective_exposure_penalty_nnls` unless overridden.",
        "The collaborator factor aggregate is higher-is-better, so the DSP loss form was fit to `-y_factor`.",
        "",
        "## Fit",
        "",
        f"- Rows: `{summary['fit_row_count']}`.",
        f"- Tasks in aggregate: `{summary['n_task_cols_used']}`.",
        f"- Domains: `{summary['n_domains']}`.",
        f"- Variant: `{summary['variant']}`.",
        f"- Parameter count: `{summary['total_param_count']}`.",
        (
            f"- Train RMSE / R2 / Spearman: `{summary['train_rmse']:.4f}` / "
            f"`{summary['train_r2']:.4f}` / `{summary['train_spearman']:.4f}`."
        ),
        (
            f"- OOF RMSE / R2 / Spearman: `{summary['oof_rmse']:.4f}` / "
            f"`{summary['oof_r2']:.4f}` / `{summary['oof_spearman']:.4f}`."
        ),
        "",
        "## Raw Optimum",
        "",
        f"- Predicted `y_factor`: `{summary['raw_pred_y_factor']:.4f}`.",
        f"- Predicted gain vs proportional: `{summary['raw_pred_delta_vs_proportional']:.4f}`.",
        (
            f"- Nearest observed row: `{summary['raw_nearest_observed_run_name']}` "
            f"at average phase TV `{summary['raw_nearest_observed_tv']:.4f}`."
        ),
        f"- Support > 1e-3: phase 0 `{summary['phase0_support_gt_1e3']}`, phase 1 `{summary['phase1_support_gt_1e3']}`.",
        (
            f"- Max phase weights: phase 0 `{summary['phase0_max_weight']:.4f}`, "
            f"phase 1 `{summary['phase1_max_weight']:.4f}`."
        ),
        "",
        "## Top Raw-Optimum Domains",
        "",
        *top_lines,
        "",
        "## Comparison",
        "",
    ]
    for row in summary["mixture_comparisons"]:
        lines.append(
            f"- `{row['label']}`: TV `{row['mean_phase_tv']:.4f}`, "
            f"corr `{row['flat_phase_weight_correlation']:.4f}`, "
            f"predicted delta `{row['pred_delta_vs_reference']:.4f}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- This is a raw unconstrained model optimum; it is not Pareto-guardrailed and can be extrapolative.",
            "- The fit target is exactly the reproduced collaborator factor aggregate from the sent matrix zip.",
            "- Comparison rows are prediction-space comparisons under this fitted DSP model, not observed evaluations.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    """Run the canonical DSP aggregate fit."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = DatasetBundle(
        name=args.dataset_name,
        raw_path=args.raw_csv,
        noise_path=args.noise_csv,
        metadata_path=args.metadata_csv,
        drop_incomplete_task_cols=args.drop_incomplete_task_cols,
    )
    print(f"loading aggregate data from {bundle.raw_path}", flush=True)
    data = load_data(bundle)
    raw = data["raw"]
    domains = data["domains"]
    w0 = data["w0"]
    w1 = data["w1"]
    c0 = data["c0"]
    c1 = data["c1"]
    z = data["z"]
    noise_share = data["noise_share"]
    assert isinstance(raw, pd.DataFrame)
    assert isinstance(domains, list)
    assert isinstance(w0, np.ndarray)
    assert isinstance(w1, np.ndarray)
    assert isinstance(c0, np.ndarray)
    assert isinstance(c1, np.ndarray)
    assert isinstance(z, np.ndarray)
    assert isinstance(noise_share, np.ndarray)
    aggregate = aggregate_targets(z, noise_share)
    y_factor = np.asarray(aggregate["y_factor"], dtype=float)
    y_loss = -y_factor
    variant = dsp.VARIANTS[args.variant]
    packet = weights_to_packet(raw, y_loss, domains, w0, w1, c0, c1)

    print(
        f"fitting {variant.name}: rows={len(y_factor)} domains={len(domains)} "
        f"maxiter={args.maxiter} starts_top_k={args.coarse_top_k}",
        flush=True,
    )
    model, tuning = dsp.fit_variant(
        packet,
        variant,
        maxiter=args.maxiter,
        coarse_top_k=args.coarse_top_k,
        basin_hopping_iters=args.basin_hopping_iters,
    )
    tuning.to_csv(args.output_dir / "tuning.csv", index=False)

    print("optimizing raw two-phase simplex optimum", flush=True)
    raw_result, raw_weights = dsp.optimize_raw(
        model,
        num_starts=args.optimum_starts,
        observed_start_weights=packet.w,
        max_observed_starts=args.max_observed_starts,
    )

    train_pred_y = -dsp.predict(model, packet.w)
    oof_pred_y = -dsp.oof_predictions(packet, model)
    observed_predictions_frame(packet, model, y_factor).to_csv(args.output_dir / "observed_predictions.csv", index=False)

    w0_prop, w1_prop = proportional_weights(c0, c1)
    proportional = np.stack([w0_prop, w1_prop], axis=0)
    v4 = read_dashboard_v4(domains)
    lcb = read_reproduced_lcb(domains)
    mixtures: dict[str, np.ndarray] = {
        "proportional": proportional,
        "raw_dsp_optimum": raw_weights,
    }
    if v4 is not None:
        mixtures["dashboard_v4"] = v4
    if lcb is not None:
        mixtures["collaborator_lcb"] = lcb

    pred_loss_by_label = {
        label: float(dsp.predict(model, weights[None, :, :])[0]) for label, weights in mixtures.items()
    }
    pred_y_by_label = {label: -value for label, value in pred_loss_by_label.items()}
    comparisons = [
        mixture_comparison(
            f"raw_dsp_optimum_vs_{label}",
            raw_weights,
            weights,
            pred_y_by_label["raw_dsp_optimum"],
            pred_y_by_label[label],
        )
        for label, weights in mixtures.items()
        if label != "raw_dsp_optimum"
    ]

    raw_distances = dsp.average_phase_tv_distance(packet.w, raw_weights[None, :, :])
    nearest_idx = int(np.argmin(raw_distances))
    top_weights = weights_frame("raw_dsp_optimum", domains, raw_weights, c0, c1, model)
    all_weight_frames = [weights_frame(label, domains, weights, c0, c1, model) for label, weights in mixtures.items()]
    pd.concat(all_weight_frames, ignore_index=True).to_csv(args.output_dir / "mixture_weights.csv", index=False)
    top_weights.to_csv(args.output_dir / "raw_optimum_weights.csv", index=False)
    pd.DataFrame(comparisons).to_csv(args.output_dir / "mixture_comparisons.csv", index=False)
    write_mixture_plot(mixtures, domains, c0, c1, args.output_dir / "mixture_plot.html")

    model_metrics = {
        "dataset": bundle.name,
        "raw_path": str(bundle.raw_path),
        "noise_path": str(bundle.noise_path),
        "metadata_path": str(bundle.metadata_path),
        "variant": model.variant.name,
        "variant_key": args.variant,
        "description": model.variant.description,
        "fit_row_count": len(y_factor),
        "n_task_cols_used": len(data["task_cols"]),
        "n_task_cols_all": len(data["task_cols_all"]),
        "n_incomplete_task_cols": len(data["incomplete_task_cols"]),
        "n_domains": len(domains),
        "target_correlations": aggregate["target_correlations"],
        "total_param_count": model.total_param_count,
        "m_dependent_params_per_domain": model.m_dependent_params_per_domain,
        "gamma": float(model.params.get("gamma", float("nan"))),
        "active_benefit_coef_count": int(np.sum(model.benefit_coef > EPS)),
        "active_penalty_coef_count": int(np.sum(model.penalty_coef > EPS)),
        "raw_pred_loss": float(raw_result.fun),
        "raw_pred_y_factor": -float(raw_result.fun),
        "raw_pred_delta_vs_proportional": pred_y_by_label["raw_dsp_optimum"] - pred_y_by_label["proportional"],
        "raw_nearest_observed_tv": float(raw_distances[nearest_idx]),
        "raw_nearest_observed_run_name": str(packet.frame.iloc[nearest_idx][packet.name_col]),
        "raw_nearest_observed_y_factor": float(y_factor[nearest_idx]),
        "raw_optimum_success": bool(raw_result.success),
        "raw_optimum_message": str(raw_result.message),
        "mixture_comparisons": comparisons,
        "pred_y_by_label": pred_y_by_label,
        **prediction_metrics(y_factor, train_pred_y, "train"),
        **prediction_metrics(y_factor, oof_pred_y, "oof"),
        **phase_stats(raw_weights),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(model_metrics, indent=2, sort_keys=True))
    pd.DataFrame([model_metrics]).drop(columns=["mixture_comparisons", "target_correlations", "pred_y_by_label"]).to_csv(
        args.output_dir / "summary.csv",
        index=False,
    )
    (args.output_dir / "model.json").write_text(json.dumps(dsp.model_to_json(model, model_metrics), indent=2))
    write_report(args.output_dir, model_metrics, top_weights)

    print(
        textwrap.dedent(
            f"""
            wrote {args.output_dir}
            variant={model.variant.name}
            rows={len(y_factor)} domains={len(domains)} params={model.total_param_count}
            train: rmse={model_metrics['train_rmse']:.4f} r2={model_metrics['train_r2']:.4f}
            train spearman={model_metrics['train_spearman']:.4f}
            oof:   rmse={model_metrics['oof_rmse']:.4f} r2={model_metrics['oof_r2']:.4f}
            oof spearman={model_metrics['oof_spearman']:.4f}
            raw optimum predicted y_factor={model_metrics['raw_pred_y_factor']:.4f}
            delta_vs_prop={model_metrics['raw_pred_delta_vs_proportional']:.4f}
            nearest observed={model_metrics['raw_nearest_observed_run_name']}
            nearest tv={model_metrics['raw_nearest_observed_tv']:.4f}
            """
        ).strip(),
        flush=True,
    )


if __name__ == "__main__":
    main()
