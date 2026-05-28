# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit canonical DSP after holding out weakly predictable factor metrics.

This is a pruning variant of the collaborator Grug-v4 aggregate reproduction:
construct the 5-factor aggregate using only metrics marked ``predictable`` by
the DSP-readiness audit, then fit the canonical effective-exposure DSP model to
that pruned aggregate.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.fit_grug_v4_aggregate_canonical_dsp import (
    DEFAULT_METADATA_CSV,
    DEFAULT_NOISE_CSV,
    DEFAULT_RAW_CSV,
    DEFAULT_VARIANT,
    EPS,
    mixture_comparison,
    observed_predictions_frame,
    phase_stats,
    prediction_metrics,
    read_dashboard_v4,
    read_reproduced_lcb,
    weights_frame,
    weights_to_packet,
    write_mixture_plot,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.reproduce_collaborator_grug_v4_aggregate import (
    DatasetBundle,
    aggregate_targets,
    load_data,
    proportional_weights,
    write_factor_loadings_plot,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp

PRUNING_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "reference_outputs/predictable_metric_factor_pruning_20260527"
)
DEFAULT_SPLIT_CSV = PRUNING_OUTPUT_DIR / "metric_predictable_vs_heldout.csv"
DEFAULT_OUTPUT_DIR = PRUNING_OUTPUT_DIR / "heldout5_canonical_dsp"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="sent_raw_metric_matrix_300m_zip_predictable_only")
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--noise-csv", type=Path, default=DEFAULT_NOISE_CSV)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--readiness-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant", choices=sorted(dsp.VARIANTS), default=DEFAULT_VARIANT)
    parser.add_argument("--maxiter", type=int, default=dsp.FIT_MAXITER)
    parser.add_argument("--coarse-top-k", type=int, default=dsp.START_TOP_K)
    parser.add_argument("--basin-hopping-iters", type=int, default=3)
    parser.add_argument("--optimum-starts", type=int, default=64)
    parser.add_argument("--max-observed-starts", type=int, default=64)
    parser.add_argument("--drop-incomplete-task-cols", action="store_true")
    return parser.parse_args()


def task_split(readiness_csv: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return readable split tables and metric lists."""
    split = pd.read_csv(readiness_csv)
    required = {"metric", "readiness_bucket"}
    missing = sorted(required - set(split.columns))
    if missing:
        raise ValueError(f"Readiness CSV missing required columns: {missing}")
    kept = split.loc[split["readiness_bucket"].eq("predictable"), "metric"].astype(str).tolist()
    heldout = split.loc[~split["readiness_bucket"].eq("predictable"), "metric"].astype(str).tolist()
    if not kept:
        raise ValueError("Readiness CSV has no predictable metrics")
    if not heldout:
        raise ValueError("Readiness CSV has no held-out metrics")
    return split, kept, heldout


def subset_data(data: dict[str, object], kept_metrics: list[str]) -> dict[str, object]:
    """Subset aggregate arrays to the predictable metric set."""
    task_cols = list(data["task_cols"])
    task_signs = np.asarray(data["task_signs"], dtype=float)
    z = np.asarray(data["z"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    noise_share = np.asarray(data["noise_share"], dtype=float)
    kept_set = set(kept_metrics)
    unknown = sorted(kept_set - set(task_cols))
    if unknown:
        raise ValueError("Predictable metric split references metrics absent from loaded data:\n" + "\n".join(unknown))
    keep_indices = np.asarray([index for index, metric in enumerate(task_cols) if metric in kept_set], dtype=int)
    return {
        **data,
        "task_cols": [task_cols[index] for index in keep_indices],
        "task_signs": task_signs[keep_indices],
        "x": x[:, keep_indices],
        "z": z[:, keep_indices],
        "noise_share": noise_share[keep_indices],
        "task_cols_all_before_pruning": task_cols,
        "n_pruned_task_cols": int(len(task_cols) - len(keep_indices)),
    }


def write_factor_tables(
    output_dir: Path,
    task_cols: list[str],
    aggregate: dict[str, object],
    split: pd.DataFrame,
) -> None:
    """Write factor score/loading artifacts for the pruned aggregate."""
    factor_loadings = np.asarray(aggregate["factor_loadings"], dtype=float)
    factor_scores = np.asarray(aggregate["factor_scores"], dtype=float)
    loadings = pd.DataFrame(
        factor_loadings,
        columns=[f"F{index + 1}" for index in range(factor_loadings.shape[1])],
    )
    loadings.insert(0, "metric", task_cols)
    readiness_cols = [
        column
        for column in [
            "metric",
            "readiness_bucket",
            "dsp_oof_r",
            "dsp_oof_spearman",
            "proportional_signal_to_noise",
            "dominant_factor",
            "dominant_loading",
        ]
        if column in split.columns
    ]
    loadings = loadings.merge(split[readiness_cols], on="metric", how="left")
    loadings.to_csv(output_dir / "pruned_factor_loadings.csv", index=False)
    pd.DataFrame(
        factor_scores,
        columns=[f"F{index + 1}" for index in range(factor_scores.shape[1])],
    ).to_csv(output_dir / "pruned_factor_scores.csv", index=False)
    write_factor_loadings_plot(task_cols, factor_loadings, output_dir / "pruned_factor_loadings.html")


def read_full_baseline_summary() -> dict[str, Any] | None:
    """Read the previous full-metric canonical DSP summary if present."""
    path = (
        Path(__file__).resolve().parent
        / "reference_outputs/collaborator_grug_v4_aggregate_repro_20260525/canonical_dsp_sent_zip/summary.json"
    )
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_report(
    output_dir: Path,
    summary: dict[str, Any],
    top_weights: pd.DataFrame,
    split: pd.DataFrame,
) -> None:
    """Write a concise report for the pruned aggregate fit."""
    heldout_cols = [
        column
        for column in [
            "metric",
            "readiness_bucket",
            "dsp_oof_r",
            "dsp_oof_spearman",
            "proportional_signal_to_noise",
        ]
        if column in split.columns
    ]
    heldout_lines = [
        (
            f"- `{row['metric']}`: bucket `{row['readiness_bucket']}`, "
            f"DSP r `{row.get('dsp_oof_r', float('nan')):.3f}`, "
            f"prop SNR `{row.get('proportional_signal_to_noise', float('nan')):.3f}`."
        )
        for _, row in split.loc[~split["readiness_bucket"].eq("predictable"), heldout_cols].iterrows()
    ]
    top_lines = [
        (
            f"- `{row['domain']}`: w0 `{row['phase_0_weight']:.4f}`, "
            f"w1 `{row['phase_1_weight']:.4f}`, epochs `{row['total_epochs']:.3f}`."
        )
        for _, row in top_weights.head(12).iterrows()
    ]
    comparison_lines = [
        (
            f"- `{row['label']}`: TV `{row['mean_phase_tv']:.4f}`, "
            f"corr `{row['flat_phase_weight_correlation']:.4f}`, "
            f"predicted delta `{row['pred_delta_vs_reference']:.4f}`."
        )
        for row in summary["mixture_comparisons"]
    ]
    full = summary.get("full_metric_baseline")
    full_lines = []
    if full is not None:
        full_lines = [
            "",
            "## Full-Metric Baseline Comparison",
            "",
            f"- Full-metric raw predicted gain vs proportional: `{full['raw_pred_delta_vs_proportional']:.4f}`.",
            f"- Full-metric OOF Spearman: `{full['oof_spearman']:.4f}`.",
            (
                "- This comparison is across differently scaled factor targets, so use it for shape/fit diagnostics, "
                "not absolute utility."
            ),
        ]
    lines = [
        "# Predictable-Metric Factor DSP Fit",
        "",
        "The factor aggregate is rebuilt after holding out all metrics not marked `predictable` by the DSP-readiness audit.",
        "Held-out metrics remain guardrail/evaluation targets, but they do not define the factor score optimized here.",
        "",
        "## Held-Out Metrics",
        "",
        *heldout_lines,
        "",
        "## Fit",
        "",
        f"- Rows: `{summary['fit_row_count']}`.",
        f"- Predictable metrics used: `{summary['n_task_cols_used']}`.",
        f"- Held-out metrics: `{summary['n_heldout_metrics']}`.",
        f"- Variant: `{summary['variant']}`.",
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
        f"- Predicted pruned `y_factor`: `{summary['raw_pred_y_factor']:.4f}`.",
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
        "## Mixture-Space Comparisons",
        "",
        *comparison_lines,
        *full_lines,
        "",
        "## Caution",
        "",
        "- This is a raw unconstrained DSP optimum, not a validation-ready mixture by itself.",
        "- Because the held-out tasks are excluded from the optimized target, they must be checked as guardrails.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    """Run the predictable-only aggregate DSP fit."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split, kept_metrics, heldout_metrics = task_split(args.readiness_csv)
    split.to_csv(args.output_dir / "metric_predictable_vs_heldout_input.csv", index=False)
    split[split["readiness_bucket"].eq("predictable")].to_csv(args.output_dir / "kept_metrics.csv", index=False)
    split[~split["readiness_bucket"].eq("predictable")].to_csv(args.output_dir / "heldout_metrics.csv", index=False)

    bundle = DatasetBundle(
        name=args.dataset_name,
        raw_path=args.raw_csv,
        noise_path=args.noise_csv,
        metadata_path=args.metadata_csv,
        drop_incomplete_task_cols=args.drop_incomplete_task_cols,
    )
    print(f"loading raw matrix from {bundle.raw_path}", flush=True)
    data = subset_data(load_data(bundle), kept_metrics)
    raw = data["raw"]
    domains = data["domains"]
    w0 = data["w0"]
    w1 = data["w1"]
    c0 = data["c0"]
    c1 = data["c1"]
    z = data["z"]
    noise_share = data["noise_share"]
    task_cols = data["task_cols"]
    assert isinstance(raw, pd.DataFrame)
    assert isinstance(domains, list)
    assert isinstance(w0, np.ndarray)
    assert isinstance(w1, np.ndarray)
    assert isinstance(c0, np.ndarray)
    assert isinstance(c1, np.ndarray)
    assert isinstance(z, np.ndarray)
    assert isinstance(noise_share, np.ndarray)
    assert isinstance(task_cols, list)

    print(
        f"constructing 5-factor aggregate from {len(task_cols)} predictable metrics; "
        f"holding out {len(heldout_metrics)}",
        flush=True,
    )
    aggregate = aggregate_targets(z, noise_share)
    y_factor = np.asarray(aggregate["y_factor"], dtype=float)
    y_loss = -y_factor
    write_factor_tables(args.output_dir, task_cols, aggregate, split)

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
    mixtures: dict[str, np.ndarray] = {
        "proportional": proportional,
        "raw_dsp_optimum": raw_weights,
    }
    v4 = read_dashboard_v4(domains)
    lcb = read_reproduced_lcb(domains)
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
    pd.concat(
        [weights_frame(label, domains, weights, c0, c1, model) for label, weights in mixtures.items()],
        ignore_index=True,
    ).to_csv(args.output_dir / "mixture_weights.csv", index=False)
    top_weights.to_csv(args.output_dir / "raw_optimum_weights.csv", index=False)
    pd.DataFrame(comparisons).to_csv(args.output_dir / "mixture_comparisons.csv", index=False)
    write_mixture_plot(mixtures, domains, c0, c1, args.output_dir / "mixture_plot.html")

    full_baseline = read_full_baseline_summary()
    model_metrics = {
        "dataset": bundle.name,
        "raw_path": str(bundle.raw_path),
        "noise_path": str(bundle.noise_path),
        "metadata_path": str(bundle.metadata_path),
        "readiness_path": str(args.readiness_csv),
        "variant": model.variant.name,
        "variant_key": args.variant,
        "description": model.variant.description,
        "fit_row_count": len(y_factor),
        "n_task_cols_used": len(task_cols),
        "n_heldout_metrics": len(heldout_metrics),
        "heldout_metrics": heldout_metrics,
        "n_task_cols_all_before_pruning": len(data["task_cols_all_before_pruning"]),
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
        "full_metric_baseline": full_baseline,
        **prediction_metrics(y_factor, train_pred_y, "train"),
        **prediction_metrics(y_factor, oof_pred_y, "oof"),
        **phase_stats(raw_weights),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(model_metrics, indent=2, sort_keys=True))
    pd.DataFrame([model_metrics]).drop(
        columns=["heldout_metrics", "mixture_comparisons", "target_correlations", "pred_y_by_label", "full_metric_baseline"]
    ).to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "model.json").write_text(json.dumps(dsp.model_to_json(model, model_metrics), indent=2))
    write_report(args.output_dir, model_metrics, top_weights, split)

    print(
        textwrap.dedent(
            f"""
            wrote {args.output_dir}
            variant={model.variant.name}
            rows={len(y_factor)} domains={len(domains)} params={model.total_param_count}
            metrics_used={len(task_cols)} heldout={len(heldout_metrics)}
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
