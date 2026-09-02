# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402,E501

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "scipy", "scikit-learn", "tabulate"]
# ///
"""Fit DSP variants to the issue #5416 aggregate task metric at 300M/6B.

The issue #5416 aggregate is higher-is-better. The DSP fitting path is written
as a loss minimizer with nonnegative benefit/penalty semantics, so this script
fits ``objective_metric = -issue5416_aggregate`` and reports metrics back in the
aggregate-score orientation.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False

from experiments.domain_phase_mix.exploratory.two_phase_many.fit_dsp_canonical_variants_300m import (
    VARIANTS,
    _fit_variant,
    _metrics,
    _oof_predictions,
    _optimize_model,
    _packet_from_frame,
    _plot_raw_optimum,
    _predict,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.issue5416_aggregate import (
    fit_issue5416_projection,
    score_issue5416_aggregate,
    write_issue5416_projection,
)
from experiments.domain_phase_mix.static_batch_selection import average_phase_tv_distance

SCRIPT_DIR = Path(__file__).resolve().parent
MATRIX_DIR = SCRIPT_DIR / "metric_registry" / "raw_metric_matrix_300m"
SIGNAL_CSV = MATRIX_DIR / "raw_metric_matrix_300m.csv"
VARIABLE_NOISE_CSV = MATRIX_DIR / "noise_baseline_run00097_variable_subset_300m.csv"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "dsp_issue5416_aggregate_300m_20260510"

TARGET_SCORE = "issue5416_aggregate"
TARGET_LOSS = "objective_metric"
VARIANT_NAMES = (
    "dsp_no_phase_penalty_nnls",
    "dsp_phase_benefit_penalty_nnls",
    "dsp_effective_exposure_penalty_nnls",
    "dsp_phase_benefit_saturation_penalty_nnls",
    "dsp_saturation_penalty_split_nnls",
)


def _selected_variants() -> list[Any]:
    by_name = {variant.name: variant for variant in VARIANTS}
    missing = sorted(set(VARIANT_NAMES) - set(by_name))
    if missing:
        raise ValueError(f"Unknown DSP variants: {missing}")
    return [by_name[name] for name in VARIANT_NAMES]


def _build_target_frame() -> tuple[pd.DataFrame, Any]:
    signal = pd.read_csv(SIGNAL_CSV, low_memory=False)
    noise = pd.read_csv(VARIABLE_NOISE_CSV, low_memory=False)
    if len(signal) != 242:
        raise ValueError(f"Expected 242 signal rows, found {len(signal)}")
    if len(noise) != 10:
        raise ValueError(f"Expected 10 variable-noise rows, found {len(noise)}")
    projection = fit_issue5416_projection(signal_frame=signal, noise_frame=noise)
    scores = score_issue5416_aggregate(signal, projection, fail_missing=True)
    if scores.isna().any():
        raise ValueError("Issue #5416 aggregate did not score every signal row")
    frame = signal.copy()
    frame[TARGET_SCORE] = scores
    frame[TARGET_LOSS] = -scores
    return frame, projection


def _observed_score_row(packet: Any, model: Any, scores: np.ndarray) -> dict[str, Any]:
    pred_loss = _predict(model, packet.w, packet)
    pred_score = -pred_loss
    predicted_rank = np.argsort(-pred_score)
    actual_rank = np.argsort(-scores)
    best_pred_idx = int(predicted_rank[0])
    top8 = predicted_rank[:8]
    return {
        "best_pred_observed_run": str(packet.frame.iloc[best_pred_idx][packet.name_col]),
        "best_pred_observed_pred_score": float(pred_score[best_pred_idx]),
        "best_pred_observed_actual_score": float(scores[best_pred_idx]),
        "best_pred_observed_actual_rank": int(np.where(actual_rank == best_pred_idx)[0][0] + 1),
        "pred_top8_mean_actual_score": float(np.mean(scores[top8])),
        "pred_top8_best_actual_score": float(np.max(scores[top8])),
        "actual_best_score": float(np.max(scores)),
        "actual_best_run": str(packet.frame.iloc[int(actual_rank[0])][packet.name_col]),
    }


def _score_metrics(packet: Any, model: Any, raw_result: Any, weights: np.ndarray) -> dict[str, Any]:
    base = _metrics(packet, model, raw_result, weights)
    scores = -packet.y
    pred_score = -_predict(model, packet.w, packet)
    oof_score = -_oof_predictions(packet, model)
    score_residual = oof_score - scores
    distances = average_phase_tv_distance(packet.w, weights[None, :, :])
    nearest_idx = int(np.argmin(distances))
    base.update(
        {
            "target_score_mean": float(np.mean(scores)),
            "target_score_std": float(np.std(scores, ddof=1)),
            "score_train_rmse": float(np.sqrt(np.mean((pred_score - scores) ** 2))),
            "score_cv_rmse": float(np.sqrt(np.mean(score_residual**2))),
            "score_cv_r2": float(1.0 - np.mean(score_residual**2) / np.var(scores, ddof=1)),
            "score_cv_mae": float(np.mean(np.abs(score_residual))),
            "score_oof_spearman": float(spearmanr(scores, oof_score).statistic),
            "score_oof_pearson": float(pearsonr(scores, oof_score).statistic),
            "raw_predicted_optimum_score": float(-raw_result.fun),
            "raw_nearest_observed_score": float(scores[nearest_idx]),
        }
    )
    base.update(_observed_score_row(packet, model, scores))
    return base


def _plot_predicted_vs_actual_score(packet: Any, model: Any, variant_dir: Path) -> None:
    scores = -packet.y
    pred = -_predict(model, packet.w, packet)
    oof = -_oof_predictions(packet, model)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(scores, pred, label="train fit", alpha=0.45, s=26)
    ax.scatter(scores, oof, label="OOF", alpha=0.8, s=28)
    lo = min(float(scores.min()), float(pred.min()), float(oof.min()))
    hi = max(float(scores.max()), float(pred.max()), float(oof.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Actual issue #5416 aggregate")
    ax.set_ylabel("Predicted issue #5416 aggregate")
    ax.set_title(model.variant.name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(variant_dir / "predicted_vs_actual_issue5416.png", dpi=180)
    plt.close(fig)


def _write_report(summary: pd.DataFrame, projection: Any) -> None:
    cols = [
        "variant",
        "total_param_count",
        "score_cv_rmse",
        "score_cv_r2",
        "score_oof_spearman",
        "score_oof_pearson",
        "cv_foldmean_regret_at_1",
        "best_pred_observed_run",
        "best_pred_observed_actual_score",
        "best_pred_observed_actual_rank",
        "raw_predicted_optimum_score",
        "raw_nearest_observed_tv",
        "raw_nearest_observed_run_name",
        "raw_nearest_observed_score",
        "phase0_max_weight",
        "phase1_max_weight",
        "fitted_gamma_benefit",
        "fitted_gamma_saturation",
        "fitted_gamma_penalty",
    ]
    table = summary.loc[:, cols].sort_values("score_cv_rmse")
    lines = [
        "# DSP Fits to Issue #5416 Aggregate at 300M/6B",
        "",
        "The issue #5416 aggregate is higher-is-better. Fits here use `objective_metric = -issue5416_aggregate` so the DSP loss-minimization code can be reused without changing the nonnegative benefit/penalty semantics.",
        "",
        "- Signal rows: 242",
        "- Variable-subset noise rows for projection: 10",
        f"- Selected aggregate items: {len(projection.task_columns)}",
        f"- Horn-selected factors: {projection.factor_count}",
        "",
        "## Summary",
        "",
        table.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
        "- `score_oof_spearman` is the main rank-fit number: higher means the form ranks mixtures better for the aggregate.",
        "- `score_cv_rmse` is in aggregate-score units. The observed aggregate standard deviation is included in `summary.csv` as `target_score_std`.",
        "- Raw optima remain diagnostic only when `raw_nearest_observed_tv` is large or phase weights collapse.",
        "",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, projection = _build_target_frame()
    write_issue5416_projection(projection, OUTPUT_DIR / "issue5416_projection.json")
    target_cols = ["run_name", "is_qsplit240_core", TARGET_SCORE, TARGET_LOSS]
    frame.loc[:, target_cols].to_csv(OUTPUT_DIR / "target_scores.csv", index=False)

    packet = _packet_from_frame(frame, name="dsp_issue5416_aggregate_300m").base
    summary_rows: list[dict[str, Any]] = []
    for variant in _selected_variants():
        variant_dir = OUTPUT_DIR / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        model, trace = _fit_variant(packet, variant)
        trace.to_csv(variant_dir / "fit_trace.csv", index=False)
        raw_result, weights = _optimize_model(model, packet)
        pd.DataFrame(
            {
                "domain": packet.domain_names,
                "phase_0": weights[0],
                "phase_1": weights[1],
            }
        ).to_csv(variant_dir / "raw_optimum_weights.csv", index=False)
        _plot_raw_optimum(packet, variant, weights, variant_dir)
        _plot_predicted_vs_actual_score(packet, model, variant_dir)
        params_payload = {
            key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in model.params.items()
        }
        (variant_dir / "params.json").write_text(
            json.dumps(
                {
                    "variant": asdict(variant),
                    "params": params_payload,
                    "intercept": model.intercept,
                    "benefit_coef": model.benefit_coef.tolist(),
                    "penalty_coef": model.penalty_coef.tolist(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        summary_rows.append(_score_metrics(packet, model, raw_result, weights))

    summary = pd.DataFrame.from_records(summary_rows)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    _write_report(summary, projection)
    print(f"Wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
