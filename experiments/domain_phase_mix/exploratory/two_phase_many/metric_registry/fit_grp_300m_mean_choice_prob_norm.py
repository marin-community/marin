# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "plotly", "scipy", "scikit-learn", "tabulate", "torch"]
# ///
"""Fit GRP no-L2 to 300M mean choice_prob_norm, excluding MMLU-Pro."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy import stats

from experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_grp_power_family_penalty_no_l2_retune import (
    REG_FIXED,
    VARIANT_NAME,
    _no_l2_param_keys,
    _parameter_counts,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.build_metric_registry import (
    METRICS_WIDE_CSV,
    WEIGHT_PREFIXES,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.fit_grp_300m_mmlu_choice_prob_norm import (
    BLOCK_VARIANTS,
    DEFAULT_COARSE_TOP_K,
    DEFAULT_METHOD,
    DEFAULT_PROB_EPS,
    DEFAULT_RANDOM_STARTS,
    RUN_SET,
    SCALE,
    _metric_oof_summary,
    _model_options,
    _objective_diagnostics,
    _refine_rows,
    _write_optimum_weight_tables,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.fit_grp_no_l2_benchmark_aggregates import (
    AggregateObjective,
    FAMILY_SCHEMES,
    _expanded_start_bank,
    _family_shares,
    _model_target_to_metric,
    _packet_from_frame,
    _plot_predictions,
    _plot_residuals,
    _prediction_rows,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_penalty_calibration import (
    build_penalty_calibration_surrogate,
    compute_penalty_calibration_metrics,
    optimize_penalty_calibration_model,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.plot_grp_phase_comparison import (
    TEXT_MUTED_COLOR,
    _plot_cc_block,
    _plot_non_cc_block,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.plot_grp_vs_proportional import (
    _display_non_cc_label,
    _grp_domain_order,
)
from experiments.domain_phase_mix.static_batch_selection import average_phase_tv_distance

COHORT = "signal"
CV_SEED = 0
TARGET_COLUMN = "mean_choice_prob_norm_no_mmlu_pro"
TARGET_SLUG = f"{TARGET_COLUMN}_logit"
DISPLAY_NAME = "Mean choice_prob_norm excluding MMLU-Pro"
OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent / "reference_outputs" / "grp_300m_mean_choice_prob_norm_no_mmlu_pro_20260428"
)
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
PARAMS_CSV = OUTPUT_DIR / "params.csv"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"
REPORT_MD = OUTPUT_DIR / "report.md"
OPTIMUM_DIAGNOSTICS_CSV = OUTPUT_DIR / "optimum_diagnostics.csv"
TARGET_COLUMNS_JSON = OUTPUT_DIR / "target_columns.json"
EXCLUDED_SUITES = frozenset({"mmlu_pro_5shot"})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--coarse-top-k", type=int, default=DEFAULT_COARSE_TOP_K)
    parser.add_argument("--random-starts", type=int, default=DEFAULT_RANDOM_STARTS)
    parser.add_argument("--prob-eps", type=float, default=DEFAULT_PROB_EPS)
    parser.add_argument("--family-scheme", action="append", choices=FAMILY_SCHEMES, default=[])
    parser.add_argument("--block-variant", action="append", choices=BLOCK_VARIANTS, default=[])
    return parser.parse_args()


def _suite_name(metric_column: str) -> str:
    pieces = metric_column.split("/")
    if len(pieces) < 3 or pieces[0] != "lm_eval":
        raise ValueError(f"Unexpected lm-eval metric column: {metric_column}")
    return pieces[1]


def _choice_prob_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if not column.startswith("lm_eval/") or not column.endswith("/choice_prob_norm"):
            continue
        suite = _suite_name(column)
        if suite in EXCLUDED_SUITES:
            continue
        accuracy_column = column.rsplit("/", 1)[0] + "/acc"
        if accuracy_column not in frame.columns:
            continue
        columns.append(column)
    if not columns:
        raise ValueError("No choice_prob_norm columns found for the aggregate target")
    return sorted(columns)


def _fit_frame() -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(METRICS_WIDE_CSV, low_memory=False)
    target_columns = _choice_prob_columns(frame)
    mask = frame["scale"].eq(SCALE) & frame["cohort"].eq(COHORT) & frame["is_qsplit240_core"].fillna(False)
    source = frame.loc[mask].copy()
    source = source.dropna(subset=target_columns).copy()
    source[TARGET_COLUMN] = source[target_columns].mean(axis=1)

    weight_columns = sorted(column for column in source.columns if column.startswith(WEIGHT_PREFIXES))
    id_columns = [
        column
        for column in (
            "registry_run_key",
            "run_id",
            "run_name",
            "scale",
            "cohort",
            "source_run_name",
            "source_experiment",
            "wandb_run_id",
            "checkpoint_root",
            "status",
            "is_qsplit240_core",
        )
        if column in source.columns
    ]
    fit_frame = source[id_columns + weight_columns + [TARGET_COLUMN]].rename(columns={TARGET_COLUMN: "objective_metric"})
    fit_frame["objective_metric_key"] = TARGET_COLUMN
    fit_frame = fit_frame.dropna(axis=1, how="all")
    fit_frame = fit_frame.reset_index(drop=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fit_frame.to_csv(OUTPUT_DIR / "fit_dataset.csv", index=False)
    TARGET_COLUMNS_JSON.write_text(
        json.dumps(
            {
                "target": TARGET_COLUMN,
                "excluded_suites": sorted(EXCLUDED_SUITES),
                "n_target_columns": len(target_columns),
                "target_columns": target_columns,
                "n_rows": len(fit_frame),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return fit_frame, target_columns


def _aggregate_objective() -> AggregateObjective:
    return AggregateObjective(
        slug=TARGET_SLUG,
        source_column="objective_metric",
        display_name=DISPLAY_NAME,
        family="choice_prob_norm",
        higher_is_better=True,
        transform="logit_probability",
    )


def _plot_optimum_phase_comparison(
    *,
    objective_dir: Path,
    packet,
    opt_kind: str,
    target_columns: list[str],
) -> Path:
    weights_long = pd.read_csv(objective_dir / "optimum_weights.csv")
    weights_long = weights_long.loc[weights_long["opt_kind"].eq(opt_kind)].copy()
    if len(weights_long) != len(packet.base.domain_names):
        raise ValueError(f"Expected {len(packet.base.domain_names)} rows for {opt_kind}, got {len(weights_long)}")

    weights_by_domain = weights_long.set_index("domain_name")
    weights = np.array(
        [
            weights_by_domain.loc[packet.base.domain_names, "phase0_weight"].to_numpy(dtype=float),
            weights_by_domain.loc[packet.base.domain_names, "phase1_weight"].to_numpy(dtype=float),
        ]
    )
    non_cc_indices, cc_indices = _grp_domain_order(packet.base.domain_names, weights)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(24, 15),
        gridspec_kw={"width_ratios": [1.0, 1.65], "wspace": 0.30},
        facecolor="white",
    )
    _plot_non_cc_block(
        ax=axes[0],
        indices=non_cc_indices,
        labels=[_display_non_cc_label(packet.base.domain_names[idx]) for idx in non_cc_indices],
        weights=weights,
        phase0_multipliers=packet.base.c0,
        phase1_multipliers=packet.base.c1,
        title="Non-CC Domains",
    )
    _plot_cc_block(
        ax=axes[1],
        domain_names=packet.base.domain_names,
        indices=cc_indices,
        weights=weights,
        phase0_multipliers=packet.base.c0,
        phase1_multipliers=packet.base.c1,
        title="CC Domains",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        "GRP optimum: mean choice_prob_norm excluding MMLU-Pro",
        fontsize=31,
        y=0.985,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.947,
        f"{opt_kind} mixture; 300M/6B qsplit-core; mean over {len(target_columns)} choice-probability metrics",
        ha="center",
        va="center",
        fontsize=18,
        color=TEXT_MUTED_COLOR,
    )
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=18, bbox_to_anchor=(0.5, 0.925))
    fig.text(
        0.5,
        0.07,
        "Epoch labels; 80/20 WSD.",
        ha="center",
        va="center",
        fontsize=14,
        color=TEXT_MUTED_COLOR,
    )
    fig.subplots_adjust(top=0.88, left=0.12, right=0.985, bottom=0.10, wspace=0.30)
    plot_path = objective_dir / f"{opt_kind}_optimum_phase_comparison.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for domain_name, phase0_weight, phase1_weight, c0, c1 in zip(
        packet.base.domain_names,
        weights[0],
        weights[1],
        packet.base.c0,
        packet.base.c1,
        strict=True,
    ):
        rows.append(
            {
                "domain": domain_name,
                "phase0_weight": float(phase0_weight),
                "phase0_epochs": float(phase0_weight * c0),
                "phase1_weight": float(phase1_weight),
                "phase1_epochs": float(phase1_weight * c1),
            }
        )
    pd.DataFrame(rows).to_csv(objective_dir / f"{opt_kind}_optimum_phase_comparison.csv", index=False)
    return plot_path


def _fit_one(
    *,
    fit_frame: pd.DataFrame,
    target_columns: list[str],
    family_scheme: str,
    block_variant: str,
    method: str,
    coarse_top_k: int,
    random_starts: int,
    prob_eps: float,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    objective = _aggregate_objective()
    packet = _packet_from_frame(fit_frame, objective, prob_eps, family_scheme)
    start_bank = _expanded_start_bank(random_starts)
    model_options = _model_options(block_variant)
    coarse, best, refine = _refine_rows(packet, start_bank, coarse_top_k, method, model_options)
    params = {key: float(best[key]) for key in _no_l2_param_keys()}
    params["reg"] = REG_FIXED

    model = build_penalty_calibration_surrogate(
        packet,
        params=params,
        variant_name=VARIANT_NAME,
        **model_options,
    ).fit(packet.base.w, packet.base.y)
    full_metrics = compute_penalty_calibration_metrics(packet, model, seed=CV_SEED)
    raw_result, phase0, phase1 = optimize_penalty_calibration_model(packet, model, seed=CV_SEED)
    raw_weights = np.stack([phase0, phase1], axis=0)
    raw_distances = average_phase_tv_distance(packet.base.w, raw_weights[None, :, :])
    nearest_idx = int(np.argmin(raw_distances))

    train_pred_target = model.predict(packet.base.w)
    pred_rows = _prediction_rows(packet, params, objective)
    best_observed_idx = int(np.argmax(pred_rows["actual_metric"]))
    predicted_observed_idx = int(np.argmin(train_pred_target))
    raw_metric = float(_model_target_to_metric(float(raw_result.fun), objective))
    raw_nearest_metric = float(pred_rows.loc[nearest_idx, "actual_metric"])
    best_metric = float(pred_rows.loc[best_observed_idx, "actual_metric"])
    predicted_observed_metric = float(pred_rows.loc[predicted_observed_idx, "actual_metric"])

    objective_dir = OUTPUT_DIR / f"{TARGET_SLUG}__{family_scheme}__{block_variant}"
    objective_dir.mkdir(parents=True, exist_ok=True)
    coarse.to_csv(objective_dir / "coarse.csv", index=False)
    refine.to_csv(objective_dir / "refine.csv", index=False)
    pred_rows.to_csv(objective_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(
        {
            "domain_name": packet.base.domain_names,
            "phase0_weight": phase0,
            "phase0_epochs": phase0 * packet.base.c0,
            "phase1_weight": phase1,
            "phase1_epochs": phase1 * packet.base.c1,
        }
    ).to_csv(objective_dir / "raw_optimum_weights.csv", index=False)
    optimum_rows, optimum_weight_map = _objective_diagnostics(
        packet,
        model,
        objective,
        pred_rows,
        float(raw_result.fun),
        raw_weights,
    )
    optimum_rows.to_csv(objective_dir / "optimum_diagnostics.csv", index=False)
    _write_optimum_weight_tables(objective_dir, packet, optimum_weight_map)
    _plot_predictions(pred_rows, objective_dir / "predicted_vs_actual.html", DISPLAY_NAME)
    _plot_residuals(pred_rows, objective_dir / "residuals.html", DISPLAY_NAME)
    raw_plot_path = _plot_optimum_phase_comparison(
        objective_dir=objective_dir,
        packet=packet,
        opt_kind="raw",
        target_columns=target_columns,
    )

    actual = pred_rows["actual_metric"].to_numpy(dtype=float)
    predicted = pred_rows["predicted_metric"].to_numpy(dtype=float)
    deployment = {
        f"{row.opt_kind!s}_{key}": value
        for row in optimum_rows.itertuples(index=False)
        for key, value in {
            "predicted_metric": float(row.predicted_metric),
            "optimism_vs_best_observed_metric": float(row.optimism_vs_best_observed_metric),
            "nearest_observed_metric": float(row.nearest_observed_metric),
            "nearest_observed_regret": float(row.nearest_observed_regret),
            "nearest_observed_tv": float(row.nearest_observed_tv),
        }.items()
    }

    summary = {
        "slug": TARGET_SLUG,
        "target": TARGET_COLUMN,
        "display_name": DISPLAY_NAME,
        "family_scheme": family_scheme,
        "block_variant": block_variant,
        "scale": SCALE,
        "run_set": RUN_SET,
        "cohort": COHORT,
        "transform": "logit_probability",
        "n": len(packet.base.y),
        "n_target_columns": len(target_columns),
        "method": method,
        "coarse_top_k": int(coarse_top_k),
        "start_bank_size": len(start_bank),
        "best_observed_run_name": str(pred_rows.loc[best_observed_idx, "run_name"]),
        "best_observed_metric": best_metric,
        "predicted_observed_run_name": str(pred_rows.loc[predicted_observed_idx, "run_name"]),
        "predicted_observed_metric": predicted_observed_metric,
        "predicted_observed_regret": float(best_metric - predicted_observed_metric),
        "raw_predicted_optimum_metric": raw_metric,
        "raw_nearest_observed_run_name": str(pred_rows.loc[nearest_idx, "run_name"]),
        "raw_nearest_observed_metric": raw_nearest_metric,
        "raw_nearest_observed_regret": float(best_metric - raw_nearest_metric),
        "raw_nearest_observed_tv": float(raw_distances[nearest_idx]),
        "raw_phase_comparison_png": str(raw_plot_path),
        "actual_metric_spearman_with_predicted_metric": float(stats.spearmanr(actual, predicted).statistic),
        **deployment,
        **_metric_oof_summary(pred_rows, higher_is_better=True),
        **{
            key: float(value)
            for key, value in full_metrics.items()
            if isinstance(value, int | float | np.integer | np.floating)
        },
        **{
            key: float(value) for key, value in best.items() if isinstance(value, int | float | np.integer | np.floating)
        },
        **_family_shares(packet, raw_weights),
    }
    params_row = {
        "slug": TARGET_SLUG,
        "family_scheme": family_scheme,
        "block_variant": block_variant,
        **params,
        **_parameter_counts(packet),
    }
    return summary, params_row, optimum_rows


def _write_report(summary: pd.DataFrame, optimum: pd.DataFrame) -> None:
    display_columns = [
        "family_scheme",
        "block_variant",
        "n",
        "n_target_columns",
        "metric_oof_rmse",
        "metric_oof_spearman",
        "metric_oof_pearson",
        "metric_oof_regret_at_1",
        "actual_metric_min",
        "actual_metric_max",
        "actual_metric_std",
        "predicted_metric_std",
        "best_observed_metric",
        "predicted_observed_metric",
        "predicted_observed_regret",
        "raw_predicted_optimum_metric",
        "raw_nearest_observed_metric",
        "raw_nearest_observed_regret",
        "raw_nearest_observed_tv",
        "raw_phase0_broad_text_share",
        "raw_phase0_tech_code_share",
        "raw_phase0_reasoning_share",
        "raw_phase1_broad_text_share",
        "raw_phase1_tech_code_share",
        "raw_phase1_reasoning_share",
    ]
    optimum_columns = [
        "family_scheme",
        "block_variant",
        "opt_kind",
        "predicted_metric",
        "optimism_vs_best_observed_metric",
        "nearest_observed_metric",
        "nearest_observed_regret",
        "nearest_observed_tv",
    ]
    body = [
        "# 300M Mean choice_prob_norm GRP Fit",
        "",
        "## Setup",
        "",
        f"- Scale: `{SCALE}`.",
        f"- Run set: `{RUN_SET}`.",
        "- Model: GRP power-family-penalty no-L2 body.",
        "- Target: row-wise mean of all lm-eval `choice_prob_norm` metrics that have matching `acc`, "
        "excluding `mmlu_pro_5shot`.",
        "- Transform: logit probability; higher is better, then negated internally so the existing "
        "minimization path can optimize it.",
        "",
        "## Fit Summary",
        "",
        summary[display_columns].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Optimum Diagnostics",
        "",
        optimum[optimum_columns].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
        "- This is a proxy target for benchmark accuracy, not a loss target.",
        "- Treat raw simplex optima as candidates only if the nearest-observed TV and optimism diagnostics "
        "are reasonable.",
        "- The phase comparison plot uses the same all-domain, epoch-labeled setup as "
        "`surrogate_search/grp_phase_comparison.png`.",
        "",
    ]
    REPORT_MD.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    family_schemes = tuple(args.family_scheme) if args.family_scheme else ("synthetic_tech",)
    block_variants = tuple(args.block_variant) if args.block_variant else ("full",)
    fit_frame, target_columns = _fit_frame()

    rows: list[dict[str, Any]] = []
    param_rows: list[dict[str, Any]] = []
    optimum_rows: list[pd.DataFrame] = []
    for family_scheme in family_schemes:
        for block_variant in block_variants:
            print(
                f"fitting {TARGET_SLUG} family_scheme={family_scheme} block_variant={block_variant}",
                flush=True,
            )
            summary, params, optimum = _fit_one(
                fit_frame=fit_frame,
                target_columns=target_columns,
                family_scheme=family_scheme,
                block_variant=block_variant,
                method=args.method,
                coarse_top_k=args.coarse_top_k,
                random_starts=args.random_starts,
                prob_eps=args.prob_eps,
            )
            rows.append(summary)
            param_rows.append(params)
            optimum.insert(0, "block_variant", block_variant)
            optimum.insert(0, "family_scheme", family_scheme)
            optimum_rows.append(optimum)
            print(
                f"fit {family_scheme}/{block_variant}: "
                f"metric_oof_rmse={summary['metric_oof_rmse']:.6f} "
                f"metric_oof_spearman={summary['metric_oof_spearman']:.6f} "
                f"raw_nearest_tv={summary['raw_nearest_observed_tv']:.3f}",
                flush=True,
            )

    summary_frame = pd.DataFrame.from_records(rows)
    params_frame = pd.DataFrame.from_records(param_rows)
    optimum_frame = pd.concat(optimum_rows, ignore_index=True) if optimum_rows else pd.DataFrame()
    summary_frame.to_csv(SUMMARY_CSV, index=False)
    params_frame.to_csv(PARAMS_CSV, index=False)
    optimum_frame.to_csv(OPTIMUM_DIAGNOSTICS_CSV, index=False)
    SUMMARY_JSON.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(summary_frame, optimum_frame)
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {PARAMS_CSV}")
    print(f"Wrote {OPTIMUM_DIAGNOSTICS_CSV}")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
