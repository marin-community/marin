# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Benchmark current phase models on matched cosine and WSD StarCoder panels.

The primary comparison uses the same 96 mixture coordinates under the original
cosine schedule and the later 50/50 boundary-aligned WSD schedule. This avoids
confounding learning-rate schedule with coordinate coverage. The full 143-row
cosine surface is included as a secondary dense-panel benchmark.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_nested_coverage_dsp as geometry,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (  # noqa: E402
    plot_separate_heads_starcoder_u_shape_fit as separate_heads,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "starcoder_cosine_wsd_phase_models_20260711"
COSINE_DATA = SCRIPT_DIR.parent / "paper_plots/data/two_phase_starcoder_combined_143_from_wandb.csv"
WSD_DATA = (
    SCRIPT_DIR.parent
    / "starcoder_wsd_boundary_aligned_repeat_outputs"
    / "two_phase_feature_bayes_linear_20260313_211537/proxy_results.csv"
)
TARGET = "eval/paloma/dolma_100_programing_languages/bpb"
DOMAIN_NAMES = ["nemotron_full", "starcoder"]
WEIGHT_COLUMNS = [
    "phase_0_nemotron_full",
    "phase_0_starcoder",
    "phase_1_nemotron_full",
    "phase_1_starcoder",
]
MODEL_NAMES = ["effective_exposure", "effective_exposure_geometry", "separate_heads"]
MODEL_LABELS = {
    "effective_exposure": "Eff-exp DSP",
    "effective_exposure_geometry": "Eff-exp DSP + geometry",
    "separate_heads": "Separate heads",
}
PARAMETER_COUNTS = {
    "effective_exposure": 10,
    "effective_exposure_geometry": 12,
    "separate_heads": 11,
}
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def inferred_epoch_multipliers(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    multipliers = np.zeros((2, len(DOMAIN_NAMES)), dtype=float)
    for phase in range(2):
        for domain_index, domain in enumerate(DOMAIN_NAMES):
            weight_column = f"phase_{phase}_{domain}"
            epoch_domain = "nemotron" if domain == "nemotron_full" else domain
            epoch_column = f"phase_{phase}_{epoch_domain}_epochs"
            observed = frame[weight_column].to_numpy(dtype=float) > 1e-10
            ratios = frame.loc[observed, epoch_column].to_numpy(dtype=float) / frame.loc[
                observed, weight_column
            ].to_numpy(dtype=float)
            multipliers[phase, domain_index] = float(np.median(ratios))
    return multipliers[0], multipliers[1]


def normalized_weights(frame: pd.DataFrame) -> np.ndarray:
    w0 = frame[WEIGHT_COLUMNS[:2]].to_numpy(dtype=float)
    w1 = frame[WEIGHT_COLUMNS[2:]].to_numpy(dtype=float)
    if not np.allclose(w0.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Phase-0 weights do not sum to one")
    if not np.allclose(w1.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Phase-1 weights do not sum to one")
    return np.stack([w0, w1], axis=1)


def dataset(
    name: str,
    frame: pd.DataFrame,
    target_column: str,
    c0: np.ndarray,
    c1: np.ndarray,
) -> pooled.Dataset:
    if not frame[target_column].notna().all():
        raise ValueError(f"{name} has missing target values")
    result = pooled.Dataset(
        name=name,
        frame=frame.reset_index(drop=True),
        y=frame[target_column].to_numpy(dtype=float),
        weights=normalized_weights(frame),
        c0=np.asarray(c0, dtype=float),
        c1=np.asarray(c1, dtype=float),
        domain_names=list(DOMAIN_NAMES),
    )
    geometry.assert_unique_mixtures(result)
    return result


def matched_indices(cosine: pd.DataFrame, wsd: pd.DataFrame) -> np.ndarray:
    cosine_weights = cosine[WEIGHT_COLUMNS].to_numpy(dtype=float)
    wsd_weights = wsd[WEIGHT_COLUMNS].to_numpy(dtype=float)
    indices = []
    maximum_distance = 0.0
    for row in wsd_weights:
        distance = np.max(np.abs(cosine_weights - row[None, :]), axis=1)
        index = int(np.argmin(distance))
        maximum_distance = max(maximum_distance, float(distance[index]))
        indices.append(index)
    if maximum_distance > 1e-10:
        raise ValueError(f"WSD coordinate match failed: max distance={maximum_distance:.3g}")
    if len(set(indices)) != len(indices):
        raise ValueError("WSD coordinates do not map uniquely to the cosine panel")
    return np.asarray(indices, dtype=int)


def load_datasets() -> tuple[list[pooled.Dataset], pd.DataFrame]:
    cosine = pd.read_csv(COSINE_DATA)
    wsd = pd.read_csv(WSD_DATA)
    c0, c1 = inferred_epoch_multipliers(cosine)
    source_indices = matched_indices(cosine, wsd)
    cosine_matched = cosine.iloc[source_indices].copy().reset_index(drop=True)
    cosine_matched["matched_wsd_run_name"] = wsd["run_name"].to_numpy()
    if not np.allclose(
        cosine_matched[WEIGHT_COLUMNS].to_numpy(dtype=float),
        wsd[WEIGHT_COLUMNS].to_numpy(dtype=float),
        atol=1e-10,
    ):
        raise ValueError("Matched cosine and WSD coordinates are not in the same order")
    coordinate_table = wsd[["run_name", "source_idx", "source_run_id", *WEIGHT_COLUMNS]].copy()
    coordinate_table.insert(1, "cosine_row_index", source_indices)
    coordinate_table["cosine_bpb"] = cosine_matched[TARGET].to_numpy(dtype=float)
    coordinate_table["wsd_bpb"] = wsd["actual_bpb"].to_numpy(dtype=float)
    coordinate_table["wsd_minus_cosine_bpb"] = coordinate_table["wsd_bpb"] - coordinate_table["cosine_bpb"]
    datasets = [
        dataset("cosine_full_143", cosine, TARGET, c0, c1),
        dataset("cosine_matched_96", cosine_matched, TARGET, c0, c1),
        dataset("wsd_matched_96", wsd, "actual_bpb", c0, c1),
    ]
    return datasets, coordinate_table


def fit_predictions(
    data: pooled.Dataset,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    alpha0, alpha1 = geometry.phase_fractions(data)
    predictions: dict[str, np.ndarray] = {}
    for config in (
        geometry.FitConfig("effective_exposure", False),
        geometry.FitConfig(
            "effective_exposure_geometry",
            True,
            "effective_exposure",
            (0, 1),
        ),
    ):
        model = geometry.fit_model(
            data,
            train_indices,
            config,
            linear_reg=0.01,
            maxiter=16,
            coarse_top_k=2,
        )
        predictions[config.name] = geometry.predict(model, data.weights[test_indices], alpha0, alpha1)
    separate_heads_model = separate_heads.fit_separate_heads(geometry.packet(data, train_indices))
    predictions["separate_heads"] = separate_heads_model.predict(data.weights[test_indices])
    return predictions


def benchmark_dataset(
    data: pooled.Dataset,
    seeds: list[int],
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    all_indices = np.arange(data.n)
    for seed in seeds:
        folds = [
            (train, test) for train, test in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(all_indices)
        ]
        oof = {model_name: np.zeros(data.n, dtype=float) for model_name in MODEL_NAMES}
        for fold_index, (train_indices, test_indices) in enumerate(folds):
            print(
                f"{data.name}: seed={seed} fold={fold_index + 1}/{n_splits}",
                flush=True,
            )
            fold_predictions = fit_predictions(data, train_indices, test_indices)
            for model_name, prediction in fold_predictions.items():
                oof[model_name][test_indices] = prediction
        for model_name, prediction in oof.items():
            row = asdict(pooled.metrics(data, model_name, seed, prediction, folds))
            row["nominal_param_count"] = PARAMETER_COUNTS[model_name]
            row["target_std"] = float(np.std(data.y))
            row["oof_nrmse"] = row["oof_rmse"] / row["target_std"]
            row["oof_r2"] = 1.0 - float(np.sum((prediction - data.y) ** 2) / np.sum((data.y - np.mean(data.y)) ** 2))
            metric_rows.append(row)
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "dataset": data.name,
                        "model": model_name,
                        "seed": seed,
                        "row_index": all_indices,
                        "observed_bpb": data.y,
                        "oof_predicted_bpb": prediction,
                        "residual": prediction - data.y,
                    }
                )
            )
    full_fit_rows = []
    full_predictions = fit_predictions(data, all_indices, all_indices)
    for model_name, prediction in full_predictions.items():
        full_fit_rows.append(
            {
                "dataset": data.name,
                "model": model_name,
                "n_rows": data.n,
                "nominal_param_count": PARAMETER_COUNTS[model_name],
                "train_rmse": float(np.sqrt(np.mean((prediction - data.y) ** 2))),
                "train_r2": 1.0 - float(np.sum((prediction - data.y) ** 2) / np.sum((data.y - np.mean(data.y)) ** 2)),
                "train_spearman": float(spearmanr(data.y, prediction).statistic),
            }
        )
    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_rows, ignore_index=True),
        pd.DataFrame(full_fit_rows),
    )


def summarize(raw_metrics: pd.DataFrame) -> pd.DataFrame:
    summary = pooled.summarize(raw_metrics)
    extra = raw_metrics.groupby(["dataset", "model"], as_index=False).agg(
        target_std=("target_std", "first"),
        oof_nrmse_mean=("oof_nrmse", "mean"),
        oof_nrmse_std=("oof_nrmse", "std"),
        oof_r2_mean=("oof_r2", "mean"),
        oof_r2_std=("oof_r2", "std"),
    )
    return summary.merge(extra, on=["dataset", "model"], validate="one_to_one")


def write_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Out-of-fold RMSE", "Out-of-fold Spearman"),
        horizontal_spacing=0.12,
    )
    colors = ["#1a9850", "#fee08b", "#d73027"]
    for model_index, model_name in enumerate(MODEL_NAMES):
        frame = summary[summary["model"] == model_name]
        figure.add_trace(
            go.Bar(
                x=frame["dataset"],
                y=frame["oof_rmse_mean"],
                error_y={"type": "data", "array": frame["oof_rmse_std"]},
                name=MODEL_LABELS[model_name],
                marker_color=colors[model_index],
                legendgroup=model_name,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Bar(
                x=frame["dataset"],
                y=frame["oof_spearman_mean"],
                error_y={"type": "data", "array": frame["oof_spearman_std"]},
                name=MODEL_LABELS[model_name],
                marker_color=colors[model_index],
                legendgroup=model_name,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    figure.update_layout(
        title=(
            "Current phase models on the StarCoder cosine and WSD surfaces"
            "<br><sup>Matched panels use the same 96 mixture coordinates; "
            "bars are means over three 5-fold CV seeds</sup>"
        ),
        barmode="group",
        height=600,
        width=1400,
        legend_title="Model",
        margin={"l": 80, "r": 40, "t": 110, "b": 120},
    )
    figure.update_yaxes(title_text="BPB RMSE (lower is better)", row=1, col=1)
    figure.update_yaxes(title_text="Spearman (higher is better)", row=1, col=2)
    figure.write_html(
        output_dir / "starcoder_cosine_wsd_model_fit.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )


def write_report(
    summary: pd.DataFrame,
    full_fit: pd.DataFrame,
    coordinate_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    display = summary[
        [
            "dataset",
            "model",
            "n_rows",
            "nominal_param_count",
            "oof_rmse_mean",
            "oof_rmse_std",
            "oof_nrmse_mean",
            "oof_r2_mean",
            "oof_spearman_mean",
            "oof_spearman_std",
            "fold_mean_regret_at_1_mean",
            "lower_tail_optimism_mean",
        ]
    ].copy()
    matched = display[display["dataset"].isin(["cosine_matched_96", "wsd_matched_96"])]
    best_cosine = matched.loc[matched["dataset"].eq("cosine_matched_96")].sort_values("oof_rmse_mean").iloc[0]
    best_wsd = matched.loc[matched["dataset"].eq("wsd_matched_96")].sort_values("oof_rmse_mean").iloc[0]
    schedule_delta = (
        coordinate_table["wsd_minus_cosine_bpb"].mean(),
        coordinate_table["wsd_minus_cosine_bpb"].std(ddof=1),
    )
    report = f"""# StarCoder cosine vs WSD phase-model benchmark

## Design

- Primary comparison: the same 96 phase-mixture coordinates under the original
  50/50 cosine schedule and the later 50/50 boundary-aligned WSD schedule.
- Secondary comparison: all 143 rows in the denser cosine surface.
- Models: effective-exposure DSP, effective-exposure DSP plus phase-TV and
  aggregate-HHI geometry, and the exact separate-heads form used in the
  validated KL sweep.
- Evaluation: three repeated shuffled 5-fold CV seeds. Every model is refit
  inside every fold. In-sample fits are reported separately and are not the
  primary result.
- Target: `{TARGET}` (BPB, lower is better).

The WSD-minus-cosine BPB difference over matched coordinates has mean
{schedule_delta[0]:.6f} and standard deviation {schedule_delta[1]:.6f}. This is
a schedule response, not a paired training-seed estimate.

## Out-of-fold results

{display.to_markdown(index=False, floatfmt=".6f")}

On matched coordinates, the lowest-RMSE cosine model is
**{MODEL_LABELS[str(best_cosine['model'])]}**
({best_cosine['oof_rmse_mean']:.6f}), while the lowest-RMSE WSD model is
**{MODEL_LABELS[str(best_wsd['model'])]}** ({best_wsd['oof_rmse_mean']:.6f}).

## Full-data in-sample fits

{full_fit.to_markdown(index=False, floatfmt=".6f")}

## Interpretation guardrails

- Compare cosine and WSD primarily on the matched 96 rows. The 143-row cosine
  result answers a different coverage question.
- RMSE is in raw BPB units and therefore depends on target spread; normalized RMSE and Spearman help compare panels.
- The geometry model here uses the current two-term correction only: phase
  total variation and aggregate concentration. Older cross-swarm artifacts
  used an additional phase-1 concentration term.
- These are fit diagnostics, not evidence that an unconstrained surrogate optimum transfers to 3e18 validation.
"""
    (output_dir / "report.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets, coordinate_table = load_datasets()
    raw_metrics = []
    predictions = []
    full_fit = []
    for data in datasets:
        metrics_frame, prediction_frame, full_fit_frame = benchmark_dataset(
            data,
            parse_int_list(args.seeds),
            args.n_splits,
        )
        raw_metrics.append(metrics_frame)
        predictions.append(prediction_frame)
        full_fit.append(full_fit_frame)

    raw = pd.concat(raw_metrics, ignore_index=True)
    oof_predictions = pd.concat(predictions, ignore_index=True)
    full_fit_frame = pd.concat(full_fit, ignore_index=True)
    summary = summarize(raw)
    raw.to_csv(args.output_dir / "cv_metrics_by_seed.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    oof_predictions.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    full_fit_frame.to_csv(args.output_dir / "full_fit_metrics.csv", index=False)
    coordinate_table.to_csv(args.output_dir / "matched_coordinate_table.csv", index=False)
    write_plot(summary, args.output_dir)
    write_report(summary, full_fit_frame, coordinate_table, args.output_dir)
    print(summary.to_string(index=False))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
