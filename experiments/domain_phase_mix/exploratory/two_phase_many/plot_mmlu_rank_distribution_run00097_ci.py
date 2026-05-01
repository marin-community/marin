# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot MMLU rank distributions with run_00097 and two rerun confidence intervals highlighted."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from experiments.domain_phase_mix.determinism_analysis import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    _bootstrap_mean_and_std_ci,
    _collect_manifest_results_frame,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.plot_mmlu_rank_distribution import (
    METRICS,
    _prepare_frame,
    _ranked_frame,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_fixed_subset_study import (
    NAME as FIXED_SUBSET_NAME,
    WANDB_ENTITY,
    WANDB_PROJECT,
    build_run_specs as build_fixed_subset_run_specs,
)
from experiments.domain_phase_mix.launch_two_phase_many_run_00097_seed_study import (
    NAME as ORIGINAL_SEED_NAME,
    build_run_specs as build_original_seed_run_specs,
)

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_CI_PATH = OUTPUT_DIR / "mmlu_rank_distribution_run00097_ci.png"
OUTPUT_STD_PATH = OUTPUT_DIR / "mmlu_rank_distribution_run00097_std.png"
OUTPUT_PREDICTION_PATH = OUTPUT_DIR / "mmlu_rank_distribution_run00097_prediction_interval.png"
OUTPUT_CSV = OUTPUT_DIR / "mmlu_rank_distribution_run00097_intervals.csv"
RUN_NAME = "run_00097"
COHORT_COLORS = {
    "original_seed_jitter": "#F28E2B",
    "fixed_subset": "#59A14F",
}
INTERVAL_MODES = {
    "ci": {
        "output_path": OUTPUT_CI_PATH,
        "title_suffix": "rerun confidence intervals",
        "legend_suffix": "95 pct CI",
        "summary_text": "bars = 95 pct bootstrap CI of rerun mean",
        "low_column": "ci_low",
        "high_column": "ci_high",
    },
    "std": {
        "output_path": OUTPUT_STD_PATH,
        "title_suffix": "rerun standard-deviation bars",
        "legend_suffix": "±1 std",
        "summary_text": "bars = ±1 std across reruns",
        "low_column": "std_low",
        "high_column": "std_high",
    },
    "prediction_interval": {
        "output_path": OUTPUT_PREDICTION_PATH,
        "title_suffix": "rerun prediction intervals",
        "legend_suffix": "95 pct PI",
        "summary_text": "bars = 95 pct prediction interval for one rerun",
        "low_column": "prediction_low",
        "high_column": "prediction_high",
    },
}


def _manifest_payload(experiment_name: str, specs: list[object]) -> dict[str, object]:
    return {
        "experiment_name": experiment_name,
        "runs": [asdict(spec) for spec in specs],
    }


def _seed_sweep_results() -> dict[str, pd.DataFrame]:
    original_specs = [spec for spec in build_original_seed_run_specs() if spec.cohort == "seed_sweep"]
    fixed_specs = [spec for spec in build_fixed_subset_run_specs() if spec.cohort == "seed_sweep"]
    original_df = _collect_manifest_results_frame(
        manifest_payload=_manifest_payload(ORIGINAL_SEED_NAME, original_specs),
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        metric_prefixes=("lm_eval/",),
        extra_metrics=tuple(metric for metric, _, _ in METRICS),
    )
    fixed_df = _collect_manifest_results_frame(
        manifest_payload=_manifest_payload(FIXED_SUBSET_NAME, fixed_specs),
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        metric_prefixes=("lm_eval/",),
        extra_metrics=tuple(metric for metric, _, _ in METRICS),
    )
    return {
        "original_seed_jitter": original_df[original_df["status"] == "completed"].copy(),
        "fixed_subset": fixed_df[fixed_df["status"] == "completed"].copy(),
    }


def _interval_rows(repeat_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for cohort, frame in repeat_frames.items():
        for metric, _, _ in METRICS:
            values = frame[metric].dropna().to_numpy(dtype=np.float64)
            mean_ci, _ = _bootstrap_mean_and_std_ci(
                values,
                n_boot=DEFAULT_BOOTSTRAP_SAMPLES,
                seed=DEFAULT_BOOTSTRAP_SEED + (0 if cohort == "original_seed_jitter" else 1),
            )
            n = int(values.size)
            mean_value = float(values.mean())
            std_value = float(values.std(ddof=1))
            std_low = mean_value - std_value
            std_high = mean_value + std_value
            if n <= 1:
                prediction_low = mean_value
                prediction_high = mean_value
            else:
                t_crit = float(scipy_stats.t.ppf(0.975, df=n - 1))
                prediction_half_width = t_crit * std_value * float(np.sqrt(1.0 + 1.0 / n))
                prediction_low = mean_value - prediction_half_width
                prediction_high = mean_value + prediction_half_width
            rows.append(
                {
                    "cohort": cohort,
                    "metric": metric,
                    "n": n,
                    "mean": mean_value,
                    "std": std_value,
                    "ci_low": float(mean_ci[0]),
                    "ci_high": float(mean_ci[1]),
                    "std_low": std_low,
                    "std_high": std_high,
                    "prediction_low": prediction_low,
                    "prediction_high": prediction_high,
                }
            )
    return pd.DataFrame(rows)


def _implied_rank(ranked: pd.DataFrame, value: float, *, lower_is_better: bool, metric: str) -> float:
    ranks = ranked["rank"].to_numpy(dtype=float)
    metric_values = ranked[metric].to_numpy(dtype=float)
    if lower_is_better:
        return float(np.interp(value, metric_values, ranks))
    return float(np.interp(value, metric_values[::-1], ranks[::-1]))


def _plot_mode(df: pd.DataFrame, interval_df: pd.DataFrame, *, mode: str) -> None:
    mode_config = INTERVAL_MODES[mode]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, len(METRICS), figsize=(15, 6), dpi=180)
    fig.suptitle(f"Two-phase many-domain swarm: {RUN_NAME} with {mode_config['title_suffix']}", fontsize=18, y=0.98)

    cmap = plt.colormaps["RdYlGn_r"]

    for ax, (metric, lower_is_better, title) in zip(np.atleast_1d(axes), METRICS, strict=True):
        ranked = _ranked_frame(df, metric, lower_is_better)
        ranks = ranked["rank"].to_numpy()
        values = ranked[metric].to_numpy()

        color_positions = np.linspace(0.0, 1.0, len(ranked))
        point_colors = cmap(color_positions)

        ax.plot(ranks, values, color="#4C78A8", linewidth=2.0, alpha=0.95, zorder=1)
        ax.scatter(ranks, values, c=point_colors, s=24, edgecolors="none", alpha=0.88, zorder=2)

        original = ranked.loc[ranked["run_name"] == RUN_NAME].iloc[0]
        original_rank = float(original["rank"])
        original_value = float(original[metric])
        ax.scatter(
            [original_rank],
            [original_value],
            marker="o",
            s=88,
            facecolors="white",
            edgecolors="#222222",
            linewidths=1.4,
            zorder=5,
            label=f"{RUN_NAME} in swarm",
        )

        for cohort, color in COHORT_COLORS.items():
            row = interval_df[(interval_df["cohort"] == cohort) & (interval_df["metric"] == metric)].iloc[0]
            mean_value = float(row["mean"])
            interval_low = float(row[mode_config["low_column"]])
            interval_high = float(row[mode_config["high_column"]])
            x = _implied_rank(ranked, mean_value, lower_is_better=lower_is_better, metric=metric)
            ax.errorbar(
                [x],
                [mean_value],
                yerr=[[mean_value - interval_low], [interval_high - mean_value]],
                fmt="o",
                ms=7,
                capsize=5,
                elinewidth=2.0,
                color=color,
                markerfacecolor=color,
                markeredgecolor="#222222",
                zorder=6,
                label=(
                    f"{RUN_NAME} reruns ({mode_config['legend_suffix']})"
                    if cohort == "original_seed_jitter"
                    else f"{RUN_NAME} fixed subset ({mode_config['legend_suffix']})"
                ),
            )
            ax.annotate(
                f"rank {x:.1f}",
                xy=(x, mean_value),
                xytext=(6, 6 if cohort == "fixed_subset" else -14),
                textcoords="offset points",
                fontsize=8.5,
                color=color,
            )

        annotation_x = 0.62 if metric == "lm_eval/mmlu_5shot/choice_logprob" else 0.05
        annotation_ha = "left"
        ax.annotate(
            f"{RUN_NAME}\nrank {int(original_rank)}\nvalue {original_value:.4f}",
            xy=(original_rank, original_value),
            xytext=(annotation_x, 0.22),
            textcoords="axes fraction",
            fontsize=9,
            color="#222222",
            arrowprops={"arrowstyle": "-", "color": "#444444", "lw": 1.0},
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "alpha": 0.92,
                "edgecolor": "#777777",
            },
            ha=annotation_ha,
        )

        best_label = "min" if lower_is_better else "max"
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Rank (1 = best)")
        ax.set_ylabel(metric)
        ax.set_xlim(1, len(ranked))
        summary_x = 0.98 if metric == "lm_eval/mmlu_5shot/choice_logprob" else 0.02
        summary_ha = "right" if metric == "lm_eval/mmlu_5shot/choice_logprob" else "left"
        ax.text(
            summary_x,
            0.98,
            (
                f"n = {len(ranked)}\n"
                f"{best_label} = {values[0]:.4f}\n"
                f"median = {np.median(values):.4f}\n"
                f"{'max' if lower_is_better else 'min'} = {values[-1]:.4f}\n" + mode_config["summary_text"]
            ),
            transform=ax.transAxes,
            va="top",
            ha=summary_ha,
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "alpha": 0.9,
                "edgecolor": "#CCCCCC",
            },
        )
        ax.legend(loc="lower right", fontsize=9, frameon=True)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = mode_config["output_path"]
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main() -> None:
    df = _prepare_frame()
    repeat_frames = _seed_sweep_results()
    interval_df = _interval_rows(repeat_frames)
    interval_df.to_csv(OUTPUT_CSV, index=False)

    for mode in INTERVAL_MODES:
        _plot_mode(df, interval_df, mode=mode)

    print(f"Saved summary to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
