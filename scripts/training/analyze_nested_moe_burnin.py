#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze the matched compute-optimal E256 and fixed25 burn-in."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import wandb

from scripts.training.analyze_nested_moe_fixed import (
    GLOBAL_STEP,
    TIMING_WARMUP_STEPS,
    HistoryPoint,
    binned_medians,
    final_value,
    histories,
    paired_domains,
    timing_summary,
)

ENTITY = "marin-community"
PROJECT = "marin"
TOKENS_PER_STEP = 32 * 8192
GPU_COUNT = 16
OUTPUT_DIR = Path("docs/reports/assets")
OUTPUT_PREFIX = "nested-model-training-burnin"
CONTROL = "e256"
TREATMENT = "fixed25"
FORECAST_TOKENS_BILLIONS = (10.0, 20.0, 100.0, 1_000.0)
QUALITY_FORECAST_TOKENS_BILLIONS = (10.0, 20.0)
COOLDOWN_START_STEP = round(16_840 * 0.8)

RUNS = {
    CONTROL: "nest-burn-001-e256-d768-s8192-e256-c4p14e18-reference-r26",
    TREATMENT: "nest-burn-001-fixed25-d768-s8192-e256-c4p14e18-reference-r26",
}
LABELS = {
    CONTROL: "E256 control",
    TREATMENT: "Fixed25",
}
COLORS = {
    CONTROL: "#24292f",
    TREATMENT: "#0969da",
}


@dataclass(frozen=True)
class AnalysisConfig:
    project: str
    control_run: str
    treatment_run: str
    tokens_per_step: int
    gpu_count: int
    output_prefix: str
    quality_fit_end_step: int
    figure_title: str


def _parse_args() -> AnalysisConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--control-run", default=RUNS[CONTROL])
    parser.add_argument("--treatment-run", default=RUNS[TREATMENT])
    parser.add_argument("--tokens-per-step", type=int, default=TOKENS_PER_STEP)
    parser.add_argument("--gpu-count", type=int, default=GPU_COUNT)
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--quality-fit-end-step", type=int, default=COOLDOWN_START_STEP)
    parser.add_argument("--figure-title", default="Compute-optimal d768 fixed-chain burn-in")
    args = parser.parse_args()
    if args.tokens_per_step <= 0:
        raise ValueError("tokens-per-step must be positive")
    if args.gpu_count <= 0:
        raise ValueError("gpu-count must be positive")
    if args.quality_fit_end_step <= 0:
        raise ValueError("quality-fit-end-step must be positive")
    return AnalysisConfig(
        project=args.project,
        control_run=args.control_run,
        treatment_run=args.treatment_run,
        tokens_per_step=args.tokens_per_step,
        gpu_count=args.gpu_count,
        output_prefix=args.output_prefix,
        quality_fit_end_step=args.quality_fit_end_step,
        figure_title=args.figure_title,
    )


def _common_horizon(all_history: Mapping[str, Mapping[str, list[HistoryPoint]]], metric: str) -> int:
    endpoints = [max(point.step for point in history[metric]) for history in all_history.values() if history[metric]]
    if len(endpoints) != len(all_history):
        return -1
    return min(endpoints)


def _plot_paloma(
    all_history: Mapping[str, Mapping[str, list[HistoryPoint]]],
    config: AnalysisConfig,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    common_horizon = _common_horizon(all_history, "paloma_macro")
    for arm, history in all_history.items():
        points = [point for point in history["paloma_macro"] if point.step <= common_horizon]
        axes[0].plot(
            [point.step * config.tokens_per_step / 1e9 for point in points],
            [point.value for point in points],
            marker="o",
            color=COLORS[arm],
            label=LABELS[arm],
        )
    axes[0].set_title("Full E256 mode")
    axes[0].set_xlabel("Training tokens (billions)")
    axes[0].set_ylabel("Paloma macro loss (lower is better)")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    treatment = all_history[TREATMENT]
    for count, linestyle in ((128, "-"), (16, "--")):
        points = [point for point in treatment[f"paloma_e{count}"] if point.step <= common_horizon]
        axes[1].plot(
            [point.step * config.tokens_per_step / 1e9 for point in points],
            [point.value for point in points],
            marker="o",
            linestyle=linestyle,
            color=COLORS[TREATMENT],
            label=f"Fixed25 E{count}",
        )
    axes[1].set_title("Extractable fixed subsets")
    axes[1].set_xlabel("Training tokens (billions)")
    axes[1].set_ylabel("Paloma macro loss (lower is better)")
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    figure.suptitle(config.figure_title)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"{config.output_prefix}-paloma.png", dpi=180)
    plt.close(figure)


def _plot_series(
    all_history: Mapping[str, Mapping[str, list[HistoryPoint]]],
    *,
    metric: str,
    title: str,
    ylabel: str,
    output_suffix: str,
    config: AnalysisConfig,
    scale: float = 1.0,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 4.8))
    common_horizon = _common_horizon(all_history, metric)
    for arm, history in all_history.items():
        points = [point for point in history[metric] if point.step <= common_horizon]
        x, y = binned_medians(points)
        axis.plot(x * config.tokens_per_step / 1e9, y * scale, color=COLORS[arm], label=LABELS[arm])
    axis.set_title(title)
    axis.set_xlabel("Training tokens (billions)")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"{config.output_prefix}-{output_suffix}.png", dpi=180)
    plt.close(figure)


def _runtime_forecast(
    step_seconds: float,
    tokens_billions: float,
    config: AnalysisConfig,
) -> dict[str, float | int]:
    steps = round(tokens_billions * 1e9 / config.tokens_per_step)
    optimizer_hours = steps * step_seconds / 3600
    return {
        "steps": steps,
        "optimizer_hours": optimizer_hours,
        "gpu_hours": optimizer_hours * config.gpu_count,
    }


def _log_linear_fit(points: list[HistoryPoint], config: AnalysisConfig) -> dict[str, Any] | None:
    phase_zero = [point for point in points if 0 < point.step <= config.quality_fit_end_step]
    if len(phase_zero) < 3:
        return None
    tokens_billions = np.asarray([point.step * config.tokens_per_step / 1e9 for point in phase_zero])
    losses = np.asarray([point.value for point in phase_zero])
    slope, intercept = np.polyfit(np.log(tokens_billions), losses, deg=1)
    fitted = intercept + slope * np.log(tokens_billions)
    residual_sum = float(np.sum(np.square(losses - fitted)))
    total_sum = float(np.sum(np.square(losses - np.mean(losses))))
    return {
        "points": len(phase_zero),
        "through_step": phase_zero[-1].step,
        "slope_per_log_token": float(slope),
        "intercept": float(intercept),
        "r_squared": 1.0 - residual_sum / total_sum if total_sum > 0 else None,
        "forecasts": {
            str(tokens): float(intercept + slope * np.log(tokens)) for tokens in QUALITY_FORECAST_TOKENS_BILLIONS
        },
    }


def _paired_delta_fit(
    control: list[HistoryPoint],
    treatment: list[HistoryPoint],
    config: AnalysisConfig,
) -> dict[str, Any] | None:
    control_by_step = {point.step: point.value for point in control}
    paired = [
        HistoryPoint(point.step, point.value - control_by_step[point.step])
        for point in treatment
        if point.step in control_by_step
    ]
    return _log_linear_fit(paired, config)


def _paired_delta_summary(control: list[HistoryPoint], treatment: list[HistoryPoint]) -> dict[str, Any]:
    control_by_step = {point.step: point.value for point in control}
    deltas = {
        point.step: point.value - control_by_step[point.step] for point in treatment if point.step in control_by_step
    }
    values = list(deltas.values())
    if not values:
        return {"points": 0, "deltas": {}}
    return {
        "points": len(values),
        "through_step": max(deltas),
        "treatment_better": sum(value < 0 for value in values),
        "mean_delta": statistics.fmean(values),
        "median_delta": statistics.median(values),
        "min_delta": min(values),
        "max_delta": max(values),
        "deltas": {str(step): value for step, value in sorted(deltas.items())},
    }


def _domain_history(run: wandb.apis.public.Run, prefix: str) -> dict[int, dict[str, float]]:
    suffix = "/loss"
    metrics = {
        key.removeprefix(prefix).removesuffix(suffix): key
        for key, value in run.summary.items()
        if key.startswith(prefix)
        and key.endswith(suffix)
        and not key.endswith(("/macro_loss", "/micro_loss"))
        and isinstance(value, (int, float))
    }
    by_step: dict[int, dict[str, float]] = {}
    items = list(metrics.items())
    for start in range(0, len(items), 8):
        chunk = dict(items[start : start + 8])
        for row in run.scan_history(keys=[GLOBAL_STEP, *chunk.values()], page_size=10_000):
            step = row.get(GLOBAL_STEP)
            if not isinstance(step, (int, float)):
                continue
            step_values = by_step.setdefault(int(step), {})
            for domain, metric in chunk.items():
                value = row.get(metric)
                if isinstance(value, (int, float)):
                    step_values[domain] = float(value)
    return by_step


def _latest_aligned_domains(
    control_run: wandb.apis.public.Run,
    treatment_run: wandb.apis.public.Run,
) -> dict[str, Any]:
    prefix = "eval/paloma/"
    control = _domain_history(control_run, prefix)
    treatment = _domain_history(treatment_run, prefix)
    common_steps = sorted(control.keys() & treatment.keys())
    if not common_steps:
        return {"through_step": None, "domains": 0, "deltas": {}}
    step = common_steps[-1]
    return {"through_step": step, **paired_domains(treatment[step], control[step])}


def main(config: AnalysisConfig) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=60)
    run_ids = {
        CONTROL: config.control_run,
        TREATMENT: config.treatment_run,
    }
    runs = {arm: api.run(f"{ENTITY}/{config.project}/{run_id}") for arm, run_id in run_ids.items()}
    all_history = {arm: histories(run, include_nested=arm == TREATMENT) for arm, run in runs.items()}
    timing_horizon = _common_horizon(all_history, "step_duration")
    timing = {
        arm: timing_summary([point for point in history["step_duration"] if point.step <= timing_horizon])
        for arm, history in all_history.items()
    }
    baseline_step = timing[CONTROL].median_step_seconds

    summaries: dict[str, dict[str, Any]] = {}
    for arm, run in runs.items():
        history = all_history[arm]
        last_step = max((point.step for point in history["train_loss"]), default=-1)
        median_step = timing[arm].median_step_seconds
        hook_seconds = [point.value for point in history["hook_duration"]]
        loading_seconds = [
            point.value for point in history["loading_time"] if TIMING_WARMUP_STEPS <= point.step <= timing_horizon
        ]
        evaluation_hooks = [value for value in hook_seconds if value >= 10.0]
        run_runtime = run.summary.get("_runtime")
        elapsed_seconds = float(run_runtime) if isinstance(run_runtime, (int, float)) else None
        summaries[arm] = {
            "run_name": run_ids[arm],
            "url": run.url,
            "state": run.state,
            "final_step": last_step,
            "tokens_billions": (last_step + 1) * config.tokens_per_step / 1e9,
            "full_paloma_macro": final_value(history["paloma_macro"]),
            "full_paloma_micro": final_value(history["paloma_micro"]),
            "e128_paloma_macro": final_value(history["paloma_e128"]),
            "e16_paloma_macro": final_value(history["paloma_e16"]),
            "final_train_loss": final_value(history["train_loss"]),
            "final_cross_entropy_loss": final_value(history["cross_entropy_loss"]),
            "terminal_overflow": final_value(history["overflow"]),
            "median_step_seconds": median_step,
            "step_overhead_vs_e256": median_step / baseline_step - 1.0,
            "gpu_hours_per_billion_tokens": 1e9 / config.tokens_per_step * median_step * config.gpu_count / 3600,
            "evaluation_hook_count": len(evaluation_hooks),
            "median_evaluation_hook_seconds": statistics.median(evaluation_hooks) if evaluation_hooks else None,
            "total_logged_hook_seconds": sum(hook_seconds),
            "median_loading_seconds": statistics.median(loading_seconds) if loading_seconds else None,
            "p90_loading_seconds": float(np.quantile(loading_seconds, 0.9)) if loading_seconds else None,
            "p99_loading_seconds": float(np.quantile(loading_seconds, 0.99)) if loading_seconds else None,
            "loading_stall_count_10s": sum(value >= 10.0 for value in loading_seconds),
            "total_logged_loading_seconds": sum(loading_seconds),
            "elapsed_run_seconds": elapsed_seconds,
            "elapsed_gpu_hours": elapsed_seconds * config.gpu_count / 3600 if elapsed_seconds is not None else None,
            "runtime_forecasts": {
                str(tokens_billions): _runtime_forecast(median_step, tokens_billions, config)
                for tokens_billions in FORECAST_TOKENS_BILLIONS
            },
        }

    result = {
        "schema_version": 1,
        "tokens_per_step": config.tokens_per_step,
        "gpu_count": config.gpu_count,
        "timing_through_step": timing_horizon,
        "summaries": summaries,
        "timing": {arm: asdict(value) for arm, value in timing.items()},
        "evaluation_history": {
            arm: {
                metric: [asdict(point) for point in history[metric]]
                for metric in ("paloma_macro", "paloma_micro", "paloma_e128", "paloma_e16")
            }
            for arm, history in all_history.items()
        },
        "quality_log_linear_fits": {
            arm: _log_linear_fit(history["paloma_macro"], config) for arm, history in all_history.items()
        },
        "full_mode_delta_log_linear_fit": _paired_delta_fit(
            all_history[CONTROL]["paloma_macro"],
            all_history[TREATMENT]["paloma_macro"],
            config,
        ),
        "full_mode_paired_delta_summary": _paired_delta_summary(
            all_history[CONTROL]["paloma_macro"],
            all_history[TREATMENT]["paloma_macro"],
        ),
        "full_mode_domain_comparison": _latest_aligned_domains(runs[CONTROL], runs[TREATMENT]),
    }
    (OUTPUT_DIR / f"{config.output_prefix}-results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    with (OUTPUT_DIR / f"{config.output_prefix}-summary.csv").open("w", newline="") as file:
        fieldnames = list(next(iter(summaries.values())).keys())
        writer = csv.DictWriter(file, fieldnames=["arm", *fieldnames])
        writer.writeheader()
        for arm, summary in summaries.items():
            writer.writerow({"arm": arm, **summary})

    _plot_paloma(all_history, config)
    _plot_series(
        all_history,
        metric="train_loss",
        title="Training loss (100-step medians; fixed25 mixes E256, E128, and E16 rows)",
        ylabel="Training loss",
        output_suffix="loss",
        config=config,
    )
    _plot_series(
        all_history,
        metric="step_duration",
        title="Compiled optimizer-step time (100-step medians)",
        ylabel="Step time (ms)",
        output_suffix="step-time",
        config=config,
        scale=1_000.0,
    )


if __name__ == "__main__":
    main(_parse_args())
