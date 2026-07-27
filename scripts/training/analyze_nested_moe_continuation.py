#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze the paired 20.4B-token nested-MoE experiment."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY = "marin-community"
PROJECT = "marin_moe"
TOKENS_PER_STEP = 256 * 2048
BASE_STEPS = 8192
CONTINUATION_STEPS = 30_720
GPU_COUNT = 64
TIMING_WARMUP_STEPS = 1024
TIMING_BLOCK_STEPS = 200
OUTPUT_DIR = Path("docs/reports/assets")
TELLTALE_CSV = OUTPUT_DIR / "nested-model-training-final-telltale.csv"

GLOBAL_STEP = "global_step"
TRAIN_LOSS = "train/loss"
STEP_DURATION = "throughput/duration"
TOKENS_PER_SECOND = "throughput/tokens_per_second"
HOOK_TIME = "throughput/hook_time"
LOADING_TIME = "throughput/loading_time"
OVERFLOW = "train/router/capacity_overflow_rate_mean"
PALOMA_MACRO = "eval/paloma/macro_loss"
PALOMA_MICRO = "eval/paloma/micro_loss"

BASE_RUNS: Mapping[str, str] = MappingProxyType(
    {
        "large_control": "nest-moe-001-full-d768-s2048-e256-cost-r25",
        "small_control": "nest-moe-002-full-d768-s2048-e128-cost-r25",
        "ladder25": "nest-moe-006-full-d768-s2048-e256-cost-r25",
        "ladder50": "nest-moe-007-full-d768-s2048-e256-cost-r25",
    }
)
CONTINUATION_RUNS: Mapping[str, str] = MappingProxyType(
    {
        "large_control": "nest-moe-001-full-d768-s2048-e256-extend16b-r31",
        "small_control": "nest-moe-002-full-d768-s2048-e128-extend16b-r31",
        "ladder25": "nest-moe-006-full-d768-s2048-e256-extend16b-r31",
        "ladder50": "nest-moe-007-full-d768-s2048-e256-extend16b-r31",
    }
)
LABELS: Mapping[str, str] = MappingProxyType(
    {
        "large_control": "E256 control",
        "small_control": "E128 control",
        "ladder25": "Power ladder 25%",
        "ladder50": "Power ladder 50%",
    }
)
COLORS: Mapping[str, str] = MappingProxyType(
    {
        "large_control": "#24292f",
        "small_control": "#6e7781",
        "ladder25": "#0969da",
        "ladder50": "#cf222e",
    }
)
NESTED_COUNTS = (128, 32, 8, 1)
NESTED_OFFSETS: Mapping[int, tuple[int, ...]] = MappingProxyType(
    {
        128: (0, 1),
        32: (0, 2, 4, 6),
        8: (0, 8, 16, 24),
        1: (0, 64, 128, 192),
    }
)


@dataclass(frozen=True)
class HistoryPoint:
    step: int
    value: float


@dataclass(frozen=True)
class TimingSummary:
    samples: int
    median_step_seconds: float
    p10_step_seconds: float
    p90_step_seconds: float
    median_tokens_per_second: float
    block_bootstrap_ci95_low: float
    block_bootstrap_ci95_high: float


def nested_paloma_metric(count: int, offset: int) -> str:
    suffix = "" if offset == 0 else f"_offset{offset}"
    return f"eval/nested_e{count}{suffix}/paloma/macro_loss"


def histories(
    run: wandb.apis.public.Run,
    metrics: Mapping[str, str],
) -> dict[str, list[HistoryPoint]]:
    values = {name: [] for name in metrics}
    for row in run.scan_history(page_size=10_000):
        step = row.get(GLOBAL_STEP)
        if step is None:
            continue
        for name, metric in metrics.items():
            value = row.get(metric)
            if isinstance(value, (int, float)) and math.isfinite(value):
                values[name].append(HistoryPoint(step=int(step), value=float(value)))
    return values


def block_bootstrap_ci(values: list[HistoryPoint]) -> tuple[float, float]:
    block_steps = TIMING_BLOCK_STEPS if len(values) >= 1000 else 10
    blocks = [
        [point.value for point in values[index : index + block_steps]]
        for index in range(0, len(values), block_steps)
        if len(values[index : index + block_steps]) == block_steps
    ]
    if len(blocks) < 2:
        return math.nan, math.nan
    medians = np.asarray([statistics.median(block) for block in blocks])
    rng = np.random.default_rng(20260727)
    samples = rng.choice(medians, size=(10_000, len(medians)), replace=True)
    low, high = np.quantile(np.median(samples, axis=1), (0.025, 0.975))
    return float(low), float(high)


def timing_summary(duration: list[HistoryPoint]) -> TimingSummary:
    steady = [point for point in duration if point.step >= TIMING_WARMUP_STEPS]
    if not steady:
        raise ValueError("No post-warmup timing samples")
    values = [point.value for point in steady]
    median = statistics.median(values)
    ci_low, ci_high = block_bootstrap_ci(steady)
    return TimingSummary(
        samples=len(values),
        median_step_seconds=median,
        p10_step_seconds=float(np.quantile(values, 0.1)),
        p90_step_seconds=float(np.quantile(values, 0.9)),
        median_tokens_per_second=TOKENS_PER_STEP / median,
        block_bootstrap_ci95_low=ci_low,
        block_bootstrap_ci95_high=ci_high,
    )


def telltale_histories() -> dict[str, dict[str, list[HistoryPoint]]]:
    """Load task-zero scalar snapshots exported from durable finelog."""
    run_to_arm = {run_name: arm for arm, run_name in CONTINUATION_RUNS.items()}
    metric_columns = {
        "train_loss": "loss",
        "duration": "duration",
        "tokens_per_second": "tokens_per_second",
        "hook": "hook",
        "loading": "loading",
        "overflow": "overflow",
    }
    values = {arm: {metric: [] for metric in metric_columns} for arm in CONTINUATION_RUNS}
    with TELLTALE_CSV.open(newline="") as input_file:
        for row in csv.DictReader(input_file):
            arm = run_to_arm[row["run"]]
            step = int(float(row["step"]))
            for metric, column in metric_columns.items():
                value = row.get(column)
                if value:
                    values[arm][metric].append(HistoryPoint(step=step, value=float(value)))
    return values


def rolling_median(points: list[HistoryPoint], window: int) -> tuple[list[int], list[float]]:
    steps = []
    values = []
    for index in range(window - 1, len(points)):
        steps.append(points[index].step)
        values.append(statistics.median(point.value for point in points[index - window + 1 : index + 1]))
    return steps, values


def effective_train_tokens(phase_step: int) -> float:
    return (BASE_STEPS + phase_step + 1) * TOKENS_PER_STEP / 1e9


def effective_eval_tokens(phase_step: int) -> float:
    return (BASE_STEPS + phase_step) * TOKENS_PER_STEP / 1e9


def combined_eval_points(
    base: list[HistoryPoint],
    continuation: list[HistoryPoint],
) -> tuple[list[float], list[float]]:
    tokens = [(point.step + 1) * TOKENS_PER_STEP / 1e9 for point in base]
    values = [point.value for point in base]
    tokens.extend(effective_eval_tokens(point.step) for point in continuation)
    values.extend(point.value for point in continuation)
    return tokens, values


def plot_loss(
    base_histories: Mapping[str, Mapping[str, list[HistoryPoint]]],
    continuation_histories: Mapping[str, Mapping[str, list[HistoryPoint]]],
) -> None:
    fig, axis = plt.subplots(figsize=(9, 5.2))
    for arm in BASE_RUNS:
        base_steps, base_values = rolling_median(base_histories[arm]["train_loss"], 200)
        continuation_window = 200 if len(continuation_histories[arm]["train_loss"]) >= 5000 else 10
        continuation_steps, continuation_values = rolling_median(
            continuation_histories[arm]["train_loss"], continuation_window
        )
        axis.plot(
            [(step + 1) * TOKENS_PER_STEP / 1e9 for step in base_steps],
            base_values,
            color=COLORS[arm],
            linewidth=1.3,
        )
        axis.plot(
            [effective_train_tokens(step) for step in continuation_steps],
            continuation_values,
            color=COLORS[arm],
            label=LABELS[arm],
            linewidth=1.3,
        )
    boundary = BASE_STEPS * TOKENS_PER_STEP / 1e9
    axis.axvline(boundary, color="#8c959f", linestyle="--", linewidth=1)
    axis.text(boundary + 0.15, axis.get_ylim()[1] - 0.05, "optimizer reset", color="#57606a")
    axis.set_xlabel("Effective training tokens (billions)")
    axis.set_ylabel("Logged mixed-objective cross-entropy loss")
    axis.set_title("Training stability; objectives differ across arms")
    axis.text(
        0.01,
        0.02,
        "Ladder loss includes harder restricted-submodel rows; use full-mode Paloma for quality.",
        transform=axis.transAxes,
        color="#57606a",
        fontsize=9,
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "nested-model-training-final-loss.png", dpi=180)
    plt.close(fig)


def plot_timing(continuation_histories: Mapping[str, Mapping[str, list[HistoryPoint]]]) -> None:
    fig, axis = plt.subplots(figsize=(9, 5.2))
    for arm in CONTINUATION_RUNS:
        points = [point for point in continuation_histories[arm]["duration"] if point.step >= TIMING_WARMUP_STEPS]
        window = 200 if len(points) >= 5000 else 10
        steps, values = rolling_median(points, window)
        axis.plot(steps, values, color=COLORS[arm], label=LABELS[arm])
    axis.set_xlabel("Continuation optimizer step")
    axis.set_ylabel("Step compute time (seconds)")
    axis.set_title("16.1B-token continuation step time")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "nested-model-training-final-step-time.png", dpi=180)
    plt.close(fig)


def plot_full_paloma(
    base_histories: Mapping[str, Mapping[str, list[HistoryPoint]]],
    continuation_histories: Mapping[str, Mapping[str, list[HistoryPoint]]],
) -> None:
    fig, axis = plt.subplots(figsize=(9, 5.2))
    for arm in BASE_RUNS:
        tokens, values = combined_eval_points(
            base_histories[arm]["paloma"],
            continuation_histories[arm]["paloma"],
        )
        axis.plot(tokens, values, marker="o", color=COLORS[arm], label=LABELS[arm])
    axis.axvline(
        BASE_STEPS * TOKENS_PER_STEP / 1e9,
        color="#8c959f",
        linestyle="--",
        linewidth=1,
    )
    axis.set_xlabel("Effective training tokens (billions)")
    axis.set_ylabel("Paloma macro loss")
    axis.set_title("Full-model Paloma loss")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "nested-model-training-final-paloma.png", dpi=180)
    plt.close(fig)


def plot_nested_offsets(
    continuation_histories: Mapping[str, Mapping[str, list[HistoryPoint]]],
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
    level_colors = ("#0969da", "#8250df", "#bf8700", "#1a7f37")
    for row, arm in enumerate(("ladder25", "ladder50")):
        for column, (count, color) in enumerate(zip(NESTED_COUNTS, level_colors, strict=True)):
            axis = axes[row, column]
            for offset in NESTED_OFFSETS[count]:
                points = continuation_histories[arm][f"paloma_e{count}_offset{offset}"]
                axis.plot(
                    [effective_eval_tokens(point.step) for point in points],
                    [point.value for point in points],
                    marker="o",
                    color=color,
                    alpha=0.8,
                    label=f"offset {offset}",
                )
            axis.set_title(f"{LABELS[arm]} E{count}")
            axis.set_xlabel("Tokens (billions)")
            axis.set_ylabel("Paloma macro loss")
            axis.grid(alpha=0.2)
            axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "nested-model-training-final-offsets.png", dpi=180)
    plt.close(fig)


def final_value(points: list[HistoryPoint]) -> float | None:
    return points[-1].value if points else None


def runtime_forecasts(median_step: float) -> dict[str, dict[str, float | int]]:
    forecasts = {}
    for tokens_billions in (10.0, 16.10612736, 20.0, 1000.0):
        steps = math.ceil(tokens_billions * 1e9 / TOKENS_PER_STEP)
        seconds = steps * median_step
        forecasts[str(tokens_billions)] = {
            "steps": steps,
            "optimizer_seconds": seconds,
            "optimizer_hours": seconds / 3600,
            "gpu_hours": seconds * GPU_COUNT / 3600,
        }
    return forecasts


def training_loss_projection(points: list[HistoryPoint]) -> dict[str, Any]:
    block_size = 200 if len(points) >= 5000 else 20
    blocks = [
        points[index : index + block_size]
        for index in range(0, len(points), block_size)
        if len(points[index : index + block_size]) == block_size and points[index].step >= TIMING_WARMUP_STEPS
    ]
    if len(blocks) < 8:
        return {}
    tokens = np.asarray([statistics.fmean(effective_train_tokens(point.step) for point in block) for block in blocks])
    losses = np.asarray([statistics.fmean(point.value for point in block) for block in blocks])
    split = max(4, round(0.8 * len(tokens)))
    slope, intercept = np.polyfit(np.log(tokens[:split]), losses[:split], 1)
    held_out = intercept + slope * np.log(tokens[split:])
    rmse = float(np.sqrt(np.mean(np.square(held_out - losses[split:]))))
    targets = (20.401094656, 40.0, 80.0)
    return {
        "model": "loss = intercept + slope * ln(effective_tokens_billions)",
        "fit_start_tokens_billions": float(tokens[0]),
        "fit_train_end_tokens_billions": float(tokens[split - 1]),
        "held_out_end_tokens_billions": float(tokens[-1]),
        "intercept": float(intercept),
        "slope": float(slope),
        "held_out_rmse": rmse,
        "projections": {str(target): float(intercept + slope * math.log(target)) for target in targets},
    }


def write_csv(summaries: Mapping[str, Mapping[str, Any]]) -> None:
    fields = (
        "arm",
        "run_name",
        "state",
        "phase_final_step",
        "effective_tokens_billions",
        "runtime_seconds",
        "gpu_hours",
        "median_step_seconds",
        "step_ci95_low",
        "step_ci95_high",
        "median_tokens_per_second",
        "step_overhead_vs_e256",
        "final_paloma_macro_loss",
        "final_paloma_micro_loss",
        "mean_overflow",
        "terminal_overflow",
        "url",
    )
    with (OUTPUT_DIR / "nested-model-training-final-summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for summary in summaries.values():
            writer.writerow({field: summary.get(field) for field in fields})


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=60)
    base_runs = {arm: api.run(f"{ENTITY}/{PROJECT}/{run_name}") for arm, run_name in BASE_RUNS.items()}
    continuation_runs = {arm: api.run(f"{ENTITY}/{PROJECT}/{run_name}") for arm, run_name in CONTINUATION_RUNS.items()}
    common_metrics = {
        "train_loss": TRAIN_LOSS,
        "duration": STEP_DURATION,
        "tokens_per_second": TOKENS_PER_SECOND,
        "hook": HOOK_TIME,
        "loading": LOADING_TIME,
        "overflow": OVERFLOW,
        "paloma": PALOMA_MACRO,
        "paloma_micro": PALOMA_MICRO,
    }
    nested_metrics = {
        f"paloma_e{count}_offset{offset}": nested_paloma_metric(count, offset)
        for count in NESTED_COUNTS
        for offset in NESTED_OFFSETS[count]
    }
    base_histories = {arm: histories(run, common_metrics) for arm, run in base_runs.items()}
    continuation_histories = {
        arm: histories(run, {**common_metrics, **nested_metrics}) for arm, run in continuation_runs.items()
    }
    telemetry_source = "wandb"
    if TELLTALE_CSV.exists():
        telemetry_source = "finelog_telltale_task_zero"
        fallback = telltale_histories()
        for arm in CONTINUATION_RUNS:
            for metric, points in fallback[arm].items():
                if points:
                    continuation_histories[arm][metric] = points
    timings = {arm: timing_summary(continuation_histories[arm]["duration"]) for arm in CONTINUATION_RUNS}
    control_step = timings["large_control"].median_step_seconds

    summaries = {}
    for arm, run in continuation_runs.items():
        timing = timings[arm]
        step = int(run.summary.get(GLOBAL_STEP, -1))
        runtime = float(run.summary.get("_runtime", math.nan))
        overflow = continuation_histories[arm]["overflow"]
        summaries[arm] = {
            "arm": arm,
            "run_name": run.name,
            "state": run.state,
            "phase_final_step": step,
            "effective_tokens_billions": effective_train_tokens(step),
            "runtime_seconds": runtime,
            "gpu_hours": runtime * GPU_COUNT / 3600,
            "median_step_seconds": timing.median_step_seconds,
            "step_ci95_low": timing.block_bootstrap_ci95_low,
            "step_ci95_high": timing.block_bootstrap_ci95_high,
            "median_tokens_per_second": timing.median_tokens_per_second,
            "step_overhead_vs_e256": timing.median_step_seconds / control_step - 1.0,
            "final_paloma_macro_loss": final_value(continuation_histories[arm]["paloma"]),
            "final_paloma_micro_loss": final_value(continuation_histories[arm]["paloma_micro"]),
            "final_nested_paloma": {
                str(count): {
                    str(offset): final_value(continuation_histories[arm][f"paloma_e{count}_offset{offset}"])
                    for offset in NESTED_OFFSETS[count]
                }
                for count in NESTED_COUNTS
                if arm.startswith("ladder")
            },
            "mean_overflow": statistics.fmean(point.value for point in overflow),
            "terminal_overflow": final_value(overflow),
            "runtime_forecasts": runtime_forecasts(timing.median_step_seconds),
            "url": run.url,
        }

    artifact = {
        "schema_version": 1,
        "tokens_per_step": TOKENS_PER_STEP,
        "base_steps": BASE_STEPS,
        "continuation_steps": CONTINUATION_STEPS,
        "gpu_count_per_arm": GPU_COUNT,
        "continuation_telemetry_source": telemetry_source,
        "initialization": {
            "mode": "weights_only",
            "optimizer": "fresh",
            "learning_rate_multiplier": 0.1,
            "warmup_steps": 512,
        },
        "base_runs": dict(BASE_RUNS),
        "continuation_runs": dict(CONTINUATION_RUNS),
        "summaries": summaries,
        "timing": {arm: asdict(summary) for arm, summary in timings.items()},
        "training_loss_projections": {
            arm: training_loss_projection(continuation_histories[arm]["train_loss"]) for arm in CONTINUATION_RUNS
        },
    }
    with (OUTPUT_DIR / "nested-model-training-final-results.json").open("w") as output:
        json.dump(artifact, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
    write_csv(summaries)
    plot_loss(base_histories, continuation_histories)
    plot_timing(continuation_histories)
    plot_full_paloma(base_histories, continuation_histories)
    plot_nested_offsets(continuation_histories)


if __name__ == "__main__":
    main()
