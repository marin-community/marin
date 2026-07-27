#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze the long-run nested-MoE timing and power-ladder experiment."""

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
from scipy.optimize import curve_fit

ENTITY = "marin-community"
PROJECT = "marin_moe"
TOKENS_PER_STEP = 256 * 2048
GPU_COUNT = 64
WARMUP_STEPS = 50
TIMING_BLOCK_STEPS = 200
EVAL_INTERVAL = 2048
OUTPUT_DIR = Path("docs/reports/assets")

GLOBAL_STEP_METRIC = "global_step"
TRAIN_LOSS_METRIC = "train/loss"
STEP_DURATION_METRIC = "throughput/duration"
THROUGHPUT_METRIC = "throughput/tokens_per_second"
HOOK_TIME_METRIC = "throughput/hook_time"
LOADING_TIME_METRIC = "throughput/loading_time"
OVERFLOW_METRIC = "train/router/capacity_overflow_rate_mean"
PALOMA_MACRO_METRIC = "eval/paloma/macro_loss"
PALOMA_MICRO_METRIC = "eval/paloma/micro_loss"
NESTED_PALOMA_METRICS: Mapping[int, str] = MappingProxyType(
    {count: f"eval/nested_e{count}/paloma/macro_loss" for count in (128, 32, 8, 1)}
)

RUNS: Mapping[str, str] = MappingProxyType(
    {
        "large_control": "nest-moe-001-full-d768-s2048-e256-cost-r25",
        "small_control": "nest-moe-002-full-d768-s2048-e128-cost-r25",
        "ladder25": "nest-moe-006-full-d768-s2048-e256-cost-r25",
        "ladder50": "nest-moe-007-full-d768-s2048-e256-cost-r25",
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


def _history(run: wandb.apis.public.Run, metric: str) -> list[HistoryPoint]:
    rows = []
    for row in run.scan_history(keys=[GLOBAL_STEP_METRIC, metric]):
        step = row.get(GLOBAL_STEP_METRIC)
        value = row.get(metric)
        if step is None or value is None:
            continue
        rows.append(HistoryPoint(step=int(step), value=float(value)))
    return rows


def _summary_number(run: wandb.apis.public.Run, metric: str) -> float:
    value = run.summary.get(metric)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric W&B summary key {metric!r} in {run.name}")
    return float(value)


def _domain_losses(run: wandb.apis.public.Run, prefix: str) -> dict[str, float]:
    suffix = "/loss"
    return {
        key.removeprefix(prefix).removesuffix(suffix): float(value)
        for key, value in run.summary.items()
        if key.startswith(prefix)
        and key.endswith(suffix)
        and not key.endswith(("/macro_loss", "/micro_loss"))
        and isinstance(value, (int, float))
    }


def _paired_domain_summary(treatment: Mapping[str, float], control: Mapping[str, float]) -> dict[str, Any]:
    if treatment.keys() != control.keys():
        raise ValueError("Paloma domain keys do not match")
    deltas = {domain: treatment[domain] - control[domain] for domain in treatment}
    values = list(deltas.values())
    return {
        "domains": len(values),
        "treatment_better": sum(value < 0 for value in values),
        "mean_delta": statistics.fmean(values),
        "median_delta": statistics.median(values),
        "min_delta": min(values),
        "max_delta": max(values),
        "deltas": deltas,
    }


def _block_bootstrap_ci(values: list[HistoryPoint]) -> tuple[float, float]:
    blocks = [
        [row.value for row in values[index : index + TIMING_BLOCK_STEPS]]
        for index in range(0, len(values), TIMING_BLOCK_STEPS)
        if len(values[index : index + TIMING_BLOCK_STEPS]) == TIMING_BLOCK_STEPS
    ]
    if len(blocks) < 2:
        return math.nan, math.nan
    block_medians = np.asarray([statistics.median(block) for block in blocks])
    rng = np.random.default_rng(20260727)
    samples = rng.choice(block_medians, size=(10_000, len(block_medians)), replace=True)
    bootstrap_medians = np.median(samples, axis=1)
    low, high = np.quantile(bootstrap_medians, (0.025, 0.975))
    return float(low), float(high)


def _timing_summary(duration: list[HistoryPoint]) -> TimingSummary:
    steady = [row for row in duration if row.step >= WARMUP_STEPS]
    values = [row.value for row in steady]
    ci_low, ci_high = _block_bootstrap_ci(steady)
    median_step = statistics.median(values)
    return TimingSummary(
        samples=len(values),
        median_step_seconds=median_step,
        p10_step_seconds=float(np.quantile(values, 0.1)),
        p90_step_seconds=float(np.quantile(values, 0.9)),
        median_tokens_per_second=TOKENS_PER_STEP / median_step,
        block_bootstrap_ci95_low=ci_low,
        block_bootstrap_ci95_high=ci_high,
    )


def _power_law(tokens_billions: np.ndarray, floor: float, scale: float, exponent: float) -> np.ndarray:
    return floor + scale * np.power(tokens_billions, -exponent)


def _loss_projection(loss: list[HistoryPoint]) -> dict[str, Any]:
    block_size = 100
    blocks = [loss[index : index + block_size] for index in range(0, len(loss), block_size)]
    complete = [block for block in blocks if len(block) == block_size]
    tokens = np.asarray(
        [statistics.fmean((row.step + 1) * TOKENS_PER_STEP / 1e9 for row in block) for block in complete]
    )
    values = np.asarray([statistics.fmean(row.value for row in block) for block in complete])
    late = tokens >= 0.2 * tokens[-1]
    tokens = tokens[late]
    values = values[late]
    split = max(3, round(0.8 * len(tokens)))
    train_tokens = tokens[:split]
    train_values = values[:split]
    test_tokens = tokens[split:]
    test_values = values[split:]

    initial_floor = max(0.0, float(train_values[-1] - 1.0))
    params, _ = curve_fit(
        _power_law,
        train_tokens,
        train_values,
        p0=(initial_floor, float(train_values[0] - initial_floor), 0.2),
        bounds=((0.0, 0.0, 0.01), (float(train_values.min()), np.inf, 2.0)),
        maxfev=50_000,
    )
    log_slope, log_intercept = np.polyfit(np.log(train_tokens), train_values, 1)

    def rmse(prediction: np.ndarray) -> float:
        if len(test_values) == 0:
            return math.nan
        return float(np.sqrt(np.mean(np.square(prediction - test_values))))

    targets = sorted({float(tokens[-1]), 8.0, 16.0, 32.0})
    return {
        "fit_start_tokens_billions": float(tokens[0]),
        "fit_train_end_tokens_billions": float(train_tokens[-1]),
        "held_out_end_tokens_billions": float(tokens[-1]),
        "power_law": {
            "floor": float(params[0]),
            "scale": float(params[1]),
            "exponent": float(params[2]),
            "held_out_rmse": rmse(_power_law(test_tokens, *params)),
            "projections": {str(target): float(_power_law(np.asarray(target), *params)) for target in targets},
        },
        "log_linear_sensitivity": {
            "intercept": float(log_intercept),
            "log_tokens_slope": float(log_slope),
            "held_out_rmse": rmse(log_intercept + log_slope * np.log(test_tokens)),
            "projections": {str(target): float(log_intercept + log_slope * math.log(target)) for target in targets},
        },
    }


def _runtime_model(
    run: wandb.apis.public.Run,
    timing: TimingSummary,
    duration: list[HistoryPoint],
    hook: list[HistoryPoint],
    loading: list[HistoryPoint],
) -> dict[str, Any]:
    runtime = _summary_number(run, "_runtime")
    duration_sum = sum(row.value for row in duration)
    hook_sum = sum(row.value for row in hook)
    loading_sum = sum(row.value for row in loading)
    final_step = max(row.step for row in hook)
    eval_hooks = [
        row.value for row in hook if (row.step > 0 and row.step % EVAL_INTERVAL == 0) or row.step == final_step
    ]
    ordinary_hooks = [
        row.value for row in hook if not ((row.step > 0 and row.step % EVAL_INTERVAL == 0) or row.step == final_step)
    ]
    ordinary_hook = statistics.median(ordinary_hooks)
    eval_increment = max(0.0, statistics.median(eval_hooks) - ordinary_hook) if eval_hooks else 0.0
    fixed = max(0.0, runtime - duration_sum - hook_sum - loading_sum)

    forecasts = {}
    for target_billions in (4.294967296, 8.0, 16.0, 32.0, 1000.0):
        steps = math.ceil(target_billions * 1e9 / TOKENS_PER_STEP)
        evaluations = math.ceil(steps / EVAL_INTERVAL)
        seconds = fixed + steps * timing.median_step_seconds + steps * ordinary_hook + evaluations * eval_increment
        forecasts[str(target_billions)] = {
            "steps": steps,
            "wall_seconds": seconds,
            "wall_hours": seconds / 3600,
            "gpu_hours": seconds * GPU_COUNT / 3600,
        }
    return {
        "wandb_runtime_seconds": runtime,
        "observed_step_compute_seconds": duration_sum,
        "observed_callback_seconds": hook_sum,
        "observed_loading_seconds": loading_sum,
        "estimated_fixed_seconds": fixed,
        "ordinary_hook_seconds_per_step": ordinary_hook,
        "evaluation_increment_seconds": eval_increment,
        "forecasts": forecasts,
    }


def _rolling_median(rows: list[HistoryPoint], window: int) -> tuple[list[int], list[float]]:
    steps = []
    values = []
    for index in range(window - 1, len(rows)):
        steps.append(rows[index].step)
        values.append(statistics.median(row.value for row in rows[index - window + 1 : index + 1]))
    return steps, values


def _plot_train_loss(histories: Mapping[str, Mapping[str, list[HistoryPoint]]]) -> None:
    fig, axis = plt.subplots(figsize=(9, 5.2))
    for arm in RUNS:
        rows = histories[arm]["train_loss"]
        steps, values = _rolling_median(rows, 200)
        axis.plot(
            [(step + 1) * TOKENS_PER_STEP / 1e9 for step in steps],
            values,
            color=COLORS[arm],
            label=LABELS[arm],
        )
    axis.set_xlabel("Training tokens (billions)")
    axis.set_ylabel("Training cross-entropy loss")
    axis.set_title("Long-run nested-MoE loss")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "nested-model-training-cost-loss.png", dpi=180)
    plt.close(fig)


def _plot_step_time(histories: Mapping[str, Mapping[str, list[HistoryPoint]]]) -> None:
    fig, axis = plt.subplots(figsize=(9, 5.2))
    for arm in RUNS:
        rows = [row for row in histories[arm]["duration"] if row.step >= WARMUP_STEPS]
        steps, values = _rolling_median(rows, 200)
        axis.plot(steps, values, color=COLORS[arm], label=LABELS[arm])
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Step compute time (seconds)")
    axis.set_title("Steady-state optimizer-step time")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "nested-model-training-step-time.png", dpi=180)
    plt.close(fig)


def _plot_paloma(histories: Mapping[str, Mapping[str, list[HistoryPoint]]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for arm in RUNS:
        rows = histories[arm]["paloma"]
        axes[0].plot(
            [(row.step + 1) * TOKENS_PER_STEP / 1e9 for row in rows],
            [row.value for row in rows],
            marker="o",
            color=COLORS[arm],
            label=LABELS[arm],
        )
    axes[0].set_title("Full model")

    level_colors = {128: "#0969da", 32: "#8250df", 8: "#bf8700", 1: "#1a7f37"}
    for axis, arm in zip(axes[1:], ("ladder25", "ladder50"), strict=True):
        for count, color in level_colors.items():
            rows = histories[arm][f"paloma_e{count}"]
            axis.plot(
                [(row.step + 1) * TOKENS_PER_STEP / 1e9 for row in rows],
                [row.value for row in rows],
                marker="o",
                color=color,
                label=f"E{count}",
            )
        axis.set_title(LABELS[arm])

    for axis in axes:
        axis.set_xlabel("Training tokens (billions)")
        axis.set_ylabel("Paloma macro loss")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "nested-model-training-power-ladder-paloma.png", dpi=180)
    plt.close(fig)


def _write_timing_csv(summaries: Mapping[str, Mapping[str, Any]]) -> None:
    fields = [
        "arm",
        "run_name",
        "final_step",
        "tokens_billions",
        "runtime_seconds",
        "gpu_hours",
        "median_step_seconds",
        "step_ci95_low",
        "step_ci95_high",
        "median_tokens_per_second",
        "steady_step_overhead_vs_e256",
        "final_paloma_macro_loss",
        "final_paloma_micro_loss",
        "mean_overflow",
        "url",
    ]
    with (OUTPUT_DIR / "nested-model-training-cost-summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for _arm, summary in summaries.items():
            writer.writerow({field: summary.get(field) for field in fields})


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=60)
    runs = {arm: api.run(f"{ENTITY}/{PROJECT}/{run_name}") for arm, run_name in RUNS.items()}
    histories: dict[str, dict[str, list[HistoryPoint]]] = {}
    timings = {}
    runtime_models = {}
    projections = {}

    for arm, run in runs.items():
        histories[arm] = {
            "train_loss": _history(run, TRAIN_LOSS_METRIC),
            "duration": _history(run, STEP_DURATION_METRIC),
            "throughput": _history(run, THROUGHPUT_METRIC),
            "hook": _history(run, HOOK_TIME_METRIC),
            "loading": _history(run, LOADING_TIME_METRIC),
            "overflow": _history(run, OVERFLOW_METRIC),
            "paloma": _history(run, PALOMA_MACRO_METRIC),
        }
        for count, metric in NESTED_PALOMA_METRICS.items():
            histories[arm][f"paloma_e{count}"] = _history(run, metric)
        timing = _timing_summary(histories[arm]["duration"])
        timings[arm] = timing
        runtime_models[arm] = _runtime_model(
            run,
            timing,
            histories[arm]["duration"],
            histories[arm]["hook"],
            histories[arm]["loading"],
        )
        projections[arm] = _loss_projection(histories[arm]["train_loss"])

    control_step = timings["large_control"].median_step_seconds
    summaries = {}
    for arm, run in runs.items():
        timing = timings[arm]
        overflow = [row.value for row in histories[arm]["overflow"]]
        final_step = int(_summary_number(run, GLOBAL_STEP_METRIC))
        runtime = _summary_number(run, "_runtime")
        summaries[arm] = {
            "arm": arm,
            "run_name": run.name,
            "state": run.state,
            "url": run.url,
            "final_step": final_step,
            "tokens_billions": (final_step + 1) * TOKENS_PER_STEP / 1e9,
            "runtime_seconds": runtime,
            "gpu_hours": runtime * GPU_COUNT / 3600,
            "median_step_seconds": timing.median_step_seconds,
            "step_ci95_low": timing.block_bootstrap_ci95_low,
            "step_ci95_high": timing.block_bootstrap_ci95_high,
            "median_tokens_per_second": timing.median_tokens_per_second,
            "steady_step_overhead_vs_e256": timing.median_step_seconds / control_step - 1.0,
            "final_paloma_macro_loss": _summary_number(run, PALOMA_MACRO_METRIC),
            "final_paloma_micro_loss": _summary_number(run, PALOMA_MICRO_METRIC),
            "final_nested_paloma": {
                str(count): run.summary.get(metric) for count, metric in NESTED_PALOMA_METRICS.items()
            },
            "mean_overflow": statistics.fmean(overflow),
            "max_overflow": max(overflow),
            "terminal_overflow": overflow[-1],
        }

    domain_comparisons = {}
    large_domains = _domain_losses(runs["large_control"], "eval/paloma/")
    small_domains = _domain_losses(runs["small_control"], "eval/paloma/")
    for arm in ("ladder25", "ladder50"):
        domain_comparisons[f"{arm}_full_vs_large_control"] = _paired_domain_summary(
            _domain_losses(runs[arm], "eval/paloma/"), large_domains
        )
        domain_comparisons[f"{arm}_e128_vs_small_control"] = _paired_domain_summary(
            _domain_losses(runs[arm], "eval/nested_e128/paloma/"), small_domains
        )

    artifact = {
        "schema_version": 1,
        "tokens_per_step": TOKENS_PER_STEP,
        "gpu_count": GPU_COUNT,
        "runs": dict(RUNS),
        "summaries": summaries,
        "timing": {arm: asdict(summary) for arm, summary in timings.items()},
        "runtime_models": runtime_models,
        "loss_projections": projections,
        "domain_comparisons": domain_comparisons,
    }
    with (OUTPUT_DIR / "nested-model-training-cost-results.json").open("w") as output:
        json.dump(artifact, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
    _write_timing_csv(summaries)
    _plot_train_loss(histories)
    _plot_step_time(histories)
    _plot_paloma(histories)


if __name__ == "__main__":
    main()
