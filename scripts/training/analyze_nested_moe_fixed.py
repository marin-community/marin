#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze the matched fixed E16 ⊂ E128 ⊂ E256 experiment."""

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
GPU_COUNT = 64
TIMING_WARMUP_STEPS = 1024
TIMING_BLOCK_STEPS = 200
OUTPUT_DIR = Path("docs/reports/assets")

GLOBAL_STEP = "global_step"
TRAIN_LOSS = "train/loss"
CROSS_ENTROPY_LOSS = "train/cross_entropy_loss"
ROUTER_AUX_LOSS = "train/router/aux_loss_weighted"
STEP_DURATION = "throughput/duration"
HOOK_DURATION = "throughput/hook_time"
LOADING_TIME = "throughput/loading_time"
OVERFLOW = "train/router/capacity_overflow_rate_mean"
PALOMA_MACRO = "eval/paloma/macro_loss"
PALOMA_MICRO = "eval/paloma/micro_loss"
SLIMPAJAMA_LOSS = "eval/slimpajama-6b/loss"
FORECAST_TOKENS_BILLIONS = (10.0, 100.0, 1_000.0)

RUNS: Mapping[str, str] = MappingProxyType(
    {
        "large_control": "nest-moe-001-full-d768-s2048-e256-fixedep16-eqb-w512-cost-r47",
        "fixed25": "nest-moe-008-full-d768-s2048-e256-fixedep16-eqb-w512-cost-r47",
        "fixed50": "nest-moe-009-full-d768-s2048-e256-fixedep16-eqb-w512-cost-r47",
    }
)
LABELS: Mapping[str, str] = MappingProxyType(
    {
        "large_control": "E256 control",
        "fixed25": "Fixed chain 25%",
        "fixed50": "Fixed chain 50%",
    }
)
COLORS: Mapping[str, str] = MappingProxyType(
    {
        "large_control": "#24292f",
        "fixed25": "#0969da",
        "fixed50": "#cf222e",
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
    block_bootstrap_ci95_low: float
    block_bootstrap_ci95_high: float


def nested_paloma_metric(count: int) -> str:
    return f"eval/nested_e{count}/paloma/macro_loss"


def histories(run: wandb.apis.public.Run, *, include_nested: bool) -> dict[str, list[HistoryPoint]]:
    metrics = {
        "train_loss": TRAIN_LOSS,
        "cross_entropy_loss": CROSS_ENTROPY_LOSS,
        "router_aux_loss": ROUTER_AUX_LOSS,
        "step_duration": STEP_DURATION,
        "hook_duration": HOOK_DURATION,
        "loading_time": LOADING_TIME,
        "overflow": OVERFLOW,
        "paloma_macro": PALOMA_MACRO,
        "paloma_micro": PALOMA_MICRO,
        "slimpajama_loss": SLIMPAJAMA_LOSS,
    }
    if include_nested:
        metrics.update(
            {
                "paloma_e128": nested_paloma_metric(128),
                "paloma_e16": nested_paloma_metric(16),
            }
        )
    values = {name: [] for name in metrics}
    # W&B explicit-key scans return only rows containing every requested key.
    # Router metrics, evaluation suites, and nested modes are optional, so a
    # grouped scan can silently discard ordinary loss and timing rows when any
    # requested metric is absent. Scan each metric independently.
    metric_groups = ((metrics, 1),)
    for group, chunk_size in metric_groups:
        items = list(group.items())
        for start in range(0, len(items), chunk_size):
            chunk = dict(items[start : start + chunk_size])
            for row in run.scan_history(keys=[GLOBAL_STEP, *chunk.values()], page_size=10_000):
                step = row.get(GLOBAL_STEP)
                if not isinstance(step, (int, float)):
                    continue
                for name, metric in chunk.items():
                    value = row.get(metric)
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        values[name].append(HistoryPoint(int(step), float(value)))
    values.setdefault("paloma_e128", [])
    values.setdefault("paloma_e16", [])
    for name, points in values.items():
        by_step = {point.step: point for point in points}
        values[name] = [by_step[step] for step in sorted(by_step)]
    return values


def timing_summary(points: list[HistoryPoint]) -> TimingSummary:
    values = [point.value for point in points if point.step >= TIMING_WARMUP_STEPS]
    if not values:
        raise ValueError("No post-warmup step-duration samples")
    blocks = [
        values[index : index + TIMING_BLOCK_STEPS]
        for index in range(0, len(values), TIMING_BLOCK_STEPS)
        if len(values[index : index + TIMING_BLOCK_STEPS]) == TIMING_BLOCK_STEPS
    ]
    if len(blocks) >= 2:
        block_medians = np.asarray([statistics.median(block) for block in blocks])
        rng = np.random.default_rng(20260727)
        samples = rng.choice(block_medians, size=(10_000, len(block_medians)), replace=True)
        low, high = np.quantile(np.median(samples, axis=1), (0.025, 0.975))
    else:
        low = high = math.nan
    return TimingSummary(
        samples=len(values),
        median_step_seconds=statistics.median(values),
        p10_step_seconds=float(np.quantile(values, 0.1)),
        p90_step_seconds=float(np.quantile(values, 0.9)),
        block_bootstrap_ci95_low=float(low),
        block_bootstrap_ci95_high=float(high),
    )


def binned_medians(points: list[HistoryPoint], width: int = 100) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(points, key=lambda point: point.step)
    bins = [ordered[index : index + width] for index in range(0, len(ordered), width)]
    x = np.asarray([statistics.median(point.step for point in group) for group in bins if group])
    y = np.asarray([statistics.median(point.value for point in group) for group in bins if group])
    return x, y


def final_value(points: list[HistoryPoint]) -> float | None:
    if not points:
        return None
    return max(points, key=lambda point: point.step).value


def domain_losses(run: wandb.apis.public.Run, prefix: str) -> dict[str, float]:
    suffix = "/loss"
    return {
        key.removeprefix(prefix).removesuffix(suffix): float(value)
        for key, value in run.summary.items()
        if key.startswith(prefix)
        and key.endswith(suffix)
        and not key.endswith(("/macro_loss", "/micro_loss"))
        and isinstance(value, (int, float))
    }


def paired_domains(treatment: Mapping[str, float], control: Mapping[str, float]) -> dict[str, Any]:
    common = sorted(treatment.keys() & control.keys())
    deltas = {domain: treatment[domain] - control[domain] for domain in common}
    values = list(deltas.values())
    if not values:
        return {"domains": 0, "deltas": {}}
    return {
        "domains": len(values),
        "treatment_better": sum(value < 0 for value in values),
        "mean_delta": statistics.fmean(values),
        "median_delta": statistics.median(values),
        "min_delta": min(values),
        "max_delta": max(values),
        "deltas": deltas,
    }


def plot_paloma(all_history: Mapping[str, Mapping[str, list[HistoryPoint]]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for arm, history in all_history.items():
        points = history["paloma_macro"]
        axes[0].plot(
            [point.step * TOKENS_PER_STEP / 1e9 for point in points],
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

    for arm in ("fixed25", "fixed50"):
        history = all_history[arm]
        for count, linestyle in ((128, "-"), (16, "--")):
            points = history[f"paloma_e{count}"]
            axes[1].plot(
                [point.step * TOKENS_PER_STEP / 1e9 for point in points],
                [point.value for point in points],
                marker="o",
                linestyle=linestyle,
                color=COLORS[arm],
                label=f"{LABELS[arm]} E{count}",
            )
    axes[1].set_title("Extractable fixed subsets")
    axes[1].set_xlabel("Training tokens (billions)")
    axes[1].set_ylabel("Paloma macro loss (lower is better)")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    figure.suptitle("Fixed E16 ⊂ E128 ⊂ E256 co-training")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "nested-model-training-fixed-paloma.png", dpi=180)
    plt.close(figure)


def plot_training_loss(all_history: Mapping[str, Mapping[str, list[HistoryPoint]]]) -> None:
    figure, axis = plt.subplots(figsize=(9, 4.8))
    for arm, history in all_history.items():
        x, y = binned_medians(history["cross_entropy_loss"])
        axis.plot(x * TOKENS_PER_STEP / 1e9, y, color=COLORS[arm], label=LABELS[arm])
    axis.set_title("Training cross-entropy (100-step medians; treatments use mixed routing modes)")
    axis.set_xlabel("Training tokens (billions)")
    axis.set_ylabel("Cross-entropy loss")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "nested-model-training-fixed-loss.png", dpi=180)
    plt.close(figure)


def plot_step_time(all_history: Mapping[str, Mapping[str, list[HistoryPoint]]]) -> None:
    figure, axis = plt.subplots(figsize=(9, 4.8))
    for arm, history in all_history.items():
        points = [point for point in history["step_duration"] if point.step >= TIMING_WARMUP_STEPS]
        x, y = binned_medians(points)
        axis.plot(x * TOKENS_PER_STEP / 1e9, y * 1_000, color=COLORS[arm], label=LABELS[arm])
    axis.set_title("Compiled optimizer-step time (100-step medians)")
    axis.set_xlabel("Training tokens (billions)")
    axis.set_ylabel("Step time (ms)")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "nested-model-training-fixed-step-time.png", dpi=180)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=60)
    runs = {arm: api.run(f"{ENTITY}/{PROJECT}/{run_id}") for arm, run_id in RUNS.items()}
    all_history = {arm: histories(run, include_nested=arm != "large_control") for arm, run in runs.items()}
    timing = {arm: timing_summary(history["step_duration"]) for arm, history in all_history.items()}
    baseline_step = timing["large_control"].median_step_seconds

    summaries: dict[str, dict[str, Any]] = {}
    for arm, run in runs.items():
        history = all_history[arm]
        last_step = max((point.step for point in history["train_loss"]), default=-1)
        median_step = timing[arm].median_step_seconds
        summaries[arm] = {
            "run_name": RUNS[arm],
            "url": run.url,
            "state": run.state,
            "final_step": last_step,
            "tokens_billions": (last_step + 1) * TOKENS_PER_STEP / 1e9,
            "full_paloma_macro": final_value(history["paloma_macro"]),
            "full_paloma_micro": final_value(history["paloma_micro"]),
            "slimpajama_loss": final_value(history["slimpajama_loss"]),
            "e128_paloma_macro": final_value(history["paloma_e128"]),
            "e16_paloma_macro": final_value(history["paloma_e16"]),
            "final_cross_entropy_loss": final_value(history["cross_entropy_loss"]),
            "final_router_aux_loss": final_value(history["router_aux_loss"]),
            "terminal_overflow": final_value(history["overflow"]),
            "median_step_seconds": median_step,
            "step_overhead_vs_e256": median_step / baseline_step - 1.0,
            "gpu_hours_per_billion_tokens": 1e9 / TOKENS_PER_STEP * median_step * GPU_COUNT / 3600,
            "runtime_forecasts": {
                str(tokens_billions): {
                    "steps": round(tokens_billions * 1e9 / TOKENS_PER_STEP),
                    "optimizer_hours": tokens_billions * 1e9 / TOKENS_PER_STEP * median_step / 3600,
                    "gpu_hours": tokens_billions * 1e9 / TOKENS_PER_STEP * median_step * GPU_COUNT / 3600,
                }
                for tokens_billions in FORECAST_TOKENS_BILLIONS
            },
        }

    control_domains = domain_losses(runs["large_control"], "eval/paloma/")
    domain_comparisons = {
        arm: paired_domains(domain_losses(runs[arm], "eval/paloma/"), control_domains) for arm in ("fixed25", "fixed50")
    }
    result = {
        "schema_version": 1,
        "tokens_per_step": TOKENS_PER_STEP,
        "gpu_count": GPU_COUNT,
        "summaries": summaries,
        "timing": {arm: asdict(value) for arm, value in timing.items()},
        "full_mode_domain_comparisons": domain_comparisons,
    }
    (OUTPUT_DIR / "nested-model-training-fixed-results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    with (OUTPUT_DIR / "nested-model-training-fixed-summary.csv").open("w", newline="") as file:
        fieldnames = list(next(iter(summaries.values())).keys())
        writer = csv.DictWriter(file, fieldnames=["arm", *fieldnames])
        writer.writeheader()
        for arm, summary in summaries.items():
            writer.writerow({"arm": arm, **summary})

    plot_paloma(all_history)
    plot_training_loss(all_history)
    plot_step_time(all_history)


if __name__ == "__main__":
    main()
