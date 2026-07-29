#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze matched WildChat and thinking SFT stages for nested-MoE runs."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import matplotlib.pyplot as plt
import wandb

ENTITY = "marin-community"
PROJECT = "marin_moe_sft"
GLOBAL_STEP = "global_step"
TRAIN_LOSS = "train/cross_entropy_loss"
TOTAL_LOSS = "train/loss"
STEP_DURATION = "throughput/duration"
OVERFLOW = "train/router/capacity_overflow_rate_mean"
TOKENS_PER_STEP = 32 * 8192
GPU_COUNT = 8
LOSS_WARMUP_STEPS = 100
TAIL_LOSS_STEPS = 100
OUTPUT_DIR = Path("docs/reports/assets")
OUTPUT_PREFIX = "nested-model-training-corrected-augdk-sft"
RUNS = MappingProxyType(
    {
        ("wildchat", "e256"): "nest-augdk-e256-wildchat-sft-r1",
        ("wildchat", "fixed25"): "nest-augdk-fixed25-wildchat-sft-r1",
        ("thinking", "e256"): "nest-augdk-e256-thinking-sft-r1",
        ("thinking", "fixed25"): "nest-augdk-fixed25-thinking-sft-r1",
    }
)
LABELS = MappingProxyType({"e256": "E256 control", "fixed25": "Fixed25"})
COLORS = MappingProxyType({"e256": "#24292f", "fixed25": "#0969da"})


@dataclass(frozen=True)
class HistoryPoint:
    step: int
    value: float


def _history(run: wandb.apis.public.Run, metric: str) -> list[HistoryPoint]:
    by_step: dict[int, HistoryPoint] = {}
    for row in run.scan_history(keys=[GLOBAL_STEP, metric], page_size=10_000):
        step = row.get(GLOBAL_STEP)
        value = row.get(metric)
        if isinstance(step, (int, float)) and isinstance(value, (int, float)):
            by_step[int(step)] = HistoryPoint(int(step), float(value))
    return [by_step[step] for step in sorted(by_step)]


def _final(points: list[HistoryPoint]) -> float | None:
    return points[-1].value if points else None


def _mean(points: list[HistoryPoint]) -> float | None:
    return statistics.fmean(point.value for point in points) if points else None


def _run_summary(
    run: wandb.apis.public.Run,
    *,
    tokens_per_step: int,
    gpu_count: int,
) -> dict[str, Any]:
    cross_entropy = _history(run, TRAIN_LOSS)
    total_loss = _history(run, TOTAL_LOSS)
    effective_loss = cross_entropy or total_loss
    step_duration = _history(run, STEP_DURATION)
    overflow = _history(run, OVERFLOW)
    post_warmup_loss = [point for point in effective_loss if point.step >= LOSS_WARMUP_STEPS]
    tail_loss = effective_loss[-TAIL_LOSS_STEPS:]
    post_warmup_duration = [point.value for point in step_duration if point.step >= LOSS_WARMUP_STEPS]
    completed_steps = max((point.step for point in effective_loss), default=-1) + 1
    runtime = run.summary.get("_runtime")
    runtime_seconds = float(runtime) if isinstance(runtime, (int, float)) else None
    median_step_seconds = statistics.median(post_warmup_duration) if post_warmup_duration else None
    return {
        "run_name": run.name,
        "url": run.url,
        "state": run.state,
        "stage_tokens": completed_steps * tokens_per_step,
        "completed_steps": completed_steps,
        "loss_metric": TRAIN_LOSS if cross_entropy else TOTAL_LOSS,
        "final_cross_entropy_loss": _final(effective_loss),
        "mean_cross_entropy_loss_post_warmup": _mean(post_warmup_loss),
        "mean_cross_entropy_loss_last_100": _mean(tail_loss),
        "final_total_loss": _final(total_loss),
        "terminal_overflow": _final(overflow),
        "median_step_seconds": median_step_seconds,
        "optimizer_gpu_hours": (
            median_step_seconds * completed_steps * gpu_count / 3600 if median_step_seconds is not None else None
        ),
        "runtime_seconds": runtime_seconds,
        "gpu_hours": runtime_seconds * gpu_count / 3600 if runtime_seconds is not None else None,
        "histories": {
            "cross_entropy_loss": [asdict(point) for point in cross_entropy],
            "total_loss": [asdict(point) for point in total_loss],
            "step_duration": [asdict(point) for point in step_duration],
            "overflow": [asdict(point) for point in overflow],
        },
    }


def _paired_comparison(stage_result: dict[str, dict[str, Any]]) -> dict[str, float | int | None]:
    control = stage_result["e256"]
    treatment = stage_result["fixed25"]

    def difference(metric: str) -> float | None:
        control_value = control[metric]
        treatment_value = treatment[metric]
        if control_value is None or treatment_value is None:
            return None
        return treatment_value - control_value

    control_step = control["median_step_seconds"]
    treatment_step = treatment["median_step_seconds"]
    control_loss = {
        point["step"]: point["value"]
        for point in control["histories"]["cross_entropy_loss"]
        if point["step"] >= LOSS_WARMUP_STEPS
    }
    treatment_loss = {
        point["step"]: point["value"]
        for point in treatment["histories"]["cross_entropy_loss"]
        if point["step"] >= LOSS_WARMUP_STEPS
    }
    common_steps = sorted(control_loss.keys() & treatment_loss.keys())
    paired_deltas = [treatment_loss[step] - control_loss[step] for step in common_steps]
    return {
        "final_cross_entropy_delta": difference("final_cross_entropy_loss"),
        "post_warmup_mean_cross_entropy_delta": difference("mean_cross_entropy_loss_post_warmup"),
        "last_100_mean_cross_entropy_delta": difference("mean_cross_entropy_loss_last_100"),
        "paired_steps_post_warmup": len(common_steps),
        "paired_treatment_wins_post_warmup": sum(delta < 0 for delta in paired_deltas),
        "paired_treatment_win_fraction_post_warmup": (
            sum(delta < 0 for delta in paired_deltas) / len(paired_deltas) if paired_deltas else None
        ),
        "paired_mean_cross_entropy_delta_post_warmup": statistics.fmean(paired_deltas) if paired_deltas else None,
        "paired_median_cross_entropy_delta_post_warmup": statistics.median(paired_deltas) if paired_deltas else None,
        "median_step_overhead_fraction": (
            treatment_step / control_step - 1.0
            if control_step is not None and treatment_step is not None and control_step > 0
            else None
        ),
        "runtime_delta_seconds": difference("runtime_seconds"),
        "gpu_hours_delta": difference("gpu_hours"),
    }


def _plot_loss(
    result: dict[str, dict[str, dict[str, Any]]],
    *,
    tokens_per_step: int,
    output_prefix: str,
    figure_title: str,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for axis, stage in zip(axes, ("wildchat", "thinking"), strict=True):
        for arm in ("e256", "fixed25"):
            points = result[stage][arm]["histories"]["cross_entropy_loss"]
            if not points:
                points = result[stage][arm]["histories"]["total_loss"]
            axis.plot(
                [point["step"] * tokens_per_step / 1e9 for point in points],
                [point["value"] for point in points],
                color=COLORS[arm],
                label=LABELS[arm],
                alpha=0.85,
            )
        axis.set_title(stage.capitalize())
        axis.set_xlabel("SFT tokens (billions)")
        axis.set_ylabel("Completion-masked cross-entropy")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle(figure_title)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"{output_prefix}-loss.png", dpi=180)
    plt.close(figure)


def _result_without_histories(
    result: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        stage: {
            arm: {key: value for key, value in arm_result.items() if key != "histories"}
            for arm, arm_result in stage_result.items()
        }
        for stage, stage_result in result.items()
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--tokens-per-step", type=int, default=TOKENS_PER_STEP)
    parser.add_argument("--gpu-count", type=int, default=GPU_COUNT)
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--figure-title", default="Corrected augmented d768 matched two-stage SFT")
    for stage in ("wildchat", "thinking"):
        for arm in ("e256", "fixed25"):
            parser.add_argument(f"--{stage}-{arm}-run", default=RUNS[(stage, arm)])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=60)
    result: dict[str, dict[str, dict[str, Any]]] = {"wildchat": {}, "thinking": {}}
    for stage in ("wildchat", "thinking"):
        for arm in ("e256", "fixed25"):
            run_name = getattr(args, f"{stage}_{arm}_run")
            run = api.run(f"{args.entity}/{args.project}/{run_name}")
            if run.state != "finished":
                raise RuntimeError(f"W&B run {run.name} is {run.state}, expected finished")
            result[stage][arm] = _run_summary(
                run,
                tokens_per_step=args.tokens_per_step,
                gpu_count=args.gpu_count,
            )
    for stage in ("wildchat", "thinking"):
        result[stage]["comparison"] = _paired_comparison(result[stage])

    summary_result = _result_without_histories(result)
    (OUTPUT_DIR / f"{args.output_prefix}-results.json").write_text(
        json.dumps(summary_result, indent=2, sort_keys=True) + "\n"
    )
    _plot_loss(
        result,
        tokens_per_step=args.tokens_per_step,
        output_prefix=args.output_prefix,
        figure_title=args.figure_title,
    )


if __name__ == "__main__":
    main()
