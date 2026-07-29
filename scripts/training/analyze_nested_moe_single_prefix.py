#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze the matched 10B-token single-prefix and layerwise nested-MoE sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import wandb

from scripts.training.analyze_nested_moe_burnin import (
    _latest_aligned_domains,
    _metric_history,
    _paired_delta_summary,
)
from scripts.training.analyze_nested_moe_fixed import (
    TIMING_WARMUP_STEPS,
    HistoryPoint,
    TimingSummary,
    binned_medians,
    final_value,
    histories,
    timing_summary,
)

ENTITY = "marin-community"
PROJECT = "marin_moe"
TOKENS_PER_STEP = 32 * 8192
GPU_COUNT = 8
DATAKIT_PHASE_STEP = 29_184
OUTPUT_DIR = Path("docs/reports/assets")
OUTPUT_PREFIX = "nested-model-training-single-prefix-10b"

CONTROL = "e256"
RUNS = {
    CONTROL: "nest-augdk-e256-10b-r1",
    "e128_naive": "nest-augdk-e128-naive25-10b-r1",
    "e16_naive": "nest-augdk-e16-naive25-10b-r1",
    "e128_layerwise": "nest-augdk-e128-layer25-10b-r1",
    "e16_layerwise": "nest-augdk-e16-layer25-10b-r1",
}
LABELS = {
    CONTROL: "E256 control",
    "e128_naive": "E128 naive 25%",
    "e16_naive": "E16 naive 25%",
    "e128_layerwise": "E128 layerwise 25%",
    "e16_layerwise": "E16 layerwise 25%",
}
COLORS = {
    CONTROL: "#24292f",
    "e128_naive": "#0969da",
    "e16_naive": "#cf222e",
    "e128_layerwise": "#54aeff",
    "e16_layerwise": "#ff8182",
}
LINESTYLES = {
    CONTROL: "-",
    "e128_naive": "-",
    "e16_naive": "-",
    "e128_layerwise": "--",
    "e16_layerwise": "--",
}
PREFIX_ARMS = {
    128: (CONTROL, "e128_naive", "e128_layerwise"),
    16: (CONTROL, "e16_naive", "e16_layerwise"),
}
PREFIX_COUNT = {
    "e128_naive": 128,
    "e16_naive": 16,
    "e128_layerwise": 128,
    "e16_layerwise": 16,
}
EVALUATION_METRICS = (
    "paloma_macro",
    "paloma_micro",
    "paloma_e128",
    "paloma_e16",
    "uncheatable_macro",
    "uncheatable_e128",
    "uncheatable_e16",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def _common_horizon(
    all_history: Mapping[str, Mapping[str, list[HistoryPoint]]],
    metric: str,
    arms: tuple[str, ...] | None = None,
) -> int:
    selected = arms or tuple(all_history)
    endpoints = [max(point.step for point in all_history[arm][metric]) for arm in selected if all_history[arm][metric]]
    if len(endpoints) != len(selected):
        return -1
    return min(endpoints)


def _plot_series(
    all_history: Mapping[str, Mapping[str, list[HistoryPoint]]],
    *,
    metric: str,
    title: str,
    ylabel: str,
    path: Path,
    scale: float = 1.0,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.2))
    horizon = _common_horizon(all_history, metric)
    for arm, history in all_history.items():
        points = [point for point in history[metric] if point.step <= horizon]
        x, y = binned_medians(points)
        axis.plot(
            x * TOKENS_PER_STEP / 1e9,
            y * scale,
            color=COLORS[arm],
            linestyle=LINESTYLES[arm],
            label=LABELS[arm],
        )
    axis.set_title(title)
    axis.set_xlabel("Training tokens (billions)")
    axis.set_ylabel(ylabel)
    if horizon >= DATAKIT_PHASE_STEP:
        axis.axvline(
            DATAKIT_PHASE_STEP * TOKENS_PER_STEP / 1e9,
            color="#6e7781",
            linestyle=":",
            label="Datakit phase change",
        )
    axis.grid(alpha=0.2)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_evaluations(
    all_history: Mapping[str, Mapping[str, list[HistoryPoint]]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    panels = (
        ("paloma_macro", tuple(RUNS), "Full E256 mode"),
        ("paloma_e128", PREFIX_ARMS[128], "Fixed E128 prefix"),
        ("paloma_e16", PREFIX_ARMS[16], "Fixed E16 prefix"),
    )
    for axis, (metric, arms, title) in zip(axes, panels, strict=True):
        horizon = _common_horizon(all_history, metric, arms)
        for arm in arms:
            points = [point for point in all_history[arm][metric] if point.step <= horizon]
            axis.plot(
                [point.step * TOKENS_PER_STEP / 1e9 for point in points],
                [point.value for point in points],
                marker="o",
                color=COLORS[arm],
                linestyle=LINESTYLES[arm],
                label=LABELS[arm],
            )
        axis.set_title(title)
        axis.set_xlabel("Training tokens (billions)")
        if horizon >= DATAKIT_PHASE_STEP:
            axis.axvline(
                DATAKIT_PHASE_STEP * TOKENS_PER_STEP / 1e9,
                color="#6e7781",
                linestyle=":",
                label="Datakit phase change",
            )
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylabel("Paloma macro loss (lower is better)")
    figure.suptitle("Matched d768 nested-MoE sweep")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _log_linear_fit(points: list[HistoryPoint], *, tail_points: int | None = None) -> dict[str, Any] | None:
    usable = [point for point in points if point.step > 0]
    if tail_points is not None:
        usable = usable[-tail_points:]
    if len(usable) < 3:
        return None
    tokens = np.asarray([point.step * TOKENS_PER_STEP / 1e9 for point in usable])
    losses = np.asarray([point.value for point in usable])
    slope, intercept = np.polyfit(np.log(tokens), losses, deg=1)
    fitted = intercept + slope * np.log(tokens)
    residual_sum = float(np.sum(np.square(losses - fitted)))
    total_sum = float(np.sum(np.square(losses - np.mean(losses))))
    return {
        "points": len(usable),
        "through_step": usable[-1].step,
        "through_tokens_billions": float(tokens[-1]),
        "slope_per_log_token": float(slope),
        "intercept": float(intercept),
        "r_squared": 1.0 - residual_sum / total_sum if total_sum > 0 else None,
    }


def _time_to_equivalent(
    control_points: list[HistoryPoint],
    treatment_points: list[HistoryPoint],
    control_step_seconds: float,
    treatment_step_seconds: float,
) -> dict[str, float] | None:
    control_fit = _log_linear_fit(control_points, tail_points=10)
    treatment_fit = _log_linear_fit(treatment_points, tail_points=10)
    if control_fit is None or treatment_fit is None:
        return None
    horizon = min(
        max(point.step for point in control_points),
        max(point.step for point in treatment_points),
    )
    control_loss = next(point.value for point in reversed(control_points) if point.step <= horizon)
    treatment_loss = next(point.value for point in reversed(treatment_points) if point.step <= horizon)
    slope = treatment_fit["slope_per_log_token"]
    if not isinstance(slope, float) or slope >= 0:
        return None
    horizon_tokens = horizon * TOKENS_PER_STEP / 1e9
    token_ratio = math.exp((control_loss - treatment_loss) / slope)
    equivalent_tokens = horizon_tokens * token_ratio
    return {
        "control_loss": control_loss,
        "treatment_loss": treatment_loss,
        "comparison_step": horizon,
        "comparison_tokens_billions": horizon_tokens,
        "treatment_tokens_to_control_loss_billions": equivalent_tokens,
        "extra_token_fraction": token_ratio - 1.0,
        "step_time_ratio": treatment_step_seconds / control_step_seconds,
        "time_ratio": token_ratio * treatment_step_seconds / control_step_seconds,
        "extra_time_fraction": token_ratio * treatment_step_seconds / control_step_seconds - 1.0,
    }


def _runtime_forecast(step_seconds: float, tokens_billions: float) -> dict[str, float | int]:
    steps = round(tokens_billions * 1e9 / TOKENS_PER_STEP)
    optimizer_hours = steps * step_seconds / 3600
    return {
        "steps": steps,
        "optimizer_hours": optimizer_hours,
        "gpu_hours": optimizer_hours * GPU_COUNT,
    }


def _rolling_time_to_equivalent(
    control_points: list[HistoryPoint],
    treatment_points: list[HistoryPoint],
    control_step_seconds: float,
    treatment_step_seconds: float,
) -> list[HistoryPoint]:
    treatment_by_step = {point.step: point for point in treatment_points}
    common_steps = [point.step for point in control_points if point.step in treatment_by_step]
    result = []
    for end_index in range(3, len(common_steps) + 1):
        end_step = common_steps[end_index - 1]
        estimate = _time_to_equivalent(
            [point for point in control_points if point.step <= end_step],
            [point for point in treatment_points if point.step <= end_step],
            control_step_seconds,
            treatment_step_seconds,
        )
        if estimate is not None:
            result.append(HistoryPoint(end_step, estimate["extra_time_fraction"]))
    return result


def _timing_summary(points: list[HistoryPoint]) -> TimingSummary:
    if any(point.step >= TIMING_WARMUP_STEPS for point in points):
        return timing_summary(points)
    if not points:
        raise ValueError("No step-duration samples")
    start = max(100, max(point.step for point in points) // 2)
    values = [point.value for point in points if point.step >= start]
    return TimingSummary(
        samples=len(values),
        median_step_seconds=statistics.median(values),
        p10_step_seconds=float(np.quantile(values, 0.1)),
        p90_step_seconds=float(np.quantile(values, 0.9)),
        block_bootstrap_ci95_low=math.nan,
        block_bootstrap_ci95_high=math.nan,
    )


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=60)
    runs = {arm: api.run(f"{ENTITY}/{args.project}/{run_id}") for arm, run_id in RUNS.items()}
    all_history = {arm: histories(run, include_nested=True) for arm, run in runs.items()}
    for arm, run in runs.items():
        all_history[arm]["uncheatable_macro"] = _metric_history(run, "eval/uncheatable_eval/macro_loss")
        all_history[arm]["uncheatable_e128"] = _metric_history(run, "eval/nested_e128/uncheatable_eval/macro_loss")
        all_history[arm]["uncheatable_e16"] = _metric_history(run, "eval/nested_e16/uncheatable_eval/macro_loss")
        prefix_count = PREFIX_COUNT.get(arm)
        all_history[arm]["nested_sequence_fraction"] = (
            _metric_history(run, f"train/nested/e{prefix_count}_sequence_fraction") if prefix_count else []
        )
        all_history[arm]["nested_layer_sequence_fraction"] = (
            _metric_history(run, f"train/nested/e{prefix_count}_layer_sequence_fraction") if prefix_count else []
        )

    timing_horizon = _common_horizon(all_history, "step_duration")
    timing = {
        arm: _timing_summary([point for point in history["step_duration"] if point.step <= timing_horizon])
        for arm, history in all_history.items()
    }
    control_step_seconds = timing[CONTROL].median_step_seconds
    summaries: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, dict[str, Any]] = {}
    time_to_equivalent_history: dict[str, list[HistoryPoint]] = {}
    for arm, run in runs.items():
        history = all_history[arm]
        last_step = max((point.step for point in history["train_loss"]), default=-1)
        step_seconds = timing[arm].median_step_seconds
        hook_seconds = [point.value for point in history["hook_duration"]]
        evaluation_hooks = [value for value in hook_seconds if value >= 10.0]
        run_runtime = run.summary.get("_runtime")
        elapsed_seconds = float(run_runtime) if isinstance(run_runtime, (int, float)) else None
        summaries[arm] = {
            "run_name": RUNS[arm],
            "url": run.url,
            "state": run.state,
            "final_step": last_step,
            "tokens_billions": (last_step + 1) * TOKENS_PER_STEP / 1e9,
            "full_paloma_macro": final_value(history["paloma_macro"]),
            "e128_paloma_macro": final_value(history["paloma_e128"]),
            "e16_paloma_macro": final_value(history["paloma_e16"]),
            "full_uncheatable_macro": final_value(history["uncheatable_macro"]),
            "e128_uncheatable_macro": final_value(history["uncheatable_e128"]),
            "e16_uncheatable_macro": final_value(history["uncheatable_e16"]),
            "nested_sequence_fraction": final_value(history["nested_sequence_fraction"]),
            "nested_layer_sequence_fraction": final_value(history["nested_layer_sequence_fraction"]),
            "median_step_seconds": step_seconds,
            "step_overhead_vs_e256": step_seconds / control_step_seconds - 1.0,
            "gpu_hours_per_billion_tokens": 1e9 / TOKENS_PER_STEP * step_seconds * GPU_COUNT / 3600,
            "optimizer_runtime_10b": _runtime_forecast(step_seconds, 10.0),
            "evaluation_hook_count": len(evaluation_hooks),
            "median_evaluation_hook_seconds": statistics.median(evaluation_hooks) if evaluation_hooks else None,
            "total_logged_hook_seconds": sum(hook_seconds),
            "elapsed_run_seconds": elapsed_seconds,
            "elapsed_gpu_hours": elapsed_seconds * GPU_COUNT / 3600 if elapsed_seconds is not None else None,
        }
        if arm == CONTROL:
            continue
        time_to_equivalent_history[arm] = _rolling_time_to_equivalent(
            all_history[CONTROL]["paloma_macro"],
            history["paloma_macro"],
            control_step_seconds,
            step_seconds,
        )
        comparisons[arm] = {
            "full_paloma": _paired_delta_summary(all_history[CONTROL]["paloma_macro"], history["paloma_macro"]),
            "full_uncheatable": _paired_delta_summary(
                all_history[CONTROL]["uncheatable_macro"], history["uncheatable_macro"]
            ),
            "paloma_domains": _latest_aligned_domains(runs[CONTROL], run, "eval/paloma/"),
            "uncheatable_domains": _latest_aligned_domains(runs[CONTROL], run, "eval/uncheatable_eval/"),
            "time_to_equivalent_paloma": _time_to_equivalent(
                all_history[CONTROL]["paloma_macro"],
                history["paloma_macro"],
                control_step_seconds,
                step_seconds,
            ),
            "trained_prefix_paloma": _paired_delta_summary(
                all_history[CONTROL][f"paloma_e{PREFIX_COUNT[arm]}"],
                history[f"paloma_e{PREFIX_COUNT[arm]}"],
            ),
            "trained_prefix_uncheatable": _paired_delta_summary(
                all_history[CONTROL][f"uncheatable_e{PREFIX_COUNT[arm]}"],
                history[f"uncheatable_e{PREFIX_COUNT[arm]}"],
            ),
        }

    result = {
        "schema_version": 1,
        "tokens_per_step": TOKENS_PER_STEP,
        "gpu_count": GPU_COUNT,
        "timing_through_step": timing_horizon,
        "summaries": summaries,
        "timing": {arm: asdict(value) for arm, value in timing.items()},
        "comparisons_vs_e256": comparisons,
        "time_to_equivalent_history": {
            arm: [asdict(point) for point in points] for arm, points in time_to_equivalent_history.items()
        },
        "layerwise_vs_naive": {
            f"e{count}": {
                "full_paloma": _paired_delta_summary(
                    all_history[naive]["paloma_macro"],
                    all_history[layerwise]["paloma_macro"],
                ),
                "prefix_paloma": _paired_delta_summary(
                    all_history[naive][f"paloma_e{count}"],
                    all_history[layerwise][f"paloma_e{count}"],
                ),
            }
            for count, naive, layerwise in (
                (128, "e128_naive", "e128_layerwise"),
                (16, "e16_naive", "e16_layerwise"),
            )
        },
        "quality_log_linear_fits": {
            arm: _log_linear_fit(history["paloma_macro"]) for arm, history in all_history.items()
        },
        "quality_tail_log_linear_fits": {
            arm: _log_linear_fit(history["paloma_macro"], tail_points=10) for arm, history in all_history.items()
        },
        "evaluation_history": {
            arm: {metric: [asdict(point) for point in history[metric]] for metric in EVALUATION_METRICS}
            for arm, history in all_history.items()
        },
    }
    result_path = args.output_dir / f"{args.output_prefix}-results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    with (args.output_dir / f"{args.output_prefix}-summary.csv").open("w", newline="") as file:
        fieldnames = list(next(iter(summaries.values())).keys())
        writer = csv.DictWriter(file, fieldnames=["arm", *fieldnames])
        writer.writeheader()
        for arm, summary in summaries.items():
            writer.writerow({"arm": arm, **summary})

    _plot_evaluations(all_history, args.output_dir / f"{args.output_prefix}-paloma.png")
    _plot_series(
        all_history,
        metric="train_loss",
        title="Training loss (100-step medians; treatment losses mix routing modes)",
        ylabel="Training loss",
        path=args.output_dir / f"{args.output_prefix}-loss.png",
    )
    _plot_series(
        all_history,
        metric="step_duration",
        title="Compiled optimizer-step time (100-step medians)",
        ylabel="Step time (ms)",
        path=args.output_dir / f"{args.output_prefix}-step-time.png",
        scale=1_000.0,
    )
    figure, axis = plt.subplots(figsize=(10, 5.2))
    for arm, points in time_to_equivalent_history.items():
        axis.plot(
            [point.step * TOKENS_PER_STEP / 1e9 for point in points],
            [point.value * 100 for point in points],
            marker="o",
            color=COLORS[arm],
            linestyle=LINESTYLES[arm],
            label=LABELS[arm],
        )
    axis.axhline(10.0, color="#cf222e", linestyle=":", label="10% viability threshold")
    if max((point.step for points in time_to_equivalent_history.values() for point in points), default=-1) >= (
        DATAKIT_PHASE_STEP
    ):
        axis.axvline(
            DATAKIT_PHASE_STEP * TOKENS_PER_STEP / 1e9,
            color="#6e7781",
            linestyle=":",
            label="Datakit phase change",
        )
    axis.set_title("Full-mode time to equivalent Paloma loss")
    axis.set_xlabel("Training tokens (billions)")
    axis.set_ylabel("Extra optimizer wall time vs E256 (%)")
    axis.grid(alpha=0.2)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(args.output_dir / f"{args.output_prefix}-time-to-equivalent.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
