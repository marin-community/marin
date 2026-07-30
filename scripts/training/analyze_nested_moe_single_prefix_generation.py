#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect selected single-prefix nested-MoE native generation evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY = "marin-community"
PROJECT = "marin_moe_sft"
OUTPUT_DIR = Path("docs/reports/assets")
OUTPUT_PREFIX = "nested-model-training-single-prefix-10b-generation"
ARMS = ("e256", "e128_naive", "e128_layerwise")
EXPERT_COUNTS = (256, 128)
RUNS = {
    ("e256", 256): "nest-augdk-e256-thinking-generation-e256-r2",
    ("e256", 128): "nest-augdk-e256-thinking-generation-e128-r2",
    ("e128_naive", 256): "nest-augdk-e128-naive25-thinking-generation-e256-r2",
    ("e128_naive", 128): "nest-augdk-e128-naive25-thinking-generation-e128-r2",
    ("e128_layerwise", 256): "nest-augdk-e128-layer25-thinking-generation-e256-r2",
    ("e128_layerwise", 128): "nest-augdk-e128-layer25-thinking-generation-e128-r2",
}
LABELS = {
    "e256": "E256 control checkpoint",
    "e128_naive": "E128 naive checkpoint",
    "e128_layerwise": "E128 layerwise checkpoint",
}
COLORS = {
    "e256": "#24292f",
    "e128_naive": "#0969da",
    "e128_layerwise": "#54aeff",
}
SCORE_METRICS = {
    "gsm8k_exact_match": "GSM8K exact match",
    "instruction_pass_rate": "Instruction pass rate",
}
LOSS_METRICS = ("paloma_macro_loss", "uncheatable_macro_loss")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    for arm in ARMS:
        for expert_count in EXPERT_COUNTS:
            parser.add_argument(
                f"--{arm.replace('_', '-')}-e{expert_count}-run",
                default=RUNS[(arm, expert_count)],
            )
    return parser.parse_args()


def _collect(args: argparse.Namespace) -> dict[str, dict[str, dict[str, float | str]]]:
    api = wandb.Api(timeout=60)
    result: dict[str, dict[str, dict[str, float | str]]] = {}
    for arm in ARMS:
        result[arm] = {}
        for expert_count in EXPERT_COUNTS:
            run_name = getattr(args, f"{arm}_e{expert_count}_run")
            run = api.run(f"{args.entity}/{args.project}/{run_name}")
            if run.state != "finished":
                raise RuntimeError(f"W&B run {run.name} is {run.state}, expected finished")
            values: dict[str, float | str] = {
                "run_name": run.name,
                "url": run.url,
                "checkpoint": str(run.config["checkpoint"]),
            }
            for metric in (*SCORE_METRICS, *LOSS_METRICS):
                value = run.summary.get(metric)
                if not isinstance(value, (int, float)):
                    raise ValueError(f"W&B run {run.name} has no numeric {metric}")
                values[metric] = float(value)
            runtime = run.summary.get("_runtime")
            if isinstance(runtime, (int, float)):
                values["runtime_seconds"] = float(runtime)
            result[arm][str(expert_count)] = values
    return result


def _plot(
    result: dict[str, dict[str, dict[str, float | str]]],
    *,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    positions = np.arange(len(EXPERT_COUNTS))
    width = 0.25
    panels = (
        ("gsm8k_exact_match", "GSM8K exact match", "Score"),
        ("paloma_macro_loss", "Paloma after SFT", "Macro loss (lower is better)"),
    )
    for axis, (metric, title, ylabel) in zip(axes, panels, strict=True):
        for arm_index, arm in enumerate(ARMS):
            values = [float(result[arm][str(count)][metric]) for count in EXPERT_COUNTS]
            offsets = positions + (arm_index - 1) * width
            bars = axis.bar(offsets, values, width, color=COLORS[arm], label=LABELS[arm])
            axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8)
        axis.set_title(title)
        axis.set_xticks(positions, [f"E{count} inference" for count in EXPERT_COUNTS])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set_ylim(0.0, 0.08)
    axes[1].set_ylim(3.15, 3.75)
    figure.suptitle("Selected d768 nested-MoE post-SFT generation")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = _collect(args)
    result_path = args.output_dir / f"{args.output_prefix}-results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _plot(result, path=args.output_dir / f"{args.output_prefix}.png")


if __name__ == "__main__":
    main()
