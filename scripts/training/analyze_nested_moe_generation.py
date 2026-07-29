#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect the corrected nested-MoE native generation evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MappingProxyType

import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY = "marin-community"
PROJECT = "marin_moe_sft"
OUTPUT_DIR = Path("docs/reports/assets")
OUTPUT_PREFIX = "nested-model-training-corrected-augdk-generation"
EXPERT_COUNTS = (256, 128, 16)
ARMS = ("e256", "fixed25")
LABELS = MappingProxyType({"e256": "E256 control checkpoint", "fixed25": "Fixed25 checkpoint"})
COLORS = MappingProxyType({"e256": "#24292f", "fixed25": "#0969da"})
METRICS = MappingProxyType(
    {
        "gsm8k_exact_match": "GSM8K exact match",
        "instruction_pass_rate": "Instruction pass rate",
    }
)
PERPLEXITY_METRICS = ("paloma_macro_loss", "uncheatable_macro_loss")


def _run_id(arm: str, expert_count: int) -> str:
    return f"nest-augdk-{arm}-thinking-generation-e{expert_count}-r1"


def _collect(entity: str, project: str) -> dict[str, dict[str, dict[str, float | str]]]:
    api = wandb.Api(timeout=60)
    result: dict[str, dict[str, dict[str, float | str]]] = {}
    for arm in ARMS:
        result[arm] = {}
        for expert_count in EXPERT_COUNTS:
            run = api.run(f"{entity}/{project}/{_run_id(arm, expert_count)}")
            if run.state != "finished":
                raise RuntimeError(f"W&B run {run.name} is {run.state}, expected finished")
            values: dict[str, float | str] = {
                "run_name": run.name,
                "url": run.url,
                "checkpoint": str(run.config["checkpoint"]),
            }
            for metric in (*METRICS, *PERPLEXITY_METRICS):
                value = run.summary.get(metric)
                if not isinstance(value, (int, float)):
                    raise ValueError(f"W&B run {run.name} has no numeric {metric}")
                values[metric] = float(value)
            runtime = run.summary.get("_runtime")
            if isinstance(runtime, (int, float)):
                values["runtime_seconds"] = float(runtime)
            result[arm][str(expert_count)] = values
    return result


def _plot(result: dict[str, dict[str, dict[str, float | str]]], output_prefix: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    positions = np.arange(len(EXPERT_COUNTS))
    width = 0.36
    for axis, (metric, title) in zip(axes, METRICS.items(), strict=True):
        for arm_index, arm in enumerate(ARMS):
            values = [float(result[arm][str(count)][metric]) for count in EXPERT_COUNTS]
            offsets = positions + (arm_index - 0.5) * width
            bars = axis.bar(offsets, values, width, color=COLORS[arm], label=LABELS[arm])
            axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8)
        axis.set_title(title)
        axis.set_xticks(positions, [f"E{count}" for count in EXPERT_COUNTS])
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Score")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Corrected augmented d768 post-SFT generation")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"{output_prefix}.png", dpi=180)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = _collect(args.entity, args.project)
    (OUTPUT_DIR / f"{args.output_prefix}-results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _plot(result, args.output_prefix)


if __name__ == "__main__":
    main()
