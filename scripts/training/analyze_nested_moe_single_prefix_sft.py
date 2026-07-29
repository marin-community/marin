#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze the selected three-arm single-prefix nested-MoE SFT runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import wandb

from scripts.training.analyze_nested_moe_burnin_sft import _paired_comparison, _run_summary

ENTITY = "marin-community"
PROJECT = "marin_moe_sft"
TOKENS_PER_STEP = 32 * 8192
GPU_COUNT = 8
OUTPUT_DIR = Path("docs/reports/assets")
OUTPUT_PREFIX = "nested-model-training-single-prefix-10b-sft"
STAGES = ("wildchat", "thinking")
ARMS = ("e256", "e128_naive", "e128_layerwise")
RUNS = {
    ("wildchat", "e256"): "nest-augdk-e256-wildchat-sft-r1",
    ("wildchat", "e128_naive"): "nest-augdk-e128-naive25-wildchat-sft-r1",
    ("wildchat", "e128_layerwise"): "nest-augdk-e128-layer25-wildchat-sft-r1",
    ("thinking", "e256"): "nest-augdk-e256-thinking-sft-r1",
    ("thinking", "e128_naive"): "nest-augdk-e128-naive25-thinking-sft-r1",
    ("thinking", "e128_layerwise"): "nest-augdk-e128-layer25-thinking-sft-r1",
}
LABELS = {
    "e256": "E256 control",
    "e128_naive": "E128 naive 25%",
    "e128_layerwise": "E128 layerwise 25%",
}
COLORS = {
    "e256": "#24292f",
    "e128_naive": "#0969da",
    "e128_layerwise": "#54aeff",
}
LINESTYLES = {
    "e256": "-",
    "e128_naive": "-",
    "e128_layerwise": "--",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--tokens-per-step", type=int, default=TOKENS_PER_STEP)
    parser.add_argument("--gpu-count", type=int, default=GPU_COUNT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    for stage in STAGES:
        for arm in ARMS:
            parser.add_argument(f"--{stage}-{arm.replace('_', '-')}-run", default=RUNS[(stage, arm)])
    return parser.parse_args()


def _plot_loss(
    result: dict[str, dict[str, dict[str, object]]],
    *,
    tokens_per_step: int,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for axis, stage in zip(axes, STAGES, strict=True):
        for arm in ARMS:
            histories = result[stage][arm]["histories"]
            assert isinstance(histories, dict)
            points = histories["cross_entropy_loss"] or histories["total_loss"]
            axis.plot(
                [point["step"] * tokens_per_step / 1e9 for point in points],
                [point["value"] for point in points],
                color=COLORS[arm],
                linestyle=LINESTYLES[arm],
                label=LABELS[arm],
                alpha=0.85,
            )
        axis.set_title(stage.capitalize())
        axis.set_xlabel("SFT tokens (billions)")
        axis.set_ylabel("Completion-masked training loss")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("Selected d768 nested-MoE post-training")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _without_histories(result: dict[str, dict[str, dict[str, object]]]) -> dict[str, object]:
    return {
        stage: {
            "runs": {
                arm: {key: value for key, value in result[stage][arm].items() if key != "histories"} for arm in ARMS
            },
            "comparisons_vs_e256": {
                arm: _paired_comparison({"e256": result[stage]["e256"], "fixed25": result[stage][arm]})
                for arm in ARMS
                if arm != "e256"
            },
        }
        for stage in STAGES
    }


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=60)
    result: dict[str, dict[str, dict[str, object]]] = {stage: {} for stage in STAGES}
    for stage in STAGES:
        for arm in ARMS:
            run_name = getattr(args, f"{stage}_{arm}_run")
            run = api.run(f"{args.entity}/{args.project}/{run_name}")
            if run.state != "finished":
                raise RuntimeError(f"W&B run {run.name} is {run.state}, expected finished")
            result[stage][arm] = _run_summary(
                run,
                tokens_per_step=args.tokens_per_step,
                gpu_count=args.gpu_count,
            )

    summary = _without_histories(result)
    result_path = args.output_dir / f"{args.output_prefix}-results.json"
    result_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _plot_loss(
        result,
        tokens_per_step=args.tokens_per_step,
        path=args.output_dir / f"{args.output_prefix}-loss.png",
    )


if __name__ == "__main__":
    main()
