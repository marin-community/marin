#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare trained and post-hoc E128 controls for the nested-MoE experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

from scripts.training.analyze_nested_moe_burnin import _metric_history

ENTITY = "marin-community"
PRETRAIN_PROJECT = "marin_moe"
EVAL_PROJECT = "marin_moe_sft"
TOKENS_PER_STEP = 32 * 8192
OUTPUT_DIR = Path("docs/reports/assets")
OUTPUT_PREFIX = "nested-model-training-e128-controls"

TRAINED_RUNS = {
    "standalone": ("nest-augdk-e128-standalone-10b-r1", "eval/paloma/macro_loss"),
    "nested_naive": ("nest-augdk-e128-naive25-10b-r1", "eval/nested_e128/paloma/macro_loss"),
    "nested_layerwise": ("nest-augdk-e128-layer25-10b-r1", "eval/nested_e128/paloma/macro_loss"),
    "control_prefix": ("nest-augdk-e256-10b-r1", "eval/nested_e128/paloma/macro_loss"),
}
POSTHOC_RUNS = {
    "qb_bias": "nest-augdk-e256-10b-posthoc-e128-qb-bias-r2",
    "router_norm": "nest-augdk-e256-10b-posthoc-e128-router-norm-r2",
    "hybrid": "nest-augdk-e256-10b-posthoc-e128-hybrid-r2",
    "random": "nest-augdk-e256-10b-posthoc-e128-random-r2",
}
LABELS = {
    "standalone": "Standalone E128",
    "nested_naive": "Nested E128 naive 25%",
    "nested_layerwise": "Nested E128 layerwise 25%",
    "control_prefix": "E256 first-half chop",
    "qb_bias": "E256 QB-bias score chop",
    "router_norm": "E256 router-norm score chop",
    "hybrid": "E256 hybrid score chop",
    "random": "E256 random half",
}
COLORS = {
    "standalone": "#1a7f37",
    "nested_naive": "#0969da",
    "nested_layerwise": "#54aeff",
    "control_prefix": "#6e7781",
    "qb_bias": "#bf8700",
    "router_norm": "#8250df",
    "hybrid": "#d4a72c",
    "random": "#cf222e",
}


def _finished_run(api: wandb.Api, project: str, run_id: str) -> wandb.apis.public.Run:
    run = api.run(f"{ENTITY}/{project}/{run_id}")
    if run.state != "finished":
        raise RuntimeError(f"W&B run {run.name} is {run.state}, expected finished")
    return run


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=60)
    results: dict[str, dict[str, object]] = {}

    for name, (run_id, metric) in TRAINED_RUNS.items():
        run = _finished_run(api, PRETRAIN_PROJECT, run_id)
        history = _metric_history(run, metric)
        if not history:
            raise ValueError(f"W&B run {run.name} has no {metric} history")
        results[name] = {
            "run_name": run.name,
            "url": run.url,
            "metric": metric,
            "paloma_macro_loss": history[-1].value,
            "history": [{"step": point.step, "value": point.value} for point in history],
        }

    for name, run_id in POSTHOC_RUNS.items():
        run = _finished_run(api, EVAL_PROJECT, run_id)
        loss = run.summary.get("paloma_macro_loss")
        if not isinstance(loss, (int, float)):
            raise ValueError(f"W&B run {run.name} has no numeric paloma_macro_loss")
        results[name] = {
            "run_name": run.name,
            "url": run.url,
            "paloma_macro_loss": float(loss),
            "selection_method": run.config["selection_method"],
        }

    result_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}-results.json"
    result_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    for name in TRAINED_RUNS:
        history = results[name]["history"]
        assert isinstance(history, list)
        axes[0].plot(
            [float(point["step"]) * TOKENS_PER_STEP / 1e9 for point in history],
            [float(point["value"]) for point in history],
            marker="o",
            color=COLORS[name],
            label=LABELS[name],
        )
    axes[0].set_title("E128 Paloma through training")
    axes[0].set_xlabel("Training tokens (billions)")
    axes[0].set_ylabel("Paloma macro loss (lower is better)")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)

    names = tuple(TRAINED_RUNS) + tuple(POSTHOC_RUNS)
    values = [float(results[name]["paloma_macro_loss"]) for name in names]
    positions = np.arange(len(names))
    bars = axes[1].bar(positions, values, color=[COLORS[name] for name in names])
    axes[1].bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8)
    axes[1].set_title("E128 checkpoints at the 10B-token endpoint")
    axes[1].set_xticks(positions, [LABELS[name] for name in names], rotation=35, ha="right")
    axes[1].set_ylabel("Paloma macro loss (lower is better)")
    axes[1].grid(axis="y", alpha=0.2)

    figure.suptitle("Matched E128 training and post-hoc controls")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
