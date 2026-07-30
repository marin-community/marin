#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare trained and post-hoc E128 controls for the nested-MoE experiment."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

from scripts.training.analyze_nested_moe_burnin import _metric_history

ENTITY = "marin-community"
PRETRAIN_PROJECT = "marin_moe"
EVAL_PROJECT = "marin_moe_sft"
TOKENS_PER_STEP = 32 * 8192
TRAIN_STEPS = 38_147
PHASE_BOUNDARY_STEP = 29_184
LOSS_ASYMPTOTE = 1.6
LOSS_EXPONENT = 0.0941
OUTPUT_DIR = Path("docs/reports/assets")
OUTPUT_PREFIX = "nested-model-training-e128-controls"

TRAINED_RUNS = {
    "standalone": ("nest-augdk-e128-standalone-10b-r1", "eval/paloma/macro_loss"),
    "nested_naive": ("nest-augdk-e128-naive25-10b-r1", "eval/nested_e128/paloma/macro_loss"),
    "nested_layerwise": ("nest-augdk-e128-layer25-10b-r1", "eval/nested_e128/paloma/macro_loss"),
    "control_prefix": ("nest-augdk-e256-10b-r1", "eval/nested_e128/paloma/macro_loss"),
}
TRAINED_UNCHEATABLE_METRICS = {
    "standalone": "eval/uncheatable_eval/macro_loss",
    "nested_naive": "eval/nested_e128/uncheatable_eval/macro_loss",
    "nested_layerwise": "eval/nested_e128/uncheatable_eval/macro_loss",
    "control_prefix": "eval/nested_e128/uncheatable_eval/macro_loss",
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


def _compute_multiplier_to_loss(observed_loss: float, target_loss: float) -> float:
    """Estimate the compute needed to move from an observed to a target loss."""

    if observed_loss <= LOSS_ASYMPTOTE or target_loss <= LOSS_ASYMPTOTE:
        raise ValueError("losses must exceed the scaling-law asymptote")
    return ((observed_loss - LOSS_ASYMPTOTE) / (target_loss - LOSS_ASYMPTOTE)) ** (1 / LOSS_EXPONENT)


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
        throughput = _metric_history(run, "throughput/tokens_per_second")
        if len(throughput) < 100:
            raise ValueError(f"W&B run {run.name} has fewer than 100 throughput samples")
        full_history = _metric_history(run, "eval/paloma/macro_loss")
        if not full_history:
            raise ValueError(f"W&B run {run.name} has no full-model Paloma history")
        uncheatable_history = _metric_history(run, TRAINED_UNCHEATABLE_METRICS[name])
        if not uncheatable_history:
            raise ValueError(f"W&B run {run.name} has no uncheatable evaluation history")
        results[name] = {
            "run_name": run.name,
            "url": run.url,
            "metric": metric,
            "paloma_macro_loss": history[-1].value,
            "uncheatable_macro_loss": uncheatable_history[-1].value,
            "full_paloma_macro_loss": full_history[-1].value,
            "terminal_mean_tokens_per_second": statistics.fmean(point.value for point in throughput[-100:]),
            "history": [{"step": point.step, "value": point.value} for point in history],
        }

    for name, run_id in POSTHOC_RUNS.items():
        run = _finished_run(api, EVAL_PROJECT, run_id)
        loss = run.summary.get("paloma_macro_loss")
        if not isinstance(loss, (int, float)):
            raise ValueError(f"W&B run {run.name} has no numeric paloma_macro_loss")
        uncheatable_loss = run.summary.get("uncheatable_macro_loss")
        if not isinstance(uncheatable_loss, (int, float)):
            raise ValueError(f"W&B run {run.name} has no numeric uncheatable_macro_loss")
        results[name] = {
            "run_name": run.name,
            "url": run.url,
            "paloma_macro_loss": float(loss),
            "uncheatable_macro_loss": float(uncheatable_loss),
            "selection_method": run.config["selection_method"],
        }

    standalone_loss = float(results["standalone"]["paloma_macro_loss"])
    standalone_tps = float(results["standalone"]["terminal_mean_tokens_per_second"])
    control_loss = float(results["control_prefix"]["full_paloma_macro_loss"])
    control_tps = float(results["control_prefix"]["terminal_mean_tokens_per_second"])
    independent_wall_ratio = 1 + control_tps / standalone_tps
    comparisons: dict[str, dict[str, float]] = {}
    for name in ("nested_naive", "nested_layerwise"):
        nested_prefix_loss = float(results[name]["paloma_macro_loss"])
        nested_full_loss = float(results[name]["full_paloma_macro_loss"])
        nested_tps = float(results[name]["terminal_mean_tokens_per_second"])
        prefix_compute_multiplier = _compute_multiplier_to_loss(nested_prefix_loss, standalone_loss)
        full_compute_multiplier = _compute_multiplier_to_loss(nested_full_loss, control_loss)
        joint_compute_multiplier = max(prefix_compute_multiplier, full_compute_multiplier)
        joint_wall_ratio = joint_compute_multiplier * control_tps / nested_tps
        comparisons[name] = {
            "prefix_compute_multiplier_to_standalone_loss": prefix_compute_multiplier,
            "prefix_equivalent_updates": prefix_compute_multiplier * TRAIN_STEPS,
            "full_compute_multiplier_to_control_loss": full_compute_multiplier,
            "full_equivalent_updates": full_compute_multiplier * TRAIN_STEPS,
            "joint_compute_multiplier": joint_compute_multiplier,
            "joint_wall_vs_one_e256": joint_wall_ratio,
            "independent_wall_vs_one_e256": independent_wall_ratio,
            "joint_wall_savings_vs_independent": 1 - joint_wall_ratio / independent_wall_ratio,
        }

    result_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}-results.json"
    result_path.write_text(json.dumps({"comparisons": comparisons, "runs": results}, indent=2, sort_keys=True) + "\n")

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
    axes[0].axvline(
        PHASE_BOUNDARY_STEP * TOKENS_PER_STEP / 1e9,
        color="#57606a",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label="Datakit phase boundary",
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)

    names = tuple(TRAINED_RUNS) + tuple(POSTHOC_RUNS)
    values = [float(results[name]["paloma_macro_loss"]) for name in names]
    positions = np.arange(len(names))
    axes[1].scatter(values, positions, color=[COLORS[name] for name in names], s=80, zorder=3)
    for position, value in zip(positions, values, strict=True):
        axes[1].text(value + 0.008, position, f"{value:.3f}", va="center", fontsize=8)
    axes[1].set_title("E128 checkpoints at the 10B-token endpoint")
    axes[1].set_yticks(positions, [LABELS[name] for name in names])
    axes[1].set_xlabel("Paloma macro loss (lower is better)")
    axes[1].set_xlim(min(values) - 0.04, max(values) + 0.08)
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.2)

    figure.suptitle("Matched E128 training and post-hoc controls")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
