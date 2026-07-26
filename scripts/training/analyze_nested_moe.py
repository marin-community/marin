#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the result artifact and figures for the nested-MoE experiment."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import wandb

ENTITY = "marin-community"
PRETRAIN_PROJECT = "marin_moe"
SFT_PROJECT = "marin_moe_sft"
TOKENS_PER_STEP = 256 * 2048
OUTPUT_DIR = Path("docs/reports/assets")

PRETRAIN_RUNS = {
    "large_control": "nest-moe-001-full-d768-s2048-e256-cf125-r15",
    "small_control": "nest-moe-002-full-d768-s2048-e128-cf125-r18",
    "nested25": "nest-moe-003-full-d768-s2048-e256-cf125-r17",
    "nested50": "nest-moe-004-full-d768-s2048-e256-cf125-r15",
    "untreated_subset": "nest-moe-001-full-d768-s2048-e256-subset-eval-cf125-r19",
    "cooldown": "nest-moe-005-cooldown-d768-s2048-e128-cf125-r20",
}

SFT_RUNS = {
    "large_control": "nest-moe-sft-large-d768-s2048-r23",
    "small_control": "nest-moe-sft-small-d768-s2048-r24",
    "nested25_full": "nest-moe-sft-nested_full-d768-s2048-r24",
    "nested25_breakout": "nest-moe-sft-breakout-d768-s2048-r24",
}

GPU_HOURS = {
    "large_control": 7.0112,
    "small_control": 6.653988888888889,
    "nested25": 6.67,
    "nested50": 7.2816,
    "cooldown": 5.103644444444445,
}


def _history(run: Any, metric: str) -> list[dict[str, float]]:
    rows = []
    for row in run.scan_history(keys=["global_step", metric]):
        step = row.get("global_step")
        value = row.get(metric)
        if step is None or value is None:
            continue
        rows.append({"step": int(step), "value": float(value)})
    return rows


def _value_at_or_before(rows: list[dict[str, float]], target_step: int) -> float:
    eligible = [row for row in rows if row["step"] <= target_step]
    if not eligible:
        raise ValueError(f"No metric at or before step {target_step}")
    return max(eligible, key=lambda row: row["step"])["value"]


def _summary_number(run: Any, key: str) -> float:
    value = run.summary.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric W&B summary key {key!r} in {run.name}")
    return float(value)


def _paloma_domain_losses(run: Any, prefix: str) -> dict[str, float]:
    suffix = "/loss"
    result = {}
    for key, value in dict(run.summary).items():
        if not key.startswith(prefix) or not key.endswith(suffix):
            continue
        domain = key.removeprefix(prefix).removesuffix(suffix)
        if domain in {"macro", "micro"} or "/" in domain:
            continue
        if isinstance(value, (int, float)):
            result[domain] = float(value)
    return result


def _paired_domain_summary(
    treatment: dict[str, float],
    control: dict[str, float],
) -> dict[str, float | int]:
    domains = sorted(treatment.keys() & control.keys())
    deltas = [treatment[domain] - control[domain] for domain in domains]
    return {
        "domains": len(domains),
        "treatment_better": sum(delta < 0 for delta in deltas),
        "mean_delta": statistics.fmean(deltas),
        "median_delta": statistics.median(deltas),
        "min_delta": min(deltas),
        "max_delta": max(deltas),
    }


def _load_runs(
    api: wandb.Api,
    project: str,
    names: dict[str, str],
) -> dict[str, Any]:
    return {label: api.run(f"{ENTITY}/{project}/{run_name}") for label, run_name in names.items()}


def _pretrain_artifact(runs: dict[str, Any]) -> dict[str, Any]:
    histories = {}
    for label, run in runs.items():
        histories[label] = {
            "train_loss": _history(run, "train/loss"),
            "paloma_macro_loss": _history(run, "eval/paloma/macro_loss"),
            "nested_paloma_macro_loss": _history(run, "eval/nested/paloma/macro_loss"),
            "tokens_per_second": _history(run, "throughput/tokens_per_second"),
            "overflow": _history(run, "train/router/capacity_overflow_rate_mean"),
        }

    summaries = {}
    for label, run in runs.items():
        throughput = [row["value"] for row in histories[label]["tokens_per_second"] if row["step"] >= 5]
        overflow = [row["value"] for row in histories[label]["overflow"]]
        summaries[label] = {
            "run_name": run.name,
            "url": run.url,
            "state": run.state,
            "runtime": _summary_number(run, "_runtime"),
            "final_step": int(_summary_number(run, "global_step")),
            "final_train_loss": _summary_number(run, "train/loss"),
            "final_paloma_macro_loss": run.summary.get("eval/paloma/macro_loss"),
            "final_nested_paloma_macro_loss": run.summary.get("eval/nested/paloma/macro_loss"),
            "median_tokens_per_second_step_5_plus": statistics.median(throughput),
            "mean_tokens_per_second_step_5_plus": statistics.fmean(throughput),
            "mean_overflow": statistics.fmean(overflow),
            "max_overflow": max(overflow),
            "terminal_overflow": overflow[-1],
        }

    large_domains = _paloma_domain_losses(runs["large_control"], "eval/paloma/")
    small_domains = _paloma_domain_losses(runs["small_control"], "eval/paloma/")
    nested25_full_domains = _paloma_domain_losses(runs["nested25"], "eval/paloma/")
    nested25_small_domains = _paloma_domain_losses(runs["nested25"], "eval/nested/paloma/")
    untreated_small_domains = _paloma_domain_losses(runs["untreated_subset"], "eval/nested/paloma/")
    cooldown_domains = _paloma_domain_losses(runs["cooldown"], "eval/paloma/")

    return {
        "summaries": summaries,
        "histories": histories,
        "domain_comparisons": {
            "nested25_full_vs_large_control": _paired_domain_summary(nested25_full_domains, large_domains),
            "nested25_small_vs_small_control": _paired_domain_summary(nested25_small_domains, small_domains),
            "nested25_small_vs_untreated_subset": _paired_domain_summary(
                nested25_small_domains, untreated_small_domains
            ),
            "cooldown_vs_small_control": _paired_domain_summary(cooldown_domains, small_domains),
        },
    }


def _sft_artifact(runs: dict[str, Any]) -> dict[str, Any]:
    result = {"summaries": {}, "histories": {}}
    for label, run in runs.items():
        loss = _history(run, "train/loss")
        overflow = _history(run, "train/router/capacity_overflow_rate_mean")
        throughput = _history(run, "throughput/tokens_per_second")
        result["histories"][label] = {
            "train_loss": loss,
            "overflow": overflow,
            "tokens_per_second": throughput,
        }
        result["summaries"][label] = {
            "run_name": run.name,
            "url": run.url,
            "state": run.state,
            "runtime": _summary_number(run, "_runtime"),
            "final_train_loss": loss[-1]["value"],
            "mean_train_loss_steps_2_7": statistics.fmean(row["value"] for row in loss),
            "mean_overflow_steps_2_7": statistics.fmean(row["value"] for row in overflow),
            "median_tokens_per_second_steps_3_7": statistics.median(
                row["value"] for row in throughput if row["step"] >= 3
            ),
        }
    return result


def _write_summary_csv(artifact: dict[str, Any]) -> None:
    output_path = OUTPUT_DIR / "nested-model-training-summary.csv"
    fields = [
        "arm",
        "final_train_loss",
        "final_paloma_macro_loss",
        "final_nested_paloma_macro_loss",
        "median_tokens_per_second_step_5_plus",
        "mean_tokens_per_second_step_5_plus",
        "mean_overflow",
        "max_overflow",
        "terminal_overflow",
        "runtime",
        "gpu_hours",
        "url",
    ]
    with output_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for arm, summary in artifact["pretraining"]["summaries"].items():
            row = {field: summary.get(field) for field in fields}
            row["arm"] = arm
            row["gpu_hours"] = GPU_HOURS.get(arm)
            writer.writerow(row)


def _plot_pretraining_loss(artifact: dict[str, Any]) -> None:
    histories = artifact["pretraining"]["histories"]
    labels = {
        "large_control": "E256 control",
        "small_control": "E128 control",
        "nested25": "E256 nested 25%",
        "nested50": "E256 nested 50%",
    }
    colors = {
        "large_control": "#24292f",
        "small_control": "#6e7781",
        "nested25": "#0969da",
        "nested50": "#cf222e",
    }

    fig, axis = plt.subplots(figsize=(9, 5.2))
    for arm, label in labels.items():
        rows = histories[arm]["train_loss"]
        x = [(row["step"] + 1) * TOKENS_PER_STEP / 1e6 for row in rows]
        y = [row["value"] for row in rows]
        axis.plot(x, y, color=colors[arm], alpha=0.18, linewidth=0.8)
        window = 20
        smooth_y = [statistics.fmean(y[max(0, index - window + 1) : index + 1]) for index in range(len(y))]
        axis.plot(x, smooth_y, color=colors[arm], linewidth=2.0, label=label)
    axis.set_xlabel("Training tokens (millions)")
    axis.set_ylabel("Training cross-entropy loss")
    axis.set_title("Nested-MoE pretraining loss")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "nested-model-training-pretraining-loss.png",
        dpi=180,
    )
    plt.close(fig)


def _plot_paloma(artifact: dict[str, Any]) -> None:
    histories = artifact["pretraining"]["histories"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    full_lines = {
        "large_control": ("E256 control", "#24292f"),
        "nested25": ("Nested 25%: full", "#0969da"),
        "nested50": ("Nested 50%: full", "#cf222e"),
    }
    for arm, (label, color) in full_lines.items():
        rows = histories[arm]["paloma_macro_loss"]
        axes[0].plot(
            [(row["step"] + 1) * TOKENS_PER_STEP / 1e6 for row in rows],
            [row["value"] for row in rows],
            marker="o",
            color=color,
            label=label,
        )
    axes[0].set_title("Full-model Paloma")
    axes[0].set_xlabel("Pretraining tokens (millions)")
    axes[0].set_ylabel("Paloma macro loss")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)

    small_lines = {
        "small_control": ("Standalone E128", "paloma_macro_loss", "#6e7781"),
        "nested25": ("Extracted E128", "nested_paloma_macro_loss", "#0969da"),
    }
    for arm, (label, metric, color) in small_lines.items():
        rows = histories[arm][metric]
        axes[1].plot(
            [(row["step"] + 1) * TOKENS_PER_STEP / 1e6 for row in rows],
            [row["value"] for row in rows],
            marker="o",
            color=color,
            label=label,
        )
    cooldown = histories["cooldown"]["paloma_macro_loss"]
    axes[1].plot(
        [500 * TOKENS_PER_STEP / 1e6 + (row["step"] + 1) * TOKENS_PER_STEP / 1e6 for row in cooldown],
        [row["value"] for row in cooldown],
        marker="o",
        color="#1a7f37",
        label="Extracted E128 + cooldown",
    )
    axes[1].set_title("Extracted-model Paloma and cooldown")
    axes[1].set_xlabel("Large-pretraining tokens + E128 cooldown tokens (millions)")
    axes[1].set_ylabel("Paloma macro loss")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "nested-model-training-paloma.png", dpi=180)
    plt.close(fig)


def _plot_sft(artifact: dict[str, Any]) -> None:
    histories = artifact["sft"]["histories"]
    labels = {
        "large_control": ("E256 control", "#24292f"),
        "small_control": ("E128 control", "#6e7781"),
        "nested25_full": ("Nested 25%: full", "#0969da"),
        "nested25_breakout": ("Nested 25%: breakout", "#1a7f37"),
    }
    fig, axis = plt.subplots(figsize=(9, 5.2))
    for arm, (label, color) in labels.items():
        rows = histories[arm]["train_loss"]
        axis.plot(
            [row["step"] for row in rows],
            [row["value"] for row in rows],
            marker="o",
            color=color,
            label=label,
        )
    axis.set_xlabel("SFT update")
    axis.set_ylabel("Assistant-token cross-entropy loss")
    axis.set_title("Matched WildChat SFT transfer check")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "nested-model-training-sft-loss.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=60)
    pretrain_runs = _load_runs(api, PRETRAIN_PROJECT, PRETRAIN_RUNS)
    sft_runs = _load_runs(api, SFT_PROJECT, SFT_RUNS)
    artifact = {
        "schema_version": 1,
        "tokens_per_step": TOKENS_PER_STEP,
        "analytic_flops_per_token": {
            "e256": 357_728_256,
            "e128": 356_155_392,
        },
        "gpu_hours": GPU_HOURS,
        "pretraining": _pretrain_artifact(pretrain_runs),
        "sft": _sft_artifact(sft_runs),
    }

    with (OUTPUT_DIR / "nested-model-training-results.json").open("w") as output:
        json.dump(artifact, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
    _write_summary_csv(artifact)
    _plot_pretraining_loss(artifact)
    _plot_paloma(artifact)
    _plot_sft(artifact)


if __name__ == "__main__":
    main()
