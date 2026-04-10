#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot TPU-only full-DPO adherence, highlighting the new batch-64 run."""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from plot_bloom_gpu_vs_marin_tpu_adherence import OverallStats, load_marin_prompt_collapsed_stats

matplotlib.rcParams.update(
    {
        "font.size": 9,
        "font.family": "serif",
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
    }
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TpuRunSpec:
    label: str
    summary_path: str
    color: str
    marker: str
    is_new_batch64: bool = False
    history_path: Path | None = None


RUN_SPECS = (
    TpuRunSpec(
        label=r"DPO $\beta$=0.01, lr=5e-7, batch=128",
        summary_path=(
            "gs://marin-us-east5/eval/marin_dpo_beta001_lr5e7_seed0_bloom_speceval/" "judge-gpt41/summary.json"
        ),
        color="#1f77b4",
        marker="o",
    ),
    TpuRunSpec(
        label=r"DPO $\beta$=0.01, lr=7.5e-7, batch=128",
        summary_path=(
            "gs://marin-us-east5/eval/marin_dpo_beta001_lr75e7_seed0_bloom_speceval/" "judge-gpt41/summary.json"
        ),
        color="#8fb6e0",
        marker="o",
    ),
    TpuRunSpec(
        label=r"DPO $\beta$=0.1, lr=5e-7, batch=128",
        summary_path=(
            "gs://marin-us-east5/eval/marin_dpo_beta01_lr5e7_seed0_bloom_speceval/" "judge-gpt41/summary.json"
        ),
        color="#f28e2b",
        marker="o",
    ),
    TpuRunSpec(
        label=r"DPO $\beta$=0.1, lr=5e-7, batch=64 (new)",
        summary_path=(
            "gs://marin-eu-west4/eval/"
            "marin_dpo_compare_lora_beta01_seed0_b64_step1699_bloom_speceval_seed0fullb64euw4r1/"
            "judge-gpt41/summary.json"
        ),
        color="#c03d3e",
        marker="D",
        is_new_batch64=True,
    ),
    TpuRunSpec(
        label=r"DPO $\beta$=0.1, lr=7.5e-7, batch=128",
        summary_path=(
            "gs://marin-us-central1/eval/marin_dpo_beta01_lr75e7_seed0_bloom_speceval/" "judge-gpt41/summary.json"
        ),
        color="#f6be85",
        marker="o",
    ),
)


BATCH64_MATCHUP_SPECS = (
    TpuRunSpec(
        label=r"Full DPO, $\beta$=0.1, lr=5e-7, batch=64",
        summary_path=(
            "gs://marin-eu-west4/eval/"
            "marin_dpo_compare_lora_beta01_seed0_b64_step1699_bloom_speceval_seed0fullb64euw4r1/"
            "judge-gpt41/summary.json"
        ),
        color="#c03d3e",
        marker="D",
        is_new_batch64=True,
        history_path=(
            REPO_ROOT.parent
            / "spicy-hugging-cat"
            / "scratch"
            / "wandb_dpo_data"
            / "new_dpo"
            / "bloom_speceval_v2_beta0.1_seed0_b64_v5p16-68f963"
            / "history.jsonl.gz"
        ),
    ),
    TpuRunSpec(
        label=r"LoRA, lr=5e-6, batch=64",
        summary_path=(
            "gs://marin-eu-west4/eval/"
            "marin_dpo_tune_lora_lr5e6_seed0_step1699_bloom_speceval_seed0paireuw4r2/"
            "judge-gpt41/summary.json"
        ),
        color="#4e79a7",
        marker="o",
        history_path=(
            REPO_ROOT.parent
            / "spicy-hugging-cat"
            / "scratch"
            / "wandb_dpo_data"
            / "tune_lora"
            / "bloom_speceval_v2_marin_lr5e6_seed0_b64_v5p8-274540"
            / "history.jsonl.gz"
        ),
    ),
    TpuRunSpec(
        label=r"LoRA, lr=1e-5, batch=64",
        summary_path=(
            "gs://marin-us-central1/eval/"
            "marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_cleanreexportr2/"
            "judge-gpt41/summary.json"
        ),
        color="#59a14f",
        marker="o",
        history_path=(
            REPO_ROOT.parent
            / "spicy-hugging-cat"
            / "scratch"
            / "wandb_dpo_data"
            / "tune_lora"
            / "bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d"
            / "history.jsonl.gz"
        ),
    ),
)

EVAL_ACCURACY_KEY = "eval/bloom_speceval_v2_val/dpo_accuracy"
EVAL_LOSS_KEY = "eval/bloom_speceval_v2_val/loss"
PALOMA_MACRO_LOSS_KEY = "lm_eval/paloma/macro_loss"
TRAIN_LOSS_KEY = "train/loss"


def load_metric_history(history_path: Path, metric_key: str) -> list[tuple[int, float]]:
    """Load non-null metric points from an archived W&B history file."""
    points: list[tuple[int, float]] = []
    with gzip.open(history_path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            metric_value = row.get(metric_key)
            global_step = row.get("global_step")
            if metric_value is None or global_step is None:
                continue
            points.append((int(global_step), float(metric_value)))
    return points


def final_metric_value(history_path: Path, metric_key: str) -> float:
    """Return the last non-null point for a metric from W&B history."""
    points = load_metric_history(history_path, metric_key)
    if not points:
        raise ValueError(f"No points found for {metric_key} in {history_path}")
    return points[-1][1]


def plot_tpu_adherence(comparisons: list[tuple[TpuRunSpec, OverallStats]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.6), constrained_layout=True)

    labels = [spec.label for spec, _ in comparisons]
    means = [stats.mean for _, stats in comparisons]
    ci95s = [stats.ci95 for _, stats in comparisons]
    base_y = np.arange(len(labels), dtype=float)

    score_min = min(mean - ci95 for mean, ci95 in zip(means, ci95s, strict=True))
    score_max = max(mean + ci95 for mean, ci95 in zip(means, ci95s, strict=True))
    ax.set_xlim(score_min - 0.12, score_max + 0.22)

    for idx, (spec, stats) in enumerate(comparisons):
        if spec.is_new_batch64:
            ax.axhspan(idx - 0.36, idx + 0.36, color="#f8e5e5", zorder=0)
        ax.errorbar(
            stats.mean,
            base_y[idx],
            xerr=stats.ci95,
            fmt=spec.marker,
            color=spec.color,
            ecolor=spec.color,
            elinewidth=1.2,
            capsize=3,
            capthick=1.0,
            markersize=7.0 if spec.is_new_batch64 else 6.0,
            markerfacecolor=spec.color if spec.is_new_batch64 else "white",
            markeredgewidth=1.4,
            zorder=3,
        )
        ax.text(
            stats.mean + stats.ci95 + 0.015,
            base_y[idx],
            f"{stats.mean:.2f} +/- {stats.ci95:.02f}",
            va="center",
            ha="left",
            fontsize=7.6,
            color="#222222",
            fontweight="semibold" if spec.is_new_batch64 else "normal",
        )

    ax.set_yticks(base_y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Prompt-Collapsed Mean Adherence Score (95% CI)")
    ax.set_title("TPU Full-DPO Adherence")
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="#555555",
            markeredgewidth=1.2,
            markersize=6.0,
            label="Older full fine-tuning DPO runs (batch=128)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            color="#c03d3e",
            markerfacecolor="#c03d3e",
            markeredgecolor="#c03d3e",
            markersize=7.0,
            label="New full-DPO run (batch=64)",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False)

    fig.suptitle(
        "Bloom-Format GPT-4.1 Judge on TPU Inference: New Batch-64 Run vs Older Full-DPO Runs",
        y=1.02,
        fontsize=12,
    )
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"tpu_full_dpo_adherence.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_json(comparisons: list[tuple[TpuRunSpec, OverallStats]], out_dir: Path) -> None:
    rows = []
    for spec, stats in comparisons:
        rows.append(
            {
                "label": spec.label,
                "summary_path": spec.summary_path,
                "mean": stats.mean,
                "ci95": stats.ci95,
                "std": stats.std,
                "count": stats.count,
                "is_new_batch64": spec.is_new_batch64,
            }
        )
    with (out_dir / "tpu_full_dpo_adherence.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")


def plot_batch64_matchup(comparisons: list[tuple[TpuRunSpec, OverallStats]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 3.6), constrained_layout=True)

    labels = [spec.label for spec, _ in comparisons]
    means = [stats.mean for _, stats in comparisons]
    ci95s = [stats.ci95 for _, stats in comparisons]
    base_y = np.arange(len(labels), dtype=float)

    score_min = min(mean - ci95 for mean, ci95 in zip(means, ci95s, strict=True))
    score_max = max(mean + ci95 for mean, ci95 in zip(means, ci95s, strict=True))
    ax.set_xlim(score_min - 0.12, score_max + 0.20)

    for idx, (spec, stats) in enumerate(comparisons):
        ax.errorbar(
            stats.mean,
            base_y[idx],
            xerr=stats.ci95,
            fmt=spec.marker,
            color=spec.color,
            ecolor=spec.color,
            elinewidth=1.2,
            capsize=3,
            capthick=1.0,
            markersize=7.0 if spec.is_new_batch64 else 6.2,
            markerfacecolor=spec.color if spec.is_new_batch64 else "white",
            markeredgewidth=1.4,
            zorder=3,
        )
        ax.text(
            stats.mean + stats.ci95 + 0.014,
            base_y[idx],
            f"{stats.mean:.2f} +/- {stats.ci95:.02f}",
            va="center",
            ha="left",
            fontsize=7.6,
            color="#222222",
            fontweight="semibold" if spec.is_new_batch64 else "normal",
        )

    ax.set_yticks(base_y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Prompt-Collapsed Mean Adherence Score (95% CI)")
    ax.set_title("Batch-64 Matchup on TPU")
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Bloom-Format GPT-4.1 Judge: Full DPO vs LoRA Runs, Matched Batch Size 64",
        y=1.02,
        fontsize=12,
    )
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"tpu_batch64_matchup.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_batch64_eval_alignment(comparisons: list[tuple[TpuRunSpec, OverallStats]], out_dir: Path) -> None:
    fig, (ax_eval, ax_judge) = plt.subplots(
        ncols=2,
        figsize=(8.8, 3.8),
        constrained_layout=True,
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.15]},
    )

    labels = [spec.label for spec, _ in comparisons]
    base_y = np.arange(len(labels), dtype=float)
    eval_accuracies = [final_metric_value(spec.history_path, EVAL_ACCURACY_KEY) for spec, _ in comparisons]

    ax_eval.scatter(
        eval_accuracies,
        base_y,
        s=55,
        c=[spec.color for spec, _ in comparisons],
        marker="o",
        zorder=3,
    )
    for idx, eval_accuracy in enumerate(eval_accuracies):
        ax_eval.text(
            eval_accuracy + 0.0003,
            base_y[idx],
            f"{eval_accuracy:.4f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#222222",
        )
    ax_eval.set_xlim(min(eval_accuracies) - 0.004, max(eval_accuracies) + 0.004)
    ax_eval.set_xlabel("Final eval DPO accuracy")
    ax_eval.set_title("Training-time eval signal")
    ax_eval.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax_eval.grid(axis="y", visible=False)
    ax_eval.spines["top"].set_visible(False)
    ax_eval.spines["right"].set_visible(False)

    for idx, (spec, stats) in enumerate(comparisons):
        ax_judge.errorbar(
            stats.mean,
            base_y[idx],
            xerr=stats.ci95,
            fmt=spec.marker,
            color=spec.color,
            ecolor=spec.color,
            elinewidth=1.2,
            capsize=3,
            capthick=1.0,
            markersize=7.0 if spec.is_new_batch64 else 6.2,
            markerfacecolor=spec.color if spec.is_new_batch64 else "white",
            markeredgewidth=1.4,
            zorder=3,
        )
        ax_judge.text(
            stats.mean + stats.ci95 + 0.014,
            base_y[idx],
            f"{stats.mean:.2f} +/- {stats.ci95:.02f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#222222",
        )

    ax_judge.set_xlim(
        min(stats.mean - stats.ci95 for _, stats in comparisons) - 0.10,
        max(stats.mean + stats.ci95 for _, stats in comparisons) + 0.20,
    )
    ax_judge.set_xlabel("Prompt-collapsed GPT-4.1 judge mean")
    ax_judge.set_title("Bloom-style LM-as-judge")
    ax_judge.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax_judge.grid(axis="y", visible=False)
    ax_judge.spines["top"].set_visible(False)
    ax_judge.spines["right"].set_visible(False)

    ax_eval.set_yticks(base_y)
    ax_eval.set_yticklabels(labels)
    ax_eval.invert_yaxis()

    fig.suptitle("Batch-64 seed-0: final eval accuracy tracks LM-as-judge ordering", y=1.02, fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"tpu_batch64_eval_alignment.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_batch64_loss_curves(comparisons: list[tuple[TpuRunSpec, OverallStats]], out_dir: Path) -> None:
    metrics = (
        (TRAIN_LOSS_KEY, "Train loss"),
        (EVAL_LOSS_KEY, "Eval DPO loss"),
        (PALOMA_MACRO_LOSS_KEY, "Paloma macro loss"),
    )
    fig, axes = plt.subplots(ncols=3, figsize=(11.8, 3.6), constrained_layout=True)

    for ax, (metric_key, title) in zip(axes, metrics, strict=True):
        for spec, _ in comparisons:
            points = load_metric_history(spec.history_path, metric_key)
            x = [step for step, _ in points]
            y = [value for _, value in points]
            ax.plot(x, y, color=spec.color, linewidth=1.8, label=spec.label)
            ax.scatter([x[-1]], [y[-1]], color=spec.color, s=18, zorder=3)
        ax.set_title(title)
        ax.set_xlabel("Global step")
        ax.grid(color="#e3e3e3", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Metric value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.10))
    fig.suptitle("Batch-64 seed-0 training curves: full DPO vs LoRA", y=1.18, fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"tpu_batch64_loss_curves.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_batch64_json(comparisons: list[tuple[TpuRunSpec, OverallStats]], out_dir: Path) -> None:
    rows = []
    for spec, stats in comparisons:
        rows.append(
            {
                "label": spec.label,
                "summary_path": spec.summary_path,
                "mean": stats.mean,
                "ci95": stats.ci95,
                "std": stats.std,
                "count": stats.count,
                "is_new_batch64": spec.is_new_batch64,
                "final_eval_accuracy": final_metric_value(spec.history_path, EVAL_ACCURACY_KEY),
            }
        )
    with (out_dir / "tpu_batch64_matchup.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot TPU-only full-DPO adherence.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "plot" / "output",
        help="Output directory for plots",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    comparisons = [(spec, load_marin_prompt_collapsed_stats(spec.summary_path)) for spec in RUN_SPECS]
    plot_tpu_adherence(comparisons, args.out_dir)
    write_json(comparisons, args.out_dir)
    batch64_matchup = [(spec, load_marin_prompt_collapsed_stats(spec.summary_path)) for spec in BATCH64_MATCHUP_SPECS]
    plot_batch64_matchup(batch64_matchup, args.out_dir)
    plot_batch64_eval_alignment(batch64_matchup, args.out_dir)
    plot_batch64_loss_curves(batch64_matchup, args.out_dir)
    write_batch64_json(batch64_matchup, args.out_dir)
    print(f"Saved plot to {args.out_dir / 'tpu_full_dpo_adherence.png'}")
    print(f"Saved plot to {args.out_dir / 'tpu_batch64_matchup.png'}")
    print(f"Saved plot to {args.out_dir / 'tpu_batch64_eval_alignment.png'}")
    print(f"Saved plot to {args.out_dir / 'tpu_batch64_loss_curves.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
