#!/usr/bin/env python3
# ruff: noqa: F841
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot Bloom GPU vs Marin TPU prompt-collapsed adherence for standard seed-0 runs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from rigging.filesystem import url_to_fs

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
class ComparisonSpec:
    label: str
    bloom_summary_path: Path
    marin_summary_path: str
    color: str


@dataclass(frozen=True)
class OverallStats:
    mean: float
    ci95: float
    std: float
    count: int


COMPARISONS = (
    ComparisonSpec(
        label="SFT Baseline",
        bloom_summary_path=Path(
            "/Users/ahmed/code/bloom/results/judging/dev-bloom-results-gpt-4-mini-prompts/"
            "_lfs_skampere3_0_ahmedah_models_marin_marin-8b-instruct/"
            "run_20260324_152421_7c0a6d282da6/summary.json"
        ),
        marin_summary_path=("gs://marin-us-east1/eval/marin_8b_instruct_bloom_speceval/" "judge-gpt41/summary.json"),
        color="#7f7f7f",
    ),
    ComparisonSpec(
        label=r"DPO $\beta$=0.01, lr=5e-7",
        bloom_summary_path=Path(
            "/Users/ahmed/code/bloom/results/judging/dev-bloom-results-gpt-4-mini-prompts/"
            "_lfs_skampere3_0_ahmedah_models_marin_bloom_v2_beta0-01_lr5e-7_seed0_step-849/"
            "run_20260325_183808_5ef636ab1ebc/summary.json"
        ),
        marin_summary_path=(
            "gs://marin-us-east5/eval/marin_dpo_beta001_lr5e7_seed0_bloom_speceval/" "judge-gpt41/summary.json"
        ),
        color="#1f77b4",
    ),
    ComparisonSpec(
        label=r"DPO $\beta$=0.01, lr=7.5e-7",
        bloom_summary_path=Path(
            "/Users/ahmed/code/bloom/results/judging/dev-bloom-results-gpt-4-mini-prompts/"
            "_lfs_skampere3_0_ahmedah_models_marin_bloom_v2_beta0-01_lr7-5e-7_seed0_step-849/"
            "run_20260325_192827_81d1d40ed0cb/summary.json"
        ),
        marin_summary_path=(
            "gs://marin-us-east5/eval/marin_dpo_beta001_lr75e7_seed0_bloom_speceval/" "judge-gpt41/summary.json"
        ),
        color="#aec7e8",
    ),
    ComparisonSpec(
        label=r"DPO $\beta$=0.1, lr=5e-7",
        bloom_summary_path=Path(
            "/Users/ahmed/code/bloom/results/judging/dev-bloom-results-gpt-4-mini-prompts/"
            "_lfs_skampere3_0_ahmedah_models_marin_bloom_v2_beta0-1_lr5e-7_seed0_step-849/"
            "run_20260325_195043_8929a88217b8/summary.json"
        ),
        marin_summary_path=(
            "gs://marin-us-east5/eval/marin_dpo_beta01_lr5e7_seed0_bloom_speceval/" "judge-gpt41/summary.json"
        ),
        color="#ff7f0e",
    ),
    ComparisonSpec(
        label=r"DPO $\beta$=0.1, lr=7.5e-7",
        bloom_summary_path=Path(
            "/Users/ahmed/code/bloom/results/judging/dev-bloom-results-gpt-4-mini-prompts/"
            "_lfs_skampere3_0_ahmedah_models_marin_bloom_v2_beta0-1_lr7-5e-7_seed0_step-849/"
            "run_20260325_195043_092641116cf6/summary.json"
        ),
        marin_summary_path=(
            "gs://marin-us-central1/eval/marin_dpo_beta01_lr75e7_seed0_bloom_speceval/" "judge-gpt41/summary.json"
        ),
        color="#ffbb78",
    ),
)


def _load_json(path: str) -> dict[str, Any]:
    fs, fs_path = url_to_fs(path)
    with fs.open(fs_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _compute_stats(values: list[float]) -> OverallStats:
    std = _sample_std(values)
    sem = std / math.sqrt(len(values))
    return OverallStats(mean=_mean(values), ci95=1.96 * sem, std=std, count=len(values))


def _row_prompt_key(row: dict[str, Any], statement_id: str, row_idx: int) -> str:
    question_id = row.get("question_id")
    if question_id:
        return str(question_id)
    prompt_id = row.get("source_prompt_id")
    if prompt_id:
        return f"{statement_id}:{prompt_id}"
    prompt = row.get("user_input") or row.get("prompt")
    if prompt:
        return f"{statement_id}:{prompt}"
    return f"{statement_id}:row_{row_idx}"


def _row_score(row: dict[str, Any], judges: list[str]) -> float | None:
    judgments = row.get("judgments")
    if not isinstance(judgments, dict):
        return None

    scores: list[float] = []
    for judge_id in judges:
        judgment = judgments.get(judge_id)
        if not isinstance(judgment, dict):
            continue
        score = judgment.get("score")
        if isinstance(score, (int, float)):
            scores.append(float(score))

    if not scores:
        return None
    return _mean(scores)


def load_bloom_prompt_collapsed_stats(summary_path: Path) -> OverallStats:
    summary = json.loads(summary_path.read_text())
    judges = list(summary.get("judges", []))
    per_statement_dir = summary_path.parent / "per_statement"
    overall_prompt_means: list[float] = []

    for statement_path in sorted(per_statement_dir.glob("*.json")):
        statement_data = json.loads(statement_path.read_text())
        statement_id = str(statement_data.get("statement_id") or statement_path.stem)
        prompt_samples: dict[str, list[float]] = {}
        for row_idx, row in enumerate(statement_data.get("results", [])):
            score = _row_score(row, judges)
            if score is None:
                continue
            prompt_key = _row_prompt_key(row, statement_id, row_idx)
            prompt_samples.setdefault(prompt_key, []).append(score)
        overall_prompt_means.extend(_mean(scores) for scores in prompt_samples.values())

    if not overall_prompt_means:
        raise ValueError(f"No prompt-collapsed Bloom stats found under {summary_path.parent}")
    return _compute_stats(overall_prompt_means)


def load_marin_prompt_collapsed_stats(summary_path: str) -> OverallStats:
    summary = _load_json(summary_path)
    if "overall_ci95" not in summary or "overall_mean_score" not in summary:
        fs, fs_path = url_to_fs(summary_path)
        judged_results_path = f"{Path(fs_path).parent}/judged_results.jsonl"
        prompt_scores: dict[str, list[float]] = {}
        with fs.open(judged_results_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                judgment = row.get("judgment", {})
                score = judgment.get("score")
                if score is None:
                    continue
                prompt_id = row.get("prompt_id") or row.get("user_message", "")[:80]
                behavior_id = row.get("behavior_id", "unknown")
                prompt_key = f"{behavior_id}:{prompt_id}"
                prompt_scores.setdefault(prompt_key, []).append(float(score))
        prompt_means = [_mean(scores) for scores in prompt_scores.values()]
        if not prompt_means:
            raise ValueError(f"No prompt-collapsed Marin stats found under {summary_path}")
        return _compute_stats(prompt_means)
    return OverallStats(
        mean=float(summary["overall_mean_score"]),
        ci95=float(summary["overall_ci95"]),
        std=float(summary["overall_std"]),
        count=int(summary["total_prompts"]),
    )


def plot_overall_comparison(
    comparisons: list[tuple[ComparisonSpec, OverallStats, OverallStats]],
    out_dir: Path,
) -> None:
    fig, (score_ax, delta_ax) = plt.subplots(
        1,
        2,
        figsize=(8.6, 4.8),
        gridspec_kw={"width_ratios": [4.8, 1.6], "wspace": 0.05},
        sharey=True,
        constrained_layout=True,
    )

    labels = [spec.label for spec, _, _ in comparisons]
    colors = [spec.color for spec, _, _ in comparisons]
    gpu_means = [gpu.mean for _, gpu, _ in comparisons]
    gpu_ci95s = [gpu.ci95 for _, gpu, _ in comparisons]
    tpu_means = [tpu.mean for _, _, tpu in comparisons]
    tpu_ci95s = [tpu.ci95 for _, _, tpu in comparisons]
    deltas = [tpu.mean - gpu.mean for _, gpu, tpu in comparisons]

    base_y = np.arange(len(labels), dtype=float)
    gpu_y = base_y - 0.14
    tpu_y = base_y + 0.14

    score_min = min(min(gpu.mean - gpu.ci95, tpu.mean - tpu.ci95) for _, gpu, tpu in comparisons)
    score_max = max(max(gpu.mean + gpu.ci95, tpu.mean + tpu.ci95) for _, gpu, tpu in comparisons)
    score_pad = 0.10
    score_ax.set_xlim(score_min - score_pad, score_max + 0.18)

    for idx, (spec, gpu, tpu) in enumerate(comparisons):
        score_ax.plot(
            [gpu.mean, tpu.mean],
            [gpu_y[idx], tpu_y[idx]],
            color="#b8b8b8",
            linewidth=1.0,
            zorder=1,
        )
        score_ax.errorbar(
            gpu.mean,
            gpu_y[idx],
            xerr=gpu.ci95,
            fmt="o",
            color=spec.color,
            ecolor=spec.color,
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            markersize=6.5,
            markerfacecolor="white",
            markeredgewidth=1.4,
            zorder=3,
        )
        score_ax.errorbar(
            tpu.mean,
            tpu_y[idx],
            xerr=tpu.ci95,
            fmt="s",
            color=spec.color,
            ecolor=spec.color,
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            markersize=6.2,
            markeredgewidth=1.0,
            zorder=3,
        )
        score_ax.text(
            gpu.mean + gpu.ci95 + 0.015,
            gpu_y[idx],
            f"{gpu.mean:.2f}",
            va="center",
            ha="left",
            fontsize=7.2,
            color="#444444",
        )
        score_ax.text(
            tpu.mean + tpu.ci95 + 0.015,
            tpu_y[idx],
            f"{tpu.mean:.2f}",
            va="center",
            ha="left",
            fontsize=7.2,
            color="#222222",
            fontweight="semibold",
        )

    score_ax.set_yticks(base_y)
    score_ax.set_yticklabels(labels)
    score_ax.invert_yaxis()
    score_ax.set_xlabel("Prompt-Collapsed Mean Adherence Score (95% CI, zoomed)")
    score_ax.set_title("Bloom GPU vs Marin TPU")
    score_ax.grid(axis="x", color="#d9d9d9", linewidth=0.7)
    score_ax.grid(axis="y", visible=False)
    score_ax.spines["top"].set_visible(False)
    score_ax.spines["right"].set_visible(False)

    delta_extent = max(abs(delta) for delta in deltas) + 0.03
    delta_ax.axvline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    delta_ax.barh(
        base_y,
        deltas,
        color=colors,
        alpha=0.9,
        edgecolor="none",
        height=0.34,
    )
    for idx, delta in enumerate(deltas):
        delta_ax.text(
            delta + (0.004 if delta >= 0 else -0.004),
            base_y[idx],
            f"{delta:+.02f}",
            va="center",
            ha="left" if delta >= 0 else "right",
            fontsize=7.4,
            color="#222222",
        )
    delta_ax.set_xlim(-delta_extent, delta_extent)
    delta_ax.set_xlabel("TPU - GPU")
    delta_ax.set_title("Delta")
    delta_ax.grid(axis="x", color="#e2e2e2", linewidth=0.7)
    delta_ax.grid(axis="y", visible=False)
    delta_ax.tick_params(axis="y", left=False, labelleft=False)
    delta_ax.spines["top"].set_visible(False)
    delta_ax.spines["right"].set_visible(False)
    delta_ax.spines["left"].set_visible(False)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="#444444",
            markeredgewidth=1.4,
            markersize=6.5,
            label="Bloom GPU",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            color="#444444",
            markerfacecolor="#444444",
            markeredgecolor="#444444",
            markersize=6.2,
            label="Marin TPU",
        ),
    ]
    score_ax.legend(handles=legend_handles, loc="upper left", frameon=False)

    fig.suptitle("Overall Spec Adherence Comparison", y=1.02, fontsize=12)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"gpu_vs_tpu_overall_adherence.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_comparison_json(
    comparisons: list[tuple[ComparisonSpec, OverallStats, OverallStats]],
    out_dir: Path,
) -> None:
    rows = []
    for spec, gpu, tpu in comparisons:
        rows.append(
            {
                "label": spec.label,
                "bloom_gpu": {
                    "mean": gpu.mean,
                    "ci95": gpu.ci95,
                    "std": gpu.std,
                    "count": gpu.count,
                    "summary_path": str(spec.bloom_summary_path),
                },
                "marin_tpu": {
                    "mean": tpu.mean,
                    "ci95": tpu.ci95,
                    "std": tpu.std,
                    "count": tpu.count,
                    "summary_path": spec.marin_summary_path,
                },
                "delta_tpu_minus_gpu": tpu.mean - gpu.mean,
            }
        )
    with (out_dir / "gpu_vs_tpu_overall_adherence.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Bloom GPU vs Marin TPU adherence.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "plot" / "output",
        help="Output directory for comparison plots",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    comparisons: list[tuple[ComparisonSpec, OverallStats, OverallStats]] = []
    for spec in COMPARISONS:
        bloom_stats = load_bloom_prompt_collapsed_stats(spec.bloom_summary_path)
        marin_stats = load_marin_prompt_collapsed_stats(spec.marin_summary_path)
        comparisons.append((spec, bloom_stats, marin_stats))
        print(
            f"{spec.label}: GPU={bloom_stats.mean:.4f} +/- {bloom_stats.ci95:.4f}, "
            f"TPU={marin_stats.mean:.4f} +/- {marin_stats.ci95:.4f}, "
            f"delta={marin_stats.mean - bloom_stats.mean:+.4f}"
        )

    plot_overall_comparison(comparisons, args.out_dir)
    write_comparison_json(comparisons, args.out_dir)
    print(f"Saved plot to {args.out_dir / 'gpu_vs_tpu_overall_adherence.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
