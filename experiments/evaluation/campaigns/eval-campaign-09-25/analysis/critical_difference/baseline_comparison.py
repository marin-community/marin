#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot the selected Snowball checkpoint against the complete baseline cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from plot import SURVIVAL, tracker_cells

SHORT_NAMES = {
    "Qwen/Qwen3.6-35B-A3B": "Qwen3.6",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": "Qwen3-Coder",
    "Qwen/Qwen3-Next-80B-A3B-Instruct": "Qwen3-Next",
    "inclusionAI/Ling-lite-1.5": "Ling Lite",
    "google/gemma-4-26B-A4B-it": "Gemma 4",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": "Nemotron 3.5",
    "openai/gpt-oss-20b": "GPT-OSS 20B",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "Nemotron Nano",
    "moonshotai/Kimi-Linear-48B-A3B-Instruct": "Kimi Linear",
    "arcee-ai/Trinity-Mini": "Trinity Mini",
    "IFM/K2-Horizon-MoVA-36B-A4B": "K2 Horizon",
    "zai-org/GLM-4.7-Flash": "GLM-4.7-Flash",
    "Aleph-Alpha/Kolibri-1": "Kolibri-1",
}
CI_MULTIPLIER = 1.96


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--statistics", required=True, type=Path)
    parser.add_argument("--flops", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    tracker_hash = hashlib.sha256(args.tracker.read_bytes()).hexdigest()
    if selection["tracker_sha256"] != tracker_hash:
        raise ValueError("selection and baseline plots must use the same tracker snapshot")
    winner = selection["winner"]
    benchmarks, cells = tracker_cells(args.tracker)
    flops_rows = csv_rows(args.flops)
    flops = {row["model"]: float(row["total_flops"]) for row in flops_rows}
    models = [winner, *SHORT_NAMES]
    if set(flops) != set(models):
        raise ValueError("FLOP cohort differs from the selected winner and fixed baselines")
    statistics = {(row["model"], row["benchmark"]): row for row in csv_rows(args.statistics)}
    if len(statistics) != len(models) * len(benchmarks):
        raise ValueError("statistics omit at least one model-benchmark cell")
    rows = []
    for benchmark in benchmarks:
        for model in models:
            cell = cells[model][benchmark]
            stat = statistics[(model, benchmark)]
            score = float(stat["score"])
            if abs(score - cell.score) > 1e-9 or stat["source"] != cell.results_path:
                raise ValueError(f"statistics differ from tracker: {model} / {benchmark}")
            radius = CI_MULTIPLIER * float(stat["adjusted_sem"])
            rows.append(
                {
                    "benchmark": benchmark,
                    "model": model,
                    "score_pct": 100 * score,
                    "ci_low_pct": 100 * max(0, score - radius),
                    "ci_high_pct": 100 * min(1, score + radius),
                    "scored_trials": stat["trial_count"],
                    "total_training_flops": flops[model],
                    "results_path": cell.results_path,
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "scores_with_95ci.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for benchmark in benchmarks:
        panel = [row for row in rows if row["benchmark"] == benchmark]
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.get_cmap("tab20").colors
        for index, row in enumerate(panel):
            model = str(row["model"])
            selected = model == winner
            score = float(row["score_pct"])
            lower = float(row["ci_low_pct"])
            upper = float(row["ci_high_pct"])
            ax.errorbar(
                float(row["total_training_flops"]),
                score,
                yerr=[[score - lower], [upper - score]],
                fmt="*" if selected else "o",
                markersize=14 if selected else 6,
                color="#e65c27" if selected else colors[index % len(colors)],
                ecolor="#e65c27" if selected else colors[index % len(colors)],
                markeredgecolor="#333333",
                markeredgewidth=0.5,
                capsize=2,
                zorder=4 if selected else 2,
                label="Snowball" if selected else SHORT_NAMES[model],
            )
        x_values = [float(row["total_training_flops"]) for row in panel]
        ax.set_xscale("log")
        ax.set_xlim(min(x_values) * 0.85, max(x_values) * 1.15)
        ax.set_ylim(0, min(100, max(12, max(float(row["ci_high_pct"]) for row in panel) + 8)))
        ax.set_xlabel("Estimated total training FLOPs (log scale)")
        ax.set_ylabel("Score (%)")
        ax.set_title(benchmark)
        ax.grid(alpha=0.18)
        fig.tight_layout()
        fig.subplots_adjust(right=0.72)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(args.output_dir / f"scatter_{benchmark.lower()}.{suffix}", dpi=220, bbox_inches="tight")
        plt.close(fig)
    metadata = {
        "winner": winner,
        "tracker_sha256": tracker_hash,
        "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        "statistics_sha256": hashlib.sha256(args.statistics.read_bytes()).hexdigest(),
        "flops_sha256": hashlib.sha256(args.flops.read_bytes()).hexdigest(),
        "confidence_interval": f"score +/- 1.96 x task-level SEM / sqrt({SURVIVAL}), clipped to [0, 1]",
        "benchmark_count": len(benchmarks),
        "baseline_count": len(models) - 1,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"plotted {len(benchmarks)} benchmarks with {len(models) - 1} baselines")


if __name__ == "__main__":
    main()
