# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot completed-correct quality against work, elapsed time, and reserved GPU-hours.

Use the CSV outputs of analyze_score_centering.py and analyze_score_centering_cost.py.
Pass --metrics to add consumed-loss-token curves alongside update, time, and cost curves.
Repeat --run to select the arms in one figure. Example::

    python -m experiments.post_training.plot_score_centering \
        --evaluations /tmp/evals.csv --cost /tmp/cost.csv \
        --run r19='TIS, cap 2' --run r20='TIS + SC32, cap 2' \
        --output /tmp/quality.svg
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


def plot_curves(
    evaluations: Path,
    cost: Path,
    arms: dict[str, str],
    output: Path,
    title: str,
    metrics: Path | None = None,
) -> None:
    with evaluations.open(newline="") as stream:
        quality_rows = [row for row in csv.DictReader(stream) if row["dataset"] == "all"]
    with cost.open(newline="") as stream:
        cost_rows = {(row["run"], int(row["step"])): row for row in csv.DictReader(stream)}
    token_rows = {}
    if metrics is not None:
        with metrics.open(newline="") as stream:
            token_rows = {
                (row["run"], int(row["step"])): float(row["cumulative_consumed_tokens"]) / 1e6
                for row in csv.DictReader(stream)
            }

    by_run = defaultdict(list)
    for row in quality_rows:
        run, step = row["run"], int(row["step"])
        if run not in arms:
            continue
        cost_row = cost_rows.get((run, step))
        if cost_row is None:
            raise ValueError(f"{run} step {step}: missing task cost")
        tokens_millions = 0.0 if step == 0 else token_rows.get((run, step))
        if metrics is not None and tokens_millions is None:
            raise ValueError(f"{run} step {step}: missing consumed-token metrics")
        by_run[run].append(
            (
                step,
                tokens_millions,
                float(cost_row["elapsed_from_first_gpu_task_hours"]),
                float(cost_row["reserved_gpu_hours_to_eval"]),
                100 * float(row["completed_correct_rate"]),
            )
        )
    if set(by_run) != set(arms):
        raise ValueError(f"missing evaluated arms: {sorted(set(arms) - set(by_run))}")

    x_columns = [(0, "Optimizer updates")]
    if metrics is not None:
        x_columns.append((1, "Consumed loss tokens (millions)"))
    x_columns.extend(((2, "Elapsed GPU-task hours"), (3, "Reserved H100-hours")))
    fig, axes = plt.subplots(1, len(x_columns), figsize=(5 * len(x_columns), 4.5), sharey=True, constrained_layout=True)
    markers = ("o", "s", "^", "D", "v", "P", "X", "*")
    for arm_index, (run, label) in enumerate(arms.items()):
        points = sorted(by_run[run])
        for ax, (column, _) in zip(axes, x_columns, strict=True):
            ax.plot(
                [point[column] for point in points],
                [point[4] for point in points],
                marker=markers[arm_index % len(markers)],
                markersize=5,
                linewidth=1.8,
                label=label,
            )
    for ax, (_, label) in zip(axes, x_columns, strict=True):
        ax.set_xlabel(label)
        ax.grid(alpha=0.25)
        ax.set_xlim(left=0)
    axes[0].set_ylabel("Completed correct (%)")
    axes[0].set_ylim(bottom=0)
    fig.suptitle(title)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=min(4, len(arms)))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format=output.suffix.removeprefix(".") or "svg")
    plt.close(fig)
    if output.suffix.lower() == ".svg":
        svg = output.read_text()
        output.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", required=True, type=Path)
    parser.add_argument("--cost", required=True, type=Path)
    parser.add_argument("--metrics", type=Path, help="Per-step CSV from analyze_score_centering.py --metrics-output")
    parser.add_argument("--run", action="append", required=True, metavar="RUN=LABEL")
    parser.add_argument("--title", default="Score centering: held-out quality")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    arms = {}
    for item in args.run:
        run, separator, label = item.partition("=")
        if not separator or not run or not label or run in arms:
            parser.error(f"invalid or duplicate --run {item!r}; use RUN=LABEL")
        arms[run] = label
    plot_curves(args.evaluations, args.cost, arms, args.output, args.title, args.metrics)
    print(f"Wrote quality curves to {args.output}")


if __name__ == "__main__":
    main()
