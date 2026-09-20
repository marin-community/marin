"""Plot completed-correct quality against updates, elapsed time, and reserved GPU-hours.

Use the CSV outputs of analyze_score_centering.py and analyze_score_centering_cost.py.
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


def plot_curves(evaluations: Path, cost: Path, arms: dict[str, str], output: Path, title: str) -> None:
    with evaluations.open(newline="") as stream:
        quality_rows = [row for row in csv.DictReader(stream) if row["dataset"] == "all"]
    with cost.open(newline="") as stream:
        cost_rows = {(row["run"], int(row["step"])): row for row in csv.DictReader(stream)}

    by_run = defaultdict(list)
    for row in quality_rows:
        run, step = row["run"], int(row["step"])
        if run not in arms:
            continue
        cost_row = cost_rows.get((run, step))
        if cost_row is None:
            raise ValueError(f"{run} step {step}: missing task cost")
        by_run[run].append(
            (
                step,
                float(cost_row["elapsed_from_first_gpu_task_hours"]),
                float(cost_row["reserved_gpu_hours_to_eval"]),
                100 * float(row["completed_correct_rate"]),
            )
        )
    if set(by_run) != set(arms):
        raise ValueError(f"missing evaluated arms: {sorted(set(arms) - set(by_run))}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True, constrained_layout=True)
    x_labels = ("Optimizer updates", "Elapsed GPU-task hours", "Reserved H100-hours")
    markers = ("o", "s", "^", "D", "v", "P", "X", "*")
    for arm_index, (run, label) in enumerate(arms.items()):
        points = sorted(by_run[run])
        for axis_index, ax in enumerate(axes):
            ax.plot(
                [point[axis_index] for point in points],
                [point[3] for point in points],
                marker=markers[arm_index % len(markers)],
                markersize=5,
                linewidth=1.8,
                label=label,
            )
    for ax, label in zip(axes, x_labels, strict=True):
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", required=True, type=Path)
    parser.add_argument("--cost", required=True, type=Path)
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
    plot_curves(args.evaluations, args.cost, arms, args.output, args.title)
    print(f"Wrote quality curves to {args.output}")


if __name__ == "__main__":
    main()
