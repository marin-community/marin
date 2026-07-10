# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot W&B HBM calibration for the Coral batch config study."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiments.coral.batch_config_study import DEFAULT_VERSION, STUDY_ID
from experiments.coral.batch_config_study.fetch_wandb import CaseUsage, load_case_usage

plt.switch_backend("Agg")


@dataclass(frozen=True)
class CasePoint:
    tpu: str
    model_name: str
    batch_size: int
    microbatch_size: int
    gradient_accumulation: int
    replicate_count: int
    actual_mean_gib: float
    actual_min_gib: float
    actual_max_gib: float
    actual_mean_percent: float
    actual_min_percent: float
    actual_max_percent: float
    estimated_chip_gib: float
    estimated_chip_percent: float
    ratio_mean: float


def main() -> None:
    args = _parse_args()
    _, rows = load_case_usage(args)
    points = _aggregate(rows)
    if not points:
        raise ValueError("No finished W&B runs with HBM metrics found.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _plot(points, output_path)
    print(f"Wrote {output_path}")


def _aggregate(rows: list[CaseUsage]) -> list[CasePoint]:
    rows_by_case: dict[tuple[str, str, int], list[CaseUsage]] = defaultdict(list)
    for row in rows:
        rows_by_case[(row.case.tpu, row.case.model_name, row.case.batch_size)].append(row)

    points = []
    for (tpu, model_name, batch_size), case_rows in rows_by_case.items():
        first = case_rows[0]
        actual_values = [row.max_chip_gib for row in case_rows]
        actual_percent_values = [row.max_chip_percent for row in case_rows]
        ratios = [row.microbatch_pred_multiplier for row in case_rows]
        points.append(
            CasePoint(
                tpu=tpu,
                model_name=model_name,
                batch_size=batch_size,
                microbatch_size=first.microbatch_size,
                gradient_accumulation=first.gradient_accumulation,
                replicate_count=len(case_rows),
                actual_mean_gib=mean(actual_values),
                actual_min_gib=min(actual_values),
                actual_max_gib=max(actual_values),
                actual_mean_percent=mean(actual_percent_values),
                actual_min_percent=min(actual_percent_values),
                actual_max_percent=max(actual_percent_values),
                estimated_chip_gib=first.microbatch_pred_chip_gib,
                estimated_chip_percent=first.microbatch_pred_percent,
                ratio_mean=mean(ratios),
            )
        )
    return sorted(points, key=lambda point: (point.tpu, point.batch_size, point.model_name))


def _plot(points: list[CasePoint], output_path: Path) -> None:
    colors = {
        "v5litepod-4": "#2a6fbb",
        "v5litepod-8": "#d1495b",
        "v6e-4": "#168f6a",
    }
    markers = {
        "llama150m": "o",
        "llama600m": "s",
        "llama2p4b": "^",
    }
    batch_sizes = sorted({point.batch_size for point in points})
    size_by_batch = {batch_size: 70 + 45 * i for i, batch_size in enumerate(batch_sizes)}

    max_axis_gib = max(max(point.actual_max_gib, point.estimated_chip_gib) for point in points) * 1.15
    max_axis_percent = max(max(point.actual_max_percent, point.estimated_chip_percent) for point in points) * 1.15
    max_axis_percent = max(max_axis_percent, 100)

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 7.1))
    _plot_panel(
        axes[0],
        points,
        colors,
        markers,
        size_by_batch,
        max_axis_gib,
        "Actual max per-chip HBM from W&B (GiB)",
        "Estimated per-chip microbatch HBM (GiB)",
        "GiB",
        lambda point: (point.actual_mean_gib, point.actual_min_gib, point.actual_max_gib, point.estimated_chip_gib),
    )
    _plot_panel(
        axes[1],
        points,
        colors,
        markers,
        size_by_batch,
        max_axis_percent,
        "Actual max HBM utilization from W&B (%)",
        "Estimated microbatch HBM utilization (%)",
        "% HBM",
        lambda point: (
            point.actual_mean_percent,
            point.actual_min_percent,
            point.actual_max_percent,
            point.estimated_chip_percent,
        ),
    )

    tpu_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=9,
            label=tpu,
        )
        for tpu, color in colors.items()
    ]
    model_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="#666666",
            markeredgecolor="white",
            markersize=9,
            label=model.replace("llama", "llama "),
        )
        for model, marker in markers.items()
    ]
    batch_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#aaaaaa",
            markeredgecolor="white",
            markersize=(size_by_batch[batch_size] ** 0.5),
            label=f"batch {batch_size}",
        )
        for batch_size in batch_sizes
    ]
    grad_accum_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#aaaaaa",
            markeredgecolor="#111111",
            markeredgewidth=1.7,
            markersize=9,
            label="grad accum > 1",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#aaaaaa",
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=9,
            label="grad accum = 1",
        ),
    ]
    calibration_handles = [
        Line2D([0], [0], color="#222222", linewidth=1.3, label="perfect calibration"),
        Line2D([0], [0], color="#222222", linewidth=7, alpha=0.08, label="+/-20%"),
    ]

    ratios = [point.ratio_mean for point in points]
    summary = (
        f"{sum(point.replicate_count for point in points)} finished runs, "
        f"{len(points)} TPU/model/batch groups. Shaded band is +/-20%. "
        f"Median estimated/actual: {median(ratios):.2f}x; range: {min(ratios):.2f}-{max(ratios):.2f}x."
    )
    fig.suptitle("Batch Config HBM Calibration", fontsize=16, y=0.9)
    fig.text(0.5, 0.845, summary, ha="center", fontsize=10, color="#555555")

    legend_specs = [
        (calibration_handles, "Calibration", 0.15, 0.125, 2),
        (tpu_handles, "TPU", 0.45, 0.125, 3),
        (model_handles, "Model", 0.76, 0.125, 3),
        (batch_handles, "Global batch", 0.35, 0.04, 3),
        (grad_accum_handles, "Schedule", 0.72, 0.04, 2),
    ]
    for handles, title, x_anchor, y_anchor, ncol in legend_specs:
        fig.legend(
            handles=handles,
            title=title,
            loc="lower center",
            ncol=ncol,
            frameon=False,
            bbox_to_anchor=(x_anchor, y_anchor),
            borderaxespad=0,
            columnspacing=1.0,
            handletextpad=0.5,
            fontsize=9,
            title_fontsize=10,
        )

    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.28, top=0.78, wspace=0.16)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")


def _plot_panel(
    ax,
    points: list[CasePoint],
    colors: dict[str, str],
    markers: dict[str, str],
    size_by_batch: dict[int, int],
    max_axis: float,
    x_label: str,
    y_label: str,
    title: str,
    values,
) -> None:
    ax.plot([0, max_axis], [0, max_axis], color="#222222", linewidth=1.3)
    ax.fill_between([0, max_axis], [0, max_axis * 0.8], [0, max_axis * 1.2], color="#222222", alpha=0.08)

    for point in points:
        actual_mean, actual_min, actual_max, estimated = values(point)
        xerr = [[actual_mean - actual_min], [actual_max - actual_mean]]
        ax.errorbar(
            actual_mean,
            estimated,
            xerr=xerr,
            fmt=markers.get(point.model_name, "o"),
            markersize=(size_by_batch[point.batch_size] ** 0.5),
            color=colors.get(point.tpu, "#555555"),
            markeredgecolor="#111111" if point.gradient_accumulation > 1 else "white",
            markeredgewidth=1.7 if point.gradient_accumulation > 1 else 0.8,
            elinewidth=1.0,
            capsize=2,
            alpha=0.9,
        )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(0, max_axis)
    ax.set_ylim(0, max_axis)
    ax.grid(True, color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="eric-czech")
    parser.add_argument("--project", default="marin")
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--study-id", default=STUDY_ID)
    parser.add_argument(
        "--output",
        default="experiments/coral/batch_config_study/results/batch_config_calibration.png",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
