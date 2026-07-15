# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analysis for the batch-calibration experiment: fetch measured HBM from W&B, tabulate, and plot the
estimate vs. measurement to calibrate ``batch_calibration.correction_factor``.

Each run's estimated per-chip HBM (from ``batch_calibration`` at ``correction_factor = 1``) is compared
to the measured per-chip peak. Their ratio gives the correction factor that would align the two; the
point of the study is the *range* of that factor across TPUs, models, and batch sizes.

    python -m experiments.coral.batch_calibration_experiment.analysis            # summary + plot
    python -m experiments.coral.batch_calibration_experiment.analysis --details  # + per-run table
    python -m experiments.coral.batch_calibration_experiment.analysis --validate # model param counts
"""

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
import wandb
from fray.types import get_tpu_topology
from matplotlib.lines import Line2D

from experiments.coral.batch_calibration import BYTES_PER_GIB
from experiments.coral.batch_calibration_experiment import (
    CASES,
    DEFAULT_VERSION,
    MODEL_SPECS,
    STUDY_ID,
    StudyCase,
    estimate_case,
)
from experiments.llama import llama3_tokenizer_vocab_size

plt.switch_backend("Agg")

DEFAULT_OUTPUT = "experiments/coral/batch_calibration_experiment/results/batch_calibration_results.png"


# --- fetch measured HBM ------------------------------------------------------


@dataclass(frozen=True)
class HbmUsage:
    max_chip_gib: float
    max_chip_percent: float
    observed_slice_capacity_bytes: int
    chip_count: int


@dataclass(frozen=True)
class CaseUsage:
    case: StudyCase
    run_id: str
    state: str
    microbatch_size: int
    gradient_accumulation: int
    max_chip_gib: float
    max_chip_percent: float
    full_pred_gib: float
    full_pred_percent: float
    est_chip_gib: float
    est_chip_percent: float
    est_over_measured: float  # estimated per-chip / measured per-chip; its reciprocal is the correction factor


def load_case_usage(args: argparse.Namespace) -> tuple[str, list[CaseUsage]]:
    api = wandb.Api()
    study_id = args.study_id
    case_by_run_id = {f"coral-batch-config-{study_id}-{case.name}-{args.version}": case for case in CASES}
    group = f"coral-batch-config-study-{study_id}-{args.version}"  # historical W&B group naming the collected runs
    rows = []
    for run in api.runs(f"{args.entity}/{args.project}", filters={"group": group, "state": "finished"}):
        case = case_by_run_id.get(run.id)
        if case is None:
            continue
        usage = _fetch_hbm_usage(run)
        if usage is None:
            continue
        rows.append(_case_usage(case, run.id, run.state, usage))
    rows.sort(key=lambda row: CASES.index(row.case))
    return group, rows


def _fetch_hbm_usage(run) -> HbmUsage | None:
    usage_by_chip: dict[int, list[float]] = {}
    percent_by_chip: dict[int, list[float]] = {}
    capacity_by_chip: dict[int, int] = {}
    for row in run.history(samples=5000, stream="system", pandas=False):
        for key, value in row.items():
            if not isinstance(value, (int, float)):
                continue
            if match := re.fullmatch(r"system\.tpu\.(\d+)\.hbmCapacityUsage", key):
                usage_by_chip.setdefault(int(match.group(1)), []).append(float(value))
            elif match := re.fullmatch(r"system\.tpu\.(\d+)\.hbmMemoryUsage", key):
                percent_by_chip.setdefault(int(match.group(1)), []).append(float(value))
            elif match := re.fullmatch(r"system\.tpu\.(\d+)\.hbmCapacityTotal", key):
                capacity_by_chip[int(match.group(1))] = int(value)
    if not usage_by_chip or not percent_by_chip:
        return None
    return HbmUsage(
        max_chip_gib=max(max(v) for v in usage_by_chip.values() if v) / BYTES_PER_GIB,
        max_chip_percent=max(max(v) for v in percent_by_chip.values() if v),
        observed_slice_capacity_bytes=sum(capacity_by_chip.values()),
        chip_count=len(capacity_by_chip),
    )


def _microbatch_size(case: StudyCase, per_device_parallelism: int) -> int:
    if per_device_parallelism == -1:
        return case.batch_size
    return per_device_parallelism * get_tpu_topology(case.tpu).chip_count


def _case_usage(case: StudyCase, run_id: str, state: str, usage: HbmUsage) -> CaseUsage:
    estimate = estimate_case(case)
    microbatch_size = _microbatch_size(case, estimate.per_device_parallelism)
    est_bytes = math.ceil(estimate.total_bytes * microbatch_size / case.batch_size)
    est_chip_bytes = est_bytes / usage.chip_count
    return CaseUsage(
        case=case,
        run_id=run_id,
        state=state,
        microbatch_size=microbatch_size,
        gradient_accumulation=estimate.gradient_accumulation,
        max_chip_gib=usage.max_chip_gib,
        max_chip_percent=usage.max_chip_percent,
        full_pred_gib=estimate.total_bytes / BYTES_PER_GIB,
        full_pred_percent=estimate.total_bytes / usage.observed_slice_capacity_bytes * 100,
        est_chip_gib=est_chip_bytes / BYTES_PER_GIB,
        est_chip_percent=est_bytes / usage.observed_slice_capacity_bytes * 100,
        est_over_measured=(est_chip_bytes) / (usage.max_chip_gib * BYTES_PER_GIB),
    )


def _print_summary(rows: list[CaseUsage]) -> None:
    by_case: dict[str, list[CaseUsage]] = defaultdict(list)
    for row in rows:
        by_case[row.case.name].append(row)
    headers = [
        "case",
        "tpu",
        "global_batch",
        "microbatch",
        "grad_accum",
        "n",
        "measured_gib_mean",
        "measured_pct_mean",
        "est_chip_gib",
        "est_pct",
        "corr_factor_mean",
        "corr_factor_range",
    ]
    print("\t".join(headers))
    for case in CASES:
        case_rows = by_case.get(case.name)
        if not case_rows:
            continue
        first = case_rows[0]
        corrections = [1.0 / row.est_over_measured for row in case_rows]
        print(
            "\t".join(
                [
                    case.name,
                    case.tpu,
                    str(case.batch_size),
                    str(first.microbatch_size),
                    str(first.gradient_accumulation),
                    str(len(case_rows)),
                    f"{mean(r.max_chip_gib for r in case_rows):.2f}",
                    f"{mean(r.max_chip_percent for r in case_rows):.2f}",
                    f"{first.est_chip_gib:.2f}",
                    f"{first.est_chip_percent:.2f}",
                    f"{mean(corrections):.2f}",
                    f"{min(corrections):.2f}-{max(corrections):.2f}",
                ]
            )
        )


def _print_table(rows: list[CaseUsage]) -> None:
    headers = [
        "case",
        "tpu",
        "global_batch",
        "microbatch",
        "grad_accum",
        "measured_gib",
        "measured_pct",
        "est_chip_gib",
        "est_pct",
        "corr_factor",
    ]
    print("\t".join(headers))
    for row in rows:
        print(
            "\t".join(
                [
                    row.case.name,
                    row.case.tpu,
                    str(row.case.batch_size),
                    str(row.microbatch_size),
                    str(row.gradient_accumulation),
                    f"{row.max_chip_gib:.2f}",
                    f"{row.max_chip_percent:.2f}",
                    f"{row.est_chip_gib:.2f}",
                    f"{row.est_chip_percent:.2f}",
                    f"{1.0 / row.est_over_measured:.2f}",
                ]
            )
        )


# --- plot --------------------------------------------------------------------


@dataclass(frozen=True)
class CasePoint:
    tpu: str
    model_name: str
    batch_size: int
    gradient_accumulation: int
    replicate_count: int
    measured_mean_gib: float
    measured_min_gib: float
    measured_max_gib: float
    measured_mean_percent: float
    measured_min_percent: float
    measured_max_percent: float
    est_chip_gib: float
    est_chip_percent: float
    correction_mean: float


def _aggregate(rows: list[CaseUsage]) -> list[CasePoint]:
    by_case: dict[tuple[str, str, int], list[CaseUsage]] = defaultdict(list)
    for row in rows:
        by_case[(row.case.tpu, row.case.model_name, row.case.batch_size)].append(row)
    points = []
    for (tpu, model_name, batch_size), case_rows in by_case.items():
        first = case_rows[0]
        gib = [r.max_chip_gib for r in case_rows]
        pct = [r.max_chip_percent for r in case_rows]
        corrections = [1.0 / r.est_over_measured for r in case_rows]
        points.append(
            CasePoint(
                tpu=tpu,
                model_name=model_name,
                batch_size=batch_size,
                gradient_accumulation=first.gradient_accumulation,
                replicate_count=len(case_rows),
                measured_mean_gib=mean(gib),
                measured_min_gib=min(gib),
                measured_max_gib=max(gib),
                measured_mean_percent=mean(pct),
                measured_min_percent=min(pct),
                measured_max_percent=max(pct),
                est_chip_gib=first.est_chip_gib,
                est_chip_percent=first.est_chip_percent,
                correction_mean=mean(corrections),
            )
        )
    return sorted(points, key=lambda p: (p.tpu, p.batch_size, p.model_name))


def _plot(points: list[CasePoint], output_path: Path) -> None:
    colors = {"v5litepod-4": "#2a6fbb", "v5litepod-8": "#d1495b", "v6e-4": "#168f6a"}
    markers = {"llama150m": "o", "llama600m": "s", "llama2p4b": "^"}
    batch_sizes = sorted({p.batch_size for p in points})
    size_by_batch = {b: 70 + 45 * i for i, b in enumerate(batch_sizes)}

    max_gib = max(max(p.measured_max_gib, p.est_chip_gib) for p in points) * 1.15
    max_pct = max(max(max(p.measured_max_percent, p.est_chip_percent) for p in points) * 1.15, 100)

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 7.1))
    _plot_panel(
        axes[0],
        points,
        colors,
        markers,
        size_by_batch,
        max_gib,
        "Measured per-chip HBM (GiB)",
        "Estimated per-chip HBM (GiB)",
        "Per-chip HBM (GiB)",
        lambda p: (p.measured_mean_gib, p.measured_min_gib, p.measured_max_gib, p.est_chip_gib),
    )
    _plot_panel(
        axes[1],
        points,
        colors,
        markers,
        size_by_batch,
        max_pct,
        "Measured HBM utilization (%)",
        "Estimated HBM utilization (%)",
        "HBM utilization (%)",
        lambda p: (p.measured_mean_percent, p.measured_min_percent, p.measured_max_percent, p.est_chip_percent),
    )

    tpu_handles = [_dot(color, "white", 9, tpu) for tpu, color in colors.items()]
    model_handles = [
        _marker(marker, "#666666", "white", 9, model.replace("llama", "llama ")) for model, marker in markers.items()
    ]
    batch_handles = [_dot("#aaaaaa", "white", size_by_batch[b] ** 0.5, f"batch {b}") for b in batch_sizes]
    accum_handles = [
        _dot("#aaaaaa", "#111111", 9, "grad accum > 1", edge_width=1.7),
        _dot("#aaaaaa", "white", 9, "grad accum = 1", edge_width=0.8),
    ]
    calib_handles = [
        Line2D([0], [0], color="#222222", linewidth=1.3, label="estimate = measured"),
        Line2D([0], [0], color="#222222", linewidth=7, alpha=0.08, label="+/-20%"),
    ]

    corrections = [p.correction_mean for p in points]
    summary = (
        f"{sum(p.replicate_count for p in points)} finished runs across {len(points)} TPU/model/batch groups. "
        f"Each point's correction_factor = measured / estimated per-chip HBM: "
        f"median {median(corrections):.2f}, range {min(corrections):.2f}-{max(corrections):.2f}. "
        f"Diagonal = estimate matches measured; shaded band is +/-20%."
    )
    fig.suptitle("Batch Calibration Results", fontsize=16, y=0.9)
    fig.text(0.5, 0.845, summary, ha="center", fontsize=10, color="#555555")

    legend_specs = [
        (calib_handles, "Calibration", 0.15, 0.125, 2),
        (tpu_handles, "TPU", 0.45, 0.125, 3),
        (model_handles, "Model", 0.76, 0.125, 3),
        (batch_handles, "Global batch", 0.35, 0.04, 3),
        (accum_handles, "Schedule", 0.72, 0.04, 2),
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


def _dot(face: str, edge: str, size: float, label: str, edge_width: float = 0.8) -> Line2D:
    return Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=face,
        markeredgecolor=edge,
        markeredgewidth=edge_width,
        markersize=size,
        label=label,
    )


def _marker(marker: str, face: str, edge: str, size: float, label: str) -> Line2D:
    return Line2D(
        [0], [0], marker=marker, color="none", markerfacecolor=face, markeredgecolor=edge, markersize=size, label=label
    )


def _plot_panel(ax, points, colors, markers, size_by_batch, max_axis, x_label, y_label, title, values) -> None:
    ax.plot([0, max_axis], [0, max_axis], color="#222222", linewidth=1.3)
    ax.fill_between([0, max_axis], [0, max_axis * 0.8], [0, max_axis * 1.2], color="#222222", alpha=0.08)
    for point in points:
        measured_mean, measured_min, measured_max, estimated = values(point)
        ax.errorbar(
            measured_mean,
            estimated,
            xerr=[[measured_mean - measured_min], [measured_max - measured_mean]],
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


# --- entry point -------------------------------------------------------------


def _validate() -> None:
    for spec in MODEL_SPECS:
        print(f"{spec.name}\t{spec.model.total_trainable_params(llama3_tokenizer_vocab_size):,}")


def main() -> None:
    args = _parse_args()
    if args.validate:
        _validate()
        return
    group, rows = load_case_usage(args)
    print(f"group: {group}")
    print(f"finished runs with HBM metrics: {len(rows)}")
    _print_summary(rows)
    if args.details:
        print()
        _print_table(rows)
    points = _aggregate(rows)
    if not points:
        raise ValueError("No finished W&B runs with HBM metrics found.")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _plot(points, output_path)
    print(f"\nWrote {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entity", default="eric-czech")
    parser.add_argument("--project", default="marin")
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--study-id", default=STUDY_ID)
    parser.add_argument("--details", action="store_true", help="also print the per-run table")
    parser.add_argument("--validate", action="store_true", help="print model parameter counts and exit")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
