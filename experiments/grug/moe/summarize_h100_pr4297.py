# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize the PR 4297 H100x8 XLA vs Triton validation runs from W&B."""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import wandb

DEFAULT_WANDB_ENTITY = "marin-community"
DEFAULT_WANDB_PROJECT = "marin"
STEP_KEY = "global_step"
DEFAULT_WINDOW_START = 20
DEFAULT_WINDOW_END = 94


@dataclass(frozen=True)
class MetricSpec:
    label: str
    key: str
    delta_mode: str


@dataclass(frozen=True)
class PairSpec:
    label: str
    xla_run_id: str
    triton_run_id: str


@dataclass(frozen=True)
class RunWindowSummary:
    run_id: str
    state: str
    summary_global_step: int | None
    window_rows: int
    metrics: dict[str, float]


@dataclass(frozen=True)
class PairWindowSummary:
    label: str
    xla: RunWindowSummary
    triton: RunWindowSummary


METRICS = (
    MetricSpec("examples/s", "throughput/examples_per_second", "percent"),
    MetricSpec("tokens/s", "throughput/tokens_per_second", "percent"),
    MetricSpec("MFU", "throughput/mfu", "points"),
    MetricSpec("duration", "throughput/duration", "percent"),
    MetricSpec("loss", "train/loss", "loss"),
)

DEFAULT_PAIRS = (
    PairSpec(
        label="pair1",
        xla_run_id="pr4297-grug-moe-256m-xla-20260402-024606",
        triton_run_id="pr4297-grug-moe-256m-triton-20260402-030301",
    ),
    PairSpec(
        label="pair2",
        xla_run_id="pr4297-grug-moe-256m-xla-p2-20260402-032629",
        triton_run_id="pr4297-grug-moe-256m-triton-p2-20260402-033540",
    ),
    PairSpec(
        label="pair3",
        xla_run_id="pr4297-grug-moe-256m-xla-p3-20260402-034620",
        triton_run_id="pr4297-grug-moe-256m-triton-p3-20260402-035616",
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--window-start", type=int, default=DEFAULT_WINDOW_START)
    parser.add_argument("--window-end", type=int, default=DEFAULT_WINDOW_END)
    return parser.parse_args()


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_step(value: object) -> int | None:
    step = _coerce_float(value)
    if step is None:
        return None
    return int(step)


def _window_rows(run: wandb.apis.public.Run, window_start: int, window_end: int) -> dict[int, dict[str, float]]:
    history_keys = [STEP_KEY, *[metric.key for metric in METRICS]]
    rows_by_step: dict[int, dict[str, float]] = {}
    for row in run.scan_history(keys=history_keys):
        step = _coerce_step(row.get(STEP_KEY))
        if step is None or step < window_start or step > window_end:
            continue
        metric_values: dict[str, float] = {}
        for metric in METRICS:
            value = _coerce_float(row.get(metric.key))
            if value is None:
                break
            metric_values[metric.key] = value
        else:
            rows_by_step[step] = metric_values
    return rows_by_step


def _summarize_run(
    api: wandb.Api,
    *,
    entity: str,
    project: str,
    run_id: str,
    window_start: int,
    window_end: int,
) -> RunWindowSummary:
    run = api.run(f"{entity}/{project}/{run_id}")
    rows_by_step = _window_rows(run, window_start=window_start, window_end=window_end)
    if not rows_by_step:
        raise ValueError(f"Run {run_id} had no complete rows in the {window_start}-{window_end} window.")
    metrics = {metric.key: statistics.median(row[metric.key] for row in rows_by_step.values()) for metric in METRICS}
    return RunWindowSummary(
        run_id=run_id,
        state=run.state,
        summary_global_step=_coerce_step(run.summary.get(STEP_KEY)),
        window_rows=len(rows_by_step),
        metrics=metrics,
    )


def _summarize_pairs(
    api: wandb.Api,
    *,
    entity: str,
    project: str,
    window_start: int,
    window_end: int,
) -> list[PairWindowSummary]:
    return [
        PairWindowSummary(
            label=pair.label,
            xla=_summarize_run(
                api,
                entity=entity,
                project=project,
                run_id=pair.xla_run_id,
                window_start=window_start,
                window_end=window_end,
            ),
            triton=_summarize_run(
                api,
                entity=entity,
                project=project,
                run_id=pair.triton_run_id,
                window_start=window_start,
                window_end=window_end,
            ),
        )
        for pair in DEFAULT_PAIRS
    ]


def _aggregate_variant(pairs: list[PairWindowSummary], variant: str) -> dict[str, float]:
    return {
        metric.key: statistics.median(getattr(pair, variant).metrics[metric.key] for pair in pairs) for metric in METRICS
    }


def _format_metric(metric: MetricSpec, value: float) -> str:
    if metric.label in {"examples/s", "tokens/s"}:
        return f"{value:.2f}"
    if metric.label == "MFU":
        return f"{value:.2f}%"
    if metric.label == "duration":
        return f"{value:.4f} s"
    return f"{value:.6f}"


def _format_delta(metric: MetricSpec, xla_value: float, triton_value: float) -> str:
    if metric.delta_mode == "percent":
        delta = 100.0 * (triton_value - xla_value) / xla_value
        return f"{delta:+.1f}%"
    if metric.delta_mode == "points":
        delta = triton_value - xla_value
        return f"{delta:+.2f} pts"
    loss_delta = abs(triton_value - xla_value)
    return f"matched within {loss_delta:.5f}"


def _print_coverage(pairs: list[PairWindowSummary], window_start: int, window_end: int) -> None:
    print("Run coverage:")
    print("| pair | variant | state | summary_global_step | window_rows |")
    print("| --- | --- | --- | ---: | ---: |")
    for pair in pairs:
        for variant in ("xla", "triton"):
            summary = getattr(pair, variant)
            summary_step = "n/a" if summary.summary_global_step is None else str(summary.summary_global_step)
            print(f"| {pair.label} | `{variant}` | `{summary.state}` | {summary_step} | {summary.window_rows} |")
    print()
    print(
        f"Shared comparison window: `{window_start}-{window_end}`. "
        "Only the first XLA run stopped at step `94`; the later XLA pairs reached step `99`."
    )
    print()


def _print_aggregate_table(pairs: list[PairWindowSummary]) -> None:
    xla_metrics = _aggregate_variant(pairs, "xla")
    triton_metrics = _aggregate_variant(pairs, "triton")
    print("| metric | `xla` | `triton` | delta |")
    print("| --- | ---: | ---: | ---: |")
    for metric in METRICS:
        print(
            f"| {metric.label} | `{_format_metric(metric, xla_metrics[metric.key])}` | "
            f"`{_format_metric(metric, triton_metrics[metric.key])}` | "
            f"`{_format_delta(metric, xla_metrics[metric.key], triton_metrics[metric.key])}` |"
        )


def main() -> None:
    args = _parse_args()
    api = wandb.Api()
    pairs = _summarize_pairs(
        api,
        entity=args.entity,
        project=args.project,
        window_start=args.window_start,
        window_end=args.window_end,
    )
    _print_coverage(pairs, window_start=args.window_start, window_end=args.window_end)
    _print_aggregate_table(pairs)


if __name__ == "__main__":
    main()
