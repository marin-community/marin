# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch W&B HBM usage for the Coral batch config study."""

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

import wandb
from fray.types import get_tpu_topology

from experiments.coral.batch_config import BYTES_PER_GIB
from experiments.coral.batch_config_study import (
    CASES,
    DEFAULT_VERSION,
    STUDY_ID,
    StudyCase,
    estimate_case,
)


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
    microbatch_pred_gib: float
    microbatch_pred_chip_gib: float
    microbatch_pred_percent: float
    microbatch_pred_multiplier: float


def main() -> None:
    args = _parse_args()
    group, rows = load_case_usage(args)
    print(f"group: {group}")
    print(f"finished runs with HBM metrics: {len(rows)}")
    _print_summary(rows)
    if args.details:
        print()
        _print_table(rows)


def load_case_usage(args: argparse.Namespace) -> tuple[str, list[CaseUsage]]:
    api = wandb.Api()
    study_id = args.study_id
    case_by_run_id = {f"coral-batch-config-{study_id}-{case.name}-{args.version}": case for case in CASES}
    group = f"coral-batch-config-study-{study_id}-{args.version}"
    rows = []
    for run in api.runs(
        f"{args.entity}/{args.project}",
        filters={"group": group, "state": "finished"},
    ):
        run_id = run.id
        case = case_by_run_id.get(run_id)
        if case is None:
            continue
        usage = _fetch_hbm_usage(run)
        if usage is None:
            continue
        rows.append(_case_usage(case, run_id, run.state, usage))

    rows.sort(key=lambda row: CASES.index(row.case))
    return group, rows


def _print_summary(rows: list[CaseUsage]) -> None:
    rows_by_case: dict[str, list[CaseUsage]] = defaultdict(list)
    for row in rows:
        rows_by_case[row.case.name].append(row)

    headers = [
        "case",
        "tpu",
        "global_batch",
        "microbatch",
        "grad_accum",
        "n",
        "max_hbm_gib_mean",
        "max_hbm_pct_mean",
        "micro_pred_chip_gib",
        "micro_pred_pct",
        "pred/actual_mean",
        "pred/actual_range",
    ]
    print("\t".join(headers))
    for case in CASES:
        case_rows = rows_by_case.get(case.name)
        if not case_rows:
            continue
        first = case_rows[0]
        ratios = [row.microbatch_pred_multiplier for row in case_rows]
        print(
            "\t".join(
                [
                    case.name,
                    case.tpu,
                    str(case.batch_size),
                    str(first.microbatch_size),
                    str(first.gradient_accumulation),
                    str(len(case_rows)),
                    f"{mean(row.max_chip_gib for row in case_rows):.2f}",
                    f"{mean(row.max_chip_percent for row in case_rows):.2f}",
                    f"{first.microbatch_pred_chip_gib:.2f}",
                    f"{first.microbatch_pred_percent:.2f}",
                    f"{mean(ratios):.2f}x",
                    f"{min(ratios):.2f}-{max(ratios):.2f}x",
                ]
            )
        )


def _case_usage(case: StudyCase, run_id: str, state: str, usage: HbmUsage) -> CaseUsage:
    estimate = estimate_case(case)
    microbatch_size = _microbatch_size(case, estimate.per_device_parallelism)
    microbatch_pred_bytes = math.ceil(estimate.total_bytes * microbatch_size / case.batch_size)
    full_pred_percent = estimate.total_bytes / usage.observed_slice_capacity_bytes * 100
    microbatch_pred_percent = microbatch_pred_bytes / usage.observed_slice_capacity_bytes * 100
    return CaseUsage(
        case=case,
        run_id=run_id,
        state=state,
        microbatch_size=microbatch_size,
        gradient_accumulation=estimate.gradient_accumulation,
        max_chip_gib=usage.max_chip_gib,
        max_chip_percent=usage.max_chip_percent,
        full_pred_gib=estimate.total_bytes / BYTES_PER_GIB,
        full_pred_percent=full_pred_percent,
        microbatch_pred_gib=microbatch_pred_bytes / BYTES_PER_GIB,
        microbatch_pred_chip_gib=microbatch_pred_bytes / usage.chip_count / BYTES_PER_GIB,
        microbatch_pred_percent=microbatch_pred_percent,
        microbatch_pred_multiplier=(microbatch_pred_bytes / usage.chip_count) / (usage.max_chip_gib * BYTES_PER_GIB),
    )


def _microbatch_size(case: StudyCase, per_device_parallelism: int) -> int:
    if per_device_parallelism == -1:
        return case.batch_size
    return per_device_parallelism * get_tpu_topology(case.tpu).chip_count


def _fetch_hbm_usage(run) -> HbmUsage | None:
    hbm_usage_by_chip: dict[int, list[float]] = {}
    hbm_percent_by_chip: dict[int, list[float]] = {}
    capacity_by_chip: dict[int, int] = {}

    for row in run.history(samples=5000, stream="system", pandas=False):
        for key, value in row.items():
            if not isinstance(value, (int, float)):
                continue
            if match := re.fullmatch(r"system\.tpu\.(\d+)\.hbmCapacityUsage", key):
                hbm_usage_by_chip.setdefault(int(match.group(1)), []).append(float(value))
            elif match := re.fullmatch(r"system\.tpu\.(\d+)\.hbmMemoryUsage", key):
                hbm_percent_by_chip.setdefault(int(match.group(1)), []).append(float(value))
            elif match := re.fullmatch(r"system\.tpu\.(\d+)\.hbmCapacityTotal", key):
                capacity_by_chip[int(match.group(1))] = int(value)

    if not hbm_usage_by_chip or not hbm_percent_by_chip:
        return None

    max_chip_bytes = max(max(values) for values in hbm_usage_by_chip.values() if values)
    max_chip_percent = max(max(values) for values in hbm_percent_by_chip.values() if values)
    return HbmUsage(
        max_chip_gib=max_chip_bytes / BYTES_PER_GIB,
        max_chip_percent=max_chip_percent,
        observed_slice_capacity_bytes=sum(capacity_by_chip.values()),
        chip_count=len(capacity_by_chip),
    )


def _print_table(rows: list[CaseUsage]) -> None:
    headers = [
        "case",
        "tpu",
        "global_batch",
        "microbatch",
        "grad_accum",
        "max_hbm_gib",
        "max_hbm_pct",
        "full_pred_gib",
        "full_pred_pct",
        "micro_pred_gib",
        "micro_pred_chip_gib",
        "micro_pred_pct",
        "micro_pred/actual",
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
                    f"{row.full_pred_gib:.2f}",
                    f"{row.full_pred_percent:.2f}",
                    f"{row.microbatch_pred_gib:.2f}",
                    f"{row.microbatch_pred_chip_gib:.2f}",
                    f"{row.microbatch_pred_percent:.2f}",
                    f"{row.microbatch_pred_multiplier:.2f}x",
                ]
            )
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="eric-czech")
    parser.add_argument("--project", default="marin")
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--study-id", default=STUDY_ID)
    parser.add_argument("--details", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
