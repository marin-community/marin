# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec[gcs]"]
# ///

"""Freeze disjoint East5 and Europe assignments for the remaining TPP40 rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import fsspec

from experiments.domain_phase_mix.launch_delphi_augmented_swarm_tpp40 import (
    EXPECTED_FINAL_CHECKPOINT_STEP,
    EXPECTED_PHASE0_CHECKPOINT_STEP,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import executor_status_succeeded

EXPECTED_RUNS = 280
DEFAULT_EAST5_ROOT = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/" "delphi_augmented_swarm_tpp40_phase0_checkpoint_20260815"
)
DEFAULT_EUROPE_ROOT = (
    "gs://marin-eu-west4/pinlin_calvin_xu/data_mixture/" "delphi_augmented_swarm_tpp40_phase0_checkpoint_20260815"
)
DEFAULT_SOURCE_PANEL = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_augmented_swarm_3e18_20260714/"
    "source/fit_panel_table9_macro-4f283bacb4ef269c.csv"
)
FINAL_MARKER = f"hf/step-{EXPECTED_FINAL_CHECKPOINT_STEP}/model.safetensors"
PHASE0_MARKER = f"checkpoints/step-{EXPECTED_PHASE0_CHECKPOINT_STEP}/metadata.json"
EXECUTOR_STATUS_MARKER = ".executor_status"
EXECUTOR_INFO_MARKER = ".executor_info"
RUN_ORDER_PATTERN = re.compile(r"/fit_(?P<order>\d{3})_")
FULL_TRAIN_STEPS = EXPECTED_FINAL_CHECKPOINT_STEP + 1
RESUMABLE_REMAINING_STEPS = EXPECTED_FINAL_CHECKPOINT_STEP - EXPECTED_PHASE0_CHECKPOINT_STEP
LEGACY_EAST5_PARENT = "/calvinxu/dm-delphi-augmented-swarm-tpp40-phase0ckpt-interactive-retry8-20260825"
TERMINAL_IRIS_STATES = frozenset({"failed", "killed", "succeeded"})


def freeze_snapshot(*, legacy_parent_job: str, legacy_parent_state: str, observed_at_utc: str) -> dict[str, str]:
    if legacy_parent_job != LEGACY_EAST5_PARENT:
        raise ValueError(f"Expected legacy parent {LEGACY_EAST5_PARENT}, got {legacy_parent_job}")
    if legacy_parent_state not in TERMINAL_IRIS_STATES:
        raise ValueError(f"Legacy parent is not terminal: {legacy_parent_state}")
    observed_at = datetime.fromisoformat(observed_at_utc.replace("Z", "+00:00"))
    if observed_at.tzinfo is None or observed_at.utcoffset() != UTC.utcoffset(observed_at):
        raise ValueError("Legacy-parent observation must use UTC")
    return {
        "legacy_parent_job": legacy_parent_job,
        "legacy_parent_state": legacy_parent_state,
        "legacy_parent_observed_at_utc": observed_at.isoformat(),
    }


def _orders_with_marker(root: str, marker: str) -> set[int]:
    fs, path = fsspec.core.url_to_fs(root)
    orders: set[int] = set()
    for matched_path in fs.glob(f"{path.rstrip('/')}/*/{marker}"):
        match = RUN_ORDER_PATTERN.search(f"/{matched_path}")
        if match is None:
            raise ValueError(f"Could not parse run order from {matched_path}")
        orders.add(int(match.group("order")))
    return orders


def _orders_with_success_status(root: str) -> set[int]:
    fs, path = fsspec.core.url_to_fs(root)
    orders: set[int] = set()
    for status_path in fs.glob(f"{path.rstrip('/')}/*/{EXECUTOR_STATUS_MARKER}"):
        with fs.open(status_path, "rt") as handle:
            status = "".join(handle).strip()
        if not executor_status_succeeded(status):
            continue
        match = RUN_ORDER_PATTERN.search(f"/{status_path}")
        if match is None:
            continue
        orders.add(int(match.group("order")))
    return orders


def _load_strata(source_panel: str, *, expected_runs: int) -> dict[int, str]:
    with fsspec.open(source_panel, "rt") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_runs:
        raise ValueError(f"Source panel has {len(rows)} rows, expected {expected_runs}")
    return {order: f"{row['panel_source']}|{row['source_experiment']}" for order, row in enumerate(rows)}


def assign_remaining_rows(
    *,
    completed_orders: set[int],
    phase0_orders: set[int],
    strata_by_order: dict[int, str] | None = None,
    expected_runs: int = EXPECTED_RUNS,
) -> dict[str, tuple[int, ...]]:
    all_orders = set(range(expected_runs))
    if not completed_orders <= all_orders:
        raise ValueError("Completed run orders exceed the frozen panel")
    if not phase0_orders <= all_orders:
        raise ValueError("Phase-0 run orders exceed the frozen panel")

    resumable_east5 = phase0_orders - completed_orders
    east5 = set(resumable_east5)
    europe: set[int] = set()
    strata_by_order = strata_by_order or {order: "all" for order in all_orders}
    if set(strata_by_order) != all_orders:
        raise ValueError("Strata must contain exactly one label for every frozen-panel row")
    remaining = all_orders - completed_orders - resumable_east5
    for stratum in sorted(set(strata_by_order.values())):
        east5_count = sum(strata_by_order[order] == stratum for order in east5)
        europe_count = 0
        for order in sorted(order for order in remaining if strata_by_order[order] == stratum):
            if europe_count < east5_count:
                europe.add(order)
                europe_count += 1
            elif east5_count < europe_count:
                east5.add(order)
                east5_count += 1
            elif len(europe) <= len(east5):
                europe.add(order)
                europe_count += 1
            else:
                east5.add(order)
                east5_count += 1

    if completed_orders & east5 or completed_orders & europe or east5 & europe:
        raise AssertionError("TPP40 assignments overlap")
    if completed_orders | east5 | europe != all_orders:
        raise AssertionError("TPP40 assignments do not cover the frozen panel")
    return {
        "completed": tuple(sorted(completed_orders)),
        "east5": tuple(sorted(east5)),
        "europe": tuple(sorted(europe)),
        "resumable_east5": tuple(sorted(resumable_east5)),
    }


def compact_orders(orders: tuple[int, ...]) -> str:
    if not orders:
        return ""
    groups: list[str] = []
    start = previous = orders[0]
    for order in orders[1:]:
        if order == previous + 1:
            previous = order
            continue
        groups.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = order
    groups.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(groups)


def estimated_training_compute(assignments: dict[str, tuple[int, ...]]) -> dict[str, dict[str, float | int]]:
    resumable = set(assignments["resumable_east5"])
    east5_fresh = len(set(assignments["east5"]) - resumable)
    europe_fresh = len(assignments["europe"])
    return {
        "east5": {
            "fresh_rows": east5_fresh,
            "resumable_rows": len(resumable),
            "estimated_remaining_steps": east5_fresh * FULL_TRAIN_STEPS + len(resumable) * RESUMABLE_REMAINING_STEPS,
            "estimated_full_run_equivalents": (
                east5_fresh + len(resumable) * RESUMABLE_REMAINING_STEPS / FULL_TRAIN_STEPS
            ),
        },
        "europe": {
            "fresh_rows": europe_fresh,
            "resumable_rows": 0,
            "estimated_remaining_steps": europe_fresh * FULL_TRAIN_STEPS,
            "estimated_full_run_equivalents": float(europe_fresh),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--east5-root", default=DEFAULT_EAST5_ROOT)
    parser.add_argument("--europe-root", default=DEFAULT_EUROPE_ROOT)
    parser.add_argument("--source-panel", default=DEFAULT_SOURCE_PANEL)
    parser.add_argument("--legacy-parent-job", default=LEGACY_EAST5_PARENT)
    parser.add_argument("--legacy-parent-state", choices=sorted(TERMINAL_IRIS_STATES), required=True)
    parser.add_argument("--legacy-parent-observed-at-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    freeze = freeze_snapshot(
        legacy_parent_job=args.legacy_parent_job,
        legacy_parent_state=args.legacy_parent_state,
        observed_at_utc=args.legacy_parent_observed_at_utc,
    )
    east5_final_orders = _orders_with_marker(args.east5_root, FINAL_MARKER)
    europe_final_orders = _orders_with_marker(args.europe_root, FINAL_MARKER)
    east5_success_orders = _orders_with_success_status(args.east5_root)
    europe_success_orders = _orders_with_success_status(args.europe_root)
    europe_phase0_orders = _orders_with_marker(args.europe_root, PHASE0_MARKER)
    europe_executor_info_orders = _orders_with_marker(args.europe_root, EXECUTOR_INFO_MARKER)
    europe_executor_status_orders = _orders_with_marker(args.europe_root, EXECUTOR_STATUS_MARKER)
    if europe_final_orders or europe_phase0_orders or europe_executor_info_orders or europe_executor_status_orders:
        raise ValueError(
            "Refusing to mint a new assignment after Europe training state exists; reuse the frozen assignment"
        )
    if east5_final_orders & europe_final_orders:
        raise ValueError("A final TPP40 row exists in both East5 and Europe")
    completed_orders = (east5_final_orders & east5_success_orders) | (europe_final_orders & europe_success_orders)
    phase0_orders = _orders_with_marker(args.east5_root, PHASE0_MARKER)
    strata_by_order = _load_strata(args.source_panel, expected_runs=EXPECTED_RUNS)
    assignments = assign_remaining_rows(
        completed_orders=completed_orders,
        phase0_orders=phase0_orders,
        strata_by_order=strata_by_order,
    )
    strata = {
        group: dict(sorted(Counter(strata_by_order[order] for order in orders).items()))
        for group, orders in assignments.items()
    }
    payload: dict[str, object] = {
        "east5_root": args.east5_root,
        "europe_root": args.europe_root,
        "source_panel": args.source_panel,
        "final_marker": FINAL_MARKER,
        "phase0_marker": PHASE0_MARKER,
        "executor_status_marker": EXECUTOR_STATUS_MARKER,
        "executor_info_marker": EXECUTOR_INFO_MARKER,
        "expected_runs": EXPECTED_RUNS,
        "freeze": freeze,
        "observed": {
            "east5_final": sorted(east5_final_orders),
            "europe_final": sorted(europe_final_orders),
            "europe_phase0": sorted(europe_phase0_orders),
            "europe_executor_info": sorted(europe_executor_info_orders),
            "europe_executor_status": sorted(europe_executor_status_orders),
            "east5_success": sorted(east5_success_orders),
            "europe_success": sorted(europe_success_orders),
            "east5_final_without_success": sorted(east5_final_orders - east5_success_orders),
            "europe_final_without_success": sorted(europe_final_orders - europe_success_orders),
        },
        "assignments": {name: list(orders) for name, orders in assignments.items()},
        "strata": strata,
        "estimated_training_compute": estimated_training_compute(assignments),
        "run_order_args": {
            "east5": compact_orders(assignments["east5"]),
            "europe": compact_orders(assignments["europe"]),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["assignment_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
