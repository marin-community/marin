#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Migration script to correct FLOP calculations for v5p TPU speedrun results."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from fray.v1.cluster.ray.tpu import TPU_CONFIGS

TPU_CONFIG_BY_NAME = {config.name: config for config in TPU_CONFIGS}


def _slice_multiplier(value) -> int:
    if isinstance(value, int):
        return value
    if value is None:
        return 1
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        try:
            return max(value)
        except ValueError:
            return 1
    raise TypeError(f"Unsupported slice count value: {value!r}")


def _chips_per_slice(tpu_type: str) -> tuple[int, int]:
    config = TPU_CONFIG_BY_NAME.get(tpu_type)
    if config is None:
        raise ValueError(f"Unknown TPU type: {tpu_type}")

    try:
        _, suffix = tpu_type.split("-", maxsplit=1)
        reported_per_slice = int(suffix)
    except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unexpected TPU type format: {tpu_type}") from exc

    return config.chip_count, reported_per_slice


def _correct_run_info(run_info: dict) -> bool:
    resources = run_info.get("resources")
    if not isinstance(resources, dict):
        return False

    tpu_type = resources.get("tpu_type")
    if not isinstance(tpu_type, str) or not tpu_type.startswith("v5p-"):
        return False

    actual_per_slice, reported_per_slice = _chips_per_slice(tpu_type)
    if actual_per_slice == reported_per_slice:
        return False

    current_flops = run_info.get("training_hardware_flops")
    if current_flops is None:
        return False

    slice_multiplier = _slice_multiplier(resources.get("slice_count", 1))
    total_chips = actual_per_slice * slice_multiplier
    total_devices = reported_per_slice * slice_multiplier

    correction_ratio = actual_per_slice / reported_per_slice
    corrected_flops = current_flops * correction_ratio

    run_info["training_hardware_flops"] = corrected_flops
    run_info.setdefault("num_devices", total_devices)
    run_info["num_chips"] = total_chips

    training_time = run_info.get("training_time")
    if training_time and training_time > 0:
        run_info["device_flops"] = corrected_flops / (training_time * total_chips)

    return True


def _update_file(path: Path, *, dry_run: bool) -> bool:
    with path.open() as f:
        data = json.load(f)

    changed = False
    for entry in data.get("runs", []):
        run_info = entry.get("run_info")
        if isinstance(run_info, dict) and _correct_run_info(run_info):
            changed = True

    if changed and not dry_run:
        with path.open("w") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    return changed


def _iter_speedrun_results(paths: list[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from path.rglob("speedrun_results.json")
        elif path.name == "speedrun_results.json":
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path.cwd()],
        help="Directories or files to scan for speedrun_results.json files.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show files that would be updated without writing them.")
    args = parser.parse_args()

    files = list(_iter_speedrun_results(args.paths))
    if not files:
        print("No speedrun_results.json files found.")
        return

    updated = 0
    for file_path in files:
        if _update_file(file_path, dry_run=args.dry_run):
            updated += 1
            action = "Would update" if args.dry_run else "Updated"
            print(f"{action}: {file_path}")

    if not updated:
        print("No v5p speedrun results required updates.")


if __name__ == "__main__":
    main()
