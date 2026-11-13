# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect or execute approved native SFTSpec cells inside an Iris CPU coordinator."""

import argparse
import json
import logging
from dataclasses import asdict

import draccus
from marin.training.training import TrainLmOnPodConfig
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.snowball_isoflop.budget import budgeted_map
from experiments.post_training.snowball_isoflop.config import (
    PREFIX,
    VERSION,
    Cell,
    build_pod,
    build_spec,
    calibration_cells,
    primary_cells,
)

logger = logging.getLogger(__name__)


def run_cell(cell: Cell) -> str:
    return run_training_pod(cell, build_pod(cell))


def run_training_pod(cell: Cell, pod: TrainLmOnPodConfig) -> str:
    logger.info("Starting native SFTSpec cell %s at %s", cell.name, pod.output_path)
    build_spec(cell).model.run(pod)
    metadata = json.loads(
        (StoragePath(pod.output_path) / f"checkpoints/step-{cell.stop_step - 1}/metadata.json").read_text()
    )
    if metadata["step"] != cell.stop_step - 1 or metadata["is_temporary"]:
        raise ValueError(f"Final checkpoint not permanently committed for {cell.name}")
    logger.info("Completed native SFTSpec cell %s", cell.name)
    return pod.output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("calibration", "A"), required=True)
    parser.add_argument("--size", type=int, choices=(512, 768, 1024))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--calibration-report")
    parser.add_argument("--max-concurrent", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--initial-concurrent", type=int, choices=range(1, 9))
    parser.add_argument("--h100-hour-limit", type=float, required=False)
    args = parser.parse_args()
    cells = calibration_cells() if args.phase == "calibration" else primary_cells()
    if args.size is not None:
        cells = tuple(cell for cell in cells if cell.size == args.size)
    if not args.run:
        print(
            json.dumps([{"cell": asdict(c), "train_config": draccus.encode(build_pod(c).train_config)} for c in cells])
        )
        return
    if args.phase == "A":
        if args.calibration_report is None:
            parser.error("A requires --calibration-report from all six successful native calibration cells")
        report = json.loads(StoragePath(args.calibration_report).read_text())
        if (
            report["version"] != VERSION
            or not report["all_six_cells_valid"]
            or report["projected_primary_gpu_hours"] > 2200
        ):
            raise ValueError("Calibration did not pass the approved primary-grid gate")
        cells = tuple(
            sorted(
                cells, key=lambda cell: report["inclusive_step_seconds"][str(cell.size)] * cell.stop_step, reverse=True
            )
        )
    if args.h100_hour_limit is None or not 0 < args.h100_hour_limit <= 2200:
        parser.error("Execution requires --h100-hour-limit within the remaining approved A ceiling")
    logging.basicConfig(level=logging.INFO)
    for cell in cells:
        gate = build_pod(cell).train_config.initial_eval_match
        if gate is None or not StoragePath(gate.reference_path).exists():
            raise ValueError(f"Missing upstream reference evaluation for d{cell.size}; no training jobs submitted")
    manager_root = StoragePath(f"{PREFIX}/{VERSION}/manager-{args.phase}-{args.size or 'all'}")
    if args.initial_concurrent is not None:
        if args.initial_concurrent > args.max_concurrent:
            parser.error("Initial concurrency exceeds the maximum")
        (manager_root / "concurrency.json").write_text(json.dumps({"max_concurrent": args.initial_concurrent}))
    (manager_root / "submitted-cells.json").write_text(json.dumps([asdict(c) for c in cells], indent=2))
    budgeted_map({cell.name: cell for cell in cells}, run_cell, manager_root, args.h100_hour_limit, args.max_concurrent)


if __name__ == "__main__":
    main()
