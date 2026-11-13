# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect or execute approved native SFTSpec cells inside an Iris CPU coordinator."""

import argparse
import contextvars
import json
import logging
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict

import draccus
from rigging.filesystem.storage_path import StoragePath

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
    pod = build_pod(cell)
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
    logging.basicConfig(level=logging.INFO)
    for cell in cells:
        gate = build_pod(cell).train_config.initial_eval_match
        if gate is None or not StoragePath(gate.reference_path).exists():
            raise ValueError(f"Missing upstream reference evaluation for d{cell.size}; no training jobs submitted")
    manager_root = StoragePath(f"{PREFIX}/{VERSION}/manager-{args.phase}-{args.size or 'all'}")
    (manager_root / "submitted-cells.json").write_text(json.dumps([asdict(c) for c in cells], indent=2))
    # Context propagation keeps the enclosing Iris/Fray client in each coordinator thread.
    with ThreadPoolExecutor(max_workers=min(8, len(cells))) as executor:
        pending = iter(cells)
        futures = {}
        for _ in range(min(8, len(cells))):
            cell = next(pending)
            futures[executor.submit(contextvars.copy_context().run, run_cell, cell)] = cell
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                cell = futures.pop(future)
                output = future.result()
                (manager_root / f"completed-{cell.name}.json").write_text(json.dumps({"output": output}))
                cell = next(pending, None)
                if cell is not None:
                    futures[executor.submit(contextvars.copy_context().run, run_cell, cell)] = cell


if __name__ == "__main__":
    main()
