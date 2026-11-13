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
from iris.client.client import iris_ctx
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.snowball_isoflop.budget import allocation_ledger
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
    parser.add_argument("--max-concurrent", type=int, choices=range(1, 9), default=8)
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
    (manager_root / "submitted-cells.json").write_text(json.dumps([asdict(c) for c in cells], indent=2))
    context = iris_ctx()
    if context.client is None or context.job_id is None:
        raise RuntimeError("A budgeted phase must run inside an Iris coordinator")
    allocation_ledger(context.client, context.job_id)
    concurrency = min(args.max_concurrent, len(cells))
    # Context propagation keeps the enclosing Iris/Fray client in each coordinator thread.
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        pending = iter(cells)
        futures = {}
        try:
            for _ in range(concurrency):
                cell = next(pending)
                futures[executor.submit(contextvars.copy_context().run, run_cell, cell)] = cell
            while futures:
                completed, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
                ledger = allocation_ledger(context.client, context.job_id)
                (manager_root / "allocation-ledger.json").write_text(json.dumps(ledger, indent=2))
                # Reserve one minute of maximum allocation for polling and cancellation.
                if ledger["h100_hours"] + concurrency * 8 / 60 >= args.h100_hour_limit:
                    raise RuntimeError("Approved experiment H100-hour limit reached")
                for future in completed:
                    cell = futures.pop(future)
                    output = future.result()
                    (manager_root / f"completed-{cell.name}.json").write_text(json.dumps({"output": output}))
                    cell = next(pending, None)
                    if cell is not None:
                        futures[executor.submit(contextvars.copy_context().run, run_cell, cell)] = cell
        except Exception:
            logger.exception("Stopping this experiment tree after a cell or budget-monitor failure")
            context.client.job(context.job_id).cancel()
            raise


if __name__ == "__main__":
    main()
