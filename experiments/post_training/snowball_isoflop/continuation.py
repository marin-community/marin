# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Continue explicitly verified A trajectories without overwriting earlier artifacts."""

import argparse
import json
import logging
from dataclasses import dataclass

import draccus
from iris.client.client import iris_ctx
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.snowball_isoflop.budget import budgeted_map
from experiments.post_training.snowball_isoflop.config import (
    PREFIX,
    RUNTIME_FIELDS,
    VERSION,
    Cell,
    build_pod,
)
from experiments.post_training.snowball_isoflop.run import run_training_pod


@dataclass(frozen=True)
class Continuation:
    cell: Cell
    checkpoint_path: str | None
    completed_updates: int


def run_continuation(continuation: Continuation) -> str:
    pod = build_pod(continuation.cell, resume_checkpoint=continuation.checkpoint_path)
    return run_training_pod(continuation.cell, pod)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manager-root", required=True)
    parser.add_argument("--calibration-report", required=True)
    parser.add_argument("--guard-config", required=True)
    parser.add_argument("--h100-hour-limit", required=True, type=float)
    parser.add_argument("--max-concurrent", required=True, type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    continuations = draccus.decode(tuple[Continuation, ...], json.loads(StoragePath(args.manifest).read_text()))
    report = json.loads(StoragePath(args.calibration_report).read_text())
    if not report["all_six_cells_valid"] or not report["expansion_gate_passed"] or report["version"] != VERSION:
        raise ValueError("A continuation requires the successful six-cell calibration")
    context = iris_ctx()
    guard = json.loads(StoragePath(args.guard_config).read_text())
    if context.job_id is None or not any(
        budget["phase"] == "A" and context.job_id.to_wire() in budget["roots"] for budget in guard["budgets"]
    ):
        raise ValueError("Register the continuation root in the aggregate A guard before execution")
    if not 0 < args.h100_hour_limit <= 700 or not 1 <= args.max_concurrent <= 2:
        raise ValueError("Continuation exceeds its approved per-size tranche or concurrency")
    for continuation in continuations:
        cell = continuation.cell
        pod = build_pod(cell, resume_checkpoint=continuation.checkpoint_path)
        if not pod.output_path.startswith(f"{PREFIX}/{VERSION}/A-host512-r3-"):
            raise ValueError("Continuations require fresh owned artifact identities")
        if (StoragePath(pod.output_path) / "consumed-config.json").exists():
            raise ValueError("Continuation output already exists")
        environment = pod.env_vars or {}
        if {key: environment.get(key) for key in RUNTIME_FIELDS} != report["runtime_environment_by_cell"][
            f"{cell.size}-{cell.mask}"
        ]:
            raise ValueError("Continuation runtime differs from its calibration")
        if continuation.checkpoint_path is None:
            if continuation.completed_updates != 0:
                raise ValueError("A fresh trajectory must start at zero")
        else:
            metadata = json.loads((StoragePath(continuation.checkpoint_path) / "metadata.json").read_text())
            if (
                metadata["step"] + 1 != continuation.completed_updates
                or not 0 < continuation.completed_updates < cell.stop_step
            ):
                raise ValueError("Checkpoint update count does not match the verified continuation")
    manager = StoragePath(args.manager_root)
    (manager / "submitted-continuations.json").write_text(StoragePath(args.manifest).read_text())
    budgeted_map(
        {row.cell.name: row for row in continuations},
        run_continuation,
        manager,
        args.h100_hour_limit,
        args.max_concurrent,
    )


if __name__ == "__main__":
    main()
