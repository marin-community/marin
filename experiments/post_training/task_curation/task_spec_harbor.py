# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare unchanged converted TaskSpecs for production Harbor trials."""

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from taskcompendium.harbor.adapter import declared_execution_limits, write_harbor_task
from taskcompendium.models import ExecutionRequirements
from taskcompendium.pipeline.resource_calibration import INITIAL_REQUIREMENTS

from experiments.post_training.task_curation.oracle_filter import row_task, write_task_map


def prepare(cohort: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    tasks = output / "tasks"
    paths = sorted(cohort.glob("*.parquet")) if cohort.is_dir() else [cohort]
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                task = row_task(row)
                if task is None:
                    continue
                # This is a trial budget, never a mutation or admission of the original task.
                shape = task.execution_requirements or ExecutionRequirements.model_validate(
                    {**INITIAL_REQUIREMENTS.model_dump(), **declared_execution_limits(task)}
                )
                directory = tasks / task.id
                if directory.parent != tasks:
                    raise ValueError(f"Task ID is not a package directory name: {task.id}")
                write_harbor_task(task, directory, shape)
    write_task_map(cohort, tasks, output / "task-map.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.cohort, args.output)


if __name__ == "__main__":
    main()
