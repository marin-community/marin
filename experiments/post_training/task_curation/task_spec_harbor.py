# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare unchanged converted TaskSpecs for production Harbor trials."""

import argparse
import importlib
from pathlib import Path

import pyarrow.parquet as pq
from pydantic import TypeAdapter
from taskcompendium.harbor.adapter import declared_execution_limits, write_harbor_task
from taskcompendium.harbor.oracle import write_oracle_command
from taskcompendium.harbor.protected_runtime import ExpectedRuntime, RuntimeRole, write_runtime_binding
from taskcompendium.models import ExecutionRequirements
from taskcompendium.pipeline.models import Controls
from taskcompendium.pipeline.resource_calibration import INITIAL_REQUIREMENTS

from experiments.post_training.task_curation.oracle_filter import row_runtime, row_task, write_task_map


def prepare(
    cohort: Path,
    output: Path,
    *,
    source_revision: str,
    controls: Controls | None = None,
    runtime_expectations: dict[RuntimeRole, ExpectedRuntime] | None = None,
) -> None:
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
                if controls is None and runtime_expectations is None:
                    continue
                runtime = row_runtime(row)
                if runtime is None:
                    raise ValueError(f"Explicit oracle/runtime bindings require the original runtime: {task.id}")
                if controls is not None and controls.golden is not None:
                    write_oracle_command(task, runtime, controls.golden, directory, source_revision=source_revision)
                if runtime_expectations is not None:
                    write_runtime_binding(task, runtime, runtime_expectations, directory)
    write_task_map(cohort, tasks, output / "task-map.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True, help="Pinned code revision declaring the controls")
    parser.add_argument("--controls", help="Declared Controls object as module:attribute")
    parser.add_argument("--runtime-expectations", type=Path, help="JSON mapping of source roles to expected runtimes")
    args = parser.parse_args()
    controls = None
    if args.controls is not None:
        module, attribute = args.controls.split(":", 1)
        controls = getattr(importlib.import_module(module), attribute)
        if not isinstance(controls, Controls):
            raise ValueError("--controls must name a declared Controls object")
    expectations = (
        TypeAdapter(dict[RuntimeRole, ExpectedRuntime]).validate_json(args.runtime_expectations.read_text())
        if args.runtime_expectations is not None
        else None
    )
    prepare(
        args.cohort,
        args.output,
        source_revision=args.source_revision,
        controls=controls,
        runtime_expectations=expectations,
    )


if __name__ == "__main__":
    main()
