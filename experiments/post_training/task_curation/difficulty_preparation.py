# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize unchanged sampled contracts for production Harbor candidate trials."""

import argparse
import json
import os
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.harbor.adapter import declared_execution_limits, unsupported_harbor_contract, write_harbor_task
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ExecutionRequirements, NoGrader, judge_spec
from taskcompendium.pipeline.resource_calibration import INITIAL_REQUIREMENTS
from taskcompendium.runtime.staging import resource_files

from experiments.post_training.task_curation.difficulty_prepared import PreparedSource, PreparedTask
from experiments.post_training.task_curation.difficulty_samples import DifficultySource
from experiments.post_training.task_curation.prebuilt import validate_native_package
from experiments.post_training.task_curation.sample_qc import panel_tasks


def prepare_source(source: DifficultySource, root: Path) -> PreparedSource:
    tasks, contracts = panel_tasks(source.panel)
    prepared = []
    for task, contract in zip(tasks, contracts, strict=True):
        if task.id not in {selected.task_id for selected in source.sample.tasks}:
            raise ValueError("Prepared task is outside the frozen difficulty selection")
        selected = next(item for item in source.sample.tasks if item.task_id == task.id)
        if selected.task_sha256 != contract["task_sha256"]:
            raise ValueError("Prepared task differs from the frozen sample receipt")
        native = isinstance(task.grader, NoGrader) and "harbor" in task.tags
        reason = None if native else unsupported_harbor_contract(task)
        if reason is not None:
            reason = f"awaiting_executability_filtered_admission_population:{reason}"
        judge = judge_spec(task.grader)
        if source.backend == "daytona":
            reason = "daytona_snapshot_preflight_pending" if native else "faithful_daytona_task_spec_factory_pending"
        if judge is not None and (judge.model != "glm-5.3" or judge.temperature != 0.7):
            reason = f"production_judge_route_or_temperature_not_qualified:{judge.model}:temperature={judge.temperature}"
        directory = root / canonical_sha256(source.declaration.id)[:20] / canonical_sha256(task.id)[:20]
        if reason is None:
            if native:
                validate_native_package(task)
                directory.mkdir(parents=True, exist_ok=False)
                for file in resource_files((*task.resources.verifier, *task.resources.oracle), ""):
                    target = directory / file.path.removeprefix("/")
                    if target.exists() and target.read_bytes() != file.data:
                        raise ValueError(f"Native source roles disagree on resource bytes: {file.path}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(file.data)
                    target.chmod(file.mode)
                    if file.mtime_ns is not None:
                        os.utime(target, ns=(file.mtime_ns, file.mtime_ns))
            else:
                shape = ExecutionRequirements.model_validate(
                    {
                        **INITIAL_REQUIREMENTS.model_dump(),
                        **declared_execution_limits(task),
                    }
                )
                write_harbor_task(task, directory, shape)
        prepared.append(
            PreparedTask(
                task_id=task.id,
                path=str(directory) if reason is None else None,
                workload="native" if native else "task_spec",
                backend=source.backend,
                reason=reason,
                judge_model=None if judge is None else judge.model,
            )
        )
    return PreparedSource(sample=source.sample, sample_receipt=source.sample_receipt, tasks=prepared)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--sources-sha256", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source-id", action="append")
    args = parser.parse_args()
    configure_coreweave_s3()
    if args.root.exists():
        raise FileExistsError(args.root)
    path = StoragePath(args.sources)
    if file_sha256(path) != args.sources_sha256:
        raise ValueError("Difficulty sources differ from the frozen selection")
    sources = [DifficultySource.model_validate(row) for row in json.loads(path.read_text())]
    if args.source_id:
        sources = [source for source in sources if source.declaration.id in args.source_id]
        if {source.declaration.id for source in sources} != set(args.source_id):
            raise ValueError("Requested source is absent from the frozen selection")
    args.root.mkdir(parents=True)
    prepared = [prepare_source(source, args.root / "tasks") for source in sources]
    (args.root / "sources.json").write_text(
        json.dumps([source.model_dump(mode="json") for source in prepared], indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
